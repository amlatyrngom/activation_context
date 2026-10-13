"""Paired whole-history compaction/QA items and exact recorded-run self-distillation."""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import replace
import random
import re
import typing as t

from ..agent.agent_utils import ModelDialect
from ..dataset.dataset import DataModality, DatasetDocument, DatasetDocumentChunk, DatasetQAExample
from ..dataset.loaders.trajectory_utils import submit_answer_definition
from ..dataset.dataset_utils import message_text, render_messages
from .ac_model_training import ActivationContextTrainingItem, ActivationContextTrainer, ActivationContextTrainingConfig
from .ac_model_utils import (AC_PART_TYPE, direct_parts, is_ac_part, assistant_target_ids, without_stop_token,
                             prepare_training_history, normalize_public_assistants, complete_public_turns, is_dense_piece)
from ..common.ac_parts import COMPACTION_INSTRUCTIONS, ac_part, tools_in_system_text

if t.TYPE_CHECKING:
    from ..agent.agent_config import AgentRunResult
    from ..harness import HarnessRuntime

TRAJ_QA_SYSTEM_PROMPT = "You answer questions about agent trajectories (transcripts of an agent's reasoning, tool calls and tool results)."
TRAJ_QA_INSTRUCTIONS = "Answer the question from the trajectories above (some may be unrelated). Call submit_answer with the answer."
RECONSTRUCTION_SYSTEM_PROMPT = "You are a careful assistant. You reproduce text exactly when asked."
RECONSTRUCTION_INSTRUCTION = "Reproduce the passage above exactly, word for word, and stop at its end."
PASSAGE_INSTRUCTION = "Reproduce Passage {k} from the passages above exactly, word for word, and stop at its end."
LOOKUP_INSTRUCTION = "Find where the following text occurs in the passages above and continue it exactly, word for word.\nText: {cue}"
SPAN_INSTRUCTION = ("Find where the first text occurs in the passages above and reproduce exactly, word for word, what follows it, "
                    "stopping right before the second text.\nStart: {cue}\nEnd: {end_cue}")
MULTISPAN_SEPARATOR = "\n---\n"
MULTISPAN_INSTRUCTION = ("Reproduce the following parts of the passage above exactly, word for word, in the order listed, separating "
                         "consecutive parts with a line containing only ---. Each part is what follows its start text (or begins the "
                         "passage) and stops right before its end text (or at the passage's end).\n{reads}")
MULTISPAN_READS = {"start": "{i}. From the beginning of the passage up to: {end_cue}", "mid": "{i}. From: {cue} up to: {end_cue}",
                   "end": "{i}. From: {cue} to the end of the passage"}
MULTISPAN_PASSAGES_INSTRUCTION = ("Reproduce the following parts of the passages above exactly, word for word, in the order listed, separating "
                                  "consecutive parts with a line containing only ---. Each part is what follows its start text and stops "
                                  "right before its end text.\n{reads}")


def join_passages(texts: t.Sequence[str]) -> str:
    """Several passages as one text, in the RAG evidence format ("Passage 1:\\n...\\n\\nPassage 2:\\n...")."""
    return "\n\n".join(f"Passage {i + 1}:\n{text}" for i, text in enumerate(texts))


DENSITY_FLOOR = 0.05   # added to a window's dense share before the power, so plain prose keeps a (small) draw weight


def _dense_mask(text: str, offsets: t.Sequence[t.Sequence[int]]) -> list[int]:
    """1 for every token whose text holds a digit, a code/math symbol or a non-ASCII character that is neither a letter nor
    punctuation (math symbols, non-Latin digits, box/arrow glyphs; curly quotes and dashes of typeset prose do not count):
    numbers, identifiers, formulas, tables — the material the rows carry well but cued access finds badly. Else 0."""
    out = []
    for start, end in offsets:
        out.append(1 if is_dense_piece(text[start:end]) else 0)
    return out


def _window_weights(mask: t.Sequence[int], window: int, power: float, count: int) -> list[float]:
    """Draw weight of each of the first `count` start positions (count <= len(mask) - 1, so no window is empty):
    (DENSITY_FLOOR + dense share of [i, i + window))^power, the window clipped at the end of the passage."""
    if not 0 < count < len(mask) + 1 or window < 1:
        raise ValueError("count must be within the mask and window positive")
    prefix = [0]
    for value in mask:
        prefix.append(prefix[-1] + value)
    n = len(mask)
    return [(DENSITY_FLOOR + (prefix[min(n, i + window)] - prefix[i]) / (min(n, i + window) - i)) ** power for i in range(count)]


DIGIT_RUN = re.compile(r"[0-9]+")


def _perturb_digits(text: str, rng: random.Random) -> str | None:
    """`text` with every run of digits replaced by a different random run of the same length (a leading nonzero digit stays
    nonzero, so numbers keep their magnitude), or None when the text has no digits: the decoy twin of a passage."""
    if not DIGIT_RUN.search(text):
        return None

    def redraw(match: re.Match) -> str:
        run = match.group()
        for _ in range(8):
            new = "".join(rng.choice("123456789" if i == 0 and run[0] != "0" and len(run) > 1 else "0123456789") for i in range(len(run)))
            if new != run:
                return new
        return run
    return DIGIT_RUN.sub(redraw, text)


def _weighted_index(rng: random.Random, weights: t.Sequence[float]) -> int:
    """One index drawn proportionally to `weights` (one rng.random() call)."""
    total = sum(weights)
    if not total > 0:
        raise ValueError("Draw weights must sum to a positive number")
    r = rng.random() * total
    for index, weight in enumerate(weights):
        r -= weight
        if r < 0:
            return index
    return len(weights) - 1


def _cue_cut(text: str, fraction: float) -> int:
    """Character offset near `fraction` of the text, moved forward to the next whitespace so the cue ends on a word."""
    cut = max(1, int(len(text) * fraction))
    while cut < len(text) and not text[cut].isspace():
        cut += 1
    return cut


def _segment_content(segments: list[dict], ac_name: str) -> list[dict]:
    """The content list of a nested passage: its text segments verbatim and one activation_context part per child segment, in order."""
    content = []
    for segment in segments:
        if segment["kind"] == "child":
            content.append(ac_part([{"role": "user", "content": segment["text"]}], ac_name, segment["ratio"]))
        elif segment["kind"] == "text":
            if segment["text"]:
                content.append({"type": "text", "text": segment["text"]})
        else:
            raise ValueError(f"unknown segment kind {segment['kind']!r}")
    return content


def _first_child_span(segments: list[dict]) -> tuple[int, int] | None:
    """Character span of the first child segment inside the concatenation of the segment texts (None without a child)."""
    offset = 0
    for segment in segments:
        if segment["kind"] == "child":
            return offset, offset + len(segment["text"])
        offset += len(segment["text"])
    return None


RAG_QA_SYSTEM_PROMPT = "You answer questions from the passages you are given."
RAG_QA_INSTRUCTIONS = "Answer the question from the passages above (some may be unrelated). Call submit_answer with the answer."
MESSAGE_TOKEN_OVERHEAD = 8                    # template tokens per message, on top of its text (estimate)
MIN_COMPACTION_TOKENS = 1024                  # a trajectory shorter than this (before its completion) yields no compaction item


# (It can't be perfect since we are taking external trajectories, but we should not kneecap ourselves; e.g., use things like submit answer where possible, etc.).
# If not perform on semi-on-policy fine-tuning from rollouts is always on the table.
# Can also please suggest some way to mimic subagent spawning? I am drawing a blank.
# My only idea involves: craft a fake multi-task combining 2-5 trajectories, the prompt asks it to do one specific one given its id. It takes the parent context is maps the id to the given task, showing that it can "see" its parent.
# But this is a bit too fake (only teaches id reading; probably worse than qa). It's not a big  deal if we end up having to roll our own if there are no clean synthetic approaches.
class ActivationContextStudyGenerator:
    """Item generators bound to one AC model (its name goes into the parts; its target tokenizer counts tokens)."""

    def __init__(self, harness: "HarnessRuntime", ac_model_name: str) -> None:
        self.harness = harness
        self.ac_model = harness.module_manager.get_ac_model(ac_model_name)
        self.ac_name = ac_model_name
        self.tokenizer = self.ac_model.target.tokenizer
        self.dialect = ModelDialect.for_tokenizer(self.tokenizer)
        self._token_cache: dict[str, int] = {}
        self.harvest_report: dict = {}             # counts of the last items_from_run_results call

    # ------------------------------------------------------------------------------------------ helpers
    def count_tokens(self, text: str) -> int:
        count = self._token_cache.get(text)
        if count is None:
            count = len(self.tokenizer.encode(text, add_special_tokens=False))
            if len(self._token_cache) < 200_000:
                self._token_cache[text] = count
        return count

    def message_tokens(self, message: dict) -> int:
        total = self.count_tokens(message_text(message)) + MESSAGE_TOKEN_OVERHEAD
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            total += self.count_tokens(json.dumps(function.get("arguments", {}), ensure_ascii=False)) + MESSAGE_TOKEN_OVERHEAD
        return total

    def assistant_text(self, message: dict) -> str:
        """What the model would have produced for this turn: its text, then its calls in the dialect's format."""
        text = message_text(message)
        calls = [self.dialect.preferred_format.render(call.get("function", call)["name"], call.get("function", call).get("arguments", {}))
                 for call in message.get("tool_calls") or []]
        return "\n".join(piece for piece in [text.rstrip()] + calls if piece)

    @staticmethod
    def _sample_ratio(rng: random.Random, ratios_range: tuple[float, float]) -> float:
        low, high = min(ratios_range), max(ratios_range)
        return float(rng.uniform(low, high))

    def _documents(self, dataset_id: str, modality: DataModality | None) -> list[DatasetDocument]:
        dataset = self.harness.dataset_manager.loaded_datasets[dataset_id]
        return [document for document in dataset.documents.values() if modality is None or document.modality == modality]

    def _system_messages(self, document: DatasetDocument) -> list[dict]:
        kwargs = document.trajectory_kwargs or {}
        return [{"role": "system", "content": kwargs["system_prompt"]}] if kwargs.get("system_prompt") else []

    # ------------------------------------------------------------------------------------------ compaction
    def generate_compaction_samples(
        self, dataset_id: str, num_samples: int, seed: int = 0,
        depth_range: tuple[int, int] = (1, 2),
        threshold_range_tokens: tuple[int, int] = (8192, 32768),
        ratios_range: tuple[float, float] = (1.0 / 8.0, 1.0 / 16.0),
        max_post_compaction_tokens: int = 8192,
        max_total_tokens: int = 72_000,
        min_trajectory_tokens: int = MIN_COMPACTION_TOKENS,
        *, max_assistant_tokens: int | None = None,
    ) -> list[ActivationContextTrainingItem]:
        """Complete public continuations; every assistant output is a target, within both exact budgets."""
        if max_post_compaction_tokens < 1 or max_total_tokens < 1 or min(depth_range) < 1:
            raise ValueError("Invalid compaction depth or token budgets")
        if max_assistant_tokens is None:
            from ..harness.vllm_wrapper import VLLMWrapper
            defaults = VLLMWrapper.recommended_chat_kwargs(self.ac_model.target.model_config.model_id)
            max_assistant_tokens = (defaults.get("sampling_params") or {}).get("max_tokens", 1024)
        if max_assistant_tokens < 1:
            raise ValueError("max_assistant_tokens must be positive")
        rng = random.Random(seed)
        documents = self._documents(dataset_id, DataModality.TRAJECTORY)
        assert documents, f"{dataset_id} holds no trajectory documents"
        rng.shuffle(documents)
        items = []
        for attempt in range(4 * num_samples + len(documents)):
            if len(items) >= num_samples:
                break
            item = self._compaction_item(documents[attempt % len(documents)], dataset_id, rng, len(items), seed,
                        depth_range, threshold_range_tokens, ratios_range, max_total_tokens, min_trajectory_tokens,
                        max_post_compaction_tokens, max_assistant_tokens)
            if item is not None:
                items.append(item)
        if len(items) < num_samples:
            print(f"AC study - produced {len(items)}/{num_samples} compaction items; complete candidates exhausted")
        return items

    def _compaction_item(self, document, dataset_id, rng, index, seed, depth_range, threshold_range_tokens,
                         ratios_range, max_total_tokens, min_trajectory_tokens, max_post_compaction_tokens,
                         max_assistant_tokens) -> ActivationContextTrainingItem | None:
        source = document.trajectory or []
        if len(source) < 2 or source[0].get("role") != "user":
            return None
        try:
            messages, tools = normalize_public_assistants(self.tokenizer, source,
                        (document.trajectory_kwargs or {}).get("tools"), max_assistant_tokens)
            first = next((index for index, message in enumerate(messages) if message.get("role") == "assistant"), len(messages))
            initial = messages[:first]
            groups = complete_public_turns(messages[first:])
        except ValueError:                    # an indivisible public call/reply cannot be a complete candidate
            return None
        if len(groups) < 2:
            return None
        system = self._system_messages(document)
        depth = min(rng.randint(min(depth_range), max(depth_range)), len(groups) - 1)
        if depth < min(depth_range):
            return None
        costs = [sum(self.message_tokens(message) for message in group) for group in groups]
        budget = min(sum(costs[:-1]), max_total_tokens - max_post_compaction_tokens)
        threshold = min(rng.randint(min(threshold_range_tokens), max(threshold_range_tokens)), max(1, budget // depth))
        cuts, offset = [], 0
        for level in range(depth):
            total, end = 0, offset
            limit = len(groups) - (depth - level)
            while end < limit and (total < threshold or end == offset):
                total += costs[end]
                end += 1
            cuts.append(end)
            offset = end
        before = [message for group in groups[:cuts[-1]] for message in group]
        prefix = system + initial + before
        chat_kwargs = {"enable_thinking": True}
        continuation, targets, exact_tokens, post_tokens = [], [], 0, 0
        for group in groups[cuts[-1]:]:
            proposed = continuation + group
            proposed_targets = targets + [assistant_target_ids(self.tokenizer, group[0], tools, chat_kwargs)]
            ids, _, positions = prepare_training_history(self.tokenizer, prefix + proposed, len(prefix), proposed_targets, [], tools=tools,
                                                        chat_template_kwargs=chat_kwargs)
            if positions[0] + 1 < min_trajectory_tokens:
                return None
            cost = len(ids) - (positions[0] + 1)
            if cost > max_post_compaction_tokens or len(ids) > max_total_tokens:
                break
            continuation, targets, exact_tokens, post_tokens = proposed, proposed_targets, len(ids), cost
        if not targets:
            return None
        ratio, nested, offset = self._sample_ratio(rng, ratios_range), None, 0
        for end in cuts:
            frame = [{"role": "user", "content": [nested]}] if nested is not None else initial
            segment = [message for group in groups[offset:end] for message in group]
            nested = ac_part(deepcopy(system + frame + segment), self.ac_name, ratio, kind="compaction")
            if tools:
                nested["tools"] = deepcopy(tools)
            offset = end
        student_prefix = system + initial + [{"role": "user", "content": [nested,
                            {"type": "text", "text": "\n\n" + COMPACTION_INSTRUCTIONS}]}]
        item = ActivationContextTrainingItem(
            item_id=f"compaction:{dataset_id}:{seed}:{index}", kind="compaction",
            in_context_messages=prefix + continuation, ac_messages=deepcopy(student_prefix + continuation),
            in_context_start=len(prefix), ac_start=len(student_prefix), assistant_token_ids=targets,
            tools=tools, dataset_id=dataset_id, doc_ids=[document.doc_id],
            info={"depth": depth, "thresholds": [threshold] * depth, "ratio": ratio,
                  "chat_template_kwargs": chat_kwargs,
                  "in_context_tokens": exact_tokens, "post_compaction_tokens": post_tokens,
                  "max_assistant_tokens": max_assistant_tokens, "assistant_turns": len(targets),
                  "observed_terminal_answer": (document.trajectory_kwargs or {}).get("observed_terminal_answer")})
        # Student wrappers and row counts also count against the overall guard.
        try:
            ActivationContextTrainer(self.harness, ActivationContextTrainingConfig(max_example_tokens=max_total_tokens)).build_example(self.ac_model, item)
        except ValueError as error:
            if "max_example_tokens" in str(error):
                return None
            raise
        return item

    # ------------------------------------------------------------------------------------------ run-result harvesting
    def items_from_run_results(
        self, runs: list["AgentRunResult"], *, loss_kind: t.Literal["kl", "sft"] = "kl",
        selection: t.Literal["score_1", "score_1_or_unscored", "all"] | None = None,
        weight_of: t.Callable[["AgentRunResult"], float] | None = None,
        text_only: bool = False,
    ) -> list[ActivationContextTrainingItem]:
        """Harvest whole segments; text-only SFT retains ordinary context and trains only the reader.

        Text-only outputs are rendered with the student's tokenizer, including recorded thinking
        and tool calls. The default AC path preserves the original sampled token IDs exactly.
        """
        from ..agent_training.agent_training_utils import activation_messages_of
        from ..agent.rollout_caching import config_key
        if loss_kind not in ("kl", "sft"):
            raise ValueError(loss_kind)
        if text_only and loss_kind != "sft":
            raise ValueError("Text-only rollout rehearsal uses SFT")
        selection = selection or ("all" if loss_kind == "kl" else "score_1")
        if selection not in ("all", "score_1", "score_1_or_unscored"):
            raise ValueError(selection)
        report = self.harvest_report = {"roots": len(runs), "runs": 0, "candidates": 0, "accepted": 0,
                                      "loss_kind": loss_kind, "selection": selection, "text_only": text_only, "skipped": {}}
        items = []
        for root in runs:
            if selection != "all" and root.score != 1 and not (selection == "score_1_or_unscored" and root.score is None):
                self._harvest_skip("selection")
                continue
            weight = 1.0 if weight_of is None else float(weight_of(root))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("AC item weights must be finite and nonnegative")
            if weight == 0:
                self._harvest_skip("zero_weight")
                continue
            root_key = f"{config_key(root.agent_config)}:{root.seed}"
            for source_key, run in self._rollout_sources(root, root_key):
                report["runs"] += 1
                source_model = self.harness.loaded_models.get(run.agent_config.model_name)
                if source_model is None:
                    raise ValueError(f"Unknown recorded tokenizer source {run.agent_config.model_name!r}")
                source_tokenizer = source_model.tokenizer
                if not text_only and (source_tokenizer.get_vocab() != self.tokenizer.get_vocab()
                        or source_tokenizer.special_tokens_map != self.tokenizer.special_tokens_map
                        or source_tokenizer.chat_template != self.tokenizer.chat_template):
                    raise ValueError("Recorded assistant tokens require a compatible tokenizer and chat template")
                tools = self._run_tools(run)
                for segment_index, segment in enumerate([*run.compactions, run]):
                    report["candidates"] += 1
                    if text_only and segment.ac_model_name:
                        raise ValueError("Text-only rehearsal needs trajectories collected without AC")
                    chat_kwargs = ({"enable_thinking": True} if text_only else
                                   source_model.chat_template_kwargs(segment.agent_config.call_kwargs))
                    targets = []
                    for step in segment.trajectory:
                        assistants = [message for message in step.get("messages") or [] if message.get("role") == "assistant"]
                        if step.get("role") == "assistant":
                            if len(assistants) != 1:
                                raise ValueError("Recorded assistant step needs exactly one assistant message")
                            ids = (assistant_target_ids(self.tokenizer, assistants[0], tools, chat_kwargs)
                                   if text_only else list(step.get("token_ids") or []))
                            if not ids:
                                raise ValueError("Recorded assistant step needs one message and its nonempty sampled token IDs; regenerate this run")
                            targets.append(ids)
                        elif assistants:
                            raise ValueError("Recorded input step contains an assistant output")
                    if not targets:
                        self._harvest_skip("empty_segment")
                        continue
                    config = replace(segment.agent_config, ac_model_name=None if text_only else self.ac_name)
                    student = activation_messages_of(segment, config)
                    start = len(segment.prompt_messages)
                    if segment.ac_model_name:
                        recorded = [(student[:start], segment.prompt_ac_spans)]
                        cursor = start
                        for step in segment.trajectory:
                            end = cursor + len(step.get("messages") or [])
                            recorded.append((student[cursor:end], step.get("ac_spans") or []))
                            cursor = end
                        if any(len(direct_parts(messages)) != len(spans) for messages, spans in recorded):
                            raise ValueError("Recorded AC spans need matching raw activation parts; regenerate this run")
                    teacher_prefix = self._expand_parts(student[:start]) if loss_kind == "kl" else None
                    if not segment.ac_model_name and not text_only:
                        # Text runs also compress their complete initial frame, retaining raw injected parts within it.
                        frame, suffix = student[:start], student[start:]
                        system = [message for message in frame if message.get("role") == "system"]
                        part = ac_part(deepcopy(frame), self.ac_name, self.ac_model.config.default_compression_ratio)
                        if tools:
                            part["tools"] = deepcopy(tools)
                        student = system + [{"role": "user", "content": [part]}] + suffix
                        start = len(system) + 1
                    teacher = None
                    teacher_start = 0
                    if loss_kind == "kl":
                        teacher = teacher_prefix + self._expand_parts(student[start:])
                        teacher_start = len(teacher_prefix)
                    item = ActivationContextTrainingItem(
                        item_id=f"rollout:{source_key}:{segment_index}" + (":text_only" if text_only else ""),
                        kind="agent_sft" if text_only else "compaction",
                        in_context_messages=teacher, ac_messages=deepcopy(student),
                        in_context_start=teacher_start, ac_start=start, assistant_token_ids=targets,
                        tools=tools, weight=weight, dataset_id="agent_runs", doc_ids=[root_key],
                        info={"source": "agent_run", "source_key": source_key, "root_key": root_key,
                              "chat_template_kwargs": chat_kwargs, "text_only": text_only,
                              "segment": segment_index, "score": root.score,
                              "completion_source": "recorded_text_student_tokens" if text_only else "recorded",
                              "channels": sorted({part.get("kind") or "unknown" for part in direct_parts(student)}),
                              "ac_model_name": segment.ac_model_name, "ac_model_version": segment.ac_model_version})
                    ActivationContextTrainer(self.harness, ActivationContextTrainingConfig(loss_kind=loss_kind)).build_example(self.ac_model, item)
                    items.append(item)
        report["accepted"] = len(items)
        return items

    def _harvest_skip(self, reason: str) -> None:
        skipped = self.harvest_report["skipped"]
        skipped[reason] = skipped.get(reason, 0) + 1

    def _rollout_sources(self, root: "AgentRunResult", key: str) -> t.Iterator[tuple[str, "AgentRunResult"]]:
        yield key, root
        for segment_index, segment in enumerate([*root.compactions, root]):
            for child_index, child in enumerate(segment.subagent_results):
                yield from self._rollout_sources(child, f"{key}/s{segment_index}c{child_index}")

    def _run_tools(self, run: "AgentRunResult") -> list[dict] | None:
        from ..agent.agent import Agent
        agent = Agent(self.harness, run.agent_config)
        agent.dialect = self.dialect
        agent._prepare_tools()
        return agent.template_tools

    def _expand_parts(self, messages: list[dict]) -> list[dict]:
        """
        The teacher's view of messages with parts: text only, each part expanded once by its channel. A
        compaction tree becomes the earlier history it holds (its instructions/summary text is dropped); a
        tool-output part replaces the truncated text with the full output; search extras follow the visible
        passages; a subagent's segment (either direction) becomes a transcript. No message of the future is added.
        """
        from activation.common.ac_parts import strip_ac_parts
        out: list[dict] = []
        plain: list[dict] = []                                                      # out[i] with its parts removed instead of expanded
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                out.append(dict(message))
                plain.append(dict(message))
                continue
            tree = next((part for part in content if is_ac_part(part) and part.get("kind") == "compaction"), None)
            if tree is not None:
                expanded = self._expand_tree(tree)
                # A nested source frame must remain visible even when the reader has another system prompt.
                framed = []
                for entry in expanded:
                    if entry.get("role") == "system" and out:
                        if entry in out:
                            continue
                        entry = {"role": "user", "content": "Source system frame:\n" + message_text(entry)}
                    framed.append(entry)
                expanded = framed
                # The tree starts with the task the prompt already shows: drop the tree's copy, keep the prompt's (its parts expanded).
                # Nested trees held it expanded, flat views (09-26) stripped; a framed source system may come first.
                first = next((i for i, entry in enumerate(expanded) if entry.get("role") == "user"
                              and not message_text(entry).startswith("Source system frame:\n")), None)
                if out and first is not None and expanded[first] in (out[-1], plain[-1]):
                    del expanded[first]
                out.extend(expanded)
                plain.extend(expanded)
                continue
            pieces: list[str] = []
            drop_next_text = False
            for part in content:
                if is_ac_part(part):
                    kind = part.get("kind")
                    inner = part.get("messages") or []
                    if kind == "tool_output":
                        pieces.append(next((m.get("content", "") for m in inner if m.get("role") == "tool"), ""))
                        drop_next_text = True                                       # the truncated copy would repeat the output
                    elif kind == "search":                                          # old records: a tool message; 09-26: evidence, then the question
                        evidence = next((m for m in inner if m.get("role") == "tool"), inner[0] if inner else {})
                        pieces.append("\n\n" + evidence.get("content", ""))
                    elif kind == "subagent_return":
                        pieces.append("Subagent transcript:\n" + render_messages(self._expand_parts(tools_in_system_text(inner, part.get("tools")))) + "\n\n")
                    else:                                                           # subagent_prompt, parent_context, untagged: a transcript
                        pieces.append("Parent transcript:\n" + render_messages(self._expand_parts(tools_in_system_text(inner, part.get("tools")))) + "\n\n")
                elif isinstance(part, dict):
                    if drop_next_text and part.get("type") == "text":
                        drop_next_text = False
                        continue
                    pieces.append(part.get("text", ""))
                else:
                    pieces.append(str(part))
            out.append(dict(message, content="".join(pieces)))
            plain.append(strip_ac_parts([message])[0])
        return out

    def _expand_tree(self, tree: dict) -> list[dict]:
        """The history a compaction tree holds, as text: its first user message (recursively) then the segment through the compact call."""
        inner = tools_in_system_text(tree.get("messages") or [], tree.get("tools"))
        if not inner:
            return []
        expanded = self._expand_parts([inner[0]]) + self._expand_parts(inner[1:])
        return expanded

    # ------------------------------------------------------------------------------------------ trajectory QA
    def _study_examples(self, dataset_id: str, num_samples: int, seed: int, modality: DataModality, caching_id: str | None) -> list[DatasetQAExample]:
        examples = self.harness.dataset_manager.synthesize_study_examples_qa(
            dataset_id, num_samples, base_seed=seed, caching_id=caching_id, modality=modality)
        return [example for example in examples if example.positive_chunk_ids and example.gold_answers and example.gold_answers[0]]

    def _chunk_window(self, document: DatasetDocument, center: DatasetDocumentChunk | None, budget_tokens: int) -> list[dict]:
        """Messages of the document's chunks around `center` (or from the start) up to the token budget, in order."""
        chunks = sorted(document.chunks.values(), key=lambda chunk: chunk.chunk_start)
        if not chunks:
            return []
        center_index = 0 if center is None else next(i for i, chunk in enumerate(chunks) if chunk.chunk_id == center.chunk_id)
        low = high = center_index
        window = self._source_window_messages(document, chunks[low:high + 1], include_task=True)
        while True:
            grew = False
            for left, right in ((low - 1, high), (low, high + 1)):
                if left < 0 or right >= len(chunks):
                    continue
                candidate = self._source_window_messages(document, chunks[left:right + 1], include_task=True)
                if self.count_tokens(render_messages(candidate)) <= budget_tokens:
                    low, high, window, grew = left, right, candidate, True
                    break
            if not grew:
                break
        # A central structured call is indivisible: retain its evidence and expose actual size in item metadata.
        return window

    def _source_window_messages(self, document: DatasetDocument, chunks: list[DatasetDocumentChunk], *, include_task: bool = False) -> list[dict]:
        spans: dict[int, tuple[int, int, bool]] = {}
        for chunk in chunks:
            if chunk.source_spans is None:
                raise ValueError("Trajectory index has no source spans; rebuild it before study generation")
            for index, start, end, calls in chunk.source_spans:
                if index in spans:
                    old_start, old_end, old_calls = spans[index]
                    start, end, calls = min(start, old_start), max(end, old_end), calls or old_calls
                spans[index] = (start, end, calls)
        if include_task:
            task_index = next((i for i, message in enumerate(document.trajectory or []) if message.get("role") == "user"), None)
            if task_index is not None:
                spans[task_index] = (0, len(message_text(document.trajectory[task_index])), True)
        result = []
        for index, (start, end, calls) in sorted(spans.items()):
            message = deepcopy(document.trajectory[index])
            text = message_text(message)
            if start != 0 or end != len(text):
                message["content"] = text[start:end]
                message.pop("reasoning", None)
                message.pop("reasoning_content", None)
            if not calls:
                message.pop("tool_calls", None)
            result.append(message)
        return result

    def _token_boundary(self, text: str, ids: list[int], position: int) -> tuple[str, str, int]:
        for cut in range(position, 0, -1):
            prefix = self.tokenizer.decode(ids[:cut], clean_up_tokenization_spaces=False)
            suffix = self.tokenizer.decode(ids[cut:], clean_up_tokenization_spaces=False)
            if prefix + suffix == text:
                return prefix, suffix, cut
        raise ValueError("Tokenizer cannot preserve the recorded turn at a nonempty token boundary")

    @staticmethod
    def _with_source_task(document: DatasetDocument, messages: list[dict]) -> list[dict]:
        task = next((message for message in document.trajectory or [] if message.get("role") == "user"), None)
        return ([deepcopy(task)] if task is not None and (not messages or messages[0] != task) else []) + messages

    def generate_trajectory_qa_samples(
        self,
        dataset_id: str,
        num_samples: int,
        depth_range: tuple[int, int] = (1, 2),
        trajectory_tokens_range: tuple[int, int] = (8192, 32768),
        distractors_range: tuple[int, int] = (0, 2),
        ratios_range: tuple[float, float] = (1.0 / 16.0, 1.0 / 32.0),
        seed: int = 0,
        caching_id: str | None = None,
        *, encoder_sees_question: bool = False,
    ) -> list[ActivationContextTrainingItem]:
        """
        Study questions (`traj_qa`) on the dataset's trajectory chunks; per question a window of the
        source trajectory grown to a token budget, plus distractor trajectories at the same budget.
        Depth 2 wraps the window's part in a second part (a parent's view of a subagent's compaction).
        """
        rng = random.Random(seed)
        dataset = self.harness.dataset_manager.loaded_datasets[dataset_id]
        index = self.harness.dataset_manager.dataset_indexes[dataset_id]
        examples = self._study_examples(dataset_id, num_samples, seed, DataModality.TRAJECTORY, caching_id)
        max_distractors = max(distractors_range)
        ranked_docs = index.bm25_query_docs_many_frozen([example.query for example in examples], top_k=max_distractors + 4,
                                                        excluded_doc_ids=[list(example.positive_doc_ids or []) for example in examples]) if examples and max_distractors else [[] for _ in examples]
        items = []
        for item_index, (example, candidates) in enumerate(zip(examples, ranked_docs)):
            chunk = index.chunks[example.positive_chunk_ids[0]]
            document = dataset.documents[chunk.doc_id]
            budget = rng.randint(min(trajectory_tokens_range), max(trajectory_tokens_range))
            window = self._with_source_task(document, self._chunk_window(document, chunk, budget))
            num_distractors = min(rng.randint(min(distractors_range), max_distractors), len(candidates))
            distractors = [self._with_source_task(candidate, self._chunk_window(candidate, None, budget)) for candidate in candidates[:num_distractors]]
            ratio = self._sample_ratio(rng, ratios_range)
            depth = rng.randint(min(depth_range), max(depth_range))
            gold_position = rng.randrange(num_distractors + 1)
            item = self.trajectory_qa_item(document, window, example.query, example.gold_answers[0],
                item_id=f"traj_qa:{dataset_id}:{seed}:{item_index}", compression_ratio=ratio, depth=depth,
                distractors=list(zip(candidates[:num_distractors], distractors)), gold_position=gold_position,
                encoder_sees_question=encoder_sees_question)
            item.info.update(budget=budget, example_id=example.example_id, gold_chunk_text=chunk.chunk_text,
                             window_tokens=self.count_tokens(render_messages(window)),
                             window_over_budget=self.count_tokens(render_messages(window)) > budget)
            items.append(item)
        return items

    def trajectory_qa_item(
        self, document: DatasetDocument, messages: list[dict], question: str, answer: str, *,
        item_id: str, compression_ratio: float = 1.0 / 16.0, depth: int = 1,
        distractors: t.Sequence[tuple[DatasetDocument, list[dict]]] = (), gold_position: int = 0,
        encoder_sees_question: bool = False, parent_messages: list[dict] | None = None,
    ) -> ActivationContextTrainingItem:
        """Pair a history with QA; optional compressed parent conditions the fresh-history encoder.

        Select and freeze `messages` before synthesizing questions for a question-blind study. The
        full-context reference receives the same source frames, histories and distractor order.
        """
        if not messages or not 0 <= gold_position <= len(distractors) or depth < 1:
            raise ValueError("Trajectory QA needs a nonempty history, positive depth and valid source position")
        sources = list(distractors)
        sources.insert(gold_position, (document, messages))
        prefix = f"Task: {question}\n\n"
        suffix = f"{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"
        content = [{"type": "text", "text": prefix}]
        full = [prefix]
        from .ac_model_utils import qa_encoder_messages
        if parent_messages is not None and depth != 1:
            raise ValueError("parent_messages supplies the recursive level; keep depth=1")
        encoder_question = qa_encoder_messages([], question, self.ac_name, compression_ratio, encoder_sees_question=encoder_sees_question)
        for number, (source, history) in enumerate(sources, start=1):
            kwargs = source.trajectory_kwargs or {}
            history = self._with_source_task(source, history)
            frame = self._system_messages(source)
            part = ac_part(qa_encoder_messages(frame + history, question, self.ac_name, compression_ratio,
                encoder_sees_question=encoder_sees_question, parent_messages=parent_messages), self.ac_name, compression_ratio)
            if kwargs.get("tools"):
                part["tools"] = deepcopy(kwargs["tools"])
            if depth >= 2:
                part = ac_part(deepcopy(frame + [history[0]]) + [{"role": "user", "content": [part]}] + deepcopy(encoder_question),
                               self.ac_name, compression_ratio)
                if kwargs.get("tools"):
                    part["tools"] = deepcopy(kwargs["tools"])
            label = f"Trajectory {number}:\n"
            content.extend([{"type": "text", "text": label}, part, {"type": "text", "text": "\n\n"}])
            source_frame = tools_in_system_text(deepcopy(frame + history), kwargs.get("tools"))
            full.extend([label, render_messages(deepcopy(parent_messages or []) + source_frame), "\n\n"])
        content.append({"type": "text", "text": suffix})
        full.append(suffix)
        tools = [submit_answer_definition()]
        output = self.dialect.rendering.assistant_message("", [{"id": "answer", "name": "submit_answer", "arguments": {"answer": answer.strip()}}])
        system = [{"role": "system", "content": TRAJ_QA_SYSTEM_PROMPT}]
        return ActivationContextTrainingItem(
            item_id=item_id, kind="traj_qa",
            in_context_messages=system + [{"role": "user", "content": "".join(full)}, output],
            ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)],
            in_context_start=len(system) + 1, ac_start=len(system) + 1,
            assistant_token_ids=[assistant_target_ids(self.tokenizer, output, tools)], tools=tools,
            dataset_id=document.dataset_id,
            doc_ids=[document.doc_id] + [source.doc_id for source, _ in distractors],
            info={"depth": depth, "distractors": len(distractors), "ratio": compression_ratio,
                  "gold_position": gold_position, "question": question, "answer": answer.strip(),
                  "matched_full_context": True, "encoder_sees_question": encoder_sees_question,
                  "encoder_conditioning": "parent" if parent_messages is not None else "question" if encoder_sees_question else "summary"},
        )

    # ------------------------------------------------------------------------------------------ RAG QA
    def generate_rag_qa_samples(
        self,
        dataset_id: str,
        num_samples: int,
        depth_range: tuple[int, int] = (1, 2),
        distractors_token_range: tuple[int, int] = (8192, 32768),
        ratios_range: tuple[float, float] = (1.0 / 16.0, 1.0 / 32.0),
        seed: int = 0,
        caching_id: str | None = None,
        *, source_mode: t.Literal["chunk", "window", "document", "related_chunks"] = "chunk",
        max_source_tokens: int | None = None, max_source_chunks: int = 3,
        prompt_kind: t.Literal["qa", "code", "numeric", "constraints"] = "qa",
        require_multiple_sources: bool = False, max_questions_per_source: int = 1,
        study_context: str | None = None, progress: t.Callable[[dict], None] | None = None,
        encoder_sees_question: bool = True, parent_messages: list[dict] | None = None,
    ) -> list[ActivationContextTrainingItem]:
        """Synthesize questions, retain all selected evidence, and match AC/full-context distractors."""
        from .ac_model_utils import qa_encoder_messages
        if parent_messages is not None and depth_range != (1, 1):
            raise ValueError("parent_messages supplies the recursive level; use depth_range=(1, 1)")
        rng = random.Random(seed)
        index = self.harness.dataset_manager.dataset_indexes[dataset_id]
        examples = self.harness.dataset_manager.synthesize_study_examples_qa(
            dataset_id, num_samples, base_seed=seed, caching_id=caching_id, modality=DataModality.TEXT,
            source_mode=source_mode, max_source_tokens=max_source_tokens, max_source_chunks=max_source_chunks,
            prompt_kind=prompt_kind, require_multiple_sources=require_multiple_sources, max_questions_per_source=max_questions_per_source,
            study_context=study_context, progress=progress)
        ranked = index.bm25_query_many_frozen([example.query for example in examples], top_k=64,
                                              excluded_doc_ids=[list(dict.fromkeys([*(example.positive_doc_ids or []),
                                                  *(u["doc_id"] for u in (example.study_source or {}).get("units", []))])) for example in examples]) if examples else []
        items = []
        for item_index, (example, candidates) in enumerate(zip(examples, ranked)):
            if not example.gold_answers or not example.positive_chunk_ids:
                continue
            source = example.study_source
            if source:
                positives = [DatasetDocumentChunk(f"{example.example_id}:source:{i}", u["doc_id"], dataset_id, u["text"], 0)
                             for i, u in enumerate(source["units"])]
            else:
                positives = [index.chunks[key] for key in example.positive_chunk_ids]
            gold = positives[0]
            budget = rng.randint(min(distractors_token_range), max(distractors_token_range))
            passages = list(positives)
            total = sum(self.count_tokens(p.chunk_text) for p in positives)
            for candidate in candidates:
                cost = self.count_tokens(candidate.chunk_text)
                if total + cost > budget:
                    continue
                passages.append(candidate)
                total += cost
            rng.shuffle(passages)
            ratio = self._sample_ratio(rng, ratios_range)
            depth = rng.randint(min(depth_range), max(depth_range))
            question = example.query
            encoder_task = qa_encoder_messages([], question, self.ac_name, ratio, encoder_sees_question=encoder_sees_question)
            passages_text = "\n\n".join(f"Passage {number}:\n{passage.chunk_text}" for number, passage in enumerate(passages, start=1))
            part = ac_part(qa_encoder_messages([{ "role": "user", "content": passages_text}], question, self.ac_name, ratio,
                encoder_sees_question=encoder_sees_question, parent_messages=parent_messages), self.ac_name, ratio)
            if depth >= 2:
                part = ac_part(deepcopy(encoder_task) + [{"role": "user", "content": [part]}], self.ac_name, ratio)
            content = [{"type": "text", "text": f"Task: {question}\n\nPassages:\n"}, part,
                       {"type": "text", "text": f"\n\n{RAG_QA_INSTRUCTIONS}\nQuestion: {question}"}]
            parent_text = render_messages(parent_messages) + "\n\n" if parent_messages else ""
            in_context_user = {"role": "user", "content": parent_text + f"Task: {question}\n\nPassages:\n{passages_text}\n\n{RAG_QA_INSTRUCTIONS}\nQuestion: {question}"}
            system = [{"role": "system", "content": RAG_QA_SYSTEM_PROMPT}]
            output = self.dialect.rendering.assistant_message("", [{"id": "answer", "name": "submit_answer", "arguments": {"answer": example.gold_answers[0].strip()}}])
            items.append(ActivationContextTrainingItem(
                item_id=f"rag_qa:{dataset_id}:{seed}:{item_index}",
                kind="rag_qa",
                in_context_messages=system + [in_context_user, output],
                ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)],
                in_context_start=len(system) + 1, ac_start=len(system) + 1,
                assistant_token_ids=[assistant_target_ids(self.tokenizer, output, [submit_answer_definition()])],
                tools=[submit_answer_definition()],
                dataset_id=dataset_id,
                doc_ids=sorted({passage.doc_id for passage in passages}),
                info={"depth": depth, "budget": budget, "passages": len(passages), "ratio": ratio,
                      "gold_position": passages.index(gold), "example_id": example.example_id, "gold_chunk_text": "\n\n".join(p.chunk_text for p in positives), "answer": example.gold_answers[0],
                      "passages_text": passages_text, "study_source": source, "matched_full_context": True,
                      "encoder_sees_question": encoder_sees_question,
                      "encoder_conditioning": "parent" if parent_messages is not None else "question" if encoder_sees_question else "summary"},
            ))
        return items

    def compose_reconstruction_inputs(self, passages: list[dict], count: int, *, token_range: tuple[int, int] = (1024, 16384),
                                      bm25_fraction: float = .5, seed: int = 0, neighbours: int = 128,
                                      query_words: int = 64, targets: list[tuple[int, t.Any]] | None = None) -> list[dict]:
        """
        `count` long multi-passage inputs for generate_reconstruction_samples, from a pool of {"text", "doc_id", "source"}
        passages ("tokens" is used when present, else the reader tokenizer counts). Each input draws a token budget
        log-uniform in `token_range` and a seed passage, then fills the budget with distinct passages that fit: either the
        seed's BM25 neighbours in rank order (the seed's first `query_words` words as the query: same-topic distractors, as in
        RAG retrieval), with probability `bm25_fraction`, or uniformly random ones (unrelated neighbours, as in long
        trajectories). The passage order is then shuffled. A seed filling more than half its budget is redrawn (up to 16
        times; a seed longer than its budget stands alone). Returns
        {"texts", "doc_ids", "sources", "source": "composed", "composition", "budget_tokens", "input_tokens"} (input tokens
        count the "Passage i:" headers).
        `targets` (default None: the draws above, unchanged): one (pool index, seed) pair per input, `count` of them. The pair's
        passage is that input's seed passage and the pair's seed alone drives the input's draws (budget, composition, fill,
        order), so an input is the same whatever else the call composes. A fixed seed passage cannot be redrawn: the budget is
        redrawn instead (up to 16 times; then the range's top). Such inputs also carry "target_index" (the seed passage's position after the
        shuffle, honoured by generate_reconstruction_samples), "target_pool_index" and "pool_indices".
        """
        low, high = token_range
        if not 0 < low <= high:
            raise ValueError("token_range must be (low, high) with 0 < low <= high")
        if not 0 <= bm25_fraction <= 1:
            raise ValueError("bm25_fraction must be in [0, 1]")
        if not passages or count < 0:
            raise ValueError("Need passages and a nonnegative count")
        if targets is not None and len(targets) != count:
            raise ValueError("targets needs one (pool index, seed) pair per input")
        rng = random.Random(seed)
        texts = [passage["text"].strip() for passage in passages]
        lengths = [passage.get("tokens") for passage in passages]
        missing = [index for index, length in enumerate(lengths) if length is None]
        for start in range(0, len(missing), 4096):
            chunk = missing[start:start + 4096]
            for index, ids in zip(chunk, self.tokenizer([texts[i] for i in chunk], add_special_tokens=False)["input_ids"]):
                lengths[index] = len(ids)
        header = len(self.tokenizer.encode("\n\nPassage 10:\n", add_special_tokens=False))
        plans = []
        for _ in range(count if targets is None else 0):
            budget, bm25, first = math.exp(rng.uniform(math.log(low), math.log(high))), rng.random() < bm25_fraction, rng.randrange(len(texts))
            for _ in range(16):   # a seed filling more than half the budget leaves no room for neighbours: redraw it
                if lengths[first] + header <= budget / 2:
                    break
                first = rng.randrange(len(texts))
            plans.append((budget, bm25, first))
        rngs = [random.Random(f"{seed}:{own}") for _, own in targets or []]   # targets: one rng per input (its plan, then its fill)
        for (first, _), own in zip(targets or [], rngs):
            budget, bm25 = math.exp(own.uniform(math.log(low), math.log(high))), own.random() < bm25_fraction
            for _ in range(16):   # the seed passage is fixed: redraw the budget instead
                if lengths[first] + header <= budget / 2:
                    break
                budget = math.exp(own.uniform(math.log(low), math.log(high)))
            else:
                budget = float(high)
            plans.append((budget, bm25, first))
        ranked = {}
        seeds = sorted({first for _, bm25, first in plans if bm25})
        if seeds:
            from ..common.bm25 import BM25Index
            _, found = BM25Index(texts).search([" ".join(texts[s].split()[:query_words]) for s in seeds], neighbours + 1)
            ranked = {s: [int(j) for j in row if j >= 0 and j != s] for s, row in zip(seeds, found)}
        inputs = []
        for index, (budget, bm25, first) in enumerate(plans):
            rng = rngs[index] if targets is not None else rng
            chosen, seen, total, misses = [first], {texts[first]}, lengths[first] + header, 0
            candidates = iter(ranked[first]) if bm25 else iter(lambda: rng.randrange(len(texts)), None)
            for j in candidates:
                if total >= budget - header or misses >= 32:
                    break
                if texts[j] in seen or total + lengths[j] + header > budget:
                    misses += 1
                    continue
                chosen.append(j)
                seen.add(texts[j])
                total += lengths[j] + header
                misses = 0
            rng.shuffle(chosen)
            inputs.append({"texts": [texts[j] for j in chosen], "doc_ids": [passages[j].get("doc_id") for j in chosen],
                           "sources": [passages[j].get("source") for j in chosen], "source": "composed",
                           "composition": "bm25" if bm25 else "random", "budget_tokens": round(budget), "input_tokens": total})
            if targets is not None:
                inputs[-1].update(target_index=chosen.index(first), target_pool_index=first, pool_indices=list(chosen))
        return inputs

    def generate_reconstruction_samples(
        self,
        passages: list[dict],
        ratio: float | None = None,
        variant: str = "reconstruct",
        cue_fraction: float = 0.2,
        seed: int = 0,
        system_prompt: str = RECONSTRUCTION_SYSTEM_PROMPT,
        *, ratios_range: tuple[float, float] = (1.0 / 10.0, 1.0 / 30.0),
        cue_tokens: int | tuple[int, int] = 24, target_tokens: tuple[int, int] = (128, 256), mask_stop: bool = False,
        density_power: float = 0., decoy: float = 0., reads: tuple[int, int] = (2, 5), boundary_share: float = 0.25,
        multispan_passages: bool = False,
    ) -> list[ActivationContextTrainingItem]:
        """
        Dense, oracle-free items: the reader sees text only through its compressed rows and reproduces some of it.
        `passages` are {"text", "doc_id", "source"} (any text; no dataset registration needed), or {"texts": [...], "doc_ids",
        "source"} for several passages read as ONE part in the RAG evidence format ("Passage 1:\\n...\\n\\nPassage 2:\\n...";
        compose_reconstruction_inputs builds long ones). Variants (the item's kind is the variant name):
          "reconstruct"  the whole input is the target;
          "continue"     the first `cue_fraction` of the input stays in plain text after the part and the rest is the target;
          "passage"      (several passages) reproduce Passage k, k from the input's "target_index" or drawn uniformly;
          "lookup"       a `cue_tokens`-token cue from anywhere inside one passage (k and the start drawn uniformly) follows
                         the part, and the reader continues it for a length drawn in `target_tokens`, stopping at that
                         passage's end; passages shorter than the cue plus `target_tokens[0]` are never chosen, and a cue
                         occurring more than once in the input is redrawn (an input without a usable cue yields no item);
          "span"         a lookup with a second cue marking the end: the reader reproduces the text between the two cues
                         (`target_tokens` tokens of it, both cues unique in the input and inside the passage) and stops
                         right before the end cue, so the end of the target is predictable.
        `cue_tokens` is a fixed cue length or a (low, high) range drawn per cue. `mask_stop` drops the end-of-turn token from
        "lookup" targets: their length is drawn, so the stop position is unpredictable and its loss is noise.
        `density_power` > 0 points "passage", "lookup" and "span" targets at dense material: the target passage k ("passage")
        or the (passage, cue start) pair ("lookup"/"span") is drawn with weight (DENSITY_FLOOR + dense share)^power, the dense
        share being the fraction of tokens holding a digit, a code/math symbol or a non-ASCII non-letter (_dense_mask) in the
        passage, or in the window from the cue start over the cue, the longest target and the end cue. At 0 (the default)
        every draw is the uniform one above, in the same rng order, so earlier packets reproduce exactly. Items record the
        drawn window's dense share (cue, target and end cue; the passage for "passage") as info["window_density"] whatever the power.
        `decoy` > 0 is the share of multi-passage "passage"/"lookup"/"span" items (drawn after the target) whose input gets a
        digit-perturbed twin of the target passage (_perturb_digits: every digit run redrawn, same length) in place of one
        other passage, so the reader must tell the right copy from its near-identical neighbour; a target passage without
        digits, or a cue the twin also contains, leaves the item unchanged. The twin's doc id is the target's + "#decoy";
        info["decoy_index"] is the twin's position. At 0 (the default) no rng call is added.
        "multispan" reads one passage alone (no multi-passage input): `reads` (low, high) non-overlapping spans of `target_tokens`
        tokens each, listed in random order (not passage order) and reproduced in that order joined by MULTISPAN_SEPARATOR. A read
        is "mid" (from a `cue_tokens` start cue up to an end cue, the span variant's form), "start" (from the passage's beginning up
        to an end cue: the opening is never shown) or "end" (from a start cue to the passage's end): the start read and the end read
        each appear with probability `boundary_share`. Every cue is unique in the passage; passages holding the separator are
        skipped. Every read ends at a defined boundary, so the stop position is predictable. info["reads"] lists the reads in
        passage order (kind, start token, cue/target/end-cue tokens), info["read_order"] the listed order.
        `multispan_passages` (default False: a multi-passage input raises, as before) lets "multispan" read a multi-passage input:
        every read lies inside ONE passage (the input's "target_index", else drawn among the passages long enough for one read),
        all reads are "mid" reads (cue to end cue; a passage's beginning or end is not addressable among several), every cue is
        unique in the whole input, and the instruction is MULTISPAN_PASSAGES_INSTRUCTION. Only the target passage is tokenized.
        info["passage_index"] is that passage; read positions count its own tokens.
        The in-context view holds the plain text, so a KL teacher would copy; these items are meant for SFT. Item ids hash
        the item's content (variant, source, doc ids, input, ratio, instruction, target), so the same sample always gets the
        same id and a repeat meant to count twice needs its own doc id.

        Nested (two-level) passages carry `segments` instead of `text` ("reconstruct" and "continue" only): an ordered list of
        {"kind": "text", "text"} and {"kind": "child", "text", "ratio"} entries, plus `ratio` (the passage's own compression;
        the call's `ratio` when absent) and optionally `variant_only` (the passage is rendered in that variant alone). The
        part's message content is then a list: the text segments verbatim and one activation_context part per child segment
        (the child's text at the child's ratio), in order, so the encoder reads its own rows next to text. The plain view and
        the targets use the concatenation of all segment texts; the "continue" cue ends `cue_fraction` into the first child,
        so the continuation crosses from the child's rows into the parent's text. Flat passages render as before.
        """
        if variant not in ("reconstruct", "continue", "passage", "lookup", "span", "multispan"):
            raise ValueError("variant must be 'reconstruct', 'continue', 'passage', 'lookup', 'span' or 'multispan'")
        if not 0 < cue_fraction < 1:
            raise ValueError("cue_fraction must be in (0, 1)")
        cue_range = (cue_tokens, cue_tokens) if isinstance(cue_tokens, int) else tuple(cue_tokens)
        if len(cue_range) != 2 or not 0 < cue_range[0] <= cue_range[1] or not 0 < target_tokens[0] <= target_tokens[1]:
            raise ValueError("cue_tokens must be a positive length or a (low, high) range and target_tokens (low, high) with 0 < low <= high")
        if mask_stop and variant != "lookup":
            raise ValueError("mask_stop applies to the 'lookup' variant (a span's end is predictable)")
        if not density_power >= 0:
            raise ValueError("density_power must be a nonnegative number (0: uniform draws)")
        if not 0 <= decoy <= 1:
            raise ValueError("decoy must be a share in [0, 1]")
        if not 0 < reads[0] <= reads[1] or not 0 <= boundary_share <= 1:
            raise ValueError("reads must be a positive (low, high) range and boundary_share a share in [0, 1]")
        rng = random.Random(seed)
        items = []
        for passage in passages:
            if passage.get("variant_only") not in (None, variant):
                continue
            segments = passage.get("segments")
            texts = [text.strip() for text in passage["texts"]] if "texts" in passage else None
            if texts is not None and (segments or not texts or not all(texts)):
                raise ValueError("A multi-passage input needs non-empty texts and no segments")
            if segments and variant in ("passage", "lookup", "span", "multispan"):
                raise ValueError(f"Nested passages support 'reconstruct' and 'continue', not {variant!r}")
            if variant == "passage" and texts is None:
                raise ValueError("The 'passage' variant needs a multi-passage input (texts)")
            if texts is not None:
                text = join_passages(texts)
            else:
                text = "".join(segment["text"] for segment in segments) if segments else passage["text"].strip()
            if not text:
                continue
            shown = texts if texts is not None else [text]
            starts = [0] if texts is None else [len(join_passages(texts[:k])) + (2 if k else 0) + len(f"Passage {k + 1}:\n") for k in range(len(texts))]
            child = _first_child_span(segments) if segments else None
            part_ratio = passage.get("ratio", ratio)
            if part_ratio is None:
                part_ratio = self._sample_ratio(rng, ratios_range)
            info = {"children": sum(1 for s in segments if s["kind"] == "child"), "child_ratios": [s["ratio"] for s in segments if s["kind"] == "child"]} if segments else {}
            if variant == "continue":
                cut = _cue_cut(text, cue_fraction) if child is None else child[0] + _cue_cut(text[child[0]:child[1]], cue_fraction)
                cue, target = text[:cut].rstrip(), text[cut:].lstrip()
                if not cue or not target or (child is not None and not child[0] < cut < child[1]):
                    continue
                instruction = f"The passage above begins with:\n{cue}\n\nContinue the passage from there, exactly as written, and stop at its end."
                output_text = target
                if segments:
                    info["cue_chars"] = cut
            elif variant == "passage":
                k = passage.get("target_index")
                cue_strings, pos_in = [], 0   # no cue to keep unique; the target starts the passage
                mask_of = lambda p: _dense_mask(p, self.tokenizer(p, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"])
                if k is None and density_power:   # dense passages drawn more often; the uniform draw below is untouched at power 0
                    k = _weighted_index(rng, [(DENSITY_FLOOR + sum(m) / max(1, len(m))) ** density_power for m in map(mask_of, texts)])
                k = rng.randrange(len(texts)) if k is None else k
                instruction, output_text = PASSAGE_INSTRUCTION.format(k=k + 1), texts[k]
                mask = mask_of(texts[k])
                info.update(passage_index=k, target_position=round(starts[k] / len(text), 4), window_density=round(sum(mask) / max(1, len(mask)), 4))
            elif variant in ("lookup", "span"):
                offsets = [self.tokenizer(p, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"] for p in shown]
                cues = 2 if variant == "span" else 1   # a span's end cue must also fit inside the passage
                eligible = [k for k, o in enumerate(offsets) if len(o) >= cues * cue_range[1] + target_tokens[0]
                            and passage.get("target_index") in (None, k)]
                if not eligible:
                    continue
                draw_cue = lambda: cue_range[0] if cue_range[0] == cue_range[1] else rng.randint(*cue_range)   # a fixed cue draws nothing: earlier packets' lookups keep their draws
                masks = {k: _dense_mask(shown[k], offsets[k]) for k in eligible}
                for _ in range(8):   # a cue occurring twice in the input (shared boilerplate) is ambiguous: redraw it
                    if density_power:   # cue lengths first, then one weighted draw of (passage, start) over every admissible start
                        cue_len, end_len = draw_cue(), draw_cue() if variant == "span" else 0
                        window = cue_len + target_tokens[1] + end_len
                        slices = [(k, _window_weights(masks[k], window, density_power, len(offsets[k]) - cue_len - end_len - target_tokens[0] + 1)) for k in eligible]
                        flat = _weighted_index(rng, [w for _, weights in slices for w in weights])
                        for k, weights in slices:
                            if flat < len(weights):
                                start = flat
                                break
                            flat -= len(weights)
                        length = rng.randint(target_tokens[0], target_tokens[1] if variant == "lookup" else min(target_tokens[1], len(offsets[k]) - start - cue_len - end_len))
                    else:
                        k = rng.choice(eligible)
                        cue_len, end_len = draw_cue(), draw_cue() if variant == "span" else 0
                        if variant == "lookup":   # length then start, as always; a lookup may run past the passage end (cut clamps)
                            length, start = rng.randint(*target_tokens), rng.randrange(len(offsets[k]) - cue_len - target_tokens[0] + 1)
                        else:                     # a span never does: the end cue must fit after the target
                            start = rng.randrange(len(offsets[k]) - cue_len - end_len - target_tokens[0] + 1)
                            length = rng.randint(target_tokens[0], min(target_tokens[1], len(offsets[k]) - start - cue_len - end_len))
                    cut = lambda i, k=k: offsets[k][i][0] if i < len(offsets[k]) else len(shown[k])   # token starts: no split characters
                    cue, output_text = shown[k][cut(start):cut(start + cue_len)], shown[k][cut(start + cue_len):cut(start + cue_len + length)]
                    end_cue = shown[k][cut(start + cue_len + length):cut(start + cue_len + length + end_len)] if variant == "span" else ""
                    if cue.strip() and output_text.strip() and text.count(cue) == 1 and (variant == "lookup" or (end_cue.strip() and text.count(end_cue) == 1)):
                        break
                else:
                    continue
                instruction = LOOKUP_INSTRUCTION.format(cue=cue) if variant == "lookup" else SPAN_INSTRUCTION.format(cue=cue, end_cue=end_cue)
                cue_strings, pos_in = [cue] + ([end_cue] if variant == "span" else []), cut(start + cue_len)   # for the decoy below
                span_mask = masks[k][start:start + cue_len + length + end_len]
                info.update(passage_index=k, cue_start=start, cue_tokens=cue_len, target_position=round((starts[k] + cut(start + cue_len)) / len(text), 4),
                            window_density=round(sum(span_mask) / max(1, len(span_mask)), 4),
                            **({"end_cue_tokens": end_len} if variant == "span" else {"stop_masked": True} if mask_stop else {}))
            elif variant == "multispan":
                if texts is not None and not multispan_passages:
                    raise ValueError("The 'multispan' variant reads one passage alone (no multi-passage input)")
                body, origin = text, 0   # the passage the reads lie in (and its character offset): the input itself, or one passage of a multi-passage input
                if texts is not None:
                    k = passage.get("target_index")
                    if k is None:
                        long_enough = [i for i, p in enumerate(texts) if self.count_tokens(p) >= 2 * cue_range[1] + target_tokens[0]]
                        if not long_enough:
                            continue
                        k = rng.choice(long_enough)
                    body, origin = texts[k], starts[k]
                if MULTISPAN_SEPARATOR in body:
                    continue
                offsets = self.tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
                n = len(offsets)
                cut = lambda i: offsets[i][0] if i < n else len(body)
                draw_cue = lambda: cue_range[0] if cue_range[0] == cue_range[1] else rng.randint(*cue_range)
                width = lambda run: run[1] + run[2] + run[3]
                for _ in range(8):   # a cue occurring twice in the passage is ambiguous: redraw the reads
                    count = rng.randint(*reads)
                    kinds = (["start"] * (rng.random() < boundary_share) + ["end"] * (rng.random() < boundary_share))[:count] if texts is None else []
                    kinds += ["mid"] * (count - len(kinds))
                    # Each read is the token run [cue][target][end cue]; the start read begins at 0 without a cue, the end read ends
                    # at the passage end without an end cue, the mid reads sit at random non-overlapping places in between. A short
                    # passage keeps as many reads as fit (mid reads dropped first, then the end read).
                    runs = [[kind, 0 if kind == "start" else draw_cue(), rng.randint(*target_tokens), 0 if kind == "end" else draw_cue()] for kind in kinds]
                    while len(runs) > 1 and sum(width(r) for r in runs) > n:
                        runs.pop(max(range(len(runs)), key=lambda i: (runs[i][0] == "mid", runs[i][0] == "end", i)))
                    low = sum(width(r) for r in runs if r[0] == "start")
                    high = n - sum(width(r) for r in runs if r[0] == "end")
                    mids = [r for r in runs if r[0] == "mid"]
                    free = high - low - sum(width(r) for r in mids)
                    if free < 0:
                        parts = None
                        continue
                    gaps = sorted(rng.randint(0, free) for _ in mids)   # the mids' offsets inside the free space, in passage order
                    starts, used = {}, 0
                    for r, gap in zip(mids, gaps):
                        starts[id(r)] = low + gap + used
                        used += width(r)
                    parts = []   # (kind, start token, cue, target, end cue, cue tokens, target tokens, end-cue tokens) in passage order
                    for r in runs:
                        kind, cue_len, length, end_len = r
                        start = 0 if kind == "start" else n - width(r) if kind == "end" else starts[id(r)]
                        parts.append((kind, start, body[cut(start):cut(start + cue_len)], body[cut(start + cue_len):cut(start + cue_len + length)],
                                      body[cut(start + cue_len + length):cut(start + cue_len + length + end_len)], cue_len, length, end_len))
                    parts.sort(key=lambda part: part[1])
                    cue_texts = [part[2] for part in parts if part[5]] + [part[4] for part in parts if part[7]]
                    if (all(part[3].strip() for part in parts) and all(c.strip() and text.count(c) == 1 for c in cue_texts)
                            and len(set(cue_texts)) == len(cue_texts)):
                        break
                else:
                    continue
                order = rng.sample(range(len(parts)), len(parts))   # the listed order: random, not the passage order
                lines = [MULTISPAN_READS[parts[j][0]].format(i=i + 1, cue=parts[j][2], end_cue=parts[j][4]) for i, j in enumerate(order)]
                instruction = (MULTISPAN_INSTRUCTION if texts is None else MULTISPAN_PASSAGES_INSTRUCTION).format(reads="\n".join(lines))
                output_text = MULTISPAN_SEPARATOR.join(parts[j][3] for j in order)
                mask = _dense_mask(body, offsets)
                read_tokens = sum(part[6] for part in parts)
                if texts is not None:
                    info["passage_index"] = k
                info.update(reads=[{"kind": part[0], "start": part[1], "cue_tokens": part[5], "tokens": part[6], "end_cue_tokens": part[7]} for part in parts],
                            read_order=order, target_position=round((origin + cut(parts[order[0]][1] + parts[order[0]][5])) / len(text), 4),
                            window_density=round(sum(sum(mask[part[1] + part[5]:part[1] + part[5] + part[6]]) for part in parts) / max(1, read_tokens), 4))
            else:
                instruction = RECONSTRUCTION_INSTRUCTION
                output_text = text
            doc_ids = [d for d in passage.get("doc_ids") or [] if d] if texts is not None else [passage["doc_id"]] if passage.get("doc_id") else []
            if decoy and texts is not None and len(texts) >= 2 and variant in ("passage", "lookup", "span") and rng.random() < decoy:
                # A digit-perturbed twin of the target passage replaces one other passage: the reader must tell the right copy from
                # its near-identical neighbour (the index names it for "passage"; the cue's own digits for "lookup"/"span", so a cue
                # the twin also contains is ambiguous and the item keeps its original input).
                twin = _perturb_digits(shown[k], rng)
                j = rng.choice([i for i in range(len(texts)) if i != k])
                if twin is not None and all(join_passages(shown[:j] + [twin] + shown[j + 1:]).count(c) == 1 for c in cue_strings):
                    shown = shown[:j] + [twin] + shown[j + 1:]
                    text = join_passages(shown)
                    starts = [len(join_passages(shown[:i])) + (2 if i else 0) + len(f"Passage {i + 1}:\n") for i in range(len(shown))]
                    if len(doc_ids) == len(texts):
                        doc_ids[j] = f"{doc_ids[k]}#decoy"
                    info.update(decoy_index=j, target_position=round((starts[k] + pos_in) / len(text), 4))
            if texts is not None:
                info.update(passages=len(texts), composition=passage.get("composition"), sources=passage.get("sources"),
                            budget_tokens=passage.get("budget_tokens"))
            label = "Passages:\n" if texts is not None else "Passage:\n"
            part = ac_part([{"role": "user", "content": _segment_content(segments, self.ac_name) if segments else text}], self.ac_name, part_ratio)
            content = [{"type": "text", "text": label}, part, {"type": "text", "text": f"\n\n{instruction}"}]
            output = {"role": "assistant", "content": output_text}
            system = [{"role": "system", "content": system_prompt}]
            source = passage.get("source", "text")
            key = hashlib.sha256(json.dumps([variant, source, doc_ids, segments or shown, part_ratio, system_prompt, instruction, output_text]
                                            + (["mask_stop"] if mask_stop else []), ensure_ascii=False).encode()).hexdigest()[:20]
            target_ids = assistant_target_ids(self.tokenizer, output)
            if mask_stop:
                target_ids = without_stop_token(self.tokenizer, target_ids)
            items.append(ActivationContextTrainingItem(
                item_id=f"{variant}:{source}:{key}",
                kind=variant,
                in_context_messages=system + [{"role": "user", "content": f"{label}{text}\n\n{instruction}"}, output],
                ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)],
                in_context_start=len(system) + 1, ac_start=len(system) + 1,
                assistant_token_ids=[target_ids],
                tools=None,
                dataset_id=passage.get("source", ""),
                doc_ids=doc_ids,
                info={"ratio": part_ratio, "variant": variant, "chars": len(text),
                      "passage_tokens": len(self.tokenizer.encode(text, add_special_tokens=False)) if texts is not None else self.count_tokens(text), **info},
            ))
        return items

    def multi_target_item(self, items: t.Sequence[ActivationContextTrainingItem], *, item_id: str) -> ActivationContextTrainingItem:
        """Read two or three separate AC inputs and return their gold responses in order."""
        from ..common.ac_parts import COMPACTION_SUMMARY_PROMPT
        if len(items) not in (2, 3) or len({item.kind for item in items}) != 1:
            raise ValueError("Combine two or three items of the same task")
        system = [{"role": "system", "content":
            "Complete each numbered request using its own context. Return responses in order, separated by a line containing ---. Do not call tools."}]
        content, full, answers = [], [], []
        for number, item in enumerate(items, 1):
            parts = list(direct_parts(item.ac_messages[:item.ac_start]))
            if not parts or item.info.get("multi_target_count"):
                raise ValueError("Each constituent needs its own AC context and one target")
            if item.kind in ("rag_qa", "traj_qa"):
                request, answer = "Answer: " + item.info["question"], item.info["answer"]
            elif item.kind == "compaction_summary":
                request, answer = COMPACTION_SUMMARY_PROMPT, message_text(item.ac_messages[item.ac_start])
            elif item.kind == "reconstruct":
                request, answer = RECONSTRUCTION_INSTRUCTION, message_text(item.ac_messages[item.ac_start])
            else:
                raise ValueError("Multi-target training supports QA, summaries and reconstruction")
            label = f"Request {number}: {request}\nContext:\n"
            content.append({"type": "text", "text": label})
            content.extend(deepcopy(parts))
            content.append({"type": "text", "text": "\n\n"})
            plain = answer if item.kind == "reconstruct" else "\n\n".join(
                render_messages(tools_in_system_text(_expand_parts(part["messages"]), part.get("tools"))) for part in parts)
            full.append(label + plain + "\n\n")
            answers.append(answer)
        output = {"role": "assistant", "content": "\n---\n".join(answers)}
        kwargs = {"enable_thinking": False}
        return ActivationContextTrainingItem(item_id, items[0].kind + ":multi",
            system + [{"role": "user", "content": "".join(full)}, deepcopy(output)],
            system + [{"role": "user", "content": content}, deepcopy(output)], 2, 2,
            [assistant_target_ids(self.tokenizer, output, None, kwargs)],
            info={"family": items[0].info.get("family"), "multi_target_count": len(items),
                  "constituent_ids": [item.item_id for item in items], "source": "multiple",
                  "chat_template_kwargs": kwargs, "matched_full_context": True})

    def transform_for_eval_reference(self, items: list[ActivationContextTrainingItem],
                                     kind: t.Literal["no_context", "recent_text", "full_context"]) -> list[ActivationContextTrainingItem]:
        """Preserve prefixes/suffixes and targets; QA's text reference is oracle gold, not retrieval."""
        if kind not in ("no_context", "recent_text", "full_context"):
            raise ValueError(kind)
        output = []
        for item in items:
            if kind == "full_context":
                if item.in_context_messages is None:
                    raise ValueError(f"{item.item_id}: full context requires an in-context view")
                output.append(replace(item, item_id=item.item_id + ":full_context", ac_messages=deepcopy(item.in_context_messages),
                                      ac_start=item.in_context_start, info=dict(item.info, reference_label="full context, same distractors" if item.info.get("matched_full_context") else "in-context reference")))
                continue
            messages = deepcopy(item.ac_messages)
            part_index = 0
            for message in messages:
                if not isinstance(message.get("content"), list):
                    continue
                pieces = []
                for part in message["content"]:
                    if part.get("type") != AC_PART_TYPE:
                        pieces.append(part.get("text", ""))
                        continue
                    if kind == "recent_text":
                        rows = self.ac_model.part_view_rows(part["messages"], part.get("compression_target"), part.get("tools"))
                        if item.kind == "compaction":
                            text = render_messages(_expand_parts(part["messages"]))
                            ids = self.tokenizer.encode(text, add_special_tokens=False)[-rows:]
                        elif item.kind == "rag_qa" or part_index == item.info.get("gold_position", 0):
                            ids = self.tokenizer.encode(item.info.get("gold_chunk_text", ""), add_special_tokens=False)[:rows]
                        else:
                            ids = []
                        text = self.tokenizer.decode(ids)
                        # A cut through a byte token can grow when decoded and tokenized again.
                        while ids and len(self.tokenizer.encode(text, add_special_tokens=False)) > rows:
                            ids = ids[1:] if item.kind == "compaction" else ids[:-1]
                            text = self.tokenizer.decode(ids)
                        pieces.append(text)
                    part_index += 1
                message["content"] = "".join(pieces)
            output.append(replace(item, item_id=item.item_id + ":" + kind, ac_messages=messages,
                                  info=dict(item.info, reference_label="oracle gold text" if kind == "recent_text" and item.kind != "compaction" else kind)))
        return output


def _expand_parts(messages: list[dict]) -> list[dict]:
    result = []
    for original in messages:
        message = deepcopy(original)
        if isinstance(message.get("content"), list):
            message["content"] = "".join(render_messages(_expand_parts(part["messages"])) if part.get("type") == AC_PART_TYPE
                                         else part.get("text", "") for part in message["content"])
        result.append(message)
    return result


def item_origin(item: ActivationContextTrainingItem) -> tuple[str, str, str]:
    origin = (item.doc_ids[0] if item.doc_ids else item.item_id) if item.kind == "compaction" else item.info.get("example_id", item.item_id)
    return item.dataset_id, "document" if item.kind == "compaction" else "question", origin


def split_items(items: list[ActivationContextTrainingItem], held: int, train: int) -> tuple[list[ActivationContextTrainingItem], list[ActivationContextTrainingItem], int]:
    """Source-balanced held items; origins excluded from training (QA holds out questions, not contexts)."""
    by_source: dict[str, list[ActivationContextTrainingItem]] = {}
    for item in items:
        by_source.setdefault(item.dataset_id, []).append(item)
    held_items = []
    held = min(max(0, held), len(items))
    cumulative = 0
    for source_items in by_source.values():
        cumulative += len(source_items)
        quota = round(held * cumulative / len(items)) - len(held_items)
        held_items.extend(source_items[:quota])
    held_ids = {item.item_id for item in held_items}
    origins = {item_origin(item) for item in held_items}
    train_items, shared = [], 0
    for item in items:
        if item.item_id in held_ids:
            continue
        if item_origin(item) in origins:
            shared += 1
        else:
            train_items.append(item)
    return held_items, train_items[:max(0, train)], shared
