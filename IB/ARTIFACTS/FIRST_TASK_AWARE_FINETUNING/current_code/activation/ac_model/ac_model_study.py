"""Paired whole-history compaction/QA items and exact recorded-run self-distillation."""
from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace
import random
import typing as t

from ..agent.agent_utils import ModelDialect
from ..dataset.dataset import DataModality, DatasetDocument, DatasetDocumentChunk, DatasetQAExample
from ..dataset.loaders.trajectory_utils import submit_answer_definition
from ..dataset.dataset_utils import message_text, render_messages
from .ac_model_training import ActivationContextTrainingItem, ActivationContextTrainer, ActivationContextTrainingConfig
from .ac_model_utils import (AC_PART_TYPE, direct_parts, is_ac_part, assistant_target_ids,
                             prepare_training_history, normalize_public_assistants, complete_public_turns)
from ..common.ac_parts import COMPACTION_INSTRUCTIONS, ac_part, tools_in_system_text

if t.TYPE_CHECKING:
    from ..agent.agent_config import AgentRunResult
    from ..harness import HarnessRuntime

TRAJ_QA_SYSTEM_PROMPT = "You answer questions about agent trajectories (transcripts of an agent's reasoning, tool calls and tool results)."
TRAJ_QA_INSTRUCTIONS = "Answer the question from the trajectories above (some may be unrelated). Call submit_answer with the answer."
RECONSTRUCTION_SYSTEM_PROMPT = "You are a careful assistant. You reproduce text exactly when asked."
RECONSTRUCTION_INSTRUCTION = "Reproduce the passage above exactly, word for word, and stop at its end."


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
    ) -> list[ActivationContextTrainingItem]:
        """One whole-history item per nonempty segment, preserving every selected recorded output."""
        from ..agent_training.agent_training_utils import activation_messages_of
        from ..agent.rollout_caching import config_key
        if loss_kind not in ("kl", "sft"):
            raise ValueError(loss_kind)
        selection = selection or ("all" if loss_kind == "kl" else "score_1")
        if selection not in ("all", "score_1", "score_1_or_unscored"):
            raise ValueError(selection)
        report = self.harvest_report = {"roots": len(runs), "runs": 0, "candidates": 0, "accepted": 0,
                                      "loss_kind": loss_kind, "selection": selection, "skipped": {}}
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
                if (source_tokenizer.get_vocab() != self.tokenizer.get_vocab()
                        or source_tokenizer.special_tokens_map != self.tokenizer.special_tokens_map
                        or source_tokenizer.chat_template != self.tokenizer.chat_template):
                    raise ValueError("Recorded assistant tokens require a compatible tokenizer and chat template")
                tools = self._run_tools(run)
                for segment_index, segment in enumerate([*run.compactions, run]):
                    report["candidates"] += 1
                    targets = []
                    for step in segment.trajectory:
                        assistants = [message for message in step.get("messages") or [] if message.get("role") == "assistant"]
                        if step.get("role") == "assistant":
                            ids = list(step.get("token_ids") or [])
                            if len(assistants) != 1 or not ids:
                                raise ValueError("Recorded assistant step needs one message and its nonempty sampled token IDs; regenerate this run")
                            targets.append(ids)
                        elif assistants:
                            raise ValueError("Recorded input step contains an assistant output")
                    if not targets:
                        self._harvest_skip("empty_segment")
                        continue
                    config = replace(segment.agent_config, ac_model_name=self.ac_name)
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
                    if not segment.ac_model_name:
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
                        item_id=f"rollout:{source_key}:{segment_index}", kind="compaction",
                        in_context_messages=teacher, ac_messages=deepcopy(student),
                        in_context_start=teacher_start, ac_start=start, assistant_token_ids=targets,
                        tools=tools, weight=weight, dataset_id="agent_runs", doc_ids=[root_key],
                        info={"source": "agent_run", "source_key": source_key, "root_key": root_key,
                              "chat_template_kwargs": source_model.chat_template_kwargs(segment.agent_config.call_kwargs),
                              "segment": segment_index, "score": root.score, "completion_source": "recorded",
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
        out: list[dict] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                out.append(dict(message))
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
                if out and expanded and out[-1] == expanded[0]:
                    out.pop()                                                       # the tree starts with the task the prompt already shows
                out.extend(expanded)
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
                    elif kind == "search":
                        pieces.append("\n\n" + next((m.get("content", "") for m in inner if m.get("role") == "tool"), ""))
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
            question = example.query
            def part_for(source: DatasetDocument, messages: list[dict]) -> dict:
                kwargs = source.trajectory_kwargs or {}                              # the frame the trajectory ran with, when the loader kept it
                frame = [{"role": "system", "content": kwargs["system_prompt"]}] if kwargs.get("system_prompt") else []
                part = ac_part(frame + messages, self.ac_name, ratio)
                if kwargs.get("tools"):
                    part["tools"] = deepcopy(kwargs["tools"])
                if depth >= 2:
                    part = ac_part(frame + [messages[0], {"role": "user", "content": [part]}], self.ac_name, ratio)
                if kwargs.get("tools"):
                    part["tools"] = deepcopy(kwargs["tools"])
                return part

            parts = [("gold", part_for(document, window))] + [("distractor", part_for(candidate, messages))
                                                                for candidate, messages in zip(candidates[:num_distractors], distractors)]
            rng.shuffle(parts)
            content = [{"type": "text", "text": f"Task: {question}\n\n"}]
            for number, (_, part) in enumerate(parts, start=1):
                content.extend([{"type": "text", "text": f"Trajectory {number}:\n"}, part, {"type": "text", "text": "\n\n"}])
            content.append({"type": "text", "text": f"{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"})
            source_frame = tools_in_system_text(self._system_messages(document) + window, (document.trajectory_kwargs or {}).get("tools"))
            in_context_user = {"role": "user", "content": f"Trajectory:\n{render_messages(source_frame)}\n\n{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"}
            system = [{"role": "system", "content": TRAJ_QA_SYSTEM_PROMPT}]
            output = self.dialect.rendering.assistant_message("", [{"id": "answer", "name": "submit_answer", "arguments": {"answer": example.gold_answers[0].strip()}}])
            items.append(ActivationContextTrainingItem(
                item_id=f"traj_qa:{dataset_id}:{seed}:{item_index}",
                kind="traj_qa",
                in_context_messages=system + [in_context_user, output],
                ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)],
                in_context_start=len(system) + 1, ac_start=len(system) + 1,
                assistant_token_ids=[assistant_target_ids(self.tokenizer, output, [submit_answer_definition()])],
                tools=[submit_answer_definition()],
                dataset_id=dataset_id,
                doc_ids=[document.doc_id] + [candidate.doc_id for candidate in candidates[:num_distractors]],
                info={"depth": depth, "budget": budget, "distractors": num_distractors, "ratio": ratio,
                      "gold_position": next(i for i, (kind, _) in enumerate(parts) if kind == "gold"), "example_id": example.example_id,
                      "gold_chunk_text": chunk.chunk_text, "answer": example.gold_answers[0],
                      "window_tokens": self.count_tokens(render_messages(window)),
                      "window_over_budget": self.count_tokens(render_messages(window)) > budget},
            ))
        return items

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
    ) -> list[ActivationContextTrainingItem]:
        """
        Study questions on text chunks; the in-context prefix is the gold passage and the question,
        the AC prefix one part holding the gold and bm25 distractor passages (the gold's document
        excluded) shuffled up to the token budget. Depth 2 wraps that part once more.
        """
        rng = random.Random(seed)
        index = self.harness.dataset_manager.dataset_indexes[dataset_id]
        examples = self._study_examples(dataset_id, num_samples, seed, DataModality.TEXT, caching_id)
        ranked = index.bm25_query_many_frozen([example.query for example in examples], top_k=64,
                                              excluded_doc_ids=[list(example.positive_doc_ids or []) for example in examples]) if examples else []
        items = []
        for item_index, (example, candidates) in enumerate(zip(examples, ranked)):
            gold = index.chunks[example.positive_chunk_ids[0]]
            budget = rng.randint(min(distractors_token_range), max(distractors_token_range))
            passages = [gold]
            total = self.count_tokens(gold.chunk_text)
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
            task_message = {"role": "user", "content": f"Task: {question}"}
            passages_text = "\n\n".join(f"Passage {number}:\n{passage.chunk_text}" for number, passage in enumerate(passages, start=1))
            part = ac_part([task_message, {"role": "user", "content": passages_text}], self.ac_name, ratio)
            if depth >= 2:
                part = ac_part([task_message, {"role": "user", "content": [part]}], self.ac_name, ratio)
            content = [{"type": "text", "text": f"Task: {question}\n\nPassages:\n"}, part,
                       {"type": "text", "text": f"\n\n{RAG_QA_INSTRUCTIONS}\nQuestion: {question}"}]
            in_context_user = {"role": "user", "content": f"Passage:\n{gold.chunk_text}\n\n{RAG_QA_INSTRUCTIONS}\nQuestion: {question}"}
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
                      "gold_position": passages.index(gold), "example_id": example.example_id, "gold_chunk_text": gold.chunk_text, "answer": example.gold_answers[0]},
            ))
        return items

    def generate_reconstruction_samples(
        self,
        passages: list[dict],
        ratio: float = 1.0 / 16.0,
        variant: str = "reconstruct",
        cue_fraction: float = 0.2,
        seed: int = 0,
        system_prompt: str = RECONSTRUCTION_SYSTEM_PROMPT,
    ) -> list[ActivationContextTrainingItem]:
        """
        Dense, oracle-free items: the reader sees a passage only through its compressed rows and reproduces it.
        `passages` are {"text", "doc_id", "source"} (any text; no dataset registration needed). Variants:
        "reconstruct" (the whole passage is the target) and "continue" (the first `cue_fraction` of the passage stays
        in plain text after the part and the rest is the target). The in-context view holds the plain passage, so a
        KL teacher would copy; these items are meant for SFT. Kind is the variant name.

        Nested (two-level) passages carry `segments` instead of `text`: an ordered list of {"kind": "text", "text"} and
        {"kind": "child", "text", "ratio"} entries, plus `ratio` (the passage's own compression; the call's `ratio` when
        absent) and optionally `variant_only` (the passage is rendered in that variant alone). The part's message content
        is then a list: the text segments verbatim and one activation_context part per child segment (the child's text
        at the child's ratio), in order, so the encoder reads its own rows next to text. The plain view and the targets
        use the concatenation of all segment texts; the "continue" cue ends `cue_fraction` into the first child, so the
        continuation crosses from the child's rows into the parent's text. Flat passages render as before.
        """
        if variant not in ("reconstruct", "continue"):
            raise ValueError("variant must be 'reconstruct' or 'continue'")
        if not 0 < cue_fraction < 1:
            raise ValueError("cue_fraction must be in (0, 1)")
        items = []
        for index, passage in enumerate(passages):
            if passage.get("variant_only") not in (None, variant):
                continue
            segments = passage.get("segments")
            text = "".join(segment["text"] for segment in segments) if segments else passage["text"].strip()
            if not text:
                continue
            child = _first_child_span(segments) if segments else None
            part_ratio = passage.get("ratio", ratio) if segments else ratio
            nested_info = {"children": sum(1 for s in segments if s["kind"] == "child"), "child_ratios": [s["ratio"] for s in segments if s["kind"] == "child"]} if segments else {}
            if variant == "continue":
                cut = _cue_cut(text, cue_fraction) if child is None else child[0] + _cue_cut(text[child[0]:child[1]], cue_fraction)
                cue, target = text[:cut].rstrip(), text[cut:].lstrip()
                if not cue or not target or (child is not None and not child[0] < cut < child[1]):
                    continue
                instruction = f"The passage above begins with:\n{cue}\n\nContinue the passage from there, exactly as written, and stop at its end."
                output_text = target
                if segments:
                    nested_info["cue_chars"] = cut
            else:
                instruction = RECONSTRUCTION_INSTRUCTION
                output_text = text
            part = ac_part([{"role": "user", "content": _segment_content(segments, self.ac_name) if segments else text}], self.ac_name, part_ratio)
            content = [{"type": "text", "text": "Passage:\n"}, part, {"type": "text", "text": f"\n\n{instruction}"}]
            output = {"role": "assistant", "content": output_text}
            system = [{"role": "system", "content": system_prompt}]
            items.append(ActivationContextTrainingItem(
                item_id=f"{variant}:{passage.get('source', 'text')}:{seed}:{index}",
                kind=variant,
                in_context_messages=system + [{"role": "user", "content": f"Passage:\n{text}\n\n{instruction}"}, output],
                ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)],
                in_context_start=len(system) + 1, ac_start=len(system) + 1,
                assistant_token_ids=[assistant_target_ids(self.tokenizer, output)],
                tools=None,
                dataset_id=passage.get("source", ""),
                doc_ids=[passage["doc_id"]] if passage.get("doc_id") else [],
                info={"ratio": part_ratio, "variant": variant, "chars": len(text), "passage_tokens": self.count_tokens(text), **nested_info},
            ))
        return items

    def transform_for_eval_reference(self, items: list[ActivationContextTrainingItem],
                                     kind: t.Literal["no_context", "recent_text"]) -> list[ActivationContextTrainingItem]:
        """Preserve prefixes/suffixes and targets; QA's text reference is oracle gold, not retrieval."""
        if kind not in ("no_context", "recent_text"):
            raise ValueError(kind)
        output = []
        for item in items:
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
                        pieces.append(self.tokenizer.decode(ids))
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
