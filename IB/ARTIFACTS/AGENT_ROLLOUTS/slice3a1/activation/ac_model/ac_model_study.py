"""
Training items for the AC model, three kinds (ac_model_training.ActivationContextTrainingItem):

- compaction: a trajectory cut at depth d (1-2) into segments of `threshold` tokens; the in-context
  prefix is the real messages up to the last cut (an assistant turn may be cut mid-text), the AC
  prefix nests the segments (`AC_d{AC_{d-1}{...}, segment d}`) in one user message followed by the
  compaction instructions; the completion is the rest of the cut turn or the next assistant turn.
  Trains the model to "see" its own compactions.
- traj_qa: a study question about a trajectory chunk (dataset_study, `traj_qa`); the in-context
  prefix holds the transcript window around the chunk, the AC prefix the window and 0-2 distractor
  trajectories (bm25 top documents for the question, the source excluded) as separate parts, each
  with the task text first so the compressor knows what the reader is after. Trains "seeing" a
  subagent or parent.
- rag_qa: a study question about a text chunk; the in-context prefix holds the gold passage, the AC
  prefix one part with the gold and bm25 distractor passages shuffled to a token budget.

Every item's completion is real text (the trajectory's own continuation, the gold answer): no
sampling is needed at generation time. Token budgets are counted with the target tokenizer.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
import random
import typing as t

from ..agent.agent_utils import ModelDialect
from ..dataset.dataset import DataModality, DatasetDocument, DatasetDocumentChunk, DatasetQAExample
from ..dataset.loaders.trajectory_utils import submit_answer_definition
from ..dataset.dataset_utils import message_text, render_messages
from .ac_model_training import ActivationContextTrainingItem
from .ac_model_utils import AC_PART_TYPE

if t.TYPE_CHECKING:
    from ..harness import HarnessRuntime

COMPACTION_INSTRUCTIONS = (
    "The conversation so far has been compacted into the activation context above. Continue the task "
    "from exactly where it left off, as if the full conversation were still in front of you."
)
TRAJ_QA_SYSTEM_PROMPT = "You answer questions about agent trajectories (transcripts of an agent's reasoning, tool calls and tool results)."
TRAJ_QA_INSTRUCTIONS = "Answer the question from the trajectories above (some may be unrelated). Call submit_answer with the answer."
RAG_QA_SYSTEM_PROMPT = "You answer questions from the passages you are given."
RAG_QA_INSTRUCTIONS = "Answer the question from the passages above (some may be unrelated). Call submit_answer with the answer."
MESSAGE_TOKEN_OVERHEAD = 8                    # template tokens per message, on top of its text (estimate)
MIN_COMPACTION_TOKENS = 1024                  # a trajectory shorter than this (before its completion) yields no compaction item


def ac_part(messages: list[dict], ac_name: str, compression_target: float) -> dict:
    return {"type": AC_PART_TYPE, "ac_name": ac_name, "compression_target": compression_target, "messages": messages}


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
        self,
        dataset_id: str,
        num_samples: int,
        seed: int = 0,
        depth_range: tuple[int, int] = (1, 2),
        threshold_range_tokens: tuple[int, int] = (8192, 32768),
        ratios_range: tuple[float, float] = (1.0 / 8.0, 1.0 / 16.0),
        max_post_compaction_tokens: int = 8192,
        immediate_continuation_ratio: float = 0.25, # Still bias towards many trajectories being immediate continuations, as that's most important.
        completion_max_tokens: int = 512,
        max_total_tokens: int = 72_000,
        min_trajectory_tokens: int = MIN_COMPACTION_TOKENS,
    ) -> list[ActivationContextTrainingItem]:
        """
        One item per sampled trajectory document (documents are drawn with replacement once every
        document was used; a document shorter than `min_trajectory_tokens` before its completion is
        skipped). Segment thresholds are scaled down to fit the trajectory and `max_total_tokens`.
        """
        if max_post_compaction_tokens < 0 or not 0 <= immediate_continuation_ratio <= 1 or completion_max_tokens < 1:
            raise ValueError("Invalid continuation budget or immediate mixture")
        rng = random.Random(seed)
        documents = self._documents(dataset_id, DataModality.TRAJECTORY)
        assert documents, f"{dataset_id} holds no trajectory documents"
        order = list(documents)
        rng.shuffle(order)
        items: list[ActivationContextTrainingItem] = []
        attempts = 0
        requested_immediate = rng.random() < immediate_continuation_ratio
        while len(items) < num_samples and attempts < 4 * num_samples + len(order):
            document = order[attempts % len(order)]
            attempts += 1
            item = self._compaction_item(document, dataset_id, rng, len(items), seed, depth_range, threshold_range_tokens,
                                         ratios_range, completion_max_tokens, max_total_tokens, min_trajectory_tokens,
                                         max_post_compaction_tokens, requested_immediate)
            if item is not None:
                items.append(item)
                requested_immediate = rng.random() < immediate_continuation_ratio
        if len(items) < num_samples:
            print(f"AC study - produced {len(items)}/{num_samples} compaction items; continuation candidates exhausted")
        return items

    def _compaction_item(self, document: DatasetDocument, dataset_id: str, rng: random.Random, index: int, seed: int,
                         depth_range: tuple[int, int], threshold_range_tokens: tuple[int, int], ratios_range: tuple[float, float],
                         completion_max_tokens: int, max_total_tokens: int, min_trajectory_tokens: int,
                         max_post_compaction_tokens: int = 8192, immediate: bool = True) -> ActivationContextTrainingItem | None:
        messages = document.trajectory or []
        if len(messages) < 3 or messages[0]["role"] != "user":
            return None
        tools = (document.trajectory_kwargs or {}).get("tools") or None
        system = self._system_messages(document)
        task = messages[0]
        body = messages[1:]
        tokens = [self.message_tokens(message) for message in body]
        # The completion must be an assistant turn after the last cut: the body up to the last assistant turn is cuttable.
        last_assistant = max((i for i, message in enumerate(body) if message["role"] == "assistant"), default=-1)
        if last_assistant < 1:
            return None
        available = sum(tokens[:last_assistant + 1])                            # the last assistant turn may be cut mid-text
        if available < min_trajectory_tokens:
            return None
        depth = rng.randint(min(depth_range), max(depth_range))
        thresholds = [rng.randint(min(threshold_range_tokens), max(threshold_range_tokens)) for _ in range(depth)]
        budget = min(max_total_tokens - completion_max_tokens - self.message_tokens(task) - 256, int(available * 0.9))
        if budget < 256:
            return None
        if sum(thresholds) > budget:
            scale = budget / sum(thresholds)
            thresholds = [max(256, int(threshold * scale)) for threshold in thresholds]
        # Cuts: (message index, character offset in that assistant text or None for a boundary).
        cuts: list[tuple[int, int | None]] = []
        cumulative = 0
        target_total = 0
        position = 0
        for threshold in thresholds:
            target_total += threshold
            while position < len(body) and cumulative + tokens[position] <= target_total:
                cumulative += tokens[position]
                position += 1
            if position >= len(body):
                return None
            message = body[position]
            text = message_text(message)
            if message["role"] == "assistant" and not message.get("tool_calls") and len(text) > 200 and cumulative < target_total:
                fraction = (target_total - cumulative) / max(1, tokens[position])
                full_ids = self.tokenizer.encode(text, add_special_tokens=False)
                cut_tokens = max(1, min(len(full_ids) - 1, int(len(full_ids) * min(0.9, max(0.1, fraction)))))
                partial, remainder, cut_tokens = self._token_boundary(text, full_ids, cut_tokens)
                offset = len(partial)
                cuts.append((position, offset))
                break                                                              # a mid-turn cut ends the chain (the rest of the turn is the completion)
            cuts.append((position, None))
        if not cuts:
            return None
        # Segments between cuts, the completion after the last one.
        segments: list[list[dict]] = []
        start = 0
        partial_text = ""
        for cut_index, (position, offset) in enumerate(cuts):
            if offset is None:
                segments.append(body[start:position])
                start = position
            else:
                message = body[position]
                text = message_text(message)
                segments.append(body[start:position] + [{"role": "assistant", "content": text[:offset]}])
                partial_text = text[:offset]
                start = position
        segments = [segment for segment in segments if segment]
        if not segments:
            return None
        last_position, last_offset = cuts[-1]
        visible: list[dict] = []
        teacher_partial_ids: list[int] | None = None
        if last_offset is not None:
            turn_text = message_text(body[last_position])
            full_ids = self.tokenizer.encode(turn_text, add_special_tokens=False)
            # The cut was selected at an exact target-token boundary above.
            teacher_partial_ids = full_ids[:cut_tokens]
            completion_ids = full_ids[cut_tokens:]
            immediate_target = last_position
            visible_after_cut = [{**body[last_position], "content": turn_text[last_offset:]}]
        else:
            immediate_target = next((i for i in range(last_position, len(body)) if body[i]["role"] == "assistant"), None)
            if immediate_target is None:
                return None
            visible_after_cut = []
        if immediate:
            target_position = immediate_target
            if last_offset is None:
                visible = body[last_position:target_position]
                completion_ids = self.tokenizer.encode(self.assistant_text(body[target_position]), add_special_tokens=False)
            in_context_tail = body[:target_position]
        else:
            candidates = []
            for target_position in range(immediate_target + 1, len(body)):
                if body[target_position]["role"] != "assistant":
                    continue
                suffix = (visible_after_cut + body[last_position + 1:target_position] if last_offset is not None
                          else body[last_position:target_position])
                suffix_tokens = sum(self.message_tokens(message) for message in suffix)
                teacher_tokens = sum(tokens[:target_position]) + self.message_tokens(task)
                if suffix_tokens <= max_post_compaction_tokens and teacher_tokens + completion_max_tokens + 256 <= max_total_tokens:
                    candidates.append((target_position, suffix))
            if not candidates:
                return None
            target_position, visible = rng.choice(candidates)
            in_context_tail = body[:target_position]
            partial_text, teacher_partial_ids = "", None
            completion_ids = self.tokenizer.encode(self.assistant_text(body[target_position]), add_special_tokens=False)
        if not completion_ids:
            return None
        complete = len(completion_ids) <= completion_max_tokens
        completion_ids = completion_ids[:completion_max_tokens]
        # Avoid a cap ending inside a multibyte character; preserve original IDs.
        while completion_ids and self.tokenizer.decode(completion_ids, clean_up_tokenization_spaces=False).endswith("\ufffd"):
            completion_ids = completion_ids[:-1]
        completion_text = self.tokenizer.decode(completion_ids, clean_up_tokenization_spaces=False)
        if not completion_text.strip():
            return None
        visible_tokens = self._visible_tokens(visible, tools)
        if visible_tokens > max_post_compaction_tokens:
            return None
        if sum(tokens[:target_position]) + len(teacher_partial_ids or []) + len(completion_ids) + self.message_tokens(task) + 256 > max_total_tokens:
            return None
        ratio = self._sample_ratio(rng, ratios_range)
        nested: dict | None = None
        for segment in segments:
            inner = [{"role": "user", "content": [nested]}] if nested is not None else [task]   # the innermost segment carries the task: what the reader is after
            nested = ac_part(inner + segment, self.ac_name, ratio)
        ac_user = {"role": "user", "content": [nested, {"type": "text", "text": "\n\n" + COMPACTION_INSTRUCTIONS}]}
        item = ActivationContextTrainingItem(
            item_id=f"compaction:{dataset_id}:{seed}:{index}",
            kind="compaction",
            in_context_prefix=system + [task] + in_context_tail,
            ac_prefix=system + [task, ac_user] + deepcopy(visible),
            completion_text=completion_text,
            completion_complete=complete,
            teacher_partial_text=partial_text,
            teacher_partial_token_ids=teacher_partial_ids,
            completion_token_ids=completion_ids,
            tools=tools,
            dataset_id=dataset_id,
            doc_ids=[document.doc_id],
            info={"depth": len(segments), "thresholds": thresholds, "ratio": ratio, "in_context_tokens": sum(tokens[:len(in_context_tail)]),
                  "immediate": immediate, "visible_suffix_tokens": visible_tokens,
                  "delay_turns": target_position - immediate_target,
                  "observed_terminal_answer": (document.trajectory_kwargs or {}).get("observed_terminal_answer")},
        )
        prefix = self.tokenizer.apply_chat_template(item.in_context_prefix, tools=tools, add_generation_prompt=True, tokenize=True)
        prefix = prefix["input_ids"] if hasattr(prefix, "keys") else prefix
        eot = self.ac_model.target.model_config.model_description.eot_token
        end_tokens = len(self.tokenizer.encode(eot, add_special_tokens=False)) if complete and eot else 0
        exact_tokens = len(prefix) + len(teacher_partial_ids or []) + len(completion_ids) + end_tokens
        if exact_tokens > max_total_tokens:
            return None
        item.info["in_context_tokens"] = exact_tokens
        return item

    def _visible_tokens(self, visible: list[dict], tools: list[dict] | None) -> int:
        if not visible:
            return 0
        anchor = [{"role": "user", "content": ""}]
        def count(messages: list[dict]) -> int:
            ids = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True)
            return len(ids["input_ids"] if hasattr(ids, "keys") else ids)
        return max(0, count(anchor + visible) - count(anchor))

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
            def part_for(messages: list[dict]) -> dict:
                part = ac_part(messages, self.ac_name, ratio)
                if depth >= 2:
                    part = ac_part([messages[0], {"role": "user", "content": [part]}], self.ac_name, ratio)
                return part

            parts = [("gold", part_for(window))] + [("distractor", part_for(messages)) for messages in distractors]
            rng.shuffle(parts)
            content = [{"type": "text", "text": f"Task: {question}\n\n"}]
            for number, (_, part) in enumerate(parts, start=1):
                content.extend([{"type": "text", "text": f"Trajectory {number}:\n"}, part, {"type": "text", "text": "\n\n"}])
            content.append({"type": "text", "text": f"{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"})
            in_context_user = {"role": "user", "content": f"Trajectory:\n{render_messages(window)}\n\n{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"}
            system = [{"role": "system", "content": TRAJ_QA_SYSTEM_PROMPT}]
            items.append(ActivationContextTrainingItem(
                item_id=f"traj_qa:{dataset_id}:{seed}:{item_index}",
                kind="traj_qa",
                in_context_prefix=system + [in_context_user],
                ac_prefix=system + [{"role": "user", "content": content}],
                completion_text=self.dialect.preferred_format.render("submit_answer", {"answer": example.gold_answers[0].strip()}),
                tools=[submit_answer_definition()],
                completion_complete=True,
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
            items.append(ActivationContextTrainingItem(
                item_id=f"rag_qa:{dataset_id}:{seed}:{item_index}",
                kind="rag_qa",
                in_context_prefix=system + [in_context_user],
                ac_prefix=system + [{"role": "user", "content": content}],
                completion_text=self.dialect.preferred_format.render("submit_answer", {"answer": example.gold_answers[0].strip()}),
                tools=[submit_answer_definition()],
                completion_complete=True,
                dataset_id=dataset_id,
                doc_ids=sorted({passage.doc_id for passage in passages}),
                info={"depth": depth, "budget": budget, "passages": len(passages), "ratio": ratio,
                      "gold_position": passages.index(gold), "example_id": example.example_id, "gold_chunk_text": gold.chunk_text, "answer": example.gold_answers[0]},
            ))
        return items

    def transform_for_eval_reference(self, items: list[ActivationContextTrainingItem],
                                     kind: t.Literal["no_context", "recent_text"]) -> list[ActivationContextTrainingItem]:
        """Preserve prefixes/suffixes and targets; QA's text reference is oracle gold, not retrieval."""
        if kind not in ("no_context", "recent_text"):
            raise ValueError(kind)
        output = []
        for item in items:
            messages = deepcopy(item.ac_prefix)
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
                        rows = self.ac_model.part_view_rows(part["messages"], part.get("compression_target"))
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
            output.append(replace(item, item_id=item.item_id + ":" + kind, ac_prefix=messages,
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
