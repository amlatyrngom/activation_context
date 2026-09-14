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
import random
import typing as t

from ..agent.agent_utils import ModelDialect
from ..dataset.dataset import DataModality, DatasetDocument, DatasetDocumentChunk, DatasetQAExample
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
TRAJ_QA_INSTRUCTIONS = "Answer the question from the trajectories above (some may be unrelated). Reply with the answer only."
RAG_QA_SYSTEM_PROMPT = "You answer questions from the passages you are given."
RAG_QA_INSTRUCTIONS = "Answer the question from the passages above (some may be unrelated). Reply with the answer only."
MESSAGE_TOKEN_OVERHEAD = 8                    # template tokens per message, on top of its text (estimate)
MIN_COMPACTION_TOKENS = 1024                  # a trajectory shorter than this (before its completion) yields no compaction item


def ac_part(messages: list[dict], ac_name: str, compression_target: float) -> dict:
    return {"type": AC_PART_TYPE, "ac_name": ac_name, "compression_target": compression_target, "messages": messages}


class ActivationContextStudyGenerator:
    """Item generators bound to one AC model (its name goes into the parts; its target tokenizer counts tokens)."""

    def __init__(self, harness: "HarnessRuntime", ac_model_name: str):
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
        completion_max_tokens: int = 512,
        max_total_tokens: int = 72_000,
        min_trajectory_tokens: int = MIN_COMPACTION_TOKENS,
    ) -> list[ActivationContextTrainingItem]:
        """
        One item per sampled trajectory document (documents are drawn with replacement once every
        document was used; a document shorter than `min_trajectory_tokens` before its completion is
        skipped). Segment thresholds are scaled down to fit the trajectory and `max_total_tokens`.
        """
        rng = random.Random(seed)
        documents = self._documents(dataset_id, DataModality.TRAJECTORY)
        assert documents, f"{dataset_id} holds no trajectory documents"
        order = list(documents)
        rng.shuffle(order)
        items: list[ActivationContextTrainingItem] = []
        attempts = 0
        while len(items) < num_samples and attempts < 4 * num_samples + len(order):
            document = order[attempts % len(order)]
            attempts += 1
            item = self._compaction_item(document, dataset_id, rng, len(items), seed, depth_range, threshold_range_tokens,
                                         ratios_range, completion_max_tokens, max_total_tokens, min_trajectory_tokens)
            if item is not None:
                items.append(item)
        return items

    def _compaction_item(self, document, dataset_id, rng, index, seed, depth_range, threshold_range_tokens, ratios_range,
                         completion_max_tokens, max_total_tokens, min_trajectory_tokens) -> ActivationContextTrainingItem | None:
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
        budget = min(max_total_tokens, int(available * 0.9))
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
                offset = int(len(text) * min(0.9, max(0.1, fraction)))
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
        if last_offset is not None:
            completion_text = message_text(body[last_position])[last_offset:]
            in_context_tail = body[:last_position]
        else:
            next_assistant = next((i for i in range(last_position, len(body)) if body[i]["role"] == "assistant"), None)
            if next_assistant is None:
                return None
            completion_text = self.assistant_text(body[next_assistant])
            in_context_tail = body[:next_assistant]
        completion_text = completion_text.strip()
        if not completion_text:
            return None
        completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)
        complete = len(completion_ids) <= completion_max_tokens
        if not complete:
            completion_text = self.tokenizer.decode(completion_ids[:completion_max_tokens])
        ratio = self._sample_ratio(rng, ratios_range)
        nested: dict | None = None
        for segment in segments:
            inner = [{"role": "user", "content": [nested]}] if nested is not None else [task]   # the innermost segment carries the task: what the reader is after
            nested = ac_part(inner + segment, self.ac_name, ratio)
        ac_user = {"role": "user", "content": [nested, {"type": "text", "text": "\n\n" + COMPACTION_INSTRUCTIONS}]}
        return ActivationContextTrainingItem(
            item_id=f"compaction:{dataset_id}:{seed}:{index}",
            kind="compaction",
            in_context_prefix=system + [task] + in_context_tail,
            ac_prefix=system + [task, ac_user],
            completion_text=completion_text,
            completion_complete=complete,
            teacher_partial_text=partial_text if last_offset is not None else "",
            tools=tools,
            dataset_id=dataset_id,
            doc_ids=[document.doc_id],
            info={"depth": len(segments), "thresholds": thresholds, "ratio": ratio, "in_context_tokens": sum(tokens[:len(in_context_tail)])},
        )

    # ------------------------------------------------------------------------------------------ trajectory QA
    def _study_examples(self, dataset_id: str, num_samples: int, seed: int, modality: DataModality, caching_id: str | None) -> list[DatasetQAExample]:
        examples = self.harness.dataset_manager.synthesize_study_examples_qa(
            dataset_id, num_samples, base_seed=seed, caching_id=caching_id, modality=modality)
        return [example for example in examples if example.positive_chunk_ids and example.gold_answers and example.gold_answers[0]]

    def _chunk_window(self, document: DatasetDocument, center: DatasetDocumentChunk | None, budget_tokens: int) -> list[dict]:
        """Messages of the document's chunks around `center` (or from the start) up to the token budget, in order."""
        chunks = sorted(document.chunks.values(), key=lambda chunk: chunk.chunk_start)
        costs = [self.count_tokens(chunk.chunk_text) for chunk in chunks]
        if center is None:
            center_index = 0
        else:
            center_index = next(i for i, chunk in enumerate(chunks) if chunk.chunk_id == center.chunk_id)
        low = high = center_index
        total = costs[center_index]
        while True:
            grew = False
            if low > 0 and total + costs[low - 1] <= budget_tokens:
                low -= 1
                total += costs[low]
                grew = True
            if high < len(chunks) - 1 and total + costs[high + 1] <= budget_tokens:
                high += 1
                total += costs[high]
                grew = True
            if not grew:
                break
        return [message for chunk in chunks[low:high + 1] for message in chunk.chunk_messages or [{"role": "user", "content": chunk.chunk_text}]]

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
            window = self._chunk_window(document, chunk, budget)
            num_distractors = min(rng.randint(min(distractors_range), max_distractors), len(candidates))
            distractors = [self._chunk_window(candidate, None, budget) for candidate in candidates[:num_distractors]]
            ratio = self._sample_ratio(rng, ratios_range)
            depth = rng.randint(min(depth_range), max(depth_range))
            question = example.query
            task_message = {"role": "user", "content": f"Task: {question}"}

            def part_for(messages: list[dict]) -> dict:
                part = ac_part([task_message] + messages, self.ac_name, ratio)
                if depth >= 2:
                    part = ac_part([task_message, {"role": "user", "content": [part]}], self.ac_name, ratio)
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
                completion_text=example.gold_answers[0].strip(),
                completion_complete=True,
                dataset_id=dataset_id,
                doc_ids=[document.doc_id] + [candidate.doc_id for candidate in candidates[:num_distractors]],
                info={"depth": depth, "budget": budget, "distractors": num_distractors, "ratio": ratio,
                      "gold_position": next(i for i, (kind, _) in enumerate(parts) if kind == "gold"), "example_id": example.example_id,
                      "gold_chunk_text": chunk.chunk_text},
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
                completion_text=example.gold_answers[0].strip(),
                completion_complete=True,
                dataset_id=dataset_id,
                doc_ids=sorted({passage.doc_id for passage in passages}),
                info={"depth": depth, "budget": budget, "passages": len(passages), "ratio": ratio,
                      "gold_position": passages.index(gold), "example_id": example.example_id, "gold_chunk_text": gold.chunk_text},
            ))
        return items
