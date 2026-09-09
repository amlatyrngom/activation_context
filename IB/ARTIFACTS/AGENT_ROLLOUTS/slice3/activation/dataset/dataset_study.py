"""
Generate study material for a given dataset: questions with reference answers from its chunks (text
chunks, or trajectory chunks: a question about a fact inside an agent's turns), and oracle labels of
which chunks help answer a question. Both steps are cached under the synced folder when the caller
gives a caching id (dataset_caching.py). Nothing is written into the dataset object: the callers get
the examples back and decide what to keep.
"""
import json
import random
import time
import typing as t

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import StructuredOutputsParams

from ..common.data_syncing import resolve_path
from .dataset import (
    SYNTHETIC,
    DataModality,
    DataSplit,
    DatasetQAExample,
    LoadedDataset,
    DatasetDocumentChunk,
)
from .dataset_caching import StudyCache, StudyCacheKey
from .dataset_utils import safe_truncate_embedding_chunk, shuffle_fill_truncate

LABEL_PROMPT_TOKEN_LIMIT = 18_000   # under the engine's 20k context, with room for the answer
STUDY_CACHE_FOLDER = "STUDY"
from ..harness.vllm_wrapper import RECOMMENDED_BATCH_SIZE

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime


STUDY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "has_valuable_question": {"type": "boolean"},
        "question": {"type": ["string", "null"]},
        "answer": {"type": ["string", "null"]},
    },
    "required": ["has_valuable_question", "question", "answer"],
    "additionalProperties": False,
}

LABEL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "positives": {"type": "array", "items": {"type": "integer"}},
        "negatives": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["positives", "negatives"],
    "additionalProperties": False,
}



def make_study_prompt(study_context: str | None, modality: DataModality = DataModality.TEXT) -> tuple[str, str, dict]:
    """
    Returns system prompt, instructions, json schema. The trajectory variant asks about a fact inside
    an agent's turns; both let the model declare a snippet worthless (`has_valuable_question: false`).
    """
    system_prompt = """
# Core Guidelines
- You are a study/exam-like questions formulator.
- You are given a snippet from the corpus students are expected to study and understand.
- From this, your goal is to generate a hard question (unanswerable with common knowledge or simple keyword look ups).
    - This is IMPORTANT: the question should be **hard**, or it won't test the student's learning.
    - The question should have an accompanying answer.
- Your priority is: a hard question along with its expected, correct answer.
- By default, keep your question short to medium-short unless the exam recommends longer questions.
- When the snippet holds nothing worth asking about (boilerplate, navigation, an empty listing, pure formatting), set has_valuable_question to false and leave question and answer null.
"""
    if modality == DataModality.TRAJECTORY:
        system_prompt += """
# Trajectory Snippets
- The snippet is part of an agent's conversation: its reasoning, its tool calls and the tool results.
- Ask about a specific fact that appears in these turns (a value a tool returned, a decision the agent took, an error it saw, a file or quantity it found), phrased so someone who read the turns can answer it.
- Never ask something answerable from the task statement alone, and never ask about the formatting of the transcript.
"""

    if study_context:
        system_prompt += f"""
# Exam Context
The following emphasizes the kind of questions students are expected to handle.
```
-----
{study_context}
-----
```
Frame your questions in light of this.
"""
    system_prompt += """
# Response Format
Your answer must be formatted as a json object like:
```json
{
    "has_valuable_question": true|false,
    "question": "..."|null,
    "answer": "..."|null
}
```
"""
    instructions = f"""
# Instructions
Generate the exam-like question for student's study.
"""
    return system_prompt, instructions, STUDY_JSON_SCHEMA


def make_label_prompt() -> tuple[str, str, dict]:
    """
    Returns system prompt, instructions and json schema for the snippet labeling task.
    """
    system_prompt = """
# Core Guidelines
- You are a relevance judge determining what snippets from a corpus students should pay attention to answer a question.
- You are given the study question, its reference answer, and numbered corpus snippets.
- Label each snippet index as:
    - **positive**: the snippet contains information that clearly helps answer the question.
    - **negative**: the snippet contains information that clearly does not help answer the question.
    - Leave out anything where you are unsure.
- Be careful:
    - The snippets are intentionally chosen to be *related* to the question without necessarily helping.
    - Don't mark something as positive just because of shared vocabulary.
- Every listed index must come from the snippet numbering; never list the same index on both sides.
"""
    system_prompt += """
# Response Format
Your answer must be formatted as a json object of snippet indices like:
```json
{
    "positives": [0, 2],
    "negatives": [3, 5, 6]
}
```
Each is a list of indexes that should map to one of the given snippet indices.
"""
    instructions = """
Label the snippets as positives / negatives for this question. Leave out ambiguous ones.
"""
    return system_prompt, instructions, LABEL_JSON_SCHEMA


class DatasetStudyGenerator:
    def __init__(self, harness: "HarnessRuntime", loaded_dataset: LoadedDataset):
        self.harness = harness
        self.loaded_dataset = loaded_dataset
        self.dataset_id = loaded_dataset.dataset_id
        self.dataset_index = harness.dataset_manager.dataset_indexes[self.dataset_id]
        self.qa_model_name = harness.harness_config.dataset_study_qa_model_name
        self.label_model_name = harness.harness_config.dataset_study_label_model_name
        self.qa_batch_size = harness.harness_config.dataset_study_qa_batch_size or RECOMMENDED_BATCH_SIZE
        self.label_batch_size = harness.harness_config.dataset_study_label_batch_size or RECOMMENDED_BATCH_SIZE
        self.chunk_input_limit = harness.harness_config.dataset_study_chunk_input_limit
        self.chat_kwargs = harness.harness_config.dataset_study_chat_kwargs
        self.label_top_k = harness.harness_config.dataset_study_label_top_k
        self.label_pool_max_chars = harness.harness_config.dataset_study_label_pool_max_chars
        self.cache = StudyCache(resolve_path(STUDY_CACHE_FOLDER))
        assert self.dataset_index.bm25_index is not None, "Study requires built bm25 indexes."


    def _chunk_modality(self, chunk: DatasetDocumentChunk) -> DataModality:
        return self.loaded_dataset.documents[chunk.doc_id].modality

    def _sample_study_chunks(self, num_samples: int, base_seed: int, modality: DataModality | None) -> list[DatasetDocumentChunk]:
        """A seeded shuffle of the chunks of the requested modality (every chunk before any repeat), cut to num_samples."""
        rng = random.Random(base_seed) # Reproducible study sets: the first n of a larger request are the same chunks.
        chunk_ids = [chunk_id for chunk_id in self.dataset_index.chunk_ids
                     if modality is None or self._chunk_modality(self.dataset_index.chunks[chunk_id]) == modality]
        chunk_ids = shuffle_fill_truncate(chunk_ids, num_samples, rng)
        return [self.dataset_index.chunks[chunk_id] for chunk_id in chunk_ids]

    def _make_engine_chat_kwargs(self, json_schema: dict, seed: int) -> dict:
        chat_kwargs = dict(self.chat_kwargs or {})
        sampling_params_kwargs = dict(chat_kwargs.pop("sampling_params", None) or {})
        sampling_params_kwargs.setdefault("max_tokens", 2048)
        sampling_params_kwargs["seed"] = seed
        sampling_params_kwargs["structured_outputs"] = StructuredOutputsParams(json=json_schema)
        chat_kwargs["sampling_params"] = sampling_params_kwargs
        return chat_kwargs

    def _example_from_row(self, row: dict, kind: str, base_seed: int) -> DatasetQAExample:
        chunk = self.dataset_index.chunks[row["chunk_id"]]
        return DatasetQAExample(
            example_id=f"{self.dataset_id}:study:{kind}:{base_seed}:{row['index']}",
            dataset_id=self.dataset_id,
            query=row["question"],
            gold_answers=[row["answer"]],
            origin=SYNTHETIC,
            split=DataSplit.TRAIN,
            positive_doc_ids=[chunk.doc_id],
            positive_chunk_ids=[chunk.chunk_id],
        )


    def generate_examples_qa(
        self,
        num_samples: int,
        base_seed: int = 0,
        study_context: str | None = None,
        caching_id: str | None = None,
        modality: DataModality | None = None,
    ) -> list[DatasetQAExample]:
        """
        One question per sampled chunk (num_samples chunks of the given modality, or of every modality
        when None; the trajectory prompt variant for trajectory chunks), minus the chunks the model
        declares junk. Cached rows (caching_id) are served first, the engine generates the tail with
        seed `base_seed + batch index`; `study_context` is added to the prompt. Nothing is written into
        the dataset: the examples are returned (origin SYNTHETIC, positive chunk = the source chunk).
        """
        loaded_model = self.harness.loaded_models[self.qa_model_name]
        stats = self.loaded_dataset.stats
        kind = "traj_qa" if modality == DataModality.TRAJECTORY else "qa"
        study_samples = self._sample_study_chunks(num_samples, base_seed, modality)
        key = None
        cached_rows: list[dict] = []
        if caching_id is not None:
            key = StudyCacheKey(caching_id, self.dataset_id, loaded_model.model_config.model_id, kind, base_seed)
            for index, row in enumerate(self.cache.read(key)):
                if index >= len(study_samples) or row.get("chunk_id") != study_samples[index].chunk_id:
                    break                                                          # the sampler changed under the cache: stop trusting it
                cached_rows.append(row)
        examples: list[DatasetQAExample] = []
        for row in cached_rows:
            if row.get("junk"):
                stats.study_num_junk_skipped += 1
            else:
                examples.append(self._example_from_row(row, kind, base_seed))
        stats.study_num_cached += len(cached_rows)
        pending = list(enumerate(study_samples))[len(cached_rows):]
        if not pending:
            return examples
        # vllm batches a whole conversation list inside one chat() call, so a
        # simple single-threaded loop needs no locks.
        loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
        system_prompt, instructions, json_schema = make_study_prompt(study_context, modality or DataModality.TEXT)
        total_generation_time = 0
        reporting_interval = 0
        for batch_index, batch_start in enumerate(range(0, len(pending), self.qa_batch_size)):
            batch = pending[batch_start:batch_start + self.qa_batch_size]
            chat_kwargs = self._make_engine_chat_kwargs(json_schema, base_seed + batch_index)
            conversations = []
            for _, chunk in batch:
                snippet = self.dataset_index.get_chunk_section(chunk)
                snippet = safe_truncate_embedding_chunk(snippet, self.chunk_input_limit)
                conversations.append([
                    {
                        "role": "system", "content": [
                            {"type": "text", "text": system_prompt},
                        ]
                    },
                    {
                        "role": "user", "content": [
                            {"type": "text", "text": f"# Corpus Snippet\n```\n{snippet}\n```\n{instructions}"},
                        ]
                    },
                ])
            batch_start_time = time.time()
            outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
            elapsed = time.time() - batch_start_time
            total_generation_time += elapsed
            stats.study_batch_latencies.append(elapsed)
            new_rows = []
            for (index, chunk), output in zip(batch, outputs):
                stats.study_prompt_tokens.append(output.prompt_token_count)
                stats.study_output_tokens.append(output.output_token_count)
                try:
                    qa_pair = json.loads(output.text)
                    valuable = bool(qa_pair["has_valuable_question"])
                    question = str(qa_pair.get("question") or "").strip()
                    answer = str(qa_pair.get("answer") or "").strip()
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Structured outputs make this rare (e.g. max_tokens cutoff). Not cached: a later pass retries.
                    stats.study_num_parse_failures += 1
                    continue
                if not valuable or not question or not answer:
                    stats.study_num_junk_skipped += 1
                    new_rows.append({"index": index, "chunk_id": chunk.chunk_id, "junk": True})
                    continue
                row = {"index": index, "chunk_id": chunk.chunk_id, "question": question, "answer": answer}
                new_rows.append(row)
                examples.append(self._example_from_row(row, kind, base_seed))
            if key is not None:
                self.cache.append(key, new_rows)
            if reporting_interval <= 0:
                print(
                    f"{self.dataset_id} - Study ({kind}): {batch_start + len(batch)}/{len(pending)} chunks, "
                    f"{len(examples)} questions, {stats.study_num_junk_skipped} junk. {total_generation_time:.2f}s."
                )
                reporting_interval = max(1, len(pending) // 20)
            reporting_interval -= len(batch)
        return examples


    def _build_label_pool(
        self,
        example: DatasetQAExample,
        ranked_chunks: list[DatasetDocumentChunk],
    ) -> list[tuple[DatasetDocumentChunk, str]]:
        """
        Return (chunk, snippet text) pairs in bm25 rank order until the pool max chars in reach.
        """
        known_positives = set(example.positive_chunk_ids or [])
        pool = []
        total_chars = 0
        for chunk in ranked_chunks:
            if chunk.chunk_id in known_positives:
                continue
            snippet = safe_truncate_embedding_chunk(chunk.chunk_text, self.chunk_input_limit)
            pool.append((chunk, snippet))
            total_chars += len(snippet)
            if total_chars >= self.label_pool_max_chars:
                break
        return pool


    def _apply_labels(
        self,
        example: DatasetQAExample,
        pool: list[tuple[DatasetDocumentChunk, str]],
        positives: list[int],
        negatives: list[int],
    ):
        """
        Applies the oracle labels. Treats them as authoritative in cases of conflicts.
        Ambiguous items are dropped.
        """
        stats = self.loaded_dataset.stats
        positive_set = set(positives) - set(negatives)
        negative_set = set(negatives) - set(positives)
        example.positive_chunk_ids = list(example.positive_chunk_ids or [])
        example.hard_negative_chunk_ids = list(example.hard_negative_chunk_ids or [])
        for index in sorted(positive_set):
            chunk, _ = pool[index]
            if chunk.chunk_id in example.hard_negative_chunk_ids:
                example.hard_negative_chunk_ids.remove(chunk.chunk_id) # The oracle verdict wins.
            if chunk.chunk_id not in example.positive_chunk_ids:
                example.positive_chunk_ids.append(chunk.chunk_id)
        for index in sorted(negative_set):
            chunk, _ = pool[index]
            if chunk.chunk_id not in example.hard_negative_chunk_ids:
                example.hard_negative_chunk_ids.append(chunk.chunk_id)
        stats.study_num_label_positives += len(positive_set)
        stats.study_num_label_negatives += len(negative_set)
        stats.study_num_label_ambiguous += len(pool) - len(positive_set) - len(negative_set)

    def _inherit_labels(
        self,
        example: DatasetQAExample,
        pool: list[tuple[DatasetDocumentChunk, str]],
    ):
        """
        One chunk per known positive / hard-negative document inherits the document's label,
        but only when the oracle never saw that chunk: a single-chunk document contributes its
        chunk, a multi-chunk document its best bm25 chunk for the query (skipped when that
        chunk scores zero). A candidate that was in the pool keeps the oracle's verdict,
        including abstention. A chunk never lands on both sides: an existing label wins.
        """
        stats = self.loaded_dataset.stats
        pool_chunk_ids = {chunk.chunk_id for chunk, _ in pool}
        positive_docs = list(example.positive_doc_ids or [])
        negative_docs = list(example.hard_negative_doc_ids or [])
        multi_chunk_docs = [
            doc_id for doc_id in positive_docs + negative_docs
            if len(self.loaded_dataset.documents[doc_id].chunks) > 1
        ]
        ranked = (
            self.dataset_index.bm25_rank_doc_chunks(example.query, multi_chunk_docs)
            if multi_chunk_docs else {}
        )
        example.positive_chunk_ids = list(example.positive_chunk_ids or [])
        example.hard_negative_chunk_ids = list(example.hard_negative_chunk_ids or [])
        for doc_ids, own_ids, other_ids, counter in [
            (positive_docs, example.positive_chunk_ids, example.hard_negative_chunk_ids, "study_num_label_inherited_positives"),
            (negative_docs, example.hard_negative_chunk_ids, example.positive_chunk_ids, "study_num_label_inherited_negatives"),
        ]:
            for doc_id in doc_ids:
                document = self.loaded_dataset.documents[doc_id]
                if len(document.chunks) == 1:
                    candidate = next(iter(document.chunks.values()))
                else:
                    candidate, top_score = ranked[doc_id][0]
                    if top_score <= 0:
                        stats.study_num_label_inherit_skipped += 1
                        continue
                if candidate.chunk_id in pool_chunk_ids:
                    continue # The oracle saw it; its verdict (or abstention) stands.
                if candidate.chunk_id in own_ids or candidate.chunk_id in other_ids:
                    continue
                own_ids.append(candidate.chunk_id)
                setattr(stats, counter, getattr(stats, counter) + 1)


    def generate_examples_labels(
        self,
        examples: list[DatasetQAExample],
        caching_id: str | None = None,
    ) -> list[DatasetQAExample]:
        """
        Oracle-label exactly these examples (the caller selects them, e.g. by origin); examples already
        labeled or without a reference answer are skipped. Labels are cached by example id under
        caching_id. Returns the examples that ended up labeled in this pass.
        """
        stats = self.loaded_dataset.stats
        examples = [example for example in examples if not example.oracle_labeled and example.gold_answers and example.gold_answers[0]]
        if not examples:
            return []
        loaded_model = self.harness.loaded_models[self.label_model_name]
        key = None
        labeled: list[DatasetQAExample] = []
        if caching_id is not None:
            key = StudyCacheKey(caching_id, self.dataset_id, loaded_model.model_config.model_id, "labels", 0,
                                label_model_id=loaded_model.model_config.model_id)
            cached = self.cache.read_by_id(key, "example_id")
            remaining = []
            for example in examples:
                row = cached.get(example.example_id)
                if row is None:
                    remaining.append(example)
                    continue
                example.positive_chunk_ids = list(row.get("positive_chunk_ids") or [])
                example.hard_negative_chunk_ids = list(row.get("hard_negative_chunk_ids") or [])
                example.oracle_labeled = True
                stats.study_num_labels_cached += 1
                labeled.append(example)
            examples = remaining
            if not examples:
                return labeled
        loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
        system_prompt, instructions, json_schema = make_label_prompt()
        chat_kwargs = self._make_engine_chat_kwargs(json_schema, self.harness.harness_config.dataset_study_seed)
        all_ranked = self.dataset_index.bm25_query_many_frozen(
            [example.query for example in examples],
            top_k=self.label_top_k,
            excluded_doc_ids=[example.excluded_doc_ids for example in examples],
        )
        example_pool_pairs = []
        for example, ranked in zip(examples, all_ranked):
            pool = self._build_label_pool(example, ranked)
            if pool:
                example_pool_pairs.append((example, pool))
            else:
                # Nothing for the oracle to judge: inheritance alone labels the example.
                self._inherit_labels(example, pool)
                example.oracle_labeled = True
                labeled.append(example)
                self._cache_label(key, example)
        total_label_time = 0
        reporting_interval = 0
        for batch_start in range(0, len(example_pool_pairs), self.label_batch_size):
            batch = example_pool_pairs[batch_start:batch_start + self.label_batch_size]
            conversations = []
            for example, pool in batch:
                snippets = "\n".join(
                    f"[Index={index}]\n```\n{snippet}\n```"
                    for index, (_, snippet) in enumerate(pool)
                )
                user_text = (
                    f"# Question\n{example.query}\n\n"
                    f"# Reference Answer\n{example.gold_answers[0]}\n\n"
                    f"# Snippets\n{snippets}\n{instructions}"
                )
                conversations.append([
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "text", "text": user_text}]},
                ])
            batch_start_time = time.time()
            try:
                outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
            except VLLMValidationError:
                # One pool over the engine's context fails the whole batch (glyph-dense chunks tokenize
                # near one token per character): skip those prompts, unlabeled, and run the rest.
                lengths = [len(loaded_model.tokenizer("\n".join(part["text"] for message in conversation for part in message["content"])).input_ids)
                           for conversation in conversations]
                kept = [index for index, length in enumerate(lengths) if length <= LABEL_PROMPT_TOKEN_LIMIT]
                skipped = len(conversations) - len(kept)
                stats.study_num_label_parse_failures += skipped
                print(f"{self.dataset_id} - Study labels: {skipped} prompts over {LABEL_PROMPT_TOKEN_LIMIT} tokens skipped (max {max(lengths)}).")
                batch = [batch[index] for index in kept]
                outputs = loaded_model.engine_chat_many([conversations[index] for index in kept], chat_kwargs=chat_kwargs) if kept else []
            elapsed = time.time() - batch_start_time
            total_label_time += elapsed
            stats.study_label_batch_latencies.append(elapsed)
            for (example, pool), output in zip(batch, outputs):
                stats.study_label_prompt_tokens.append(output.prompt_token_count)
                stats.study_label_output_tokens.append(output.output_token_count)
                stats.study_label_pool_sizes.append(len(pool))
                try:
                    labels = json.loads(output.text)
                    positives = [int(index) for index in labels["positives"]]
                    negatives = [int(index) for index in labels["negatives"]]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    # Left unlabeled (oracle_labeled stays False) so a later pass can retry cleanly.
                    stats.study_num_label_parse_failures += 1
                    continue
                in_range = lambda indexes: [index for index in indexes if 0 <= index < len(pool)]
                if len(in_range(positives)) + len(in_range(negatives)) < len(positives) + len(negatives):
                    stats.study_num_label_parse_failures += 1 # Out-of-range index: count, keep the valid ones.
                self._apply_labels(example, pool, in_range(positives), in_range(negatives))
                self._inherit_labels(example, pool)
                example.oracle_labeled = True
                labeled.append(example)
                self._cache_label(key, example)
            if reporting_interval <= 0:
                print(
                    f"{self.dataset_id} - Study labels: {batch_start + len(batch)}/{len(example_pool_pairs)} questions, "
                    f"+{stats.study_num_label_positives} pos / {stats.study_num_label_negatives} neg / "
                    f"{stats.study_num_label_ambiguous} ambiguous. {total_label_time:.2f}s."
                )
                reporting_interval = max(1, len(example_pool_pairs) // 20)
            reporting_interval -= len(batch)
        return labeled

    def _cache_label(self, key: StudyCacheKey | None, example: DatasetQAExample) -> None:
        if key is None:
            return
        self.cache.append(key, [{"example_id": example.example_id, "positive_chunk_ids": list(example.positive_chunk_ids or []),
                                 "hard_negative_chunk_ids": list(example.hard_negative_chunk_ids or [])}])
