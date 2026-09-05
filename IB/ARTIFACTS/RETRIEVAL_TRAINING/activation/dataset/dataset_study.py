"""
Generate study material for a given dataset.
Calls a moderately powerful model (e.g., up to ~40B, likely quantized) with well-chosen snippets to generate questions.
Study generation is simple. Until study budget is reached:
- Select random chunk.
- Prompt.
- Parse.
Retrieval label generation is also simple:
- Ask the model to pick positives/hard-negative from the top bm25 pool.
- Use native labels (treating the top-ranked chunk as a positive/negative).
- Labeler always wins.
"""
import json
import random
import time
import typing as t

from vllm.sampling_params import StructuredOutputsParams

from .dataset import (
    DataOrigin,
    DataSplit,
    LabeledRetrievalQAExample,
    LoadedDataset,
    DatasetDocumentChunk,
)
from .dataset_utils import safe_truncate_embedding_chunk, shuffle_fill_truncate
from ..harness.vllm_wrapper import RECOMMENDED_BATCH_SIZE

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime


STUDY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["question", "answer"],
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



def make_study_prompt(study_context: str | None) -> tuple[str, str, dict]:
    """
    Returns system prompt, instructions, json schema.
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
    "question": "...",
    "answer": "..."
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
        self.study_context = loaded_dataset.study_context
        self.qa_model_name = harness.harness_config.dataset_study_qa_model_name
        self.label_model_name = harness.harness_config.dataset_study_label_model_name
        self.qa_batch_size = harness.harness_config.dataset_study_qa_batch_size or RECOMMENDED_BATCH_SIZE
        self.label_batch_size = harness.harness_config.dataset_study_label_batch_size or RECOMMENDED_BATCH_SIZE
        self.chunk_input_limit = harness.harness_config.dataset_study_chunk_input_limit
        self.chat_kwargs = harness.harness_config.dataset_study_chat_kwargs
        self.study_seed = harness.harness_config.dataset_study_seed
        self.label_top_k = harness.harness_config.dataset_study_label_top_k
        self.label_pool_max_chars = harness.harness_config.dataset_study_label_pool_max_chars
        assert self.dataset_index.bm25_index is not None, "Study requires built bm25 indexes."


    def _sample_study_chunks(self, num_samples: int) -> list[DatasetDocumentChunk]:
        rng = random.Random(self.study_seed) # Reproducible study sets.
        chunk_ids = shuffle_fill_truncate(list(self.dataset_index.chunk_ids), num_samples, rng)  # every chunk before any repeat
        return [self.dataset_index.chunks[chunk_id] for chunk_id in chunk_ids]

    def _make_engine_chat_kwargs(self, json_schema: dict) -> dict:
        chat_kwargs = dict(self.chat_kwargs or {})
        sampling_params_kwargs = dict(chat_kwargs.pop("sampling_params", None) or {})
        sampling_params_kwargs.setdefault("max_tokens", 2048)
        sampling_params_kwargs.setdefault("seed", self.study_seed)
        sampling_params_kwargs["structured_outputs"] = StructuredOutputsParams(json=json_schema)
        chat_kwargs["sampling_params"] = sampling_params_kwargs
        return chat_kwargs


    def generate_examples_qa(self, num_samples: int) -> list[LabeledRetrievalQAExample]:
        loaded_model = self.harness.loaded_models[self.qa_model_name]
        # vllm batches a whole conversation list inside one chat() call, so a
        # simple single-threaded loop needs no locks.
        loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
        study_samples = self._sample_study_chunks(num_samples)
        system_prompt, instructions, json_schema = make_study_prompt(self.study_context)
        chat_kwargs = self._make_engine_chat_kwargs(json_schema)
        stats = self.loaded_dataset.stats
        examples: list[LabeledRetrievalQAExample] = []
        example_count = 0
        total_generation_time = 0
        reporting_interval = 0
        for batch_start in range(0, len(study_samples), self.qa_batch_size):
            batch = study_samples[batch_start:batch_start + self.qa_batch_size]
            conversations = []
            for chunk in batch:
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
            for chunk, output in zip(batch, outputs):
                stats.study_prompt_tokens.append(output.prompt_token_count)
                stats.study_output_tokens.append(output.output_token_count)
                try:
                    qa_pair = json.loads(output.text)
                    question = str(qa_pair["question"]).strip()
                    answer = str(qa_pair["answer"]).strip()
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Structured outputs make this rare (e.g. max_tokens cutoff).
                    stats.study_num_parse_failures += 1
                    continue
                if not question or not answer:
                    stats.study_num_parse_failures += 1
                    continue
                example = LabeledRetrievalQAExample(
                    example_id=f"{self.dataset_id}:study:{example_count}",
                    dataset_id=self.dataset_id,
                    query=question,
                    gold_answers=[answer],
                    origin=DataOrigin.SYNTHETIC,
                    split=DataSplit.TRAIN,
                    positive_doc_ids=[chunk.doc_id],
                    positive_chunk_ids=[chunk.chunk_id],
                )
                self.loaded_dataset.labeled_retrieval_examples[example.example_id] = example
                examples.append(example)
                example_count += 1
            if reporting_interval <= 0:
                print(
                    f"{self.dataset_id} - Study: {batch_start + len(batch)}/{len(study_samples)} chunks, "
                    f"{example_count} questions. {total_generation_time:.2f}s."
                )
                reporting_interval = max(1, len(study_samples) // 20)
            reporting_interval -= len(batch)
        return examples


    def _select_examples_to_label(self, num_samples: int, synthetic_only: bool) -> list[LabeledRetrievalQAExample]:
        """
        Select examples to label.
        """
        candidates = [
            example
            for example in self.loaded_dataset.labeled_retrieval_examples.values()
            if not example.oracle_labeled
            and example.gold_answers and example.gold_answers[0]
            and (not synthetic_only or example.origin == DataOrigin.SYNTHETIC)
        ]
        rng = random.Random(self.study_seed)
        rng.shuffle(candidates)
        return candidates[:num_samples]

    def _build_label_pool(
        self,
        example: LabeledRetrievalQAExample,
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
        example: LabeledRetrievalQAExample,
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
        example: LabeledRetrievalQAExample,
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


    def generate_examples_labels(self, num_samples: int, synthetic_only: bool) -> list[LabeledRetrievalQAExample]:
        """
        Oracle-label up to num_samples examples that were not labeled yet.
        When synthetic_only is true, only synthetically generated questions are considered.
        Returns the examples that ended up labeled in this pass.
        """
        examples = self._select_examples_to_label(num_samples, synthetic_only)
        if not examples:
            return []
        loaded_model = self.harness.loaded_models[self.label_model_name]
        loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
        system_prompt, instructions, json_schema = make_label_prompt()
        chat_kwargs = self._make_engine_chat_kwargs(json_schema)
        stats = self.loaded_dataset.stats
        all_ranked = self.dataset_index.bm25_query_many_frozen(
            [example.query for example in examples],
            top_k=self.label_top_k,
            excluded_doc_ids=[example.excluded_doc_ids for example in examples],
        )
        labeled: list[LabeledRetrievalQAExample] = []
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
            outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
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
            if reporting_interval <= 0:
                print(
                    f"{self.dataset_id} - Study labels: {batch_start + len(batch)}/{len(example_pool_pairs)} questions, "
                    f"+{stats.study_num_label_positives} pos / {stats.study_num_label_negatives} neg / "
                    f"{stats.study_num_label_ambiguous} ambiguous. {total_label_time:.2f}s."
                )
                reporting_interval = max(1, len(example_pool_pairs) // 20)
            reporting_interval -= len(batch)
        return labeled
