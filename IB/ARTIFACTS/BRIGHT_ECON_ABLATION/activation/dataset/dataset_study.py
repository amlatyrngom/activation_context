"""
Generate study material for a given dataset.
Calls a moderately powerful model (e.g., up to ~40B, likely quantized) with well-chosen snippets to generate questions.
Study generation is simple. Until study budget is reached:
- Select random chunk.
- Prompt.
- Parse.
Two study kinds share that loop: a hard question with its answer (origin SYNTHETIC_QA) and a
search-friendly description that becomes the query (origin SYNTHETIC_DESCRIPTION).
Retrieval label generation is also simple:
- Ask the model to pick positives/hard-negative from the top bm25 pool.
- Use native labels (treating the top-ranked chunk as a positive/negative).
- Labeler always wins.
Programmatic examples (origin PROGRAMMATIC) need no model: a span of a chunk is the query and the
chunk its positive, with in-batch negatives only; they live in their own dict and are never labeled.
Generation and labeling outputs can be cached by a caller-chosen caching id (dataset_caching.py).
Prompts live in dataset_study_prompts.py.
"""
import json
import random
import time
import typing as t
from dataclasses import dataclass

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import StructuredOutputsParams

from .dataset import (
    DataOrigin,
    DataSplit,
    LabeledRetrievalQAExample,
    LoadedDataset,
    DatasetDocumentChunk,
)
from .dataset_caching import StudyCache, StudyCacheKey
from .dataset_study_prompts import (
    label_messages,
    make_label_prompt,
    make_study_description_prompt,
    make_study_qa_prompt,
    study_messages,
)
from .dataset_utils import safe_truncate_embedding_chunk, shuffle_fill_truncate
from ..harness.vllm_wrapper import RECOMMENDED_BATCH_SIZE

LABEL_PROMPT_TOKEN_LIMIT = 18_000   # under the engine's 20k context, with room for the answer

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime


@dataclass(frozen=True)
class StudyKind:
    """What differs between the two generated study kinds: the prompt, the parser and the origin."""
    tag: str
    origin: DataOrigin
    make_prompt: t.Callable[[str | None], tuple[str, str, dict]]
    parse: t.Callable[[str], tuple[str, list[str] | None] | None]
    """Output text -> (query, gold answers) or None when the output yields no example."""


def _parse_qa(text: str) -> tuple[str, list[str] | None] | None:
    qa_pair = json.loads(text)
    question = str(qa_pair["question"]).strip()
    answer = str(qa_pair["answer"]).strip()
    if not question or not answer:
        return None
    return question, [answer]


def _parse_description(text: str) -> tuple[str, list[str] | None] | None:
    parsed = json.loads(text)
    description = parsed.get("retrieval_summarization")
    if not parsed["needs_abstraction_bridge"] or not description or not str(description).strip():
        return None                                                    # the snippet needs no bridge: no example
    return str(description).strip(), None


QA_KIND = StudyKind("study", DataOrigin.SYNTHETIC_QA, make_study_qa_prompt, _parse_qa)
DESCRIPTION_KIND = StudyKind("description", DataOrigin.SYNTHETIC_DESCRIPTION, make_study_description_prompt, _parse_description)
ORIGIN_KIND_TAGS = {QA_KIND.origin: QA_KIND.tag, DESCRIPTION_KIND.origin: DESCRIPTION_KIND.tag}   # label cache file per kind


def programmatic_query(text: str, rng: random.Random, min_chars: int, max_chars: int) -> str:
    """
    A random contiguous span of the text, min_chars to max_chars long (the whole text when it is
    shorter than min_chars), snapped outward to whitespace so it starts and ends on whole words.
    """
    text = text.strip()
    if len(text) <= min_chars:
        return text
    length = rng.randint(min_chars, min(max_chars, len(text)))
    start = rng.randint(0, len(text) - length)
    end = start + length
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return text[start:end].strip()


class DatasetStudyGenerator:
    def __init__(self, harness: "HarnessRuntime", loaded_dataset: LoadedDataset):
        self.harness = harness
        self.loaded_dataset = loaded_dataset
        self.dataset_id = loaded_dataset.dataset_id
        self.dataset_index = harness.dataset_manager._get_or_create_index(self.dataset_id)
        self.study_context = None # @AI: Add me later
        self.qa_model_name = harness.harness_config.dataset_study_qa_model_name
        self.label_model_name = harness.harness_config.dataset_study_label_model_name
        self.qa_batch_size = harness.harness_config.dataset_study_qa_batch_size or RECOMMENDED_BATCH_SIZE
        self.label_batch_size = harness.harness_config.dataset_study_label_batch_size or RECOMMENDED_BATCH_SIZE
        self.chunk_input_limit = harness.harness_config.dataset_study_chunk_input_limit
        self.chat_kwargs = harness.harness_config.dataset_study_chat_kwargs
        self.study_seed = harness.harness_config.dataset_study_seed
        self.label_top_k = harness.harness_config.dataset_study_label_top_k
        self.label_pool_max_chars = harness.harness_config.dataset_study_label_pool_max_chars
        self.programmatic_min_chars = harness.harness_config.dataset_study_programmatic_min_chars
        self.programmatic_max_chars = harness.harness_config.dataset_study_programmatic_max_chars
        self.cache = StudyCache(harness.harness_config.cache_storage_dir)
        assert self.dataset_index.bm25_index is not None, "Study requires built bm25 indexes."


    def _sample_study_chunks(self, num_samples: int, seed: int | None = None) -> list[DatasetDocumentChunk]:
        rng = random.Random(self.study_seed if seed is None else seed) # Reproducible study sets.
        chunk_ids = shuffle_fill_truncate(list(self.dataset_index.chunk_ids), num_samples, rng)  # every chunk before any repeat
        return [self.dataset_index.chunks[chunk_id] for chunk_id in chunk_ids]

    def _make_engine_chat_kwargs(self, json_schema: dict, seed: int | None = None) -> dict:
        """vllm seeds its sampler per request (same prompt + same seed = same text), hence a seed per generation pass."""
        chat_kwargs = dict(self.chat_kwargs or {})
        sampling_params_kwargs = dict(chat_kwargs.pop("sampling_params", None) or {})
        sampling_params_kwargs.setdefault("max_tokens", 2048)
        sampling_params_kwargs.setdefault("seed", self.study_seed if seed is None else seed)
        sampling_params_kwargs["structured_outputs"] = StructuredOutputsParams(json=json_schema)
        chat_kwargs["sampling_params"] = sampling_params_kwargs
        return chat_kwargs

    def _model_id(self, model_name: str) -> str:
        return self.harness.loaded_models[model_name].model_config.model_id

    # ----------------------------------------------------------------------------- generation
    def generate_examples_qa(self, num_samples: int, caching_id: str | None = None) -> list[LabeledRetrievalQAExample]:
        return self._generate_examples(num_samples, QA_KIND, caching_id)

    def generate_examples_descriptions(self, num_samples: int, caching_id: str | None = None) -> list[LabeledRetrievalQAExample]:
        return self._generate_examples(num_samples, DESCRIPTION_KIND, caching_id)

    def _register_generated(self, kind: StudyKind, chunk: DatasetDocumentChunk, query: str, gold_answers: list[str] | None, index: int) -> LabeledRetrievalQAExample:
        example = LabeledRetrievalQAExample(
            example_id=f"{self.dataset_id}:{kind.tag}:{index}",
            dataset_id=self.dataset_id,
            query=query,
            gold_answers=gold_answers,
            origin=kind.origin,
            split=DataSplit.TRAIN,
            positive_doc_ids=[chunk.doc_id],
            positive_chunk_ids=[chunk.chunk_id],
        )
        self.loaded_dataset.labeled_retrieval_examples[example.example_id] = example
        return example

    def _generate_examples(self, num_samples: int, kind: StudyKind, caching_id: str | None) -> list[LabeledRetrievalQAExample]:
        """
        One pass per corpus-size slice of the request, pass k seeded with study_seed + k: its own
        shuffle (every chunk once), its own sampling seed (a chunk revisited gets a new question or
        description) and its own cache file. A request within the corpus size is one pass at the
        study seed, so existing caches stay valid; pass 1 of seed s is the file of pass 0 of seed s + 1.
        """
        corpus_size = len(self.dataset_index.chunk_ids)
        starts = list(range(0, num_samples, corpus_size))
        examples: list[LabeledRetrievalQAExample] = []
        for pass_index, start in enumerate(starts):
            pass_samples = min(corpus_size, num_samples - start)
            print(
                f"{self.dataset_id} - Study ({kind.tag}): pass {pass_index + 1}/{len(starts)}, {pass_samples} chunks, "
                f"seed {self.study_seed + pass_index} ({len(examples)} examples so far).", flush=True,
            )
            examples.extend(self._generate_examples_pass(
                pass_samples, kind, caching_id, seed=self.study_seed + pass_index, first_index=len(examples),
            ))
        return examples

    def _generate_examples_pass(
        self, num_samples: int, kind: StudyKind, caching_id: str | None, seed: int, first_index: int,
    ) -> list[LabeledRetrievalQAExample]:
        """
        One example per sampled chunk that the generator turns into a query (questions always,
        descriptions when the snippet needs a bridge). With a caching id, cached rows serve the
        sampled prefix they cover (matched by chunk id) and the rest is generated and appended.
        """
        study_samples = self._sample_study_chunks(num_samples, seed)
        stats = self.loaded_dataset.stats
        examples: list[LabeledRetrievalQAExample] = []
        cache_key = None
        if caching_id:
            cache_key = StudyCacheKey(caching_id, self.dataset_id, self._model_id(self.qa_model_name), kind.tag, seed)
        cached_rows = self.cache.read(cache_key) if cache_key else []
        next_index = first_index                                           # example ids keep counting across passes
        num_from_cache = 0
        for row in cached_rows:                                            # cached prefix: same chunks in the same order
            if num_from_cache >= len(study_samples) or row.get("chunk_id") != study_samples[num_from_cache].chunk_id:
                break
            chunk = study_samples[num_from_cache]
            num_from_cache += 1
            if row.get("query"):
                examples.append(self._register_generated(kind, chunk, row["query"], row.get("gold_answers"), next_index))
                next_index += 1
        if num_from_cache:
            print(f"{self.dataset_id} - Study ({kind.tag}): {num_from_cache} chunks from the cache ({len(examples)} examples).")
        remaining = study_samples[num_from_cache:]
        if not remaining:
            return examples

        loaded_model = self.harness.loaded_models[self.qa_model_name]
        # vllm batches a whole conversation list inside one chat() call, so a
        # simple single-threaded loop needs no locks.
        loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
        system_prompt, instructions, json_schema = kind.make_prompt(self.study_context)
        chat_kwargs = self._make_engine_chat_kwargs(json_schema, seed)
        total_generation_time = 0
        reporting_interval = 0
        for batch_start in range(0, len(remaining), self.qa_batch_size):
            batch = remaining[batch_start:batch_start + self.qa_batch_size]
            conversations = []
            for chunk in batch:
                snippet = self.dataset_index.get_chunk_section(chunk)
                snippet = safe_truncate_embedding_chunk(snippet, self.chunk_input_limit)
                conversations.append(study_messages(system_prompt, instructions, snippet))
            batch_start_time = time.time()
            outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
            elapsed = time.time() - batch_start_time
            total_generation_time += elapsed
            stats.study_batch_latencies.append(elapsed)
            cache_rows = []
            for chunk, output in zip(batch, outputs):
                stats.study_prompt_tokens.append(output.prompt_token_count)
                stats.study_output_tokens.append(output.output_token_count)
                try:
                    parsed = kind.parse(output.text)
                except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                    # Structured outputs make this rare (e.g. max_tokens cutoff).
                    stats.study_num_parse_failures += 1
                    parsed = None
                    cache_rows.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "query": None, "failed": True})
                    continue
                if parsed is None:                                         # a valid answer that yields no example
                    if kind is QA_KIND:
                        stats.study_num_parse_failures += 1
                    else:
                        stats.study_num_description_no_bridge += 1
                    cache_rows.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "query": None})
                    continue
                query, gold_answers = parsed
                examples.append(self._register_generated(kind, chunk, query, gold_answers, next_index))
                next_index += 1
                cache_rows.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "query": query, "gold_answers": gold_answers})
            if cache_key:
                self.cache.append(cache_key, cache_rows)
            if reporting_interval <= 0:
                print(
                    f"{self.dataset_id} - Study ({kind.tag}): {num_from_cache + batch_start + len(batch)}/{len(study_samples)} chunks, "
                    f"{len(examples)} examples. {total_generation_time:.2f}s."
                )
                reporting_interval = max(1, len(remaining) // 20)
            reporting_interval -= len(batch)
        return examples

    def generate_programmatic_examples(self, num_samples: int) -> list[LabeledRetrievalQAExample]:
        """
        Programmatic examples for massive AC model training: a random span of a sampled chunk is
        the query, the chunk is its own positive, negatives come from the batch at training time.
        Chunks repeat past the corpus size with a different span each time. Not cached, never
        labeled, stored apart from the labeled examples.
        """
        chunks = self._sample_study_chunks(num_samples)
        rng = random.Random(self.study_seed + 1)
        examples: list[LabeledRetrievalQAExample] = []
        start_index = len(self.loaded_dataset.programmatic_retrieval_examples)
        for offset, chunk in enumerate(chunks):
            query = programmatic_query(chunk.chunk_text, rng, self.programmatic_min_chars, self.programmatic_max_chars)
            example = LabeledRetrievalQAExample(
                example_id=f"{self.dataset_id}:programmatic:{start_index + offset}",
                dataset_id=self.dataset_id,
                query=query,
                origin=DataOrigin.PROGRAMMATIC,
                split=DataSplit.TRAIN,
                positive_doc_ids=[chunk.doc_id],
                positive_chunk_ids=[chunk.chunk_id],
            )
            self.loaded_dataset.programmatic_retrieval_examples[example.example_id] = example
            examples.append(example)
        print(f"{self.dataset_id} - {len(examples)} programmatic examples ({len(self.dataset_index.chunk_ids)} chunks in the corpus).")
        return examples

    # ----------------------------------------------------------------------------- labeling
    def _select_examples_to_label(self, num_samples: int, origins: list[DataOrigin] | None) -> list[LabeledRetrievalQAExample]:
        """
        Select examples to label: unlabeled ones with something to judge against (a reference
        answer, or a description), of the given origins (None: every origin).
        """
        candidates = [
            example
            for example in self.loaded_dataset.labeled_retrieval_examples.values()
            if not example.oracle_labeled
            and ((example.gold_answers and example.gold_answers[0]) or example.origin == DataOrigin.SYNTHETIC_DESCRIPTION)
            and (origins is None or example.origin in origins)
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


    def generate_examples_labels(
        self, num_samples: int, origins: list[DataOrigin] | None = None, caching_id: str | None = None,
        label_model_name: str | None = None,
    ) -> list[LabeledRetrievalQAExample]:
        """
        Oracle-label up to num_samples examples of the given origins (None: all) that were not
        labeled yet; descriptions are judged alone, questions with their reference answer. With a
        caching id, examples whose labels are cached take them without a model call; new labels
        are appended. label_model_name overrides the configured labeler (a probe compares two).
        Returns the examples that ended up labeled in this pass.
        """
        examples = self._select_examples_to_label(num_samples, origins)
        if not examples:
            return []
        label_model_name = label_model_name or self.label_model_name
        stats = self.loaded_dataset.stats
        cache_key = None
        if caching_id:
            # One file per study kind: questions and descriptions are labeled by different runs (and nodes),
            # and a shared file name would make their synced copies overwrite each other.
            kinds = sorted({ORIGIN_KIND_TAGS.get(example.origin, "native") for example in examples})
            cache_key = StudyCacheKey(caching_id, self.dataset_id, self._model_id(self.qa_model_name), "labels-" + "+".join(kinds),
                                      self.study_seed, label_model_id=self._model_id(label_model_name))
        cached = self.cache.read_by_id(cache_key, "example_id") if cache_key else {}
        labeled: list[LabeledRetrievalQAExample] = []
        to_label = []
        for example in examples:
            row = cached.get(example.example_id)
            if row is None:
                to_label.append(example)
                continue
            example.positive_chunk_ids = list(row["positive_chunk_ids"])
            example.hard_negative_chunk_ids = list(row["hard_negative_chunk_ids"]) or None
            example.oracle_labeled = True
            labeled.append(example)
        if labeled:
            print(f"{self.dataset_id} - Study labels: {len(labeled)} examples from the cache.")
        if not to_label:
            return labeled

        loaded_model = self.harness.loaded_models[label_model_name]
        loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
        prompts = {for_qa: make_label_prompt(for_qa) for for_qa in (True, False)}
        chat_kwargs = self._make_engine_chat_kwargs(prompts[True][2])
        all_ranked = self.dataset_index.bm25_query_many_frozen(
            [example.query for example in to_label],
            top_k=self.label_top_k,
            excluded_doc_ids=[example.excluded_doc_ids for example in to_label],
        )
        example_pool_pairs = []
        cache_rows = []
        for example, ranked in zip(to_label, all_ranked):
            pool = self._build_label_pool(example, ranked)
            if pool:
                example_pool_pairs.append((example, pool))
            else:
                # Nothing for the oracle to judge: inheritance alone labels the example.
                self._inherit_labels(example, pool)
                example.oracle_labeled = True
                labeled.append(example)
                cache_rows.append(self._label_row(example))
        if cache_key:
            self.cache.append(cache_key, cache_rows)
        total_label_time = 0
        reporting_interval = 0
        for batch_start in range(0, len(example_pool_pairs), self.label_batch_size):
            batch = example_pool_pairs[batch_start:batch_start + self.label_batch_size]
            conversations = []
            for example, pool in batch:
                for_qa = example.origin != DataOrigin.SYNTHETIC_DESCRIPTION
                system_prompt, instructions, _ = prompts[for_qa]
                conversations.append(label_messages(system_prompt, instructions, example, [snippet for _, snippet in pool], for_qa))
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
            cache_rows = []
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
                cache_rows.append(self._label_row(example))
            if cache_key:
                self.cache.append(cache_key, cache_rows)
            if reporting_interval <= 0:
                print(
                    f"{self.dataset_id} - Study labels: {batch_start + len(batch)}/{len(example_pool_pairs)} examples, "
                    f"+{stats.study_num_label_positives} pos / {stats.study_num_label_negatives} neg / "
                    f"{stats.study_num_label_ambiguous} ambiguous. {total_label_time:.2f}s."
                )
                reporting_interval = max(1, len(example_pool_pairs) // 20)
            reporting_interval -= len(batch)
        return labeled

    @staticmethod
    def _label_row(example: LabeledRetrievalQAExample) -> dict:
        return {"example_id": example.example_id, "positive_chunk_ids": list(example.positive_chunk_ids or []),
                "hard_negative_chunk_ids": list(example.hard_negative_chunk_ids or [])}

    # ----------------------------------------------------------------------------- selection
    def _training_candidates(self, origins: list[DataOrigin] | None, oracle_labeled_only: bool) -> list[LabeledRetrievalQAExample]:
        """Examples of the given origins from both dicts, chunk labels inherited where missing; no positive chunk drops one."""
        stats = self.loaded_dataset.stats
        pools = [self.loaded_dataset.labeled_retrieval_examples.values()]
        if origins is None or DataOrigin.PROGRAMMATIC in origins:
            pools.append(self.loaded_dataset.programmatic_retrieval_examples.values())
        selected: list[LabeledRetrievalQAExample] = []
        for pool in pools:
            for example in pool:
                if origins is not None and example.origin not in origins:
                    continue
                if oracle_labeled_only and not example.oracle_labeled:
                    continue
                if not example.oracle_labeled and not example.positive_chunk_ids:
                    self._inherit_labels(example, pool=[])
                if example.positive_chunk_ids:
                    selected.append(example)
                else:
                    stats.training_select_num_dropped_no_positive += 1
        return selected

    def select_training_data(
        self,
        num_samples: int,
        origins: list[DataOrigin] | None = None,
        oracle_labeled_only: bool = True,
        val_ratio: float = 0.1,
        max_reporting_size: int = 50,
        force_partition: bool = True, # Most datasets only have training. This forces a val set.
        seed: int = 0,
    ) -> tuple[list[LabeledRetrievalQAExample], list[LabeledRetrievalQAExample], list[LabeledRetrievalQAExample]]:
        """
        Select training data. Returns tuples with the following:
        - training data: actually used to update the gradients.
        - validation: small amount of data for validation (~10% in general).
        - reporting: trivial amount of data (subset of the validation set). Used for plotting.
        There is a general min of 10 for training and validation, and 1 for reporting regardless of the fractions.

        num_samples examples are selected before the split, of the given origins (None: every
        origin, programmatic included); validation is carved from them by val_ratio unless
        force_partition is False and the dataset carries native validation-split examples.
        Oracle-labeled examples are used as they are. Without the oracle, an example inherits
        chunk labels from its document-level labels: one chunk per positive document and, where
        the dataset carries native hard negatives, one chunk per hard-negative document. Examples
        without a positive chunk are dropped and counted in the dataset stats. Test-split examples
        are never selected here; they belong to select_testing_data.
        """
        dataset_id = self.dataset_id
        stats = self.loaded_dataset.stats
        selected = [example for example in self._training_candidates(origins, oracle_labeled_only) if example.split != DataSplit.TEST]
        rng = random.Random(seed)
        native_validation = [example for example in selected if example.split == DataSplit.VAL]
        if not force_partition and native_validation:
            training_pool = [example for example in selected if example.split != DataSplit.VAL]
            rng.shuffle(training_pool)
            rng.shuffle(native_validation)
            training_data = training_pool[:num_samples]
            validation_data = native_validation[:max(10, round(num_samples * val_ratio))]
        else:
            rng.shuffle(selected)
            chosen = selected[:num_samples]
            num_validation = max(10, round(len(chosen) * val_ratio))
            validation_data = chosen[:num_validation]
            training_data = chosen[num_validation:]
        assert len(training_data) >= 10, (
            f"{dataset_id} - Only {len(training_data)} training examples after the split; need at least 10 "
            f"({len(selected)} selectable, {stats.training_select_num_dropped_no_positive} dropped without a positive chunk)."
        )
        assert len(validation_data) >= 10, f"{dataset_id} - Only {len(validation_data)} validation examples; need at least 10."
        reporting_data = validation_data[:max(1, min(max_reporting_size, len(validation_data)))]
        print(
            f"{dataset_id} - Selected {len(training_data)} training / {len(validation_data)} validation / "
            f"{len(reporting_data)} reporting examples."
        )
        return training_data, validation_data, reporting_data

    def select_testing_data(self, num_samples: int | None = None, seed: int = 0) -> list[LabeledRetrievalQAExample]:
        """
        The native test-split examples (BRIGHT: every labeled query) with chunk labels inherited
        from their gold documents, shuffled by the seed and truncated to num_samples (None: all).
        Examples without a positive chunk are dropped and counted.
        """
        stats = self.loaded_dataset.stats
        selected = []
        for example in self.loaded_dataset.labeled_retrieval_examples.values():
            if example.split != DataSplit.TEST or example.origin != DataOrigin.NATIVE:
                continue
            if not example.oracle_labeled and not example.positive_chunk_ids:
                self._inherit_labels(example, pool=[])
            if example.positive_chunk_ids:
                selected.append(example)
            else:
                stats.training_select_num_dropped_no_positive += 1
        rng = random.Random(seed)
        rng.shuffle(selected)
        if num_samples is not None:
            selected = selected[:num_samples]
        print(f"{self.dataset_id} - Selected {len(selected)} testing examples.")
        return selected
