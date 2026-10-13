"""
Generate study material for a given dataset: questions with reference answers from its chunks (text
chunks, or trajectory chunks: a question about a fact inside an agent's turns), and oracle labels of
which chunks help answer a question. Both steps are cached under the synced folder when the caller
gives a caching id (dataset_caching.py). Nothing is written into the dataset object: the callers get
the examples back and decide what to keep.
"""
import json
import hashlib
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


from .dataset_study_prompts import make_label_prompt, make_source_study_prompt


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

    def _study_source(self, chunk, mode, max_chunks, max_tokens, tokenizer):
        """Select once and retain exactly the text shown to the generator."""
        doc = self.loaded_dataset.documents[chunk.doc_id]
        chunks = sorted(doc.chunks.values(), key=lambda c: c.chunk_start)
        if mode == "document":
            if doc.text is None:
                return None
            selected = chunks
            units = [{"doc_id": doc.doc_id, "chunk_ids": [c.chunk_id for c in selected], "text": doc.text}]
        elif mode == "window":
            position = next(i for i, c in enumerate(chunks) if c.chunk_id == chunk.chunk_id)
            start = max(0, position - max_chunks // 2)
            selected = chunks[start:start + max_chunks]
            units = []
            covered = selected[0].chunk_start
            for candidate in selected:
                end = candidate.chunk_start + len(candidate.chunk_text)
                # Keep adjacent chunks independently citable without repeating their overlap.
                text = (doc.text[covered:end] if doc.text is not None
                        else candidate.chunk_text[max(0, covered - candidate.chunk_start):])
                if text:
                    units.append({"doc_id": doc.doc_id, "chunk_ids": [candidate.chunk_id], "text": text})
                covered = max(covered, end)
        elif mode == "related_chunks":
            selected = [chunk] + self.dataset_index.bm25_query_many_frozen(
                [chunk.chunk_text], top_k=max_chunks - 1, excluded_doc_ids=[[chunk.doc_id]])[0] if max_chunks > 1 else [chunk]
            units = [{"doc_id": c.doc_id, "chunk_ids": [c.chunk_id], "text": c.chunk_text} for c in selected]
        else:
            text = safe_truncate_embedding_chunk(self.dataset_index.get_chunk_section(chunk), self.chunk_input_limit)
            units = [{"doc_id": chunk.doc_id, "chunk_ids": [chunk.chunk_id], "text": text}]
        text = "\n\n".join(f"Source {i}:\n{unit['text']}" for i, unit in enumerate(units))
        if max_tokens is not None and len(tokenizer.encode(text, add_special_tokens=False)) > max_tokens:
            return None  # Explicit skip, never silently shorten a whole document.
        return {"mode": mode, "units": units, "text": text}

    def generate_examples_qa(
        self, num_samples: int, base_seed: int = 0, study_context: str | None = None,
        caching_id: str | None = None, modality: DataModality | None = None,
        *, progress: t.Callable[[dict], None] | None = None,
        source_mode: t.Literal["chunk", "window", "document", "related_chunks"] = "chunk",
        max_source_tokens: int | None = None, max_source_chunks: int = 3,
        prompt_kind: t.Literal["qa", "code", "numeric", "constraints"] = "qa",
        require_multiple_sources: bool = False, max_questions_per_source: int = 1,
    ) -> list[DatasetQAExample]:
        """Generate up to N questions per source attempt; retain exact evidence and cache whole requests."""
        if source_mode not in ("chunk", "window", "document", "related_chunks"):
            raise ValueError(f"Unknown source_mode: {source_mode}")
        if max_source_chunks < 1 or max_questions_per_source < 1 or (max_source_tokens is not None and max_source_tokens < 1):
            raise ValueError("Study source and question budgets must be positive")
        model = self.harness.loaded_models[self.qa_model_name]
        stats = self.loaded_dataset.stats
        system, instructions, schema = make_source_study_prompt(study_context, modality or DataModality.TEXT,
            prompt_kind, max_questions_per_source, require_multiple_sources)
        recipe = hashlib.sha256(json.dumps([system, instructions, schema, source_mode, max_source_tokens,
            max_source_chunks, self.chat_kwargs, self.chunk_input_limit], sort_keys=True, default=str).encode()).hexdigest()[:16]
        key = StudyCacheKey(caching_id, self.dataset_id, model.model_config.model_id, "qa_" + recipe, base_seed) if caching_id else None
        cached = self.cache.read_by_id(key, "request_id") if key else {}
        samples = self._sample_study_chunks(num_samples, base_seed, modality)
        requests = []
        skipped = 0
        for index, chunk in enumerate(samples):
            source = self._study_source(chunk, source_mode, max_source_chunks, max_source_tokens, model.tokenizer)
            if source is None or (require_multiple_sources and len(source["units"]) < 2):
                skipped += 1
                continue
            identity = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
            requests.append({"request_id": f"{recipe}:{base_seed}:{index}:{identity}", "source": source})
        rows = {r["request_id"]: cached[r["request_id"]] for r in requests if r["request_id"] in cached}
        pending = [r for r in requests if r["request_id"] not in rows]
        def report(phase):
            if progress:
                progress({"phase": phase, "completed": len(rows) + skipped, "total": len(samples),
                          "accepted": sum(len(r["questions"]) for r in rows.values()), "cached": len(requests) - len(pending),
                          "skipped_sources": skipped, "failed": stats.study_num_parse_failures})
        report("loading QA model" if pending else "QA complete")
        if pending:
            model.ensure_engine_loaded()
        for start in range(0, len(pending), self.qa_batch_size):
            batch = pending[start:start + self.qa_batch_size]
            conversations = [[{"role": "system", "content": system},
                              {"role": "user", "content": r["source"]["text"] + "\n\n" + instructions}] for r in batch]
            report("generating QA")
            began = time.monotonic()
            outputs = model.engine_chat_many(conversations, chat_kwargs=self._make_engine_chat_kwargs(schema, base_seed + start))
            stats.study_batch_latencies.append(time.monotonic() - began)
            if len(outputs) != len(batch):
                raise RuntimeError("QA engine returned the wrong batch size")
            new_rows = []
            for request, output in zip(batch, outputs):
                stats.study_prompt_tokens.append(output.prompt_token_count)
                stats.study_output_tokens.append(output.output_token_count)
                questions = []
                try:
                    result = json.loads(output.text)
                    seen = set()
                    for q in result["questions"][:max_questions_per_source]:
                        question, answer = q["question"].strip(), q["answer"].strip()
                        try:
                            # JSON permits lone surrogate escapes; cached/model text must be valid UTF-8.
                            question.encode("utf-8")
                            answer.encode("utf-8")
                        except UnicodeEncodeError:
                            stats.study_num_parse_failures += 1
                            continue
                        refs = sorted(set(q["evidence_indices"]))
                        if not question or not answer or question.casefold() in seen:
                            continue
                        if not refs or any(type(i) is not int or not 0 <= i < len(request["source"]["units"]) for i in refs):
                            continue
                        if require_multiple_sources and len(refs) < 2:
                            continue
                        seen.add(question.casefold())
                        questions.append({"question": question, "answer": answer, "evidence_indices": refs})
                except (ValueError, KeyError, TypeError, AttributeError):
                    stats.study_num_parse_failures += 1
                # Cache empty/failed attempts too: stable source positions across restart.
                row = dict(request, questions=questions)
                rows[request["request_id"]] = row
                new_rows.append(row)
                stats.study_num_junk_skipped += int(not questions)
            if key:
                self.cache.append(key, new_rows)
            report("generating QA")
        examples = []
        for request in requests:
            row = rows[request["request_id"]]
            for child, qa in enumerate(row["questions"]):
                supports = [row["source"]["units"][i] for i in qa["evidence_indices"]]
                examples.append(DatasetQAExample(
                    example_id=f"{self.dataset_id}:study:{row['request_id']}:{child}", dataset_id=self.dataset_id,
                    query=qa["question"], gold_answers=[qa["answer"]], origin=SYNTHETIC, split=DataSplit.TRAIN,
                    positive_doc_ids=list(dict.fromkeys(u["doc_id"] for u in supports)),
                    positive_chunk_ids=list(dict.fromkeys(c for u in supports for c in u["chunk_ids"])),
                    study_source=dict(row["source"], request_id=row["request_id"])))
        stats.study_num_cached += len(requests) - len(pending)
        report("QA complete")
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
