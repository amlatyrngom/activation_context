# Study labeling round

Diff cards for the study-labeling handoff: unified chunking with the `atomic` flag removed, an
in-document bm25 scoring pair (`score_all` on the adapter, `bm25_rank_doc_chunks` on the index),
the three label knobs and eleven label stats, the selection function, the rewritten label
contract with inheritance, the harness import cleanup, prompt fixes, manager returns, and both
tests. Every card is the exact delta of the workspace file against your current `/source` tree;
the staged copies under `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/activation/` match the cards.

## Validation

| Check | Result |
| --- | --- |
| `test_basic_dataset_loading.py` on CPU (bm25 + dense, basic + public) | `4 passed in 308.33s`, including unified-chunk assertions and the in-doc scoring check |
| Labeling contract on CPU with `engine_chat_many` replaced by fixed JSON (private, `IB/TMP/RETRIEVAL_TRAINING/private_label_contract_check.py`, not part of the handoff) | all listed behaviors hold: chunk fields only, doc ids untouched, out-of-range counted with valid indices kept, unparseable leaves `oracle_labeled` False and inherits nothing, absent single-chunk positive document inherits a positive chunk, zero-score multi-chunk document is skipped and counted, an abstained pool chunk blocks inheritance for its document, no chunk on both sides, unparseable example is selectable again |
| `test_basic_dataset_study.py` on a fresh RTX PRO 6000 node (after the port into user source) | `1 passed in 700.76s`: 16 synthetic + 20 native examples labeled, 0 parse failures, 56 oracle positives / 269 negatives / 15 ambiguous, 49 inherited positives, 0 inherit skips; doc ids unchanged. Log: `IB/TMP/STUDY_LABELING/sky_study_test.log` |
| `test_basic_dataset_loading.py` after the port into user source | `4 passed in 347.18s` |
| `test_basic_dataset_study.py` again after the stats follow-up (explicit `ensure_engine_loaded()` before both timed loops; total prompt/output tokens and label tokens/s in `summarize()`), fresh RTX PRO 6000, no manual JIT env prefix | `1 passed in 776.68s`; engine load 647.5 s outside the stats; study 38.9 output tok/s (first batch still holds ~50 s of Triton kernel JIT during inference; second batch ~500 tok/s), labels 11.2k prompt tok/s. Log: `IB/TMP/STUDY_LABELING/sky_study_test_2.log` |

After the port, `runtime_config.py` additionally sets `MAX_JOBS=3` and `NVCC_THREADS=1` through `os.environ.setdefault` after `load_dotenv()`. The first GPU attempt without them was OOM-killed by the NVFP4 kernel JIT's compiler fan-out (`IB/TMP/STUDY_LABELING/sky_study_test_oom_attempt1.log`). The `runtime_config.py` card below predates that addition; the staged copy includes it.

## Two judgment calls to confirm

- **Oracle verdict versus a pre-existing chunk label.** If the oracle marks positive a chunk that
  already sits in `hard_negative_chunk_ids` (only possible if a loader pre-filled chunk-level
  negatives, which none does today), the chunk moves to the positives. "Oracle verdicts are
  final for every chunk in the pool" seemed to imply that; say so if you would rather keep the
  earlier label.
- **`IB/CANON/ROOT_CANON.md`** is shared write-through, so the chunk-id convention line is
  already updated on your side and carries no card.

## Diffs to apply

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_study.py</span>
    <span class="card-oneliner">Selection, label contract, inheritance, prompt fixes, harness import cleanup.</span>
    <span class="card-badge">Diff</span>
  </summary>

`_select_examples_to_label` (unlabeled, has a reference answer, synthetic filter, seeded shuffle, no replacement); `_build_label_pool` keeps only the reference-chunk skip; `_apply_labels` writes chunk ids only and drops both-sided indices; new `_inherit_labels`; `generate_example_labels` passes `excluded_doc_ids`, batches only non-empty pools with `dataset_study_label_batch_size`, handles the four per-example outcomes, sets `oracle_labeled`, returns the labeled list. No `TARGET_DEVICE` import or explicit engine placement in generation. Exact delta vs `/source/activation/dataset/dataset_study.py`:

````diff-python
diff --git a/activation/source/activation/dataset/dataset_study.py b/activation/activation/dataset/dataset_study.py
index 363873d..a71ae61 100644
--- a/activation/source/activation/dataset/dataset_study.py
+++ b/activation/activation/dataset/dataset_study.py
@@ -7,6 +7,14 @@ Study generation is simple. Until study budget is reached:
 - Select random chunk.
 - Prompt.
 - Parse.
+
+Study labeling turns each question into chunk-level retrieval labels:
+- Take the bm25 top-k chunks for the question (the reference chunk never enters the pool).
+- Gather them in rank order until the pool char budget is reached.
+- Ask the study model which snippets are positives / negatives; anything left out is ambiguous.
+- Oracle verdicts are final for every chunk in the pool. Doc-level ids are never written.
+- Known positive / hard-negative documents the oracle never saw a chunk of inherit one chunk
+  (the document's best bm25 chunk for the question) with the document's label.
 """
 import json
 import random
@@ -58,10 +66,10 @@ def make_study_prompt(study_context: str | None) -> tuple[str, str, dict]:
 # Core Guidelines
 - You are a study/exam-like questions formulator.
 - You are given a snippet from the corpus students are expected to study and understand.
-- From this, your goal is to generate a hard question (unanswerable with common knowledge or simple keywork look ups).
+- From this, your goal is to generate a hard question (unanswerable with common knowledge or simple keyword look ups).
     - This is IMPORTANT: the question should be **hard**, or it won't test the student's learning.
     - The question should have an accompanying answer.
-- Your priorities is: a hard question along with its expected, correct answer.
+- Your priority is: a hard question along with its expected, correct answer.
 - By default, keep your question short to medium-short unless the exam recommends longer questions.
 """
 
@@ -106,7 +114,7 @@ def make_label_prompt() -> tuple[str, str, dict]:
     - **negative**: the snippet contains information that clearly does not help answer the question.
     - Leave out anything where you are unsure.
 - Be careful:
-    - The snippets are intentionally chosen to *related* to the question without necessarily helping.
+    - The snippets are intentionally chosen to be *related* to the question without necessarily helping.
     - Don't mark something as positive just because of shared vocabulary.
 - Every listed index must come from the snippet numbering; never list the same index on both sides.
 """
@@ -116,7 +124,7 @@ Your answer must be formatted as a json object of snippet indices like:
 ```json
 {
     "positives": [0, 2],
-    "negatives": [3, 5, 6],
+    "negatives": [3, 5, 6]
 }
 ```
 Each is a list of indexes that should map to one of the given snippet indices.
@@ -136,6 +144,7 @@ class DatasetStudyGenerator:
         self.study_context = loaded_dataset.study_context
         self.study_model_name = harness.harness_config.dataset_study_model_name
         self.batch_size = harness.harness_config.dataset_study_batch_size
+        self.label_batch_size = harness.harness_config.dataset_study_label_batch_size
         self.chunk_input_limit = harness.harness_config.dataset_study_chunk_input_limit
         self.chat_kwargs = harness.harness_config.dataset_study_chat_kwargs
         self.study_seed = harness.harness_config.dataset_study_seed
@@ -159,12 +168,10 @@ class DatasetStudyGenerator:
         return chat_kwargs
 
 
-    def generate_study_examples(self, num_samples: int):
-        from ..harness.hf_utils import TARGET_DEVICE
+    def generate_study_examples(self, num_samples: int) -> list[LabeledRetrievalQAExample]:
         loaded_model = self.harness.loaded_models[self.study_model_name]
         # vllm batches a whole conversation list inside one chat() call, so a
-        # simple single-threaded loop needs no locks.
-        loaded_model.engine_to_device(TARGET_DEVICE)
+        # simple single-threaded loop needs no locks. engine_chat_many places the engine.
         study_samples = self._sample_study_chunks(num_samples)
         system_prompt, instructions, json_schema = make_study_prompt(self.study_context)
         chat_kwargs = self._make_engine_chat_kwargs(json_schema)
@@ -235,9 +242,20 @@ class DatasetStudyGenerator:
 
     def _select_examples_to_label(self, num_samples: int, synthetic_only: bool) -> list[LabeledRetrievalQAExample]:
         """
-        Select the set of examples to label.
+        Up to num_samples not-yet-labeled examples with a reference answer (the label prompt
+        shows it), synthetic only when requested, in seeded random order without replacement:
+        relabeling an example would overwrite its labels.
         """
-        pass # @AI: Implement me.
+        candidates = [
+            example
+            for example in self.loaded_dataset.labeled_retrieval_examples.values()
+            if not example.oracle_labeled
+            and example.gold_answers and example.gold_answers[0]
+            and (not synthetic_only or example.origin == DataOrigin.SYNTHETIC)
+        ]
+        rng = random.Random(self.study_seed)
+        rng.shuffle(candidates)
+        return candidates[:num_samples]
 
     def _build_label_pool(
         self,
@@ -245,12 +263,10 @@ class DatasetStudyGenerator:
         ranked_chunks: list[DatasetDocumentChunk],
     ) -> list[tuple[DatasetDocumentChunk, str]]:
         """
-        (chunk, snippet text) pairs in bm25 rank order, skipping chunks already labeled
-        positive (the source chunk), gathered append-then-break until the pool char budget.
+        (chunk, snippet text) pairs in bm25 rank order, skipping chunks already in
+        positive_chunk_ids (the reference chunk), gathered append-then-break until the pool
+        char budget.
         """
-        # @AI: Let's think about what do positive and hard negative doc ids.
-        # My intuition is to prepend the top 2 matching chunks from these docs to the ranked candidates.
-        # Meaning two additional calls to the bm25 index (one for positives and one for negatives)  
         known_positives = set(example.positive_chunk_ids or [])
         pool = []
         total_chars = 0
@@ -273,43 +289,87 @@ class DatasetStudyGenerator:
         negatives: list[int],
     ):
         """
-        Positives extend positive_chunk_ids / positive_doc_ids; negatives extend
-        hard_negative_chunk_ids, and hard_negative_doc_ids only for documents that have no
-        positive chunk for this example. Indices listed on both sides are ambiguous.
+        Oracle verdicts are final for every chunk in the pool: positives go to
+        positive_chunk_ids, negatives to hard_negative_chunk_ids, and doc-level ids are never
+        written. Indices listed on both sides are dropped from both sets; the reference positive
+        never enters the pool, so it cannot be contradicted.
         """
-        # @AI: This code should drop ambiguous negatives (keep reference positives always).
         stats = self.loaded_dataset.stats
         positive_set = set(positives) - set(negatives)
         negative_set = set(negatives) - set(positives)
         example.positive_chunk_ids = list(example.positive_chunk_ids or [])
-        example.positive_doc_ids = list(example.positive_doc_ids or [])
         example.hard_negative_chunk_ids = list(example.hard_negative_chunk_ids or [])
-        example.hard_negative_doc_ids = list(example.hard_negative_doc_ids or [])
         for index in sorted(positive_set):
             chunk, _ = pool[index]
+            if chunk.chunk_id in example.hard_negative_chunk_ids:
+                example.hard_negative_chunk_ids.remove(chunk.chunk_id) # The oracle verdict wins.
             if chunk.chunk_id not in example.positive_chunk_ids:
                 example.positive_chunk_ids.append(chunk.chunk_id)
-            if chunk.doc_id not in example.positive_doc_ids:
-                example.positive_doc_ids.append(chunk.doc_id)
-        positive_docs = set(example.positive_doc_ids)
         for index in sorted(negative_set):
             chunk, _ = pool[index]
             if chunk.chunk_id not in example.hard_negative_chunk_ids:
                 example.hard_negative_chunk_ids.append(chunk.chunk_id)
-            if chunk.doc_id not in positive_docs and chunk.doc_id not in example.hard_negative_doc_ids:
-                example.hard_negative_doc_ids.append(chunk.doc_id)
         stats.study_num_label_positives += len(positive_set)
         stats.study_num_label_negatives += len(negative_set)
         stats.study_num_label_ambiguous += len(pool) - len(positive_set) - len(negative_set)
 
 
+    def _inherit_labels(
+        self,
+        example: LabeledRetrievalQAExample,
+        pool: list[tuple[DatasetDocumentChunk, str]],
+    ):
+        """
+        One chunk per known positive / hard-negative document inherits the document's label,
+        but only when the oracle never saw that chunk: a single-chunk document contributes its
+        chunk, a multi-chunk document its best bm25 chunk for the query (skipped when that
+        chunk scores zero). A candidate that was in the pool keeps the oracle's verdict,
+        including abstention. A chunk never lands on both sides: an existing label wins.
+        """
+        stats = self.loaded_dataset.stats
+        pool_chunk_ids = {chunk.chunk_id for chunk, _ in pool}
+        positive_docs = list(example.positive_doc_ids or [])
+        negative_docs = list(example.hard_negative_doc_ids or [])
+        multi_chunk_docs = [
+            doc_id for doc_id in positive_docs + negative_docs
+            if len(self.loaded_dataset.documents[doc_id].chunks) > 1
+        ]
+        ranked = (
+            self.dataset_index.bm25_rank_doc_chunks(example.query, multi_chunk_docs)
+            if multi_chunk_docs else {}
+        )
+        example.positive_chunk_ids = list(example.positive_chunk_ids or [])
+        example.hard_negative_chunk_ids = list(example.hard_negative_chunk_ids or [])
+        for doc_ids, own_ids, other_ids, counter in [
+            (positive_docs, example.positive_chunk_ids, example.hard_negative_chunk_ids, "study_num_label_inherited_positives"),
+            (negative_docs, example.hard_negative_chunk_ids, example.positive_chunk_ids, "study_num_label_inherited_negatives"),
+        ]:
+            for doc_id in doc_ids:
+                document = self.loaded_dataset.documents[doc_id]
+                if len(document.chunks) == 1:
+                    candidate = next(iter(document.chunks.values()))
+                else:
+                    candidate, top_score = ranked[doc_id][0]
+                    if top_score <= 0:
+                        stats.study_num_label_inherit_skipped += 1
+                        continue
+                if candidate.chunk_id in pool_chunk_ids:
+                    continue # The oracle saw it; its verdict (or abstention) stands.
+                if candidate.chunk_id in own_ids or candidate.chunk_id in other_ids:
+                    continue
+                own_ids.append(candidate.chunk_id)
+                setattr(stats, counter, getattr(stats, counter) + 1)
+
 
-    def generate_example_labels(self, num_samples: int, synthetic_only: bool):
+    def generate_example_labels(self, num_samples: int, synthetic_only: bool) -> list[LabeledRetrievalQAExample]:
         """
-        Generate positive/negative labels for the right number of samples.
+        Oracle-label up to num_samples examples that were not labeled yet.
         When synthetic_only is true, only synthetically generated questions are considered.
+        Returns the examples that ended up labeled in this pass.
         """
         examples = self._select_examples_to_label(num_samples, synthetic_only)
+        if not examples:
+            return []
         loaded_model = self.harness.loaded_models[self.study_model_name]
         system_prompt, instructions, json_schema = make_label_prompt()
         chat_kwargs = self._make_engine_chat_kwargs(json_schema)
@@ -317,16 +377,23 @@ class DatasetStudyGenerator:
         all_ranked = self.dataset_index.bm25_query_many_frozen(
             [example.query for example in examples],
             top_k=self.label_top_k,
+            excluded_doc_ids=[example.excluded_doc_ids for example in examples],
         )
-        pools = [
-            self._build_label_pool(example, ranked)
-            for example, ranked in zip(examples, all_ranked)
-        ]
-        example_pool_pairs = list(zip(examples, pools))
+        labeled: list[LabeledRetrievalQAExample] = []
+        example_pool_pairs = []
+        for example, ranked in zip(examples, all_ranked):
+            pool = self._build_label_pool(example, ranked)
+            if pool:
+                example_pool_pairs.append((example, pool))
+            else:
+                # Nothing for the oracle to judge: inheritance alone labels the example.
+                self._inherit_labels(example, pool)
+                example.oracle_labeled = True
+                labeled.append(example)
         total_label_time = 0
         reporting_interval = 0
-        for batch_start in range(0, len(examples), self.batch_size):
-            batch = example_pool_pairs[batch_start:batch_start + self.batch_size]
+        for batch_start in range(0, len(example_pool_pairs), self.label_batch_size):
+            batch = example_pool_pairs[batch_start:batch_start + self.label_batch_size]
             conversations = []
             for example, pool in batch:
                 snippets = "\n".join(
@@ -356,21 +423,22 @@ class DatasetStudyGenerator:
                     positives = [int(index) for index in labels["positives"]]
                     negatives = [int(index) for index in labels["negatives"]]
                 except (json.JSONDecodeError, KeyError, TypeError, ValueError):
+                    # Left unlabeled (oracle_labeled stays False) so a later pass can retry cleanly.
                     stats.study_num_label_parse_failures += 1
                     continue
                 in_range = lambda indexes: [index for index in indexes if 0 <= index < len(pool)]
                 if len(in_range(positives)) + len(in_range(negatives)) < len(positives) + len(negatives):
                     stats.study_num_label_parse_failures += 1 # Out-of-range index: count, keep the valid ones.
                 self._apply_labels(example, pool, in_range(positives), in_range(negatives))
+                self._inherit_labels(example, pool)
+                example.oracle_labeled = True
+                labeled.append(example)
             if reporting_interval <= 0:
                 print(
-                    f"{self.dataset_id} - Study labels: {batch_start + len(batch)}/{len(examples)} questions, "
+                    f"{self.dataset_id} - Study labels: {batch_start + len(batch)}/{len(example_pool_pairs)} questions, "
                     f"+{stats.study_num_label_positives} pos / {stats.study_num_label_negatives} neg / "
                     f"{stats.study_num_label_ambiguous} ambiguous. {total_label_time:.2f}s."
                 )
-                reporting_interval = max(1, len(examples) // 20)
+                reporting_interval = max(1, len(example_pool_pairs) // 20)
             reporting_interval -= len(batch)
-
-
-
-
+        return labeled
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_manager.py</span>
    <span class="card-oneliner">Both study entry points return the examples they produced or labeled.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/dataset_manager.py`:

````diff-python
diff --git a/activation/source/activation/dataset/dataset_manager.py b/activation/activation/dataset/dataset_manager.py
index 4b23dc0..f1fa89b 100644
--- a/activation/source/activation/dataset/dataset_manager.py
+++ b/activation/activation/dataset/dataset_manager.py
@@ -54,19 +54,20 @@ class DatasetManager:
         return
 
     
-    def synthesize_study_examples(self, dataset_id: str, num_samples: int):
-        """Synthesize study examples"""
+    def synthesize_study_examples(self, dataset_id: str, num_samples: int) -> list[LabeledRetrievalQAExample]:
+        """Synthesize study examples. Returns the examples produced."""
         study_generator = self._get_or_create_study_generator(dataset_id)
-        study_generator.generate_study_examples(num_samples)
+        return study_generator.generate_study_examples(num_samples)
 
     def label_study_examples(
         self,
         dataset_id: str,
         num_samples: int,
         synthetic_only: bool = False,
-    ):
+    ) -> list[LabeledRetrievalQAExample]:
+        """Oracle-label up to num_samples not-yet-labeled examples. Returns the examples labeled."""
         study_generator = self._get_or_create_study_generator(dataset_id)
-        study_generator.generate_example_labels(num_samples, synthetic_only)
+        return study_generator.generate_example_labels(num_samples, synthetic_only)
 
 
     def select_training_data(
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset.py</span>
    <span class="card-oneliner">Drop `atomic`; add the label stats and their summary keys.</span>
    <span class="card-badge">Diff</span>
  </summary>

Existing stat names untouched. Exact delta vs `/source/activation/dataset/dataset.py`:

````diff-python
diff --git a/activation/source/activation/dataset/dataset.py b/activation/activation/dataset/dataset.py
index 2e5aa4d..8c321bb 100644
--- a/activation/source/activation/dataset/dataset.py
+++ b/activation/activation/dataset/dataset.py
@@ -47,9 +47,6 @@ class DatasetDocument:
     text: str
     """Actual text"""
 
-    atomic: bool = False
-    """Whether document should be chunked using a much larger chunk size."""
-
     origin: DataOrigin = DataOrigin.NATIVE
     """Source of the doc"""
 
@@ -154,6 +151,17 @@ class DatasetStats:
     study_output_tokens: list[int] = field(default_factory=list)
     study_batch_latencies: list[float] = field(default_factory=list)
     study_num_parse_failures: int = 0
+    study_label_prompt_tokens: list[int] = field(default_factory=list)
+    study_label_output_tokens: list[int] = field(default_factory=list)
+    study_label_batch_latencies: list[float] = field(default_factory=list)
+    study_label_pool_sizes: list[int] = field(default_factory=list)
+    study_num_label_positives: int = 0
+    study_num_label_negatives: int = 0
+    study_num_label_ambiguous: int = 0
+    study_num_label_parse_failures: int = 0
+    study_num_label_inherited_positives: int = 0
+    study_num_label_inherited_negatives: int = 0
+    study_num_label_inherit_skipped: int = 0
 
     def summarize(self) -> dict:
         # Returns average statistics.
@@ -192,6 +200,18 @@ class DatasetStats:
                 if self.study_batch_latencies else 0.0
             ),
             "study_num_parse_failures": self.study_num_parse_failures,
+            "num_study_label_requests": len(self.study_label_prompt_tokens),
+            "avg_study_label_prompt_tokens": average(self.study_label_prompt_tokens),
+            "avg_study_label_output_tokens": average(self.study_label_output_tokens),
+            "total_study_label_latency": total(self.study_label_batch_latencies),
+            "avg_study_label_pool_size": average(self.study_label_pool_sizes),
+            "study_num_label_positives": self.study_num_label_positives,
+            "study_num_label_negatives": self.study_num_label_negatives,
+            "study_num_label_ambiguous": self.study_num_label_ambiguous,
+            "study_num_label_parse_failures": self.study_num_label_parse_failures,
+            "study_num_label_inherited_positives": self.study_num_label_inherited_positives,
+            "study_num_label_inherited_negatives": self.study_num_label_inherited_negatives,
+            "study_num_label_inherit_skipped": self.study_num_label_inherit_skipped,
         }
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_index.py</span>
    <span class="card-oneliner">Unified chunking (no atomic branch, no `chunk_max_size`); `bm25_rank_doc_chunks`.</span>
    <span class="card-badge">Diff</span>
  </summary>

One `score_all` call per example; each requested document's chunks come back sorted by descending score, zeros included. Exact delta vs `/source/activation/dataset/dataset_index.py`:

````diff-python
diff --git a/activation/source/activation/dataset/dataset_index.py b/activation/activation/dataset/dataset_index.py
index 524d947..deed55b 100644
--- a/activation/source/activation/dataset/dataset_index.py
+++ b/activation/activation/dataset/dataset_index.py
@@ -28,7 +28,6 @@ class DatasetIndex:
         self.dataset_id = loaded_dataset.dataset_id
         self.chunk_size_chars = harness.harness_config.doc_chunk_size_chars
         self.chunk_overlap_chars = harness.harness_config.doc_chunk_overlap_chars
-        self.chunk_max_size = harness.harness_config.doc_chunk_max_atomic_size
         self.embedding_model_name = harness.harness_config.doc_embedding_model_name
         self.batch_size = harness.harness_config.doc_embedding_batch_size
         self.embedding_input_limit = harness.harness_config.doc_embedding_input_limit_chars
@@ -48,10 +47,9 @@ class DatasetIndex:
         print(f"{self.dataset_id} - Chunking.")
         chunks_list = [] # pre sort.
         for document in self.loaded_dataset.documents.values():
-            chunk_size = self.chunk_max_size if document.atomic else self.chunk_size_chars
-            step_size = chunk_size - self.chunk_overlap_chars
+            step_size = self.chunk_size_chars - self.chunk_overlap_chars
             pieces = [
-                (document.text[start:start + chunk_size], start)
+                (document.text[start:start + self.chunk_size_chars], start)
                 for start in range(0, len(document.text), step_size)
             ]
             pieces = [x for x in pieces if x[0]] # filter our empty strings
@@ -134,8 +132,7 @@ class DatasetIndex:
         for excluded in all_excluded_sets:
             for doc_id in excluded:
                 document = self.loaded_dataset.documents[doc_id]
-                chunk_size = self.chunk_max_size if document.atomic else self.chunk_size_chars
-                step_size = chunk_size - self.chunk_overlap_chars
+                step_size = self.chunk_size_chars - self.chunk_overlap_chars
                 num_chunks = -(-len(document.text) // step_size)  # ceil division
                 max_chunks = max(max_chunks, num_chunks)
         return 5 + max_chunks if max_chunks else 0
@@ -287,6 +284,29 @@ class DatasetIndex:
         ]
 
 
+    def bm25_rank_doc_chunks(
+        self,
+        query: str,
+        doc_ids: list[str],
+    ) -> dict[str, list[tuple[DatasetDocumentChunk, float]]]:
+        """
+        Score every indexed chunk against one query, then return each requested document's own
+        chunks sorted by descending bm25 score. A document whose chunks all score zero still
+        returns them (with zero scores); the caller decides what to do.
+        """
+        assert self.bm25_index is not None, "bm25_rank_doc_chunks needs the bm25 index."
+        scores = self.bm25_index.score_all(query)
+        ranked: dict[str, list[tuple[DatasetDocumentChunk, float]]] = {}
+        for doc_id in doc_ids:
+            document = self.loaded_dataset.documents[doc_id]
+            scored = [
+                (chunk, float(scores[self.chunk_id_indexes[chunk_id]]))
+                for chunk_id, chunk in document.chunks.items()
+            ]
+            scored.sort(key=lambda pair: pair[1], reverse=True)
+            ranked[doc_id] = scored
+        return ranked
+
     def get_top_documents(self, chunks: list[DatasetDocumentChunk]) -> list[DatasetDocument]:
         seen = set()
         docs = list()
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/common/bm25.py</span>
    <span class="card-oneliner">`score_all(query)`: per-chunk bm25 score vector.</span>
    <span class="card-badge">Diff</span>
  </summary>

Same stop-word settings as `search`; doc-id agnostic. Exact delta vs `/source/activation/common/bm25.py`:

````diff-python
diff --git a/activation/source/activation/common/bm25.py b/activation/activation/common/bm25.py
index 6b23ae0..324873b 100644
--- a/activation/source/activation/common/bm25.py
+++ b/activation/activation/common/bm25.py
@@ -24,6 +24,11 @@ class BM25Index:
         )
         self.build_time = time.time() - build_start_time
 
+    def score_all(self, query: str) -> np.ndarray:
+        """bm25 score of one query against every indexed document, as a float32 vector."""
+        query_tokens = bm25s.tokenize([query], stopwords="en", return_ids=False, show_progress=False)[0]
+        return self.retriever.get_scores(query_tokens).astype(np.float32)
+
     def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
         """
         faiss-shaped search: (scores [Q, k] float32, indices [Q, k] int64), sorted by decreasing
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/loaders/bright.py</span>
    <span class="card-oneliner">Drop `atomic=True`.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/loaders/bright.py`:

````diff-python
diff --git a/activation/source/activation/dataset/loaders/bright.py b/activation/activation/dataset/loaders/bright.py
index c8d3981..1300b6c 100644
--- a/activation/source/activation/dataset/loaders/bright.py
+++ b/activation/activation/dataset/loaders/bright.py
@@ -77,7 +77,6 @@ class BrightDataset:
                 doc_id=doc_id,
                 dataset_id=dataset_id,
                 text=str(row["content"]),
-                atomic=True, # BRIGHT passages are already retrieval units.
             )
         examples: dict[str, LabeledRetrievalQAExample] = {}
         for row in selected:
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/loaders/msmarco.py</span>
    <span class="card-oneliner">Drop `atomic=True` (two sites).</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/loaders/msmarco.py`:

````diff-python
diff --git a/activation/source/activation/dataset/loaders/msmarco.py b/activation/activation/dataset/loaders/msmarco.py
index 9f57102..2102acb 100644
--- a/activation/source/activation/dataset/loaders/msmarco.py
+++ b/activation/activation/dataset/loaders/msmarco.py
@@ -74,7 +74,6 @@ class MsMarcoDataset:
                         doc_id=doc_id,
                         dataset_id=dataset_id,
                         text=text,
-                        atomic=True,
                     )
                     documents[doc_id] = doc
                     documents_dedup[text] = doc
@@ -116,7 +115,6 @@ class MsMarcoDataset:
                         doc_id=f"{prefix}:{index}",
                         dataset_id=dataset_id,
                         text=text,
-                        atomic=True,
                     )
                     documents[doc.doc_id] = doc
                     documents_dedup[text] = doc
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/loaders/nq.py</span>
    <span class="card-oneliner">Drop `atomic=True` (two sites).</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/loaders/nq.py`:

````diff-python
diff --git a/activation/source/activation/dataset/loaders/nq.py b/activation/activation/dataset/loaders/nq.py
index 541c976..edf1516 100644
--- a/activation/source/activation/dataset/loaders/nq.py
+++ b/activation/activation/dataset/loaders/nq.py
@@ -63,7 +63,6 @@ class NqDataset:
                     doc_id=f"nq:{split}:{ordinal}",
                     dataset_id=dataset_id,
                     text=text,
-                    atomic=True, # NQ passages are already retrieval units.
                 )
                 documents[doc.doc_id] = doc
                 documents_dedup[text] = doc
@@ -89,7 +88,6 @@ class NqDataset:
                     doc_id=f"nq:{split}:{ordinal}",
                     dataset_id=dataset_id,
                     text=text,
-                    atomic=True,
                 )
                 documents[doc.doc_id] = doc
                 documents_dedup[text] = doc
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/loaders/scifact.py</span>
    <span class="card-oneliner">Drop `atomic=True`.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/loaders/scifact.py`:

````diff-python
diff --git a/activation/source/activation/dataset/loaders/scifact.py b/activation/activation/dataset/loaders/scifact.py
index e5a5369..ce85f21 100644
--- a/activation/source/activation/dataset/loaders/scifact.py
+++ b/activation/activation/dataset/loaders/scifact.py
@@ -80,7 +80,6 @@ class SciFactDataset:
                 doc_id=doc_id,
                 dataset_id=dataset_id,
                 text=f"{title}\n\n{text}" if title else text,
-                atomic=True, # One abstract is one retrieval unit.
             )
         examples: dict[str, LabeledRetrievalQAExample] = {}
         for query_id, gold_ids in selected:
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/loaders/sciq.py</span>
    <span class="card-oneliner">Drop `atomic=True` (two sites).</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/loaders/sciq.py`:

````diff-python
diff --git a/activation/source/activation/dataset/loaders/sciq.py b/activation/activation/dataset/loaders/sciq.py
index 4e671ba..e870fa8 100644
--- a/activation/source/activation/dataset/loaders/sciq.py
+++ b/activation/activation/dataset/loaders/sciq.py
@@ -60,7 +60,6 @@ class SciQDataset:
                     doc_id=f"sciq:{split}:{ordinal}",
                     dataset_id=dataset_id,
                     text=text,
-                    atomic=True, # One support passage is one retrieval unit.
                 )
                 documents[doc.doc_id] = doc
                 documents_dedup[text] = doc
@@ -87,7 +86,6 @@ class SciQDataset:
                     doc_id=f"sciq:{split}:{ordinal}",
                     dataset_id=dataset_id,
                     text=text,
-                    atomic=True,
                 )
                 documents[doc.doc_id] = doc
                 documents_dedup[text] = doc
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/runtime_config.py</span>
    <span class="card-oneliner">Add the three label knobs; remove `dataset_study_num_samples`.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/harness/runtime_config.py`:

````diff-python
diff --git a/activation/source/activation/harness/runtime_config.py b/activation/activation/harness/runtime_config.py
index 6145ce1..2613a4d 100644
--- a/activation/source/activation/harness/runtime_config.py
+++ b/activation/activation/harness/runtime_config.py
@@ -27,10 +27,12 @@ class HarnessRuntimeConfig:
     # Dataset Study
     dataset_study_batch_size: int = 16
     dataset_study_model_name: str|None = None
-    dataset_study_num_samples: int|None = None
     dataset_study_chunk_input_limit: int|None = None # Limit for fast testing.
     dataset_study_chat_kwargs: dict|None = None
     dataset_study_seed: int = 0 # Seeds chunk sampling and generation sampling.
+    dataset_study_label_top_k: int = 10 # bm25 candidates per question.
+    dataset_study_label_pool_max_chars: int = 32768 # Char budget of the snippet pool shown to the labeler.
+    dataset_study_label_batch_size: int = 16 # Label prompts carry the whole pool, so they get their own batch size.
 
 @dataclass
 class HarnessStats:
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/tests/test_basic_dataset_loading.py</span>
    <span class="card-oneliner">Unified-chunking assertions and the in-doc scoring check on the bm25 branch.</span>
    <span class="card-badge">Diff</span>
  </summary>

Passed on CPU for both variants, basic and public: `4 passed in 308.33s`. Exact delta vs `/source/activation/tests/test_basic_dataset_loading.py`:

````diff-python
diff --git a/activation/source/activation/tests/test_basic_dataset_loading.py b/activation/activation/tests/test_basic_dataset_loading.py
index 9e54174..35d3f39 100644
--- a/activation/source/activation/tests/test_basic_dataset_loading.py
+++ b/activation/activation/tests/test_basic_dataset_loading.py
@@ -10,7 +10,7 @@ from activation.dataset.loaders import (
 )
 from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig, TARGET_DEVICE
 
-# Last element is longer than the max atomic size.
+# Last element spans several chunks at the test chunk size.
 # Second-to-last is shorter than the chunk size.
 TEST_DOCUMENTS = [
     "Gravity pulls masses together.",
@@ -23,7 +23,7 @@ TEST_DOCUMENTS = [
     "Whales are mammals.",
     "Light bends through glass.",
     "Oxygen.", # Short
-    "Atomic passage larger than sixteen characters.",
+    "Longer passage spanning several chunks.",
 ]
 
 EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
@@ -67,7 +67,6 @@ def _run_basic_dataset_loading(retrieval: str):
     harness_config = HarnessRuntimeConfig(
         doc_chunk_size_chars=8,
         doc_chunk_overlap_chars=2,
-        doc_chunk_max_atomic_size=16,
         **_embedding_model_config(retrieval),
     )
     harness = HarnessRuntime(harness_config)
@@ -79,7 +78,6 @@ def _run_basic_dataset_loading(retrieval: str):
                 doc_id=str(i),
                 dataset_id=dataset_id,
                 text=text,
-                atomic = (i == len(TEST_DOCUMENTS) - 1), # For testing.
             )
             for i, text in enumerate(TEST_DOCUMENTS)}
     )
@@ -87,16 +85,14 @@ def _run_basic_dataset_loading(retrieval: str):
     assert dataset_id in harness.dataset_manager.loaded_datasets
     _build_indexes(harness, retrieval)
     doc_index = harness.dataset_manager.dataset_indexes[dataset_id]
-    # Some chunking assertions.
+    # Some chunking assertions: every document chunks at size 8 with overlap 2 (step 6).
     num_docs = len(TEST_DOCUMENTS)
-    assert doc_index.chunks[f"{num_docs-1}:0"].chunk_text == TEST_DOCUMENTS[num_docs-1][:16]
-    assert doc_index.chunks[f"{num_docs-1}:1"].chunk_text == TEST_DOCUMENTS[num_docs-1][14:30]
+    long_text = TEST_DOCUMENTS[num_docs-1]
+    assert doc_index.chunks[f"{num_docs-1}:0"].chunk_text == long_text[:8]
+    assert doc_index.chunks[f"{num_docs-1}:1"].chunk_text == long_text[6:14]
+    assert len(dataset.documents[str(num_docs-1)].chunks) == -(-(len(long_text) - 2) // 6)
     assert doc_index.chunks[f"{num_docs-2}:0"].chunk_text == TEST_DOCUMENTS[num_docs-2]
-    assert all(
-        len(chunk.chunk_text) <= 8
-        for chunk in doc_index.chunks.values()
-        if not dataset.documents[chunk.doc_id].atomic
-    )
+    assert all(len(chunk.chunk_text) <= 8 for chunk in doc_index.chunks.values())
     if retrieval == "dense":
         assert doc_index.dense_faiss_index.ntotal == len(doc_index.chunks)
     else:
@@ -118,6 +114,14 @@ def _run_basic_dataset_loading(retrieval: str):
     if retrieval == "bm25":
         # A query sharing no term with the corpus returns nothing rather than arbitrary chunks.
         assert doc_index.bm25_query_many_frozen(["zzzz unmatched"], top_k) == [[]]
+        # In-doc scoring: the second chunk of the multi-chunk document ranks first for its own text.
+        long_doc_id = str(num_docs-1)
+        second_chunk = doc_index.chunks[f"{long_doc_id}:1"]
+        ranked = doc_index.bm25_rank_doc_chunks(second_chunk.chunk_text, [long_doc_id])[long_doc_id]
+        assert ranked[0][0].chunk_id == second_chunk.chunk_id and ranked[0][1] > 0
+        assert len(ranked) == len(dataset.documents[long_doc_id].chunks)
+        unmatched = doc_index.bm25_rank_doc_chunks("zzzz unmatched", [long_doc_id])[long_doc_id]
+        assert all(score == 0.0 for _, score in unmatched)
 
 
 def _elide(text: str, max_chars: int) -> str:
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/tests/test_basic_dataset_study.py</span>
    <span class="card-oneliner">Explicit sample count, two labeling passes (synthetic, then native), doc-id invariance and inheritance asserts.</span>
    <span class="card-badge">Diff</span>
  </summary>

GPU-marked; not run this session. Exact delta vs `/source/activation/tests/test_basic_dataset_study.py`:

````diff-python
diff --git a/activation/source/activation/tests/test_basic_dataset_study.py b/activation/activation/tests/test_basic_dataset_study.py
index 8078e9e..3b78f37 100644
--- a/activation/source/activation/tests/test_basic_dataset_study.py
+++ b/activation/activation/tests/test_basic_dataset_study.py
@@ -2,7 +2,6 @@ import json
 
 import pytest
 
-from activation.dataset import DatasetStudyGenerator
 from activation.dataset.dataset import DataOrigin
 from activation.dataset.loaders import BrightDataset
 from activation.harness import (
@@ -17,6 +16,7 @@ pytestmark = pytest.mark.gpu
 
 STUDY_NUM_QUESTIONS = 16
 STUDY_BATCH_SIZE = 8
+STUDY_LABEL_BATCH_SIZE = 8
 if SUPPORTS_FP4:
     STUDY_MODEL_NAME = "unsloth/Qwen3.8-27B-NVFP4"
 else:
@@ -47,8 +47,8 @@ def test_basic_dataset_study():
             ),
         },
         dataset_study_model_name=STUDY_MODEL_NAME,
-        dataset_study_num_samples=STUDY_NUM_QUESTIONS,
         dataset_study_batch_size=STUDY_BATCH_SIZE,
+        dataset_study_label_batch_size=STUDY_LABEL_BATCH_SIZE,
         dataset_study_chat_kwargs={
             # Qwen3.8 instruct-mode sampling recommendation.
             "sampling_params": {
@@ -67,10 +67,18 @@ def test_basic_dataset_study():
         max_corpus_documents=100,
     )
     harness.dataset_manager.build_bm25_indexes()
-    # Free the embedding model before the engine claims its memory fraction.
+    native = [
+        example
+        for example in dataset.labeled_retrieval_examples.values()
+        if example.origin == DataOrigin.NATIVE
+    ]
+    native_doc_ids_before = {
+        example.example_id: (list(example.positive_doc_ids or []), list(example.hard_negative_doc_ids or []))
+        for example in native
+    }
     study_model = harness.loaded_models[STUDY_MODEL_NAME]
     try:
-        harness.dataset_manager.synthesize_study_examples(dataset.dataset_id)
+        harness.dataset_manager.synthesize_study_examples(dataset.dataset_id, STUDY_NUM_QUESTIONS)
         synthetic = [
             example
             for example in dataset.labeled_retrieval_examples.values()
@@ -86,6 +94,40 @@ def test_basic_dataset_study():
             print(f"\n[{example.example_id}] (doc {example.positive_doc_ids[0]})")
             print(f"  Q: {example.query}")
             print(f"  A: {example.gold_answers[0]}")
+
+        # Labeling pass 1: the synthetic questions.
+        labeled = harness.dataset_manager.label_study_examples(
+            dataset.dataset_id, STUDY_NUM_QUESTIONS, synthetic_only=True,
+        )
+        assert labeled and all(example.origin == DataOrigin.SYNTHETIC for example in labeled)
+        for example in labeled:
+            assert example.oracle_labeled
+            assert not set(example.positive_chunk_ids) & set(example.hard_negative_chunk_ids or [])
+        assert any(example.hard_negative_chunk_ids for example in labeled)
+        print(f"\n=== {len(labeled)} synthetic questions labeled ===")
+        for example in labeled[:4]:
+            print(f"\n[{example.example_id}]")
+            print(f"  positives: {example.positive_chunk_ids}")
+            print(f"  hard negatives: {example.hard_negative_chunk_ids}")
+
+        # Labeling pass 2: the native examples. Doc-level ids are never touched; chunk-level
+        # positives arrive from the oracle or by inheritance from the known positive documents.
+        stats_before = dataset.stats.study_num_label_positives + dataset.stats.study_num_label_inherited_positives
+        labeled_native = harness.dataset_manager.label_study_examples(
+            dataset.dataset_id, len(native), synthetic_only=False,
+        )
+        assert labeled_native and all(example.origin == DataOrigin.NATIVE for example in labeled_native)
+        for example in labeled_native:
+            assert example.oracle_labeled
+            assert (list(example.positive_doc_ids or []), list(example.hard_negative_doc_ids or [])) == native_doc_ids_before[example.example_id]
+            assert not set(example.positive_chunk_ids or []) & set(example.hard_negative_chunk_ids or [])
+        stats_after = dataset.stats.study_num_label_positives + dataset.stats.study_num_label_inherited_positives
+        assert stats_after > stats_before
+        print(f"\n=== {len(labeled_native)} native examples labeled ===")
+        for example in labeled_native[:4]:
+            print(f"\n[{example.example_id}]")
+            print(f"  positives: {example.positive_chunk_ids}")
+            print(f"  hard negatives: {example.hard_negative_chunk_ids}")
         print("\n=== dataset stats ===")
         print(json.dumps(dataset.stats.summarize(), indent=2))
     finally:
````

</details>

