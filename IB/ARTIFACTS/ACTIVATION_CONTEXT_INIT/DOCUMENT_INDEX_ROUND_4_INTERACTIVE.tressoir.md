# Document index — unskewed loaders and exclusions round

This round lands the anti-skew direction from our chat: your embedding-input truncator and `filter_fn` finished and wired, per-query `excluded_doc_ids` respected at retrieval time, BRIGHT simplified to `max_examples: int | None` with the distractor machinery removed, and a review of your partial application with each finding fixed in the staged files. Deltas below are exact against your current source.

## Review of your partial application

Four findings, all corrected in the staged files (hunks in the exact delta below):

- **Every index build crashed.** Both embedding paths called `safe_truncate_embedding_chunk(text)` without the limit argument — a `TypeError` on the first build. Both call sites now pass `self.embedding_input_limit` (your `None`-tolerant signature makes the call unconditional, which is nice).
- **Exclusions were silently never applied.** In the result loop, `if doc.doc_id in excluded_doc_ids:` tested membership against the outer list-of-lists parameter instead of the per-query `excluded` set, so nothing ever matched and every excluded document sailed through. It now tests `excluded`.
- **`_build_excluded_sets` couldn't handle the default.** With `excluded_doc_ids=None` it computed `len(None)`; with an empty list it built a zero-length placeholder that `zip` then used to truncate all results. It also returned the raw input lists rather than the sets it had just built. It now takes `num_queries`, defaults correctly, and returns the sets. Your flat `+10` over-fetch stays as designed — worth knowing it can undershoot `top_k` if a query's excluded documents own more than ten of the top-scoring chunks.
- **The `chunk_ids` cache wasn't used.** `query_many_frozen` still recomputed `list(self.chunks.keys())`; it now reads `self.chunk_ids`.

## Changes per your direction

- **`excluded_doc_ids`** replaces the chunk-level field — doc-level is indeed the natural grain for BRIGHT, whose `excluded_ids` name documents. The BRIGHT loader populates it (dropping the `'N/A'` sentinel), `query_many_frozen(queries, top_k, excluded_doc_ids=…)` filters every chunk of an excluded document, and the test passes each inspected example's exclusions and proves the mechanism by excluding a probe's own document.
- **BRIGHT accepts `max_examples: int | None` and the distractor logic is gone.** The corpus is exactly the gold passages of the selected examples; with `None`, every labeled example loads (biology: 103 examples, 372 gold passages). The loader shrank by ~15 lines.
- **Truncator config** stays as you wrote it (`doc_embedding_input_limit_chars`, `None`-passthrough inside `safe_truncate_embedding_chunk`); the test sets 512 on CPU, `None` on GPU.

## Corpus-composition notes to hold knowingly

- **Gold-only corpora**: with distractors removed, the only "background" documents for one query are other examples' golds. At `n=20` biology that's a 104-doc corpus where every document answers *some* query — fine for plumbing sanity, but retrieval difficulty is not benchmark-like. `max_examples=None` restores the full labeled set (and its golds), though still not the domain's full 57k-document corpus.
- **Real exclusions are rare**: all 103 biology examples carry `excluded_ids=['N/A']`, so the doc-level exclusion currently only fires in the synthetic test probe. The mechanism matters more once metrics rounds use domains/examples that do carry exclusions.
- **SciFact still has the seeded-distractor logic** you removed from BRIGHT — see the decision below.

## Exact delta — staged files vs your source

`activation/dataset/document_index.py`

```diff-python
@@
         chunk_texts = [
-            safe_truncate_embedding_chunk(text) for text, _ in self.chunks.values()
+            safe_truncate_embedding_chunk(text, self.embedding_input_limit)
+            for text, _ in self.chunks.values()
         ]
@@
-    def _build_excluded_sets(self, excluded_doc_ids: list[list[str] | None] | None = None) -> tuple[list[set], int]:
+    def _build_excluded_sets(self, excluded_doc_ids: list[list[str] | None] | None, num_queries: int) -> tuple[list[set], int]:
         """
         Return (excluded sets, additional fetch count)
         """
         if not excluded_doc_ids:
-            excluded_doc_ids = [[] for _ in range(len(excluded_doc_ids))]
+            excluded_doc_ids = [[] for _ in range(num_queries)]
         excluded_sets = [set(excluded or []) for excluded in excluded_doc_ids]
         if any(len(excluded) > 0 for excluded in excluded_sets):
-            return (excluded_doc_ids, 10)
-        return (excluded_doc_ids, 0)
+            return (excluded_sets, 10)
+        return (excluded_sets, 0)
@@
         stats = self.loaded_dataset.stats
         embedding_model = self.harness.loaded_models[self.embedding_model_name]
-        chunk_ids = list(self.chunks.keys())
         stats.query_batch_sizes.append(len(queries))
@@
-            batch_texts = [safe_truncate_embedding_chunk(query) for _, query in batch]
+            batch_texts = [
+                safe_truncate_embedding_chunk(query, self.embedding_input_limit)
+                for _, query in batch
+            ]
@@
-        all_excluded_sets, extra_fetches = self._build_excluded_sets(excluded_doc_ids)
+        all_excluded_sets, extra_fetches = self._build_excluded_sets(excluded_doc_ids, len(queries))
         _all_scores, all_indices = self.faiss_index.search(
             query_embeddings,
-            k=min(top_k + extra_fetches, len(chunk_ids)), # +10 handles most cases of exclusion.
+            k=min(top_k + extra_fetches, len(self.chunk_ids)), # +10 handles most cases of exclusion.
         )
@@
-                chunk_id = chunk_ids[index]
+                chunk_id = self.chunk_ids[index]
                 _, doc = self.chunks[chunk_id]
-                if doc.doc_id in excluded_doc_ids:
+                if doc.doc_id in excluded:
                     continue
```

`activation/dataset/loaders/bright.py`

```diff-python
@@
 import ast
-import random
 import time
@@
         """
-        Take up to max_examples labeled examples, keep every gold passage they
-        reference (overfetching documents rather than dropping examples), and add
-        up to max_examples seed-sampled distractor passages from the same domain.
+        Take up to max_examples labeled examples (all of them when None) and keep
+        every gold passage they reference. Other examples' golds act as the
+        natural distractors of a subsampled corpus.
         """
@@
-        doc_rows = load_dataset("xlangai/BRIGHT", "documents", split=domain)
-        doc_ids = [str(doc_id) for doc_id in doc_rows["id"]]
-        gold_positions = [p for p, doc_id in enumerate(doc_ids) if doc_id in wanted_doc_ids]
-        candidate_positions = [p for p, doc_id in enumerate(doc_ids) if doc_id not in wanted_doc_ids]
-        # Seeded uniform sample avoids the topical clustering of a prefix.
-        sampled_distractors = random.Random(0).sample(
-            candidate_positions,
-            min(max_examples, len(candidate_positions)),
-        )
         documents: dict[str, DatasetDocument] = {}
-        for position in sorted(gold_positions + sampled_distractors):
-            row = doc_rows[position]
+        for row in load_dataset("xlangai/BRIGHT", "documents", split=domain):
             doc_id = str(row["id"])
+            if doc_id not in wanted_doc_ids:
+                continue
             documents[doc_id] = DatasetDocument(
@@
                 positive_doc_ids=gold_ids,
-                excluded_chunk_ids=excluded_ids or None,
+                excluded_doc_ids=excluded_ids or None,
             )
```

`activation/tests/test_basic_dataset_loading.py`

```diff-python
@@
-        # Exclusions are respected, both as an exact chunk id and as a whole doc id.
+        # Exclusions are respected: excluding the probe's own document removes it.
         excluded_results = doc_index.query_many_frozen(
-            [probe_text, probe_text],
+            [probe_text],
             top_k=5,
-            excluded_chunk_ids=[[probe_chunk_id], [probe_doc.doc_id]],
+            excluded_doc_ids=[[probe_doc.doc_id]],
         )
-        assert all(chunk_id != probe_chunk_id for chunk_id, _ in excluded_results[0])
-        assert all(doc.doc_id != probe_doc.doc_id for _, doc in excluded_results[1])
+        assert all(doc.doc_id != probe_doc.doc_id for _, doc in excluded_results[0])
         # Manual-inspection printout: 2 labeled queries with their top 5 chunks.
         inspected = list(dataset.labeled_retrieval_examples.values())[:2]
         all_results = doc_index.query_many_frozen(
             [example.query for example in inspected],
             top_k=5,
-            excluded_chunk_ids=[example.excluded_chunk_ids for example in inspected],
+            excluded_doc_ids=[example.excluded_doc_ids for example in inspected],
         )
```

## Validation

- `test_basic_dataset_loading`: passed (13s) after the fixes.
- BRIGHT loads at `n=20` (104 docs / 20 examples) and `n=None` (372 docs / 103 examples) without embedding.
- `test_public_dataset_loading`: **passed in 15m39s on CPU** with all five datasets at `n=20`, 512-char embedding inputs, and per-example exclusions passed through the inspection queries. Corpora: bright/bio 104 docs, bright/econ 97, nq 20, ms_marco 175, scifact 35, sciq 20 — 18 embedding batches total, ~25–87s each. Printed retrieval stays sensible on the gold-only corpora: e.g. bright/biology query 0 ranks its three gold passages 1–3 with topically-adjacent golds of other examples (phototropism) as natural distractors below; scifact claim 0's gold ranks 2 of 35.

Run `activation/tests/test_basic_dataset_loading.py` yourself after applying:

```bash
uv run pytest activation/tests/test_basic_dataset_loading.py -s
```

<article class="decision" data-tressoir-decision data-decision-state="unresolved"
  aria-labelledby="round4-scifact-distractors">
  <header class="decision-header">
    <div>
      <h3 class="decision-title" id="round4-scifact-distractors">Should SciFact mirror BRIGHT's distractor removal?</h3>
      <p class="decision-context">SciFact still carries the seeded-sample distractor logic; BRIGHT no longer does. Claims have ~1 gold abstract each, so a golds-only SciFact corpus at n=20 would hold only ~15–20 abstracts.</p>
    </div>
    <span class="decision-state" data-decision-indicator role="status" aria-live="polite">Unresolved</span>
  </header>
  <fieldset class="decision-options">
    <legend class="visually-hidden">Decision answers</legend>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="round4.scifact_distractors.mirror">
      <span><strong>Mirror BRIGHT — golds only, max_examples: int | None</strong><small>One consistent policy across separated-corpus loaders; smallest code.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="round4.scifact_distractors.keep">
      <span><strong>Keep SciFact's seeded distractors</strong><small>Its 1-gold-per-claim shape makes golds-only corpora very small; distractors keep retrieval non-trivial.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="round4.scifact_distractors.defer">
      <span><strong>Defer to the metrics round</strong><small>Decide when retrieval quality is actually measured.</small></span>
    </label>
  </fieldset>
  <div class="field decision-feedback">
    <label for="round4-scifact-response">Free Response</label>
    <textarea id="round4-scifact-response" rows="2" data-tressoir-input="round4.scifact_distractors.feedback"
      data-tressoir-autogrow="2:6" placeholder="A different corpus policy, a distractor count, anything else…"></textarea>
  </div>
</article>

## Deferred

- GPU runs and retrieval-quality metrics (recall@k against golds) — the natural next round, where `excluded_doc_ids` starts doing real work.
- Exact exclusion accounting in place of the flat +10 over-fetch, only if undershooting `top_k` ever matters in practice.
- NQ against a full BEIR-style corpus with qrels.
