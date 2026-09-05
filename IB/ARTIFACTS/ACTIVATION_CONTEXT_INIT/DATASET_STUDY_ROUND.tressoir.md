# Dataset study round

Batched report for the post-prototype round: the `@AI` integration fixes, the dataset-legality
batch (`max_corpus_documents`, chunker tail guard, SciFact qrels guard), the new engine chat
surface, and the synthetic study-question generator — GPU-validated end to end by
`test_basic_dataset_study` on an RTX PRO 6000 (NVFP4) node. Everything below is staged in
`IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/activation/` in source shape, and the **Diffs to adapt**
section carries the exact per-file delta against the current `/source` tree.

## Headline validation

`activation/tests/test_basic_dataset_study.py` — BRIGHT biology, 20 examples, 100-document
corpus budget, 16 sampled chunks, `unsloth/Qwen3.8-27B-NVFP4` with the benchmark-winning engine
formula:

- **`1 passed in 805.05s (13:25)`** on `ac-study-rtx` (g7e.2xlarge, ap-northeast-1). ~9.5 min of
  that is one-time engine setup (115 s weight load, 51 s torch.compile, 335 s warmup profiling).
- 16 chunks → **39 synthetic questions, 0 structured-output parse failures**.
- Generation throughput: 16 requests in 65.6 s total; avg 780 prompt / 308 output tokens.
- An L40S run of the same test exercises the other `SUPPORTS_FP4` branch
  (`Qwen/Qwen3.8-27B-FP8`); result recorded in the cross-check section below.
- **Final revalidation** after both adaptation review rounds (one-question schema,
  `synthesize_study_examples` path, `HarnessStats`): **`1 passed in 291.89s (4:51)`** on the warm
  RTX node — 16/16 chunks → exactly 16 questions, interval prints and per-model harness stats
  visible (engine load 255.6s dominates; embedding model 4.7s load / 0.2s free).

Sample output (verbatim from the run):

> **Q:** Based on the text, what specific physiological mechanism allows newborn mammals to
> deviate from the standard rule that most peptides longer than four amino acids are not
> absorbed, and what is the functional biological consequence of this deviation?
>
> **A:** Newborn mammals can absorb intact proteins at the small intestine. This enables passive
> immunity by allowing the transfer of immunoglobulins from the mother to the newborn via milk.

Questions are multi-hop, grounded in the sampled chunk, and carry faithful answers; each example
records `positive_doc_ids`, `positive_chunk_ids`, `origin=SYNTHETIC`, `split=TRAIN`.

## Diffs to adapt

Your second adaptation pass is merged — everything applied verbatim, including your progress
prints around model/engine placement. The two cards below are the **complete residual delta**
against the current `/source` tree. Embedding-layer timings are recorded into the existing
load/free lists (grouped per model by `summarize()`), keeping `HarnessStats` schema-stable;
split them into dedicated fields later if you want the copy cost separated from full-model
moves.

The probe files (`tests/probe_speed_ab.py`, `tests/probe_family_ab.py`, `tests/test_q4_engine.py`)
are **IB-only** by decision: they stay in the staged handoff tree for reproducibility but are
not part of the main-code delta and carry no cards here.

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_study.py</span>
    <span class="card-oneliner">Clears the leftover interval-print marker (the logic itself is already applied).</span>
    <span class="card-badge">Diff</span>
  </summary>

Clears the leftover interval-print marker (the logic itself is already applied). Exact delta vs `/source/activation/dataset/dataset_study.py`:

````diff-python
--- a/activation/dataset/dataset_study.py
+++ b/activation/dataset/dataset_study.py
@@ -178,7 +178,6 @@ class DatasetStudyGenerator:
                 )
                 self.loaded_dataset.labeled_retrieval_examples[example.example_id] = example
                 example_count += 1
-            # @AI: Interval printing logic here.
             if reporting_interval <= 0:
                 print(
                     f"{self.dataset_id} - Study: {batch_start + len(batch)}/{len(study_samples)} chunks, "
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/loaded_model.py</span>
    <span class="card-oneliner">Embedding-layer timing with your standard prints; records into the existing HarnessStats lists.</span>
    <span class="card-badge">Diff</span>
  </summary>

Embedding-layer timing with your standard prints; records into the existing `HarnessStats` lists. Exact delta vs `/source/activation/harness/loaded_model.py`:

````diff-python
--- a/activation/harness/loaded_model.py
+++ b/activation/harness/loaded_model.py
@@ -138,27 +138,33 @@ class LoadedModel:
             return
         device_change_start_time = time.time()
         harness_stats = self.harness.harness_stats
-
         if device != FREE_DEVICE and self.current_embedding_layer_device != FREE_DEVICE:
             # Regular device change (cpu -> gpu or gpu -> cpu).
+            print(f"{self.model_config.model_name} - Moving embedding layer.")
             self.embedding_layer.to(device)
             self.current_embedding_layer_device = device
+            elapsed = time.time() - device_change_start_time
+            print(f"{self.model_config.model_name} - Moved embedding layer in {elapsed:.2f}s")
             harness_stats.model_loading_times.append(
-                (self.model_config.model_name, time.time() - device_change_start_time)
+                (self.model_config.model_name, elapsed)
             )
             return
         if device == FREE_DEVICE and self.current_embedding_layer_device != FREE_DEVICE:
             # Free.
+            print(f"{self.model_config.model_name} - Freeing embedding layer.")
             self.embedding_layer = None
             gc.collect()
             if self.current_embedding_layer_device.startswith("cuda"):
                 torch.cuda.empty_cache()
             self.current_embedding_layer_device = device
+            elapsed = time.time() - device_change_start_time
+            print(f"{self.model_config.model_name} - Freed embedding layer in {elapsed:.2f}s")
             harness_stats.model_freeing_times.append(
-                (self.model_config.model_name, time.time() - device_change_start_time)
+                (self.model_config.model_name, elapsed)
             )
             return
         # Bring over from scratch. Need to temporarily use the whole model.
+        print(f"{self.model_config.model_name} - Copying embedding layer.")
         starting_model_device = self.current_model_device
         if starting_model_device == FREE_DEVICE:
             self.model_to_device(SOURCE_DEVICE) # Use cpu for efficiency.
@@ -168,8 +174,10 @@ class LoadedModel:
         self.current_embedding_layer_device = device
         if starting_model_device == FREE_DEVICE:
             self.model_to_device(FREE_DEVICE) # Restore starting model device.
+        elapsed = time.time() - device_change_start_time
+        print(f"{self.model_config.model_name} - Copied embedding layer in {elapsed:.2f}s")
         harness_stats.model_loading_times.append(
-            (self.model_config.model_name, time.time() - device_change_start_time)
+            (self.model_config.model_name, elapsed)
         )
 
     def engine_to_device(self, device: str):
````

</details>

## Family micro-benchmark (RTX PRO 6000, non-reasoning)

FP8 vs NVFP4 across three families via `tests/probe_family_ab.py` (IB-only probe): speed by the
established methodology (batch 8, greedy, ~6000-char prefill, 128-token decode subtraction),
plus seeded study QA on four identical BRIGHT-biology chunks (`dataset_study_seed=0`, pipeline
sampling held constant).

| arm | config | prefill tok/s | decode tok/s | study 4 chunks |
|---|---|---:|---:|---:|
| qwen3.8-fp8 (`Qwen/Qwen3.8-27B-FP8`) | formula | 10,113 | 736.0 | 2.8s |
| qwen3.8-fp4 (`unsloth/Qwen3.8-27B-NVFP4`) | formula | 12,981 | 933.4 | 2.1s |
| nemotron-fp8 (`RedHatAI/...Lightning-30B-A3B-FP8`) | **base** | 43,621 | 1,207.9 | 1.6s |
| nemotron-fp4 (`nvidia/...Lightning-30B-A3B-NVFP4`) | formula | 30,732 | 1,876.7 | 3.8s |
| qwen3.6moe-fp8 (`Qwen/Qwen3.6-35B-A3B-FP8`) | formula | 32,764 | 1,129.7 | 4.4s |
| qwen3.6moe-fp4 (`nvidia/Qwen3.6-35B-A3B-NVFP4`) | formula | 28,553 | 2,217.3 | 5.2s |

Notes:

- "formula" = the good formula (kv fp8 + MTP-2 + hybrid-safe `max_num_seqs`); "base" = no spec
  decode / default KV. The Nemotron FP8 (RedHatAI) engine dies at init on the MTP spec-decode
  path (`NemotronHMTPModel` resolves, then EngineCore aborts), so its row is base-config —
  cross-quant comparison for Nemotron is therefore config-asymmetric. The official nvidia NVFP4
  MTP head works.
- A3B MoE architectures dominate the dense 27B on both phases. Best decode: qwen3.6moe-fp4
  (2,217 tok/s, ×3 qwen3.8-fp8). Best prefill: nemotron-fp8 base (43,621) — MTP costs prefill
  on these MoEs (nemotron 43.6k base vs 30.7k formula; qwen3.6 32.8k with MTP).
- Harness support needed for Nemotron: `LayerType.FFN_ONLY` (its `nemotron_h` stack interleaves
  23 mamba / 23 MoE / 6 attention entries).
- Study-QA quality on identical seeded chunks: **Qwen3.8** writes precise, targeted questions of
  moderate length; FP8 vs FP4 outputs are sometimes token-identical (e.g. the superfecundation
  chunk) — quantization barely perturbs generation. **Nemotron Lightning** is the tersest and
  drifts easiest toward single-hop lookup questions (weakest adherence to "hard"). **Qwen3.6
  MoE** writes the most elaborate multi-hop compare-and-contrast questions with essay-length
  answers — strongest "hard" adherence, but verbose and fond of "According to the provided
  text". All 24 generations parsed cleanly (0 structured-output failures).

## Dataset-legality batch

### `max_corpus_documents` on all five loaders

The agreed semantics, applied uniformly:

- Gold document ids are **unconditionally** added.
- `None` → the whole reference pool is loaded.
- Otherwise an additional `max(0, max_corpus_documents - len(wanted_doc_ids))` non-gold
  documents are taken in corpus order.

`activation/dataset/dataset_utils.py` gained the shared helper:

```python
def extra_corpus_budget(max_corpus_documents: int | None, num_wanted: int) -> int | None:
    """None means unlimited; otherwise extra non-gold docs on top of the golds."""
    if max_corpus_documents is None:
        return None
    return max(0, max_corpus_documents - num_wanted)
```

Loader shapes:

- **BRIGHT, SciFact** (corpus-file datasets): the corpus loop keeps golds unconditionally and
  decrements the extra budget on non-golds. This also fixes the original bug — the corpus was
  previously *golds-only* (`if doc_id not in wanted_doc_ids: continue`), so retrieval never saw
  distractors.
- **NQ, SciQ, MS-MARCO** (pair datasets): the row iterator is shared with `take_to_budget`;
  after `max_examples` is reached the loop keeps scanning rows **adding documents only** until
  the corpus budget is exhausted (dedup respected; SciQ skips empty supports; MS-MARCO adds
  whole rows' passages then stops once the budget is spent).
- `max_corpus_documents` participates in `make_dataset_id` as `corpus=` so cached ids differ.

CPU verification: BRIGHT `n=5, mcd=30` → 30 docs with all 22 golds present; SciFact `mcd=0` →
5 docs == the 5 golds; SciQ `mcd=25` → 25 docs / 5 examples. Full
`test_basic_dataset_loading.py` passes on CPU (18.3 s) with `MAX_CORPUS_DOCUMENTS` wired into
every loader plus a new doc-level retrieval assertion.

### Chunker tail-duplicate guard

`activation/dataset/dataset_index.py` — the overlap chunker could emit a final chunk fully
contained in the previous one (a duplicate-embedding "free lottery ticket" for that text):

```python
if len(pieces) > 1 and pieces[-1][1] + len(pieces[-1][0]) <= pieces[-2][1] + len(pieces[-2][0]):
    pieces.pop()
```

### SciFact qrels guard

`activation/dataset/loaders/scifact.py` — qrels rows can reference query ids with no query text
(previously a `KeyError`); query texts are built first and unmatched rows are skipped:

```python
if query_id not in query_texts:
    continue
```

### Split normalization (user-applied)

`normalize_split_name` mapping `dev`/`validation`/`val` → `VAL` was applied by the user in their
prototype round; no agent change was needed.

## Prototype integration fixes

Small breakages found while wiring the user's prototype round together:

- `activation/harness/model_config.py`: `engine_kargs` → `engine_kwargs` (the
  `engine_to_device` merge site expected the latter).
- `activation/dataset/dataset_index.py`: `self.chunk_ids_indexes` → `self.chunk_id_indexes`
  (attribute was declared under the second name).
- `activation/dataset/dataset.py`: study stat stubs were not legal dataclass fields; now
  `study_prompt_tokens: list[int] = field(default_factory=list)` (same for output tokens and
  batch latencies) and `study_num_parse_failures: int = 0`, with `summarize()` extended
  (`num_study_requests`, token averages, `total_study_latency`, `study_output_tokens_per_s`,
  parse-failure count).
- `activation/harness/runtime_config.py`: `dataset_study_batch_size = 16` was missing its type
  annotation, so it silently wasn't a dataclass field.
- `activation/dataset/dataset_study.py`: `make_study_prompt` had a stray `self` parameter and no
  return; `_genenerate_study_material` typo did not match its call site.
- Stale `document_index.py` deleted after the user's rename to `dataset_index.py`.

## New engine surface

### `engine_chat_many`

`activation/harness/loaded_model.py` — batched chat against the vLLM engine with per-request
token accounting, built for the study generator but generally useful:

```python
@dataclass
class EngineChatOutput:
    text: str
    prompt_token_count: int
    output_token_count: int

def engine_chat_many(self, conversations, lora_name=None, chat_kwargs=None) -> list[EngineChatOutput]:
    ...
```

- Deep-copies conversations; merges caller `sampling_params` over defaults
  (`temperature=1.0`, `max_tokens=1024`); passes `chat_template_kwargs={"enable_thinking": False}`.
- Structured outputs ride the same path:
  `sampling_params_kwargs["structured_outputs"] = StructuredOutputsParams(json=schema)` — this is
  the vLLM 0.28 API (the old `guided_decoding` surface is gone).

### Doc-level retrieval

`activation/dataset/dataset_index.py` — `query_docs_many_frozen(queries, top_k)` over-fetches
`5 * top_k` chunks via `query_many_frozen`, then dedups through `get_top_documents` and returns
the top `top_k` documents per query. Covered by a new assertion in the basic loading test.

## Study generator

`activation/dataset/dataset_study.py` — `DatasetStudyGenerator(harness, dataset)`:

1. Samples chunks with `random.Random(dataset_study_seed)` (with replacement;
   `dataset_study_num_samples` or all chunks).
2. Builds prompts from `get_chunk_section(chunk)` (3× chunk-size extension when enabled),
   truncated by `safe_truncate_embedding_chunk` to `dataset_study_chunk_input_limit`.
3. Batches through `engine_chat_many` with a strict JSON schema — one `{question, answer}`
   object per chunk, `additionalProperties: false` — with `max_tokens` defaulting to 2048 and
   the generation `seed` defaulting to `dataset_study_seed`. (The GPU validation runs below
   predate the one-question correction; they used a 1–5-questions array schema, hence 39
   questions from 16 chunks.)
4. Each parsed QA becomes a `LabeledRetrievalQAExample`:
   `example_id=f"{dataset_id}:study:{n}"`, `origin=DataOrigin.SYNTHETIC`, `split=DataSplit.TRAIN`,
   `positive_doc_ids=[chunk.doc_id]`, `positive_chunk_ids=[chunk.chunk_id]`,
   `gold_answers=[answer]`.
5. Stats populate per batch (prompt/output tokens, latencies, parse failures) with progress
   prints per batch.

Config knobs on `HarnessRuntimeConfig`: `dataset_study_model_name`, `dataset_study_num_samples`,
`dataset_study_batch_size`, `dataset_study_chunk_input_limit`, `dataset_study_chat_kwargs`,
`dataset_study_seed`.

### The engine formula (no further A/B — carried from the benchmark round)

`activation/tests/test_basic_dataset_study.py` uses the winning configuration from
`probe_speed_ab.py` (NVFP4 + MTP-2 was 12,881 prefill / 911.6 decode tok/s vs 10,136 / 331.6 for
default FP8):

```python
STUDY_ENGINE_KWARGS = {
    "max_model_len": 20_000,  # No power-of-2 requirement; sized with buffer.
    "max_num_seqs": 64,                       # hybrid: one Mamba block per decode seq
    "limit_mm_per_prompt": {"image": 0, "video": 0},
    "kv_cache_dtype": "fp8",
    "gpu_memory_utilization": 0.9,
    "speculative_config": {"method": "mtp", "num_speculative_tokens": 2},
    "enable_lora": False,
}
```

with Qwen3.8 instruct sampling (`temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`)
via `dataset_study_chat_kwargs`, and `unsloth/Qwen3.8-27B-NVFP4` when `SUPPORTS_FP4` (Blackwell,
`torch.cuda.get_device_capability() >= (10, 0)`) else `Qwen/Qwen3.8-27B-FP8`. The test frees the
embedding model (`model_to_device(FREE_DEVICE)`) after index build so the engine's memory
fraction is honest.

## FP8 / L40S cross-check

Same test on `ac-study-probe` (g6e.xlarge L40S, eu-north-1), exercising the non-FP4 branch with
`Qwen/Qwen3.8-27B-FP8`: **`1 passed in 746.14s (12:26)`** — also 39 questions from 16 chunks
with 0 parse failures. Engine init was lighter (227.7 s, no FP4 JIT) but generation slower:
131.7 s total study latency at 33.5 output tok/s per request stream, versus 65.6 s at 75.1 on
the RTX/NVFP4 node — consistent with the earlier speed A/B. Both `SUPPORTS_FP4` branches of the
test are validated.

## Operational notes

- **Zone-pinning capacity trap**: a stopped SkyPilot cluster can only restart in its original
  AZ. During the AWS GPU capacity crunch this pinned every warm cluster to dry zones while a
  fresh `sky setup` was free to shop all regions — the fix was launching fresh named clusters
  and letting failover walk the region list (L40S landed in eu-north-1, RTX in ap-northeast-1
  after us-east-1/2, us-west-2, ap-northeast-2 and eu-north-1 all refused the RTX shape).
- **FP4 CUTLASS JIT vs host RAM**: first-time NVFP4 kernel builds spawn ~10 parallel `cicc`
  processes at ~5.4 GB each and OOM a 62 GB host; RTX runs get
  `MAX_JOBS=3 NVCC_THREADS=1 RAY_memory_monitor_refresh_ms=0`.
- Engine setup dominates test wall-clock (~9.5 of 13.4 min); repeat runs on a warm node reuse
  the torch.compile and FlashInfer autotune caches.

## Deferred (user's call, unchanged)

- Seeded prefix sampling for corpus top-up (#1) — limits are test-only.
- Eval-split migration.
- Doc-level top-k retrieval in the eval path (#3) — being solved in the user's prototype.
