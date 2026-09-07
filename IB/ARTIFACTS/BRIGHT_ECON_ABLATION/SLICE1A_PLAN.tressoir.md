# Slice 1a — BRIGHT economics ablation: short probes that size the overnight runs

Your diff asks for three ways of fitting the BRIGHT economics corpus (programmatic spans, generated descriptions, generated questions), a test-set evaluation against a trained embedder, a prefix-plus-view activation-context model, a Qwen3-4B base, result caching and a synced working folder. The deliverable of this slice is not the overnight runs themselves: it is three short probe runs (one per mode, in parallel on three nodes, about 20 minutes each) that measure every stage's rate on the real pipeline and print how many samples fit in a 6-hour round. Round 3: everything is implemented and the three probes have run (all three nodes torn down). The measured rates, the budget tables, the Round 1 and Round 2 commands and the labeler comparison are in `SLICE1A_ROUND.tressoir.md` next to the 25 diff cards; this page carries the completion reports per milestone. The headline: **Round 1 fits one epoch over 150k programmatic spans, all 50.8k chunks in description mode, and 25k chunks in question mode; Round 2 fits the same study sets after a 50k or 25k programmatic pretraining.** The Qwen3-Embedding-4B baseline under the in-batch policy is rank-1 0.61, MRR@10 0.67, nDCG@10 0.64. Round 4 (closing): Round 1 ran on 2026-09-06 — description fitting matches the reference (0.61 / 0.67 / 0.63 vs 0.61 / 0.67 / 0.64), questions plateau at 0.58 / 0.65 / 0.59 regardless of volume, programmatic spans degrade with volume (0.39); the description learning curves on the test set climb in the first ~150 examples and reach the reference band by 3k–5k. Your decision: the benchmark is saturated, move on; Round 2 not run. Final cards and the full record are in `SLICE1A_ROUND.tressoir.md`; every node is torn down.

## Executive Summary

### Goal

Answer, with measured numbers, "how many economics chunks can each mode generate, label, train on and evaluate in 6 hours on one RTX PRO 6000", and hand you the launch commands for Round 1 (each mode alone) and Round 2 (programmatic pretraining followed by cached description or question training).

### The corpus, so the numbers below make sense

| quantity | value |
| --- | --- |
| documents | 50,220, mean 394 chars (about 100 tokens), max 39.7k chars |
| chunks at 4,096 chars | 50,811, essentially one per document |
| labeled test queries | 103, mean 740 chars, mean 7.8 gold documents (max 85) |

"4x the corpus" is about 200k programmatic examples. Every mode's cost is dominated by training, because economics chunks are short and the 4B FP8 generator writes 200k answers per GPU-hour.

### The three modes

| mode | query | positive | hard negatives | oracle stages | new code |
| --- | --- | --- | --- | --- | --- |
| programmatic | a span of the chunk, 128–1,024 chars (decision 1) | the chunk | none, in-batch only | none | `generate_programmatic_examples` |
| description | a generated search-friendly description of the chunk | the chunk, plus labeler positives | labeler negatives from the BM25 top-10 | generation + labeling, cached | description prompt, description-mode labeling |
| question | a generated hard question (existing pipeline) | the chunk, plus labeler positives | labeler negatives | generation + labeling, cached | origin split, cache |

Evaluation is the same for all: the whole test set, the 103 native queries with one chunk per gold document, scored as one batch (the pool is the batch's distinct candidates: about 800 gold chunks, every query ranked against all of them) before and after training, next to two references on the same batch: the frozen base and Qwen3-Embedding-4B. Nothing beyond the batch is embedded, per the standing decision, and the batch is the full test set, so the number is deterministic and has no small tail batch. On BRIGHT the golds are true positives, so in-batch nDCG@10 is a fair trend signal; 103 queries make it noisy to about ±0.05. The probe report states the baseline under exactly this policy: Qwen3-Embedding-4B's in-batch rank-1, MRR@10 and nDCG@10 over the 103 queries against the 800-chunk gold pool, which is the target line for every mode.

### What the probes do

Each probe runs the mode's full pipeline on a small sample and writes `probe_timing.json` plus the usual live report:

| stage | probe size | what is measured |
| --- | --- | --- |
| generation (description, question) | 512 chunks | engine load time, chunks per second, parse failures |
| labeling (description, question) | the 512 generated examples | engine load, labels per second, positives and negatives per example |
| training | 512 labeled examples, or 2,048 programmatic ones; 2 epochs | batch size from the heuristic and probe, step time, real tokens per second, tokens per example, peak memory |
| evaluation | all 103 test queries as one batch, twice (before, after), plus the two references | seconds per pass, the in-batch metrics; the baseline line under the policy |
| labeling, second labeler (description, question) | the same 512 examples | labels per second on the 4B FP8 generator used as labeler, agreement with the A3B labels |

From the rates it prints the largest sample count whose total time fits 5.5 hours (30 minutes of safety), and the time for 25k, 50k, 100k and 200k samples. Round 2 (programmatic then cached description or question training in one process) is sized from the same rates: the study outputs come from the cache written in Round 1, so Round 2 pays training only.

Actual probe wall times: programmatic 12 minutes, description 27, question 35; the difference is the A3B labeler's 11-minute engine load and the labeled examples' larger training cost. All three ran in parallel on fresh nodes with `--sync`; the reports and the cache landed in `IB/TMP/SYNC/BRIGHT_ECON_ABLATION/` while they ran.

### Measured budget (round 3)

The probes replaced the guesses. Every mode trains at about 3k real tokens per second; a programmatic example is 248 tokens (0.106 s), a labeled example carries its ten-chunk pool (1,750–2,390 tokens, 0.59–0.79 s). Generation runs at over 100 chunks per second and labeling at 11–23 labels per second, so the study steps are minutes and training is the budget. The 4B marks 70 % of chunks as needing no abstraction bridge, so the corpus yields about 15.4k descriptions.

| mode | seconds per chunk (4B labeler) | fixed | fits in 5.5 h | Round 1 choice |
| --- | --- | --- | --- | --- |
| programmatic | 0.106 | 268 s | 184k spans | 150k spans, 4.5 h |
| description | 0.265 (0.30 examples per chunk) | 589 s | 72k chunks, more than the corpus | all 50,811 chunks, 3.9 h |
| question | 0.643 | 589 s | 29.9k chunks | 25k chunks, 4.6 h |

Round 2 (programmatic then the cached study set, one process, about 340 s fixed because no engine loads when the cache is complete): 50k spans + the corpus descriptions in 5.0 h; 25k spans + the 25k questions in 5.0 h. Commands, the A3B variant and the sample outputs are in `SLICE1A_ROUND.tressoir.md`.

### Architecture of the changes

Where the work lands under `activation/`:

```
bench/bright_econ_ablation/            one thin script per mode, shared _common.py (args, harness, data, cache, eval, timing)
   |                                    reports and cache under IB/TMP/SYNC/BRIGHT_ECON_ABLATION/ via data_syncing.resolve_path
dataset/dataset_study.py               generate_examples_{qa,descriptions} share one generation loop; labeling knows both kinds
dataset/dataset_study_prompts.py       every prompt, schema and message builder
dataset/dataset_caching.py             JSONL cache of generation and labeling outputs, prefix reuse
dataset/dataset_manager.py             origins filter, description entry point, select_testing_data
dataset/dataset.py                     DataOrigin split, programmatic_retrieval_examples
retrieval/retrieval_ac.py              P prefix rows + V view rows, d_ac 1024, 8 layers
retrieval/retrieval_model.py           document instruction; [prefix][instruction + text][EOS][views]
retrieval/retrieval_trainer.py         train(reporting only), eval(eval_data) with the references
common/data_syncing.py + cloud/sky.py  IB/TMP/SYNC mirrored before, during and after `exec --sync`; `exec --watch`
```

### Order of work

M1 data and study kinds, M3 model and trainer, M4 benches, M2 sync and cache, then M5 the probes. M2's cache and path resolution are needed by the probes; the `exec --sync` loop can land right after the probes if time is short, with `watch` covering the probes as today.

## Accepted decisions (round 2)

| # | decision | your answer | effect |
| --- | --- | --- | --- |
| 1 | programmatic query | a random contiguous span of 128–1,024 chars of the chunk; the chunk is the positive | `generate_programmatic_examples` in M1; the span bounds are the two config knobs you added |
| 2 | generator | RedHatAI/Qwen3.5-4B-FP8-dynamic at max_num_seqs 256 | descriptions and questions; added to the engine recommendations; model id in the cache key |
| 3 | base and references | Qwen/Qwen3-4B, Qwen3-Embedding-4B as the trained reference, frozen base as the floor | bench defaults in M4 |
| 4 | prefix-plus-view before the probes | yes: P=16, V=8, d_ac 1,024, 8 layers | M3 lands before M4 |
| 5 | description-mode labeling | the description alone, as "Description" | label prompt variant in M1 |
| 6 | probe layout | three fresh nodes in parallel, one per mode, torn down after | M5 |

### Your notes, folded in

- **Labeler: A3B or the 4B?** You remember labeling being the slow part. The record agrees in wall time and disagrees in rate: in the multi-GPU study test the A3B labeled at 28.4k prompt tokens per second, but its prompts were 32k-char pools (about 8k tokens each), so it did about 3.5 labels per second while the generator did 512 output tokens per second on 200-token answers, about 2.5 questions per second. Both stages were engine-load dominated at that size. On economics the pool is the BM25 top-10 of ~400-char chunks, about 1.2k tokens per label, so either labeler should run far faster than then. Whether the 4B dense (about 4B parameters per token) beats the A3B MoE (3B active, but MoE prefill is less efficient in vLLM) is a probe question, so the description and question probes label the same 512 examples with both and report labels per second and label agreement. Round 1 takes the faster one unless the agreement is poor, in which case the plan says so and you pick. The 4B is already resident as the generator, so probing it as a labeler costs no extra engine load.
- **Eval policy and the baseline.** Agreed: the probes evaluate the full test set. `eval` now scores all 103 queries as one batch (pool: their distinct gold chunks, about 800), so the baseline is one well-defined number per metric under the in-batch policy, printed in the probe report's evaluation table and drawn as the reference line. The general `eval` keeps the reporting-size batches for sets larger than 128 queries.
- **Orphan nodes** (round 3 status: the three probe nodes are torn down; `ac-fp4-probe` and the untracked instance still need your two commands below). `sky ls` showed six stopped clusters from three days ago (`ac-fp4-probe`, `ac-family-ab`, `ac-study-probe`, `ac-study-rtx`, `ac-q4-probe`, `ac-loaded-model-tests`); five are torn down. A direct EC2 listing of us-east-1, us-east-2, eu-north-1, ap-northeast-1 and us-west-2 found two more stopped SkyPilot instances in us-east-1: `ac-fp4-probe` (still tracked) and an untracked `sky-ac-test-cf861108-head` (g6e.xlarge, stopped since Aug 31). My permission mode refused both terminations, so they need your two commands (below). One running untagged `t3.2xlarge` from Aug 25 (`i-0918b8f73e9a0fe03`) is not a SkyPilot node and was left alone; it looks like a dev host. The two commands, through `activation/cloud/sky.py` and the wrapper's AWS CLI:

```
uv run sky teardown ac-fp4-probe
uv tool run --no-config --python 3.13 --from awscli==1.46.1 aws ec2 terminate-instances --region us-east-1 --instance-ids i-0f794f54eae4b1c41
```

## Milestones

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M1 — Data model, origins, study kinds, prompts file</span>
    <span class="card-oneliner">Three synthetic origins with an origins filter everywhere; description generation and labeling next to questions; programmatic examples; test-data selection; every prompt in one file.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `dataset.py`: the origin split, `programmatic_retrieval_examples`, a `study_num_description_no_bridge` counter. `dataset_study_prompts.py`: every prompt, the three schemas, `make_label_prompt(for_qa)`, `study_messages`, `label_messages`. `dataset_study.py` rewritten around a `StudyKind` record (tag, origin, prompt maker, parser): one `_generate_examples` loop for questions and descriptions, `generate_programmatic_examples` (seeded spans of 128–1,024 chars snapped to whitespace, ids `<dataset>:programmatic:<n>`), `origins` filters, description-mode labeling (`for_qa = origin != SYNTHETIC_DESCRIPTION`), `select_training_data` moved here, `select_testing_data`. `dataset_manager.py`: the new entry points.

**Drifts.** `StudyKind` is a small dataclass, not an enum. A description row is cached for every chunk, including the "no bridge" ones (query null), so the prefix rule of the cache holds. The 2 % reporting hold-out applies to study sets too (502 of 512 questions trained in the probe). Nothing else moved from the plan.

**Actual diffs.** Exact per-file deltas are the first five cards of `SLICE1A_ROUND.tressoir.md`; not repeated here.

**Validation.** All three study paths ran on the GPU probes: 512 descriptions (155 kept, 0 parse failures), 512 questions (0 parse failures), 2,048 programmatic spans; labels 155 and 512 examples with both labelers. The CPU end-to-end training test passes with the new selection. The study test itself needs an engine and was not rerun.

#### Planning Overview

`dataset.py` splits `SYNTHETIC` into `SYNTHETIC_QA`, `SYNTHETIC_DESCRIPTION`, `PROGRAMMATIC` and adds `programmatic_retrieval_examples` (a separate dict, so 200k cheap examples never mix with labeled ones and never enter labeling). The `synthetic_only` flag becomes `origins: list[DataOrigin] | None` (None = all) on `label_study_examples`, `_select_examples_to_label` and `select_training_data`; the study test passes `[SYNTHETIC_QA]` where it passed `synthetic_only=True`.

`dataset_study_prompts.py` receives the two study prompts, the label prompt with a `for_qa` switch, the three JSON schemas and the message builders (`study_messages(system, instructions, snippet)`, `label_messages(...)`), so `dataset_study.py` keeps only the loops. The description prompt is your draft with the response schema (`needs_abstraction_bridge`, `retrieval_summarization`); a chunk marked as needing no bridge yields no example and is counted. The label prompt in description mode shows "# Description" instead of "# Question / # Reference Answer" (decision 5).

`generate_examples_qa` and the new `generate_examples_descriptions` become two thin wrappers over one `_generate_examples(kind)` loop that differs only in prompt, parser and origin. `generate_programmatic_examples(num_samples)` samples chunks with the seeded shuffle (repeats allowed past the corpus size, a different span each time), builds the query per decision 1, and stores into `programmatic_retrieval_examples` with `positive_chunk_ids=[chunk]` and no hard negatives. `select_training_data` moves into `dataset_study.py` as you noted, reads both dicts through `origins`, and keeps its split logic; `select_testing_data(dataset_id, num_samples=None)` returns the native TEST examples with inherited chunk labels (one chunk per gold document; BRIGHT economics documents are single-chunk), shuffled by the seed, truncated when asked.

#### Planned Changes

`activation/dataset/dataset.py · DataOrigin, LoadedDataset`

```diff
@@ activation/dataset/dataset.py — DataOrigin @@
-    SYNTHETIC = auto()
+    SYNTHETIC_QA = auto()
+    SYNTHETIC_DESCRIPTION = auto()
+    PROGRAMMATIC = auto()
@@ LoadedDataset @@
+    programmatic_retrieval_examples: dict[str, LabeledRetrievalQAExample] = field(default_factory=dict)
+    """Programmatic examples (chunk spans), kept apart because of their volume; never labeled."""
```

`activation/dataset/dataset_study_prompts.py · new`

```diff
+STUDY_QA_JSON_SCHEMA, STUDY_DESCRIPTION_JSON_SCHEMA, LABEL_JSON_SCHEMA
+def make_study_qa_prompt(study_context) -> (system, instructions, schema)          # moved
+def make_study_description_prompt(study_context) -> (system, instructions, schema) # your draft + instructions block
+def make_label_prompt(for_qa: bool = True) -> (system, instructions, schema)       # "Description" wording when not for_qa
+def study_messages(system_prompt, instructions, snippet) -> list[dict]
+def label_messages(system_prompt, instructions, example, pool_snippets, for_qa) -> list[dict]
```

`activation/dataset/dataset_study.py · DatasetStudyGenerator`

```diff
@@ generation @@
-    def generate_examples_qa(self, num_samples):
-        ... (loop) ...
+    def generate_examples_qa(self, num_samples):
+        return self._generate_examples(num_samples, StudyKind.QA)
+    def generate_examples_descriptions(self, num_samples):
+        return self._generate_examples(num_samples, StudyKind.DESCRIPTION)
+    def _generate_examples(self, num_samples, kind):
+        # one loop: prompt = kind.prompt(study_context); parse = kind.parse(output.text) -> (query, gold_answers | None) or None
+        # origin = kind.origin; example ids f"{dataset_id}:{kind.tag}:{n}"; stats counters per kind
@@ labeling @@
-    def _select_examples_to_label(self, num_samples, synthetic_only):
+    def _select_examples_to_label(self, num_samples, origins: list[DataOrigin] | None):
-            and example.gold_answers and example.gold_answers[0]
+            and (example.gold_answers or example.origin == DataOrigin.SYNTHETIC_DESCRIPTION)
-    def generate_examples_labels(self, num_samples, synthetic_only):
+    def generate_examples_labels(self, num_samples, origins):
+        # per example: for_qa = example.origin != SYNTHETIC_DESCRIPTION; prompt and messages from the prompts file
@@ programmatic @@
+    def generate_programmatic_examples(self, num_samples) -> list[LabeledRetrievalQAExample]:
+        chunks = self._sample_study_chunks(num_samples)          # seeded; repeats past the corpus size
+        rng = random.Random(self.study_seed + 1)
+        for n, chunk in enumerate(chunks):
+            query = programmatic_query(chunk.chunk_text, rng, self.programmatic_min_chars, self.programmatic_max_chars)  # decision 1
+            example = LabeledRetrievalQAExample(example_id=f"{self.dataset_id}:programmatic:{n}", query=query,
+                origin=DataOrigin.PROGRAMMATIC, split=DataSplit.TRAIN, positive_doc_ids=[chunk.doc_id], positive_chunk_ids=[chunk.chunk_id])
+            self.loaded_dataset.programmatic_retrieval_examples[example.example_id] = example
@@ selection (moved from dataset_manager) @@
+    def select_training_data(self, num_samples, origins, oracle_labeled_only, val_ratio, max_reporting_size, force_partition, seed)
+    def select_testing_data(self, num_samples=None, seed=0)     # native TEST examples, inherited chunk labels
```

`activation/dataset/dataset_manager.py · DatasetManager`

```diff
-    def synthesize_study_examples_qa(self, dataset_id, num_samples):
+    def synthesize_study_examples_qa(self, dataset_id, num_samples, caching_id=None):        # cache in M2
+    def synthesize_study_examples_description(self, dataset_id, num_samples, caching_id=None):
+    def synthesize_programmatic_examples(self, dataset_id, num_samples):
-    def label_study_examples(self, dataset_id, num_samples, synthetic_only=False):
+    def label_study_examples(self, dataset_id, num_samples, origins=None, caching_id=None):
-    def select_training_data(self, ..., synthetic_only=False, ...):   (body moved to dataset_study.py)
+    def select_training_data(self, dataset_id, num_samples, origins=None, ...): return generator.select_training_data(...)
+    def select_testing_data(self, dataset_id, num_samples=None): return generator.select_testing_data(...)
```

`activation/harness/runtime_config.py` gains `dataset_study_programmatic_min_chars = 128`, `dataset_study_programmatic_max_chars = 1024`, `cache_storage_dir = None` (your lines). The study test replaces `synthetic_only=True` by `origins=[DataOrigin.SYNTHETIC_QA]`.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M2 — Cache and synced working folder</span>
    <span class="card-oneliner">Generation and labeling outputs cached as JSONL under a path that is local on your machine and remote on a node; `exec --sync` keeps IB/TMP/SYNC mirrored both ways.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `dataset_caching.py` (`StudyCacheKey`, `StudyCache.read/append/read_by_id`, torn last line tolerated, flush per batch); cache hooks in generation (prefix reuse by chunk id) and labeling (by example id; when every row is cached no engine is loaded, which is what makes Round 2 cheap); `data_syncing.py` (`resolve_path`, `ACTIVATION_SYNC_ROOT`); `sky.py` (`exec --sync`: update-only push, a pull thread every 15 s, a final pull; `exec --watch REMOTE LOCAL`; a `sync` command).

**Drifts.** `read(key)` returns every row and the caller slices the prefix (it also checks each row's chunk id), instead of `read(key, num_rows)`. The `--sync`, `--watch` and `--interval` flags go before the node name. The cache key includes the generator model id for labels too (labels of a generator's examples).

**Actual diffs.** Cards `dataset_caching.py`, `data_syncing.py`, `sky.py` in `SLICE1A_ROUND.tressoir.md`.

**Validation.** The four cache files (descriptions, questions, labels per labeler) and the three probe report folders appeared under `IB/TMP/SYNC/BRIGHT_ECON_ABLATION/` on the host while the probes ran, and the final pull brought `probe_timing.json`.

#### Planning Overview

`dataset_caching.py`: `StudyCache(root)` with one JSONL file per key `(caching_id, dataset_id, model_id, kind, seed)`, rows in generation order. `get(num_samples)` returns the first n rows when the file holds at least n (the seeded shuffle makes the first n of a larger request the same as a request for n, so a 50k cache serves a 25k request; a 100k request regenerates the tail after the cached 50k). Labels are keyed the same way plus `label_model_id`, one row per example id with its positive and negative chunk ids. Wired into `_generate_examples` and `generate_examples_labels` behind `caching_id`; nothing cached when `caching_id` is None or `cache_storage_dir` is unset. Cheap steps (programmatic examples, selection) are not cached, as your note says.

`data_syncing.py`: `resolve_path(relative)` returns `IB/TMP/SYNC/<relative>` locally and `~/activation_artifacts/SYNC/<relative>` on a node (the wrapper exports `ACTIVATION_SYNC_ROOT`); creates the folder. `sky.py`: `exec --sync` pushes `IB/TMP/SYNC/` to the node before the job (rsync, update-only so newer remote files survive), pulls every 15 s while the job runs, and pulls once more when it ends; `exec --watch <remote-folder> <local-folder>` folds today's `watch` loop into `exec`. Your two notes: `watch` already handles folders (`-a` is recursive), and its `--until-file` looks at the top level only, which `--sync` no longer needs. A `sync` command re-runs one push and pull for an orphaned session.

#### Planned Changes

`activation/dataset/dataset_caching.py · new`

```diff
+@dataclass
+class StudyCacheKey: caching_id, dataset_id, model_id, kind, seed, label_model_id=None
+class StudyCache:
+    def __init__(self, root: str | None)          # None: disabled
+    def read(self, key, num_rows=None) -> list[dict] | None
+    def append(self, key, rows: list[dict])         # flushes per batch, so a killed run keeps its prefix
```

`activation/common/data_syncing.py · new`

```diff
+SYNC_ROOT_ENV = "ACTIVATION_SYNC_ROOT"
+def local_sync_root() -> Path: PROJECT_ROOT / "IB/TMP/SYNC"
+def resolve_path(relative: str) -> Path: (os.environ.get(SYNC_ROOT_ENV) or local root) / relative, mkdir
```

`activation/cloud/sky.py · exec_cmd, sync`

```diff
-def exec_cmd(name, cmd):
+def exec_cmd(name, cmd, *, sync=False, watch: tuple[str, str] | None = None):
+    if sync: _push_sync(record)                                  # IB/TMP/SYNC -> ~/activation_artifacts/SYNC, -au
+    puller = _start_pull_loop(record, ...) if sync or watch else None   # every 15 s, thread, nothing deleted locally
     ... run "sky exec" with ACTIVATION_SYNC_ROOT=/root/activation_artifacts/SYNC in the env ...
+    finally: stop the loop, final pull
+def sync(name): one push, one pull
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M3 — Prefix-plus-view AC model, document instruction, train and eval split</span>
    <span class="card-oneliner">The AC model writes P rows before the text and V rows after EOS; documents get an instruction; training reports on the reporting batch only and `eval` scores a held-out set with the references.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `retrieval_ac.py`: `row_queries` of P + V rows, `num_rows`, defaults P=16, V=8, d_ac 1,024, 8 layers. `retrieval_model.py`: `DEFAULT_DOCUMENT_INSTRUCTION`, the sequence `[pad][P prefix][instruction + text][EOS][V views]` with per-row prefix placement by `scatter`. `retrieval_batching.py`: estimates count P + V rows and the document instruction. `retrieval_baseline.py`: the baseline embedder gets an empty document instruction (the Qwen3-Embedding convention). `retrieval_trainer.py`: `train` without validation; `eval(model, reporter, eval_data, config, label, progress)` scores a set of at most 128 queries as one batch (larger sets in reporting-size batches) with the frozen base and the baseline on the same batch and returns `{name: (loss, metrics)}`. Reporter: an evaluations table, the metrics plot shared by training and eval points, reference lines. Config: `reporting_size`; the validation stats are gone.

**Drifts.** The one-batch rule for small sets comes from your round-2 note (the plan had reporting-size batches). `progress` places the eval point on the plot. The e2e test uses P=4, V=4, d_ac 256, 2 layers on the CPU.

**Actual diffs.** Cards `retrieval_ac.py` through `retrieval/__init__.py` in `SLICE1A_ROUND.tressoir.md`.

**Validation.** CPU end-to-end test `1 passed in 963 s`; batch-composition invariance with prefix rows holds (max diff 1e-6). On the probes the batch heuristic passed its real-batch probe on the first attempt in every mode (128 programmatic, 20 description, 23 question) with peaks of 41.6–48.8 GB.

#### Planning Overview

`StandardRetrievalACModel(num_prefix_tokens=16, num_view_tokens=8, d_ac_model=1024, num_ac_layers=8)` keeps one learned query table of P+V rows; `forward` returns `[B, P+V, d_model]` and `embed_batch` splits it: the sequence becomes `[pad][P prefix rows][instruction + text tokens][EOS][V view rows]`, left-padded, attention mask over prefix..views, positions by cumulative sum, readout the last V rows. The prefix rows carry the byte-level reading of the text into the frozen base before it reads the tokens; the view rows read out after. `num_vectors` stays V. The batching estimate counts P+V rows per sequence, and the sizing formula's AC term uses the new width and depth. Defaults change for every caller, including the CPU test (P=4, V=4 there).

`DEFAULT_DOCUMENT_INSTRUCTION = "Instruct: Summarize this document for retrieval\nDocument: "` prefixes documents the way the query instruction prefixes queries; both are constructor arguments. Judgment call: the wording; any short instruction works since LoRA and the prefix adapt to it.

`RetrievalTrainer.train(config, model, reporter, training_data, reporting_data)` loses `validation_data`, the epoch-end validation and the references; the reporting batch and the epoch-0 point stay. `eval(model, reporter, eval_data, config, label)` batches `eval_data` at the reporting size, scores the model and the references (frozen base, baseline embedder) on the same batches, writes an "eval: {label}" table row and reference lines, and returns `{name: (loss, metrics)}`. The benches call it before and after training on the test queries. The e2e test calls `train` then `eval` on its validation split and keeps its asserts.

#### Planned Changes

`activation/retrieval/retrieval_ac.py · StandardRetrievalACModel`

```diff
-        d_ac_model: int | None = None,      # None: d_model // 4
+        d_ac_model: int | None = None,      # None: 1024
+        num_prefix_tokens: int = 16,        # rows placed before the text tokens
         num_view_tokens: int = 8,
-        num_ac_layers: int | None = None,   # None: base layer count // 4
+        num_ac_layers: int | None = None,   # None: 8
-        self.view_queries = nn.Parameter(torch.randn(num_view_tokens, self.d_ac_model) * 0.02)
+        self.row_queries = nn.Parameter(torch.randn(num_prefix_tokens + num_view_tokens, self.d_ac_model) * 0.02)  # P then V
     def forward(self, texts) -> torch.Tensor:
-        """[B, V, d_model] view rows, one group per text."""
+        """[B, P + V, d_model]: P prefix rows then V view rows, one group per text."""
```

`activation/retrieval/retrieval_model.py · RetrievalModel.embed_batch`

```diff
+DEFAULT_DOCUMENT_INSTRUCTION = "Instruct: Summarize this document for retrieval\nDocument: "
-        if is_query:
-            texts = [self.query_instruction + text for text in texts]
+        texts = [(self.query_instruction if is_query else self.document_instruction) + text for text in texts]
-        num_views = self.num_vectors if self.ac_model is not None else 0
-        length = max(len(row) for row in token_rows) + num_views
+        num_prefix, num_views = (self.ac_model.num_prefix_tokens, self.num_vectors) if self.ac_model is not None else (0, 0)
+        length = num_prefix + max(len(row) for row in token_rows) + num_views
         ... left-pad the tokens into positions [length - num_views - len(row), length - num_views) as today ...
+        attention_mask[index, length - num_views - len(row) - num_prefix : length] = 1
-                inputs_embeds = torch.cat([inputs_embeds[:, :length - num_views], view_rows], dim=1)
+                rows = (self.ac_model(texts) * self.view_scale).to(base_dtype)          # [B, P+V, d_model]
+                prefix_rows, view_rows = rows[:, :num_prefix], rows[:, num_prefix:]
+                token_part = inputs_embeds[:, num_prefix : length - num_views]
+                inputs_embeds = torch.cat([inputs_embeds[:, :pad_end], prefix_rows, token_part_after_pad, view_rows], dim=1)
+                # per-row placement: prefix rows sit directly before each row's first real token (left padding varies per row)
```

The per-row placement is the one subtle part: with left padding the prefix rows must follow each row's padding, so `embed_batch` scatters the P rows at `length - num_views - len(row) - num_prefix` per row instead of one `cat`. The CPU invariance check (`IB/TMP/BATCHING/invariant_cpu.py`) is rerun so rows stay independent of batch composition.

`activation/retrieval/retrieval_batching.py`: `estimated_tokens(text_length, num_rows)` with `num_rows = P + V`; the sizing AC term uses `d_ac_model` and `num_ac_layers` from the model.

`activation/retrieval/retrieval_trainer.py · train, eval`

```diff
-    def train(self, training_config, retrieval_model, retrieval_reporter, training_data, reporting_data, validation_data=None):
+    def train(self, training_config, retrieval_model, retrieval_reporter, training_data, reporting_data):
-        validation_batches = fixed_batches(validation_data, ...); references ...; validate(0) ... validate(epoch)
+        (reporting points and epoch-0 point only)
+    @torch.no_grad()
+    def eval(self, retrieval_model, retrieval_reporter, eval_data, config, label="eval") -> dict[str, tuple[float, dict]]:
+        batches = [one batch of all eval_data] if len(eval_data) <= 128 else fixed_batches(eval_data, dataset_index, config.reporting_size, config.seed)
+        results = {"model": self._evaluate(model, batches, T)} | self.evaluate_references(config, model, batches)
+        reporter.report_eval(label, results); return results
```

`retrieval_training_config.py`: `reporting_size: int = 50` (the eval batch size), `validation_*` stats become `eval_results: dict[str, dict]`. Reporter: `report_eval` adds an "Evaluations" table (label, embedder, loss, metrics) and keeps the reference lines on the metrics plot.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M4 — The three benches and their probe mode</span>
    <span class="card-oneliner">One shared module, three thin scripts, a `--probe` flag, datetime-stamped reports and a timing file with the budget arithmetic.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `_common.py` (arguments, harness, corpus, model, `StageTimer` with net-of-load seconds, two-labeler labeling with Jaccard agreement, the full-test-set eval, `extrapolate`, `run`), the three thin scripts, `--probe`, `--label-model-ids`, `--pretrain-programmatic` (Round 2 in one process with an eval after each phase), `probe_timing.json` / `run_summary.json`, a "timing" text block in the report.

**Drifts.** `run_stage` became a `StageTimer.stage(name, units)` context manager. The first version timed the engine loads inside the study stages; the bench now subtracts the harness's load timings per stage and treats every load as a fixed cost, and `IB/TMP/BRIGHT_ECON_ABLATION/recompute.py` applied the same rule to the finished probes (the numbers in the round document). The extrapolation gives a one-epoch table (`round_1`) and one at the probe's epochs.

**Actual diffs.** Cards `_common.py` and the three `bench_*_fitting.py` in `SLICE1A_ROUND.tressoir.md`.

**Validation.** Three probes, EXIT 0 each; logs `IB/TMP/BRIGHT_ECON_ABLATION/probe_{prog,desc,qa}.log`.

#### Planning Overview

`bench/bright_econ_ablation/_common.py` holds everything the modes share: the argument parser (`--probe`, `--num-samples`, `--epochs`, `--base-model-id`, `--baseline-model-id`, `--generator-model-id`, `--label-model-id`, `--caching-id`, `--seed`, `--report-root`), the harness build (base, baseline, generator and labeler configs; `cache_storage_dir` from `resolve_path("BRIGHT_ECON_ABLATION/CACHE")`), the dataset load (`BrightDataset.load(harness, max_examples=None, domain="economics", max_corpus_documents=None)`, BM25 index), the model build (LoRA r128, P+V AC, no head), `run_stage(name, fn)` timing wrapper, the evaluation (`select_testing_data` then `trainer.eval` before and after training), and `write_timing(...)` which stores per-stage seconds and rates and the extrapolation table.

The three scripts each define one `make_training_data(harness, dataset, n)`: programmatic calls `synthesize_programmatic_examples(n)`; description calls `synthesize_study_examples_description(n, caching_id)` then `label_study_examples(n, origins=[SYNTHETIC_DESCRIPTION], caching_id)`; question does the same with the QA kind. Engines are freed before the base loads (`engine_to_device(FREE)`), so one node runs generation, labeling and training in sequence. `--probe` sets n to 512 (2,048 for programmatic), epochs to 2, labels the 512 examples with both labelers (`--label-model-ids`, the A3B and the 4B generator; agreement and labels per second go into the timing file) and the report folder to `PROBE_REPORTS/<mode>_<YYYYmmdd_HHMM>/`; without it n defaults to the mode's Round 1 size once the probes fix it, and the folder is `REPORTS/<mode>_<stamp>/`. Round 2 is `--pretrain-programmatic <n>`: a programmatic phase, then the cached description or question phase, in one process, with eval after each phase.

The extrapolation printed at the end of a probe: for n in {25k, 50k, 100k, 200k} the wall time of each stage from the measured rates plus the fixed costs (engine loads, base load, eval), and the largest n under 5.5 hours. That table is the deliverable of M5.

#### Planned Changes

`activation/bench/bright_econ_ablation/_common.py · new`

```diff
+MODES = {"programmatic": ..., "description": ..., "question": ...}
+def parse_args(mode) -> argparse.Namespace                 # --probe sets probe sizes and the PROBE_REPORTS folder
+def build_harness(args) -> HarnessRuntime                  # base, baseline, generator, labeler; cache dir via resolve_path
+def load_economics(harness) -> LoadedDataset               # whole corpus, BM25
+def build_model(harness, args) -> RetrievalModel           # LoRA r128, AC P=16 V=8 d_ac 1024 layers 8
+class StageTimer: with timer.stage("generation", units=n): ...   -> seconds, units per second
+def evaluate(trainer, model, reporter, harness, dataset, config, label) -> dict   # all 103 test queries, one batch, with references
+def write_timing(folder, timer, training_stats, extrapolation) -> probe_timing.json
+def extrapolate(timer, tokens_per_example, epochs) -> {n: hours} + largest n under 5.5 h
```

`activation/bench/bright_econ_ablation/bench_programmatic_fitting.py`

```diff
+def make_training_data(harness, dataset, n):
+    return harness.dataset_manager.synthesize_programmatic_examples(dataset.dataset_id, n)
+if __name__ == "__main__": run(mode="programmatic", make_training_data)
```

`bench_description_fitting.py` and `bench_qa_fitting.py` differ only in the two study calls and the origin.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M5 — The probe runs and the budget table (deliverable)</span>
    <span class="card-oneliner">Three probes in parallel on fresh nodes; the measured rates, the samples-per-6-hours table for Round 1 and Round 2, and the launch commands.</span>
    <span class="card-badge">Completed</span>
  </summary>

#### Completion report

Done: `ac-econ-prog`, `ac-econ-desc`, `ac-econ-qa` ran the three probes in parallel (12, 27 and 35 minutes) and are torn down. Reports and timing files: `IB/TMP/SYNC/BRIGHT_ECON_ABLATION/PROBE_REPORTS/{programmatic_20260905_2116,description_20260905_2120,question_20260905_2120}/`. The budget table, the Round 1 and Round 2 commands, the labeler comparison (the 4B labels 1.5x faster than the A3B and skips its 11-minute load; Jaccard agreement 0.87 on question positives, 0.75 on description positives; recommendation: the 4B for Round 1) and four sample descriptions and questions are in `SLICE1A_ROUND.tressoir.md`. Evaluation on the 103 test queries as one batch: baseline 0.61 / 0.67 / 0.64 (rank-1 / MRR@10 / nDCG@10), frozen base 0.01 / 0.03 / 0.02, the model after the probe's short training 0.44 / 0.51 / 0.48 (2,007 spans), 0.54 / 0.62 / 0.60 (145 descriptions), 0.59 / 0.65 / 0.60 (502 questions).

**Round 1 and the curves (closing).** Round 1 (one epoch each): description 18.6k examples 0.61 / 0.67 / 0.63 (3 h 49 m); question 24.5k 0.58 / 0.65 / 0.59 (4 h 07 m); programmatic 147k spans 0.39 / 0.45 / 0.39 (4 h 39 m); reference 0.61 / 0.67 / 0.64. Description curves on the test queries (`--report-on-test`, every 5 %): 968 and 1,832-example runs plateau at 0.53–0.57 after ~150 examples; the full pass-0 run reaches 0.62–0.64 at 20–30 % of its epoch (stopped there on your call). Study cache complete for both kinds and both passes with the 4B labeler (archived as `IB/TMP/SAVED_RESULTS/sep6_queries_and_labels_4B.tgz`). Details, incidents and fixes: `SLICE1A_ROUND.tressoir.md`.

Original plan: three nodes (`ac-econ-prog`, `ac-econ-desc`, `ac-econ-qa`), each `uv run sky exec --sync <node> uv run python -m activation.bench.bright_econ_ablation.bench_<mode>_fitting --probe`. Output per mode: the live report, `probe_timing.json`, four sample descriptions or questions with their labels from both labelers, the before/after in-batch metrics on the full test set next to the two references (the Qwen3-Embedding-4B line is the target under the in-batch policy). The plan's completion report will carry one table (mode × stage rate × fixed cost × largest n in 5.5 h × time for 50k/100k/200k) and the Round 1 and Round 2 commands with the chosen sample counts. Nodes are torn down after the probes.

</details>

## Judgment calls

- **Programmatic examples are not cached** and never labeled; they are cheap and live in their own dict.
- **Test queries are all 103 native BRIGHT queries.** BRIGHT has no train split, and no synthetic example is built from a query, so the only overlap with training is the corpus itself, which is the point of the ablation.
- **Eval pools are the batch's candidates.** With 7.8 golds per query and batches of 50, a query is ranked against about 400 chunks, all golds of the batch's queries. That is the same in-batch rule as training feedback.
- **Engines and the base share one node in sequence.** Generation, labeling and training never overlap; each engine is freed before the next model loads. Parallelism is across modes (nodes), not within a mode.
- **No model checkpointing.** Round 2 runs both phases in one process, as your note says.
- **Reports and the cache are the only things under `IB/TMP/SYNC/`**; logs stay in the node's job log and are downloaded on demand.
- **Round 3:** Round 1 sizes sit 50–90 minutes under the budget on purpose; question mode gets half the corpus because a question costs as much as three description chunks; the description "no bridge" escape stays (it is what makes the corpus affordable); junk chunks (a number, a URL) reach both modes and a chunk filter is a cheap follow-up; the evaluation pool is the 800 gold chunks of the test queries, the deterministic in-batch policy, so a corpus-wide number remains a slice-2 item.
