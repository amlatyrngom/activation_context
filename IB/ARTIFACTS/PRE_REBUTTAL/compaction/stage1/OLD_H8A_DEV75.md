# Old compaction model H8a on the stage-1 dev-75 proxy (2026-11-16)

Quick reference: where the paper's compaction model (H8a) sits on the same dev-75 proxy as the new stage-1 runs.

## What was run

- Pair: `S3:node_final_20261018/pc-a/pc/pairs/h8a_final` (`pc-b` holds byte-identical files, same ETags). Reader adapter
  sha256 `3ab1f420c5ea…`.
- Scorer: `cmp_proxy_eval.py` from this folder, with the ORCD bundle `cmp_stage1_code_v1` (same code, environment, panel
  `cmp1/data/panel_dev75` and target tokens as `cmp_stage1_job.sh`). Deterministic kernels. `dev75/none` and NLL_full
  come from `cmp1/data/proxy_start_pair.json` (`--references-from`), not recomputed.
- Job: `cmp_h8a_proxy_job.sh` (ORCD `mit_preemptable`, one RTX PRO 6000): `--limit 5` check (job 29254284), then all 75
  (job 29254743). Outputs: `old_h8a_dev75/{deep,rowsonly}.json` (+ `.items.jsonl`, logs, the limit-5 files).

## H8a is a deep model, not Input Embeddings

The paper describes H8a as Input Embeddings, but its `ac_modules.pt` has `intervention_frequency -1` with trained (non-zero) delta
heads at reader layers 3, 7, …, 31. So it was **trained with residual deltas** (the CSAIL `run_channel_mixture` path). But the
paper **served it rows only** (`run_resume.py --rows-only`: vLLM gets the rows and the deltas are dropped). It loads in the
current code as-is: strict checkpoint load, interleaved layout, the same 2B encoder and 9B reader. It passes the proxy's
`res` mode check, so no `--mode` flag was needed. Two variants:

- **deep**: rows plus deltas, as trained.
- **rowsonly**: `--deltas-off` (new flag), which uses the trainer's own `deltas_off` path. It keeps the rows and withholds the
  delta payload, as the paper served H8a.

## Result (all 75 prefixes, 497,993 target tokens per group)

Gap closed = (0.6449 − NLL) / (0.6449 − 0.5057), where NLL_none 0.6449 and NLL_full 0.5057 are the untuned 9B's (shared reference).

| Model | NLL 1/10 (r10) | NLL 1/5 (r5) | Gap closed 1/10 | Gap closed 1/5 |
|---|---|---|---|---|
| H8a rows only (as the paper served it) | **0.5216** | **0.5110** | **0.886** | **0.962** |
| H8a deep (rows + deltas, as trained) | 0.5456 | 0.5414 | 0.713 | 0.744 |
| New `S1_res_kl_c20`, update 41 (`.held.jsonl`) | 0.6297 | 0.6302 | 0.109 | 0.105 |
| New `S1_res_kl_c162024`, update 41 | 0.6303 | 0.6306 | 0.105 | 0.102 |
| Start pair `R_res_rq9fr` (update 0) | 0.7135 | 0.7447 | −0.493 | −0.717 |

Not comparable, listed for information only:
- H8a's own teacher (the 9B with H8a's reader adapter, reading the full prefix) has NLL 0.4780, below the untuned 9B's 0.5057.
- KL to that teacher: rows only 0.0571 (r10) / 0.0470 (r5); deep 0.0778 / 0.0735. These are relative to H8a's adapted reader, not to the
  untuned 9B that the new runs are scored against.

## Reader vs encoder split (rows only, jobs 29257333 limit 5, then 29258032)

- **(a) `readernone`**: H8a's adapted reader on the `none` arm (head + the none-arm note, no rows). This is what the LoRA gives without
  any compacted context. Its r5/r10 reproduce `rowsonly` exactly.
- **(b) `zeroreader`**: H8a's encoder (rows only) with R_res_rq9fr's all-zero reader adapter, i.e. the untuned frozen 9B, as the new
  stage-1 runs. The two adapters were checked first: same 256 tensors, shapes, r128, alpha 256 and target modules. The scorer used a
  copy of `cmp1/data/teacher_dev75.pt` (adapter hash matched, `teacher: stored`), and its teacher NLL is 0.5057, the untuned 9B.
  So its **KL is comparable** with the new runs.

| Model | NLL 1/10 | NLL 1/5 | Gap 1/10 | Gap 1/5 | KL 1/10 | KL 1/5 |
|---|---|---|---|---|---|---|
| (a) H8a reader, no rows (`none` arm) | 0.6418 (no rows) | | 0.022 | | not comp. | |
| (b) **H8a encoder rows only + untuned frozen 9B** | **0.5917** | **0.5758** | **0.382** | **0.496** | **0.0833** | **0.0689** |
| H8a encoder + H8a reader, rows only (above) | 0.5216 | 0.5110 | 0.886 | 0.962 | not comp. | |
| New `S1_res_kl_c20`, update 81 (`.held.jsonl`) | 0.6289 | 0.6291 | 0.115 | 0.113 | 0.1120 | 0.1121 |
| New `S1_res_kl_c162024`, update 81 | 0.6291 | 0.6297 | 0.113 | 0.109 | 0.1120 | 0.1120 |
| Untuned 9B, no rows (reference) | 0.6449 | | 0 | | 0.1328 | |

Reading of the split (NLL, 1/10 and 1/5):
- The reader LoRA alone is worth almost nothing: 0.6449 → 0.6418.
- H8a's encoder into the untuned frozen 9B closes 0.38 / 0.50 of the gap, against about 0.11 for the new runs at update 81.
- The adapted reader adds the rest, reaching 0.89 / 0.96.
Against `S1_res_kl_c20` at update 81, H8a's lead splits as follows:
- At 1/10 the lead is 0.6289 − 0.5216 = 0.107. The encoder accounts for 0.037 (about 35 %) and the adapter reading the rows for
  0.070 (about 65 %).
- At 1/5 the lead is 0.6291 − 0.5110 = 0.118. The encoder accounts for 0.053 (about 45 %) and the adapter for 0.065 (about 55 %).
The adapter matters only together with rows: on its own it does almost nothing.

## Caveats on comparability

- **NLL is comparable**: same panel, items, target tokens (the ORIGINAL untuned-9B continuations) and NLL definition.
- **H8a has a trained reader LoRA**, and the new stage-1 runs keep the reader frozen (row (b) of the split removes it). Part of H8a's lower NLL is the reader adapter
  itself. Its own full-prefix teacher is at 0.4780, so on this panel even the adapted reader with the full prefix beats the untuned
  9B. This is why gap closed can approach 1. It is not an apples-to-apples encoder comparison.
- **H8a's training targets were close to this panel**: the untuned 9B's own continuations (RFT, shortest correct of 3 samples).
  Dev-75 was used earlier to promote H5, H8a's starting point. H8a's material includes evaluation-side runs, but dev-75 tasks
  are held out from every training set (DATA_INVENTORY).
- Deep scoring below rows-only is unexpected for a model trained with deltas. Possible causes, not checked: a change since
  October in how the HF path applies deltas, or deltas that hurt on this panel. The rows-only number does not depend on the
  delta path and matches how the paper reported H8a, so it is the headline here.
