# Compaction agent evaluation on dev-75 (pre-rebuttal C4) — Hidden Residuals pairs in vLLM

Started 2026-11-16 ~20:00Z. Dev-75 only (`IB/TMP/PAPER_COMPACTION/node_sync/prefixes_dev9_75/`, S3 copy
`pre_rebuttal_20261109/compaction/pc_inputs/prefixes_dev9_75/`); eval-200 (test) is not fetched or run.

## 1. Wiring (code: `compaction/agent_eval/`, commit 795cda1)

- **`run_resume_res.py`** wraps the kit's `run_resume.py` (kit = `IB/TMP/PAPER_COMPACTION/kit/`, identical to the S3 copy
  `pc_inputs/kit/`). Text arms run unchanged. For the `ac_<rows>` arms without `--rows-only`:
  - the AC model is built from the fields saved in `ac_modules.pt` (run_qa.pair_setup's rule) and the saved model ids must be
    [Qwen/Qwen3.5-2B, Qwen/Qwen3.5-9B];
  - it checks and logs: checkpoint intervention layers = loaded layers = the engine's `hf_overrides` layers, the engine
    architecture `Qwen3_5ResForConditionalGeneration`, and `inputs_embeds_size = (1 + k) · 4096`; a failed check stops the run;
  - a reader adapter whose LoRA B matrices are all zero (frozen reader) is not sent to the engine, so the AC arm reads with the
    untuned 9B exactly like the text arms;
  - a monitor line every 60 s: host memory, process-tree RSS, engine requests/min, request payload (sent, and what the stock
    full-length tensor would have been), prompt / cached / output tokens.
  - `--rows-only` keeps the kit's old behavior (rows alone).
- **Payload fix (main code, small):** a deep model's request tensor now stops after its last AC row
  (`Agent.prepare_request`), and `activation.vllm_residuals.allow_short_prompt_embeds` (run by the plugin in every vLLM
  process) relaxes vLLM's one length-equality check. vLLM's runner already copies rows only where the tensor has them and embeds
  every token-id position itself; the prefix-cache hash of a block past the end hashes an empty slice (same on every turn).
  Why: the stock rule sends the whole prompt as a (1 + 8) · 4096-wide bf16 tensor every turn, 73,728 bytes per token, so late in
  a 50k-token continuation one request would be about 4 GB (and is SHA-256 hashed by the engine core every turn). With the cut,
  the payload stays at the first prompt's size (head + rows + instructions). Rows-only models are untouched.
- **`first_turn_check.py`**: first prompt of the ac arm built as run_resume builds it; vLLM greedy with the cut tensor vs the
  full-length tensor (must agree) vs deltas zeroed (scale); then HF with the intervention hooks on the same ids/rows/deltas.
- **`dev75_node.sh`**: SkyPilot spot node job (pattern of `cmp_node.sh` / `pool_seed_node.sh`), paper 50k settings, rollout
  caches mirrored to S3 `compaction/agent_eval_dev75/ROLLOUTS/` every 10 min, DONE markers.
- **`report.py`**: per-arm tables and paired comparisons (bootstrap over prefixes, McNemar exact).

## 2. Existing dev-75 arms (inventory)

All 75 rows each, untuned Qwen3.5-9B reader, 50k settings (budget 50,000, trajectory cap 90,000, max duration 6,000 s), run in
October (local `IB/TMP/PAPER_COMPACTION/node_sync/dev9_results/`; S3 `host_ib_tmp_20261109/PAPER_COMPACTION/node_sync/` and
`node_final_20261018/pc-{a,b}/SYNC/ROLLOUTS/`):

| cache | arm | seeds |
|---|---|---|
| `pc_dev9_s{0,1}_full` | full prefix | 0, 1 |
| `pc_dev9_s{0,1}_summary_write` / `_summary` | summary, **capped** at 500 words (the paper's old arm) | 0, 1 |
| `pc_dev9_s{0,1}_ac_{4000,2000}` | old AC pair H0, rows only | 0, 1 |
| `pc_dev9h5_s{0,1}_ac_{4000,2000}` | old AC pair H5, rows only | 0, 1 |
| `pc_dev9h5x2_s{0,1}_ac_{4000,2000}` | old H5x2 | 0, 1 |
| `pc_dev9h3_s0_ac_{4000,2000}` | old H3 | 0 |

Missing on dev-75: **no-context (`none`) at any seed**, **uncapped summary at any seed**, and every arm at seeds 2–4.

## 3. Smoke and agents per GPU (one spot g7e.4xlarge: 1 × RTX PRO 6000, 16 vCPU, 124 GiB RAM; ap-south-1)

**First-turn check** (`first_turn_check.py`, `S1_lcrec100_kl_both_c20` final, ac_4000, first 2 dev prefixes, greedy 48 tokens,
fp8 KV; S3 `agent_eval_dev75/checks/first_turn_S1_lcrec100_kl_both_c20_r4000.json`). The engine got the residual architecture,
layers [3, 7, …, 31] and inputs_embeds_size 36,864.

| prefix | prompt tokens | request tensor (cut / full length) | cut vs full length | vLLM vs HF hooks: max / mean \|Δ logprob\| (top-20), top-1, common greedy prefix | deltas zeroed: max / mean \|Δ\| |
|---|---|---|---|---|---|
| embulk__embulk__761__757 | 6,340 | 464.9 / 467.4 MB | identical (0.0, 48/48) | 0.250 / 0.075, agree, 24/48 | 5.22 / 1.97 |
| pinpoint-apm__pinpoint__2078__2077 | 4,814 | 352.4 / 354.9 MB | identical (0.0, 48/48) | 0.135 / 0.059, agree, 40/48 | 3.20 / 1.99 |

The vLLM–HF difference is the size the vLLM residual README measured (0.37–0.50 max, bf16 KV); the deltas move the log-probs by 3–5.

**End to end** (4 prefixes, seed 2, 50k settings, tag `cdv1smoke_s2`): ac_4000 4/4 solved, mean continuation 11,812 tokens;
ac_2000 3/4, 10,182 tokens; all 8 runs submitted, no errors.

**Load** (ac_4000, dev prefixes 5–75 = 71 runs, seed 2, throwaway tags `cdv1load{16,32,48}_s2`; monitor lines every 60 s):

| agents / GPU | peak host memory used | peak process-tree RSS | steady output tokens/s | wall time | solved | max_tool_errors / no_tool_call |
|---|---|---|---|---|---|---|
| 16 | 44.5 GiB | 34.9 GiB | ~600 | ~19 min | 48 / 71 | 4 / 5 |
| 32 | 57.8 GiB | 51.3 GiB | ~950 | ~16 min | 51 / 71 | 2 / 3 |
| 48 | 70.5 GiB | 59.3 GiB | ~1,100–1,240 | ~16 min | 42 / 71 | 7 / 6 |

Request payload with the cut (all three loads): mean 377–411 MB, max 487 MB for the compaction part alone (1.2–1.8 GB when the
harness adds a later tool-output AC part, which happens in a few runs as in the paper's runs). The stock full-length tensor
would have averaged 1.20–1.23 GB per request with a maximum of 4.05 GB, about 3× the memory at every load.
**Choice: 32 agents per GPU on g7e.4xlarge** (the paper's AC setting); 48 is no faster end to end and shows more tool-error
and no-tool-call finishes (the canon's sandbox limit is about 40 agents). At 32, a g7e.2xlarge (64 GB) would be tight.

(Sections 4–5 are filled in as the runs finish.)
