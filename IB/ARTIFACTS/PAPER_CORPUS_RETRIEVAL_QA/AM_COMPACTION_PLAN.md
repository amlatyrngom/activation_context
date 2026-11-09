# Compaction comparison on Qwen3-4B / LongHealth at 1/10 — agent-facing plan (Oct 19)

Source of truth for `AM_COMPACTION_PLAN.tressoir.md`. Explainer: `ATTENTION_MATCHING_EXPLAINER.tressoir.html`. Status: **Green-lit ≈15:20Z Oct 19, top priority; running on EC2 cqa-ec2**. Supersedes the Llama-3.1-8B setting of `FREE_WEIGHTS_PLAN.md`. The hook design, the batched handler and decisions D1–D4 there carry over.

## Request (user, verbatim, Oct 19 ≈14:50Z)

> I think we need a slight change of plans here. Take a look at this then: https://arxiv.org/html/2602.16284v1#S1 Same idea again, except that we'll need values for all documents. This time, we have to use Qwen 3 4B to have all the baselines. But we can a 1/10th comparison across the techniques. (Figure 3, Longhealth) From that paper, can you also glean a better approach than free kv (still using special input handler)? Can you write a simplified .tressoir.html explainer of their technique (I couldn't quite grasp it)? Since things are so fast now, we can just try many things.

## Decisions (user, Oct 19 ≈15:00–15:20Z): these override the sections below

- **Order:** step 1, then step 2a, is the first green light.
  - **Step 1** (= Phase H1): AM's closed-form cache, built by AM's own code (`kit/am_export.py`), served through `AMKVTable` in our harness. Target ≈0.75.
  - **Step 2a** (= G2): start from that cache and train keys, β and values.
  - Phase B (AM's own runs) only calibrates the harness.
- **Loss:** the trainer's existing `loss_kind="kl"`, top-32 log-probs plus remainder ("Reuse it"). The teacher is Qwen3-4B-Instruct-2507 reading **only the sample's own patient**, not the five-patient context.
- **Data:** one merged set per patient, as follows.
  - ≈75% self-study (Cartridges seed types).
  - ≈25% eval-shaped questions: new MC questions with AM's instruction ("think step by step very briefly … letter"). The whole response is the target when it parses ("It was never my intention to force a single letter").
  - Near-duplicates of the 400 eval questions are filtered out. Key agreement is logged, not filtered.
  - ≈1k samples per patient eventually; start with 200.
- **Later (2b):** SFT on answers from Qwen 3.8 27B, as a way to possibly exceed full context. Use medium reasoning if no-reasoning behaves oddly. User: "it's quite possible that Qwen-3-4B single right context is actually th right upper-bound", so the single-patient eval (`am_eval.py --views right`) is reported as the upper bound.
- **Defaults accepted** ("Use with your defaults, unless they are, like, absurdly suboptimal"):
  - uniform per-head budget first (`t0.1_fastest-u`), then AM's nonuniform budget (padded per layer, β masked);
  - reader frozen first, then LoRA 2e-5;
  - AM's protocol with the model's default sampling: 1 pass for snapshots, 3 passes for finals.
- **Kit as built** (replaces the file names under "Kit" below): `am_lh.py` (AM's data/prompt/parser), `am_export.py` (runs AM's compaction and saves the per-patient cache), `am_handler.py` (`AMKVTable`), `am_eval.py` (views am / full / none / right), `am_train.py` (2a).
- **Key positions as built:** keys are stored rotated back by the patient's start and returned with `kv_positions = 0`; the reader rotates them by the part's start. This differs from the "unrotated" sentence under "Main-source changes".

## Setting (taken from the paper and the AM repo `IB/TMP/PAPER_CORPUS_QA/am_compaction`)

- **Model:** `Qwen/Qwen3-4B-Instruct-2507`. Standard Qwen3-4B config, to verify on load: 36 layers, 32 query heads, 8 KV heads, head dim 128. Key norm (`k_norm`) is already handled by `LayerInputCapture.finish` (kv_transfer.py:113).
- **Data:** `data/longhealth_benchmark_v5.json`, dataset `longhealth5`:
  - patients 1–20, as 4 contexts × 5 patients (≈60k tokens each);
  - 100 MC questions per context, 400 in all.
  - "Values for all documents" means 20 handler entries, one per patient.
- **Size:** t/T = 0.10 over article tokens; chat-template tokens are kept as they are. That is ≈6k entries per context and ≈24k in all.
- **Chunking:** KV-based, 5 chunks per context. The whole context is prefilled once and each chunk's slice is compacted separately. This maps onto per-patient handler entries whose keys were computed with the earlier patients in view.
- **Eval:**
  - AM's `run_qa_evaluation` with `format_question` and `parse_model_choice`;
  - the model's default decoding, max 2048 new tokens, `--max-model-len 131072`.
- **AM launcher:** `scripts/main/qwen-longhealth5.sh` (ORCD mit_preemptable, 1 H200, ≤6 h per array entry).
  - Configs: `t0.1_fastest` = AM-HighestAttnKeys with `ss-plus-repeat`, on-policy, budget `head_budget_optimization/head_budgets/Qwen3-4B-Instruct-2507/optimized_agnostic.json`, max ratio per head 0.75.
  - Also `t0.1_fast` ×4 (AM-OMP-fast), `t0.1_kvzip2`, `t0.1_h2o`, `t0.1_snapkv`, `t0.99_orig2` (original + no context), and `t0.99_summarize` ×2.
  - Requirements: torch 2.8.0, transformers 4.57.1, vllm 0.11.0, Python 3.12.
- **Figure 3 at 0.10** (read off the plot, ±0.02):

  | Method | Accuracy |
  |---|---|
  | Original cache | ≈0.81 |
  | AM-HighestAttnKeys | ≈0.75 |
  | KVzip | ≈0.75 |
  | Cartridges | ≈0.68 |
  | H2O+ | ≈0.47 |
  | SnapKV / PyramidKV | ≈0.35 |
  | No context | ≈0.34 |

  - AM-OMP has no point at 0.10 (≈0.74 at 0.05).
  - Summarization: ≈0.34–0.41, points only up to 0.05.

## Sizes (arithmetic, not measured)

- One entry = 36 layers × (key + value) × 8 heads × 128 = 73,728 numbers, plus a β of 36 × 8 = 288.
- Per context, 6k entries → ≈0.44B numbers (≈1.8 GB fp32; ≈7 GB with gradients and Adam).
- All four contexts → ≈7 GB of parameters, ≈28 GB with gradients and Adam. One H200 holds this next to the bf16 reader (≈8 GB).
- A training step attends ≈6k entries plus about 1–2k sample tokens. With no encoder and no 60k prefill, the projection is ≈1–3 s per update of 8 samples on one H200 (to be measured in G0).

## Main-source changes

1. **Special-input hook:** `ActivationContextModel.special_input_handler`, a batched `list[EncodeRequest] -> list[ACOutput | None]`, as in FREE_WEIGHTS_PLAN, with a third optimizer group, `special_inputs.pt`, and load refusal without a handler.
2. **Per-entry attention bias β** in `activation/harness/kv_transfer.py`:
   - Payload key `kv_bias`: `{layer: [M, kv_heads]}`, optional.
   - `bind_kv_prefix` builds a per-layer bias `[B, kv_heads, P]`, zero on padded entries. It travels in `PrefixKV`, which gains `biases`.
   - The existing attention pre-hook in `install_kv_hooks` replaces that layer's `attention_mask` with a float mask: `where(bool_mask, 0, -inf)`, plus the bias on the entry columns, expanded over the query-head groups to `[B, 32, S, P+S]`. SDPA and eager attention both take an additive float mask.
   - `generate_with_kv_prefix` steps receive the same bias for the entry columns.
   - With no `kv_bias` in any payload, the boolean path is unchanged; a bitwise test enforces this.
   - Memory, transient per layer: B × 32 × S × (P+S) × 2 bytes ≈ 0.2 GB at B=1, S=400, P=6k.
3. **Sampling in `generate_with_kv_prefix`** (temperature / top-p / top-k), so that our harness can use AM's default decoding. Greedy stays the default.

Ordinary text gets the existing rotary treatment. Kept original keys are stored unrotated, with the original token index as `kv_positions` and the full chunk length as `kv_length`. This is exact, because AM keys are a subset of the original keys.

## Kit (outside main source): `IB/TMP/PAPER_AM/kit/`

- `contexts.py`: builds the 4 contexts exactly as AM's `load_longhealth_data` does. Each patient becomes a part, and the chat template is kept fixed outside the parts.
- `am_table.py · AMKVTable(nn.Module)`, the handler. Per patient p and per layer:
  - `keys[p][l]` `[t_p, 8, 128]`;
  - `values[p][l]`;
  - `beta[p][l]` `[t_p, 8]`;
  - `positions[p]`.
  - `trainable=False|True`. It recognizes the 20 patient parts by content hash and is strict.
- Initializers:
  - `init_am(algorithm=highest_attn|omp_fast)`: calls AM's compaction functions on our captured full-context keys, values and queries (their code is imported as is), with the precomputed head budget.
  - `init_real_tokens`: D1 of FREE_WEIGHTS_PLAN, bin-end tokens, β = log(T/t).
  - `init_full`: all tokens, β = 0. This is the harness calibration.
- `selfstudy.py`: Cartridges seed prompts (structuring / summarization / question / use case / creative) via vLLM. Qwen3-4B-Instruct-2507 sees the full context. N conversations per context, with the teacher's top-20 log-probs per token stored. Eval questions are never used.
- `train_am.py`: distillation (KL against the full-context teacher on the self-study answer tokens, as in Cartridges).
  - Parameter groups: keys, β, values (rate probe), and optionally the reader LoRA at 2e-5.
  - Snapshots every N updates are evaluated with AM's protocol on the 400 questions.

## Runs

**Phase B: baselines with AM's own code and eval** (ORCD, 1 H200 each):
- B1 `t0.99_orig2` (original + no context);
- B2 `t0.1_fastest` (AM-HAK);
- B3 `t0.1_kvzip2`;
- B4 `t0.1_h2o`;
- B5 `t0.1_snapkv`;
- B6 `t0.1_fast` ×4 (AM-OMP-fast at 0.10, missing from the figure);
- B7 summarize (optional).
- Cartridges: the published value only (≈hours per context).

**Phase H: our harness through the handler** (no training):
- H0 `init_full` and no context. These must match B1 within ±2.5 points (400 questions, SE ≈2.2).
- H1 `init_am(highest_attn)` must match B2.
- H0 and H1 together calibrate the harness and give the "AC interface hosts AM" number.

**Phase G: gradient arms** (frozen reader unless noted). G0 measures s/update and memory over 20 updates. Then:
- G1 free KV from real tokens (the FREE_WEIGHTS_PLAN arm, now with β), distillation;
- G2 **AM init → gradient on keys, β, values** (the proposed approach), distillation;
- G3 G2 + reader LoRA 2e-5;
- G4 (optional) AM init → AM repo's layer-local `OptimJointCompaction` (no full backward);
- G5 (optional) G2 at 0.05 / 0.02 / 0.01, where Cartridges ≥ AM.

Rate probe for keys and values: 2e-2 / 2e-3 / 2e-4. β gets its own rate (same grid ×10).

## Why G2 should beat both AM and free KV (hypothesis)

- It starts at AM's ≈0.75, where Cartridges-style training starts at real-token KV, far below that.
- The gradient removes AM's two limits: keys restricted to original keys, and the per-layer objective in place of the answer loss.
- The authors name exactly this as future work. They attribute Cartridges' win at 100× to the wider search space.
- An update is reader-sized, so many arms fit in hours.

## Validation (CPU, tiny Qwen3 fixture)

1. With no handler and no bias, outputs are bitwise identical.
2. `kv_bias` = log c on a duplicated key gives the same logits as c copies of that key (the worked example).
3. `init_full` through the handler gives reader logits equal to plain full-context reading.
4. Gradients reach keys, β and values, and none reach the AC modules.
5. The save/load round trip works.
6. `init_am` on a fixture reproduces AM's own compacted attention output to ≤1e-4.

## Compute windows (as of 14:55Z Oct 19)

- ORCD:
  - at the 4-GPU cap: D15b until ≈18:40Z, H2L until ≈00:30Z;
  - maintenance 04:00Z Oct 20 → 01:00Z Oct 21.
- CSAIL down until 06:00Z Oct 20.
- EC2: H3 until ≈04:15Z.
- So Phase B fits on ORCD between ≈18:40Z and 04:00Z: each config ≤6 h, at most 2 concurrent while H2L runs. Phases H and G need the main-source changes plus a GPU, i.e. CSAIL after 06:00Z Oct 20 (or ORCD slots before 04:00Z if the code is ready).
