# Qwen3-4B LongHealth at 1/10: Attention Matching, free KV, and a better hybrid

New setting for the free-weights idea, following your Oct 19 change of plans. The model is Qwen3-4B-Instruct-2507, on all 20 LongHealth patients (AM's 4 contexts × 5), compared at 1/10 against Figure 3 of Attention Matching (AM; arXiv 2602.16284). Every patient is served by our special-input handler. The proposed improvement starts from AM's closed-form cache and then trains it end-to-end. A plain-language explainer of AM is in [ATTENTION_MATCHING_EXPLAINER.tressoir.html](./ATTENTION_MATCHING_EXPLAINER.tressoir.html).

**Status (Oct 19):** green-lit at ≈15:20Z as the top priority, in your order: **step 1**, then **step 2a**.
- **Step 1:** AM's closed-form cache, built by their own code and served through our handler. It should reproduce ≈0.75.
- **Step 2a:** start from that cache and train it with our KL loss.

Both run on the EC2 node. Phase B, AM's own runs, now only serves as a calibration check.

## 1. Executive Summary

**Target to beat at 1/10** (Figure 3, Qwen3-4B LongHealth, read off the plot, ±0.02):

| Method | Accuracy |
|---|---|
| Full cache | ≈0.81 |
| AM, highest-attention keys | ≈0.75 |
| KVzip | ≈0.75 |
| Cartridges | ≈0.68 |
| H2O+ | ≈0.47 |
| SnapKV / PyramidKV | ≈0.35 |
| No context | ≈0.34 |

AM-OMP has no 1/10 point (≈0.74 at 1/20). Summarization is at ≈0.34–0.41.

**The proposal, "AM start, then gradient":** initialize the handler's keys, per-entry biases β and values with AM's closed-form solution. That takes minutes and should land near ≈0.75 on day one. Then train all three end-to-end, the way Cartridges does.
- It removes AM's two stated limits: kept keys must be original keys, and it matches each layer's attention rather than the answers.
- It removes Cartridges' slow start.
- The AM authors name this as their future work ("directly optimizing compact keys"), and attribute Cartridges' win at 100× to exactly that wider search.

**Why it is cheap:** with no encoder and no 60k-token read, a training step attends ≈6k entries plus a short sample. Projection, not measured: ≈1–3 s per update on one H200.

**Two things change in the main code:**
1. the special-input hook already planned;
2. a per-entry bias β in the reader's attention (the β that AM needs).

Both are off by default, and tests check that behaviour without them is bitwise identical.

**Baselines are reproducible as they are:** AM's repository ships the exact ORCD launcher for this figure (`scripts/main/qwen-longhealth5.sh`) and a precomputed head budget for Qwen3-4B-Instruct-2507. So we can rerun AM, KVzip, H2O, SnapKV, original and no-context with their own evaluation code. Cartridges is cited from the paper, because it takes hours per context.

## 2. Arms

**Phase B: their code, their evaluation** (ORCD, one H200 per configuration, ≤6 h each):

| Arm | Configuration in their launcher | Purpose |
|---|---|---|
| B1 | `t0.99_orig2` | full cache + no context |
| B2 | `t0.1_fastest` | AM, highest-attention keys |
| B3 | `t0.1_kvzip2` | KVzip |
| B4 | `t0.1_h2o` | H2O |
| B5 | `t0.1_snapkv` | SnapKV |
| B6 | `t0.1_fast` (4 jobs) | AM-OMP-fast at 1/10, missing from the figure |
| B7 | `t0.99_summarize` (optional) | summarization |

**Phase H: AM implemented in our harness, through the handler** (no training; this is the cache G2 starts from). Their selection, β and value-fit functions are imported as they are; our side captures the queries, keys and values, places the entries and evaluates. Phase B exists only to check that our harness reproduces their numbers.

| Arm | What | Must match |
|---|---|---|
| H0 | every token through the handler, plus no context | B1 within ±2.5 pts (400 questions) |
| H1 | AM highest-attention cache through the handler | B2 |

**Phase G: trained caches through the handler** (reader frozen unless noted):

| Arm | Start | Trained | Notes |
|---|---|---|---|
| G0 | — | — | speed and memory over 20 updates |
| G1 | real tokens at bin ends, β = log(T/t) | keys, β, values | the free-KV arm of the earlier plan, now with β |
| **G2** | **AM cache** | **keys, β, values** | **the proposal** |
| G3 | AM cache | keys, β, values + reader LoRA 2e-5 | the AC angle: the reader also adapts |
| G4 (optional) | AM cache | AM repo's per-layer joint fit | no full backward; cheaper middle ground |
| G5 (optional) | AM cache | as G2 at 1/20, 1/50, 1/100 | where Cartridges ≥ AM |

Rates for keys and values: 2e-2 / 2e-3 / 2e-4. β gets its own rate.

## 3. Data and training signal

- **Records and questions:** AM's LongHealth file. It has 20 patients in 4 contexts of 5, ≈60k tokens each, and 100 multiple-choice questions per context.
  - Each patient is one handler entry, compacted from the full-context read (AM's "KV-based chunking"). Its keys have therefore seen the earlier patients.
  - Size is measured over article tokens; chat-template tokens are kept as they are.
- **Training signal, as you decided (Oct 19 ≈15:00–15:20Z):** one merged data set, trained with our existing KL loss (top-32 log-probs plus remainder).
  - **Teacher:** Qwen3-4B-Instruct-2507 reading **only that sample's own patient**, which gives higher-quality answers than reading all five.
  - **≈75% self-study:** the teacher answers seed prompts (structure, summarize, ask a question, use case, creative).
  - **≈25% eval-shaped:** new multiple-choice questions with AM's "think step by step very briefly … letter" instruction. The whole response is the target whenever it parses, not a single letter.
  - **Overlap control:** near-duplicates of the 400 evaluation questions are removed.
  - **Size:** 200 samples per patient to start, ≈1k eventually.
- **Upper bound:** the reader with only the right patient as text (the `right` view), next to the five-patient full text.
- **Later (2b):** fine-tuning on answers from Qwen 3.8 27B, as a way to possibly exceed full context.
- **Evaluation:** AM's own prompt and answer parser, the model's default decoding, and up to 2048 new tokens. Our generator gains optional sampling for this.

## 4. Main-source changes

`activation/harness/kv_transfer.py`: payloads may carry `kv_bias` `{layer: [entries, kv_heads]}`.
- The existing per-attention pre-hook turns that layer's boolean mask into an additive float mask with β on the entry columns.
- Without `kv_bias`, the boolean path is untouched.

`kv_transfer.py · bind_kv_prefix` and the attention pre-hook:

```python
# bind_kv_prefix: per layer, [B, kv_heads, P] biases (0 on padded entries), carried by PrefixKV
# attention pre-hook (layer l): additive mask = where(bool_mask, 0, -inf); entry columns += bias[l] expanded to the 32 query heads
```

`activation/ac_model/*`: the batched `special_input_handler` hook from FREE_WEIGHTS_PLAN, unchanged in design:
- a third optimizer group;
- `special_inputs.pt`;
- `load()` refuses that file when no handler is attached.

Kit, outside the main source (`IB/TMP/PAPER_AM/kit/`):
- the context builder;
- `AMKVTable`: the handler for the 20 patients, holding keys, β, values and positions;
- initializers: AM via their own functions imported as they are, real tokens, and full;
- self-study generation;
- the trainer and the evaluation in AM's format.

## 5. Validation before any GPU run (CPU, tiny Qwen3)

1. No handler and no bias: outputs are bitwise identical to today.
2. A bias of log 3 on one key gives the same logits as three copies of that key (the explainer's worked example).
3. Every token through the handler gives the same logits as ordinary full-context reading.
4. Gradients reach keys, β and values only.
5. A saved table loads back identically.
6. Our AM initialization reproduces AM's own compacted attention output on a fixture.

## 6. Schedule (as of 14:55Z Oct 19)

- **ORCD:**
  - full until D15b ends (≈18:40Z);
  - H2L runs until ≈00:30Z;
  - maintenance from 04:00Z Oct 20.
  - Phase B fits between ≈18:40Z and 04:00Z, mostly two jobs at a time.
- **CSAIL** returns at 06:00Z Oct 20. Phases H and G go there, on one H200 or GH200 per arm.
- **The code and CPU tests** can be done before then.

## 7. Relation to the earlier free-weights plan

- The Llama-3.1-8B, patients 1–10, 1/15 setting is paused in favour of this one.
- The hook, the batched handler and your decisions D1–D4 carry over.
- The encoder arms (key/value, rows, rows + residuals) on Qwen3-4B need an encoder trained on the new model. That comes later, once this comparison stands.

## 8. Decisions

<article class="decision" data-tressoir-decision data-decision-state="unresolved" aria-labelledby="am-go-question">
  <header class="decision-header">
    <div>
      <h3 class="decision-title" id="am-go-question">Which parts can start now?</h3>
      <p class="decision-context">Check every part that has your go-ahead.</p>
    </div>
    <span class="decision-state" data-decision-indicator role="status" aria-live="polite">Unresolved</span>
  </header>
  <fieldset class="decision-options">
    <legend class="visually-hidden">Decision answers</legend>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="am.go.baselines">
      <span><strong>Phase B baselines with AM's own code (recommended now)</strong><small>Their launcher runs on ORCD as GPUs free up after ≈18:40Z, before the 04:00Z maintenance. No change to our code.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="am.go.code">
      <span><strong>Main-source hook + β bias + kit, with the CPU tests</strong><small>Off by default and bitwise-tested. Ready for CSAIL at 06:00Z Oct 20.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="am.go.train">
      <span><strong>Phase H and G runs as soon as the code passes</strong><small>Calibration first (H0, H1), then G1–G3 with the rate probe.</small></span>
    </label>
  </fieldset>
  <div class="field decision-feedback">
    <label for="am-go-response">Free Response</label>
    <textarea id="am-go-response" rows="2" data-tressoir-input="am.go.feedback" data-tressoir-autogrow="2:6" placeholder="Add anything the choices miss…"></textarea>
  </div>
</article>

<article class="decision" data-tressoir-decision data-decision-state="unresolved" aria-labelledby="am-signal-question">
  <header class="decision-header">
    <div>
      <h3 class="decision-title" id="am-signal-question">What should the trained caches learn from?</h3>
      <p class="decision-context">Applies to G1–G5.</p>
    </div>
    <span class="decision-state" data-decision-indicator role="status" aria-live="polite">Unresolved</span>
  </header>
  <fieldset class="decision-options">
    <legend class="visually-hidden">Decision answers</legend>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="am.signal.distill">
      <span><strong>Self-study distillation from the full-context model (recommended)</strong><small>Cartridges' objective. No answer labels; trains on everything the full-context model would say, the same signal as Cartridges.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="am.signal.mcq">
      <span><strong>Synthesized multiple-choice questions with letter answers</strong><small>Our corpus setup. Patients 11–20 would need new questions, and the answer format differs from AM's evaluation.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="am.signal.both">
      <span><strong>Both, as separate arms</strong><small>Doubles G2; cheap at the projected speed.</small></span>
    </label>
  </fieldset>
  <div class="field decision-feedback">
    <label for="am-signal-response">Free Response</label>
    <textarea id="am-signal-response" rows="2" data-tressoir-input="am.signal.feedback" data-tressoir-autogrow="2:6" placeholder="Add anything the choices miss…"></textarea>
  </div>
</article>
