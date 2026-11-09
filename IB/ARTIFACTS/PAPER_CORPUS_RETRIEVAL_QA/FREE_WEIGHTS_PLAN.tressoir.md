# Free weights for the ten corpus records

A Cartridges-style arm for Corpus Q/A: the ten patient records are recognized and answered by trained values (keys/values, input rows, or rows + residual deltas) instead of the encoder, through a small batched hook that the main code exposes and the corpus kit fills. The aim is a much faster update and a direct "AC is pluggable and compatible with earlier techniques" result. Your decisions are integrated below; implementation waits for your green light.

## 1. Executive Summary

**Goal.** Keep the reader, data, positions and evaluation exactly as in the key/value arms. Replace only "what the reader attends for record r": instead of the encoder's output, it attends a trained table of key/value numbers for that record. The encoder is about 95 % of today's update time (113 s on a GH200: 30 s encoder forward, 78 s encoder backward, ≈5 s everything else), so removing it should make an update several times faster.

**Where the code lives.**

| Piece | Location | Size |
|---|---|---|
| Hook: `ActivationContextModel.special_input_handler` | main source, `activation/ac_model/ac_model.py` | small, generic, default `None` (no change in behaviour) |
| Handler protocol (`__call__`, `parameters`, `state_dict`) | main source, `activation/ac_model/ac_model_utils.py` | a few lines |
| Third optimizer group with its own rate | main source, `ac_model_training.py` | small |
| `FreeKVTable`: recognizes the 10 records, owns the trained values | outside, `IB/TMP/PAPER_CORPUS_QA/kit_free/free_kv.py` | the actual logic |
| Runner and eval wiring | outside, `kit_free/` | scripts |

**Data flow** through `ActivationContextModel.encode_batch`:

```text
example → parts (records) → ActivationContextModel.encode_batch
                               │
                               ├─ handler(request) → ACOutput (free keys/values)   ← one of the 10 records
                               │
                               └─ None → encoder (unchanged)                         ← anything else (strict mode: error)
                                          │
              reader (LoRA, positions, masks: unchanged) ← keys/values per layer
```

**What a record's free values are.**
- At 1/15, each record gets ≈630–810 positions, ≈7,200 in all.
- Each position holds one key and one value per reader layer: 32 layers × 8 heads × 128, about 65k numbers.
- That is ≈472M trained numbers, about 7.5 GB with gradients and optimizer state. This fits next to the reader on one H200 or a 96 GB RTX.
- The positions are computed exactly as the encoder places them. So at update 0, with the recommended initialization, the free arm is identical to the key/value arm.

**Plan of runs** (same even-letter data and order as H2, ratio 1/15, 8 pairs per update, reader 2e-5, all ten records in every example):

| Run | Purpose | Length |
|---|---|---|
| F0 | Measure seconds per update and memory | 20 updates |
| F1 | Rate for the free values: 2e-2 (Cartridges), 2e-3, 2e-4 | ≈200 updates each |
| F2 | Best rate to the 6,160-pair cap, then more passes while it still improves | to the cap and beyond |
| F3 (optional) | Reader frozen, as in pure Cartridges | like F1's best |

**Projection, not measured:** reader-only updates of ≈5–10 s on one H200. That compares with 33 s for K15 on two H200s and 167 s for H3 on the RTX node, so the 6,160-pair cap would take about 1–2 h.

**Boundaries.**
- The free arm only knows these ten records; it cannot read a new document, and the encoder arms can. That is the honest comparison for the paper.
- With no handler attached, the main code behaves byte-identically. A test enforces this.

## 2. Accepted Decisions (your answers, Oct 19)

| # | Decision | Notes |
|---|---|---|
| D1 | **Start from the base model's own keys/values, at real record tokens** | Your note: the untrained encoder has no notion of a "marker". Default: the base model reads the record with no markers, and each free position takes the key/value of the real token at the end of its bin. That covers the record evenly at the same positions. Alternative, run as one extra probe: the record's last V tokens (your "[-T:]"). |
| D2 | **Reader LoRA trains at 2e-5** | Only the source of the keys/values differs from the key/value arm. |
| D3 | **Rate probe: 2e-2, 2e-3, 2e-4** | ≈200 updates each, picked by balanced accuracy, no-context view as the leak check. |
| D4 | **All three channels** | Key/value, input rows, and input rows + residual deltas. Hill-climb key/value first, then reuse its settings for the other two. |
| — | **Batched handler** | The handler takes the whole batch of requests and returns one answer per request, None where the encoder should run. That gives it the option to batch, even though our ten records gain nothing from it. |
| — | **Priority: high** | It should be fast, and we are short on GPUs for a while. If we have too few questions compared with Cartridges, synthesize more and better ones. |

## 3. Milestones

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M1 — The hook in the main source</span>
    <span class="card-oneliner">A generic, optional special-input handler on ActivationContextModel, saved and loaded with checkpoints, with its own optimizer group.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

- **Single entry point.** Every encode in training, evaluation and rollout goes through `encode_batch`, so the hook sits there, before the row cache.
- **Caching.** A handler answer is never cached, because its values change each update. Requests the handler declines go to the encoder exactly as today.
- **Optimizer and checkpoints.**
  - The handler's parameters get a third optimizer group with its own rate.
  - Checkpoints gain an optional `special_inputs.pt` next to `ac_modules.pt`.
  - Loading a checkpoint that has one without a handler attached is refused, so the trained values can never be silently ignored.
  - Old two-group checkpoints still resume.

#### Planned Changes

`activation/ac_model/ac_model_utils.py · SpecialInputHandler`

```diff-python
@@ ac_model_utils.py — after EncodeRequest @@
+class SpecialInputHandler(t.Protocol):
+    """Answers some encode requests itself (e.g. trained free values for a fixed corpus). Batched: one answer per request,
+    None where the encoder should run (the "selected indices" are the non-None positions)."""
+    def __call__(self, requests: list[EncodeRequest]) -> "list[ACOutput | None]": ...
+    def parameters(self) -> t.Iterable[nn.Parameter]: ...
+    def state_dict(self) -> dict: ...
+    def load_state_dict(self, state: dict) -> t.Any: ...
```

`activation/ac_model/ac_model.py · ActivationContextModel`

```diff-python
@@ __init__ @@
         self._dense_vocab: torch.Tensor | None = None
+        self.special_input_handler: "SpecialInputHandler | None" = None   # set from outside; None = every request is encoded
@@ trainable_parameters @@
+    def special_input_parameters(self) -> list[nn.Parameter]:
+        """The handler's trained values (empty without a handler); the trainer gives them their own rate."""
+        handler = self.special_input_handler
+        return [] if handler is None else [p for p in handler.parameters() if p.requires_grad]
@@ encode_batch — before the cache lookup @@
+        special = self.special_input_handler(requests) if self.special_input_handler is not None else [None] * len(requests)
+        if len(special) != len(requests):
+            raise ValueError("a special-input handler returns one answer per request")
         for index, request in enumerate(requests):
+            if special[index] is not None:
+                results[index] = special[index]   # never row-cached: its values change every update
+                continue
             cached = self.cache.get(...) if self.mode == ROLLOUT else None
⋯ unchanged lines omitted ⋯
@@ save @@
+        if self.special_input_handler is not None:
+            torch.save({"handler": type(self.special_input_handler).__name__,
+                        "state": self.special_input_handler.state_dict()}, folder / SPECIAL_INPUTS_FILE)
@@ load @@
+        special = folder / SPECIAL_INPUTS_FILE
+        if special.exists():
+            if self.special_input_handler is None:
+                raise ValueError(f"{folder} holds trained special inputs: attach the handler before loading")
+            self.special_input_handler.load_state_dict(torch.load(special, map_location="cpu")["state"])
```

`activation/ac_model/ac_model_training.py · optimizer groups`

```diff-python
@@ config @@
     learning_rate_ac: float = 1e-4             # AC modules and the side LoRA
+    learning_rate_special_inputs: float = 0.0  # a special-input handler's trained values (free weights)
@@ optimizer construction (≈line 696) @@
                 groups = [{"params": ac_parameters, "lr": config.learning_rate_ac}, {"params": target_parameters, "lr": config.learning_rate_target_lora}]
+                special = ac_model.special_input_parameters()
+                if special:
+                    groups.append({"params": special, "lr": config.learning_rate_special_inputs})
⋯ the resume layout check accepts 2 groups (old) and 3 groups (with a handler) ⋯
⋯ after a step that moved only the special-input group, invalidate_encoded_rows() still runs (cached encodes keyed by version) ⋯
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M2 — FreeKVTable, outside the main source</span>
    <span class="card-oneliner">Recognizes the ten record parts and returns their trained keys/values at the encoder's own positions.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

- **Recognition** uses the same content hash as the row cache, over each record part's exact messages, ratio and tools.
  - The builder checks that the packet has exactly 10 distinct record parts and maps each to its patient.
  - In strict mode an unknown part raises, so the arm can never quietly fall back to the encoder.
- **Positions and lengths** are computed once, the way `_encode_kv_batch` does: interleaved markers at 1/15 over the record's own tokens.
- **Initialization** (D1) copies the untrained encoder's keys/values, under no gradient. So update 0 equals the key/value arm.

#### Planned Changes

`IB/TMP/PAPER_CORPUS_QA/kit_free/free_kv.py · FreeKVTable`

```diff-python
@@ new file @@
+class FreeKVTable(nn.Module):
+    """Trained keys/values for a fixed set of record parts (Cartridges-style), served through ActivationContextModel.special_input_handler."""
+    def __init__(self, ac_model, record_requests: list[EncodeRequest], *, init: str = "fresh", strict: bool = True):
+        super().__init__()
+        self.keys_by_request = {request_key(r): i for i, r in enumerate(record_requests)}   # exactly 10, checked
+        outputs = initial_outputs(ac_model, record_requests, init)     # handler detached, no grad
+        self.keys = nn.ParameterList(...)    # per record, per layer [V_r, 8, 128] fp32
+        self.values = nn.ParameterList(...)
+        self.positions, self.lengths = ...   # = the encoder's kv_positions / kv_length for each record
+
+    def __call__(self, requests):
+        return [self.answer(request) for request in requests]   # batched interface; per-record answers here
+
+    def answer(self, request):
+        index = self.keys_by_request.get(request_key(request))
+        if index is None:
+            if self.strict: raise KeyError("not one of the ten records")
+            return None
+        return ACOutput(zero_rows(index), kv_values={layer: (k, v) for layer, k, v in self.layer_values(index)},
+                        kv_positions=self.positions[index], kv_length=self.lengths[index], kv_overwrite=False)
⋯ state_dict / load_state_dict inherited from nn.Module; build_from_packet(packet) collects the 10 distinct parts ⋯
```

`IB/TMP/PAPER_CORPUS_QA/kit_free/tests/test_free_kv.py`

```diff-python
@@ new file (tiny Llama, CPU) @@
+def test_no_handler_is_byte_identical(): ...
+def test_fresh_init_matches_encoder_logits(): ...
+def test_gradients_reach_free_values_only(): ...
+def test_save_load_round_trip_and_refusal(): ...
+def test_strict_unknown_part_raises(): ...
+def test_three_group_resume_and_old_two_group_resume(): ...
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M3 — Runner, evaluation and the speed check (F0)</span>
    <span class="card-oneliner">kit_free runner and eval wiring, then 20 updates on one H200 to measure seconds per update and memory.</span>
    <span class="card-badge">TBD</span>
  </summary>

- **Runner:** `kit_free/run_free.sh` follows `kit_hill/run_hill.sh`.
  - New settings: the free-values channel, the free-values rate, the initialization and whether the reader is frozen.
  - It builds the packet, attaches the table and trains.
- **Evaluation:** the evaluation attaches the same table before loading a snapshot.
- **Hardware:** one H200 on ORCD (preemptable, before its 04:00Z Oct 20 maintenance), or CSAIL after it returns at 06:00Z Oct 20.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M4 — Runs F1, F2 (and F3)</span>
    <span class="card-oneliner">Rate probe, then the long run on the pairs axis, compared with K15, K15c and the hill-climbing arms.</span>
    <span class="card-badge">TBD</span>
  </summary>

- **Metrics:** each snapshot gets all-ten, right-record and no-context accuracy, balanced accuracy and the letter counts.
- **Comparison:** at matched pairs against K15, K15c, H1a and H2.
- **Record:** results go to `research/CORPUS_FREE_WEIGHTS.md`.

</details>
