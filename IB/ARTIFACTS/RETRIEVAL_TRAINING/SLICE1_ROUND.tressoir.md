# Slice 1 round — activation-context retrieval training

Diff cards for the slice 1 handoff: the harness `ModuleManager` (PEFT LoRA adapters named on one
wrapped base, retrieval AC registry), the base-model hooks, the byte-level AC model, the
multi-vector `RetrievalModel`, batching with GPU sizing and the OOM probe, the trainer with the
form-A loss and the live Plotly report, `select_training_data`, the end-to-end test, the
acceptance-run bench and the `peft` dependency. Every card is the exact delta of the workspace
file against your current `/source` tree; the staged copies under
`IB/ARTIFACTS/RETRIEVAL_TRAINING/activation/` match the cards (`SLICE1_pyproject.toml` beside
them; `uv.lock` is not carded: `uv lock` after adding `peft>=0.20.0` resolves peft 0.20.0 and
accelerate 1.14.0).

## Round 2: your `@AI` notes, applied

- **LoRA ownership** moved out of `LoadedModel` into `ModuleManager`: one PEFT wrapper per base
  (`peft_models[model_name]`), adapters added by name; `decoder_forward(..., lora_name)` selects
  the adapter per pass through `lora_context` (None disables every injected adapter);
  `free_lora(model_name, lora_name, checkpoint_path=None)` asserts on a checkpoint path, deletes
  the adapter and unloads the wrapper after the last one; `model_to_device(FREE)` refuses while
  `has_loras`. The test frees its LoRA, then the base, and checks that a pass without a LoRA name
  differs from one with it. Base weights are present once; each adapter adds only its A/B matrices.
- **`get_base_model()`** was the LoRA'd model (PEFT injects into the base's own linear layers);
  the forward now uses `loaded_model.model` directly and the docstring says why.
- **`register_retrieval_ac`** takes a `StandardRetrievalACModel` (annotation under
  `TYPE_CHECKING`, no base class) and reads `base_model_name` directly.
- **AC dropout**: knob removed; 0.1 inside the byte pooling FFN, the encoder layers and the view
  cross-attention, none on the view queries or the output projection. The 8-byte window attention
  runs without dropout: SDPA's dropout path refuses the flattened B × windows batch (> 65535 rows).
- **Log axes** label ticks as 1e-4 instead of 100μ.
- **Test asserts**: the step loss falls by an order of magnitude and the reporting loss dips below
  its first value; `last < first` on 4 held-out queries was noise (it failed once on the node).
- **MRR@10 and nDCG@10** next to rank-1, in-batch (the batch's distinct candidates with
  same-document collisions masked, binary relevance over own positives), on the reporting batch at
  every reporting point and on the validation split per epoch; stats carry the dicts.
- **Reporter refactor**: `common/reporting.py` holds the job-agnostic `HtmlReporter` and the
  vendored assets moved to `common/report_assets/`; `retrieval_reporter.py` is
  `RetrievalReporter(HtmlReporter)` with the widget layout and `report_*` events; the trainer has no
  reporting code left.
- **`retrieval_dense_index.py`** untouched.
- **Self-contained report page** (your html-skill update): the data is embedded and the Tressoir
  linear CSS/JS, CodeMirror and Plotly come from pinned HTTPS URLs; no `report_assets/` folder,
  nothing copied beside `report.tressoir.html`. The `common/report_assets` card is gone.

## Round 4: reference embedders on the validation batches

- **Question**: with `ac_name=None` and `lora_name=None` (and `d_embedding_result=None`) a `RetrievalModel`
  is the frozen base alone: EOS-token readout of the last hidden state, L2-normalized, no head, nothing
  trainable (`train()` asserts). That is a floor, not a trained reference.
- **New `retrieval/retrieval_baseline.py`**: `BaselineEmbedder` puts a harness embedding model
  (`Qwen/Qwen3-Embedding-0.6B` by default) behind the scoring interface of `RetrievalModel`
  (`embed_batch` → [B, 1, d] via `simple_vector_embed_many`, the same instruction prefix, MaxSim with
  V = 1 = the cosine), and `frozen_base_reference` builds the floor on the resident base.
- **Trainer**: `evaluate_references(config, retrieval_model, batches)` scores each reference on the
  validation batches (same pool, loss and metrics as the model under training) before the first step,
  then frees the baseline model; the untrained model gets an epoch-0 reporting point and an epoch-0
  validation point so every curve starts at its floor. Config: `baseline_model_name` (None skips),
  `reference_frozen_base` (True). Stats: `reference_losses`, `reference_metrics`.
- **Report**: flat lines `<reference> <metric>` on the metrics plot across the whole run plus a
  "Reference embedders" table. Bench flag `--baseline-model-id` (`none` to skip).
- **Nothing beyond the batch is embedded**: the references honor the decision that learning feedback
  stays in-batch. The comparison is a trend, not a corpus-level benchmark.
- **Result (950 / 50 / 50, one epoch, the heuristic's batch 128, 8 steps; `IB/TMP/BASELINE_EVAL/run/`)**:
  frozen base rank-1 0.00 / MRR@10 0.00 / nDCG@10 0.00 (loss 6.07); Qwen3-Embedding-0.6B 0.32 / 0.55 /
  0.65 (loss 2.53); the untrained model 0.02 / 0.04 / 0.05 (loss 7.79); after 8 steps 0.32 / 0.51 /
  0.62 (loss 2.22). Throughput at batch 128 with checkpointing: 7.2 s/step, 16 forwards/step, 15.0k real
  tokens/s, peak 19.7 GB, 604 s per 10k examples.
- **Why the trained reference is low**: MS-MARCO's native hard negatives are the query's other Bing
  passages (`hard_negative_doc_ids` = the row's non-selected passages), which are topically relevant, so
  in-batch rank-1 asks "which passage did the annotator select", a noisy ceiling for any embedder.
  Our model matches Qwen3-Embedding on that pool after 8 steps; nothing more can be read from it at
  this size. A corpus-level comparison belongs to the slice 2 index.

## Validation

| Check | Result |
| --- | --- |
| Pure-Python helper checks on the agent host (`IB/TMP/RETRIEVAL_SLICE1/private_helpers_check.py`: batches, dedup, length groups, fixed batches, reporter data flow) | all pass |
| `test_basic_retrieval_training.py --slow` on the Sky node (fp32 base, 32 MS-MARCO rows → 10 / 10 / 4, batch 2, 20 steps, checkpointing on) before the review fixes | `1 passed in 57.43s`; loss 12.1 → 0.6, peak 3.5 GB (`sky_e2e_test.log`) |
| Same test after the review fixes | `1 passed in 35.61s`; reporting loss 3.9 → 1.56, validation rank-1 0.60 (`sky_e2e_test_run3.log`) |
| Acceptance run 1 (pre-review): 950 / 50 / 50 MS-MARCO, batch 32 (probe passed on attempt 1), 4 epochs, 120 steps | reporting loss 45.8 → 1.88 at epoch 1, then up to 5.4 at flat rank-1 0.24–0.30; validation 1.85 / 2.08 / 5.49 / 5.37; 4.0 s/step, 7.2k real tok/s, peak 8.5 GB. Diagnosed by the reviewer as the MaxSim sum over 8 views (effective τ 0.0025). Log `sky_acceptance_run.log`, summary `gpu_run/run1_summary.txt`; the report folder was lost (see below) |
| Acceptance run 2 (post-review): same data and batch, 2 epochs, 60 steps | reporting loss 5.99 (≈ log 500) → 1.85 → 2.00; validation loss 1.853 / 2.002 with rank-1 0.34 / 0.38; 4.1 s/step, 7.0k real tok/s, 25.7 % padding, peak 8.5 GB, 1288 s per 10k examples. Report `IB/TMP/RETRIEVAL_SLICE1/gpu_run/report.tressoir.html` (+ `training_stats.json`, `report_light.png`), log `sky_acceptance_run2.log` |
| Report page inside the extension pipeline | mock verified earlier in a replay of the extension's `notebook-webview.js`; the run-2 page renders all five plots headlessly with no console errors |
| Round 2 test on a fresh node (`ac-slice1b`), with the LoRA lifecycle and per-pass adapter checks | `1 passed in 37.10s` (LoRA lifecycle: per-pass adapter selection differs from the plain base, `free_lora` then base free; `sky_e2e_test_run6.log`) (`sky_e2e_test_run4.log`) |
| Round 2 acceptance run (same shape as run 2) | 950 / 50 / 50, batch 32, 2 epochs, 60 steps: reporting loss 5.38 → 1.89 → 2.19; validation in-batch rank-1 0.34 → 0.42, MRR@10 0.55 → 0.59, nDCG@10 0.66 → 0.68; 4.0 s/step, 7.2k real tokens/s, peak 8.5 GB, 1258 s per 10k examples (`sky_acceptance_run5.log`, report `IB/TMP/RETRIEVAL_SLICE1/gpu_run3/`). An identical run one commit earlier (`sky_acceptance_run4.log`, before the pooling-attention dropout was removed and the metrics renamed) gave rank-1 0.24 → 0.28, so run-to-run spread at this size is ±0.1 in-batch rank-1 and the dropout-versus-no-dropout gap (run 2: 0.34 → 0.38) is within it. (`sky_acceptance_run3.log`, report `IB/TMP/RETRIEVAL_SLICE1/gpu_run3/`) |
| Round 4 e2e test on the agent host (CPU, fp32, 20 steps) with the frozen-base reference and the epoch-0 points | `1 passed in 737.24s` (`IB/TMP/BASELINE_EVAL/cpu_test.log`) |
| Round 4 run on `ac-baseline` (torn down): 950 / 50 / 50, one epoch, batch 128 from the heuristic (probe passed on attempt 1), Qwen3-Embedding-0.6B as the baseline | references and curves as in Round 4 above; `sky_exec.log`, report `IB/TMP/BASELINE_EVAL/run/report.tressoir.html` |
| Independent review (M8) | one blocker (view sum) and six should-fix items, all applied; defaults kept; details in the plan's M8 card |

## Judgment calls to confirm

- **MaxSim is the mean over query views**, not the sum in the v3 cards, so scores stay in [-1, 1]
  for any V and τ 0.02 keeps its single-cosine meaning. Canon carries the line.
- **AC view rows are rescaled**: unit RMS × learned scalar in the AC, × the base embedding-row RMS
  in `embed_batch`, so they enter the frozen base at token-embedding scale.
- **Adapter names are sanitized** (`qwen3-0.6b-retrieval-lora` → `qwen3_0_6b_retrieval_lora` as
  the PEFT adapter name; the LoRA name stays the registry key).
- **`batch_size=32` for the 1000-query run** instead of `None`: the heuristic would have chosen a
  larger batch (peak 8.5 GB of 96 at 32), and a fixed batch keeps runs comparable. The heuristic and
  probe are exercised when `--batch-size` is omitted.
- **Bench defaults**: 2 epochs (4 overfits 950 queries), `int(n × 1.25) + 8` rows loaded so the
  drop of unlabeled queries still leaves n; the report folder is `expanduser`ed inside the script.
- **Validation batches use the reporting size** so the two loss curves see the same number of
  in-batch candidates; at 5 % / 50 the two sets are the same 50 queries and coincide at epoch ends.
- **The old IB-only `activation/training/` package** (single-vector full fine-tune) was removed
  from the staging folder by the generator; it had no git copy. Only its `losses.py` was restored to
  `IB/TMP/RETRIEVAL_TRAINING/old_training_package/`. Say so if you want the rest reconstructed.
- **Heavy runs stay off the agent host**: the first local run at the plan's test shape peaked at
  29.5 GB (PEFT keeps fp32 copies of every LoRA input outside autocast; without checkpointing ~1 MB
  of saved activations per token) and was OOM-killed. The test shape shrank and both runs moved to
  the Sky node; canon and the agent-container note carry the rule.

## Diffs to apply

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/tests/test_basic_retrieval_training.py</span>
    <span class="card-oneliner">The end-to-end test: fp32 Qwen3-0.6B, 32 MS-MARCO rows, batch 2, 20 steps, checkpointing on; a live report, the loss-trend assert, the frozen-base reference and the epoch-0 validation point.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/tests/test_basic_retrieval_training.py`:

````diff-python
diff --git asource/activation/tests/test_basic_retrieval_training.py bworkspace/activation/tests/test_basic_retrieval_training.py
index b16dbe7..6daf478 100644
--- asource/activation/tests/test_basic_retrieval_training.py
+++ bworkspace/activation/tests/test_basic_retrieval_training.py
@@ -9,6 +9,7 @@ from activation.dataset.loaders import MsMarcoDataset
 from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
 from activation.harness.hf_utils import FREE_DEVICE
 from activation.retrieval import (
+    METRIC_NAMES,
     RetrievalModel,
     RetrievalReporter,
     RetrievalTrainer,
@@ -78,7 +79,8 @@ def test_basic_retrieval_training():
     assert sum(step_losses[-5:]) < 0.5 * sum(step_losses[:5]), step_losses
     reporting_losses = [loss for _progress, loss in stats.reporting_losses]
     assert min(reporting_losses) < reporting_losses[0], reporting_losses
-    assert len(stats.validation_losses) == 4
+    assert len(stats.validation_losses) == 5, "epoch 0 (untrained) plus one per epoch"
+    assert "frozen base" in stats.reference_metrics and set(stats.reference_metrics["frozen base"]) == set(METRIC_NAMES)
     assert os.path.exists(os.path.join(REPORT_FOLDER, "report.tressoir.html"))
     assert os.path.exists(os.path.join(REPORT_FOLDER, "report_data.json"))
````

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/harness/module_manager.py</span>
    <span class="card-oneliner">`LoraConfig` and `ModuleManager`: named PEFT adapters on one wrapped base, AC registry, adapter parameters.</span>
    <span class="card-badge">Unchanged</span>
  </summary>

No change to `activation/harness/module_manager.py`.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/harness/loaded_model.py</span>
    <span class="card-oneliner">`decoder_forward` (decoder without the LM head, inside the module manager's LoRA scope), the free guard, `trust_remote_code` on every loader.</span>
    <span class="card-badge">Unchanged</span>
  </summary>

No change to `activation/harness/loaded_model.py`.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/harness/hf_utils.py</span>
    <span class="card-oneliner">`d_ff` in the description and its pretty print; `make_peft_lora_config`; `trust_remote_code` on config/tokenizer/processor loads.</span>
    <span class="card-badge">Unchanged</span>
  </summary>

No change to `activation/harness/hf_utils.py`.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/harness/model_config.py</span>
    <span class="card-oneliner">`ModelDescription.d_ff`; `ModelConfig.trust_remote_code` (default off).</span>
    <span class="card-badge">Unchanged</span>
  </summary>

No change to `activation/harness/model_config.py`.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/harness/runtime.py</span>
    <span class="card-oneliner">`self.module_manager = ModuleManager(self)`.</span>
    <span class="card-badge">Unchanged</span>
  </summary>

No change to `activation/harness/runtime.py`.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/harness/__init__.py</span>
    <span class="card-oneliner">Exports `ModuleManager` and `LoraConfig`.</span>
    <span class="card-badge">Unchanged</span>
  </summary>

No change to `activation/harness/__init__.py`.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/harness/lora.py</span>
    <span class="card-oneliner">Placeholder removed: LoRA lives in `module_manager.py`.</span>
    <span class="card-badge">Delete</span>
  </summary>

Delete `activation/harness/lora.py`.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/__init__.py</span>
    <span class="card-oneliner">Package exports.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/retrieval/__init__.py`:

````diff-python
diff --git asource/activation/retrieval/__init__.py bworkspace/activation/retrieval/__init__.py
index e6e2e70..1e36961 100644
--- asource/activation/retrieval/__init__.py
+++ bworkspace/activation/retrieval/__init__.py
@@ -1,5 +1,6 @@
 from .retrieval_ac import StandardRetrievalACModel
 from .retrieval_model import RetrievalModel
+from .retrieval_baseline import BaselineEmbedder, frozen_base_reference
 from .retrieval_batching import (
     RetrievalBatch,
     make_batches,
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_ac.py</span>
    <span class="card-oneliner">`StandardRetrievalACModel`: bytes → windowed pooling (per-text window validity, so rows do not depend on batch padding) → bidirectional layers → V view rows in `d_model`.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/retrieval/retrieval_ac.py`:

````diff-python
diff --git asource/activation/retrieval/retrieval_ac.py bworkspace/activation/retrieval/retrieval_ac.py
index f7fa974..4abf8d4 100644
--- asource/activation/retrieval/retrieval_ac.py
+++ bworkspace/activation/retrieval/retrieval_ac.py
@@ -113,7 +113,6 @@ class StandardRetrievalACModel(nn.Module):
         harness: "HarnessRuntime",
         base_model_name: str,
         d_ac_model: int | None = None,      # None: d_model // 4
-        num_prefix_tokens: int | None = 16, # @AI: This now becomes the prompt prefix.
         num_view_tokens: int = 8,
         num_ac_layers: int | None = None,   # None: base layer count // 4
     ):
@@ -122,9 +121,9 @@ class StandardRetrievalACModel(nn.Module):
         self.harness = harness
         self.base_model_name = base_model_name
         self.d_model = description.d_model
-        self.d_ac_model = d_ac_model or 1024
+        self.d_ac_model = d_ac_model or self.d_model // 4
         self.num_view_tokens = num_view_tokens
-        self.num_ac_layers = num_ac_layers or 8
+        self.num_ac_layers = num_ac_layers or max(1, len(description.layer_descriptions) // 4)
         self.input_limit_chars = harness.harness_config.doc_embedding_input_limit_chars
         self.gradient_checkpointing = False
         num_heads = max(1, self.d_ac_model // 64)
@@ -141,7 +140,7 @@ class StandardRetrievalACModel(nn.Module):
             for _ in range(self.num_ac_layers)
         ])
         self.final_norm = nn.LayerNorm(self.d_ac_model)
-        self.view_queries = nn.Parameter(torch.randn(num_view_tokens, self.d_ac_model) * 0.02) # @AI: P+V
+        self.view_queries = nn.Parameter(torch.randn(num_view_tokens, self.d_ac_model) * 0.02)
         self.view_attention = nn.MultiheadAttention(self.d_ac_model, num_heads, dropout=dropout, batch_first=True)
         self.view_norm = nn.LayerNorm(self.d_ac_model)
         self.output_projection = nn.Linear(self.d_ac_model, self.d_model)
@@ -166,10 +165,7 @@ class StandardRetrievalACModel(nn.Module):
         return ids.to(self.device), mask.to(self.device)
 
     def forward(self, texts: list[str]) -> torch.Tensor:
-        """
-        [B, V, d_model] view rows, one group per text.
-        @AI: This should now be P+V (prefix + view).
-        """
+        """[B, V, d_model] view rows, one group per text."""
         ids, mask = self.encode_bytes(texts)
         x = self.byte_embedding(ids)
         x = x + sinusoidal_positions(x.shape[1], x.shape[2], x.device, x.dtype)[None]
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_model.py</span>
    <span class="card-oneliner">`RetrievalModel`: left-padded tokens + EOS + view rows (at embedding scale) through the frozen base with LoRA, head, view-mean MaxSim.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/retrieval/retrieval_model.py`:

````diff-python
diff --git asource/activation/retrieval/retrieval_model.py bworkspace/activation/retrieval/retrieval_model.py
index 4e874e9..824d1fa 100644
--- asource/activation/retrieval/retrieval_model.py
+++ bworkspace/activation/retrieval/retrieval_model.py
@@ -23,7 +23,6 @@ if t.TYPE_CHECKING:
 
 
 DEFAULT_QUERY_INSTRUCTION = "Instruct: Retrieve passages that answer the question\nQuery: "
-# @AI: I think there should be a DOCUMENT_INSTRUCTION like Summarize this document or something like that.
 
 
 class RetrievalModel(nn.Module):
@@ -111,7 +110,7 @@ class RetrievalModel(nn.Module):
         """
         texts = [safe_truncate_embedding_chunk(text, self.input_limit_chars) for text in texts]
         if is_query:
-            texts = [self.query_instruction + text for text in texts] # @AI: Distinguish query and doc?
+            texts = [self.query_instruction + text for text in texts]
         tokenizer = self.loaded_model.tokenizer
         token_rows = tokenizer(texts, add_special_tokens=False, padding=False)["input_ids"]
         token_rows = [row + [self.eos_token_id] for row in token_rows]
@@ -131,11 +130,10 @@ class RetrievalModel(nn.Module):
         padded_ids, attention_mask = padded_ids.to(device), attention_mask.to(device)
 
         with self._autocast():
-            # @AI: Handle prefix and view here.
             view_rows = None
             if self.ac_model is not None:                                               # [B, V, d_model] at embedding scale
                 view_rows = (self.ac_model(texts) * self.view_scale).to(base_dtype)
-            inputs_embeds = embedding_layer(padded_ids)
+            inputs_embeds = embedding_layer(padded_ids)                                 # padding rows are masked out
             if view_rows is not None:
                 inputs_embeds = torch.cat([inputs_embeds[:, :length - num_views], view_rows], dim=1)
             position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min(0)
````

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_batching.py</span>
    <span class="card-oneliner">`RetrievalBatch`, epoch and fixed batches, candidate dedup, forwards cut by a padded-token budget, sizing from the heaviest real batch, the real-batch probe.</span>
    <span class="card-badge">Unchanged</span>
  </summary>

No change to `activation/retrieval/retrieval_batching.py`.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_baseline.py</span>
    <span class="card-oneliner">Reference embedders for the in-batch metrics: `BaselineEmbedder` (a harness embedding model such as Qwen3-Embedding-0.6B behind the `RetrievalModel` scoring interface) and `frozen_base_reference` (the base alone: no LoRA, AC or head).</span>
    <span class="card-badge">New file</span>
  </summary>

Exact delta vs `/source/activation/retrieval/retrieval_baseline.py`:

````diff-python
diff --git aworkspace/activation/retrieval/retrieval_baseline.py bworkspace/activation/retrieval/retrieval_baseline.py
new file mode 100644
index 0000000..8a6d236
--- /dev/null
+++ bworkspace/activation/retrieval/retrieval_baseline.py
@@ -0,0 +1,64 @@
+"""
+Reference embedders for the in-batch metrics: the same validation batches scored by a well-trained
+single-vector embedder (Qwen/Qwen3-Embedding-0.6B by default) and by the frozen base alone, so a
+training curve has a trained reference and a floor even at small scales. Nothing beyond the batch
+is embedded; corpus-level comparisons belong to the slice 2 index.
+"""
+import typing as t
+
+import torch
+
+from ..dataset.dataset_utils import safe_truncate_embedding_chunk
+from ..harness.hf_utils import FREE_DEVICE, TARGET_DEVICE
+from .retrieval_model import DEFAULT_QUERY_INSTRUCTION, RetrievalModel
+
+if t.TYPE_CHECKING:
+    from ..harness import HarnessRuntime
+
+
+class BaselineEmbedder:
+    """
+    A harness embedding model behind the scoring interface of RetrievalModel (embed_batch [B, 1, d],
+    similarity, num_vectors, device, query_instruction, input_limit_chars, ac_model), so the trainer
+    evaluates it with the same batches, loss and metrics as the model under training. Queries get
+    the same instruction prefix (the Qwen3-Embedding "Instruct: ...\\nQuery: " format).
+    """
+    num_vectors = 1
+    ac_model = None
+
+    def __init__(self, harness: "HarnessRuntime", model_name: str, query_instruction: str = DEFAULT_QUERY_INSTRUCTION):
+        self.harness = harness
+        self.model_name = model_name
+        self.loaded_model = harness.loaded_models[model_name]
+        assert self.loaded_model.model_config.model_description.is_embedding_model, f"{model_name} is not an embedding model"
+        self.query_instruction = query_instruction
+        self.input_limit_chars = harness.harness_config.doc_embedding_input_limit_chars
+        self.device = torch.device(TARGET_DEVICE)
+        self.real_tokens_embedded = self.padded_tokens_embedded = self.forwards_embedded = 0
+
+    def embed_batch(self, texts: list[str], is_query: bool) -> torch.Tensor:
+        """[B, 1, d], L2-normalized, on the CPU (the harness embedding path returns there)."""
+        texts = [safe_truncate_embedding_chunk(text, self.input_limit_chars) for text in texts]
+        if is_query:
+            texts = [self.query_instruction + text for text in texts]
+        embeddings = self.loaded_model.simple_vector_embed_many(texts)                 # [B, d] normalized
+        self.forwards_embedded += 1
+        return embeddings.float().unsqueeze(1)
+
+    similarity = staticmethod(RetrievalModel.similarity)
+
+    def free(self) -> None:
+        self.loaded_model.model_to_device(FREE_DEVICE)
+
+
+def frozen_base_reference(harness: "HarnessRuntime", retrieval_model: RetrievalModel) -> RetrievalModel:
+    """
+    The base of the model under training with no LoRA, no AC model and no head: the EOS readout of
+    the frozen language model, L2-normalized. It shares the resident base, so it costs no memory.
+    """
+    reference = RetrievalModel(
+        harness, retrieval_model.base_model_name, d_embedding_result=None, ac_name=None, lora_name=None,
+        query_instruction=retrieval_model.query_instruction,
+    )
+    reference.set_training_mode(False)
+    return reference
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_training_config.py</span>
    <span class="card-oneliner">`RetrievalTrainingConfig` (audited defaults, `baseline_model_name`, `reference_frozen_base`) and `RetrievalTrainingStats` with `summarize()` and the reference results.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/retrieval/retrieval_training_config.py`:

````diff-python
diff --git asource/activation/retrieval/retrieval_training_config.py bworkspace/activation/retrieval/retrieval_training_config.py
index 746938b..b804bca 100644
--- asource/activation/retrieval/retrieval_training_config.py
+++ bworkspace/activation/retrieval/retrieval_training_config.py
@@ -27,6 +27,8 @@ class RetrievalTrainingConfig:
     memory_headroom_fraction: float = 0.1
     reporting_fraction: float = 0.1         # of an epoch
     seed: int = 0
+    baseline_model_name: str | None = None  # a harness embedding model scored on the validation batches as a trained reference
+    reference_frozen_base: bool = True      # also score the frozen base alone (no LoRA, AC or head) as the floor
 
 
 @dataclass
@@ -47,6 +49,8 @@ class RetrievalTrainingStats:
     validation_losses: list[tuple[int, float]] = field(default_factory=list)            # (epoch, loss)
     validation_metrics: list[tuple[int, dict[str, float]]] = field(default_factory=list)  # (epoch, in-batch metrics)
     total_validation_time: float = 0.0
+    reference_losses: dict[str, float] = field(default_factory=dict)                    # {reference name: validation loss}
+    reference_metrics: dict[str, dict[str, float]] = field(default_factory=dict)        # {reference name: in-batch metrics}
     peak_memory_bytes: int = 0
     batch_sizing: str = ""                                                              # the printed arithmetic
 
@@ -99,4 +103,6 @@ class RetrievalTrainingStats:
             "last_reporting_metrics": self.reporting_metrics[-1][1] if self.reporting_metrics else None,
             "validation_losses": list(self.validation_losses),
             "validation_metrics": list(self.validation_metrics),
+            "reference_losses": dict(self.reference_losses),
+            "reference_metrics": dict(self.reference_metrics),
         }
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_trainer.py</span>
    <span class="card-oneliner">The trainer: form-A loss, in-batch rank-1 / MRR@10 / nDCG@10, no-decay groups, references scored on the validation batches before training, epoch-0 point, validation at epoch end, report_* calls only.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/retrieval/retrieval_trainer.py`:

````diff-python
diff --git asource/activation/retrieval/retrieval_trainer.py bworkspace/activation/retrieval/retrieval_trainer.py
index 478edc2..e0a0ab4 100644
--- asource/activation/retrieval/retrieval_trainer.py
+++ bworkspace/activation/retrieval/retrieval_trainer.py
@@ -39,6 +39,7 @@ from .retrieval_batching import (
     probe_batch_size,
     recommended_batch_size,
 )
+from .retrieval_baseline import BaselineEmbedder, frozen_base_reference
 from .retrieval_model import RetrievalModel
 from .retrieval_reporter import METRIC_NAMES, RetrievalReporter
 from .retrieval_training_config import RetrievalTrainingConfig, RetrievalTrainingStats
@@ -124,7 +125,10 @@ class RetrievalTrainer:
 
     @torch.no_grad()
     def _evaluate(self, retrieval_model: RetrievalModel, batches: list[RetrievalBatch], temperature: float) -> tuple[float, dict[str, float]]:
-        """Example-weighted loss and in-batch metrics over the given batches, in eval mode."""
+        """
+        Example-weighted loss and in-batch metrics over the given batches, in eval mode. Any embedder
+        with the RetrievalModel scoring interface works here, including a BaselineEmbedder.
+        """
         total_loss, total_examples = 0.0, 0
         totals = {name: 0.0 for name in METRIC_NAMES}
         for batch in batches:
@@ -137,6 +141,29 @@ class RetrievalTrainer:
         divisor = max(1, total_examples)
         return total_loss / divisor, {name: value / divisor for name, value in totals.items()}
 
+    def evaluate_references(
+        self, config: RetrievalTrainingConfig, retrieval_model: RetrievalModel, batches: list[RetrievalBatch],
+    ) -> dict[str, tuple[float, dict[str, float]]]:
+        """
+        {reference name: (loss, in-batch metrics)} on the given batches for the references the config
+        asks for: the frozen base alone (the floor) and a well-trained baseline embedder (the target).
+        The baseline model is loaded for the pass and freed afterwards; the frozen base shares the
+        resident base. Same batches, pool, loss and metrics as the model under training.
+        """
+        references: dict[str, tuple[float, dict[str, float]]] = {}
+        if not batches:
+            return references
+        if config.reference_frozen_base:
+            reference = frozen_base_reference(self.harness, retrieval_model)
+            references["frozen base"] = self._evaluate(reference, batches, config.temperature)
+        if config.baseline_model_name:
+            baseline = BaselineEmbedder(self.harness, config.baseline_model_name, retrieval_model.query_instruction)
+            try:
+                references[config.baseline_model_name] = self._evaluate(baseline, batches, config.temperature)
+            finally:
+                baseline.free()
+        return references
+
     # ----------------------------------------------------------------------------- training
     def _dataset_indexes(self, *example_lists: list[LabeledRetrievalQAExample]) -> DatasetIndexes:
         dataset_ids = {example.dataset_id for examples in example_lists for example in examples}
@@ -177,11 +204,12 @@ class RetrievalTrainer:
         retrieval_reporter: RetrievalReporter,
         training_data: list[LabeledRetrievalQAExample],
         reporting_data: list[LabeledRetrievalQAExample],
-        validation_data: list[LabeledRetrievalQAExample] | None = None, # @AI: Remove from here. Move to eval. Reporting is enough here.
+        validation_data: list[LabeledRetrievalQAExample] | None = None,
     ) -> RetrievalTrainingStats:
         """
         Holds base residency for the whole run. Order: ensure_resident -> batch size (config or
-        recommended) -> probe -> epochs. Raises before the first real step if the probe cannot fit.
+        recommended) -> probe -> references (frozen base, baseline embedder) -> epoch-0 point ->
+        epochs. Raises before the first real step if the probe cannot fit.
         """
         config = training_config
         reporter = retrieval_reporter
@@ -233,7 +261,17 @@ class RetrievalTrainer:
             "batch_size": batch_size,
         }
         lora_config = self.harness.module_manager.get_lora_config(retrieval_model.lora_name) if retrieval_model.lora_name else None
-        reporter.initialize_run(config, retrieval_model, lora_config, counts, batch_sizing, groups)
+        reference_names = (["frozen base"] if config.reference_frozen_base else []) + ([config.baseline_model_name] if config.baseline_model_name else [])
+        reporter.initialize_run(config, retrieval_model, lora_config, counts, batch_sizing, groups, reference_names)
+
+        # References on the validation batches (the reporting batch when there is no validation split).
+        reference_batches = validation_batches or reporting_batches
+        start = time.time()
+        for name, (loss, metrics) in self.evaluate_references(config, retrieval_model, reference_batches).items():
+            stats.reference_losses[name], stats.reference_metrics[name] = loss, metrics
+            reporter.report_reference(name, loss, metrics)
+        stats.total_validation_time += time.time() - start
+        retrieval_model.set_training_mode(True, config.gradient_checkpointing)
         if device.type == "cuda":
             torch.cuda.reset_peak_memory_stats(device)
 
@@ -252,9 +290,24 @@ class RetrievalTrainer:
             stats.reporting_metrics.append((progress, metrics))
             reporter.report_reporting_point(step, progress, loss, metrics)
 
+        def validate(epoch: int) -> None:
+            if not validation_batches:
+                return
+            start = time.time()
+            retrieval_model.set_training_mode(False)
+            validation_loss, validation_metrics = self._evaluate(retrieval_model, validation_batches, config.temperature)
+            retrieval_model.set_training_mode(True, config.gradient_checkpointing)
+            stats.total_validation_time += time.time() - start
+            stats.validation_losses.append((epoch, validation_loss))
+            stats.validation_metrics.append((epoch, validation_metrics))
+            reporter.report_validation(epoch, validation_loss, validation_metrics)
+
         run_start = time.time()
         step = 0
         last_report_step = -1
+        report_progress(0, 0.0)                                                       # the untrained model: epoch 0 of every curve
+        validate(0)
+        reporter.render(force=True)
         for epoch in range(1, config.epochs + 1):
             rng = random.Random(config.seed + epoch)
             epoch_start = time.time()
@@ -304,15 +357,7 @@ class RetrievalTrainer:
             if last_report_step != step:
                 report_progress(step, step / steps_per_epoch)
                 last_report_step = step
-            if validation_batches:
-                start = time.time()
-                retrieval_model.set_training_mode(False)
-                validation_loss, validation_metrics = self._evaluate(retrieval_model, validation_batches, config.temperature)
-                retrieval_model.set_training_mode(True, config.gradient_checkpointing)
-                stats.total_validation_time += time.time() - start
-                stats.validation_losses.append((epoch, validation_loss))
-                stats.validation_metrics.append((epoch, validation_metrics))
-                reporter.report_validation(epoch, validation_loss, validation_metrics)
+            validate(epoch)
             reporter.report_epoch(str(epoch), epoch_steps, epoch_shapes, epoch_time, peak_memory(), running=False)
             reporter.render(force=True)
 
@@ -321,12 +366,3 @@ class RetrievalTrainer:
         retrieval_model.set_training_mode(False)
         reporter.report_finished(step, time.time() - run_start)
         return stats
-
-
-    def eval(
-        self,
-        retrieval_model: RetrievalModel,
-        retrieval_reporter: RetrievalReporter,
-        eval_data: list[LabeledRetrievalQAExample] | None = None, # @AI: can be validation data or test data.
-    ):
-        pass # @AI: Implement me.
\ No newline at end of file
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_reporter.py</span>
    <span class="card-oneliner">`RetrievalReporter(HtmlReporter)`: the run's widget layout, the reference lines and table, and the report_* events the trainer calls.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/retrieval/retrieval_reporter.py`:

````diff-python
diff --git asource/activation/retrieval/retrieval_reporter.py bworkspace/activation/retrieval/retrieval_reporter.py
index 91255b3..f78d889 100644
--- asource/activation/retrieval/retrieval_reporter.py
+++ bworkspace/activation/retrieval/retrieval_reporter.py
@@ -48,11 +48,16 @@ class RetrievalReporter(HtmlReporter):
         counts: dict,
         batch_sizing: str,
         groups: dict[str, list],
+        reference_names: list[str] = (),
     ) -> None:
         """Create every widget of the run and write the first page."""
         self.total_steps = counts["total_steps"]
         self.epochs = config.epochs
         reporting_examples = counts["reporting_examples"]
+        reference_note = ""
+        if reference_names:
+            reference_note = (" Flat lines: " + ", ".join(reference_names) + " scored on the same validation batches before "
+                              "training (the frozen base alone is the floor, a well-trained embedder the target).")
         self.initialize_line_plot(
             "reporting", "Reporting and validation loss",
             f"Reporting: the fixed {reporting_examples}-query batch every {config.reporting_fraction:.0%} of an epoch. "
@@ -65,10 +70,18 @@ class RetrievalReporter(HtmlReporter):
             "Rank-1 (top candidate is an own positive), MRR@10 and nDCG@10 (binary relevance over own positives), all "
             "in-batch: each query is ranked against the batch's distinct candidates (its own positives and hard negatives "
             "plus the other queries' candidates) with same-document collisions masked, not against the corpus. "
-            "The reporting batch at every reporting point, the validation split at each epoch end.",
+            "The reporting batch at every reporting point, the validation split at each epoch end (epoch 0: untrained)."
+            + reference_note,
             "epochs", "metric",
-            [f"reporting {metric}" for metric in METRIC_NAMES] + [f"validation {metric}" for metric in METRIC_NAMES],
+            [f"reporting {metric}" for metric in METRIC_NAMES] + [f"validation {metric}" for metric in METRIC_NAMES]
+            + [f"{name} {metric}" for name in reference_names for metric in METRIC_NAMES],
         )
+        if reference_names:
+            self.initialize_table(
+                "references", "Reference embedders",
+                "Loss and in-batch metrics on the validation batches (same pool, loss and metrics as the model under training).",
+                ["reference", "loss", *METRIC_NAMES],
+            )
         self.initialize_line_plot(
             "step_loss", "Training loss per step", "Raw per-step loss (faint) with a 25-step moving average.",
             "step", "loss", ["loss"], smoothing_window=25,
@@ -124,6 +137,13 @@ class RetrievalReporter(HtmlReporter):
         self.set_status(last_reporting_loss=f"{loss:.3f}", **{f"reporting_{name}": f"{value:.2f}" for name, value in metrics.items()})
         print(f"Step {step}/{self.total_steps} (epoch {progress:.2f}): reporting loss {loss:.4f}, {_format_metrics(metrics)}")
 
+    def report_reference(self, name: str, loss: float, metrics: dict[str, float]) -> None:
+        """One flat line per metric across the whole run, plus a table row."""
+        for x in (0.0, float(self.epochs)):
+            self.add_data_point("metrics", {"x": x, **{f"{name} {metric}": value for metric, value in metrics.items()}})
+        self.add_data_point("references", {"reference": name, "loss": f"{loss:.4f}", **{metric: f"{value:.3f}" for metric, value in metrics.items()}})
+        print(f"Reference {name}: validation loss {loss:.4f}, {_format_metrics(metrics)}")
+
     def report_validation(self, epoch: int, loss: float, metrics: dict[str, float]) -> None:
         self.add_data_point("reporting", {"x": float(epoch), "validation": loss})
         self.add_data_point("metrics", {"x": float(epoch), **{f"validation {name}": value for name, value in metrics.items()}})
````

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/common/reporting.py</span>
    <span class="card-oneliner">`HtmlReporter`: named widgets, the self-contained Tressoir page with pinned HTTPS assets, atomic (mode 644) `report.tressoir.html` + `report_data.json`.</span>
    <span class="card-badge">Unchanged</span>
  </summary>

No change to `activation/common/reporting.py`.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_dense_index.py</span>
    <span class="card-oneliner">Unchanged placeholder (slice 2).</span>
    <span class="card-badge">Unchanged</span>
  </summary>

No change to `activation/retrieval/retrieval_dense_index.py`.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_manager.py</span>
    <span class="card-oneliner">`select_training_data`: filters, label inheritance with an empty pool, seeded split, minimums, reporting subset.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/dataset_manager.py`:

````diff-python
diff --git asource/activation/dataset/dataset_manager.py bworkspace/activation/dataset/dataset_manager.py
index ff17f45..7313c35 100644
--- asource/activation/dataset/dataset_manager.py
+++ bworkspace/activation/dataset/dataset_manager.py
@@ -55,20 +55,16 @@ class DatasetManager:
         return
 
     
-    def synthesize_study_examples_qa(self, dataset_id: str, num_samples: int, caching_id: str|None=None):
+    def synthesize_study_examples_qa(self, dataset_id: str, num_samples: int):
         """Synthesize study examples"""
         study_generator = self._get_or_create_study_generator(dataset_id)
         return study_generator.generate_examples_qa(num_samples)
 
-    def synthesize_study_examples_description(self, args): # @AI: Implement me. includes caching id.
-        pass
-
     def label_study_examples(
         self,
         dataset_id: str,
         num_samples: int,
-        origins: list[str]|None, # None=all origins. @AI: Propagate this.
-        caching_id: str|None = None
+        synthetic_only: bool = False,
     ):
         study_generator = self._get_or_create_study_generator(dataset_id)
         return study_generator.generate_examples_labels(num_samples, synthetic_only)
@@ -78,7 +74,7 @@ class DatasetManager:
         self,
         dataset_id: str,
         num_samples: int,
-        origins: list[str]|None, # @AI Also propagate.
+        synthetic_only: bool = False,
         oracle_labeled_only: bool = True,
         val_ratio: float = 0.1,
         max_reporting_size: int = 50,
@@ -99,8 +95,6 @@ class DatasetManager:
         where the dataset carries native hard negatives, one chunk per hard-negative document.
         Examples without a positive chunk are dropped and counted in the dataset stats.
         """
-        # @AI: Move into dataset_study.py.
-        # Double-check: either qa-gen alone, or description-gen alone is enough to re-use 
         loaded_dataset = self.loaded_datasets[dataset_id]
         stats = loaded_dataset.stats
         candidates = [
@@ -143,11 +137,3 @@ class DatasetManager:
             f"{len(reporting_data)} reporting examples."
         )
         return training_data, validation_data, reporting_data
-
-
-    def select_testing_data(
-        self,
-        dataset_id: str,
-        num_samples: int|None = None, # None=all
-    ):
-        pass # @AI: Implement me.
\ No newline at end of file
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset.py</span>
    <span class="card-oneliner">`DatasetStats.training_select_num_dropped_no_positive`.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/dataset.py`:

````diff-python
diff --git asource/activation/dataset/dataset.py bworkspace/activation/dataset/dataset.py
index bfcdead..19c4ab3 100644
--- asource/activation/dataset/dataset.py
+++ bworkspace/activation/dataset/dataset.py
@@ -7,9 +7,7 @@ class DataOrigin(StrEnum):
     """Who authored one registerd labeled example"""
     NATIVE = auto()
     EXTERNAL = auto()
-    SYNTHETIC_QA = auto()
-    SYNTHETIC_DESCRIPTION = auto()
-    PROGRAMMATIC = auto()
+    SYNTHETIC = auto()
 
 
 class DataSplit(StrEnum):
@@ -244,10 +242,7 @@ class LoadedDataset:
     """List of documents."""
 
     labeled_retrieval_examples: dict[str, LabeledRetrievalQAExample] = field(default_factory=dict)
-    """List of labeled retrieval examples."""
-
-    programmatic_retrieval_examples: dict[str, LabeledRetrievalQAExample] = field(default_factory=dict)
-    """List of programmatic retrieval examples. Separated due to potentially large volume."""
+    """List of labeled retrieval examples"""
 
     labeled_messages_examples: dict[str, LabeledMessagesExample] = field(default_factory=dict)
     """List of labeled messages."""
````

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/dataset_utils.py</span>
    <span class="card-oneliner">`shuffle_fill_truncate`: whole reshuffled passes until the budget is covered, then truncate (replaces sampling with replacement).</span>
    <span class="card-badge">Unchanged</span>
  </summary>

No change to `activation/dataset/dataset_utils.py`.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_study.py</span>
    <span class="card-oneliner">`_sample_study_chunks` uses `shuffle_fill_truncate`.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/dataset_study.py`:

````diff-python
diff --git asource/activation/dataset/dataset_study.py bworkspace/activation/dataset/dataset_study.py
index 3876d1f..05d74a7 100644
--- asource/activation/dataset/dataset_study.py
+++ bworkspace/activation/dataset/dataset_study.py
@@ -53,9 +53,9 @@ LABEL_JSON_SCHEMA = {
 
 
 
-def make_study_qa_prompt(study_context: str | None) -> tuple[str, str, dict]:
+def make_study_prompt(study_context: str | None) -> tuple[str, str, dict]:
     """
-    Returns system prompt, instructions, json schema for study q/a tasks.
+    Returns system prompt, instructions, json schema.
     """
     system_prompt = """
 # Core Guidelines
@@ -63,12 +63,9 @@ def make_study_qa_prompt(study_context: str | None) -> tuple[str, str, dict]:
 - You are given a snippet from the corpus students are expected to study and understand.
 - From this, your goal is to generate a hard question (unanswerable with common knowledge or simple keyword look ups).
     - This is IMPORTANT: the question should be **hard**, or it won't test the student's learning.
-    - There should be a reasonably high lexical distance to prevent simple keyword lookups.
     - The question should have an accompanying answer.
 - Your priority is: a hard question along with its expected, correct answer.
 - By default, keep your question short to medium-short unless the exam recommends longer questions.
-- Other notes:
-    - Avoid things like "according to the text". Ask the question as is.
 """
 
     if study_context:
@@ -99,57 +96,7 @@ Generate the exam-like question for student's study.
     return system_prompt, instructions, STUDY_JSON_SCHEMA
 
 
-def make_study_description_prompt(study_context: str | None) -> tuple[str, str, dict]:
-    """
-    Returns system prompt, instructions, json schema for study description tasks.
-    """
-    system_prompt = """
-# Core Guidelines
-You perform a search-friendly summarization.
-- You are given a snippet from a corpus.
-- You make summary that makes it easy for students to perform various kinds of search for semantic information retrieval.
-
-# What a good summarization does
-- It bridges an abstraction gap:
-    - The snippet contains the concrete information like:
-        - The code for specific algorithm.
-        - A specific biological process.
-        - An agentic trace.
-    - The students search more abstract terms like:
-        - Efficient sorting algorithms with custom comparators.
-        - Biological processes that transform molecule X.
-        - Traces where an agent is stuck on formatting issues.
-    - Notice how the properties the students are looking are clearly from the snippets:
-        - They are just not not explicitly stated as a distinct property yet, making it hard to search for them.
-        - There can also be many such properties. List them all.
-- The description should covers the gap between the two:
-    - It identifies key, specific properties of the document that make such implicit searches easier.
-    - It avoids restating things that already searchable within the documents.
-    - It avoids being so vague the property is likely shared by many other things within the corpus.
-        - It's distinctive: a summary maps strongly to the given snippet.
-- It's relatively concise.
-    
-# What a good description does NOT do
-- It does not summarize or paraphrase the surface content. The raw document is already indexed; restating it adds nothing.
-    - To that end, you are given a boolean for `needs_abstraction_bridge`. When set, you can mark the snippet as not needing an abstraction bridge.
-- It does not invent properties the text cannot support. If a property is ambiguous, omit it rather than guess.
-- It's neither too abstract nor too specific. In case of doubt though, prioritize specifity.
- 
-# Response Format
-Your answer must be formatted as a json object like:
-```json
-{
-    "needs_abstraction_bridge": true|false,
-    "retrieval_summarization": "..."|null
-}
-```
-"""
-    if study_context:
-        # The following emphasizes what descriptions should aim to extractio.
-        pass
-    # @AI: Implement me.
-
-def make_label_prompt(for_qa: bool = True) -> tuple[str, str, dict]:
+def make_label_prompt() -> tuple[str, str, dict]:
     """
     Returns system prompt, instructions and json schema for the snippet labeling task.
     """
@@ -199,8 +146,6 @@ class DatasetStudyGenerator:
         self.study_seed = harness.harness_config.dataset_study_seed
         self.label_top_k = harness.harness_config.dataset_study_label_top_k
         self.label_pool_max_chars = harness.harness_config.dataset_study_label_pool_max_chars
-        self.programmatic_min_chars = harness.harness_config.dataset_study_programmatic_min_chars
-        self.programmatic_max_chars = harness.harness_config.dataset_study_programmatic_max_chars
         assert self.dataset_index.bm25_index is not None, "Study requires built bm25 indexes."
 
 
@@ -219,16 +164,13 @@ class DatasetStudyGenerator:
         return chat_kwargs
 
 
-    def generate_examples_descriptions(self, num_samples: int) -> list[LabeledRetrievalQAExample]:
-        pass # @AI: Refactor with generate_examples_qa. Only the prompt/parsing should differ. Uses the qa model.
-
     def generate_examples_qa(self, num_samples: int) -> list[LabeledRetrievalQAExample]:
         loaded_model = self.harness.loaded_models[self.qa_model_name]
         # vllm batches a whole conversation list inside one chat() call, so a
         # simple single-threaded loop needs no locks.
         loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
         study_samples = self._sample_study_chunks(num_samples)
-        system_prompt, instructions, json_schema = make_study_qa_prompt(self.study_context)
+        system_prompt, instructions, json_schema = make_study_prompt(self.study_context)
         chat_kwargs = self._make_engine_chat_kwargs(json_schema)
         stats = self.loaded_dataset.stats
         examples: list[LabeledRetrievalQAExample] = []
@@ -241,7 +183,6 @@ class DatasetStudyGenerator:
             for chunk in batch:
                 snippet = self.dataset_index.get_chunk_section(chunk)
                 snippet = safe_truncate_embedding_chunk(snippet, self.chunk_input_limit)
-                # @AI: Move to prompt file.
                 conversations.append([
                     {
                         "role": "system", "content": [
@@ -450,8 +391,6 @@ class DatasetStudyGenerator:
                     f"[Index={index}]\n```\n{snippet}\n```"
                     for index, (_, snippet) in enumerate(pool)
                 )
-                # @AI: Support the case with query only: Those should say: Description.
-                # Also move into prompt file.
                 user_text = (
                     f"# Question\n{example.query}\n\n"
                     f"# Reference Answer\n{example.gold_answers[0]}\n\n"
@@ -494,13 +433,3 @@ class DatasetStudyGenerator:
                 reporting_interval = max(1, len(example_pool_pairs) // 20)
             reporting_interval -= len(batch)
         return labeled
-
-
-    def generate_programmatic_examples(self, num_samples: int):
-        """
-        Generate programmatic examples for massive AC Model training.
-        """
-        study_samples = self._sample_study_chunks(num_samples)
-        # @AI: simple programmatic formation.
-        # The chunk is its own positive.
-        # only use the in-batch negatives as negatives.
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/cloud/sky.py</span>
    <span class="card-oneliner">`watch`: rsync a run's remote artifact folder into `IB/TMP/...` every N seconds until a sentinel file arrives.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/cloud/sky.py`:

````diff-python
diff --git asource/activation/cloud/sky.py bworkspace/activation/cloud/sky.py
index 342f489..840948c 100644
--- asource/activation/cloud/sky.py
+++ bworkspace/activation/cloud/sky.py
@@ -614,7 +614,6 @@ def download(
     return _run_rsync(*args).returncode
 
 
-# @AI: Double-check this works with directories too.
 def watch(
     name: str,
     remote_path: str,
@@ -668,7 +667,6 @@ def watch(
         return 0
 
 
-# @AI: make this more practical by allow --watch  etc.
 def exec_cmd(name: str, cmd: str | list[str]) -> int:
     command = [cmd] if isinstance(cmd, str) else cmd.copy()
     if command[:1] == ["--"]:
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/bench/live_report_demo.py</span>
    <span class="card-oneliner">The live-sync demo: a small `HtmlReporter` page rewritten every few seconds, then `done.json`.</span>
    <span class="card-badge">New file</span>
  </summary>

Exact delta vs `/source/activation/bench/live_report_demo.py`:

````diff-python
diff --git aworkspace/activation/bench/live_report_demo.py bworkspace/activation/bench/live_report_demo.py
new file mode 100644
index 0000000..2eff7ac
--- /dev/null
+++ bworkspace/activation/bench/live_report_demo.py
@@ -0,0 +1,49 @@
+"""
+Writes a small live report for a few minutes so `sky watch` can be exercised end to end:
+
+    uv run sky exec <node> uv run python -m activation.bench.live_report_demo --seconds 180
+    uv run sky watch <node> '~/activation_artifacts/live_report_demo/' IB/TMP/LIVE_REPORT_DEMO/ --until-file done.json
+
+The page under the watched folder morphs in the editor as the points arrive; `done.json` is written last.
+"""
+import argparse
+import json
+import math
+import os
+import random
+import time
+
+from activation.common.reporting import HtmlReporter
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--seconds", type=int, default=180)
+    parser.add_argument("--interval", type=float, default=5.0)
+    parser.add_argument("--report-folder", default=os.path.expanduser("~/activation_artifacts/live_report_demo"))
+    args = parser.parse_args()
+    args.report_folder = os.path.expanduser(args.report_folder)
+    reporter = HtmlReporter(args.report_folder, "Live report demo", "A curve that grows while you watch.",
+                            eyebrow="Watch demo", refresh_seconds=10, min_render_interval_seconds=0)
+    reporter.initialize_line_plot("curve", "Decaying noisy curve", "One point per tick.", "tick", "value", ["value"])
+    reporter.initialize_table("ticks", "Ticks", "", ["tick", "time", "value"])
+    rng = random.Random(0)
+    start = time.time()
+    tick = 0
+    while time.time() - start < args.seconds:
+        tick += 1
+        value = math.exp(-tick / 20) + 0.05 * rng.random()
+        reporter.add_data_point("curve", {"x": tick, "value": value})
+        reporter.add_data_point("ticks", {"tick": tick, "time": time.strftime("%H:%M:%S"), "value": f"{value:.3f}"})
+        reporter.set_status(tick=str(tick), elapsed=f"{time.time() - start:.0f}s")
+        reporter.render(force=True)
+        print(f"tick {tick} value {value:.3f}", flush=True)
+        time.sleep(args.interval)
+    reporter.finish()
+    with open(os.path.join(args.report_folder, "done.json"), "w") as handle:
+        json.dump({"ticks": tick, "seconds": time.time() - start}, handle)
+    print("done", flush=True)
+
+
+if __name__ == "__main__":
+    main()
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/bench/cheap_synthetic_bench.py</span>
    <span class="card-oneliner">The CHEAP_SYNTHETIC generator bench: study-prompt QA throughput per (model, concurrency) with a live report.</span>
    <span class="card-badge">New file</span>
  </summary>

Exact delta vs `/source/activation/bench/cheap_synthetic_bench.py`:

````diff-python
diff --git aworkspace/activation/bench/cheap_synthetic_bench.py bworkspace/activation/bench/cheap_synthetic_bench.py
new file mode 100644
index 0000000..0fde34c
--- /dev/null
+++ bworkspace/activation/bench/cheap_synthetic_bench.py
@@ -0,0 +1,186 @@
+"""
+Throughput of small generators for a CHEAP_SYNTHETIC phase: the study prompt on real MS-MARCO
+chunks, one JSON {question, answer} per chunk, measured per model and engine concurrency. Writes
+a live report (watch it with `sky watch`) and `cheap_synthetic_results.json` last.
+
+    uv run sky exec <node> uv run python -m activation.bench.cheap_synthetic_bench \\
+        --model-ids inclusionAI/Ling-3.0-tiny-fp8 RedHatAI/Qwen3.5-4B-FP8-dynamic \\
+        --max-num-seqs 128 256 512 --num-chunks 2048
+    uv run sky watch <node> '~/activation_artifacts/cheap_synthetic/' IB/TMP/CHEAP_SYNTHETIC/run/ \\
+        --until-file cheap_synthetic_results.json
+"""
+import argparse
+import gc
+import json
+import os
+import random
+import time
+import traceback
+
+import torch
+from vllm.sampling_params import StructuredOutputsParams
+
+from activation.common.reporting import HtmlReporter
+from activation.dataset.dataset_study import STUDY_JSON_SCHEMA, make_study_prompt
+from activation.dataset.dataset_utils import safe_truncate_embedding_chunk, shuffle_fill_truncate
+from activation.dataset.loaders import MsMarcoDataset
+from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
+
+COLUMNS = ["model", "max_num_seqs", "structured", "thinking", "requests", "prompt tok/s", "output tok/s",
+           "requests/s", "QA per GPU-hour", "avg output tokens", "parse failures", "engine load", "run time", "note"]
+
+
+def load_chunks(harness: HarnessRuntime, num_chunks: int, seed: int) -> list[str]:
+    dataset = MsMarcoDataset.load(harness, max_examples=max(256, num_chunks // 8), max_corpus_documents=0)
+    index = harness.dataset_manager._get_or_create_index(dataset.dataset_id)
+    chunk_ids = shuffle_fill_truncate(list(index.chunk_ids), num_chunks, random.Random(seed))
+    return [index.get_chunk_section(index.chunks[chunk_id]) for chunk_id in chunk_ids]
+
+
+def conversations_for(snippets: list[str], chunk_input_limit: int) -> list[list[dict]]:
+    system_prompt, instructions, _ = make_study_prompt(None)
+    out = []
+    for snippet in snippets:
+        snippet = safe_truncate_embedding_chunk(snippet, chunk_input_limit)
+        out.append([
+            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
+            {"role": "user", "content": [{"type": "text", "text": f"# Corpus Snippet\n```\n{snippet}\n```\n{instructions}"}]},
+        ])
+    return out
+
+
+def chat_kwargs_for(structured: bool, thinking: bool, seed: int, max_tokens: int) -> dict:
+    sampling = {"max_tokens": max_tokens, "seed": seed, "temperature": 0.7, "top_p": 0.9}
+    if structured:
+        sampling["structured_outputs"] = StructuredOutputsParams(json=STUDY_JSON_SCHEMA)
+    return {"sampling_params": sampling, "chat_template_kwargs": {"enable_thinking": thinking}}
+
+
+def run_config(model_id: str, max_num_seqs: int, structured: bool, thinking: bool, snippets: list[str], args, seed: int) -> dict:
+    engine_kwargs = {"max_num_seqs": max_num_seqs, "kv_cache_dtype": args.kv_cache_dtype}
+    trust_remote_code = model_id.startswith("inclusionAI/")                        # Ling ships its own modeling code
+    if trust_remote_code:
+        engine_kwargs["enable_lora"] = False                                        # vLLM 0.28: BailingMoeV3 has no LoRA support
+    config = HarnessRuntimeConfig(model_configs={
+        model_id: ModelConfig(model_id, model_id, engine_kwargs=engine_kwargs, trust_remote_code=trust_remote_code),
+    })
+    harness = HarnessRuntime(config)
+    loaded = harness.loaded_models[model_id]
+    load_start = time.time()
+    loaded.ensure_engine_loaded()
+    load_time = time.time() - load_start
+    conversations = conversations_for(snippets, args.chunk_input_limit)
+    chat_kwargs = chat_kwargs_for(structured, thinking, seed, args.max_tokens)
+    # Warm-up: one small batch, untimed.
+    loaded.engine_chat_many(conversations[:min(64, len(conversations))], chat_kwargs=chat_kwargs)
+    batch_size = 4 * max_num_seqs
+    prompt_tokens = output_tokens = failures = 0
+    start = time.time()
+    for batch_start in range(0, len(conversations), batch_size):
+        outputs = loaded.engine_chat_many(conversations[batch_start:batch_start + batch_size], chat_kwargs=chat_kwargs)
+        for output in outputs:
+            prompt_tokens += output.prompt_token_count
+            output_tokens += output.output_token_count
+            try:
+                qa = json.loads(output.text)
+                if not str(qa["question"]).strip() or not str(qa["answer"]).strip():
+                    failures += 1
+            except (json.JSONDecodeError, KeyError, TypeError):
+                failures += 1
+    elapsed = time.time() - start
+    sample = outputs[0].text[:400] if outputs else ""
+    loaded.engine_to_device(FREE_DEVICE)
+    del harness
+    gc.collect()
+    torch.cuda.empty_cache()
+    n = len(conversations)
+    return {
+        "model": model_id, "max_num_seqs": max_num_seqs, "structured": structured, "thinking": thinking, "requests": n,
+        "prompt_tokens_per_s": prompt_tokens / elapsed, "output_tokens_per_s": output_tokens / elapsed,
+        "requests_per_s": n / elapsed, "qa_per_gpu_hour": 3600 * (n - failures) / elapsed,
+        "avg_output_tokens": output_tokens / n, "parse_failures": failures, "parse_failure_fraction": failures / n,
+        "engine_load_s": load_time, "run_s": elapsed, "sample_output": sample,
+    }
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--model-ids", nargs="+", default=["inclusionAI/Ling-3.0-tiny-fp8", "RedHatAI/Qwen3.5-4B-FP8-dynamic"])
+    parser.add_argument("--max-num-seqs", nargs="+", type=int, default=[128, 256, 512])
+    parser.add_argument("--structured", choices=["on", "off", "both"], default="on")
+    parser.add_argument("--thinking", choices=["on", "off", "both"], default="off")
+    parser.add_argument("--num-chunks", type=int, default=2048)
+    parser.add_argument("--chunk-input-limit", type=int, default=2048)
+    parser.add_argument("--max-tokens", type=int, default=512)
+    parser.add_argument("--kv-cache-dtype", default="auto")
+    parser.add_argument("--seed", type=int, default=0)
+    parser.add_argument("--report-folder", default="~/activation_artifacts/cheap_synthetic")
+    args = parser.parse_args()
+    args.report_folder = os.path.expanduser(args.report_folder)
+
+    reporter = HtmlReporter(args.report_folder, "Cheap synthetic generators",
+                            f"Study-prompt QA generation on {args.num_chunks} MS-MARCO chunks per configuration, one RTX PRO 6000.",
+                            eyebrow="CHEAP_SYNTHETIC bench", refresh_seconds=15, min_render_interval_seconds=0)
+    reporter.initialize_bar_plot("throughput", "Output tokens per second", "Decode throughput per configuration.", "configuration", "tokens/s", ["output tok/s"])
+    reporter.initialize_bar_plot("qa_rate", "QA pairs per GPU-hour", "Parsed pairs only.", "configuration", "pairs/hour", ["QA/hour"])
+    reporter.initialize_table("results", "Results", "One row per (model, concurrency, structured output, thinking).", COLUMNS)
+    reporter.set_text("plan", "Plan", json.dumps(vars(args), indent=2))
+    reporter.set_status(configurations="0", state="loading chunks")
+    reporter.render(force=True)
+
+    seed_harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={}))
+    snippets = load_chunks(seed_harness, args.num_chunks, args.seed)
+    del seed_harness
+    structured_options = {"on": [True], "off": [False], "both": [True, False]}[args.structured]
+    thinking_options = {"on": [True], "off": [False], "both": [False, True]}[args.thinking]
+    results = []
+    total = len(args.model_ids) * len(args.max_num_seqs) * len(structured_options) * len(thinking_options)
+    done = 0
+    for model_id in args.model_ids:
+        for max_num_seqs in args.max_num_seqs:
+            for structured in structured_options:
+                for thinking in thinking_options:
+                    label = f"{model_id.split('/')[-1]} · {max_num_seqs} · {'json' if structured else 'free'}{' · think' if thinking else ''}"
+                    reporter.set_status(configurations=f"{done} / {total}", state=f"running {label}")
+                    reporter.add_data_point("results", {"model": model_id, "max_num_seqs": max_num_seqs, "structured": structured,
+                                                        "thinking": thinking, "note": "running", "running": True})
+                    reporter.render(force=True)
+                    row_start = time.time()
+                    try:
+                        result = run_config(model_id, max_num_seqs, structured, thinking, snippets, args, args.seed)
+                        note = ""
+                    except Exception as error:  # a load or kernel failure is a result too
+                        traceback.print_exc()
+                        result = {"model": model_id, "max_num_seqs": max_num_seqs, "structured": structured, "thinking": thinking,
+                                  "error": f"{type(error).__name__}: {str(error)[:300]}", "run_s": time.time() - row_start}
+                        note = result["error"]
+                        gc.collect(); torch.cuda.empty_cache()
+                    results.append(result)
+                    done += 1
+                    print(json.dumps(result, indent=1), flush=True)
+                    if "error" not in result:
+                        reporter.add_data_point("throughput", {"label": label, "output tok/s": result["output_tokens_per_s"]})
+                        reporter.add_data_point("qa_rate", {"label": label, "QA/hour": result["qa_per_gpu_hour"]})
+                    reporter.add_data_point("results", {
+                        "model": model_id, "max_num_seqs": max_num_seqs, "structured": structured, "thinking": thinking,
+                        "requests": result.get("requests", ""),
+                        "prompt tok/s": f"{result['prompt_tokens_per_s']:.0f}" if "prompt_tokens_per_s" in result else "",
+                        "output tok/s": f"{result['output_tokens_per_s']:.0f}" if "output_tokens_per_s" in result else "",
+                        "requests/s": f"{result['requests_per_s']:.1f}" if "requests_per_s" in result else "",
+                        "QA per GPU-hour": f"{result['qa_per_gpu_hour']:,.0f}" if "qa_per_gpu_hour" in result else "",
+                        "avg output tokens": f"{result['avg_output_tokens']:.0f}" if "avg_output_tokens" in result else "",
+                        "parse failures": f"{result['parse_failure_fraction']:.1%}" if "parse_failure_fraction" in result else "",
+                        "engine load": f"{result['engine_load_s']:.0f} s" if "engine_load_s" in result else "",
+                        "run time": f"{result['run_s']:.0f} s", "note": note,
+                    })
+                    reporter.set_status(configurations=f"{done} / {total}", state="between configurations")
+                    reporter.render(force=True)
+    reporter.set_text("samples", "Sample outputs", "\n\n".join(f"## {r['model']} · {r['max_num_seqs']}\n{r.get('sample_output', r.get('error', ''))}" for r in results))
+    reporter.set_status(state="finished")
+    reporter.finish()
+    with open(os.path.join(args.report_folder, "cheap_synthetic_results.json"), "w") as handle:
+        json.dump({"args": vars(args), "results": results}, handle, indent=1)
+
+
+if __name__ == "__main__":
+    main()
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/bench/retrieval_training_bench.py</span>
    <span class="card-oneliner">The acceptance-run script (1000 / 50, 2 epochs by default; `--baseline-model-id` Qwen3-Embedding-0.6B as the trained reference; the 50k shape with two flags).</span>
    <span class="card-badge">New file</span>
  </summary>

Exact delta vs `/source/activation/bench/retrieval_training_bench.py`:

````diff-python
diff --git aworkspace/activation/bench/retrieval_training_bench.py bworkspace/activation/bench/retrieval_training_bench.py
new file mode 100644
index 0000000..019a05d
--- /dev/null
+++ bworkspace/activation/bench/retrieval_training_bench.py
@@ -0,0 +1,90 @@
+"""
+Slice-1 acceptance run: train the LoRA + activation-context retrieval model on MS-MARCO and write
+the live report. The 1000-query / 50-reporting run on one GPU is the acceptance shape; the same
+script with 50k queries and one epoch is the production shape.
+
+    uv run python -m activation.bench.retrieval_training_bench --num-queries 1000 --epochs 2 \
+        --report-folder ~/activation_artifacts/retrieval_slice1
+"""
+import argparse
+import json
+import os
+import time
+
+import torch
+
+from activation.dataset.loaders import MsMarcoDataset
+from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
+from activation.retrieval import (
+    RetrievalModel,
+    RetrievalReporter,
+    RetrievalTrainer,
+    RetrievalTrainingConfig,
+    StandardRetrievalACModel,
+)
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
+    parser.add_argument("--base-model-id", default="Qwen/Qwen3-0.6B")
+    parser.add_argument("--baseline-model-id", default="Qwen/Qwen3-Embedding-0.6B",
+                        help="well-trained embedder scored on the validation batches as a reference; 'none' to skip")
+    parser.add_argument("--num-queries", type=int, default=1000, help="queries selected before the validation split")
+    parser.add_argument("--val-ratio", type=float, default=0.05)
+    parser.add_argument("--reporting-size", type=int, default=50)
+    parser.add_argument("--epochs", type=int, default=2)
+    parser.add_argument("--batch-size", type=int, default=None, help="examples per step; default: sized for the GPU")
+    parser.add_argument("--lora-rank", type=int, default=128)
+    parser.add_argument("--num-view-tokens", type=int, default=8)
+    parser.add_argument("--d-embedding-result", type=int, default=None)
+    parser.add_argument("--chunk-size-chars", type=int, default=4096)
+    parser.add_argument("--report-folder", default=os.path.expanduser("~/activation_artifacts/retrieval_slice1"))
+    parser.add_argument("--seed", type=int, default=0)
+    args = parser.parse_args()
+    args.report_folder = os.path.expanduser(args.report_folder)                       # a quoted "~" reaches us unexpanded
+
+    base_model_name = args.base_model_id.split("/")[-1].lower()
+    lora_name, ac_name = f"{base_model_name}-retrieval-lora", f"{base_model_name}-retrieval-ac"
+    model_configs = {base_model_name: ModelConfig(base_model_name, args.base_model_id)}
+    baseline_model_name = None
+    if args.baseline_model_id.lower() != "none":
+        baseline_model_name = args.baseline_model_id.split("/")[-1].lower()
+        model_configs[baseline_model_name] = ModelConfig(baseline_model_name, args.baseline_model_id)
+    harness = HarnessRuntime(HarnessRuntimeConfig(
+        model_configs=model_configs,
+        doc_chunk_size_chars=args.chunk_size_chars,
+        doc_embedding_input_limit_chars=args.chunk_size_chars,
+    ))
+    # Rows without a selected passage yield no example: load extra rows so num_queries are labeled.
+    load_start = time.time()
+    dataset = MsMarcoDataset.load(harness, max_examples=int(args.num_queries * 1.25) + 8, max_corpus_documents=0)
+    harness.dataset_manager.build_bm25_indexes()
+    training_data, validation_data, reporting_data = harness.dataset_manager.select_training_data(
+        dataset.dataset_id, num_samples=args.num_queries, oracle_labeled_only=False,
+        val_ratio=args.val_ratio, max_reporting_size=args.reporting_size, seed=args.seed,
+    )
+    print(f"Data ready in {time.time() - load_start:.1f}s: {json.dumps(dataset.stats.summarize(), indent=1)}")
+
+    harness.module_manager.register_lora(lora_name, base_model_name, rank=args.lora_rank)
+    torch.manual_seed(args.seed)                                                       # seeded AC and head init
+    ac_model = StandardRetrievalACModel(harness, base_model_name, num_view_tokens=args.num_view_tokens)
+    harness.module_manager.register_retrieval_ac(ac_name, base_model_name, ac_model)
+    retrieval_model = RetrievalModel(
+        harness, base_model_name, d_embedding_result=args.d_embedding_result, ac_name=ac_name, lora_name=lora_name,
+    )
+    config = RetrievalTrainingConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, baseline_model_name=baseline_model_name)
+    reporter = RetrievalReporter(
+        args.report_folder,
+        title=f"Retrieval training — MS-MARCO train, {len(training_data):,} queries",
+        description=f"{args.base_model_id} frozen + LoRA r{args.lora_rank} + AC (V={args.num_view_tokens}), "
+                    f"{args.epochs} epochs, {len(validation_data)} validation / {len(reporting_data)} reporting queries.",
+    )
+    stats = RetrievalTrainer(harness).train(config, retrieval_model, reporter, training_data, reporting_data, validation_data)
+    summary = stats.summarize()
+    print(json.dumps(summary, indent=2))
+    with open(os.path.join(args.report_folder, "training_stats.json"), "w") as handle:
+        json.dump({"summary": summary, "harness": harness.harness_stats.summarize()}, handle, indent=1)
+
+
+if __name__ == "__main__":
+    main()
````

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">pyproject.toml</span>
    <span class="card-oneliner">`peft` dependency.</span>
    <span class="card-badge">Unchanged</span>
  </summary>

No change to `pyproject.toml`.

</details>
