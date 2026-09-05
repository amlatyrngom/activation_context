# Q4 engine round — quantized frozen oracles on vLLM

This round answers "how do we run a Q4 Qwen3.8-27B through `LoadedModel`" with live GPU probes, adds the `engine_args` override you asked for, and — along the way — removes the SkyPilot image-staleness checks per your direction. Two Q4 checkpoints are validated end to end on an L40S; the dead ends are documented so they don't get re-explored.

## Findings

- **Engine-only models never touch the HF path.** A vLLM-only `LoadedModel` loads config + tokenizer at init and nothing else; `engine_to_device`'s `model_to_device(FREE_DEVICE)` guard no-ops. Quantized repos therefore work without any HF-side accommodation.
- **`Qwen/Qwen3.8-27B` is `Qwen3_5ForConditionalGeneration`**: multimodal, hybrid 48 linear-attention + 16 full-attention layers, 262k context. Registered in vLLM 0.28 with LoRA support; our `canonical_layer_type`/`get_text_config` parsing handles it as-is.
- **Two Q4 checkpoints validated on L40S (48GB), LoRA enabled, `'Hello, World!'` both:**
  - `cyankiwi/Qwen3.8-27B-AWQ-INT4` — compressed-tensors W4A16, ~21GB (bf16 vision tower included).
  - `btbtyler09/Qwen3.8-27B-GPTQ-4bit` — GPTQ g32 via `gptq_marlin`, ~21GB.
- **Two required `engine_args`:** `max_model_len` (the native 262k blows the KV budget) and `max_num_seqs: 64` — hybrid models pin one Mamba cache block per decode sequence, and vLLM's default 256 exceeds the 191 blocks that fit beside the weights on 48GB.
- **Dead ends on vLLM 0.28:** unsloth's Q4 artifacts cannot load — upstream removed both the `bitsandbytes` and `gguf` quantization paths (the bnb probe failed at config validation, and `gguf` is absent from the load-format/quant registries). `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` hits a quark-loader bug (`AttributeError: 'dict' object has no attribute 'endswith'` in the weight-name mapper) at engine init.
- **FP4 looks probeable next** on `rtx6000pro` (Blackwell): the FP4 family survived in 0.28 (`modelopt_fp4`, `nvfp4_per_token`, `mxfp4`), and `unsloth/Qwen3.8-27B-NVFP4` (compressed-tensors, 2.6M downloads) plus `RadixArk/Qwen3.8-27B-NVFP4` (modelopt) exist. Held pending your go.
- Official FP8 repos (`Qwen/Qwen3.8-27B-FP8`) remain the zero-friction quant for engine-only models: auto-detected, no flags, no libraries.

## Harness delta

`activation/harness/model_config.py`

```diff-python
@@
     dtype: torch.dtype = torch.bfloat16
     """Model data type."""

+    engine_args: dict[str, t.Any] = field(default_factory=dict)
+    """Extra vllm.LLM kwargs. Overrides the harness defaults (e.g. enable_lora,
+    max_model_len) for engine-only models such as quantized frozen oracles."""
+
     model_description: ModelDescription|None = None
```

`activation/harness/loaded_model.py`

```diff-python
@@
             self.model_to_device(FREE_DEVICE) # Avoid dual load.
-            self.vllm_model = vllm.LLM(
+            engine_kwargs: dict[str, t.Any] = dict(
                 model=self.model_config.model_id,
                 max_loras=4,
                 max_lora_rank=self.harness.harness_config.max_lora_rank,
                 enable_lora=True,
                 dtype=self.model_config.dtype or "auto",
             )
+            engine_kwargs.update(self.model_config.engine_args)
+            self.vllm_model = vllm.LLM(**engine_kwargs)
```

`activation/tests/test_q4_engine.py` is new (staged): a `@pytest.mark.gpu` probe mirroring `test_basic_engine` over `Q4_MODELS`, with the two required `engine_args` and the dead-end notes inline. Validated: **passed in 8m16s** on the warm cluster (both models loaded, generated, and freed sequentially).

## SkyPilot delta — image demoted to warm cache

Per your direction, the lock-digest staleness checks are gone; correctness now comes from rsync + on-node `uv run` syncing the baked env to the uploaded `uv.lock`.

`activation/cloud/Dockerfile`

```diff-dockerfile
-ENV UV_NO_SYNC=1 \
+ENV UV_FROZEN=1 \
     PATH=/opt/activation/.venv/bin:/usr/local/bin:${PATH}
```

`activation/cloud/sky.py`

```diff-python
@@ def _labels
         f"{LABEL_PREFIX}-gpu-count": str(gpu_count),
-        f"{LABEL_PREFIX}-image": IMAGE_ID,
         f"{LABEL_PREFIX}-disk-gb": str(DISK_SIZE_GB),
@@ def _task_yaml
 setup: |
   test -x /usr/local/cuda/bin/nvcc
-  cmp /opt/activation/image-lock/uv.lock uv.lock || {
-    echo "uv.lock differs from the baked SkyPilot image; rebuild it" >&2
-    exit 1
-  }
   mkdir -p {REMOTE_ARTIFACTS} /root/.cache/huggingface /root/.cache/flashinfer
@@ def exec_cmd
             "env",
             "VLLM_WORKER_MULTIPROC_METHOD=spawn",
+            # The baked env is a warm cache: sync it to the uploaded lock
+            # (old images set UV_NO_SYNC=1) without re-resolving on the node.
+            "UV_NO_SYNC=0",
+            "UV_FROZEN=1",
             f"PYTHONPATH={REMOTE_WORKDIR}",
```

`setup` additionally gained (from the auto-build round plus this one): `_image_published()` + `_latest_published_image()`, so a missing lock-tagged image is built when docker exists (opt out with `--no-image-rebuild`) and otherwise **falls back to the newest published ECR image** with a printed note — docker-less machines can now create clusters. `UV_NO_SYNC=0` (not the empty string, which uv rejects as "not boolish") is what lets the *already-published old image* sync correctly with no rebuild.

All of this was exercised live: the temp cluster `ac-q4-probe` was created from this docker-less container via the fallback path (us-east-1 L40S capacity was exhausted; SkyPilot failed over to us-east-2), and every probe ran through the syncing exec wrapper.

## Operational notes

- **`/workspace/.skyignore` was missing in the agent container** (your source has it); the first upload started mirroring the 8GB local `.venv` before I killed it and restored the file. Worth checking whether the Tressoir copy step skips dotfiles generally.
- The original `ac-loaded-model-tests` cluster was untouched except for failed start attempts (eu-north-1a had no L40S capacity); it remains STOPPED with its disk.
- `ac-q4-probe` (L40S, us-east-2) autostops after 30 idle minutes; its disk keeps the synced env and both Q4 checkpoints cached.

<article class="decision" data-tressoir-decision data-decision-state="unresolved"
  aria-labelledby="q4-cluster-question">
  <header class="decision-header">
    <div>
      <h3 class="decision-title" id="q4-cluster-question">What should happen to the temp cluster ac-q4-probe?</h3>
      <p class="decision-context">It is stopped (disk retained, small cost) with warm caches useful for an FP4 probe or further quant work. Teardown deletes the disk.</p>
    </div>
    <span class="decision-state" data-decision-indicator role="status" aria-live="polite">Unresolved</span>
  </header>
  <fieldset class="decision-options">
    <legend class="visually-hidden">Decision answers</legend>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="q4.cluster.keep">
      <span><strong>Keep it stopped</strong><small>Cheap standby; caches speed up the next quant probe.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="q4.cluster.teardown">
      <span><strong>Tear it down</strong><small>`uv run sky teardown ac-q4-probe`; the FP4 probe would use a fresh rtx6000pro cluster anyway.</small></span>
    </label>
  </fieldset>
  <div class="field decision-feedback">
    <label for="q4-cluster-response">Free Response</label>
    <textarea id="q4-cluster-response" rows="2" data-tressoir-input="q4.cluster.feedback"
      data-tressoir-autogrow="2:6" placeholder="Anything else about cluster lifecycle…"></textarea>
  </div>
</article>

## Deferred

- FP4 probe on `rtx6000pro` (unsloth NVFP4 + RadixArk modelopt NVFP4) — ready to launch on your go.
- A guard in `model_to_device` refusing to HF-load a repo whose config carries `quantization_config`, if you want engine-only models to fail loudly on the HF path.
- Gated `nvidia/Qwen3.8-27B-NVFP4` (needs an HF token in `.env` if you want the official NVFP4).
