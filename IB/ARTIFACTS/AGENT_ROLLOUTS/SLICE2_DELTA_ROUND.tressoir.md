# Agent training, slice 2: the throughput delta

Delta round (2026-09-08), on top of the applied slice-2 handoff (`SLICE2_ROUND.tressoir.md`). Training only; rollouts, tools and the engine protocol are unchanged. Round-0 training of the key test went from 134.6 s (as handed off) to 72.2 s, with identical loss, ratio and gradient statistics, and both the engine and the trainer now accept 52,000-token trajectories.

## What changed, in reading order

1. **Caps at 52k.** `vllm_wrapper.py` recommends `max_model_len` 52,000 for Qwen3.5-4B/9B; `AgentTrainingConfig.max_example_tokens` is 52,000. A 52,000-token example trains in 16 s at 37 GB peak on the RTX 6000 Pro.
2. **The output head.** Its log-softmax chunks are checkpointed (autograd kept 2 GB per 2,048 positions), and its GEMM runs with bf16 operands and fp32 accumulation and output (`torch.mm(out_dtype)`): the same products as the fp32 GEMM, on the tensor cores, without the fp32 copies of the states and the 2.5 GB weight. Log-probs agree with the fp32 GEMM to 5e-5, the state gradient to 3e-4 relative (its backward uses a high and a low bf16 part). `fp32_head_matmul=True` restores the old path.
3. **Checkpointing by length.** Recompute only from `gradient_checkpointing_min_tokens` (default: 6,144 on a card with at least 80 GB, else always): +35% on rows under that, about 9 GB per 1k tokens without recompute.
4. **fla kernel configs.** fla's Triton autotune re-benchmarked its two l2norm kernels for every new sequence-length bucket, about 14 s each, in every process (51 s of a 75 s step). `fla_cache.py` loads tuned configs in fla's fuzzy mode from `fla_configs/<GPU name>/` (checked in for the RTX 6000 Pro, ten kernels). **A new GPU type needs one run of the dump probe** (see the canon note in `IB/CANON/AGENT_TRAINING_CANON.md`); without configs the trainer prints a warning and pays the stalls.
5. **Packed rows, off.** `collate(packed=True)` puts several examples in one row with per-example attention and recurrent state (position resets, `cu_seq_lens` to the gated-delta kernels through `decoder_forward(**kwargs)`, three masked spacer tokens because the gated-delta conv otherwise reads across the boundary). Measured exact, but single rows are as fast or faster at every length on this GPU, so `pack_micro_batches=False`; the config documents how to enable it (flex attention forced, 16-wide backward blocks).
6. **Slowest micro-batches per step** in the trainer's step line and `AgentTrainingStats.slow_micro_batches`: first-encounter stalls show there.

## Application handoff (for your workspace agent)

1. **Copy the staged tree**: `cp -r IB/ARTIFACTS/AGENT_ROLLOUTS/slice2_delta/activation/. activation/` (six Python files, the `fla_configs/` folder with ten JSON files). No dependency change.
2. **CPU checks**: `uv run python -m compileall -q activation`; the extended CPU check `uv run python IB/TMP/AGENT_ROLLOUTS/slice2_cpu_check.py` ends with `SLICE2 CPU CHECK OK` (packed rows against single rows to 3e-5 in fp32; adapter dtypes restored).
3. **Node**: the key test as before. Expected in the step lines: `slowest 8.0s@30251` on the first step (kernel compiles, once per node) and nothing over 3 s afterwards; `AgentTrainer - no fla kernel configs for this GPU` must not appear on the RTX 6000 Pro.
4. **Probes are not in the cards.** `training_profile.py` (extended), `engine_contention_probe.py`, `stall_probe.py` and `fla_config_dump.py` stay under `slice2_delta/probes/activation/bench/agent_probes/`; copy them in if you want them in the tree. `fla_config_dump.py` is the one that matters for a new GPU type.

## Numbers (key test, round 0, Qwen3.5-4B, RTX 6000 Pro, cached rollouts)

| | handed off (`timed_round_run1`) | delta (`timed_round_run4`) |
| --- | --- | --- |
| step 1 | 70.8 s | 33.9 s |
| step 2 | 44.7 s | 26.3 s |
| training duration | 126.4 s (42 examples, 16k cap) | 72.2 s (43 examples, 52k cap) |
| steady throughput | 2,923 tok/s | 4,830 tok/s |
| test | `1 passed in 841.77s` | `1 passed in 117.33s` |

Probe evidence (`IB/TMP/AGENT_ROLLOUTS/training_profile_run3..6*.log`, `engine_contention_run1.log`, `stall_probe_run{1,2,3}*.log`): head 3,754 → 4,249 tok/s on 10-12k rows; no recompute 4,677 → 6,334 tok/s on 5k rows at 60 GB; packed flex 4,001 tok/s; the sleeping engine costs nothing (4,665 / 4,694 / 4,697 tok/s absent / asleep / freed); with the tuned configs and a fresh Triton cache no autotuning runs.

The cards below are exact deltas against `/source` with the slice-2 handoff applied (verified byte-equal to the staged `slice2/` copies); the staged copies under `slice2_delta/activation/` are byte-for-byte what the cards produce.

## Diffs to apply

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/vllm_wrapper.py</span>
    <span class="card-oneliner">max_model_len 52,000 for the Qwen3.5 agents (the trainer&#x27;s max_example_tokens matches).</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/harness/vllm_wrapper.py`:

```diff
--- a/activation/harness/vllm_wrapper.py
+++ b/activation/harness/vllm_wrapper.py
@@ -122,9 +122,9 @@ class VLLMWrapper:
             # A3B MoE labeler: MTP taxes prefill, and labeling is prefill-bound.
             "nvidia/Qwen3.6-35B-A3B-NVFP4": base,
             "Qwen/Qwen3.6-35B-A3B-FP8": base,
-            # Agent rollouts: room for a 32k compaction threshold plus the answer.
-            "Qwen/Qwen3.5-9B": base | {"max_model_len": 40_960},
-            "Qwen/Qwen3.5-4B": base | {"max_model_len": 40_960},
+            # Agent rollouts: 50k of trajectory plus a buffer (the trainer's max_example_tokens matches).
+            "Qwen/Qwen3.5-9B": base | {"max_model_len": 52_000},
+            "Qwen/Qwen3.5-4B": base | {"max_model_len": 52_000},
         }
         if model_id not in known:
             print(f"{model_id} - No recommended engine kwargs; using the base formula without speculative decoding.")
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/loaded_model.py</span>
    <span class="card-oneliner">decoder_forward passes extra keywords to every decoder layer (cu_seq_lens_q for packed rows).</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/harness/loaded_model.py`:

```diff
--- a/activation/harness/loaded_model.py
+++ b/activation/harness/loaded_model.py
@@ -458,9 +458,11 @@ class LoadedModel:
         attention_mask: torch.Tensor | None,
         position_ids: torch.Tensor|None = None,
         lora_name: str|None = None,
+        **kwargs,
     ) -> torch.Tensor:
         """
         Last hidden state [B, S, d_model] of the causal decoder without the language-model head.
+        Extra keywords reach every decoder layer (e.g. `cu_seq_lens_q` for packed rows).
         With a LoRA name the module manager activates that adapter for this pass (PEFT injects the
         adapters into the base's own linear layers, so self.model is the LoRA'd module tree; the
         wrapper only routes and manages adapters); with None any injected adapters are disabled.
@@ -476,6 +478,7 @@ class LoadedModel:
                 position_ids=position_ids,
                 use_cache=False,
                 return_dict=True,
+                **kwargs,
             )
         return outputs.last_hidden_state
 
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent_training/agent_training_config.py</span>
    <span class="card-oneliner">52k cap; fp32_head_matmul; gradient_checkpointing_min_tokens; packed-row knobs (off); slow_micro_batches in the stats.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent_training/agent_training_config.py`:

```diff
--- a/activation/agent_training/agent_training_config.py
+++ b/activation/agent_training/agent_training_config.py
@@ -16,7 +16,7 @@ class AgentTrainingConfig:
     # The objective.
     clip_epsilon: float = 0.2                  # PPO clip on the token ratio pi_theta / pi_old
     updates_per_round: int = 4                 # gradient steps per train() call: the data is split into this many mini-batches, one pass
-    max_example_tokens: int = 16_384           # longer trajectories are dropped (counted), never truncated
+    max_example_tokens: int = 52_000           # longer trajectories are dropped (counted), never truncated; matches the engine's context (50k + buffer)
     # Optimisation (LoRA values; a full fine-tune would sit two orders of magnitude lower).
     learning_rate: float = 2e-5
     weight_decay: float = 0.0
@@ -24,11 +24,25 @@ class AgentTrainingConfig:
     adam_eps: float = 1e-8
     max_grad_norm: float = 1.0
     # Memory.
-    micro_batch_tokens: int = 32_768          # token budget per micro-batch when examples are packed with padding (pad_free_micro_batches=False)
+    micro_batch_tokens: int = 32_768          # how many tokens may share one micro-batch when examples are packed (pack_micro_batches) or padded (pad_free_micro_batches=False); unused pad-free (one example each). Never drops or splits: an example over the budget gets a micro-batch of its own, up to max_example_tokens
     pad_free_micro_batches: bool = True       # one example per micro-batch, no padding mask: sdpa takes its causal (flash) path, nothing is wasted on pads
-    length_multiple: int = 1                  # sequence length rounded up to a multiple (tail pads carry no loss). 1: rounding to 512 cost 30% on the node (profile), and no per-shape compile cost was measurable           # padded tokens per forward/backward; gradient accumulation fills the mini-batch
+    # Packed rows (measured exact with the spacers, see agent_training_utils.collate): several examples in one row under
+    # micro_batch_tokens with per-example attention and recurrent state (needs the fla kernels). Off by default: on the
+    # RTX 6000 Pro packed rows under flex attention run at 4,000 tok/s against 4,250 for single long rows, and only the
+    # short tail gains; enable with attn_implementation="flex_attention" and decoder_kwargs={"kernel_options": {"BLOCK_M1": 16,
+    # "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 16}} (flex's default backward blocks need more shared memory than sm_120 has
+    # for head_dim 256); sdpa packs too but with a dense block mask (1,800 tok/s).
+    pack_micro_batches: bool = False
+    attn_implementation: str | None = None    # set on the base model for the round and restored after (None: leave it)
+    decoder_kwargs: dict = field(default_factory=dict)  # extra keywords for every decoder pass, e.g. flex attention's kernel_options
+    length_multiple: int = 1                  # sequence length rounded up to a multiple (tail pads carry no loss). 1: rounding to 512 cost 30% on the node (profile), and no per-shape compile cost was measurable
     logits_chunk_tokens: int = 2_048           # loss positions per lm_head chunk (the full logits of a long sequence would not fit)
+    fp32_head_matmul: bool = False             # the head's GEMM in fp32 on the SIMT kernels instead of bf16 operands with fp32 accumulation (same products, 6x slower; a reference)
     gradient_checkpointing: bool = True
+    # Recompute activations only for micro-batches of at least this many tokens; shorter ones keep them (+35% on the RTX 6000 Pro,
+    # 6,334 against 4,677 tok/s at 5k tokens) at about 9 GB per 1k tokens on top of 15 GB (60 GB at 5k). None: 6,144 on a card
+    # with at least 80 GB, else 0 (always recompute). 0 always recomputes.
+    gradient_checkpointing_min_tokens: int | None = None
     # Bookkeeping.
     seed: int = 0
     checkpoint_every_round: bool = True        # adapter saved under LORAS/<lora_name>/round_<n> (+ latest) after the round
@@ -51,6 +65,7 @@ class AgentTrainingStats:
     mean_logprob: list[float] = field(default_factory=list)    # per step, mean log pi_theta of the sampled tokens (an entropy proxy)
     grad_norm: list[float] = field(default_factory=list)       # per step, before clipping
     step_seconds: list[float] = field(default_factory=list)
+    slow_micro_batches: list[list] = field(default_factory=list)   # per step: the three slowest micro-batches as [tokens, seconds] (first-encounter kernel compiles show here)
     reporting_loss: float | None = None        # the objective on reporting_data after the round, no gradient
     positive_weight_sum: float = 0.0
     negative_weight_sum: float = 0.0
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent_training/fla_cache.py</span>
    <span class="card-oneliner">New: fla&#x27;s tuned-config files per GPU type: configure (fuzzy load) and dump (from the live autotuners).</span>
    <span class="card-badge">Diff</span>
  </summary>

New file vs `/source` for `activation/agent_training/fla_cache.py`:

```diff
new file mode 100644
--- /dev/null
+++ b/activation/agent_training/fla_cache.py
@@ -0,0 +1,89 @@
+"""
+fla's Triton kernels autotune at first use, and two of them (the l2norm kernels) key that on a block
+count derived from the sequence length, so a training process re-benchmarks them for every new length
+bucket: about 14 s each, several times per round, on every process (measured: 51 s of a 75 s step).
+fla can instead load tuned configs from `{kernel_name}.json` files with fuzzy numeric matching
+(`FLA_CACHE_MODE=fuzzy`), but ships none for this GPU and never writes them. `dump_fla_configs` writes
+them from the live autotuners after one tuning pass; `configure_fla_cache` points fla at the files for
+the current GPU (under `fla_configs/<sanitized gpu name>/`, checked in per GPU type).
+"""
+from __future__ import annotations
+
+import gc
+import json
+import os
+from pathlib import Path
+
+import torch
+
+CONFIG_ROOT = Path(__file__).parent / "fla_configs"
+
+
+def gpu_config_dir(device) -> Path | None:
+    """The config folder for this device's GPU type (fla's own sanitized name), or None off the GPU."""
+    if getattr(device, "type", None) != "cuda":
+        return None
+    try:
+        from fla.ops.utils.cache import get_gpu_info
+    except ImportError:
+        return None
+    return CONFIG_ROOT / get_gpu_info()
+
+
+def configure_fla_cache(device, config_dir: Path | None = None) -> Path | None:
+    """
+    Fuzzy config loading from the GPU's folder when it holds files; returns the folder used or None.
+    fla reads its mode at import, so the module global is set alongside the environment variables.
+    """
+    config_dir = config_dir if config_dir is not None else gpu_config_dir(device)
+    if config_dir is None or not any(config_dir.glob("*.json")):
+        return None
+    try:
+        import fla.ops.utils.cache as fla_cache
+    except ImportError:
+        return None
+    os.environ["FLA_CONFIG_DIR"] = str(config_dir)
+    os.environ["FLA_CACHE_MODE"] = "fuzzy"
+    fla_cache.FLA_CACHE_MODE = fla_cache.FlaCacheMode.FUZZY
+    return config_dir
+
+
+def live_autotuners() -> list:
+    from fla.ops.utils.cache import CachedAutotuner
+    return [obj for obj in gc.get_objects() if isinstance(obj, CachedAutotuner) and getattr(obj, "cache", None)]
+
+
+def dump_fla_configs(config_dir: Path) -> dict[str, int]:
+    """Write `{kernel_name}.json` for every fla autotuner that has tuned something; returns entries per kernel."""
+    import triton
+    from fla.ops.utils.cache import AutotuneKey
+    config_dir.mkdir(parents=True, exist_ok=True)
+    written = {}
+    for tuner in live_autotuners():
+        entries = {}
+        for key, cfg in tuner.cache.items():
+            autotune_key = [value if isinstance(value, (int, float, str, bool)) or value is None else str(value) for value in key]
+            entries[AutotuneKey.key_hash(autotune_key)] = {
+                "autotune_key": autotune_key,
+                "config": {
+                    "kwargs": dict(cfg.kwargs), "num_warps": cfg.num_warps, "num_stages": cfg.num_stages,
+                    "num_ctas": getattr(cfg, "num_ctas", 1), "maxnreg": getattr(cfg, "maxnreg", None),
+                },
+            }
+        if not entries:
+            continue
+        path = config_dir / f"{tuner.kernel_name}.json"
+        existing = {}
+        if path.exists():
+            try:
+                existing = json.loads(path.read_text()).get("autotune_entries") or {}
+            except (OSError, ValueError):
+                existing = {}
+        merged = {**existing, **entries}
+        most_common = max(merged.values(), key=lambda entry: sum(1 for other in merged.values() if other["config"] == entry["config"]))
+        path.write_text(json.dumps({
+            "kernel_name": tuner.kernel_name, "triton_version": triton.__version__,
+            "autotune_entries": merged, "default_config": most_common["config"],
+        }, indent=1))
+        written[tuner.kernel_name] = len(merged)
+    return written
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent_training/agent_training_utils.py</span>
    <span class="card-oneliner">Head GEMM on bf16 operands with fp32 accumulation; head chunks checkpointed; packed collation with spacers and the hybrid mask dict; checkpoint threshold; attention forcing; segments instead of rows.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent_training/agent_training_utils.py`:

```diff
--- a/activation/agent_training/agent_training_utils.py
+++ b/activation/agent_training/agent_training_utils.py
@@ -1,7 +1,7 @@
 """
 The messy parts of agent training, kept out of the trainer: turning a trajectory into one training
-sequence with its loss mask and stored log-probs, packing sequences into token-budgeted micro-batches,
-padding, and the chunked log-prob computation that avoids materialising the full logits of a long
+sequence with its loss mask and stored log-probs, grouping sequences into micro-batches (one each,
+padded, or packed into one row), padding, and the chunked log-prob computation that avoids materialising the full logits of a long
 sequence (16k positions x a 250k vocabulary would not fit).
 """
 from __future__ import annotations
@@ -10,6 +10,7 @@ import typing as t
 from dataclasses import dataclass
 
 import torch
+import torch.utils.checkpoint
 
 if t.TYPE_CHECKING:
     from .agent_trainer import AgentTrainingItem
@@ -68,21 +69,27 @@ def build_example(item: "AgentTrainingItem", item_index: int) -> TrainingExample
     )
 
 
-def micro_batches(examples: list[TrainingExample], token_budget: int, pad_free: bool = True) -> list[list[int]]:
+def micro_batches(examples: list[TrainingExample], token_budget: int, pad_free: bool = True, packed: bool = False) -> list[list[int]]:
     """
-    Indices grouped into micro-batches, longest first. Pad-free: one example each (no padding, so the
-    attention layers run their causal fast path and no token is wasted). Otherwise examples are packed
-    under `token_budget` padded tokens per micro-batch (longest-first bins keep the padding small).
+    Indices grouped into micro-batches, longest first. Packed: examples concatenated into one row
+    under `token_budget` tokens (no padding, per-sequence attention; see `collate(packed=True)`).
+    Pad-free: one example each (no padding, so the attention layers run their causal fast path and
+    no token is wasted). Otherwise examples are padded to the longest under `token_budget` padded
+    tokens per micro-batch (longest-first bins keep the padding small).
     """
     order = sorted(range(len(examples)), key=lambda index: -len(examples[index].token_ids))
-    if pad_free:
+    if pad_free and not packed:
         return [[index] for index in order]
     groups: list[list[int]] = []
     for index in order:
         length = len(examples[index].token_ids)
         for group in groups:
-            width = max(len(examples[i].token_ids) for i in group)
-            if max(width, length) * (len(group) + 1) <= token_budget:
+            if packed:
+                fits = sum(len(examples[i].token_ids) for i in group) + length <= token_budget
+            else:
+                width = max(len(examples[i].token_ids) for i in group)
+                fits = max(width, length) * (len(group) + 1) <= token_budget
+            if fits:
                 group.append(index)
                 break
         else:
@@ -93,23 +100,41 @@ def micro_batches(examples: list[TrainingExample], token_budget: int, pad_free:
 @dataclass
 class Collated:
     input_ids: torch.Tensor        # [B, S]
-    attention_mask: torch.Tensor   # [B, S]
-    position_ids: torch.Tensor     # [B, S]
+    attention_mask: torch.Tensor   # [B, S] real tokens (statistics)
+    position_ids: torch.Tensor     # [B, S]; packed: restarts at 0 for every example
     loss_mask: torch.Tensor        # [B, S] bool
     old_logprobs: torch.Tensor     # [B, S] float32
-    weights: torch.Tensor          # [B] float32
+    weights: torch.Tensor          # [E] float32, one per example in batch order
     example_indices: list[int]     # into the examples list, batch order
+    segment_ids: torch.Tensor      # [B, S] long: which example (batch order) each position belongs to
+    segment_offsets: list[int]     # where each example starts in its row (0 unless packed)
     model_attention_mask: torch.Tensor | None = None   # what the decoder gets: None = causal only (right padding never reaches a real token)
+    cu_seq_lens: torch.Tensor | None = None            # packed: int32 [E + 1] boundaries, for the linear-attention kernels (fla varlen)
+
+    @property
+    def packed(self) -> bool:
+        return self.cu_seq_lens is not None
 
 
 def collate(examples: list[TrainingExample], indices: list[int], pad_token_id: int, device, length_multiple: int = 1,
-            causal_only: bool | None = None) -> Collated:
+            causal_only: bool | None = None, packed: bool = False, spacer_tokens: int = 0) -> Collated:
     """
     Right-padded tensors on `device`. `attention_mask` marks the real tokens (statistics); the decoder
     gets `model_attention_mask`: None when nothing but tail padding is present (`causal_only`, the
     default for a single example), since under causal attention a real token never sees a later pad and
     the pads carry no loss. `length_multiple` rounds the length up so the compiled kernels see few shapes.
+
+    Packed: every example concatenated into one row with no padding and its positions restarting at 0.
+    transformers detects the resets (attention_mask None) and gives the attention layers a per-sequence
+    block mask (flex: a BlockMask; sdpa: a dense one, slow; flash: cu_seqlens), and `cu_seq_lens` goes
+    to the gated-delta layers so their state restarts too (the `fla` kernels honour it; the torch fallback
+    would not). Their short causal conv would still see the previous example's last tokens at a boundary
+    (measured: 5x the bf16 noise), so `spacer_tokens` (the conv width minus one, see `packing_spacer_tokens`)
+    of pad go between examples as their own tiny sequence; `packed_attention_masks` hands the gated-delta
+    layers a mask that zeroes them, which is exactly the zero padding a fresh sequence's conv sees.
     """
+    if packed:
+        return _collate_packed(examples, indices, pad_token_id, device, length_multiple, spacer_tokens)
     length = max(len(examples[index].token_ids) for index in indices)
     length = -(-length // max(1, length_multiple)) * max(1, length_multiple)
     if causal_only is None:
@@ -132,29 +157,121 @@ def collate(examples: list[TrainingExample], indices: list[int], pad_token_id: i
         old_logprobs=torch.tensor(old, dtype=torch.float32, device=device),
         weights=torch.tensor(weights, dtype=torch.float32, device=device),
         example_indices=list(indices),
+        segment_ids=torch.arange(len(indices), device=device)[:, None].expand(len(indices), length).contiguous(),
+        segment_offsets=[0] * len(indices),
         model_attention_mask=None if causal_only else attention_mask.to(device),
     )
 
 
-def sampled_logprobs(hidden: torch.Tensor, head: torch.nn.Module, batch: Collated, chunk_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
+def _collate_packed(examples: list[TrainingExample], indices: list[int], pad_token_id: int, device, length_multiple: int = 1,
+                    spacer_tokens: int = 0) -> Collated:
+    """One row; spacers between examples and a `length_multiple` tail are pad segments (their own sequence, no loss, masked)."""
+    tokens, positions, loss, old, segments, offsets, boundaries, real = [], [], [], [], [], [], [0], []
+
+    def pad_segment(count: int):
+        tokens.extend([pad_token_id] * count)
+        positions.extend(range(count))
+        loss.extend([False] * count)
+        old.extend([0.0] * count)
+        segments.extend([len(indices)] * count)
+        real.extend([0] * count)
+        boundaries.append(len(tokens))
+
+    for ordinal, index in enumerate(indices):
+        if ordinal and spacer_tokens:
+            pad_segment(spacer_tokens)
+        example = examples[index]
+        length = len(example.token_ids)
+        offsets.append(len(tokens))
+        tokens += example.token_ids
+        positions += list(range(length))
+        loss += example.loss_mask
+        old += example.old_logprobs
+        segments += [ordinal] * length
+        real += [1] * length
+        boundaries.append(len(tokens))
+    pad = -(-len(tokens) // max(1, length_multiple)) * max(1, length_multiple) - len(tokens)
+    if pad:
+        pad_segment(pad)
+    return Collated(
+        input_ids=torch.tensor([tokens], dtype=torch.long, device=device),
+        attention_mask=torch.tensor([real], dtype=torch.long, device=device),
+        position_ids=torch.tensor([positions], dtype=torch.long, device=device),
+        loss_mask=torch.tensor([loss], dtype=torch.bool, device=device),
+        old_logprobs=torch.tensor([old], dtype=torch.float32, device=device),
+        weights=torch.tensor([examples[index].weight for index in indices], dtype=torch.float32, device=device),
+        example_indices=list(indices),
+        segment_ids=torch.tensor([segments], dtype=torch.long, device=device),
+        segment_offsets=offsets,
+        model_attention_mask=None,
+        cu_seq_lens=torch.tensor(boundaries, dtype=torch.int32, device=device),
+    )
+
+
+class _HeadMatmul(torch.autograd.Function):
+    """
+    logits = states @ weight.T with bf16 operands on the tensor cores and fp32 accumulation and output
+    (`torch.mm(out_dtype=float32)`): the products of two bf16 values are exact in fp32, so this equals the
+    fp32 GEMM of the upcast operands up to summation order, at a fraction of the SIMT fp32 kernel's time
+    and without the fp32 copies of the states and the 2.5 GB weight. The head is frozen: only the gradient
+    to the states is formed, from the fp32 gradient split into a high and a low bf16 part (two GEMMs,
+    ~2^-16 relative error instead of bf16's 2^-8).
+    """
+
+    @staticmethod
+    def forward(ctx, states, weight):
+        ctx.save_for_backward(weight)
+        ctx.states_dtype = states.dtype
+        return torch.mm(states, weight.t(), out_dtype=torch.float32)
+
+    @staticmethod
+    def backward(ctx, grad_logits):
+        (weight,) = ctx.saved_tensors
+        high = grad_logits.to(weight.dtype)
+        low = (grad_logits - high.float()).to(weight.dtype)
+        grad_states = torch.mm(high, weight, out_dtype=torch.float32) + torch.mm(low, weight, out_dtype=torch.float32)
+        return grad_states.to(ctx.states_dtype), None
+
+
+def head_logits(states: torch.Tensor, head: torch.nn.Module, fp32_matmul: bool = False) -> torch.Tensor:
+    """fp32 logits of `states` through the frozen output head (see _HeadMatmul; the fp32 GEMM on CPU or on request)."""
+    weight = head.weight
+    bias = getattr(head, "bias", None)
+    if fp32_matmul or not states.is_cuda or weight.dtype not in (torch.bfloat16, torch.float16) or weight.dtype != states.dtype:
+        logits = torch.nn.functional.linear(states.float(), weight.float(), bias.float() if bias is not None else None)
+    else:
+        logits = _HeadMatmul.apply(states, weight)
+        if bias is not None:
+            logits = logits + bias.float()
+    return logits
+
+
+def sampled_logprobs(hidden: torch.Tensor, head: torch.nn.Module, batch: Collated, chunk_tokens: int,
+                     fp32_head_matmul: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
     """
     log pi_theta of every loss token: the decoder's hidden state at the previous position through the
-    output head, in chunks. The head runs in fp32 (its weight cast once per call; it is frozen, so no
-    gradient flows through the cast): bf16 logits of magnitude 20-30 are quantised in steps of 0.125-0.25,
-    which alone would put several percent of the tokens outside a 0.2 ratio clip at the first step.
-    Returns (logprobs [N], batch row of each [N], target ids [N]).
+    output head, in chunks, with fp32 logits (`head_logits`): bf16 logits of magnitude 20-30 are quantised
+    in steps of 0.125-0.25, which alone would put several percent of the tokens outside a 0.2 ratio clip at
+    the first step. Returns (logprobs [N], example of each in batch order [N], target ids [N]). The previous
+    position is always inside the same example: an example's first token never carries loss (build_example).
     """
     rows, positions = batch.loss_mask.nonzero(as_tuple=True)
     states = hidden[rows, positions - 1]                                                   # [N, d]
     targets = batch.input_ids[rows, positions]                                             # [N]
-    with torch.no_grad():
-        weight = head.weight.float()
-        bias = head.bias.float() if getattr(head, "bias", None) is not None else None
+    segments = batch.segment_ids[rows, positions]                                          # [N]
+
+    def chunk_logprobs(chunk_states, chunk_targets):
+        logits = head_logits(chunk_states, head, fp32_head_matmul)                          # [n, V] fp32
+        return torch.log_softmax(logits, dim=-1).gather(-1, chunk_targets[:, None])[:, 0]
+
     pieces = []
     for start in range(0, states.shape[0], chunk_tokens):
-        logits = torch.nn.functional.linear(states[start:start + chunk_tokens].float(), weight, bias)   # [n, V] fp32
-        pieces.append(torch.log_softmax(logits, dim=-1).gather(-1, targets[start:start + chunk_tokens, None])[:, 0])
-    return torch.cat(pieces) if pieces else states.new_zeros(0, dtype=torch.float32), rows, targets
+        # Checkpointed: autograd would otherwise keep every chunk's [n, V] fp32 log-softmax (2 GB per 2,048
+        # positions) until backward; recomputing the chunk there keeps the head at a constant footprint.
+        pieces.append(torch.utils.checkpoint.checkpoint(
+            chunk_logprobs, states[start:start + chunk_tokens], targets[start:start + chunk_tokens], use_reentrant=False,
+        ))
+    return torch.cat(pieces) if pieces else states.new_zeros(0, dtype=torch.float32), segments, targets
 
 
 def clipped_surrogate(logprobs: torch.Tensor, old_logprobs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, dict[str, float]]:
@@ -173,6 +290,88 @@ def clipped_surrogate(logprobs: torch.Tensor, old_logprobs: torch.Tensor, weight
     return surrogate, stats
 
 
+def set_attention_implementation(model, name: str) -> None:
+    """
+    The base model's attention implementation for training. Flex is forced on models that do not declare
+    it (Qwen3.5 does not; its text-only path works: the block mask comes from transformers' generic
+    masking and the attention layers dispatch through the same interface as sdpa).
+    """
+    if name == "flex_attention" and not getattr(type(model), "_supports_flex_attn", False):
+        print(f"{type(model).__name__} does not declare flex attention support; forcing it for the text-only training path", flush=True)
+        for module in model.modules():
+            cls = type(module)
+            if hasattr(cls, "_supports_flex_attn") and not getattr(cls, "_supports_flex_attn"):
+                cls._supports_flex_attn = True
+    if model.config._attn_implementation != name:
+        model.set_attn_implementation(name)
+
+
+CHECKPOINT_FREE_TOKENS_ON_LARGE_CARDS = 6_144      # micro-batches below this keep their activations on a card with >= 80 GB
+
+
+def checkpointing_min_tokens(config, device) -> int:
+    """The configured threshold, or the default for the card (see AgentTrainingConfig.gradient_checkpointing_min_tokens)."""
+    if config.gradient_checkpointing_min_tokens is not None:
+        return int(config.gradient_checkpointing_min_tokens)
+    if getattr(device, "type", None) == "cuda" and torch.cuda.get_device_properties(device).total_memory >= 80e9:
+        return CHECKPOINT_FREE_TOKENS_ON_LARGE_CARDS
+    return 0
+
+
+def set_checkpointing(model, config, num_tokens: int, min_tokens: int = 0) -> bool:
+    """Gradient checkpointing on for this micro-batch when configured and the batch is long enough; returns the state."""
+    wanted = bool(config.gradient_checkpointing) and num_tokens >= min_tokens
+    if wanted != bool(model.is_gradient_checkpointing):
+        if wanted:
+            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
+        else:
+            model.gradient_checkpointing_disable()
+    return wanted
+
+
+def _text_config(model):
+    return model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
+
+
+def packing_spacer_tokens(model) -> int:
+    """Pad tokens between packed examples: the gated-delta conv width minus one (0 for a plain attention model)."""
+    config = _text_config(model)
+    if "linear_attention" not in (getattr(config, "layer_types", None) or []):
+        return 0
+    return max(0, int(getattr(config, "linear_conv_kernel_dim", 4)) - 1)
+
+
+def packed_attention_masks(model, inputs_embeds: torch.Tensor, batch: Collated):
+    """
+    The decoder's attention_mask argument for a packed row: None for a plain attention model (transformers
+    finds the sequences from the position resets), else the per-layer-type dict the hybrid model builds
+    itself, with the packed causal mask for the attention layers and the real-token mask for the
+    gated-delta layers (which zero their input where it is 0: spacers and tail pads).
+    """
+    config = _text_config(model)
+    layer_types = getattr(config, "layer_types", None) or []
+    if "linear_attention" not in layer_types:
+        return None
+    from transformers.masking_utils import create_causal_mask
+    full = create_causal_mask(config, inputs_embeds, None, None, batch.position_ids)
+    masks = {layer_type: full for layer_type in set(layer_types)}
+    masks["linear_attention"] = batch.attention_mask.bool()
+    return masks
+
+
+def check_packing_support(model) -> None:
+    """
+    Packed rows need the gated-delta kernels that take `cu_seqlens` (transformers routes to the `fla`
+    package or a hub kernel; its pure torch fallback ignores the boundaries and would carry state across examples).
+    """
+    if "linear_attention" not in (getattr(_text_config(model), "layer_types", None) or []):
+        return
+    try:
+        import fla  # noqa: F401
+    except ImportError as error:
+        raise RuntimeError("pack_micro_batches needs the flash-linear-attention package (fla) for per-example recurrent state") from error
+
+
 def disable_dropout(module: torch.nn.Module) -> int:
     """Dropout layers to eval while the rest trains (checkpointing needs training mode). Returns how many."""
     count = 0
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent_training/agent_trainer.py</span>
    <span class="card-oneliner">Loads the fla configs; checkpointing per micro-batch by length; per-example segments; slowest micro-batches per step; attention implementation set and restored.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent_training/agent_trainer.py`:

```diff
--- a/activation/agent_training/agent_trainer.py
+++ b/activation/agent_training/agent_trainer.py
@@ -35,14 +35,21 @@ from activation.harness import SOURCE_DEVICE, TARGET_DEVICE
 
 from .agent_training_config import AgentTrainingConfig, AgentTrainingStats
 from .agent_training_reporter import AgentTrainingReporter
+from .fla_cache import configure_fla_cache
 from .agent_training_utils import (
     Collated,
     TrainingExample,
     build_example,
+    check_packing_support,
+    checkpointing_min_tokens,
     clipped_surrogate,
     collate,
     disable_dropout,
     micro_batches,
+    packed_attention_masks,
+    packing_spacer_tokens,
+    set_attention_implementation,
+    set_checkpointing,
     sampled_logprobs,
 )
 
@@ -143,8 +150,16 @@ class AgentTrainer:
         base = loaded_model.model
         device = next(base.parameters()).device
         base.train()
-        if config.gradient_checkpointing:
-            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
+        previous_attn = base.config._attn_implementation
+        if config.attn_implementation and config.attn_implementation != previous_attn:
+            set_attention_implementation(base, config.attn_implementation)
+        if config.pack_micro_batches:
+            check_packing_support(base)
+        self._spacer_tokens = packing_spacer_tokens(base)
+        if configure_fla_cache(device) is None and device.type == "cuda":
+            print("AgentTrainer - no fla kernel configs for this GPU: the Triton kernels autotune per length bucket (see fla_cache.py)", flush=True)
+        min_tokens = checkpointing_min_tokens(config, device)
+        set_checkpointing(base, config, min_tokens, min_tokens)
         disable_dropout(base)
         parameters = module_manager.lora_parameters(lora_name)
         assert parameters, f"LoRA {lora_name!r} has no trainable parameters"
@@ -174,10 +189,16 @@ class AgentTrainer:
                     break
                 optimizer.zero_grad(set_to_none=True)
                 totals = {"loss": 0.0, "mean_ratio": 0.0, "clip_fraction": 0.0, "mean_logprob": 0.0, "tokens": 0, "loss_tokens": 0}
-                for group in micro_batches([examples[index] for index in mini], config.micro_batch_tokens, config.pad_free_micro_batches):
-                    batch = collate(examples, [mini[i] for i in group], pad_token_id, device, config.length_multiple)
+                micro_batch_times = []
+                for group in micro_batches([examples[index] for index in mini], config.micro_batch_tokens, config.pad_free_micro_batches,
+                                           config.pack_micro_batches):
+                    batch = collate(examples, [mini[i] for i in group], pad_token_id, device, config.length_multiple,
+                                    packed=config.pack_micro_batches, spacer_tokens=self._spacer_tokens)
+                    set_checkpointing(base, config, int(batch.input_ids.numel()), min_tokens)
+                    micro_batch_started = time.time()
                     loss, batch_stats, loss_tokens = self._objective(loaded_model, lora_name, batch, len(mini))
                     loss.backward()
+                    micro_batch_times.append([int(batch.input_ids.numel()), round(time.time() - micro_batch_started, 2)])
                     totals["loss"] += float(loss.detach())
                     for key in ("mean_ratio", "clip_fraction", "mean_logprob"):
                         totals[key] += batch_stats[key] * loss_tokens
@@ -194,9 +215,12 @@ class AgentTrainer:
                 stats.mean_logprob.append(totals["mean_logprob"] / divisor)
                 stats.grad_norm.append(grad_norm)
                 stats.step_seconds.append(seconds)
+                slowest = sorted(micro_batch_times, key=lambda pair: -pair[1])[:3]
+                stats.slow_micro_batches.append(slowest)
                 print(f"AgentTrainer - round {round_index} step {step + 1}/{steps}: loss {totals['loss']:.4f} ratio {stats.mean_ratio[-1]:.3f} "
                       f"clip {stats.clip_fraction[-1]:.3f} logprob {stats.mean_logprob[-1]:.3f} grad {grad_norm:.3f} "
-                      f"{totals['tokens']} tokens in {seconds:.1f}s", flush=True)
+                      f"{totals['tokens']} tokens in {seconds:.1f}s ({len(micro_batch_times)} micro-batches, slowest "
+                      f"{', '.join(f'{t}s@{n}' for n, t in slowest)})", flush=True)
                 if reporter is not None:
                     reporter.report_step(round_index, step + 1, totals["loss"], stats.mean_ratio[-1], stats.clip_fraction[-1], stats.mean_logprob[-1],
                                          grad_norm, totals["tokens"], seconds)
@@ -205,8 +229,9 @@ class AgentTrainer:
             if reporting_examples:
                 with torch.no_grad():
                     total = 0.0
-                    for group in micro_batches(reporting_examples, config.micro_batch_tokens, config.pad_free_micro_batches):
-                        batch = collate(reporting_examples, group, pad_token_id, device, config.length_multiple)
+                    for group in micro_batches(reporting_examples, config.micro_batch_tokens, config.pad_free_micro_batches, config.pack_micro_batches):
+                        batch = collate(reporting_examples, group, pad_token_id, device, config.length_multiple,
+                                        packed=config.pack_micro_batches, spacer_tokens=self._spacer_tokens)
                         loss, _, _ = self._objective(loaded_model, lora_name, batch, len(reporting_examples))
                         total += float(loss)
                 stats.reporting_loss = total
@@ -218,6 +243,8 @@ class AgentTrainer:
             base.eval()
             if base.is_gradient_checkpointing:
                 base.gradient_checkpointing_disable()
+            if base.config._attn_implementation != previous_attn:
+                set_attention_implementation(base, previous_attn)
             if device.type == "cuda":
                 stats.peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
             loaded_model.model_to_device(SOURCE_DEVICE)          # off the GPU; the next rollout wakes the engine
@@ -235,30 +262,37 @@ class AgentTrainer:
     def _hidden_and_logprobs(self, loaded_model, lora_name: str, batch: Collated) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
         base = loaded_model.model
         inputs_embeds = base.get_input_embeddings()(batch.input_ids)
-        hidden = loaded_model.decoder_forward(inputs_embeds, batch.model_attention_mask, batch.position_ids, lora_name)
-        return sampled_logprobs(hidden, base.get_output_embeddings(), batch, self.config.logits_chunk_tokens)
+        kwargs = dict(self.config.decoder_kwargs)
+        attention_mask = batch.model_attention_mask
+        if batch.packed:
+            kwargs["cu_seq_lens_q"] = batch.cu_seq_lens                      # the gated-delta layers restart their state per example
+            attention_mask = packed_attention_masks(base, inputs_embeds, batch)
+        hidden = loaded_model.decoder_forward(inputs_embeds, attention_mask, batch.position_ids, lora_name, **kwargs)
+        return sampled_logprobs(hidden, base.get_output_embeddings(), batch, self.config.logits_chunk_tokens, self.config.fp32_head_matmul)
 
     def _objective(self, loaded_model, lora_name: str, batch: Collated, num_examples: int) -> tuple[torch.Tensor, dict[str, float], int]:
         """The negated clipped surrogate of this micro-batch, scaled so the mini-batch's total is a mean over its examples."""
-        logprobs, rows, _ = self._hidden_and_logprobs(loaded_model, lora_name, batch)
+        logprobs, segments, _ = self._hidden_and_logprobs(loaded_model, lora_name, batch)
         old = batch.old_logprobs[batch.loss_mask]
-        weights = batch.weights[rows]
+        weights = batch.weights[segments]
         surrogate, batch_stats = clipped_surrogate(logprobs, old, weights, self.config.clip_epsilon)
-        num_rows = batch.input_ids.shape[0]
-        per_sample = torch.zeros(num_rows, device=logprobs.device, dtype=torch.float32).index_add_(0, rows, surrogate)
-        counts = torch.zeros(num_rows, device=logprobs.device, dtype=torch.float32).index_add_(0, rows, torch.ones_like(surrogate))
+        count = len(batch.example_indices)
+        per_sample = torch.zeros(count, device=logprobs.device, dtype=torch.float32).index_add_(0, segments, surrogate)
+        counts = torch.zeros(count, device=logprobs.device, dtype=torch.float32).index_add_(0, segments, torch.ones_like(surrogate))
         loss = -(per_sample / counts.clamp(min=1)).sum() / max(1, num_examples)
         return loss, batch_stats, int(logprobs.shape[0])
 
     @torch.no_grad()
     def _fill_old_logprobs(self, loaded_model, lora_name: str, examples: list[TrainingExample], pad_token_id: int, device) -> None:
         """pi_old := the current policy for examples without engine log-probs (teacher / hinted / unrecorded)."""
-        for group in micro_batches(examples, self.config.micro_batch_tokens, self.config.pad_free_micro_batches):
-            batch = collate(examples, group, pad_token_id, device, self.config.length_multiple)
-            logprobs, rows, _ = self._hidden_and_logprobs(loaded_model, lora_name, batch)
+        config = self.config
+        for group in micro_batches(examples, config.micro_batch_tokens, config.pad_free_micro_batches, config.pack_micro_batches):
+            batch = collate(examples, group, pad_token_id, device, config.length_multiple, packed=config.pack_micro_batches,
+                            spacer_tokens=self._spacer_tokens)
+            logprobs, segments, _ = self._hidden_and_logprobs(loaded_model, lora_name, batch)
             positions = batch.loss_mask.nonzero(as_tuple=True)[1]
-            for row_index, example_index in enumerate(batch.example_indices):
-                mask = rows == row_index
+            for ordinal, example_index in enumerate(batch.example_indices):
+                mask = segments == ordinal
                 values = logprobs[mask].tolist()
                 for position, value in zip(positions[mask].tolist(), values):
-                    examples[example_index].old_logprobs[position] = value
+                    examples[example_index].old_logprobs[position - batch.segment_offsets[ordinal]] = value
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent_training/fla_configs/</span>
    <span class="card-oneliner">New: fla&#x27;s tuned kernel configs per GPU type (JSON, generated by fla_config_dump.py on the node); copied, not diffed.</span>
    <span class="card-badge">Diff</span>
  </summary>

Ten new files, one per fla kernel, under the GPU's sanitized name (fla's own):

- `activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_bwd_kernel_dqkwg.json`
- `activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_bwd_kernel_dv_local.json`
- `activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_fwd_kernel_o.json`
- `activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64.json`
- `activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.json`
- `activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_gated_delta_rule_fwd_kkt_solve_kernel.json`
- `activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_local_cumsum_scalar_kernel.json`
- `activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/l2norm_bwd_kernel.json`
- `activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/l2norm_fwd_kernel.json`
- `activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/recompute_w_u_fwd_kernel.json`

Copy the folder as is; the trainer picks it by `torch.cuda.get_device_name`. Another GPU type gets its own folder from one run of the dump probe (canon note).

</details>
