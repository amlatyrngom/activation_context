# Current code — source sync handoff

**Historical capture:** the [current consolidation handoff](CONSOLIDATION_HANDOFF.tressoir.md) contains the remaining changes against today's source. The already-applied package below is retained only as provenance.

The existing workspace implementation captured before the task-aware changes. Review the behavior summary in [HANDOFF.md](HANDOFF.md); full files, a patch and SHA-256 hashes are in [current_code/](current_code/). These are exact per-file differences against the captured `/source` baseline. The live `/source` tree now matches all packaged destination hashes (15:25 UTC); keep these diffs as the before/after record, and do not reapply them there. No task-aware implementation is included.

## Scope and validation

The snapshot includes all runtime Python differences, the two training test files already requested by the user, and the previously approved three-line `base_dir` caller migration in `test_basic_agent_ac.py`. That existing caller must migrate with tensor serialization; it is not another test suite. Other test changes, private probes, experiment benches, datasets and weights are excluded. `pyproject.toml` and `uv.lock` match `/source`. The retired `activation/agent/agent_programs.py` appears as an explicit deletion. Patch application and staged-file hashes are checked; prior runtime evidence and known gaps are listed in the handoff. This capture does not claim a new full GPU regression run.

## Files

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/ac_model/ac_model.py">activation/ac_model/ac_model.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/ac_model/ac_model.py`

```diff
diff --git a/activation/ac_model/ac_model.py b/activation/ac_model/ac_model.py
index c66de9c..c3313e2 100644
--- a/activation/ac_model/ac_model.py
+++ b/activation/ac_model/ac_model.py
@@ -15,7 +15,7 @@ Forward of one part: its activation_context children first (recursively, cached)
 recursive adapter into the side input space; `tokenize_with_parts` renders the part under the side
 chat template with a pad-id placeholder run per child; side input embeddings with the child rows
 written over the placeholders, scale-aligned with positional encoding -> blocked-local mixer
--> optional content pooling (stride 0 retains content), plus V separately pooled content summaries
+-> V pooled content summaries
 with one shared marker and positions -> the causal side decoder with its LoRA -> the final V
 positions -> `target_head` (into the target width) or, for a recursive child, returned for the
 parent's `recursive_adapter`.
@@ -39,14 +39,18 @@ from pathlib import Path
 
 import torch
 import torch.nn as nn
+import torch.nn.functional as F
 from torch.utils.checkpoint import checkpoint
 
 from ..common.data_syncing import resolve_path
 from ..harness.hf_utils import SOURCE_DEVICE, TARGET_DEVICE
 from .ac_model_utils import (
     AC_PART_TYPE,
+    ACOutput,
     AdaptiveSummaryPooling,
     BlockedLocalAttentionLayer,
+    DeltaHead,
+    PassageAttention,
     EncodeQueue,
     EncodeRequest,
     RowCache,
@@ -54,6 +58,7 @@ from .ac_model_utils import (
     WindowedPooling,
     direct_parts,
     normalize_messages,
+    resolve_intervention_layers,
     row_cache_key,
     sinusoidal_positions,
     tokenize_with_parts,
@@ -69,7 +74,12 @@ MODULES_FILE = "ac_modules.pt"
 SIDE_LORA_FOLDER = "side_lora"
 ROLLOUT, TRAINING = "rollout", "training"
 ARCHITECTURE_VERSION = 2
-PreparedPart = tuple[list[int], list[tuple[int, int]], list[torch.Tensor], int, bool]
+INTERVENTION_INIT_SEED_MASK = 0xDEE9    # the intervention modules draw from a forked stream (initial seed ^ mask): the shared modules and the adapters draw as before
+INTERVENTION_MODULE_PREFIXES = ("delta_heads.", "passage_attention.")   # state-dict / parameter-name prefixes of the intervention modules
+READOUT_MODULE_PREFIXES = ("row_readout.", "recursive_readout.")
+POOLING_MODULE_PREFIX = "pooling."          # the read-outs in front of the row heads: fresh (identity) when a checkpoint predates them
+DIAGNOSTIC_CAPACITY = 256        # per layer, the encoder keeps at most this many diagnostic values between trace points
+PreparedPart = tuple[list[int], list[tuple[int, int]], list[torch.Tensor], int, bool, list[torch.Tensor]]   # ids, part spans, child rows, V, recursive, child states
 
 
 class CompletedEpoch(t.TypedDict):
@@ -80,14 +90,26 @@ class CompletedEpoch(t.TypedDict):
 
 @dataclass
 class ActivationContextModelConfig:
+    """
+    The AC model's shape. The encoder is fixed: the side base's frozen embeddings -> `mixer_layers` bidirectional blocked-local
+    attention layers -> summary pooling into V rows -> one causal pass of the side base with its LoRA over [content; summaries]
+    -> the final V states, each read out through a PassageAttention over the level's key set (the passage's final side states
+    and, with `recursive_keys`, the final states its children returned), through the row head (identity-skip FFN, RMS-normalized,
+    learned scale) into the reader's input space; a recursive part's rows go through the same kind of read-out and the recursive
+    adapter instead. Optional content pooling (`input_pooling_stride`) shortens the decoder's content block to about 1/stride.
+    A deep model (`intervention_frequency` != 0 or explicit `intervention_layers`) adds, per intervention layer, a read-out in
+    which the row's final side state attends over the passage tokens' final side states (PassageAttention, zero-initialized
+    output) and a zero-initialized bottleneck head (DeltaHead) whose output the reader adds to its residual stream at the row
+    positions, at the input of that layer. Both are exactly zero at init, so a deep model starts bit-identical to the
+    input-only one, and an input-only checkpoint loads into a deep model (the heads fresh). The reader-side operation is a
+    per-request scatter-add on the hidden state entering the named layers: the same in HF and in an engine.
+    """
     ac_model_name: str                       # identifies this model; parts name it in `ac_name`
     base_side_model_name: str                # the side base (a loaded model)
     base_side_model_lora_name: str           # its adapter, registered by ModuleManager.register_ac_model when new
     target_model_name: str                   # whose input space the rows land in
     target_model_lora_name: str              # the target adapter trained alongside (ActivationContextTrainer)
     default_compression_ratio: float = 1.0 / 16.0
-    input_pooling_stride: int = 0           # zero bypasses content pooling; summaries always pool
-    input_pooling_window: int = 8
     mixer_layers: int = 2
     mixer_window: int = 256                  # block size of the blocked local attention (each row sees 256-512 rows per side)
     mixer_heads: int = 8
@@ -97,12 +119,29 @@ class ActivationContextModelConfig:
     encode_batch_max_tokens: int = 65_536    # side tokens per padded encode batch (B x longest)
     side_gradient_checkpointing: bool = True # on the side base in training mode
     checkpoint_path: str | None = None       # loaded by register_ac_model when given
+    intervention_frequency: int = 0          # deep inputs: interior reader layers receiving a residual delta per row (0 = the embedding channel only);
+                                             # -1 = every full-attention layer of the reader (0 excluded; the last-layer flag is moot) - the validated setting
+    intervention_layers: list[int] | None = None   # explicit reader layers instead of the frequency rule (with intervention_frequency 0; 0 < layer < depth, unique)
+    add_last_layer_intervention: bool = True # ... plus the last reader block (inert at frequency 0)
+    recursive_keys: bool = False             # the read-outs of a level also attend the final states its children returned (nested parts)
+    input_pooling_stride: int = 0            # content pooling before the side decoder: one vector per `input_pooling_window`-wide window every
+                                             # `stride` rows (the content block becomes about 1/stride of the tokens); 0 = off, the tokens go through
+    input_pooling_window: int = 8
 
     def __post_init__(self) -> None:
         if self.input_pooling_stride < 0:
             raise ValueError("Content pooling stride must be nonnegative")
         if self.input_pooling_stride and not 0 < self.input_pooling_stride <= self.input_pooling_window:
             raise ValueError("Positive pooling stride must not exceed its positive window")
+        if self.intervention_frequency < -1:
+            raise ValueError("intervention_frequency must be -1 (every full-attention layer), 0 or positive")
+        if self.intervention_layers is not None:
+            layers = [int(layer) for layer in self.intervention_layers]
+            if self.intervention_frequency != 0:
+                raise ValueError("explicit intervention_layers replace the frequency rule: set intervention_frequency 0")
+            if not layers or len(set(layers)) != len(layers) or any(layer <= 0 for layer in layers):
+                raise ValueError("intervention_layers must be a non-empty list of distinct positive reader layers")
+            self.intervention_layers = sorted(layers)
         if not 0 < self.min_view_rows <= self.max_view_rows:
             raise ValueError("Invalid view-row bounds")
 
@@ -135,21 +174,57 @@ class ActivationContextModelStats:
 class ActivationContextModules(nn.Module):
     """The trainable AC modules besides the side LoRA (fp32)."""
 
-    def __init__(self, config: ActivationContextModelConfig, d_side: int, d_target: int) -> None:
+    def __init__(self, config: ActivationContextModelConfig, d_side: int, d_target: int, intervention_layers: list[int] = ()) -> None:
         super().__init__()
         self.mixer = nn.ModuleList([BlockedLocalAttentionLayer(d_side, config.mixer_heads, config.mixer_window) for _ in range(config.mixer_layers)])
-        self.pooling = (WindowedPooling(d_side, config.mixer_heads, config.input_pooling_window, config.input_pooling_stride)
-                        if config.input_pooling_stride else None)
         self.summary_pooling = AdaptiveSummaryPooling(d_side, config.mixer_heads)
         self.summary_marker = nn.Parameter(torch.empty(d_side))
         self.position_scale = nn.Parameter(torch.tensor(0.1))
         self.input_scale = nn.Parameter(torch.tensor(1.0))
         self.register_buffer("embedding_scale", torch.tensor(1.0))
         self.recursive_adapter = RowHead(d_side, d_side)
-        self.target_head = RowHead(d_side, d_target)
+        # The rows start as the (normalized) final side state itself when the widths allow the identity skip.
+        self.target_head = RowHead(d_side, d_target, skip=d_side == d_target)
         self.register_buffer("scales_initialized", torch.tensor(False))
         nn.init.normal_(self.summary_marker, std=0.02)
+        # Deep inputs: one read-out block and one bottleneck head per intervention layer, registered (parameters, state-dict keys)
+        # only when the model is deep, so an input-only model's checkpoint is the one it always was. Their initialization draws
+        # from forked generators: the mixer, the row heads and the adapters injected later draw exactly the sequence they drew before.
+        self.intervention_layers: list[int] = [int(layer) for layer in intervention_layers]
+        self.delta_heads: nn.ModuleDict | None = None
+        self.passage_attention: nn.ModuleDict | None = None
+        if self.intervention_layers:
+            r = max(1, min(d_side, d_target) // 4)
+            with torch.random.fork_rng(devices=[]):
+                torch.manual_seed(torch.initial_seed() ^ INTERVENTION_INIT_SEED_MASK)
+                self.delta_heads = nn.ModuleDict({str(layer): DeltaHead(d_side, d_target, r) for layer in self.intervention_layers})
+            with torch.random.fork_rng(devices=[]):
+                torch.manual_seed((torch.initial_seed() ^ INTERVENTION_INIT_SEED_MASK) + 1)
+                self.passage_attention = nn.ModuleDict({str(layer): PassageAttention(d_side, r) for layer in self.intervention_layers})
+        # The read-outs in front of the row heads (the same block the delta heads have): identity at init (zero output projection), so
+        # a checkpoint that predates them loads with them fresh and reproduces its outputs. Forked streams, created last: every module
+        # above draws exactly the sequence it drew before.
+        r_side = max(1, d_side // 4)
+        with torch.random.fork_rng(devices=[]):
+            torch.manual_seed((torch.initial_seed() ^ INTERVENTION_INIT_SEED_MASK) + 2)
+            self.row_readout = PassageAttention(d_side, r_side)
+        with torch.random.fork_rng(devices=[]):
+            torch.manual_seed((torch.initial_seed() ^ INTERVENTION_INIT_SEED_MASK) + 3)
+            self.recursive_readout = PassageAttention(d_side, r_side)
+        self.pooling: WindowedPooling | None = None
+        if config.input_pooling_stride:
+            with torch.random.fork_rng(devices=[]):
+                torch.manual_seed((torch.initial_seed() ^ INTERVENTION_INIT_SEED_MASK) + 4)
+                self.pooling = WindowedPooling(d_side, config.mixer_heads, config.input_pooling_window, config.input_pooling_stride)
 
+    @property
+    def is_deep(self) -> bool:
+        return bool(self.intervention_layers)
+
+    @property
+    def intervention_modules(self) -> list[nn.Module]:
+        """The per-layer delta heads in layer order."""
+        return [self.delta_heads[str(layer)] for layer in self.intervention_layers] if self.delta_heads is not None else []
 
 
 class ActivationContextModel:
@@ -159,11 +234,13 @@ class ActivationContextModel:
         self.name = config.ac_model_name
         self.side = harness.loaded_models[config.base_side_model_name]
         self.target = harness.loaded_models[config.target_model_name]
-        if self.side is self.target:
-            raise ValueError("AC side and target require separate LoadedModel objects (separate loads of one model ID are allowed)")
+        if config.base_side_model_lora_name == config.target_model_lora_name:
+            raise ValueError("AC side and reader require distinct LoRA adapters")
         self.d_side = self.side.model_config.model_description.d_model
         self.d_target = self.target.model_config.model_description.d_model
-        self.modules = ActivationContextModules(config, self.d_side, self.d_target)
+        self.intervention_layers = self._resolve_intervention_layers()
+        self.modules = ActivationContextModules(config, self.d_side, self.d_target, self.intervention_layers)
+        self.intervention_diagnostics: dict[str, dict[int | str, list[torch.Tensor]]] = {"cosine": {}, "delta_rms": {}}   # per layer, per encoded example: cosine(delta, input row) and the delta RMS; the trainer drains them
         self.version = 0
         self.mode = ROLLOUT
         self.cache = RowCache(harness.harness_config.ac_row_cache_bytes)
@@ -198,6 +275,8 @@ class ActivationContextModel:
             self.queue = None
         self.mode = mode
         self.modules.train(mode == TRAINING)
+        for values in self.intervention_diagnostics.values():
+            values.clear()                                    # the training trace drains them; rollout never records them
 
     @property
     def device(self) -> torch.device:
@@ -250,8 +329,34 @@ class ActivationContextModel:
         """The AC modules' parameters (the side LoRA's come from the module manager)."""
         return list(self.modules.parameters())
 
+    # ------------------------------------------------------------------------------------------ deep inputs
+    def _resolve_intervention_layers(self) -> list[int]:
+        from ..harness.model_config import LayerType
+        layers = self.target.model_config.model_description.layer_descriptions
+        full = [layer.layer_idx for layer in layers if layer.layer_type is LayerType.FULL_ATTENTION]
+        if self.config.intervention_layers is not None:            # explicit layers: validated against the reader's depth
+            wrong = [layer for layer in self.config.intervention_layers if not 0 < layer < len(layers)]
+            if wrong:
+                raise ValueError(f"intervention_layers {wrong} lie outside the reader's interior layers 1..{len(layers) - 1}")
+            return list(self.config.intervention_layers)
+        return resolve_intervention_layers(len(layers), self.config.intervention_frequency, self.config.add_last_layer_intervention, full)
+
+    @property
+    def is_deep(self) -> bool:
+        return bool(self.intervention_layers)
+
+    def intervention_summary(self) -> dict:
+        """Layers and module sizes (for reports, checkpoints and result records); empty for an input-only model."""
+        if not self.is_deep:
+            return {}
+        return {"layers": list(self.intervention_layers), "source": "passage_attention", "attention_depth": "final", "target": "residual", "head_style": "lora",
+                "readouts": "rows,recursive,deltas", "recursive_keys": bool(self.config.recursive_keys),
+                "head_parameters": sum(parameter.numel() for parameter in self.modules.delta_heads.parameters()),
+                "attention_parameters": sum(parameter.numel() for parameter in self.modules.passage_attention.parameters())}
+
     def trainable_parameters(self) -> list[nn.Parameter]:
-        return self.parameters() + self.harness.module_manager.lora_parameters(self.config.base_side_model_lora_name)
+        """The AC modules' parameters plus the side adapter's."""
+        return [parameter for parameter in self.modules.parameters() if parameter.requires_grad] + self.harness.module_manager.lora_parameters(self.config.base_side_model_lora_name)
 
     # ------------------------------------------------------------------------------------------ encoding
     def encode(self, messages: list[dict] | str, compression_ratio: float | None = None, is_recursive: bool = False,
@@ -269,11 +374,14 @@ class ActivationContextModel:
         ratio = self.config.default_compression_ratio if compression_ratio is None else float(compression_ratio)
         return self.queue.submit(EncodeRequest(normalize_messages(messages), ratio, is_recursive, tools or None))
 
-    def encode_batch(self, requests: list[EncodeRequest]) -> list[torch.Tensor]:
-        """The rows of every request, cache first in rollout mode, the rest encoded together (children first)."""
+    def encode_batch(self, requests: list[EncodeRequest], *, with_layers: bool = False) -> list[torch.Tensor] | list[ACOutput]:
+        """The rows of every request, cache first in rollout mode, the rest encoded together (children first).
+        With `with_layers`, each result is the complete ACOutput (rows plus the deep model's per-layer deltas)."""
         if not requests:
             return []
-        results: list[torch.Tensor | None] = [None] * len(requests)
+        if self.is_deep and not with_layers:
+            raise RuntimeError(f"{self.name} is a deep AC model: encode_batch(with_layers=True) returns its rows with their deltas; a rows-only consumer would drop them silently")
+        results: list[ACOutput | None] = [None] * len(requests)
         pending: list[int] = []
         for index, request in enumerate(requests):
             cached = self.cache.get(row_cache_key(request.messages, request.compression_ratio, request.is_recursive, self.version, request.tools)) if self.mode == ROLLOUT else None
@@ -298,7 +406,8 @@ class ActivationContextModel:
         if self.queue is not None:
             self.stats.queue_batches, self.stats.queue_requests = self.queue.batches, self.queue.requests
         device = self.device
-        return [rows.to(device=device, dtype=torch.float32) for rows in results]
+        outputs = [output.to(device, torch.float32) for output in results]
+        return outputs if with_layers else [output.input_embeds for output in outputs]
 
     def _child_request(self, part: dict, parent_ratio: float) -> EncodeRequest:
         ac_name = part.get("ac_name") or self.name
@@ -307,7 +416,7 @@ class ActivationContextModel:
         return EncodeRequest(normalize_messages(part.get("messages") or []), float(ratio) if ratio is not None else parent_ratio, True,
                              part.get("tools") or None)
 
-    def _encode_requests(self, requests: list[EncodeRequest]) -> list[torch.Tensor]:
+    def _encode_requests(self, requests: list[EncodeRequest]) -> list[ACOutput]:
         """Children of every request (one recursive batch), then the requests in token-budgeted padded batches."""
         device = self.prepare()
         child_requests: list[EncodeRequest] = []
@@ -316,14 +425,20 @@ class ActivationContextModel:
             parts = direct_parts(request.messages)
             child_slices.append((len(child_requests), len(child_requests) + len(parts)))
             child_requests.extend(self._child_request(part, request.compression_ratio) for part in parts)
-        child_rows = self.encode_batch(child_requests) if child_requests else []
+        child_outputs = self.encode_batch(child_requests, with_layers=True) if child_requests else []
         prepared: list[PreparedPart] = []
         for request, (start, end) in zip(requests, child_slices):
-            rows = child_rows[start:end]
+            outputs = child_outputs[start:end]
+            rows = [output.input_embeds for output in outputs]
+            states: list[torch.Tensor] = []
+            if self.config.recursive_keys:
+                if any(output.states is None for output in outputs):
+                    raise RuntimeError("recursive_keys needs the children's final states (a recursive result without `states`)")
+                states = [output.states for output in outputs]
             ids, spans = tokenize_with_parts(self.side.tokenizer, request.messages, [r.shape[0] for r in rows], self.pad_id, tools=request.tools)
-            prepared.append((ids, spans, rows, self.num_view_rows(len(ids), request.compression_ratio), request.is_recursive))
+            prepared.append((ids, spans, rows, self.num_view_rows(len(ids), request.compression_ratio), request.is_recursive, states))
         order = sorted(range(len(prepared)), key=lambda index: len(prepared[index][0]))
-        results: list[torch.Tensor | None] = [None] * len(prepared)
+        results: list[ACOutput | None] = [None] * len(prepared)
         batch: list[int] = []
         longest = 0
         for index in order:
@@ -347,7 +462,7 @@ class ActivationContextModel:
         self.stats.phase_seconds[name] = self.stats.phase_seconds.get(name, 0.0) + now - started
         return now
 
-    def _encode_prepared_batch(self, items: list[PreparedPart], indexes: list[int], results: list[torch.Tensor | None], device: torch.device) -> None:
+    def _encode_prepared_batch(self, items: list[PreparedPart], indexes: list[int], results: list[ACOutput | None], device: torch.device) -> None:
         started = time.time()
         phase_started = started
         modules = self.modules
@@ -359,7 +474,7 @@ class ActivationContextModel:
         max_length = max(lengths)
         x = torch.zeros(batch_size, max_length, self.d_side, device=device, dtype=torch.float32)
         valid = torch.zeros(batch_size, max_length, device=device, dtype=torch.bool)
-        for row, (ids, spans, child_rows, _, _) in enumerate(items):
+        for row, (ids, spans, child_rows, _, _, _) in enumerate(items):
             with torch.no_grad():
                 embedded = embedding(torch.tensor(ids, device=device)).float()
             if spans:                                                                              # child rows over their placeholder runs
@@ -377,7 +492,7 @@ class ActivationContextModel:
         x = x + modules.position_scale * sinusoidal_positions(max_length, self.d_side, device)[None]
         phase_started = self._phase("embed", phase_started, device)
         autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
-        def run(module: nn.Module, *args: torch.Tensor | int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
+        def run(module: nn.Module, *args: torch.Tensor | int) -> torch.Tensor:
             if self.mode == TRAINING and torch.is_grad_enabled() and self.config.side_gradient_checkpointing:
                 return checkpoint(module, *args, use_reentrant=False)
             return module(*args)
@@ -386,14 +501,14 @@ class ActivationContextModel:
             for layer in modules.mixer:
                 x = run(layer, x, valid)
             phase_started = self._phase("mixer", phase_started, device)
-            if modules.pooling is not None:
+            if modules.pooling is not None:                                 # content pooling: about 1/stride of the tokens enter the decoder
                 pooled, window_valid = run(modules.pooling, x, valid)
-            phase_started = self._phase("pooling", phase_started, device)
+                phase_started = self._phase("pooling", phase_started, device)
             sequences = []
-            for row, (_, _, _, num_rows, _) in enumerate(items):
+            for row, (_, _, _, num_rows, _, _) in enumerate(items):
                 mixed = x[row, :lengths[row]]
                 content = mixed if modules.pooling is None else pooled[row][window_valid[row]]
-                summaries = run(modules.summary_pooling, mixed, num_rows)
+                summaries = run(modules.summary_pooling, mixed, num_rows)   # the summaries always pool from the unpooled content
                 summaries = (unit_rows(summaries) + modules.summary_marker
                              + modules.position_scale * sinusoidal_positions(num_rows, self.d_side, device))
                 sequences.append(torch.cat([unit_rows(content), unit_rows(summaries)], dim=0) * modules.input_scale)
@@ -410,13 +525,35 @@ class ActivationContextModel:
             attention_mask = torch.zeros(batch_size, max_seq, device=device, dtype=torch.long)
             for row, length in enumerate(seq_lengths):
                 attention_mask[row, :length] = 1
-        hidden = self.side.decoder_forward(inputs_embeds.to(base_dtype), attention_mask, lora_name=self.config.base_side_model_lora_name)
+        enabled = (self.mode == TRAINING and torch.is_grad_enabled() and self.config.side_gradient_checkpointing
+                   and getattr(self, "_training_gradient_checkpointing", True)
+                   and max_seq >= getattr(self, "_side_gradient_checkpointing_min_tokens", 0))
+        hidden = self.side.decoder_forward(inputs_embeds.to(base_dtype), attention_mask,
+                                           lora_name=self.config.base_side_model_lora_name, gradient_checkpointing=enabled)
         phase_started = self._phase("side_decoder", phase_started, device)
-        for row, (ids, _, _, num_rows, is_recursive) in enumerate(items):
+        for row, (ids, _, _, num_rows, is_recursive, child_states) in enumerate(items):
             end = seq_lengths[row]
             rows = hidden[row, end - num_rows:end].float()
-            rows = modules.recursive_adapter(rows) if is_recursive else modules.target_head(rows)
-            results[indexes[row]] = rows
+            # The level's key set: the passage positions' final side states (one fp32 copy per example, shared by every read-out)
+            # plus, with recursive_keys, the final states its children returned (the parent reads a child at the child's granularity).
+            keys = hidden[row, :end - num_rows].float()
+            if child_states:
+                keys = torch.cat([keys, *[states.to(keys.dtype) for states in child_states]], dim=0)
+            if is_recursive:
+                results[indexes[row]] = ACOutput(modules.recursive_adapter(modules.recursive_readout(rows, keys)), states=rows if self.config.recursive_keys else None)   # states ride the cache only when a parent reads them
+            else:
+                input_embeds = modules.target_head(modules.row_readout(rows, keys))
+                deltas = {}
+                if modules.is_deep:
+                    for layer in self.intervention_layers:
+                        source = modules.passage_attention[str(layer)](rows, keys)
+                        deltas[layer] = modules.delta_heads[str(layer)](source)
+                        if self.mode == TRAINING:                              # does the delta re-inject the row? cosine with the input row, per layer; and the delta's RMS (0-d tensors, read at the trace point)
+                            with torch.no_grad():
+                                cosine = F.cosine_similarity(deltas[layer].detach(), input_embeds.detach(), dim=-1).mean()
+                            self._record_diagnostic("cosine", layer, cosine)
+                            self._record_diagnostic("delta_rms", layer, self._row_rms(deltas[layer]))
+                results[indexes[row]] = ACOutput(input_embeds, deltas)
             self.stats.encodes += 1
             self.stats.side_tokens += len(ids)
             self.stats.view_rows += num_rows
@@ -424,6 +561,17 @@ class ActivationContextModel:
         self.stats.encode_batches += 1
         self.stats.encode_seconds.append(time.time() - started)
 
+    @staticmethod
+    def _row_rms(rows: torch.Tensor) -> torch.Tensor:
+        """Mean over rows of the per-row RMS, detached (0-d tensor: no host sync)."""
+        with torch.no_grad():
+            return rows.detach().float().pow(2).mean(dim=-1).sqrt().mean()
+
+    def _record_diagnostic(self, name: str, key: int | str, value: torch.Tensor) -> None:
+        values = self.intervention_diagnostics[name].setdefault(key, [])
+        values.append(value)
+        del values[:-DIAGNOSTIC_CAPACITY]                     # bounded when nothing drains it (eval-only use of a training-mode model)
+
     # ------------------------------------------------------------------------------------------ checkpoints
     def invalidate_encoded_rows(self) -> None:
         """An encoder update changes cache identity even when checkpointing is disabled or fails."""
@@ -441,11 +589,12 @@ class ActivationContextModel:
             folder = resolve_path(checkpoint_path, create=False)
         self.harness.module_manager.ensure_lora(self.config.base_side_model_lora_name)
         folder.mkdir(parents=True, exist_ok=True)
-        torch.save({"modules": self.modules.state_dict(), "config": dataclasses.asdict(self.config), "version": self.version,
+        torch.save({"modules": self.modules.state_dict(), "config": {**dataclasses.asdict(self.config), "row_head_skip": self.modules.target_head.skip}, "version": self.version,
                     "d_side": self.d_side, "d_target": self.d_target, "architecture_version": ARCHITECTURE_VERSION,
                     "model_ids": [self.side.model_config.model_id, self.target.model_config.model_id],
                     "side_lora_targets": list(self.harness.module_manager.get_lora_config(self.config.base_side_model_lora_name).target_modules),
-                    "epoch_number": self.epoch_number if epoch_number is None else epoch_number}, folder / MODULES_FILE)
+                    "epoch_number": self.epoch_number if epoch_number is None else epoch_number,
+                    **({"interventions": self.intervention_summary()} if self.is_deep else {})}, folder / MODULES_FILE)
         module_manager = self.harness.module_manager
         module_manager.save_lora(self.config.base_side_model_name, self.config.base_side_model_lora_name, str(folder / SIDE_LORA_FOLDER))
         if not checkpoint_path:
@@ -459,7 +608,9 @@ class ActivationContextModel:
         return str(folder)
 
     def load(self, checkpoint_path: str) -> None:
-        """Modules from the folder; the side LoRA re-injects from the folder's adapter on next use."""
+        """Modules from the folder; the side LoRA re-injects from the folder's adapter on next use. An input-only checkpoint loads
+        into a deep model (the shared modules loaded, the intervention modules at their seeded init); a deep checkpoint loads into
+        the same intervention layers only."""
         folder = Path(checkpoint_path)
         if not folder.is_absolute():
             folder = resolve_path(checkpoint_path, create=False)
@@ -469,14 +620,61 @@ class ActivationContextModel:
         expected_ids = [self.side.model_config.model_id, self.target.model_config.model_id]
         if state.get("model_ids") != expected_ids or state["d_side"] != self.d_side or state["d_target"] != self.d_target:
             raise ValueError("Checkpoint model identities or widths differ from the loaded bases")
-        structural = ("input_pooling_stride", "input_pooling_window", "mixer_layers", "mixer_window", "mixer_heads", "min_view_rows", "max_view_rows", "side_lora_rank", "default_compression_ratio")
+        structural = ("mixer_layers", "mixer_window", "mixer_heads", "min_view_rows", "max_view_rows", "side_lora_rank", "default_compression_ratio")
         differences = [name for name in structural if state["config"].get(name) != getattr(self.config, name)]
+        saved_stride = int(state["config"].get("input_pooling_stride", 0) or 0)
+        # An unpooled checkpoint into a pooled model is a migration (the pooler is a new module, trained from its init on top of the
+        # loaded encoder: that is how the strided arms warm-start). Every other stride change is refused, the reverse direction included.
+        pooling_fresh = saved_stride == 0 and self.config.input_pooling_stride > 0
+        if saved_stride != self.config.input_pooling_stride and not pooling_fresh:
+            differences.append(f"input_pooling_stride (checkpoint {saved_stride}, model {self.config.input_pooling_stride})")
+        elif saved_stride and int(state["config"].get("input_pooling_window", 0) or 0) != self.config.input_pooling_window:
+            differences.append("input_pooling_window")
+        if bool(state["config"].get("row_head_skip", False)) != self.modules.target_head.skip:
+            differences.append("row_head_skip")
+        if state["config"].get("encoder_passes", 1) != 1 or state["config"].get("pool_surprisal", 0.0):
+            differences.append("encoder (iterative passes or surprisal pooling are no longer part of the model)")
+        saved_interventions = state.get("interventions") or {}
+        saved_layers = [int(layer) for layer in saved_interventions.get("layers", [])]   # absent in input-only checkpoints
+        migrate = self.is_deep and not saved_layers          # input-only checkpoint into a deep model: shared modules loaded, intervention modules fresh
+        if not migrate and saved_layers != self.intervention_layers:
+            raise ValueError(f"Checkpoint intervention layers {saved_layers} differ from the configured {self.intervention_layers} (a deep checkpoint loads into the same layers only)")
+        if saved_layers:
+            saved_shape = (saved_interventions.get("source", "summary"), saved_interventions.get("attention_depth"),
+                           saved_interventions.get("target", "residual"), saved_interventions.get("head_style", "scaled"))
+            if saved_shape != ("passage_attention", "final", "residual", "lora"):
+                raise ValueError(f"Checkpoint interventions {saved_shape} are not the model's design (passage attention over the final side states, "
+                                 "zero-initialized residual deltas): that checkpoint belongs to the full-featured branch")
         targets = list(self.harness.module_manager.get_lora_config(self.config.base_side_model_lora_name).target_modules)
         if differences or state.get("side_lora_targets") != targets:
             raise ValueError(f"Checkpoint AC configuration differs: {differences or ['side_lora_targets']}")
         side_lora = folder / SIDE_LORA_FOLDER
         self._validate_adapter(side_lora, self.config.base_side_model_lora_name)
-        self.modules.load_state_dict(state["modules"])
+        modules_state = dict(state["modules"])
+        for name in list(modules_state):        # inert calibration state that earlier deep checkpoints carried
+            if name in ("intervention_calibration", "interventions_calibrated") or (name.startswith("delta_heads.") and name.split(".")[-1] in ("relative", "rms")):
+                modules_state.pop(name)
+        # Two kinds of keys may be absent: the intervention modules when an input-only checkpoint migrates into a deep model, and the
+        # row read-outs when the checkpoint predates them (identity at init: the outputs are the checkpoint's). Anything else raises.
+        deep_keys = {name for name in self.modules.state_dict() if name.startswith(INTERVENTION_MODULE_PREFIXES)} if migrate else set()
+        readout_keys = {name for name in self.modules.state_dict() if name.startswith(READOUT_MODULE_PREFIXES)}
+        pooling_keys = {name for name in self.modules.state_dict() if name.startswith(POOLING_MODULE_PREFIX)} if pooling_fresh else set()
+        expected = set(self.modules.state_dict()); saved = set(modules_state)
+        missing, unexpected = expected - saved, saved - expected               # decided before anything is written: a mismatch leaves the modules untouched
+        fresh_readouts = missing & readout_keys
+        if (missing - deep_keys - readout_keys - pooling_keys) or unexpected or (fresh_readouts and fresh_readouts != readout_keys) \
+                or (migrate and not deep_keys <= missing) or (pooling_fresh and not pooling_keys <= missing):
+            raise ValueError(f"Checkpoint modules do not match (missing {sorted(missing - deep_keys - readout_keys - pooling_keys)}, unexpected {sorted(unexpected)})")
+        self.modules.load_state_dict(modules_state, strict=False)
+        if bool(state["config"].get("recursive_keys", False)) != self.config.recursive_keys:
+            print(f"{self.name} - Note: the checkpoint was trained with recursive_keys={state['config'].get('recursive_keys', False)}; this model uses {self.config.recursive_keys} (the read-outs see a different key set)")
+        if migrate:
+            print(f"{self.name} - Migrated an input-only checkpoint into a deep model: shared modules loaded, {len(self.intervention_layers)} intervention layers fresh (seeded init)")
+        if fresh_readouts:
+            print(f"{self.name} - Row read-outs fresh (identity at init): the checkpoint predates the read-outs in front of the row heads")
+        if pooling_fresh:
+            print(f"{self.name} - Content pooling fresh (stride {self.config.input_pooling_stride}, window {self.config.input_pooling_window}): "
+                  "the checkpoint was trained on unpooled content, the pooler starts at its init")
         self.version = max(self.version, int(state.get("version", 0)))
         self.epoch_number = int(state.get("epoch_number", 0))
         self.invalidate_encoded_rows()
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/ac_model/ac_model_reporter.py">activation/ac_model/ac_model_reporter.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/ac_model/ac_model_reporter.py`

```diff
diff --git a/activation/ac_model/ac_model_reporter.py b/activation/ac_model/ac_model_reporter.py
index 762b648..6cfb6f7 100644
--- a/activation/ac_model/ac_model_reporter.py
+++ b/activation/ac_model/ac_model_reporter.py
@@ -1,4 +1,4 @@
-"""AC epoch progress, exact evaluation, training diagnostics and retained greedy samples."""
+"""AC training progress, evaluation curves, gold-answer log-loss, training diagnostics and retained greedy samples."""
 from __future__ import annotations
 
 import json
@@ -12,84 +12,218 @@ if t.TYPE_CHECKING:
         ActivationContextTrainingProgress, ActivationContextEvalSummary, ActivationContextEpochStats, CompletionSample, ReferenceName)
 
 REFERENCE_SERIES = ("no_context", "recent_text", "untrained_ac", "in_context")
+TEACHER_SERIES = "teacher (stored at training start)"
 
 
 class ActivationContextTrainingReporter(HtmlReporter):
-    def __init__(self, folder: str, title: str = "AC training", description: str = "", **kwargs: t.Any) -> None:
+    """
+    The page a training call writes. Widgets, in reading order: gold-answer log-loss of every evaluation set and the
+    stored teacher, evaluation loss (KL to the stored teacher, or cross-entropy), training loss per update, then the
+    tables (evaluations, epochs, per kind, references) and the diagnostics (agreement, gradient norm, throughput).
+    `set_labels` renames evaluation sets in plots and tables, e.g. {"reporting": "held-out, AC student"}.
+    """
+
+    def __init__(self, folder: str, title: str = "AC training", description: str = "", set_labels: dict[str, str] | None = None,
+                 **kwargs: t.Any) -> None:
         super().__init__(folder, title, description, eyebrow="AC training", **kwargs)
+        self.loss_kind: str | None = None
         self.references: dict[str, float] = {}
+        self.set_labels = dict(set_labels or {})
         self.global_step = 0
         self.epoch_x = 0.0
         self.epoch_origin: int | None = None
-        self.initialize_line_plot("kl", "Training KL", "Weighted item mean; cached teachers use coarsened KL.",
-                                  "global optimizer step", "KL", ["train"], smoothing_window=5)
-        self.initialize_line_plot("reporting", "Exact held-out KL", "Current-adapter reporting/validation, with labeled initial-adapter references.",
-                                  "epochs in this report", "KL", ["reporting", "validation", *REFERENCE_SERIES])
-        self.initialize_line_plot("agreement", "Top-1 agreement", "Mean teacher/student agreement over items.", "global optimizer step", "fraction", ["train"])
-        self.initialize_line_plot("throughput", "Token-equivalent throughput", "Original teacher + student tokens per training second; cached teacher tokens are equivalent work, not executed tokens.",
-                                  "global optimizer step", "tokens/s", ["tokens/s"])
-        self.initialize_line_plot("grad", "Gradient norm", "Before clipping.", "global optimizer step", "norm", ["grad_norm"])
-        self.initialize_table("epochs", "Epochs", "Phase times in seconds.",
-                              ["epoch", "global step", "train s", "report s", "validation s", "samples s", "checkpoint s", "total s", "checkpoint"])
-        self.initialize_table("kinds", "Per kind", "Exact reporting/validation results.", ["epoch", "set", "kind", "items", "kl", "agreement"])
-        self.initialize_table("evals", "Evaluations", "Exact current-adapter results; empty and dropped sets are counted.",
-                              ["epoch", "global step", "name", "items", "dropped", "kl", "agreement"])
-        self.initialize_table("references", "Reference measurements", "QA recent_text is oracle gold text; compaction uses the text tail.", ["name", "KL", "measured with"])
+        self.planned_updates: int | None = None
+        self._interventions_reported = 0
+        self.initialize_line_plot("gold_nll", "Gold-answer log-loss",
+                                  "Negative log-likelihood per target token of the reference completion, item mean. "
+                                  "Every reader is scored on the same tokens; lower is better and the teacher is not a ceiling.",
+                                  "epochs in this report", "nats per target token", [])
+        self.initialize_line_plot("reporting", "Evaluation loss", "Named evaluation sets and initial-reader references.",
+                                  "epochs in this report", "loss", [*REFERENCE_SERIES])
+        self.initialize_line_plot("loss", "Training loss per update", "Weighted item mean of the update's items.", "global update", "loss", ["train"], smoothing_window=5)
+        self.initialize_table("evals", "Evaluations", "One row per evaluation of a named set.",
+                              ["epoch", "global update", "set", "items", "loss kind", "loss", "agreement", "gold NLL", "token acc", "teacher gold NLL"])
+        self.initialize_table("epochs", "Epochs", "Phase times in seconds; the checkpoint column names what was actually written.",
+                              ["epoch", "global update", "train s", "panels s", "validation s", "samples s", "checkpoint s", "total s", "checkpoint"])
+        self.initialize_table("kinds", "Per kind", "Evaluation results by item kind.", ["epoch", "set", "kind", "items", "loss kind", "loss", "agreement", "gold NLL", "token acc"])
+        self.initialize_table("references", "Reference measurements", "QA recent_text is oracle gold text; compaction uses the text tail.", ["name", "loss kind", "loss", "measured with"])
+        self.initialize_line_plot("throughput", "Token throughput", "Tokens per training second.", "global update", "tokens/s", ["tokens/s"])
+        self.initialize_line_plot("grad", "Gradient norm", "Before clipping.", "global update", "norm", ["grad_norm"])
+
+    # ------------------------------------------------------------------------------------------ helpers
+    def label(self, name: str) -> str:
+        return self.set_labels.get(name, name)
+
+    def _ensure_series(self, widget: str, series: str) -> None:
+        if series not in {entry["name"] for entry in self.widgets[widget]["series"]}:
+            self.widgets[widget]["series"].append({"name": series, "x": [], "y": []})
 
+    def _clear_status(self, *keys: str) -> None:
+        for key in keys:
+            self.status.pop(key.replace("_", " "), None)
+
+    def _set_loss_kind(self, loss_kind: str) -> None:
+        if self.loss_kind == loss_kind:
+            return
+        if loss_kind not in ("kl", "sft") or self.loss_kind is not None:
+            raise ValueError("Use a separate AC reporter for each loss kind")
+        self.loss_kind = loss_kind
+        if loss_kind == "kl":
+            self.widgets["reporting"].update(title="KL to the stored teacher", y_label="KL",
+                description="Top-k coarsened KL, teacher || student, per evaluation set; the teacher is the reader as it was at training start. "
+                            "Zero would mean the student matches it exactly.")
+            self.widgets["loss"].update(title="Training KL per update", y_label="KL",
+                                        description="Coarsened KL to the stored teacher on the update's items; smoothed over 5 updates.")
+            self.initialize_line_plot("agreement", "Top-1 agreement", "Mean teacher/student agreement over items (token agreement, not answer accuracy).",
+                                      "global update", "fraction", ["train"])
+        else:
+            self.widgets["reporting"].update(title="Evaluation cross-entropy", y_label="cross-entropy",
+                                             description="Mean cross-entropy over the target tokens per evaluation set.")
+            self.widgets["loss"].update(title="Training cross-entropy per update", y_label="cross-entropy",
+                                        description="Weighted mean cross-entropy over retained assistant tokens; smoothed over 5 updates.")
+            self.remove_widget("agreement")
+            for name in ("kinds", "evals"):
+                self.widgets[name]["columns"] = [column for column in self.widgets[name]["columns"] if column not in ("agreement", "teacher gold NLL")]
+
+    # ------------------------------------------------------------------------------------------ trainer callbacks
     def initialize_training(self, config: "ActivationContextTrainingConfig", stats: "ActivationContextTrainingStats") -> None:
+        self._set_loss_kind(config.loss_kind)
         if self.epoch_origin is None:
             self.epoch_origin = stats.start_epoch - 1
         self.epoch_x = float(stats.start_epoch - 1 - self.epoch_origin)
-        self.set_status(epoch=f"0/{config.num_epochs}", phase="baseline", examples=str(stats.examples), dropped=str(stats.dropped_too_long),
-                        completion_tokens=str(stats.completion_tokens), teacher_tokens=str(stats.teacher_tokens))
+        self.set_status(phase="baseline", epoch=f"0 of {config.num_epochs}", items=str(stats.examples))
         self.set_text("config", "Training config", json.dumps(asdict(config), indent=2))
         self.render()
 
+    def report_teacher_targets(self, targets: dict) -> None:
+        coverage = targets.get("coverage")
+        self.set_status(teacher_targets=f"{targets.get('examples', 0)} items · top-{targets.get('top_k')}"
+                                        + (f" · {coverage:.1%} of the mass" if coverage is not None else ""))
+        self.render(force=True)
+
     def report_phase(self, progress: "ActivationContextTrainingProgress", phase: str) -> None:
         self.global_step = progress.global_step
         if self.epoch_origin is None:
             self.epoch_origin = progress.epoch_number if progress.epoch == 0 else progress.epoch_number - 1
         completed = progress.epoch_number if progress.epoch == 0 else progress.epoch_number - 1 + progress.step / max(1, progress.steps)
         self.epoch_x = completed - self.epoch_origin
-        self.set_status(epoch=f"{progress.epoch}/{progress.epochs}", checkpoint_epoch=str(progress.epoch_number), step=f"{progress.step}/{progress.steps}",
-                        global_step=str(progress.global_step), phase=phase, epoch_elapsed=f"{progress.epoch_elapsed_s:.1f}s", elapsed=f"{progress.elapsed_s:.1f}s")
+        if self.planned_updates is None and progress.epoch:
+            self.planned_updates = progress.global_step - ((progress.epoch - 1) * progress.steps + progress.step) + progress.epochs * progress.steps
+        done = (progress.epoch - 1) * progress.steps + progress.step if progress.epoch else 0
+        total = progress.epochs * progress.steps
+        remaining = (progress.elapsed_s / done * (total - done)) if done and total > done else None
+        self.set_status(phase=phase, epoch=f"{progress.epoch} of {progress.epochs}", update=f"{progress.step} of {progress.steps} in this epoch",
+                        global_updates=f"{progress.global_step}" + (f" of {self.planned_updates}" if self.planned_updates else ""),
+                        elapsed=_duration(progress.elapsed_s),
+                        remaining=("about " + _duration(remaining)) if remaining is not None else ("done" if done == total and total else "estimating"))
+        if phase == "training":
+            self._clear_status("evaluation_items")
         self.render()
 
     def report_step(self, progress: "ActivationContextTrainingProgress", stats: "ActivationContextTrainingStats") -> None:
+        self._set_loss_kind(stats.loss_kind)
         self.report_phase(progress, "training")
-        self.add_data_point("kl", {"x": progress.global_step, "train": stats.kl[-1]})
-        self.add_data_point("agreement", {"x": progress.global_step, "train": stats.agreement[-1]})
+        self.add_data_point("loss", {"x": progress.global_step, "train": stats.loss[-1]})
+        if stats.loss_kind == "kl":
+            self.add_data_point("agreement", {"x": progress.global_step, "train": stats.agreement[-1]})
         self.add_data_point("throughput", {"x": progress.global_step, "tokens/s": stats.step_tokens[-1] / max(stats.step_seconds[-1], 1e-6)})
         self.add_data_point("grad", {"x": progress.global_step, "grad_norm": stats.grad_norm[-1]})
+        if stats.penalty:
+            if "penalty" not in self.widgets:
+                self.initialize_line_plot("penalty", "No-context penalty",
+                    "KL of the adapted reader to the base on the same items without their rows (unweighted, sampled examples); "
+                    "a rising curve means the reader improves without the rows.", "global update", "KL", ["train"])
+            self.add_data_point("penalty", {"x": progress.global_step, "train": stats.penalty[-1]})
+            self.set_status(no_context_penalty=f"{stats.penalty[-1]:.4f}")
         if self.references:
             self.add_data_point("reporting", {"x": self.epoch_x, **self.references})
-        self.set_status(kl=f"{stats.kl[-1]:.4f}", agreement=f"{stats.agreement[-1]:.3f}", learning_rates=", ".join(f"{lr:.3g}" for lr in stats.learning_rates[-1]))
+        if stats.intervention_trace and len(stats.intervention_trace) != self._interventions_reported:
+            entry = stats.intervention_trace[-1]
+            self._interventions_reported = len(stats.intervention_trace)
+            if "intervention_grads" not in self.widgets:
+                self.initialize_line_plot("intervention_grads", "Delta head gradient norms",
+                    "Per intervention layer, before clipping. The delta is zero at init, so this starts small and must climb; a monotone decay toward zero means the path is dying.",
+                    "global update", "norm", [])
+                self.initialize_line_plot("intervention_delta_rms", "Applied delta RMS relative to the residual at the rows",
+                    "Per intervention layer, the RMS of the delta rows the reader received (mean over the encoded rows since the last point) divided by the residual RMS "
+                    "measured at the row positions on the same training forwards. In bf16 the addition rounds away below ~0.4 %; watch it cross ~1 % and settle in the 2-10 % band.",
+                    "global update", "delta RMS / row RMS", [])
+                self.initialize_line_plot("intervention_row_rms", "Residual RMS at the row positions",
+                    "Per intervention layer, measured by the reader hooks on the training forwards (reader as trained, rows present): the deltas' denominator.", "global update", "RMS", [])
+                self.initialize_line_plot("intervention_cosine", "Delta vs input row",
+                    "Per intervention layer, mean cosine between the delta rows and the input rows over the encoded rows since the last point: "
+                    "near 1 means the delta re-injects the input row; near 0, a different direction.", "global update", "cosine", [])
+                self.initialize_line_plot("intervention_attention", "Read-out blocks: out-projection weight norm",
+                    "Per intervention layer, the norm of the zero-initialized output projection of the row-over-passage attention block in the encoder: "
+                    "0 at init (the head reads the row's own state), rising as the attention path opens.", "global update", "norm", [])
+            for widget, key in (("intervention_grads", "head_grad_norm"), ("intervention_row_rms", "row_rms"), ("intervention_cosine", "cosine"),
+                                ("intervention_attention", "attention_out_norm"), ("intervention_delta_rms", "delta_rms")):
+                for layer, value in (entry.get(key) or {}).items():
+                    if value is None:
+                        continue
+                    series = f"layer {layer}" if isinstance(layer, int) or str(layer).isdigit() else f"{layer} read-out"
+                    self._ensure_series(widget, series)
+                    self.add_data_point(widget, {"x": entry["update"], series: value})
+        self.set_status(loss=f"{stats.loss[-1]:.4f}", learning_rates=", ".join(f"{lr:.3g}" for lr in stats.learning_rates[-1]))
+        if stats.loss_kind == "kl":
+            self.set_status(agreement=f"{stats.agreement[-1]:.3f}")
         self.render()
 
-    def report_reference(self, name: "ReferenceName", kl: float, *, provenance: str = "initial adapter; held set") -> None:
+    def report_reference(self, name: "ReferenceName", loss: float, *, provenance: str = "initial adapter; held set", loss_kind: str = "kl") -> None:
         assert name in REFERENCE_SERIES, name
-        self.references[name] = kl
-        self.add_data_point("references", {"name": name, "KL": kl, "measured with": provenance})
-        self.add_data_point("reporting", {"x": 0, name: kl})
+        self._set_loss_kind(loss_kind)
+        self.references[name] = loss
+        self.add_data_point("references", {"name": name, "loss kind": loss_kind, "loss": loss, "measured with": provenance})
+        self.add_data_point("reporting", {"x": 0, name: loss})
         if self.epoch_x:
-            self.add_data_point("reporting", {"x": self.epoch_x, name: kl})
+            self.add_data_point("reporting", {"x": self.epoch_x, name: loss})
         self.render()
 
     def report_eval(self, progress: "ActivationContextTrainingProgress | None", name: str, summary: "ActivationContextEvalSummary") -> None:
+        self._set_loss_kind(summary["loss_kind"])
         if progress is not None:
             self.report_phase(progress, name)
-        epoch = progress.epoch_number - (self.epoch_origin or 0) if progress else "external"
-        self.add_data_point("evals", {"epoch": epoch, "global step": self.global_step, "name": name, "items": summary["items"],
-                                      "dropped": summary["dropped_too_long"], "kl": summary["kl"], "agreement": summary["agreement"]})
-        if name in ("reporting", "validation") and summary["items"]:
-            self.add_data_point("reporting", {"x": self.epoch_x, name: summary["kl"], **self.references})
+        epoch = (round(progress.epoch_number - (self.epoch_origin or 0) - (1 - progress.step / max(1, progress.steps)), 3)
+                 if progress and progress.epoch else (0 if progress else "external"))
+        label = self.label(name)
+        gold, teacher_gold = summary.get("gold_nll"), summary.get("teacher_gold_nll")
+        self.add_data_point("evals", {"epoch": epoch, "global update": self.global_step, "set": label, "items": summary["items"],
+                                      "loss kind": summary["loss_kind"], "loss": _round(summary["loss"]),
+                                      **({"agreement": _round(summary["agreement"]), "teacher gold NLL": _round(teacher_gold)} if self.loss_kind == "kl" else {}),
+                                      "gold NLL": _round(gold), "token acc": _round(summary.get("token_accuracy"))})
+        if summary["items"]:
+            self._ensure_series("reporting", label)
+            self.add_data_point("reporting", {"x": self.epoch_x, **self.references, label: summary["loss"]})
+            accuracy = summary.get("token_accuracy")
+            if accuracy is not None:
+                if "token_accuracy" not in self.widgets:
+                    self.initialize_line_plot("token_accuracy", "Teacher-forced token accuracy",
+                        "Share of target positions where the reader's most likely token is the gold token, per evaluation set.",
+                        "epochs in this report", "fraction", [])
+                self._ensure_series("token_accuracy", label)
+                self.add_data_point("token_accuracy", {"x": self.epoch_x, label: accuracy})
+            if gold is not None:
+                self._ensure_series("gold_nll", label)
+                point = {"x": self.epoch_x, label: gold}
+                if teacher_gold is not None and name == "reporting":
+                    self._ensure_series("gold_nll", TEACHER_SERIES)
+                    point[TEACHER_SERIES] = teacher_gold
+                self.add_data_point("gold_nll", point)
+            if name == "reporting":
+                self.set_status(last_panel=f"update {self.global_step}")
         for kind, values in summary["by_kind"].items():
-            self.add_data_point("kinds", {"epoch": epoch, "set": name, "kind": kind, **values})
+            self.add_data_point("kinds", {"epoch": epoch, "set": label, "kind": kind, "items": values["items"],
+                "loss kind": values["loss_kind"], "loss": _round(values["loss"]), "gold NLL": _round(values.get("gold_nll")), "token acc": _round(values.get("token_accuracy")),
+                **({"agreement": _round(values["agreement"])} if self.loss_kind == "kl" else {})})
         self.render()
 
+    def report_evaluation_progress(self, name: str, completed: int, total: int) -> None:
+        self.set_status(phase=self.label(name), evaluation_items=f"{completed} of {total}")
+        self.render(force=completed == 0 or completed == total)
+
     def report_completions(self, progress: "ActivationContextTrainingProgress", rows: list["CompletionSample"]) -> None:
-        text = "\n\n".join(f"[{r['item_id']}] ({r['kind']}, global step {r['global_step']}, AC version {r['ac_version']})\n"
-                           f"reference: {r['reference']}\nteacher: {r['teacher']}\nstudent: {r['student']}" for r in rows)
+        text = "\n\n".join(f"[{r['item_id']}] ({r['kind']}, global update {r['global_step']}, AC version {r['ac_version']})\n"
+                           f"reference: {r['reference']}\n" + (f"teacher: {r['teacher']}\n" if r["teacher"] is not None else "") + f"student: {r['student']}" for r in rows)
         label = "Baseline" if progress.epoch == 0 else f"Epoch {progress.epoch_number - (self.epoch_origin or 0)}"
         key = f"completions_{'baseline' if progress.epoch == 0 else 'epoch'}_{progress.epoch_number:03d}"
         self.set_text(key, f"{label} inspection samples", text)
@@ -97,18 +231,27 @@ class ActivationContextTrainingReporter(HtmlReporter):
 
     def report_epoch(self, epoch: "ActivationContextEpochStats", stats: "ActivationContextTrainingStats") -> None:
         self.report_phase(epoch.progress, "epoch complete")
-        self.add_data_point("epochs", {"epoch": epoch.progress.epoch_number - (self.epoch_origin or 0), "global step": epoch.progress.global_step,
-            "train s": round(epoch.training_seconds, 2), "report s": round(epoch.reporting_seconds, 2), "validation s": round(epoch.validation_seconds, 2),
+        self.add_data_point("epochs", {"epoch": epoch.progress.epoch_number - (self.epoch_origin or 0), "global update": epoch.progress.global_step,
+            "train s": round(epoch.training_seconds, 2), "panels s": round(epoch.reporting_seconds, 2), "validation s": round(epoch.validation_seconds, 2),
             "samples s": round(epoch.sample_seconds, 2), "checkpoint s": round(epoch.checkpoint_seconds, 2),
-            "total s": round(epoch.progress.epoch_elapsed_s, 2), "checkpoint": epoch.checkpoint_path or "disabled"})
-        self.set_text(f"epoch_{epoch.progress.epoch_number:03d}", "Epoch record", json.dumps(asdict(epoch), indent=2))
+            "total s": round(epoch.progress.epoch_elapsed_s, 2), "checkpoint": epoch.checkpoint_path or "not written (disabled)"})
+        self.set_status(last_checkpoint=(epoch.checkpoint_path or "none").rsplit("/", 1)[-1])
+        self.set_text(f"epoch_{epoch.progress.epoch_number:03d}", "Epoch record",
+                      json.dumps({key: value for key, value in asdict(epoch).items() if key not in ("reporting", "validation", "deltas_off", "samples")}, indent=2))
         self.render(force=True)
 
     def report_training(self, stats: "ActivationContextTrainingStats") -> None:
-        self.set_status(phase="complete", peak_memory=f"{stats.peak_memory_bytes / 2**30:.2f} GB", elapsed=f"{stats.duration_s:.1f}s")
-        groups = {name: [r["seconds"] for r in stats.example_records if r["teacher_cached"] == cached]
-                  for name, cached in (("live", False), ("cached", True))}
-        self.set_text("rates", "Live and cached teacher examples", json.dumps({name: {"examples": len(values), "seconds": sum(values),
-            "examples_per_second": len(values) / sum(values) if sum(values) else None} for name, values in groups.items()}, indent=2))
-        self.set_text("training_summary", "Training summary", json.dumps(stats.summarize(), indent=2))
+        self.set_status(phase="complete", peak_memory=f"{stats.peak_memory_bytes / 2**30:.2f} GB", elapsed=_duration(stats.duration_s), remaining="done")
+        self._clear_status("evaluation_items")
+        self.set_text("training_summary", "Training summary", json.dumps(stats.summarize(), indent=2, default=str))
         self.render(force=True)
+
+
+def _round(value: float | None, digits: int = 4) -> float | str:
+    return round(value, digits) if isinstance(value, (int, float)) else ""
+
+
+def _duration(seconds: float) -> str:
+    seconds = max(0, int(seconds))
+    hours, minutes = divmod(seconds // 60, 60)
+    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds % 60:02d}s"
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/ac_model/ac_model_study.py">activation/ac_model/ac_model_study.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/ac_model/ac_model_study.py`

```diff
diff --git a/activation/ac_model/ac_model_study.py b/activation/ac_model/ac_model_study.py
index 19a3d87..be3ab47 100644
--- a/activation/ac_model/ac_model_study.py
+++ b/activation/ac_model/ac_model_study.py
@@ -1,22 +1,4 @@
-"""
-Training items for the AC model, three kinds (ac_model_training.ActivationContextTrainingItem):
-
-- compaction: a trajectory cut at depth d (1-2) into segments of `threshold` tokens; the in-context
-  prefix is the real messages up to the last cut (an assistant turn may be cut mid-text), the AC
-  prefix nests the segments (`AC_d{AC_{d-1}{...}, segment d}`) in one user message followed by the
-  compaction instructions; the completion is the rest of the cut turn or the next assistant turn.
-  Trains the model to "see" its own compactions.
-- traj_qa: a study question about a trajectory chunk (dataset_study, `traj_qa`); the in-context
-  prefix holds the transcript window around the chunk, the AC prefix the window and 0-2 distractor
-  trajectories (bm25 top documents for the question, the source excluded) as separate parts, each
-  with the task text first so the compressor knows what the reader is after. Trains "seeing" a
-  subagent or parent.
-- rag_qa: a study question about a text chunk; the in-context prefix holds the gold passage, the AC
-  prefix one part with the gold and bm25 distractor passages shuffled to a token budget.
-
-Every item's completion is real text (the trajectory's own continuation, the gold answer): no
-sampling is needed at generation time. Token budgets are counted with the target tokenizer.
-"""
+"""Paired whole-history compaction/QA items and exact recorded-run self-distillation."""
 from __future__ import annotations
 
 import json
@@ -30,9 +12,10 @@ from ..agent.agent_utils import ModelDialect
 from ..dataset.dataset import DataModality, DatasetDocument, DatasetDocumentChunk, DatasetQAExample
 from ..dataset.loaders.trajectory_utils import submit_answer_definition
 from ..dataset.dataset_utils import message_text, render_messages
-from .ac_model_training import ActivationContextTrainingItem
-from .ac_model_utils import AC_PART_TYPE, direct_parts, is_ac_part
-from ..common.ac_parts import COMPACTION_INSTRUCTIONS, ac_part   # shared with the agent loop (agent_tools)
+from .ac_model_training import ActivationContextTrainingItem, ActivationContextTrainer, ActivationContextTrainingConfig
+from .ac_model_utils import (AC_PART_TYPE, direct_parts, is_ac_part, assistant_target_ids,
+                             prepare_training_history, normalize_public_assistants, complete_public_turns)
+from ..common.ac_parts import COMPACTION_INSTRUCTIONS, ac_part, tools_in_system_text
 
 if t.TYPE_CHECKING:
     from ..agent.agent_config import AgentRunResult
@@ -40,6 +23,42 @@ if t.TYPE_CHECKING:
 
 TRAJ_QA_SYSTEM_PROMPT = "You answer questions about agent trajectories (transcripts of an agent's reasoning, tool calls and tool results)."
 TRAJ_QA_INSTRUCTIONS = "Answer the question from the trajectories above (some may be unrelated). Call submit_answer with the answer."
+RECONSTRUCTION_SYSTEM_PROMPT = "You are a careful assistant. You reproduce text exactly when asked."
+RECONSTRUCTION_INSTRUCTION = "Reproduce the passage above exactly, word for word, and stop at its end."
+
+
+def _cue_cut(text: str, fraction: float) -> int:
+    """Character offset near `fraction` of the text, moved forward to the next whitespace so the cue ends on a word."""
+    cut = max(1, int(len(text) * fraction))
+    while cut < len(text) and not text[cut].isspace():
+        cut += 1
+    return cut
+
+
+def _segment_content(segments: list[dict], ac_name: str) -> list[dict]:
+    """The content list of a nested passage: its text segments verbatim and one activation_context part per child segment, in order."""
+    content = []
+    for segment in segments:
+        if segment["kind"] == "child":
+            content.append(ac_part([{"role": "user", "content": segment["text"]}], ac_name, segment["ratio"]))
+        elif segment["kind"] == "text":
+            if segment["text"]:
+                content.append({"type": "text", "text": segment["text"]})
+        else:
+            raise ValueError(f"unknown segment kind {segment['kind']!r}")
+    return content
+
+
+def _first_child_span(segments: list[dict]) -> tuple[int, int] | None:
+    """Character span of the first child segment inside the concatenation of the segment texts (None without a child)."""
+    offset = 0
+    for segment in segments:
+        if segment["kind"] == "child":
+            return offset, offset + len(segment["text"])
+        offset += len(segment["text"])
+    return None
+
+
 RAG_QA_SYSTEM_PROMPT = "You answer questions from the passages you are given."
 RAG_QA_INSTRUCTIONS = "Answer the question from the passages above (some may be unrelated). Call submit_answer with the answer."
 MESSAGE_TOKEN_OVERHEAD = 8                    # template tokens per message, on top of its text (estimate)
@@ -101,259 +120,212 @@ class ActivationContextStudyGenerator:
 
     # ------------------------------------------------------------------------------------------ compaction
     def generate_compaction_samples(
-        self,
-        dataset_id: str,
-        num_samples: int,
-        seed: int = 0,
+        self, dataset_id: str, num_samples: int, seed: int = 0,
         depth_range: tuple[int, int] = (1, 2),
         threshold_range_tokens: tuple[int, int] = (8192, 32768),
         ratios_range: tuple[float, float] = (1.0 / 8.0, 1.0 / 16.0),
         max_post_compaction_tokens: int = 8192,
-        immediate_continuation_ratio: float = 0.25, # Still bias towards many trajectories being immediate continuations, as that's most important.
-        completion_max_tokens: int = 512,
         max_total_tokens: int = 72_000,
         min_trajectory_tokens: int = MIN_COMPACTION_TOKENS,
+        *, max_assistant_tokens: int | None = None,
     ) -> list[ActivationContextTrainingItem]:
-        """
-        One item per sampled trajectory document (documents are drawn with replacement once every
-        document was used; a document shorter than `min_trajectory_tokens` before its completion is
-        skipped). Segment thresholds are scaled down to fit the trajectory and `max_total_tokens`.
-        """
-        if max_post_compaction_tokens < 0 or not 0 <= immediate_continuation_ratio <= 1 or completion_max_tokens < 1:
-            raise ValueError("Invalid continuation budget or immediate mixture")
+        """Complete public continuations; every assistant output is a target, within both exact budgets."""
+        if max_post_compaction_tokens < 1 or max_total_tokens < 1 or min(depth_range) < 1:
+            raise ValueError("Invalid compaction depth or token budgets")
+        if max_assistant_tokens is None:
+            from ..harness.vllm_wrapper import VLLMWrapper
+            defaults = VLLMWrapper.recommended_chat_kwargs(self.ac_model.target.model_config.model_id)
+            max_assistant_tokens = (defaults.get("sampling_params") or {}).get("max_tokens", 1024)
+        if max_assistant_tokens < 1:
+            raise ValueError("max_assistant_tokens must be positive")
         rng = random.Random(seed)
         documents = self._documents(dataset_id, DataModality.TRAJECTORY)
         assert documents, f"{dataset_id} holds no trajectory documents"
-        order = list(documents)
-        rng.shuffle(order)
-        items: list[ActivationContextTrainingItem] = []
-        attempts = 0
-        requested_immediate = rng.random() < immediate_continuation_ratio
-        while len(items) < num_samples and attempts < 4 * num_samples + len(order):
-            document = order[attempts % len(order)]
-            attempts += 1
-            item = self._compaction_item(document, dataset_id, rng, len(items), seed, depth_range, threshold_range_tokens,
-                                         ratios_range, completion_max_tokens, max_total_tokens, min_trajectory_tokens,
-                                         max_post_compaction_tokens, requested_immediate)
+        rng.shuffle(documents)
+        items = []
+        for attempt in range(4 * num_samples + len(documents)):
+            if len(items) >= num_samples:
+                break
+            item = self._compaction_item(documents[attempt % len(documents)], dataset_id, rng, len(items), seed,
+                        depth_range, threshold_range_tokens, ratios_range, max_total_tokens, min_trajectory_tokens,
+                        max_post_compaction_tokens, max_assistant_tokens)
             if item is not None:
                 items.append(item)
-                requested_immediate = rng.random() < immediate_continuation_ratio
         if len(items) < num_samples:
-            print(f"AC study - produced {len(items)}/{num_samples} compaction items; continuation candidates exhausted")
+            print(f"AC study - produced {len(items)}/{num_samples} compaction items; complete candidates exhausted")
         return items
 
-    def _compaction_item(self, document: DatasetDocument, dataset_id: str, rng: random.Random, index: int, seed: int,
-                         depth_range: tuple[int, int], threshold_range_tokens: tuple[int, int], ratios_range: tuple[float, float],
-                         completion_max_tokens: int, max_total_tokens: int, min_trajectory_tokens: int,
-                         max_post_compaction_tokens: int = 8192, immediate: bool = True) -> ActivationContextTrainingItem | None:
-        messages = document.trajectory or []
-        if len(messages) < 3 or messages[0]["role"] != "user":
+    def _compaction_item(self, document, dataset_id, rng, index, seed, depth_range, threshold_range_tokens,
+                         ratios_range, max_total_tokens, min_trajectory_tokens, max_post_compaction_tokens,
+                         max_assistant_tokens) -> ActivationContextTrainingItem | None:
+        source = document.trajectory or []
+        if len(source) < 2 or source[0].get("role") != "user":
             return None
-        tools = (document.trajectory_kwargs or {}).get("tools") or None
-        system = self._system_messages(document)
-        task = messages[0]
-        body = messages[1:]
-        tokens = [self.message_tokens(message) for message in body]
-        # The completion must be an assistant turn after the last cut: the body up to the last assistant turn is cuttable.
-        last_assistant = max((i for i, message in enumerate(body) if message["role"] == "assistant"), default=-1)
-        if last_assistant < 1:
-            return None
-        available = sum(tokens[:last_assistant + 1])                            # the last assistant turn may be cut mid-text
-        if available < min_trajectory_tokens:
+        try:
+            messages, tools = normalize_public_assistants(self.tokenizer, source,
+                        (document.trajectory_kwargs or {}).get("tools"), max_assistant_tokens)
+            first = next((index for index, message in enumerate(messages) if message.get("role") == "assistant"), len(messages))
+            initial = messages[:first]
+            groups = complete_public_turns(messages[first:])
+        except ValueError:                    # an indivisible public call/reply cannot be a complete candidate
             return None
-        depth = rng.randint(min(depth_range), max(depth_range))
-        thresholds = [rng.randint(min(threshold_range_tokens), max(threshold_range_tokens)) for _ in range(depth)]
-        budget = min(max_total_tokens - completion_max_tokens - self.message_tokens(task) - 256, int(available * 0.9))
-        if budget < 256:
+        if len(groups) < 2:
             return None
-        if sum(thresholds) > budget:
-            scale = budget / sum(thresholds)
-            thresholds = [max(256, int(threshold * scale)) for threshold in thresholds]
-        # Cuts: (message index, character offset in that assistant text or None for a boundary).
-        cuts: list[tuple[int, int | None]] = []
-        cumulative = 0
-        target_total = 0
-        position = 0
-        for threshold in thresholds:
-            target_total += threshold
-            while position < len(body) and cumulative + tokens[position] <= target_total:
-                cumulative += tokens[position]
-                position += 1
-            if position >= len(body):
-                return None
-            message = body[position]
-            text = message_text(message)
-            if message["role"] == "assistant" and not message.get("tool_calls") and len(text) > 200 and cumulative < target_total:
-                fraction = (target_total - cumulative) / max(1, tokens[position])
-                full_ids = self.tokenizer.encode(text, add_special_tokens=False)
-                cut_tokens = max(1, min(len(full_ids) - 1, int(len(full_ids) * min(0.9, max(0.1, fraction)))))
-                partial, remainder, cut_tokens = self._token_boundary(text, full_ids, cut_tokens)
-                offset = len(partial)
-                cuts.append((position, offset))
-                break                                                              # a mid-turn cut ends the chain (the rest of the turn is the completion)
-            cuts.append((position, None))
-        if not cuts:
-            return None
-        # Segments between cuts, the completion after the last one.
-        segments: list[list[dict]] = []
-        start = 0
-        partial_text = ""
-        for cut_index, (position, offset) in enumerate(cuts):
-            if offset is None:
-                segments.append(body[start:position])
-                start = position
-            else:
-                message = body[position]
-                text = message_text(message)
-                segments.append(body[start:position] + [{"role": "assistant", "content": text[:offset]}])
-                partial_text = text[:offset]
-                start = position
-        segments = [segment for segment in segments if segment]
-        if not segments:
+        system = self._system_messages(document)
+        depth = min(rng.randint(min(depth_range), max(depth_range)), len(groups) - 1)
+        if depth < min(depth_range):
             return None
-        last_position, last_offset = cuts[-1]
-        visible: list[dict] = []
-        teacher_partial_ids: list[int] | None = None
-        if last_offset is not None:
-            turn_text = message_text(body[last_position])
-            full_ids = self.tokenizer.encode(turn_text, add_special_tokens=False)
-            # The cut was selected at an exact target-token boundary above.
-            teacher_partial_ids = full_ids[:cut_tokens]
-            completion_ids = full_ids[cut_tokens:]
-            immediate_target = last_position
-            visible_after_cut = [{**body[last_position], "content": turn_text[last_offset:]}]
-        else:
-            immediate_target = next((i for i in range(last_position, len(body)) if body[i]["role"] == "assistant"), None)
-            if immediate_target is None:
+        costs = [sum(self.message_tokens(message) for message in group) for group in groups]
+        budget = min(sum(costs[:-1]), max_total_tokens - max_post_compaction_tokens)
+        threshold = min(rng.randint(min(threshold_range_tokens), max(threshold_range_tokens)), max(1, budget // depth))
+        cuts, offset = [], 0
+        for level in range(depth):
+            total, end = 0, offset
+            limit = len(groups) - (depth - level)
+            while end < limit and (total < threshold or end == offset):
+                total += costs[end]
+                end += 1
+            cuts.append(end)
+            offset = end
+        before = [message for group in groups[:cuts[-1]] for message in group]
+        prefix = system + initial + before
+        chat_kwargs = {"enable_thinking": True}
+        continuation, targets, exact_tokens, post_tokens = [], [], 0, 0
+        for group in groups[cuts[-1]:]:
+            proposed = continuation + group
+            proposed_targets = targets + [assistant_target_ids(self.tokenizer, group[0], tools, chat_kwargs)]
+            ids, _, positions = prepare_training_history(self.tokenizer, prefix + proposed, len(prefix), proposed_targets, [], tools=tools,
+                                                        chat_template_kwargs=chat_kwargs)
+            if positions[0] + 1 < min_trajectory_tokens:
                 return None
-            visible_after_cut = []
-        if immediate:
-            target_position = immediate_target
-            if last_offset is None:
-                visible = body[last_position:target_position]
-                completion_ids = self.tokenizer.encode(self.assistant_text(body[target_position]), add_special_tokens=False)
-            in_context_tail = body[:target_position]
-        else:
-            candidates = []
-            for target_position in range(immediate_target + 1, len(body)):
-                if body[target_position]["role"] != "assistant":
-                    continue
-                suffix = (visible_after_cut + body[last_position + 1:target_position] if last_offset is not None
-                          else body[last_position:target_position])
-                suffix_tokens = sum(self.message_tokens(message) for message in suffix)
-                teacher_tokens = sum(tokens[:target_position]) + self.message_tokens(task)
-                if suffix_tokens <= max_post_compaction_tokens and teacher_tokens + completion_max_tokens + 256 <= max_total_tokens:
-                    candidates.append((target_position, suffix))
-            if not candidates:
-                return None
-            target_position, visible = rng.choice(candidates)
-            in_context_tail = body[:target_position]
-            partial_text, teacher_partial_ids = "", None
-            completion_ids = self.tokenizer.encode(self.assistant_text(body[target_position]), add_special_tokens=False)
-        if not completion_ids:
-            return None
-        complete = len(completion_ids) <= completion_max_tokens
-        completion_ids = completion_ids[:completion_max_tokens]
-        # Avoid a cap ending inside a multibyte character; preserve original IDs.
-        while completion_ids and self.tokenizer.decode(completion_ids, clean_up_tokenization_spaces=False).endswith("\ufffd"):
-            completion_ids = completion_ids[:-1]
-        completion_text = self.tokenizer.decode(completion_ids, clean_up_tokenization_spaces=False)
-        if not completion_text.strip():
-            return None
-        visible_tokens = self._visible_tokens(visible, tools)
-        if visible_tokens > max_post_compaction_tokens:
-            return None
-        if sum(tokens[:target_position]) + len(teacher_partial_ids or []) + len(completion_ids) + self.message_tokens(task) + 256 > max_total_tokens:
+            cost = len(ids) - (positions[0] + 1)
+            if cost > max_post_compaction_tokens or len(ids) > max_total_tokens:
+                break
+            continuation, targets, exact_tokens, post_tokens = proposed, proposed_targets, len(ids), cost
+        if not targets:
             return None
-        ratio = self._sample_ratio(rng, ratios_range)
-        nested: dict | None = None
-        for segment in segments:
-            inner = [{"role": "user", "content": [nested]}] if nested is not None else [task]   # the innermost segment carries the task: what the reader is after
-            nested = ac_part(deepcopy(system) + inner + segment, self.ac_name, ratio)           # a segment part carries its system message first
+        ratio, nested, offset = self._sample_ratio(rng, ratios_range), None, 0
+        for end in cuts:
+            frame = [{"role": "user", "content": [nested]}] if nested is not None else initial
+            segment = [message for group in groups[offset:end] for message in group]
+            nested = ac_part(deepcopy(system + frame + segment), self.ac_name, ratio, kind="compaction")
             if tools:
                 nested["tools"] = deepcopy(tools)
-        ac_user = {"role": "user", "content": [nested, {"type": "text", "text": "\n\n" + COMPACTION_INSTRUCTIONS}]}
+            offset = end
+        student_prefix = system + initial + [{"role": "user", "content": [nested,
+                            {"type": "text", "text": "\n\n" + COMPACTION_INSTRUCTIONS}]}]
         item = ActivationContextTrainingItem(
-            item_id=f"compaction:{dataset_id}:{seed}:{index}",
-            kind="compaction",
-            in_context_prefix=system + [task] + in_context_tail,
-            ac_prefix=system + [task, ac_user] + deepcopy(visible),
-            completion_text=completion_text,
-            completion_complete=complete,
-            teacher_partial_text=partial_text,
-            teacher_partial_token_ids=teacher_partial_ids,
-            completion_token_ids=completion_ids,
-            tools=tools,
-            dataset_id=dataset_id,
-            doc_ids=[document.doc_id],
-            info={"depth": len(segments), "thresholds": thresholds, "ratio": ratio, "in_context_tokens": sum(tokens[:len(in_context_tail)]),
-                  "immediate": immediate, "visible_suffix_tokens": visible_tokens,
-                  "delay_turns": target_position - immediate_target,
-                  "target_position": target_position, "mid_turn_cut": last_offset is not None,   # the completion's body message; run harvesting maps it back
-                  "observed_terminal_answer": (document.trajectory_kwargs or {}).get("observed_terminal_answer")},
-        )
-        prefix = self.tokenizer.apply_chat_template(item.in_context_prefix, tools=tools, add_generation_prompt=True, tokenize=True)
-        prefix = prefix["input_ids"] if hasattr(prefix, "keys") else prefix
-        eot = self.ac_model.target.model_config.model_description.eot_token
-        end_tokens = len(self.tokenizer.encode(eot, add_special_tokens=False)) if complete and eot else 0
-        exact_tokens = len(prefix) + len(teacher_partial_ids or []) + len(completion_ids) + end_tokens
-        if exact_tokens > max_total_tokens:
-            return None
-        item.info["in_context_tokens"] = exact_tokens
+            item_id=f"compaction:{dataset_id}:{seed}:{index}", kind="compaction",
+            in_context_messages=prefix + continuation, ac_messages=deepcopy(student_prefix + continuation),
+            in_context_start=len(prefix), ac_start=len(student_prefix), assistant_token_ids=targets,
+            tools=tools, dataset_id=dataset_id, doc_ids=[document.doc_id],
+            info={"depth": depth, "thresholds": [threshold] * depth, "ratio": ratio,
+                  "chat_template_kwargs": chat_kwargs,
+                  "in_context_tokens": exact_tokens, "post_compaction_tokens": post_tokens,
+                  "max_assistant_tokens": max_assistant_tokens, "assistant_turns": len(targets),
+                  "observed_terminal_answer": (document.trajectory_kwargs or {}).get("observed_terminal_answer")})
+        # Student wrappers and row counts also count against the overall guard.
+        try:
+            ActivationContextTrainer(self.harness, ActivationContextTrainingConfig(max_example_tokens=max_total_tokens)).build_example(self.ac_model, item)
+        except ValueError as error:
+            if "max_example_tokens" in str(error):
+                return None
+            raise
         return item
 
-    def _visible_tokens(self, visible: list[dict], tools: list[dict] | None) -> int:
-        if not visible:
-            return 0
-        anchor = [{"role": "user", "content": ""}]
-        def count(messages: list[dict]) -> int:
-            ids = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True)
-            return len(ids["input_ids"] if hasattr(ids, "keys") else ids)
-        return max(0, count(anchor + visible) - count(anchor))
-
     # ------------------------------------------------------------------------------------------ run-result harvesting
     def items_from_run_results(
-        self,
-        runs: list["AgentRunResult"],
-        *,
+        self, runs: list["AgentRunResult"], *, loss_kind: t.Literal["kl", "sft"] = "kl",
+        selection: t.Literal["score_1", "score_1_or_unscored", "all"] | None = None,
         weight_of: t.Callable[["AgentRunResult"], float] | None = None,
-        seed: int = 0,
-        completion_max_tokens: int | None = None,
-        text_run_kwargs: dict | None = None,
-        items_per_text_run: int = 1,
     ) -> list[ActivationContextTrainingItem]:
-        """
-        Self-distillation items from agent run results: the target reading a run as plain text (teacher)
-        against the same target reading the run's real activation-context prompt (student), on the turn
-        the agent actually sampled. Every run recorded with this AC model yields one item per assistant
-        turn whose prompt holds a part (compaction trees, subagent prompts and returns, tool-output and
-        search parts); the run's compacted segments and every subagent beneath it are visited. A run
-        without an AC model is converted to a trajectory document and cut by the 3a1 generator
-        (`text_run_kwargs` override its defaults), with the recorded sampled tokens as the completion
-        where the cut lands on a whole turn. Weights multiply a KL and must be finite and nonnegative;
-        `weight_of(run)` defaults to the run's score, zero-weight runs are skipped. Records are not
-        mutated; `harvest_report` holds candidate/accepted/skip counts.
-        """
-        rng = random.Random(seed)
-        report = self.harvest_report = {"roots": len(runs), "runs": 0, "candidates": 0, "accepted": 0, "skipped": {}}
-        items: list[ActivationContextTrainingItem] = []
+        """One whole-history item per nonempty segment, preserving every selected recorded output."""
+        from ..agent_training.agent_training_utils import activation_messages_of
+        from ..agent.rollout_caching import config_key
+        if loss_kind not in ("kl", "sft"):
+            raise ValueError(loss_kind)
+        selection = selection or ("all" if loss_kind == "kl" else "score_1")
+        if selection not in ("all", "score_1", "score_1_or_unscored"):
+            raise ValueError(selection)
+        report = self.harvest_report = {"roots": len(runs), "runs": 0, "candidates": 0, "accepted": 0,
+                                      "loss_kind": loss_kind, "selection": selection, "skipped": {}}
+        items = []
         for root in runs:
-            weight = float(root.score if weight_of is None else weight_of(root))
+            if selection != "all" and root.score != 1 and not (selection == "score_1_or_unscored" and root.score is None):
+                self._harvest_skip("selection")
+                continue
+            weight = 1.0 if weight_of is None else float(weight_of(root))
             if not math.isfinite(weight) or weight < 0:
-                raise ValueError("AC distillation item weights must be finite and nonnegative: the KL target has no sign to flip "
-                                 "(negative advantages belong to the policy-gradient path)")
-            if weight == 0.0:
+                raise ValueError("AC item weights must be finite and nonnegative")
+            if weight == 0:
                 self._harvest_skip("zero_weight")
                 continue
-            for source_key, run in self._rollout_sources(root):
+            root_key = f"{config_key(root.agent_config)}:{root.seed}"
+            for source_key, run in self._rollout_sources(root, root_key):
                 report["runs"] += 1
-                if run.ac_model_name is None:
-                    for index in range(items_per_text_run):
-                        items += self._items_from_text_run(run, f"{source_key}.{index}", weight, rng, completion_max_tokens, text_run_kwargs)
-                elif run.ac_model_name == self.ac_name:
-                    items += self._items_from_ac_run(run, source_key, weight, completion_max_tokens)
-                else:
-                    raise ValueError(f"run recorded with AC model {run.ac_model_name!r}; this generator is bound to {self.ac_name!r}")
+                source_model = self.harness.loaded_models.get(run.agent_config.model_name)
+                if source_model is None:
+                    raise ValueError(f"Unknown recorded tokenizer source {run.agent_config.model_name!r}")
+                source_tokenizer = source_model.tokenizer
+                if (source_tokenizer.get_vocab() != self.tokenizer.get_vocab()
+                        or source_tokenizer.special_tokens_map != self.tokenizer.special_tokens_map
+                        or source_tokenizer.chat_template != self.tokenizer.chat_template):
+                    raise ValueError("Recorded assistant tokens require a compatible tokenizer and chat template")
+                tools = self._run_tools(run)
+                for segment_index, segment in enumerate([*run.compactions, run]):
+                    report["candidates"] += 1
+                    targets = []
+                    for step in segment.trajectory:
+                        assistants = [message for message in step.get("messages") or [] if message.get("role") == "assistant"]
+                        if step.get("role") == "assistant":
+                            ids = list(step.get("token_ids") or [])
+                            if len(assistants) != 1 or not ids:
+                                raise ValueError("Recorded assistant step needs one message and its nonempty sampled token IDs; regenerate this run")
+                            targets.append(ids)
+                        elif assistants:
+                            raise ValueError("Recorded input step contains an assistant output")
+                    if not targets:
+                        self._harvest_skip("empty_segment")
+                        continue
+                    config = replace(segment.agent_config, ac_model_name=self.ac_name)
+                    student = activation_messages_of(segment, config)
+                    start = len(segment.prompt_messages)
+                    if segment.ac_model_name:
+                        recorded = [(student[:start], segment.prompt_ac_spans)]
+                        cursor = start
+                        for step in segment.trajectory:
+                            end = cursor + len(step.get("messages") or [])
+                            recorded.append((student[cursor:end], step.get("ac_spans") or []))
+                            cursor = end
+                        if any(len(direct_parts(messages)) != len(spans) for messages, spans in recorded):
+                            raise ValueError("Recorded AC spans need matching raw activation parts; regenerate this run")
+                    teacher_prefix = self._expand_parts(student[:start]) if loss_kind == "kl" else None
+                    if not segment.ac_model_name:
+                        # Text runs also compress their complete initial frame, retaining raw injected parts within it.
+                        frame, suffix = student[:start], student[start:]
+                        system = [message for message in frame if message.get("role") == "system"]
+                        part = ac_part(deepcopy(frame), self.ac_name, self.ac_model.config.default_compression_ratio)
+                        if tools:
+                            part["tools"] = deepcopy(tools)
+                        student = system + [{"role": "user", "content": [part]}] + suffix
+                        start = len(system) + 1
+                    teacher = None
+                    teacher_start = 0
+                    if loss_kind == "kl":
+                        teacher = teacher_prefix + self._expand_parts(student[start:])
+                        teacher_start = len(teacher_prefix)
+                    item = ActivationContextTrainingItem(
+                        item_id=f"rollout:{source_key}:{segment_index}", kind="compaction",
+                        in_context_messages=teacher, ac_messages=deepcopy(student),
+                        in_context_start=teacher_start, ac_start=start, assistant_token_ids=targets,
+                        tools=tools, weight=weight, dataset_id="agent_runs", doc_ids=[root_key],
+                        info={"source": "agent_run", "source_key": source_key, "root_key": root_key,
+                              "chat_template_kwargs": source_model.chat_template_kwargs(segment.agent_config.call_kwargs),
+                              "segment": segment_index, "score": root.score, "completion_source": "recorded",
+                              "channels": sorted({part.get("kind") or "unknown" for part in direct_parts(student)}),
+                              "ac_model_name": segment.ac_model_name, "ac_model_version": segment.ac_model_version})
+                    ActivationContextTrainer(self.harness, ActivationContextTrainingConfig(loss_kind=loss_kind)).build_example(self.ac_model, item)
+                    items.append(item)
         report["accepted"] = len(items)
         return items
 
@@ -361,71 +333,19 @@ class ActivationContextStudyGenerator:
         skipped = self.harvest_report["skipped"]
         skipped[reason] = skipped.get(reason, 0) + 1
 
-    def _rollout_sources(self, root: "AgentRunResult", key: str | None = None) -> t.Iterator[tuple[str, "AgentRunResult"]]:
-        """The run, then every subagent attached to any of its segments (recursively), each once with a path key."""
-        key = f"{root.agent_config.agent_name}:{root.seed}" if key is None else key
+    def _rollout_sources(self, root: "AgentRunResult", key: str) -> t.Iterator[tuple[str, "AgentRunResult"]]:
         yield key, root
-        for segment_index, segment in enumerate(list(root.compactions) + [root]):
+        for segment_index, segment in enumerate([*root.compactions, root]):
             for child_index, child in enumerate(segment.subagent_results):
                 yield from self._rollout_sources(child, f"{key}/s{segment_index}c{child_index}")
 
     def _run_tools(self, run: "AgentRunResult") -> list[dict] | None:
-        """The tool definitions the run's prompts were templated with (None under the inline rendering, whose listing is in the system prompt)."""
         from ..agent.agent import Agent
         agent = Agent(self.harness, run.agent_config)
         agent.dialect = self.dialect
         agent._prepare_tools()
         return agent.template_tools
 
-    def _rollout_completion(self, recorded_ids: list[int], max_tokens: int | None = None) -> tuple[list[int], bool]:
-        """
-        Recorded sampled ids as completion fields: exactly one terminal end-of-turn sequence is stripped
-        (the trainer appends it again when `completion_complete`), the rest is kept verbatim; a cap marks
-        the completion incomplete. No cap by default: a run's turn is bounded by its own sampling max_tokens.
-        """
-        eot = self.ac_model.target.model_config.model_description.eot_token or ""
-        eot_ids = self.tokenizer.encode(eot, add_special_tokens=False) if eot else []
-        ids = list(recorded_ids)
-        complete = bool(eot_ids) and len(ids) >= len(eot_ids) and ids[-len(eot_ids):] == eot_ids
-        if complete:
-            ids = ids[:-len(eot_ids)]
-        if max_tokens is not None and len(ids) > max_tokens:
-            return ids[:max_tokens], False
-        return ids, complete
-
-    def _items_from_ac_run(self, run: "AgentRunResult", source_key: str, weight: float, completion_max_tokens: int | None) -> list[ActivationContextTrainingItem]:
-        """One item per assistant turn whose student prefix (its segment's real prompt plus the steps before it) holds a part."""
-        tools = self._run_tools(run)
-        items: list[ActivationContextTrainingItem] = []
-        for k, segment in enumerate(list(run.compactions) + [run]):
-            system = [message for message in segment.prompt_messages if message.get("role") == "system"]
-            base = [message for message in segment.prompt_messages if message.get("role") != "system"]
-            history: list[dict] = []
-            for t_index, step in enumerate(segment.trajectory):
-                if step.get("role") == "assistant":
-                    student = base + history
-                    self.harvest_report["candidates"] += 1
-                    parts = direct_parts(student)
-                    if not parts:
-                        self._harvest_skip("no_parts")
-                    else:
-                        ids, complete = self._rollout_completion(step.get("token_ids") or [], completion_max_tokens)
-                        if not ids:
-                            self._harvest_skip("empty_completion")
-                        else:
-                            items.append(ActivationContextTrainingItem(
-                                item_id=f"rollout:{source_key}:{k}:{t_index}", kind="compaction",
-                                in_context_prefix=system + self._expand_parts(student), ac_prefix=deepcopy(system + student),
-                                completion_text=self.tokenizer.decode(ids, clean_up_tokenization_spaces=False),
-                                completion_token_ids=ids, completion_complete=complete, tools=tools, weight=weight,
-                                dataset_id="agent_runs", doc_ids=[source_key],
-                                info={"source": "agent_run", "source_key": source_key, "segment": k, "turn": t_index, "score": run.score,
-                                      "channels": sorted({part.get("kind") or "unknown" for part in parts}), "completion_source": "recorded",
-                                      "ac_model_name": run.ac_model_name, "ac_model_version": run.ac_model_version},
-                            ))
-                history += deepcopy(step.get("messages") or [])
-        return items
-
     def _expand_parts(self, messages: list[dict]) -> list[dict]:
         """
         The teacher's view of messages with parts: text only, each part expanded once by its channel. A
@@ -442,6 +362,15 @@ class ActivationContextStudyGenerator:
             tree = next((part for part in content if is_ac_part(part) and part.get("kind") == "compaction"), None)
             if tree is not None:
                 expanded = self._expand_tree(tree)
+                # A nested source frame must remain visible even when the reader has another system prompt.
+                framed = []
+                for entry in expanded:
+                    if entry.get("role") == "system" and out:
+                        if entry in out:
+                            continue
+                        entry = {"role": "user", "content": "Source system frame:\n" + message_text(entry)}
+                    framed.append(entry)
+                expanded = framed
                 if out and expanded and out[-1] == expanded[0]:
                     out.pop()                                                       # the tree starts with the task the prompt already shows
                 out.extend(expanded)
@@ -458,9 +387,9 @@ class ActivationContextStudyGenerator:
                     elif kind == "search":
                         pieces.append("\n\n" + next((m.get("content", "") for m in inner if m.get("role") == "tool"), ""))
                     elif kind == "subagent_return":
-                        pieces.append("Subagent transcript:\n" + render_messages(self._expand_parts(inner)) + "\n\n")
+                        pieces.append("Subagent transcript:\n" + render_messages(self._expand_parts(tools_in_system_text(inner, part.get("tools")))) + "\n\n")
                     else:                                                           # subagent_prompt, parent_context, untagged: a transcript
-                        pieces.append("Parent transcript:\n" + render_messages(self._expand_parts(inner)) + "\n\n")
+                        pieces.append("Parent transcript:\n" + render_messages(self._expand_parts(tools_in_system_text(inner, part.get("tools")))) + "\n\n")
                 elif isinstance(part, dict):
                     if drop_next_text and part.get("type") == "text":
                         drop_next_text = False
@@ -473,76 +402,12 @@ class ActivationContextStudyGenerator:
 
     def _expand_tree(self, tree: dict) -> list[dict]:
         """The history a compaction tree holds, as text: its first user message (recursively) then the segment through the compact call."""
-        inner = [message for message in tree.get("messages") or [] if message.get("role") != "system"]   # the frame is the reader's own
+        inner = tools_in_system_text(tree.get("messages") or [], tree.get("tools"))
         if not inner:
             return []
         expanded = self._expand_parts([inner[0]]) + self._expand_parts(inner[1:])
-        last = expanded[-1] if expanded else None
-        if last is not None and last.get("role") == "assistant" and last.get("tool_calls"):
-            expanded.append({"role": "tool", "content": "Context compacted."})        # the compact call's result, so the transcript stays well formed
         return expanded
 
-    def _rollout_document(self, run: "AgentRunResult", key: str, tools: list[dict] | None) -> tuple[DatasetDocument | None, list[tuple[int, int] | None]]:
-        """A text-only run as a trajectory document (task first, every segment's steps as text, compaction messages dropped) with each body message's (segment, step) origin."""
-        segments = list(run.compactions) + [run]
-        first = segments[0].prompt_messages
-        task = next((message for message in first if message.get("role") == "user"), None)
-        if task is None:
-            return None, []
-        system = next((message.get("content") for message in first if message.get("role") == "system"), None)
-        trajectory = self._expand_parts([task])
-        origins: list[tuple[int, int] | None] = []
-        for k, segment in enumerate(segments):
-            for t_index, step in enumerate(segment.trajectory):
-                expanded = self._expand_parts(step.get("messages") or [])
-                trajectory += expanded
-                origins += [(k, t_index)] * len(expanded)
-            if k < len(segments) - 1 and trajectory and trajectory[-1].get("role") == "assistant" and trajectory[-1].get("tool_calls"):
-                trajectory.append({"role": "tool", "content": "Context compacted."})
-                origins.append(None)
-        document = DatasetDocument(doc_id=key, dataset_id="agent_runs", trajectory=trajectory,
-                                   trajectory_kwargs={"tools": tools, "system_prompt": system, "answer": run.answer}, modality=DataModality.TRAJECTORY)
-        return document, origins
-
-    def _items_from_text_run(self, run: "AgentRunResult", source_key: str, weight: float, rng: random.Random,
-                             completion_max_tokens: int | None, text_run_kwargs: dict | None) -> list[ActivationContextTrainingItem]:
-        """A text-only run cut by the 3a1 generator; a whole-turn completion takes the recorded sampled tokens, a mid-turn cut keeps the re-encoded rest of the turn."""
-        tools = self._run_tools(run)
-        document, origins = self._rollout_document(run, source_key, tools)
-        self.harvest_report["candidates"] += 1
-        if document is None:
-            self._harvest_skip("no_task")
-            return []
-        kwargs = dict(depth_range=(1, 2), threshold_range_tokens=(8192, 32768), ratios_range=(1.0 / 8.0, 1.0 / 16.0), completion_max_tokens=512,
-                      max_total_tokens=72_000, min_trajectory_tokens=MIN_COMPACTION_TOKENS, max_post_compaction_tokens=8192,
-                      immediate_continuation_ratio=0.25) | dict(text_run_kwargs or {})
-        immediate = rng.random() < kwargs.pop("immediate_continuation_ratio")
-        item = self._compaction_item(document, "agent_runs", rng, 0, 0, kwargs["depth_range"], kwargs["threshold_range_tokens"],
-                                     kwargs["ratios_range"], kwargs["completion_max_tokens"], kwargs["max_total_tokens"],
-                                     kwargs["min_trajectory_tokens"], kwargs["max_post_compaction_tokens"], immediate)
-        if item is None:
-            self._harvest_skip("no_cut")
-            return []
-        item.item_id = f"rollout_text:{source_key}"
-        item.weight = weight
-        item.doc_ids = [source_key]
-        item.info.update({"source": "agent_run_text", "source_key": source_key, "score": run.score})
-        position = item.info.get("target_position")
-        origin = origins[position] if position is not None and position < len(origins) else None
-        if item.info.get("mid_turn_cut") or origin is None:
-            item.info["completion_source"] = "reencoded" if item.info.get("mid_turn_cut") else "reencoded_unmapped"
-            return [item]
-        segment = (list(run.compactions) + [run])[origin[0]]
-        step = segment.trajectory[origin[1]]
-        ids, complete = self._rollout_completion(step.get("token_ids") or [], completion_max_tokens)
-        if not ids:
-            self._harvest_skip("empty_completion")
-            return []
-        item.completion_token_ids, item.completion_complete = ids, complete
-        item.completion_text = self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
-        item.info["completion_source"] = "recorded"
-        return [item]
-
     # ------------------------------------------------------------------------------------------ trajectory QA
     def _study_examples(self, dataset_id: str, num_samples: int, seed: int, modality: DataModality, caching_id: str | None) -> list[DatasetQAExample]:
         examples = self.harness.dataset_manager.synthesize_study_examples_qa(
@@ -592,6 +457,7 @@ class ActivationContextStudyGenerator:
             if start != 0 or end != len(text):
                 message["content"] = text[start:end]
                 message.pop("reasoning", None)
+                message.pop("reasoning_content", None)
             if not calls:
                 message.pop("tool_calls", None)
             result.append(message)
@@ -648,6 +514,8 @@ class ActivationContextStudyGenerator:
                 kwargs = source.trajectory_kwargs or {}                              # the frame the trajectory ran with, when the loader kept it
                 frame = [{"role": "system", "content": kwargs["system_prompt"]}] if kwargs.get("system_prompt") else []
                 part = ac_part(frame + messages, self.ac_name, ratio)
+                if kwargs.get("tools"):
+                    part["tools"] = deepcopy(kwargs["tools"])
                 if depth >= 2:
                     part = ac_part(frame + [messages[0], {"role": "user", "content": [part]}], self.ac_name, ratio)
                 if kwargs.get("tools"):
@@ -661,16 +529,18 @@ class ActivationContextStudyGenerator:
             for number, (_, part) in enumerate(parts, start=1):
                 content.extend([{"type": "text", "text": f"Trajectory {number}:\n"}, part, {"type": "text", "text": "\n\n"}])
             content.append({"type": "text", "text": f"{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"})
-            in_context_user = {"role": "user", "content": f"Trajectory:\n{render_messages(window)}\n\n{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"}
+            source_frame = tools_in_system_text(self._system_messages(document) + window, (document.trajectory_kwargs or {}).get("tools"))
+            in_context_user = {"role": "user", "content": f"Trajectory:\n{render_messages(source_frame)}\n\n{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"}
             system = [{"role": "system", "content": TRAJ_QA_SYSTEM_PROMPT}]
+            output = self.dialect.rendering.assistant_message("", [{"id": "answer", "name": "submit_answer", "arguments": {"answer": example.gold_answers[0].strip()}}])
             items.append(ActivationContextTrainingItem(
                 item_id=f"traj_qa:{dataset_id}:{seed}:{item_index}",
                 kind="traj_qa",
-                in_context_prefix=system + [in_context_user],
-                ac_prefix=system + [{"role": "user", "content": content}],
-                completion_text=self.dialect.preferred_format.render("submit_answer", {"answer": example.gold_answers[0].strip()}),
+                in_context_messages=system + [in_context_user, output],
+                ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)],
+                in_context_start=len(system) + 1, ac_start=len(system) + 1,
+                assistant_token_ids=[assistant_target_ids(self.tokenizer, output, [submit_answer_definition()])],
                 tools=[submit_answer_definition()],
-                completion_complete=True,
                 dataset_id=dataset_id,
                 doc_ids=[document.doc_id] + [candidate.doc_id for candidate in candidates[:num_distractors]],
                 info={"depth": depth, "budget": budget, "distractors": num_distractors, "ratio": ratio,
@@ -727,14 +597,15 @@ class ActivationContextStudyGenerator:
                        {"type": "text", "text": f"\n\n{RAG_QA_INSTRUCTIONS}\nQuestion: {question}"}]
             in_context_user = {"role": "user", "content": f"Passage:\n{gold.chunk_text}\n\n{RAG_QA_INSTRUCTIONS}\nQuestion: {question}"}
             system = [{"role": "system", "content": RAG_QA_SYSTEM_PROMPT}]
+            output = self.dialect.rendering.assistant_message("", [{"id": "answer", "name": "submit_answer", "arguments": {"answer": example.gold_answers[0].strip()}}])
             items.append(ActivationContextTrainingItem(
                 item_id=f"rag_qa:{dataset_id}:{seed}:{item_index}",
                 kind="rag_qa",
-                in_context_prefix=system + [in_context_user],
-                ac_prefix=system + [{"role": "user", "content": content}],
-                completion_text=self.dialect.preferred_format.render("submit_answer", {"answer": example.gold_answers[0].strip()}),
+                in_context_messages=system + [in_context_user, output],
+                ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)],
+                in_context_start=len(system) + 1, ac_start=len(system) + 1,
+                assistant_token_ids=[assistant_target_ids(self.tokenizer, output, [submit_answer_definition()])],
                 tools=[submit_answer_definition()],
-                completion_complete=True,
                 dataset_id=dataset_id,
                 doc_ids=sorted({passage.doc_id for passage in passages}),
                 info={"depth": depth, "budget": budget, "passages": len(passages), "ratio": ratio,
@@ -742,6 +613,75 @@ class ActivationContextStudyGenerator:
             ))
         return items
 
+    def generate_reconstruction_samples(
+        self,
+        passages: list[dict],
+        ratio: float = 1.0 / 16.0,
+        variant: str = "reconstruct",
+        cue_fraction: float = 0.2,
+        seed: int = 0,
+        system_prompt: str = RECONSTRUCTION_SYSTEM_PROMPT,
+    ) -> list[ActivationContextTrainingItem]:
+        """
+        Dense, oracle-free items: the reader sees a passage only through its compressed rows and reproduces it.
+        `passages` are {"text", "doc_id", "source"} (any text; no dataset registration needed). Variants:
+        "reconstruct" (the whole passage is the target) and "continue" (the first `cue_fraction` of the passage stays
+        in plain text after the part and the rest is the target). The in-context view holds the plain passage, so a
+        KL teacher would copy; these items are meant for SFT. Kind is the variant name.
+
+        Nested (two-level) passages carry `segments` instead of `text`: an ordered list of {"kind": "text", "text"} and
+        {"kind": "child", "text", "ratio"} entries, plus `ratio` (the passage's own compression; the call's `ratio` when
+        absent) and optionally `variant_only` (the passage is rendered in that variant alone). The part's message content
+        is then a list: the text segments verbatim and one activation_context part per child segment (the child's text
+        at the child's ratio), in order, so the encoder reads its own rows next to text. The plain view and the targets
+        use the concatenation of all segment texts; the "continue" cue ends `cue_fraction` into the first child, so the
+        continuation crosses from the child's rows into the parent's text. Flat passages render as before.
+        """
+        if variant not in ("reconstruct", "continue"):
+            raise ValueError("variant must be 'reconstruct' or 'continue'")
+        if not 0 < cue_fraction < 1:
+            raise ValueError("cue_fraction must be in (0, 1)")
+        items = []
+        for index, passage in enumerate(passages):
+            if passage.get("variant_only") not in (None, variant):
+                continue
+            segments = passage.get("segments")
+            text = "".join(segment["text"] for segment in segments) if segments else passage["text"].strip()
+            if not text:
+                continue
+            child = _first_child_span(segments) if segments else None
+            part_ratio = passage.get("ratio", ratio) if segments else ratio
+            nested_info = {"children": sum(1 for s in segments if s["kind"] == "child"), "child_ratios": [s["ratio"] for s in segments if s["kind"] == "child"]} if segments else {}
+            if variant == "continue":
+                cut = _cue_cut(text, cue_fraction) if child is None else child[0] + _cue_cut(text[child[0]:child[1]], cue_fraction)
+                cue, target = text[:cut].rstrip(), text[cut:].lstrip()
+                if not cue or not target or (child is not None and not child[0] < cut < child[1]):
+                    continue
+                instruction = f"The passage above begins with:\n{cue}\n\nContinue the passage from there, exactly as written, and stop at its end."
+                output_text = target
+                if segments:
+                    nested_info["cue_chars"] = cut
+            else:
+                instruction = RECONSTRUCTION_INSTRUCTION
+                output_text = text
+            part = ac_part([{"role": "user", "content": _segment_content(segments, self.ac_name) if segments else text}], self.ac_name, part_ratio)
+            content = [{"type": "text", "text": "Passage:\n"}, part, {"type": "text", "text": f"\n\n{instruction}"}]
+            output = {"role": "assistant", "content": output_text}
+            system = [{"role": "system", "content": system_prompt}]
+            items.append(ActivationContextTrainingItem(
+                item_id=f"{variant}:{passage.get('source', 'text')}:{seed}:{index}",
+                kind=variant,
+                in_context_messages=system + [{"role": "user", "content": f"Passage:\n{text}\n\n{instruction}"}, output],
+                ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)],
+                in_context_start=len(system) + 1, ac_start=len(system) + 1,
+                assistant_token_ids=[assistant_target_ids(self.tokenizer, output)],
+                tools=None,
+                dataset_id=passage.get("source", ""),
+                doc_ids=[passage["doc_id"]] if passage.get("doc_id") else [],
+                info={"ratio": part_ratio, "variant": variant, "chars": len(text), "passage_tokens": self.count_tokens(text), **nested_info},
+            ))
+        return items
+
     def transform_for_eval_reference(self, items: list[ActivationContextTrainingItem],
                                      kind: t.Literal["no_context", "recent_text"]) -> list[ActivationContextTrainingItem]:
         """Preserve prefixes/suffixes and targets; QA's text reference is oracle gold, not retrieval."""
@@ -749,7 +689,7 @@ class ActivationContextStudyGenerator:
             raise ValueError(kind)
         output = []
         for item in items:
-            messages = deepcopy(item.ac_prefix)
+            messages = deepcopy(item.ac_messages)
             part_index = 0
             for message in messages:
                 if not isinstance(message.get("content"), list):
@@ -771,7 +711,7 @@ class ActivationContextStudyGenerator:
                         pieces.append(self.tokenizer.decode(ids))
                     part_index += 1
                 message["content"] = "".join(pieces)
-            output.append(replace(item, item_id=item.item_id + ":" + kind, ac_prefix=messages,
+            output.append(replace(item, item_id=item.item_id + ":" + kind, ac_messages=messages,
                                   info=dict(item.info, reference_label="oracle gold text" if kind == "recent_text" and item.kind != "compaction" else kind)))
         return output
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/ac_model/ac_model_training.py">activation/ac_model/ac_model_training.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/ac_model/ac_model_training.py`

```diff
diff --git a/activation/ac_model/ac_model_training.py b/activation/ac_model/ac_model_training.py
index d4e3aa3..7eacc2e 100644
--- a/activation/ac_model/ac_model_training.py
+++ b/activation/ac_model/ac_model_training.py
@@ -1,24 +1,8 @@
-"""
-Training the AC model in isolation (self-distillation): the target model reading the full context
-(the teacher, the same target adapter, no gradient) is matched by the target model reading the AC
-prefix (the student) on the completion tokens, with a full-vocabulary KL(teacher || student). One
-round trains the AC modules and the side LoRA (`learning_rate_ac`) and the target LoRA
-(`learning_rate_target_lora`) together; the AdamW state and the warm-up clock persist across
-rounds (one optimizer per AC model, parked on the host between rounds).
-
-Teacher tokens: the in-context prefix under the target chat template with the generation prompt
-open, then (for a mid-turn cut) the text the turn had already produced, then the completion text
-and, when the completion is a whole turn, the end-of-turn token. Student tokens: the AC prefix
-through `tokenize_with_parts` (rows written over the placeholder runs), then the same completion
-tokens. The loss averages over the completion positions.
-
-`drift_term_weight` > 0 adds KL(frozen base || target-with-adapter) on the in-context tokens so
-the target adapter cannot drift from the base's in-context behaviour; off by default (the teacher
-already carries the adapter).
-"""
+"""Whole-history AC self-distillation: current-reader KL or teacher-free supervised loss."""
 from __future__ import annotations
 
 import random
+import math
 import hashlib
 import json
 import shutil
@@ -26,21 +10,23 @@ from fractions import Fraction
 from pathlib import Path
 import time
 import typing as t
-from dataclasses import asdict, dataclass, field
-from contextlib import ExitStack
+from dataclasses import asdict, dataclass, field, replace
+from contextlib import ExitStack, nullcontext
 
 import torch
 import torch.nn.functional as F
 from torch.utils.checkpoint import checkpoint
 
 from ..agent_training.agent_training_utils import (
-    disable_dropout, head_logits, optimizer_state_to, persistent_optimizer, set_checkpointing, set_learning_rates,
+    disable_dropout, head_logits, optimizer_state_to, persistent_optimizer, set_learning_rates,
 )
 from ..autotuners import get_ac_training_autotuning_variables, configure_fla_runtime
-from ..harness.hf_utils import SOURCE_DEVICE, TARGET_DEVICE
+from ..harness.hf_utils import SOURCE_DEVICE, TARGET_DEVICE, Interventions, generate_with_interventions
 from ..common.data_syncing import resolve_path
 from .ac_model import ROLLOUT, TRAINING, ActivationContextModel
-from .ac_model_utils import EncodeRequest, direct_parts, tokenize_with_parts
+from ..common.ac_parts import strip_ac_parts
+from .ac_model_utils import (ACOutput, EncodeRequest, direct_parts, prepare_training_history,
+                             validate_paired_histories, validate_part_capacity)
 
 if t.TYPE_CHECKING:
     from ..harness import HarnessRuntime
@@ -64,13 +50,11 @@ ITEM_KINDS = ("compaction", "traj_qa", "rag_qa")
 class ActivationContextTrainingItem:
     item_id: str
     kind: str                                  # compaction | traj_qa | rag_qa (or a reference variant's name)
-    in_context_prefix: list[dict]              # the teacher's messages (plain text)
-    ac_prefix: list[dict]                      # the student's messages, with activation_context parts (or plain text for references)
-    completion_text: str                       # what both must predict (a whole turn or the rest of a cut turn)
-    completion_complete: bool = True           # the completion ends the turn: the end-of-turn token is a target too
-    teacher_partial_token_ids: list[int] | None = None
-    completion_token_ids: list[int] | None = None
-    teacher_partial_text: str = ""             # mid-turn cut: the text already produced, raw tokens for the teacher (the AC side holds it compressed)
+    in_context_messages: list[dict] | None    # full teacher view; unused in SFT
+    ac_messages: list[dict]                   # full student view, including all later input parts
+    in_context_start: int                     # independent logical starts, in message indexes
+    ac_start: int
+    assistant_token_ids: list[list[int]]      # one authoritative output per retained assistant message
     tools: list[dict] | None = None            # tool definitions for the template (both sides)
     weight: float = 1.0
     dataset_id: str = ""
@@ -80,42 +64,74 @@ class ActivationContextTrainingItem:
 
 @dataclass
 class ActivationContextTrainingConfig:
-    learning_rate_ac: float = 5e-4             # AC modules and the side LoRA
-    learning_rate_target_lora: float = 2e-5    # the target adapter
+    """
+    The training recipe. The defaults are the ones the reconstruction bench settled on: SFT on the gold target tokens, cosine
+    decay with a linear warm-up, one joint gradient-norm clip over the encoder and the reader adapter, the no-context penalty
+    (λ x KL(base || adapted reader) on the same history with its AC parts removed, at the target positions, on a random
+    quarter of the examples per update and scaled by 1 / fraction) whenever the reader adapter trains, and stored teacher /
+    reference targets that always hold the gold token. `learning_rate_target_lora = 0` freezes the reader adapter (it neither
+    steps nor carries the penalty): the regime that adapts the encoder alone to a dataset or task.
+    """
+    loss_kind: t.Literal["kl", "sft"] = "sft"   # "sft": cross-entropy on the target tokens; "kl": the coarsened KL to the stored teacher (the reader as it is at the
+                                               # call's start, on the in-context history; its top-k log-probs + the remainder mass are stored before the first update)
+    learning_rate_ac: float = 1e-4             # AC modules and the side LoRA
+    learning_rate_target_lora: float = 2e-5    # the target adapter (0 = frozen: no steps, no penalty)
     weight_decay: float = 0.0
     adam_betas: tuple[float, float] = (0.9, 0.95)
     adam_eps: float = 1e-8
     max_grad_norm: float = 1.0
-    warmup_updates: int = 0                    # linear warm-up of both learning rates over this many updates (counted across calls)
+    lr_schedule: str = "cosine"                # "constant" (warm-up only) or "cosine": decay to lr_final_fraction of the base rate over the planned updates
+    lr_final_fraction: float = 0.1
+    warmup_updates: int = 50                   # linear warm-up of both learning rates over this many updates (counted across calls)
+    schedule_total_updates: int | None = None  # planned updates the cosine spans (default: updates done at the call's start + steps x num_epochs, so a resume stays on the curve)
+    resume_optimizer: bool = False             # load `optimizer.pt` next to the AC checkpoint the model was loaded from (moments + updates done), once per model
+    resume_skip_steps: int = 0                 # steps of the first epoch of this call already trained by a previous attempt (resume from a mid-epoch checkpoint)
+    checkpoint_every_updates: int | None = None  # mid-epoch checkpoint (AC_MODELS/<name>/epoch_<n>_step_<s> + optimizer.pt + partial_epoch.json) every this many steps
+    checkpoint_every_epoch: bool = True        # AC_MODELS/<name>/epoch_<n> (+ latest) and LORAS/<target lora>/epoch_<n>
+    no_context_penalty_weight: float = 1.0     # λ of the no-context penalty (0 = off); applied only while the reader adapter trains
+    no_context_penalty_fraction: float = 0.25  # share of training examples carrying the penalty each update; the term is scaled by 1/fraction (unbiased)
+    teacher_cache_top_k: int = 32              # stored teacher (KL) and no-context reference (penalty): top-k log-probs + the remainder mass per target position,
+                                               # the gold token always among them; the KL is between the k+1-way coarsened distributions (a lower bound of the exact KL)
+    deltas_off_panel: bool = False             # deep models: evaluate the reporting items a second time at every panel with the rows kept and the deltas removed ("deltas_off"): held minus deltas_off is the deltas' contribution
     updates_per_epoch: int = 4                 # maximum gradient steps per epoch when examples_per_update is unset
     examples_per_update: int | None = None     # when set, overrides updates_per_epoch: ceil(items / examples_per_update) steps
-    max_example_tokens: int = 72_000           # longer teacher sequences are dropped (counted), never truncated: depth 2 at the
-                                               # default 32k compaction threshold, depth 1 up to a 50k threshold (task + completion included)
-    teacher_cache_top_k: int = 0               # > 0: the teacher's top-k log-probs (+ the remainder mass) per completion position are
-                                               # kept per item after its first pass and stand in for the teacher afterwards (epoch 2
-                                               # skips the teacher forward); training then minimizes the KL of the k+1-way coarsened
-                                               # distributions (a lower bound of the exact KL, the same loss in every epoch, the
-                                               # teacher's adapter as of the item's last exact pass). Eval is always exact.
-    drift_term_weight: float = 0.0
-    logits_chunk_tokens: int | None = None           # completion positions per lm_head chunk
+    micro_batch_examples: int = 1              # SFT: examples per right-padded reader forward (1 = one example at a time)
+    max_example_tokens: int = 72_000           # explicit error; no dropping or truncation
+    logits_chunk_tokens: int | None = None     # completion positions per lm_head chunk
     fp32_head_matmul: bool = False
     gradient_checkpointing: bool = True        # on both bases
     gradient_checkpointing_min_tokens: int | None = None
     seed: int = 0
-    checkpoint_every_epoch: bool = True        # AC_MODELS/<name>/epoch_<n> (+ latest) and LORAS/<target lora>/epoch_<n>
     num_epochs: int = 2
     reporting_interval: float = 0.1
     completion_samples: int = 4
     completion_max_new_tokens: int = 64
 
     def __post_init__(self) -> None:
+        if self.loss_kind not in ("kl", "sft") or self.max_example_tokens < 1 or self.teacher_cache_top_k < 1:
+            raise ValueError("Invalid loss kind, example capacity, or teacher cache size (the stored targets need a positive top-k)")
+        if self.no_context_penalty_weight < 0 or not 0 < self.no_context_penalty_fraction <= 1:
+            raise ValueError("no_context_penalty_weight must be nonnegative and no_context_penalty_fraction in (0, 1]")
+        if self.learning_rate_target_lora < 0 or self.learning_rate_ac < 0:
+            raise ValueError("learning rates must be nonnegative (0 freezes the reader adapter)")
+        if self.lr_schedule not in ("constant", "cosine") or not 0 <= self.lr_final_fraction <= 1 or self.warmup_updates < 0:
+            raise ValueError("lr_schedule must be 'constant' or 'cosine' with lr_final_fraction in [0, 1] and a nonnegative warm-up")
         if self.num_epochs < 1 or self.updates_per_epoch < 1 or not 0 < self.reporting_interval <= 1:
             raise ValueError("Epochs/updates must be positive and reporting_interval must be in (0, 1]")
         if self.examples_per_update is not None and self.examples_per_update < 1:
             raise ValueError("examples_per_update must be positive")
+        if self.micro_batch_examples < 1:
+            raise ValueError("micro_batch_examples must be positive")
+        if self.micro_batch_examples > 1 and self.loss_kind != "sft":
+            raise ValueError("micro_batch_examples > 1 is only implemented for loss_kind='sft'")
         if self.completion_samples < 0 or self.completion_max_new_tokens < 1:
             raise ValueError("Invalid completion inspection budget")
 
+    @property
+    def penalty_active(self) -> bool:
+        """The no-context penalty trains the reader adapter only: it is on when λ > 0 and the adapter has a learning rate."""
+        return self.no_context_penalty_weight > 0 and self.learning_rate_target_lora > 0
+
 
 @dataclass(frozen=True)
 class ActivationContextTrainingProgress:
@@ -129,27 +145,44 @@ class ActivationContextTrainingProgress:
     elapsed_s: float
 
 
-class EvalItemSummary(t.TypedDict):
+class EvalItemSummary(t.TypedDict, total=False):
+    token_accuracy: float                      # teacher-forced argmax accuracy over the target positions
     item_id: str
     kind: str
-    kl: float
-    agreement: float
+    loss_kind: str
+    loss: float | None
+    kl: float | None
+    agreement: float | None
     positions: int
     teacher_tokens: int
     student_tokens: int
+    gold_nll: float                     # mean negative log-prob of the item's target tokens under the student
+    target_logp: list[float]            # the student's log-prob of every target token, in order
+    teacher_gold_nll: float | None      # the same under the stored teacher (KL only)
+    teacher_target_logp: list[float] | None
 
 
-class EvalKindSummary(t.TypedDict):
-    kl: float
-    agreement: float
+class EvalKindSummary(t.TypedDict, total=False):
+    token_accuracy: float | None
+    loss_kind: str
+    loss: float | None
+    kl: float | None
+    agreement: float | None
+    gold_nll: float | None
+    teacher_gold_nll: float | None
     items: int
 
 
-class ActivationContextEvalSummary(t.TypedDict):
+class ActivationContextEvalSummary(t.TypedDict, total=False):
+    token_accuracy: float | None
     items: int
     dropped_too_long: int
-    kl: float
-    agreement: float
+    loss_kind: str
+    loss: float | None
+    kl: float | None
+    agreement: float | None
+    gold_nll: float | None              # item mean of the per-token target NLL under the student
+    teacher_gold_nll: float | None      # the same under the stored teacher (KL only)
     by_kind: dict[str, EvalKindSummary]
     per_item: list[EvalItemSummary]
 
@@ -158,7 +191,7 @@ class CompletionSample(t.TypedDict):
     item_id: str
     kind: str
     reference: str
-    teacher: str
+    teacher: str | None
     student: str
     epoch_number: int
     global_step: int
@@ -176,6 +209,7 @@ class ActivationContextEpochStats:
     checkpoint_seconds: float = 0.0
     reporting: ActivationContextEvalSummary | None = None
     validation: ActivationContextEvalSummary | None = None
+    deltas_off: ActivationContextEvalSummary | None = None       # deep models with deltas_off_panel: the reporting items without their deltas
     checkpoint_path: str | None = None
     target_lora_checkpoint_path: str | None = None
     samples: list[CompletionSample] = field(default_factory=list)
@@ -191,6 +225,9 @@ def reporting_boundaries(steps: int, interval: float) -> list[int]:
 
 @dataclass
 class ActivationContextTrainingStats:
+    loss_kind: str = "kl"
+    loss: list[float] = field(default_factory=list)
+    loss_by_kind: dict[str, list[float]] = field(default_factory=dict)
     start_epoch: int = 0
     epochs: list[ActivationContextEpochStats] = field(default_factory=list)
     baseline_samples: list[CompletionSample] = field(default_factory=list)
@@ -206,8 +243,15 @@ class ActivationContextTrainingStats:
     steps: int = 0
     kl: list[float] = field(default_factory=list)               # per step, item-weighted mean of each item’s mean KL
     agreement: list[float] = field(default_factory=list)        # per step, top-1 teacher/student agreement
-    drift: list[float] = field(default_factory=list)
     grad_norm: list[float] = field(default_factory=list)
+    grad_norm_ac: list[float] = field(default_factory=list)        # encoder side (AC modules + side LoRA) before clipping
+    grad_norm_reader: list[float] = field(default_factory=list)    # reader LoRA before clipping (0 while frozen)
+    penalty_counts: list[int] = field(default_factory=list)         # per update: examples that carried the no-context penalty
+    gate_trace: list[dict] = field(default_factory=list)           # the scalar gates every 25 updates
+    intervention_trace: list[dict] = field(default_factory=list)   # deep models, every 5 updates for the first 200 then every 25: per layer the head's gradient norm, the applied delta RMS relative to the
+                                                                    # residual RMS at the row positions, the delta / input-row cosine and the read-out block's output-projection norm
+    payload_bytes: int = 0                                          # deep models: delta bytes (bf16) handed to the reader over the call's training forwards
+    deltas_off: ActivationContextEvalSummary | None = None          # deep models: the reporting items without their deltas at the last panel
     step_seconds: list[float] = field(default_factory=list)
     step_tokens: list[int] = field(default_factory=list)        # teacher + student tokens per step
     kl_by_kind: dict[str, list[float]] = field(default_factory=dict)   # per example
@@ -216,7 +260,12 @@ class ActivationContextTrainingStats:
     teacher_tokens: int = 0
     student_tokens: int = 0
     side_tokens: int = 0
-    reporting: ActivationContextEvalSummary | None = None               # exact reporting items at the last epoch boundary
+    reporting: ActivationContextEvalSummary | None = None               # reporting items at the last epoch boundary
+    validation_baseline: ActivationContextEvalSummary | None = None     # validation items before the first update (when distinct from reporting)
+    teacher_targets: dict = field(default_factory=dict)                 # KL: {"examples", "top_k", "coverage"} of the stored teacher targets
+    resumed_from: str | None = None                                       # the optimizer.pt this call restored, if any
+    no_context_reference: dict = field(default_factory=dict)            # penalty: the same for the base's no-context reference
+    penalty: list[float] = field(default_factory=list)                  # per update: mean unweighted no-context KL over the sampled examples (carried forward when none)
     checkpoint_path: str | None = None
     target_lora_checkpoint_path: str | None = None
     peak_memory_bytes: int = 0
@@ -240,10 +289,13 @@ class ActivationContextTrainingStats:
         tokens = sum(self.step_tokens)
         seconds = sum(self.step_seconds)
         return {
+            "loss_kind": self.loss_kind, "loss_first": self.loss[0] if self.loss else None,
+            "loss_last": self.loss[-1] if self.loss else None,
+            "loss_by_kind": {kind: sum(values) / len(values) for kind, values in self.loss_by_kind.items() if values},
             "autotuning": self.autotuning,
             "start_epoch": self.start_epoch, "epochs": [asdict(epoch) for epoch in self.epochs],
             "placement_seconds": self.placement_seconds, "cleanup_seconds": self.cleanup_seconds, "other_seconds": self.other_seconds,
-            "baseline_samples": self.baseline_samples, "baseline_seconds": self.baseline_seconds, "global_steps": self.global_steps, "learning_rates": self.learning_rates, "items": self.items, "examples": self.examples, "dropped_too_long": self.dropped_too_long,
+            "baseline_samples": self.baseline_samples, "baseline_seconds": self.baseline_seconds, "global_steps": self.global_steps, "learning_rates": self.learning_rates, "grad_norm_ac": self.grad_norm_ac, "grad_norm_reader": self.grad_norm_reader, "penalty_counts": self.penalty_counts, "gate_trace": self.gate_trace, "intervention_trace": self.intervention_trace, "payload_bytes": self.payload_bytes, "deltas_off": self.deltas_off, "items": self.items, "examples": self.examples, "dropped_too_long": self.dropped_too_long,
             "steps": self.steps, "kl_first": round(self.kl[0], 4) if self.kl else None, "kl_last": round(self.kl[-1], 4) if self.kl else None,
             "agreement_last": round(self.agreement[-1], 4) if self.agreement else None,
             "kl_by_kind": {kind: round(sum(values) / len(values), 4) for kind, values in self.kl_by_kind.items() if values},
@@ -252,7 +304,7 @@ class ActivationContextTrainingStats:
             "reporting": self.reporting, "checkpoint": self.checkpoint_path, "target_lora_checkpoint": self.target_lora_checkpoint_path,
             "peak_memory_gb": round(self.peak_memory_bytes / 2**30, 2), "duration_s": round(self.duration_s, 1),
             "seconds_by_bucket": self.seconds_by_bucket(), "teacher_cache_hits": self.teacher_cache_hits,
-            "teacher_cache_mb": round(self.teacher_cache_bytes / 2**20, 1),
+            "teacher_cache_mb": round(self.teacher_cache_bytes / 2**20, 1), "teacher_targets": self.teacher_targets,
         }
 
 
@@ -261,10 +313,17 @@ class _Example:
     item: ActivationContextTrainingItem
     teacher_ids: list[int]
     student_ids: list[int]
+    teacher_positions: list[int]
+    student_positions: list[int]
+    target_ids: list[int]
     spans: list[tuple[int, int]]
     part_requests: list[EncodeRequest]
-    num_completion: int                        # completion tokens (+ eot) at the end of both sequences
-    teacher_cache_key: tuple[str, str, int, int]
+    teacher_cache_key: tuple[str, str, int]
+    penalty: "_Example | None" = None          # the same history without its AC parts, scored against the base (no-context penalty)
+
+    @property
+    def num_completion(self) -> int:
+        return len(self.target_ids)
 
 
 class ActivationContextTrainer:
@@ -272,36 +331,67 @@ class ActivationContextTrainer:
         self.harness = harness
         self.config = config or ActivationContextTrainingConfig()
         self.epochs_done: dict[str, int] = {}
+        self.resumed: set[str] = set()                # models whose optimizer.pt was restored in this process
         self.updates_done: dict[str, int] = {}                    # optimizer steps so far per AC model (the warm-up clock)
         self.optimizers: dict[str, torch.optim.AdamW] = {}        # one per AC model; moments survive train() calls
-        self.teacher_cache: dict[tuple[str, str, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}   # prepared-token identity -> (ids [C, k], log-probs [C, k], log remainder [C]), CPU
+        self.teacher_cache: dict[tuple[str, str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}   # prepared-token identity -> (ids [C, k], log-probs [C, k], log remainder [C]), CPU
+        self.teacher_target_logp: dict[tuple[str, str, int], torch.Tensor] = {}   # same identity -> the stored teacher's log-prob of each target token [C], CPU
+        self._teacher_texts: dict[str, str] = {}                                   # item id -> the stored teacher's greedy completion, sampled once at baseline
+        self._penalty_rng = random.Random(self.config.seed)                        # which training examples carry the no-context penalty
+        self._penalty_term: torch.Tensor | None = None                             # set by _example_loss: the weighted penalty of the last example, if sampled
+        self._last_hits: torch.Tensor | None = None                                # set by the chunked losses: argmax == target per target position
+        self._payload_bytes = 0                                                    # deep models: delta bytes handed to the reader (reset per train call)
+        self._row_rms: dict[int, list[torch.Tensor]] = {}                          # deep models: residual RMS at the row positions per layer since the last trace (0-d tensors)
 
     # ------------------------------------------------------------------------------------------ examples
-    def build_example(self, ac_model: ActivationContextModel, item: ActivationContextTrainingItem) -> _Example:
+    def build_example(self, ac_model: ActivationContextModel, item: ActivationContextTrainingItem, reference: bool = False) -> _Example:
+        """With `reference`, the example is a no-context reference: its in-context history is scored regardless of the loss kind."""
+        if not math.isfinite(item.weight) or item.weight < 0:
+            raise ValueError(f"{item.item_id}: weight must be finite and nonnegative")
         tokenizer = ac_model.target.tokenizer
-        eot = ac_model.target.model_config.model_description.eot_token
-        completion_ids = (list(item.completion_token_ids) if item.completion_token_ids is not None
-                          else tokenizer.encode(item.completion_text, add_special_tokens=False))
-        if item.completion_complete and eot:
-            completion_ids = completion_ids + tokenizer.encode(eot, add_special_tokens=False)
-        teacher_prefix = tokenizer.apply_chat_template(_flatten(item.in_context_prefix), tools=item.tools, add_generation_prompt=True, tokenize=True)
-        teacher_prefix = list(teacher_prefix["input_ids"] if hasattr(teacher_prefix, "keys") else teacher_prefix)
-        if item.teacher_partial_token_ids is not None:
-            teacher_prefix += list(item.teacher_partial_token_ids)
-        elif item.teacher_partial_text:
-            teacher_prefix += tokenizer.encode(item.teacher_partial_text, add_special_tokens=False)
-        parts = direct_parts(item.ac_prefix)
-        requests = [ac_model._child_request(part, ac_model.config.default_compression_ratio) for part in parts]
-        requests = [EncodeRequest(request.messages, request.compression_ratio, False, request.tools) for request in requests]   # top-level parts: target rows
+        chat_kwargs = item.info.get("chat_template_kwargs")
+        requests = [ac_model._child_request(part, ac_model.config.default_compression_ratio)
+                    for part in direct_parts(item.ac_messages)]
+        requests = [EncodeRequest(request.messages, request.compression_ratio, False, request.tools) for request in requests]
         lengths = [ac_model.part_view_rows(request.messages, request.compression_ratio, request.tools) for request in requests]
-        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
-        student_prefix, spans = tokenize_with_parts(tokenizer, item.ac_prefix, lengths, pad_id, tools=item.tools, add_generation_prompt=True)
-        if not completion_ids or not teacher_prefix or not student_prefix:
-            raise ValueError(f"{item.item_id}: training requires nonempty prefixes and completion")
-        teacher_ids = teacher_prefix + completion_ids
-        digest = hashlib.sha256(json.dumps(teacher_ids, separators=(",", ":")).encode()).hexdigest()
-        key = (ac_model.name, digest, len(completion_ids), self.config.teacher_cache_top_k)
-        return _Example(item, teacher_ids, student_prefix + completion_ids, spans, requests, len(completion_ids), key)
+        student_ids, spans, student_positions = prepare_training_history(tokenizer, item.ac_messages, item.ac_start,
+                            item.assistant_token_ids, lengths, tools=item.tools, chat_template_kwargs=chat_kwargs)
+        teacher_ids, teacher_positions = [], []
+        if self.config.loss_kind == "kl" or reference:
+            if item.in_context_messages is None:
+                raise ValueError(f"{item.item_id}: KL needs an in-context teacher history")
+            validate_paired_histories(item.in_context_messages, item.in_context_start, item.ac_messages, item.ac_start)
+            teacher_ids, _, teacher_positions = prepare_training_history(tokenizer, item.in_context_messages,
+                                item.in_context_start, item.assistant_token_ids, [], tools=item.tools, chat_template_kwargs=chat_kwargs)
+        for view, ids in (("teacher", teacher_ids), ("student", student_ids)):
+            if len(ids) > self.config.max_example_tokens:
+                raise ValueError(f"{item.item_id}: {view} history has {len(ids)} tokens, exceeding max_example_tokens={self.config.max_example_tokens}; regenerate or raise the guard")
+        digest = hashlib.sha256(json.dumps([teacher_ids, teacher_positions], separators=(",", ":")).encode()).hexdigest()
+        key = (ac_model.name + ("#base" if reference else ""), digest, self.config.teacher_cache_top_k)
+        example = _Example(item, teacher_ids, student_ids, teacher_positions, student_positions,
+                           [token for output in item.assistant_token_ids for token in output], spans, requests, key)
+        if not reference and requests and self.config.penalty_active:
+            # The no-context reference: the same history with its AC parts removed, scored against the base with no adapter.
+            stripped = strip_ac_parts(item.ac_messages)
+            example.penalty = self.build_example(ac_model, replace(item, item_id=item.item_id + ":no_context_reference", in_context_messages=stripped,
+                                                                   ac_messages=stripped, in_context_start=item.ac_start), reference=True)
+        return example
+
+    def _check_capacity(self, ac_model: ActivationContextModel, examples: t.Iterable[_Example]) -> None:
+        def capacity(model):
+            config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
+            return getattr(config, "max_position_embeddings", None)
+        target_limit, side_limit = capacity(ac_model.target.model), capacity(ac_model.side.model)
+        for example in examples:
+            for view, ids in (("teacher", example.teacher_ids), ("student", example.student_ids)):
+                if target_limit and len(ids) > target_limit:
+                    raise ValueError(f"{example.item.item_id}: {view} reader history has {len(ids)} tokens, exceeding supported capacity {target_limit}")
+            if side_limit:
+                for request in example.part_requests:
+                    try:
+                        validate_part_capacity(ac_model, request, side_limit)
+                    except ValueError as error:
+                        raise ValueError(f"{example.item.item_id}: {error}") from error
 
     # ------------------------------------------------------------------------------------------ training
     def train(
@@ -312,27 +402,30 @@ class ActivationContextTrainer:
         reporter: "ActivationContextTrainingReporter | None" = None,
         *,
         validation_data: t.Sequence[ActivationContextTrainingItem] = (),
+        reset_optimizer: bool = False,
     ) -> ActivationContextTrainingStats:
         """Complete epochs in one residency scope; optimizer/global step persist across calls."""
+        self._training_ac_name = ac_model_name
         config = self.config
-        assert training_data, "No training data."
-        self.teacher_cache.clear()                 # teacher targets belong to this invocation only
+        if not training_data:
+            raise ValueError("No eligible AC training data")
+        self.teacher_cache.clear()                 # teacher targets belong to this invocation only: the reader as it is at this call's start
+        self.teacher_target_logp.clear()
+        self._teacher_texts.clear()
         manager = self.harness.module_manager
         ac_model = manager.get_ac_model(ac_model_name)
         target = ac_model.target
         target_lora = ac_model.config.target_model_lora_name
         start_epoch = self._next_epoch_number(ac_model)
-        stats = ActivationContextTrainingStats(start_epoch=start_epoch, items=len(training_data))
+        stats = ActivationContextTrainingStats(loss_kind=config.loss_kind, start_epoch=start_epoch, items=len(training_data))
         started = time.monotonic()
+        if reporter is not None:
+            reporter.set_status(phase="building training examples", examples=str(len(training_data)))
+            reporter.render(force=True)
         torch.manual_seed(config.seed + start_epoch)
-        examples = []
-        for item in training_data:
-            example = self.build_example(ac_model, item)
-            if max(len(example.teacher_ids), len(example.student_ids)) > config.max_example_tokens:
-                stats.dropped_too_long += 1
-            else:
-                examples.append(example)
-        assert examples, f"every item was dropped ({stats.dropped_too_long} too long)"
+        examples = [self.build_example(ac_model, item) for item in training_data]
+        reporting_examples = [self.build_example(ac_model, item) for item in reporting_data]
+        validation_examples = [self.build_example(ac_model, item) for item in validation_data]
         stats.examples = len(examples)
         stats.completion_tokens = sum(example.num_completion for example in examples)
         stats.teacher_tokens = sum(len(example.teacher_ids) for example in examples)
@@ -340,14 +433,7 @@ class ActivationContextTrainer:
         per_step = config.examples_per_update or -(-len(examples) // min(config.updates_per_epoch, len(examples)))
         steps = -(-len(examples) // per_step)
         boundaries = set(reporting_boundaries(steps, config.reporting_interval))
-        fixed_items = []
-        if config.completion_samples:
-            for item in reporting_data:
-                example = self.build_example(ac_model, item)
-                if max(len(example.teacher_ids), len(example.student_ids)) <= config.max_example_tokens:
-                    fixed_items.append(item)
-                if len(fixed_items) == config.completion_samples:
-                    break
+        fixed_items = list(reporting_data[:config.completion_samples])
         kernel_scope = ExitStack()
         optimizer = None
         device = None
@@ -356,6 +442,9 @@ class ActivationContextTrainer:
             reporter.initialize_training(config, stats)
         try:
             placement_started = time.monotonic()
+            if reporter is not None:
+                reporter.set_status(phase="model placement and kernel preparation")
+                reporter.render(force=True)
             if target.vllm_model is not None:
                 target.engine_to_device(SOURCE_DEVICE)
             variables = self._autotuning_variables(ac_model)
@@ -366,22 +455,74 @@ class ActivationContextTrainer:
             manager.ensure_lora(target_lora)
             device = ac_model.prepare()
             ac_model.set_mode(TRAINING)
-            bases = [target.model, ac_model.side.model]
+            bases = list(dict.fromkeys([target.model, ac_model.side.model]))
             for base in bases:
                 base.train()
                 disable_dropout(base)
-            min_tokens = variables["target_gradient_checkpointing_min_tokens"]
-            side_min_tokens = variables["side_gradient_checkpointing_min_tokens"]
-            side_config = _CheckpointConfig(config.gradient_checkpointing and ac_model.config.side_gradient_checkpointing,
-                                            side_min_tokens)
-            set_checkpointing(target.model, config, min_tokens, min_tokens)
-            set_checkpointing(ac_model.side.model, side_config, side_min_tokens, side_min_tokens)
+            self._target_gradient_checkpointing_min_tokens = variables["target_gradient_checkpointing_min_tokens"]
+            ac_model._side_gradient_checkpointing_min_tokens = variables["side_gradient_checkpointing_min_tokens"]
+            ac_model._training_gradient_checkpointing = config.gradient_checkpointing
+            self._check_capacity(ac_model, [*examples, *reporting_examples, *validation_examples])
+            self._payload_bytes = 0
+            if config.loss_kind == "kl":
+                # The teacher of this call is the reader as it is now; nothing after this line recomputes it.
+                if reporter is not None:
+                    reporter.set_status(phase="teacher targets")
+                    reporter.render(force=True)
+                stats.teacher_targets = self._store_teacher_targets(ac_model, [*examples, *reporting_examples, *validation_examples], device, reporter)
+                if reporter is not None:
+                    reporter.report_teacher_targets(stats.teacher_targets)
+            penalty_examples = [example.penalty for example in examples if example.penalty is not None]
+            if penalty_examples:
+                # The penalty's reference is the base with no adapter on the no-context history: fixed for the whole call, and kept next to
+                # the checkpoints so a resumed run skips the reference pass.
+                if reporter is not None:
+                    reporter.set_status(phase="no-context reference")
+                    reporter.render(force=True)
+                reference_path = resolve_path(f"AC_MODELS/{ac_model.name}/no_context_reference.pt", create=False)
+                if reference_path.exists():
+                    loaded = self._merge_teacher_targets(reference_path, {example.teacher_cache_key for example in penalty_examples})
+                    print(f"AC resume: {loaded} no-context references loaded from {reference_path}", flush=True)
+                stats.no_context_reference = self._store_teacher_targets(ac_model, penalty_examples, device, reporter, reporting_name="no-context reference", lora=None)
+                if stats.no_context_reference["examples"] or not reference_path.exists():
+                    keys = {example.teacher_cache_key for example in penalty_examples}
+                    reference_path.parent.mkdir(parents=True, exist_ok=True)
+                    torch.save({"top_k": config.teacher_cache_top_k, "targets": {k: v for k, v in self.teacher_cache.items() if k in keys},
+                                "target_logp": {k: v for k, v in self.teacher_target_logp.items() if k in keys}}, reference_path)
             ac_parameters = ac_model.trainable_parameters()
             target_parameters = manager.lora_parameters(target_lora)
             assert ac_parameters and target_parameters
-            optimizer = persistent_optimizer(self.optimizers, ac_model_name,
-                    [{"params": ac_parameters, "lr": config.learning_rate_ac}, {"params": target_parameters, "lr": config.learning_rate_target_lora}],
+            gate_parameters = [parameter for parameter in ac_parameters if parameter.numel() == 1]     # the scalar gates (row scale, input / position / residual scales): traced
+            gate_names = {id(parameter): name for name, parameter in ac_model.modules.named_parameters() if parameter.numel() == 1}
+            groups = [{"params": ac_parameters, "lr": config.learning_rate_ac}, {"params": target_parameters, "lr": config.learning_rate_target_lora}]
+            if reset_optimizer:
+                self.optimizers.pop(ac_model_name, None)
+            optimizer = persistent_optimizer(self.optimizers, ac_model_name, groups,
                     device, betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay)
+            if config.resume_optimizer and ac_model_name not in self.resumed and ac_model.config.checkpoint_path:
+                saved = Path(ac_model.config.checkpoint_path) / "optimizer.pt"
+                if saved.exists():
+                    state = torch.load(saved, map_location="cpu", weights_only=False)
+                    saved_layout = [len(group["params"]) for group in state["optimizer"]["param_groups"]]
+                    current_layout = [len(group["params"]) for group in groups]
+                    if saved_layout == current_layout:
+                        optimizer.load_state_dict(state["optimizer"])
+                        for group, wanted in zip(optimizer.param_groups, groups):   # load_state_dict also restores the saved hyper-parameters: the arm's rates win
+                            group["lr"] = group["base_lr"] = wanted["lr"]
+                            group["betas"] = tuple(config.adam_betas); group["eps"] = config.adam_eps; group["weight_decay"] = config.weight_decay
+                        optimizer_state_to(optimizer, device)
+                        stats.resumed_from = str(saved)
+                    else:
+                        # The group layout changed since the checkpoint (a checkpoint from the full-featured branch with extra groups, a migrated
+                        # checkpoint): the moments cannot be mapped, so the optimizer starts fresh and only the update clock resumes. A requeued run never dies here.
+                        print(f"AC resume: optimizer state discarded (layout changed: saved groups {saved_layout}, current {current_layout}); "
+                              f"fresh moments, update clock resumed at {int(state['updates_done'])}", flush=True)
+                        stats.resumed_from = f"{saved} (moments discarded: layout changed)"
+                    self.updates_done[ac_model_name] = int(state["updates_done"])
+                    self.resumed.add(ac_model_name)
+                    print(f"AC resume: optimizer moments and {state['updates_done']} updates restored from {saved}", flush=True)
+            if device.type == "cuda":
+                torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True   # fp32 GEMMs (LoRA path, gates) on tensor cores
             if device.type == "cuda":
                 torch.cuda.reset_peak_memory_stats(device)
             stats.placement_seconds = time.monotonic() - placement_started
@@ -389,13 +530,24 @@ class ActivationContextTrainer:
                         self.updates_done.get(ac_model_name, 0), 0.0, time.monotonic() - started)
             phase = time.monotonic()
             if reporting_data:
-                stats.reporting = self._evaluate(ac_model, list(reporting_data), device)
+                stats.reporting = self._evaluate(ac_model, list(reporting_data), device, reporter, "reporting")
                 if reporter is not None:
                     reporter.report_eval(baseline, "reporting", stats.reporting)
+            deltas_off_panel = config.deltas_off_panel and ac_model.is_deep and bool(reporting_data)
+            if deltas_off_panel:
+                stats.deltas_off = self._evaluate(ac_model, list(reporting_data), device, reporter, "deltas_off", deltas_off=True)
+                if reporter is not None:
+                    reporter.report_eval(baseline, "deltas_off", stats.deltas_off)
+            if validation_data and list(validation_data) != list(reporting_data):
+                stats.validation_baseline = self._evaluate(ac_model, list(validation_data), device, reporter, "validation")
+                if reporter is not None:
+                    reporter.report_eval(baseline, "validation", stats.validation_baseline)
             stats.baseline_samples = self._sample_completions(ac_model, fixed_items, baseline)
             stats.baseline_seconds = time.monotonic() - phase
             if reporter is not None and stats.baseline_samples:
                 reporter.report_completions(baseline, stats.baseline_samples)
+            planned_total = config.schedule_total_updates or (self.updates_done.get(ac_model_name, 0) + steps * config.num_epochs)
+            reader_frozen = config.learning_rate_target_lora == 0.0     # a frozen reader neither steps nor takes part in the clipping norm
             for epoch in range(1, config.num_epochs + 1):
                 epoch_number = start_epoch + epoch - 1
                 epoch_started = time.monotonic()
@@ -406,33 +558,62 @@ class ActivationContextTrainer:
                 order = list(range(len(examples)))
                 random.Random(config.seed + epoch_number).shuffle(order)
                 for step in range(1, steps + 1):
+                    if epoch == 1 and step <= config.resume_skip_steps:
+                        continue        # trained by the previous attempt (same seeded order); the mid-epoch checkpoint carries the moments and the update count
                     if reporter is not None:
                         reporter.report_phase(progress(step - 1), "training")
                     step_started = time.monotonic()
                     before_side_tokens = ac_model.stats.side_tokens
                     mini = order[(step - 1) * per_step:step * per_step]
                     optimizer.zero_grad(set_to_none=True)
-                    totals = {"kl": 0.0, "agreement": 0.0, "drift": 0.0, "tokens": 0}
-                    for index in mini:
-                        example = examples[index]
+                    totals = {"loss": 0.0, "agreement": 0.0, "tokens": 0, "penalty": []}
+                    batched = config.micro_batch_examples > 1 and config.loss_kind == "sft"
+                    if batched:
+                        mini = sorted(mini, key=lambda index: len(examples[index].student_ids))   # same examples, less padding per micro-batch
+                        groups = [mini[i:i + config.micro_batch_examples] for i in range(0, len(mini), config.micro_batch_examples)]
+                    else:
+                        groups = [[index] for index in mini]
+                    for group in groups:
+                        group_examples = [examples[index] for index in group]
                         example_started = time.monotonic()
-                        set_checkpointing(target.model, config, len(example.teacher_ids), min_tokens)
-                        cached = config.teacher_cache_top_k > 0 and example.teacher_cache_key in self.teacher_cache
-                        kl, agreement, drift, _ = self._example_loss(ac_model, example, device)
-                        loss = (kl + config.drift_term_weight * drift) * (example.item.weight / len(mini))
+                        if batched:
+                            outcomes = [(result["objective"], None, result["penalty"]) for result in self._batch_loss(ac_model, group_examples, device)]
+                        else:
+                            objective, agreement, _, _ = self._example_loss(ac_model, group_examples[0], device)
+                            outcomes = [(objective, agreement, self._penalty_term)]
+                        loss = sum((objective + (penalty_term if penalty_term is not None else 0.0)) * (example.item.weight / len(mini))
+                                   for example, (objective, _, penalty_term) in zip(group_examples, outcomes))
                         loss.backward()
-                        kl_value, agreement_value = float(kl.detach()), float(agreement)
-                        stats.example_records.append({"kind": example.item.kind, "teacher_tokens": len(example.teacher_ids),
-                            "student_tokens": len(example.student_ids), "seconds": round(time.monotonic() - example_started, 3), "teacher_cached": cached})
-                        stats.teacher_cache_hits += int(cached)
-                        stats.kl_by_kind.setdefault(example.item.kind, []).append(kl_value)
-                        stats.agreement_by_kind.setdefault(example.item.kind, []).append(agreement_value)
-                        totals["kl"] += kl_value * example.item.weight
-                        totals["agreement"] += agreement_value
-                        totals["drift"] += float(drift.detach()) * example.item.weight
-                        totals["tokens"] += len(example.teacher_ids) + len(example.student_ids)
-                    grad_norm = float(torch.nn.utils.clip_grad_norm_(ac_parameters + target_parameters, config.max_grad_norm))
-                    set_learning_rates(optimizer, self.updates_done.get(ac_model_name, 0), config.warmup_updates)
+                        seconds_each = (time.monotonic() - example_started) / len(group)
+                        for example, (objective, agreement, penalty_term) in zip(group_examples, outcomes):
+                            cached = config.loss_kind == "kl" and example.teacher_cache_key in self.teacher_cache
+                            if penalty_term is not None:
+                                totals["penalty"].append(float(penalty_term.detach()) * config.no_context_penalty_fraction / config.no_context_penalty_weight)
+                            loss_value = float(objective.detach())
+                            stats.example_records.append({"kind": example.item.kind, "teacher_tokens": len(example.teacher_ids),
+                                "student_tokens": len(example.student_ids), "seconds": round(seconds_each, 3), "teacher_cached": cached})
+                            stats.teacher_cache_hits += int(cached)
+                            stats.loss_by_kind.setdefault(example.item.kind, []).append(loss_value)
+                            totals["loss"] += loss_value * example.item.weight
+                            if config.loss_kind == "kl":
+                                stats.kl_by_kind.setdefault(example.item.kind, []).append(loss_value)
+                                stats.agreement_by_kind.setdefault(example.item.kind, []).append(agreement)
+                                totals["agreement"] += agreement
+                            totals["tokens"] += len(example.teacher_ids) + len(example.student_ids)
+                    stats.penalty_counts.append(len(totals["penalty"]))
+                    if reader_frozen:
+                        for parameter in target_parameters:
+                            parameter.grad = None
+                    norm_ac = self._grad_norm(ac_parameters); norm_reader = self._grad_norm(target_parameters)
+                    head_norms = self._head_grad_norms(ac_model) if ac_model.is_deep else None       # before clipping, like norm_ac
+                    grad_norm = float(torch.nn.utils.clip_grad_norm_(ac_parameters + ([] if reader_frozen else target_parameters), config.max_grad_norm))
+                    set_learning_rates(optimizer, self.updates_done.get(ac_model_name, 0), config.warmup_updates,
+                                       total_updates=planned_total, schedule=config.lr_schedule, final_fraction=config.lr_final_fraction)
+                    stats.grad_norm_ac.append(norm_ac); stats.grad_norm_reader.append(norm_reader)
+                    if self.updates_done.get(ac_model_name, 0) % 25 == 0:
+                        stats.gate_trace.append({"update": self.updates_done.get(ac_model_name, 0), **{gate_names.get(id(parameter), str(index)): float(parameter.detach()) for index, parameter in enumerate(gate_parameters)}})
+                    if ac_model.is_deep and self._intervention_trace_due(self.updates_done.get(ac_model_name, 0)):
+                        stats.intervention_trace.append(self._intervention_trace_entry(ac_model, self.updates_done.get(ac_model_name, 0), head_norms))
                     encoder_has_grad = any(parameter.grad is not None for parameter in ac_parameters)
                     try:
                         optimizer.step()
@@ -443,9 +624,12 @@ class ActivationContextTrainer:
                     seconds = time.monotonic() - step_started
                     record.training_seconds += seconds
                     stats.steps += 1
-                    stats.kl.append(totals["kl"] / len(mini))
-                    stats.agreement.append(totals["agreement"] / len(mini))
-                    stats.drift.append(totals["drift"] / len(mini))
+                    stats.loss.append(totals["loss"] / len(mini))
+                    if config.loss_kind == "kl":
+                        stats.kl.append(stats.loss[-1])
+                        stats.agreement.append(totals["agreement"] / len(mini))
+                    if config.penalty_active:
+                        stats.penalty.append(sum(totals["penalty"]) / len(totals["penalty"]) if totals["penalty"] else (stats.penalty[-1] if stats.penalty else 0.0))
                     stats.grad_norm.append(grad_norm)
                     stats.step_seconds.append(seconds)
                     stats.step_tokens.append(totals["tokens"])
@@ -453,24 +637,31 @@ class ActivationContextTrainer:
                     stats.global_steps.append(self.updates_done[ac_model_name])
                     stats.learning_rates.append([float(group["lr"]) for group in optimizer.param_groups])
                     print(f"AC epoch {epoch}/{config.num_epochs} step {step}/{steps} global {stats.global_steps[-1]}: "
-                          f"kl {stats.kl[-1]:.4f}, agreement {stats.agreement[-1]:.3f}, {seconds:.1f}s", flush=True)
+                          f"{config.loss_kind} loss {stats.loss[-1]:.4f}, {seconds:.1f}s", flush=True)
                     if reporter is not None:
                         reporter.report_step(progress(step), stats)
+                    if config.checkpoint_every_updates and config.checkpoint_every_epoch and step < steps and step % config.checkpoint_every_updates == 0:
+                        self._save_partial(ac_model, epoch_number, step, steps)
                     if step in boundaries and reporting_data:
                         phase = time.monotonic()
                         if reporter is not None:
                             reporter.report_phase(progress(step), "reporting")
-                        record.reporting = self._evaluate(ac_model, list(reporting_data), device)
-                        record.reporting_seconds += time.monotonic() - phase
+                        record.reporting = self._evaluate(ac_model, list(reporting_data), device, reporter, "reporting")
                         stats.reporting = record.reporting
                         if reporter is not None:
                             reporter.report_eval(progress(step), "reporting", record.reporting)
+                        if deltas_off_panel:
+                            record.deltas_off = self._evaluate(ac_model, list(reporting_data), device, reporter, "deltas_off", deltas_off=True)
+                            stats.deltas_off = record.deltas_off
+                            if reporter is not None:
+                                reporter.report_eval(progress(step), "deltas_off", record.deltas_off)
+                        record.reporting_seconds += time.monotonic() - phase
                 if validation_data:
                     phase = time.monotonic()
                     if reporter is not None:
                         reporter.report_phase(progress(steps), "validation")
                     record.validation = (record.reporting if list(validation_data) == list(reporting_data) and record.reporting is not None
-                                         else self._evaluate(ac_model, list(validation_data), device))
+                                         else self._evaluate(ac_model, list(validation_data), device, reporter, "validation"))
                     record.validation_seconds = time.monotonic() - phase
                     if reporter is not None:
                         reporter.report_eval(progress(steps), "validation", record.validation)
@@ -496,6 +687,7 @@ class ActivationContextTrainer:
                 self.epochs_done[ac_model_name] = epoch_number
                 ac_model.epoch_number = epoch_number
             stats.teacher_cache_bytes = sum(sum(tensor.numel() * tensor.element_size() for tensor in entry) for entry in self.teacher_cache.values())
+            stats.payload_bytes = self._payload_bytes
         finally:
             kernel_scope.close()
             cleanup_started = time.monotonic()
@@ -505,12 +697,15 @@ class ActivationContextTrainer:
                 base.eval()
                 if base.is_gradient_checkpointing:
                     base.gradient_checkpointing_disable()
+            ac_model.__dict__.pop("_training_gradient_checkpointing", None)
+            ac_model.__dict__.pop("_side_gradient_checkpointing_min_tokens", None)
             ac_model.set_mode(ROLLOUT)
             if device is not None and device.type == "cuda":
                 stats.peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
             ac_model.release()
             target.model_to_device(SOURCE_DEVICE)
-            ac_model.side.model_to_device(SOURCE_DEVICE)
+            if ac_model.side is not target:
+                ac_model.side.model_to_device(SOURCE_DEVICE)
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()
         stats.cleanup_seconds = time.monotonic() - cleanup_started
@@ -541,10 +736,25 @@ class ActivationContextTrainer:
             raise FileExistsError(f"Epoch {epoch_number} already has checkpoint files")
         ac_path = ac_model.save(str(ac_folder), epoch_number=epoch_number)
         target_path = manager.save_lora(ac_model.target.model_config.model_name, target_lora, str(target_folder))
+        if self.teacher_cache and self.config.loss_kind == "kl":
+            self.save_teacher_targets(ac_folder / "teacher_targets.pt")     # the reference this run was trained against (an SFT run's cache only holds penalty references)
         for source, latest in ((ac_folder, ac_model.checkpoint_folder()), (target_folder, manager.checkpoint_folder(target_lora))):
             if latest.exists():
                 shutil.rmtree(latest)
             shutil.copytree(source, latest)
+        optimizer = self.optimizers.get(ac_model.name)
+        if optimizer is not None:               # resume material: moments + the update count (the LR schedule and the reader freeze window read it)
+            torch.save({"optimizer": optimizer.state_dict(), "updates_done": self.updates_done.get(ac_model.name, 0), "epoch_number": epoch_number},
+                       ac_folder / "optimizer.pt")
+            for previous in sorted(ac_folder.parent.glob("epoch_*/optimizer.pt")):
+                if previous.parent != ac_folder:
+                    previous.unlink()           # only the newest epoch keeps its optimizer (about 2x the adapter size)
+        for root in (ac_folder.parent, target_folder.parent):
+            for partial in root.glob(f"epoch_{epoch_number:03d}_step_*"):
+                shutil.rmtree(partial)      # the completed epoch supersedes its mid-epoch checkpoints
+        partial_record = ac_folder.parent / "partial_epoch.json"
+        if partial_record.exists() and json.loads(partial_record.read_text())["epoch_number"] <= epoch_number:
+            partial_record.unlink()
         record = ac_folder.parent / "completed_epoch.json"
         temporary = record.with_suffix(".json.tmp")
         temporary.write_text(json.dumps({"epoch_number": epoch_number, "ac_checkpoint": ac_relative,
@@ -552,6 +762,45 @@ class ActivationContextTrainer:
         temporary.replace(record)       # only this record declares the numbered pair complete
         return ac_path, target_path
 
+    def _save_partial(self, ac_model: ActivationContextModel, epoch_number: int, step: int, steps: int) -> None:
+        """Mid-epoch checkpoint: adapters as of `step` of `epoch_number` (saved with epoch_number - 1 so a resume restarts this epoch), optimizer.pt,
+        and AC_MODELS/<name>/partial_epoch.json; earlier partials of the same epoch are removed."""
+        manager = self.harness.module_manager
+        target_lora = ac_model.config.target_model_lora_name
+        leaf = f"epoch_{epoch_number:03d}_step_{step:05d}"
+        ac_relative, target_relative = f"AC_MODELS/{ac_model.name}/{leaf}", f"LORAS/{target_lora}/{leaf}"
+        ac_folder, target_folder = resolve_path(ac_relative, create=False), resolve_path(target_relative, create=False)
+        for folder in (ac_folder, target_folder):
+            if folder.exists():
+                shutil.rmtree(folder)
+        ac_model.save(str(ac_folder), epoch_number=epoch_number - 1)
+        manager.save_lora(ac_model.target.model_config.model_name, target_lora, str(target_folder))
+        optimizer = self.optimizers.get(ac_model.name)
+        if optimizer is not None:
+            torch.save({"optimizer": optimizer.state_dict(), "updates_done": self.updates_done.get(ac_model.name, 0), "epoch_number": epoch_number - 1, "step": step},
+                       ac_folder / "optimizer.pt")
+        record = ac_folder.parent / "partial_epoch.json"
+        temporary = record.with_suffix(".json.tmp")
+        temporary.write_text(json.dumps({"epoch_number": epoch_number, "step": step, "steps": steps, "ac_checkpoint": ac_relative,
+                                        "target_lora_checkpoint": target_relative}, indent=2) + "\n")
+        temporary.replace(record)
+        for root in (ac_folder.parent, target_folder.parent):
+            for previous in root.glob(f"epoch_{epoch_number:03d}_step_*"):
+                if previous.name != leaf:
+                    shutil.rmtree(previous)
+        print(f"AC checkpoint: epoch {epoch_number} step {step}/{steps} saved to {ac_folder}", flush=True)
+
+    def _merge_teacher_targets(self, path: Path, keys: set | None) -> int:
+        """Add the saved reference entries for `keys` (None: all) that are not stored yet (same top-k required). Returns how many were added."""
+        data = torch.load(path, map_location="cpu", weights_only=False)
+        if data["top_k"] != self.config.teacher_cache_top_k:
+            return 0
+        added = 0
+        for key, value in data["targets"].items():
+            if (keys is None or key in keys) and key not in self.teacher_cache:
+                self.teacher_cache[key] = value; self.teacher_target_logp[key] = data["target_logp"][key]; added += 1
+        return added
+
     @torch.no_grad()
     def _sample_completions(self, ac_model: ActivationContextModel, items: list[ActivationContextTrainingItem],
                             progress: ActivationContextTrainingProgress) -> list[CompletionSample]:
@@ -559,10 +808,12 @@ class ActivationContextTrainer:
             return []
         target = ac_model.target
         manager = self.harness.module_manager
-        bases = [target.model, ac_model.side.model]
+        bases = list(dict.fromkeys([target.model, ac_model.side.model]))
         modes = [(base.training, base.is_gradient_checkpointing) for base in bases]
         previous_mode = ac_model.mode
         result = []
+        model_config = target.model.config.get_text_config() if hasattr(target.model.config, "get_text_config") else target.model.config
+        capacity = getattr(model_config, "max_position_embeddings", None)
         try:
             ac_model.set_mode(TRAINING)
             for base in bases:
@@ -571,30 +822,37 @@ class ActivationContextTrainer:
                     base.gradient_checkpointing_disable()
             for item in items:
                 example = self.build_example(ac_model, item)
-                if max(len(example.teacher_ids), len(example.student_ids)) > self.config.max_example_tokens:
-                    continue
                 texts = {}
-                for side in ("teacher", "student"):
+                # The stored teacher is the reader at baseline: its completion is sampled once (before any update) and reused.
+                teacher_now = self.config.loss_kind == "kl" and item.item_id not in self._teacher_texts
+                for side in (("teacher", "student") if teacher_now else ("student",)):
                     sequence = example.teacher_ids if side == "teacher" else example.student_ids
-                    ids = torch.tensor(sequence[:-example.num_completion], device=ac_model.device)
+                    positions = example.teacher_positions if side == "teacher" else example.student_positions
+                    prefix_length = positions[0] + 1
+                    ids = torch.tensor(sequence[:prefix_length], device=ac_model.device)
                     embeds = target.model.get_input_embeddings()(ids)
+                    payload = None
                     if side == "student" and example.part_requests:
-                        rows = ac_model.encode_batch(example.part_requests)
-                        pieces, cursor = [], 0
-                        for (start, end), part_rows in zip(example.spans, rows):
-                            pieces.extend([embeds[cursor:start], part_rows.to(embeds.dtype)])
-                            cursor = end
-                        embeds = torch.cat(pieces + [embeds[cursor:]], dim=0)
+                        present = [(span, request) for span, request in zip(example.spans, example.part_requests) if span[1] <= prefix_length]
+                        outputs = ac_model.encode_batch([request for _, request in present], with_layers=True) if present else []
+                        embeds, payload = self._assemble_student(embeds, [span for span, _ in present], outputs, embeds.dtype)
                     lora = ac_model.config.target_model_lora_name
                     peft_model = manager.ensure_lora(lora)
+                    budget = min(self.config.completion_max_new_tokens, capacity - prefix_length) if capacity else self.config.completion_max_new_tokens
+                    generation = dict(max_new_tokens=budget, do_sample=False, use_cache=True,
+                                      pad_token_id=target.tokenizer.pad_token_id if target.tokenizer.pad_token_id is not None else target.tokenizer.eos_token_id)
                     with manager.lora_context(target.model_config.model_name, lora):
-                        output = peft_model.generate(inputs_embeds=embeds[None], max_new_tokens=self.config.completion_max_new_tokens,
-                                        do_sample=False, use_cache=True,
-                                        pad_token_id=target.tokenizer.pad_token_id if target.tokenizer.pad_token_id is not None else target.tokenizer.eos_token_id)
+                        if payload is not None:
+                            output = generate_with_interventions(peft_model, inputs_embeds=embeds[None], interventions=[payload], **generation)
+                        else:
+                            output = peft_model.generate(inputs_embeds=embeds[None], **generation)
                     texts[side] = target.tokenizer.decode(output[0], skip_special_tokens=True)
+                if "teacher" in texts:
+                    self._teacher_texts[item.item_id] = texts["teacher"]
                 result.append(CompletionSample(item_id=item.item_id, kind=item.kind,
-                    reference=str(item.info.get("answer") or item.info.get("observed_terminal_answer") or item.completion_text),
-                    teacher=texts["teacher"], student=texts["student"], epoch_number=progress.epoch_number,
+                    reference=target.tokenizer.decode(item.assistant_token_ids[0], skip_special_tokens=False),
+                    teacher=self._teacher_texts.get(item.item_id) if self.config.loss_kind == "kl" else None,
+                    student=texts["student"], epoch_number=progress.epoch_number,
                     global_step=progress.global_step, ac_version=ac_model.version, target_lora=ac_model.config.target_model_lora_name))
         finally:
             for base, (training, checkpointed) in zip(bases, modes):
@@ -616,11 +874,17 @@ class ActivationContextTrainer:
         *,
         reference_name: ReferenceName | None = None,
         reference_provenance: str = "initial adapter; held set",
+        teacher: str = "stored",
     ) -> ActivationContextEvalSummary:
         """
-        KL and agreement of the items without gradient (models placed as for training, moved back when `release`).
-        A supplied reporter receives this exact evaluation under reporting_name.
+        Loss, agreement and target log-probs of the items without gradient (models placed as for training, moved back
+        when `release`). KL items need stored teacher targets: `teacher="stored"` (default) fails when an item has none,
+        so a reloaded reader is never silently made its own teacher; `teacher="current"` stores missing targets from the
+        reader as it is now (right before training, or after `load_teacher_targets`, that is the intended reference).
+        A supplied reporter receives this evaluation under reporting_name.
         """
+        if teacher not in ("stored", "current"):
+            raise ValueError("teacher must be 'stored' or 'current'")
         module_manager = self.harness.module_manager
         ac_model = module_manager.get_ac_model(ac_model_name)
         target = ac_model.target
@@ -632,54 +896,94 @@ class ActivationContextTrainer:
             kernel_scope.enter_context(configure_fla_runtime(variables["fla_config"]))
             module_manager.ensure_lora(ac_model.config.target_model_lora_name)
             device = ac_model.prepare()
-            summary = self._evaluate(ac_model, items, device)
+            if teacher == "current" and self.config.loss_kind == "kl":
+                self._store_teacher_targets(ac_model, [self.build_example(ac_model, item) for item in items], device, reporter)
+            summary = self._evaluate(ac_model, items, device, reporter, reporting_name or reference_name or "evaluation")
             if reporter is not None:
                 reporter.report_eval(progress, reporting_name or reference_name or "evaluation", summary)
-                if reference_name is not None:
-                    reporter.report_reference(reference_name, summary["kl"], provenance=reference_provenance)
+                if reference_name is not None and summary["items"]:
+                    reporter.report_reference(reference_name, summary["loss"], provenance=reference_provenance, loss_kind=self.config.loss_kind)
             return summary
         finally:
             kernel_scope.close()
             if release:
                 ac_model.release()
                 target.model_to_device(SOURCE_DEVICE)
-                ac_model.side.model_to_device(SOURCE_DEVICE)
+                if ac_model.side is not target:
+                    ac_model.side.model_to_device(SOURCE_DEVICE)
 
     # ------------------------------------------------------------------------------------------ internals
     @torch.no_grad()
-    def _evaluate(self, ac_model: ActivationContextModel, items: list[ActivationContextTrainingItem], device: torch.device) -> ActivationContextEvalSummary:
+    def _evaluate(self, ac_model: ActivationContextModel, items: list[ActivationContextTrainingItem], device: torch.device,
+                  reporter: "ActivationContextTrainingReporter | None" = None, reporting_name: str = "evaluation", deltas_off: bool = False) -> ActivationContextEvalSummary:
+        """`deltas_off`: a deep model's items scored with their rows kept and their per-layer deltas withheld (the deltas' contribution)."""
+        if reporter is not None:
+            reporter.report_evaluation_progress(reporting_name, 0, len(items))
+        examples = [self.build_example(ac_model, item) for item in items]
+        self._check_capacity(ac_model, examples)
         previous_mode = ac_model.mode
-        ac_model.set_mode(TRAINING)                                          # no cache; the no_grad above makes it free of gradients
-        target_base = ac_model.target.model
-        training_modes = [target_base.training, ac_model.side.model.training]
-        target_base.eval()
-        ac_model.side.model.eval()
+        ac_model.set_mode(TRAINING)
+        bases = list(dict.fromkeys([ac_model.target.model, ac_model.side.model]))
+        training_modes = [base.training for base in bases]
+        for base in bases:
+            base.eval()
         per_item = []
-        dropped = 0
+        is_kl = self.config.loss_kind == "kl"
         try:
-            for item in items:
-                example = self.build_example(ac_model, item)
-                if len(example.teacher_ids) > self.config.max_example_tokens or len(example.student_ids) > self.config.max_example_tokens:
-                    dropped += 1
-                    continue
-                kl, agreement, _, positions = self._example_loss(ac_model, example, device, with_drift=False)
-                per_item.append({"item_id": item.item_id, "kind": item.kind, "kl": float(kl), "agreement": float(agreement), "positions": positions,
-                                 "teacher_tokens": len(example.teacher_ids), "student_tokens": len(example.student_ids)})
+            def row(example, loss, agreement, positions, target_logp, hits):
+                teacher_logp = self.teacher_target_logp.get(example.teacher_cache_key) if is_kl else None
+                return {"item_id": example.item.item_id, "kind": example.item.kind,
+                    "loss_kind": self.config.loss_kind, "loss": float(loss), "kl": float(loss) if is_kl else None,
+                    "agreement": agreement, "positions": positions,
+                    "token_accuracy": float(hits.float().mean()) if hits is not None and hits.numel() else 0.0,
+                    "teacher_tokens": len(example.teacher_ids), "student_tokens": len(example.student_ids),
+                    "gold_nll": float(-target_logp.mean()), "target_logp": [round(float(value), 4) for value in target_logp],
+                    "teacher_gold_nll": float(-teacher_logp.mean()) if teacher_logp is not None else None,
+                    "teacher_target_logp": [round(float(value), 4) for value in teacher_logp] if teacher_logp is not None else None}
+            rows: list[dict | None] = [None] * len(examples)
+            done = 0
+            if self.config.micro_batch_examples > 1 and not is_kl:
+                order = sorted(range(len(examples)), key=lambda index: len(examples[index].student_ids))
+                for start in range(0, len(order), self.config.micro_batch_examples):
+                    group = order[start:start + self.config.micro_batch_examples]
+                    for index, result in zip(group, self._batch_loss(ac_model, [examples[index] for index in group], device, with_penalty=False, deltas_off=deltas_off)):
+                        rows[index] = row(examples[index], result["objective"], None, result["num"], result["target_logp"], result["hits"])
+                    done += len(group)
+                    if reporter is not None:
+                        reporter.report_evaluation_progress(reporting_name, done, len(examples))
+            else:
+                for index, example in enumerate(examples):
+                    loss, agreement, positions, target_logp = self._example_loss(ac_model, example, device, with_penalty=False, deltas_off=deltas_off)
+                    rows[index] = row(example, loss, agreement, positions, target_logp, self._last_hits)
+                    done += 1
+                    if reporter is not None:
+                        reporter.report_evaluation_progress(reporting_name, done, len(examples))
+            per_item = rows
         finally:
-            for base, training in zip((target_base, ac_model.side.model), training_modes):
+            for base, training in zip(bases, training_modes):
                 base.train(training)
                 disable_dropout(base)
             ac_model.set_mode(previous_mode)
-        by_kind: dict[str, list[dict]] = {}
-        for row in per_item:
-            by_kind.setdefault(row["kind"], []).append(row)
-        summary = {kind: {"kl": sum(r["kl"] for r in rows) / len(rows), "agreement": sum(r["agreement"] for r in rows) / len(rows), "items": len(rows)}
-                   for kind, rows in by_kind.items()}
-        return {"items": len(per_item), "dropped_too_long": dropped, "kl": sum(r["kl"] for r in per_item) / max(1, len(per_item)),
-                "agreement": sum(r["agreement"] for r in per_item) / max(1, len(per_item)), "by_kind": summary, "per_item": per_item}
-
-    def _example_loss(self, ac_model: ActivationContextModel, example: _Example, device: torch.device, with_drift: bool = True) -> tuple[torch.Tensor, float, torch.Tensor, int]:
-        """(kl, agreement, drift, positions) of one example: teacher no-grad, student with the rows; drift on the in-context tokens when configured."""
+        def summarize(rows):
+            loss = sum(row["loss"] for row in rows) / len(rows) if rows else None
+            teacher_rows = [row["teacher_gold_nll"] for row in rows if row["teacher_gold_nll"] is not None]
+            return {"loss_kind": self.config.loss_kind, "loss": loss, "kl": loss if is_kl else None,
+                    "agreement": sum(row["agreement"] for row in rows) / len(rows) if is_kl and rows else None,
+                    "gold_nll": sum(row["gold_nll"] for row in rows) / len(rows) if rows else None,
+                    "gold_nll_token": -sum(sum(row["target_logp"]) for row in rows) / max(1, sum(len(row["target_logp"]) for row in rows)) if rows else None,   # token-weighted: long passages count by their length
+                    "token_accuracy": sum(row["token_accuracy"] for row in rows) / len(rows) if rows else None,
+                    "teacher_gold_nll": sum(teacher_rows) / len(teacher_rows) if teacher_rows else None,
+                    "items": len(rows)}
+        by_kind = {kind: summarize([row for row in per_item if row["kind"] == kind])
+                   for kind in dict.fromkeys(row["kind"] for row in per_item)}
+        return {**summarize(per_item), "dropped_too_long": 0, "by_kind": by_kind, "per_item": per_item}
+
+    def _example_loss(self, ac_model: ActivationContextModel, example: _Example, device: torch.device, with_penalty: bool = True, deltas_off: bool = False) -> tuple[torch.Tensor, float | None, int, torch.Tensor]:
+        """
+        (loss, agreement, positions, target log-probs) of one example: the student with the rows against the stored teacher
+        targets (KL) or the target tokens (SFT). With `with_penalty` (training forwards) the sampled no-context penalty term
+        lands in `_penalty_term`. The last value is the student's log-prob of every target token, detached, on `device`.
+        """
         config = self.config
         target = ac_model.target
         base = target.model
@@ -688,47 +992,314 @@ class ActivationContextTrainer:
         head = base.get_output_embeddings()
         lora = ac_model.config.target_model_lora_name
         num = example.num_completion
-        top_k = config.teacher_cache_top_k if torch.is_grad_enabled() else 0                  # eval is exact
-        cached = self.teacher_cache.get(example.teacher_cache_key) if top_k else None
-        teacher_ids = torch.tensor(example.teacher_ids, device=device)
-        teacher_states = None
-        if cached is None:
-            # Teacher.
-            with torch.no_grad():
-                teacher_hidden = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=lora)[0]
-                teacher_states = teacher_hidden[-num - 1:-1]
-            if top_k:
-                cached = self._teacher_top_k(teacher_states, head, top_k)
-                self.teacher_cache[example.teacher_cache_key] = tuple(tensor.cpu() for tensor in cached)
-                teacher_states = None
-        else:
+        is_kl = config.loss_kind == "kl"
+        cached = None
+        if is_kl:
+            cached = self.teacher_cache.get(example.teacher_cache_key)
+            if cached is None:
+                raise RuntimeError(f"{example.item.item_id}: no stored teacher targets; train() stores them before its first update "
+                                   "and eval() needs them for its items (teacher='current' stores them from the current reader)")
             cached = tuple(tensor.to(device, non_blocking=True) for tensor in cached)
-        # Student: rows for the parts (grad), written over the placeholders.
-        rows = ac_model.encode_batch(example.part_requests) if example.part_requests else []
+        target_ids = torch.tensor(example.target_ids, device=device, dtype=torch.long)
+        # Student: rows for the parts (grad), written over the placeholders; a deep model's deltas travel as one payload.
+        outputs = ac_model.encode_batch(example.part_requests, with_layers=True) if example.part_requests else []
         student_ids = torch.tensor(example.student_ids, device=device)
         with torch.no_grad():
             student_embeds = embedding(student_ids)
-        if rows:
-            assert [int(part_rows.shape[0]) for part_rows in rows] == [end - start for start, end in example.spans], "placeholder runs differ from the encoded rows"
-            pieces, cursor = [], 0
-            for (start, end), part_rows in zip(example.spans, rows):
-                pieces.extend([student_embeds[cursor:start], part_rows.to(base_dtype)])
-                cursor = end
-            pieces.append(student_embeds[cursor:])
-            student_embeds = torch.cat(pieces, dim=0)
-        student_hidden = target.decoder_forward(student_embeds[None], None, lora_name=lora)[0]
-        student_states = student_hidden[-num - 1:-1]
-        kl, agreement = self._chunked_kl(teacher_states, student_states, head, cached=cached)
-        drift = torch.zeros((), device=device)
-        if with_drift and config.drift_term_weight > 0:
-            # The adapter's in-context behaviour on the last in-context positions, held to the frozen base's.
-            prefix = len(example.teacher_ids) - num
-            window = slice(max(0, prefix - 1024), prefix)
+        student_embeds, payload = self._assemble_student(student_embeds, example.spans, outputs, base_dtype, deltas_off=deltas_off)
+        checkpointed = (config.gradient_checkpointing and torch.is_grad_enabled()
+                        and len(example.student_ids) >= getattr(self, "_target_gradient_checkpointing_min_tokens", config.gradient_checkpointing_min_tokens or 0))
+        student_hidden = target.decoder_forward(student_embeds[None], None, lora_name=lora, gradient_checkpointing=checkpointed,
+                                                interventions=[payload] if payload is not None else None)[0]
+        if torch.is_grad_enabled():
+            self._harvest_row_rms([payload])
+        student_states = student_hidden[example.student_positions]
+        if is_kl:
+            loss, agreement, target_logp = self._chunked_kl(None, student_states, head, cached=cached, target_ids=target_ids)
+        else:
+            loss, target_logp = self._chunked_sft(student_states, head, target_ids)
+            agreement = None
+        self._penalty_term = None
+        if with_penalty and example.penalty is not None and config.penalty_active and self._penalty_rng.random() < config.no_context_penalty_fraction:
+            # The reader on the same history without its rows, held to the base's stored no-context distribution: any improvement that
+            # does not need the rows (the answer prior, memorizing the question) is charged; decoding the rows is free.
+            reference = example.penalty
+            cached_reference = self.teacher_cache.get(reference.teacher_cache_key)
+            if cached_reference is None:
+                raise RuntimeError(f"{reference.item.item_id}: no stored no-context reference; train() stores it before its first update")
+            cached_reference = tuple(tensor.to(device, non_blocking=True) for tensor in cached_reference)
+            reference_ids = torch.tensor(reference.student_ids, device=device, dtype=torch.long)
             with torch.no_grad():
-                frozen_states = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=None)[0][window]
-            adapted_states = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=lora)[0][window]
-            drift, _ = self._chunked_kl(frozen_states, adapted_states, head)
-        return kl, agreement, drift, num
+                reference_embeds = embedding(reference_ids)
+            reference_checkpointed = (config.gradient_checkpointing and torch.is_grad_enabled()
+                                      and len(reference.student_ids) >= getattr(self, "_target_gradient_checkpointing_min_tokens", config.gradient_checkpointing_min_tokens or 0))
+            hits = self._last_hits
+            reference_states = target.decoder_forward(reference_embeds[None], None, lora_name=lora, gradient_checkpointing=reference_checkpointed)[0][reference.student_positions]
+            penalty, _, _ = self._chunked_kl(None, reference_states, head, cached=cached_reference)
+            self._last_hits = hits
+            self._penalty_term = (config.no_context_penalty_weight / config.no_context_penalty_fraction) * penalty
+        return loss, agreement, num, target_logp
+
+    def _batch_loss(self, ac_model: ActivationContextModel, batch: t.Sequence[_Example], device: torch.device, with_penalty: bool = True, deltas_off: bool = False) -> list[dict]:
+        """
+        SFT loss of several examples from one right-padded reader forward (and one encoder call for all their parts):
+        per example {"objective", "num", "target_logp", "hits", "penalty"}, the values `_example_loss` gives one at a
+        time. Under causal attention (and the causal recurrence of the linear-attention layers) a real token never sees
+        a later pad, so the padded rows need no mask. The sampled no-context references go through a second padded
+        forward; `penalty` is the scaled term for the sampled examples and None for the rest.
+        """
+        config = self.config
+        assert config.loss_kind == "sft", "_batch_loss is the SFT path"
+        target = ac_model.target
+        base = target.model
+        base_dtype = next(base.parameters()).dtype
+        embedding = base.get_input_embeddings()
+        head = base.get_output_embeddings()
+        lora = ac_model.config.target_model_lora_name
+        requests = [request for example in batch for request in example.part_requests]
+        outputs = ac_model.encode_batch(requests, with_layers=True) if requests else []
+        embeds, payloads, cursor_rows = [], [], 0
+        for example in batch:
+            student_ids = torch.tensor(example.student_ids, device=device)
+            with torch.no_grad():
+                student_embeds = embedding(student_ids)
+            example_outputs = outputs[cursor_rows:cursor_rows + len(example.part_requests)]
+            cursor_rows += len(example.part_requests)
+            student_embeds, payload = self._assemble_student(student_embeds, example.spans, example_outputs, base_dtype, deltas_off=deltas_off)
+            embeds.append(student_embeds)
+            payloads.append(payload)
+        hidden = self._padded_forward(target, embeds, lora, payloads)
+        if torch.is_grad_enabled():
+            self._harvest_row_rms(payloads)
+        states = torch.cat([hidden[index, example.student_positions] for index, example in enumerate(batch)], dim=0)
+        targets = torch.cat([torch.tensor(example.target_ids, device=device, dtype=torch.long) for example in batch])
+        cross_entropy, token_logp, hits = self._chunked_sft_tokens(states, head, targets)
+        results, cursor = [], 0
+        for example in batch:
+            num = example.num_completion
+            results.append({"objective": cross_entropy[cursor:cursor + num].mean(), "num": num,
+                            "target_logp": token_logp[cursor:cursor + num], "hits": hits[cursor:cursor + num], "penalty": None})
+            cursor += num
+        if with_penalty and config.penalty_active:
+            sampled = [index for index, example in enumerate(batch) if example.penalty is not None and self._penalty_rng.random() < config.no_context_penalty_fraction]
+            if sampled:
+                references = [batch[index].penalty for index in sampled]
+                cached = []
+                for reference in references:
+                    entry = self.teacher_cache.get(reference.teacher_cache_key)
+                    if entry is None:
+                        raise RuntimeError(f"{reference.item.item_id}: no stored no-context reference; train() stores it before its first update")
+                    cached.append(tuple(tensor.to(device, non_blocking=True) for tensor in entry))
+                with torch.no_grad():
+                    reference_embeds = [embedding(torch.tensor(reference.student_ids, device=device, dtype=torch.long)) for reference in references]
+                reference_hidden = self._padded_forward(target, reference_embeds, lora)
+                for position, (index, reference) in enumerate(zip(sampled, references)):
+                    penalty, _, _ = self._chunked_kl(None, reference_hidden[position, reference.student_positions], head, cached=cached[position])
+                    results[index]["penalty"] = (config.no_context_penalty_weight / config.no_context_penalty_fraction) * penalty
+        return results
+
+    def _padded_forward(self, target, embeds: list[torch.Tensor], lora: str | None, payloads: list[Interventions | None] | None = None) -> torch.Tensor:
+        """Last hidden states [B, L, d] of right-padded (zero embedding) rows; no mask, the pads sit after every real token.
+        Right padding keeps every payload's positions physical, so the per-row payloads pass through unchanged."""
+        length = max(int(item.shape[0]) for item in embeds)
+        padded = torch.stack([torch.cat([item, item.new_zeros((length - item.shape[0], item.shape[1]))]) if item.shape[0] < length else item for item in embeds])
+        checkpointed = (self.config.gradient_checkpointing and torch.is_grad_enabled()
+                        and length * len(embeds) >= getattr(self, "_target_gradient_checkpointing_min_tokens", self.config.gradient_checkpointing_min_tokens or 0))   # the threshold counts the whole batch
+        interventions = list(payloads) if payloads is not None and any(payload is not None for payload in payloads) else None
+        return target.decoder_forward(padded, None, lora_name=lora, gradient_checkpointing=checkpointed, interventions=interventions)
+
+    def _assemble_student(self, student_embeds: torch.Tensor, spans: list[tuple[int, int]], outputs: list[ACOutput], dtype: torch.dtype,
+                          deltas_off: bool = False) -> tuple[torch.Tensor, Interventions | None]:
+        """The rows written over the placeholder runs, and (deep models) one payload: the runs' positions in order with every
+        part's deltas concatenated per layer. `deltas_off` keeps the rows and withholds the payload."""
+        if not outputs:
+            return student_embeds, None
+        assert [int(output.input_embeds.shape[0]) for output in outputs] == [end - start for start, end in spans], "placeholder runs differ from the encoded rows"
+        pieces, cursor = [], 0
+        for (start, end), output in zip(spans, outputs):
+            pieces.extend([student_embeds[cursor:start], output.input_embeds.to(dtype)])
+            cursor = end
+        pieces.append(student_embeds[cursor:])
+        embeds = torch.cat(pieces, dim=0)
+        if deltas_off or not outputs[0].has_interventions:
+            return embeds, None
+        positions = [position for start, end in spans for position in range(start, end)]
+        layer_inputs = {layer: torch.cat([output.layer_inputs[layer] for output in outputs], dim=0) for layer in outputs[0].layer_inputs}
+        if torch.is_grad_enabled():                                                          # training forwards only: the panels and the samples run under no_grad
+            self._payload_bytes += sum(rows.numel() * 2 for rows in layer_inputs.values())  # bf16 at the reader boundary
+        return embeds, {"positions": positions, "layer_inputs": layer_inputs}
+
+    @staticmethod
+    def _intervention_trace_due(update: int) -> bool:
+        """Every 5 updates for the first 200 (a 50-update scale collapse is ten points, not two), every 25 afterwards."""
+        return update % (5 if update < 200 else 25) == 0
+
+    def _head_grad_norms(self, ac_model: ActivationContextModel) -> dict:
+        """Per intervention layer, the norm of the head's gradient as it stands when called (before clipping), and the read-out
+        block's under "attn:<layer>", on the same footing."""
+        modules = ac_model.modules
+        norms = {layer: self._grad_norm(list(modules.delta_heads[str(layer)].parameters())) for layer in ac_model.intervention_layers}
+        norms.update({f"attn:{layer}": self._grad_norm(list(modules.passage_attention[str(layer)].parameters())) for layer in ac_model.intervention_layers})
+        norms.update({"attn:row": self._grad_norm(list(modules.row_readout.parameters())), "attn:recursive": self._grad_norm(list(modules.recursive_readout.parameters()))})
+        return norms
+
+    def _intervention_trace_entry(self, ac_model: ActivationContextModel, update: int, head_norms: dict | None = None) -> dict:
+        """Per layer: the head's gradient norm before clipping (`head_norms`, taken next to the encoder norm; when omitted, whatever the
+        gradients hold now), the residual RMS at the row positions on the training forwards since the last trace, `delta_rms`: the RMS
+        of the applied delta relative to that residual RMS (`delta_rms_abs` the absolute value), the delta / input-row cosine, and the
+        read-out block's output-projection weight norm (0 at init; rising = the attention path opens) with its gradient norm."""
+        modules = ac_model.modules
+        head_norms = self._head_grad_norms(ac_model) if head_norms is None else head_norms
+        entry = {"update": update, "head_grad_norm": {}, "row_rms": {}, "cosine": {}, "attention_out_norm": {}, "attention_grad_norm": {}, "delta_rms": {}, "delta_rms_abs": {}}
+        def mean_of(values) -> float | None:
+            return float(torch.stack([torch.as_tensor(value, dtype=torch.float32) for value in values]).mean()) if values else None
+        for layer in ac_model.intervention_layers:
+            entry["attention_out_norm"][layer] = float(modules.passage_attention[str(layer)].out.weight.detach().norm())
+            entry["attention_grad_norm"][layer] = head_norms[f"attn:{layer}"]
+            entry["head_grad_norm"][layer] = head_norms[layer]
+            entry["row_rms"][layer] = mean_of(self._row_rms.pop(layer, []))                 # 0-d tensors, converted here (one host sync per trace point, not per forward)
+            entry["cosine"][layer] = mean_of(ac_model.intervention_diagnostics["cosine"].pop(layer, []))
+            applied = mean_of(ac_model.intervention_diagnostics["delta_rms"].pop(layer, []))
+            entry["delta_rms_abs"][layer] = applied
+            entry["delta_rms"][layer] = applied / entry["row_rms"][layer] if applied is not None and entry["row_rms"][layer] else None
+        for name in ("row", "recursive"):                                                       # the read-outs in front of the row heads, on the same footing
+            entry["attention_out_norm"][name] = float(getattr(modules, f"{name}_readout").out.weight.detach().norm())
+            entry["attention_grad_norm"][name] = head_norms[f"attn:{name}"]
+        return entry
+
+    def _harvest_row_rms(self, payloads) -> None:
+        """After a training forward: the reader hooks wrote each payload's residual RMS at its row positions per layer."""
+        for payload in payloads:
+            for layer, value in (payload or {}).pop("row_rms", {}).items():
+                self._row_rms.setdefault(int(layer), []).append(value.detach())
+
+    def _chunked_sft_tokens(self, states: torch.Tensor, head: torch.nn.Module, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
+        """(per-token cross-entropy [C] with grad, detached per-token log-probs [C], detached top-1 hits [C])."""
+        chunk = getattr(self, "_logits_chunk_tokens", self.config.logits_chunk_tokens or 64)
+        def cross_entropy(hidden, labels):
+            logp = F.log_softmax(head_logits(hidden, head, self.config.fp32_head_matmul), dim=-1)
+            token_logp = logp.gather(-1, labels[:, None])[:, 0]
+            return -token_logp, token_logp.detach(), (logp.argmax(dim=-1) == labels).detach()
+        losses, logps, hits = [], [], []
+        for start in range(0, targets.shape[0], chunk):
+            arguments = states[start:start + chunk], targets[start:start + chunk]
+            token_loss, token_logp, chunk_hits = (checkpoint(cross_entropy, *arguments, use_reentrant=False)
+                                                  if torch.is_grad_enabled() and states.requires_grad else cross_entropy(*arguments))
+            losses.append(token_loss)
+            logps.append(token_logp)
+            hits.append(chunk_hits)
+        if not losses:
+            return states.new_zeros((0,), dtype=torch.float32), targets.new_zeros((0,), dtype=torch.float32), targets.new_zeros((0,), dtype=torch.bool)
+        return torch.cat(losses), torch.cat(logps), torch.cat(hits)
+
+    def _chunked_sft(self, states: torch.Tensor, head: torch.nn.Module, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
+        """(mean cross-entropy over the target tokens, detached per-token log-probs [C])."""
+        chunk = getattr(self, "_logits_chunk_tokens", self.config.logits_chunk_tokens or 64)
+        def cross_entropy(hidden, labels):
+            logp = F.log_softmax(head_logits(hidden, head, self.config.fp32_head_matmul), dim=-1)
+            token_logp = logp.gather(-1, labels[:, None])[:, 0]
+            return -token_logp.sum(), token_logp.detach(), (logp.argmax(dim=-1) == labels).detach()
+        total = states.new_zeros((), dtype=torch.float32)
+        logps, hits = [], []
+        for start in range(0, targets.shape[0], chunk):
+            arguments = states[start:start + chunk], targets[start:start + chunk]
+            ce_sum, token_logp, chunk_hits = (checkpoint(cross_entropy, *arguments, use_reentrant=False)
+                                              if torch.is_grad_enabled() and states.requires_grad else cross_entropy(*arguments))
+            total = total + ce_sum
+            logps.append(token_logp)
+            hits.append(chunk_hits)
+        self._last_hits = torch.cat(hits) if hits else targets.new_zeros((0,), dtype=torch.bool)
+        return total / max(1, targets.shape[0]), torch.cat(logps) if logps else targets.new_zeros((0,), dtype=torch.float32)
+
+    @torch.no_grad()
+    def _store_teacher_targets(self, ac_model: ActivationContextModel, examples: t.Sequence[_Example], device: torch.device,
+                               reporter: "ActivationContextTrainingReporter | None" = None, reporting_name: str = "teacher targets",
+                               lora: str | None = "reader") -> dict:
+        """
+        Store, for every example without an entry, the reader's current top-k targets and its log-prob of each target
+        token: the fixed teacher of the training call. Returns {"examples", "top_k", "coverage"}; coverage is the mean
+        probability mass the stored top-k carry over all stored positions.
+        """
+        target = ac_model.target
+        base = target.model
+        embedding = base.get_input_embeddings()
+        head = base.get_output_embeddings()
+        lora = ac_model.config.target_model_lora_name if lora == "reader" else lora   # None: the base with no adapter (the penalty's reference)
+        k = self.config.teacher_cache_top_k
+        started = time.monotonic()
+        pending = list({example.teacher_cache_key: example for example in examples if example.teacher_cache_key not in self.teacher_cache}.values())
+        if reporter is not None:
+            reporter.report_evaluation_progress(reporting_name, 0, len(pending))
+        groups: list[list[_Example]] = [[example] for example in pending]
+        if self.config.micro_batch_examples > 1:       # several teacher views per right-padded forward (no grad; pads never precede a real token)
+            order = sorted(pending, key=lambda example: len(example.teacher_ids))
+            groups = [order[start:start + self.config.micro_batch_examples] for start in range(0, len(order), self.config.micro_batch_examples)]
+        done = 0
+        for group in groups:
+            if len(group) == 1:
+                ids = torch.tensor(group[0].teacher_ids, device=device, dtype=torch.long)
+                states_by_example = [target.decoder_forward(embedding(ids)[None], None, lora_name=lora, gradient_checkpointing=False)[0][group[0].teacher_positions]]
+            else:
+                embeds = [embedding(torch.tensor(example.teacher_ids, device=device, dtype=torch.long)) for example in group]
+                hidden = self._padded_forward(target, embeds, lora)
+                states_by_example = [hidden[index, example.teacher_positions] for index, example in enumerate(group)]
+            for example, states in zip(group, states_by_example):
+                cached = self._teacher_top_k(states, head, k)
+                targets = torch.tensor(example.target_ids, device=device, dtype=torch.long)
+                target_logp = self._target_logp(states, head, targets)
+                cached = self._with_gold(cached, targets, target_logp)          # the gold token always among the stored top-k
+                self.teacher_cache[example.teacher_cache_key] = tuple(tensor.cpu() for tensor in cached)
+                self.teacher_target_logp[example.teacher_cache_key] = target_logp.cpu()
+            done += len(group)
+            if reporter is not None:
+                reporter.report_evaluation_progress(reporting_name, done, len(pending))
+        keys = list(dict.fromkeys(example.teacher_cache_key for example in examples))
+        covered = [1.0 - self.teacher_cache[key][2].float().exp() for key in keys if key in self.teacher_cache]
+        coverage = float(torch.cat(covered).mean()) if covered and sum(c.numel() for c in covered) else None
+        return {"examples": len(pending), "top_k": k, "coverage": coverage, "seconds": round(time.monotonic() - started, 3)}
+
+    @torch.no_grad()
+    def _target_logp(self, states: torch.Tensor, head: torch.nn.Module, targets: torch.Tensor) -> torch.Tensor:
+        """Log-prob of each target token [C] from the hidden states at the predicting positions."""
+        chunk, fp32 = getattr(self, "_logits_chunk_tokens", self.config.logits_chunk_tokens or 64), self.config.fp32_head_matmul
+        out = []
+        for start in range(0, states.shape[0], chunk):
+            logp = F.log_softmax(head_logits(states[start:start + chunk], head, fp32), dim=-1)
+            out.append(logp.gather(-1, targets[start:start + chunk, None])[:, 0].float())
+        return torch.cat(out) if out else states.new_zeros((0,), dtype=torch.float32)
+
+    def save_teacher_targets(self, path: str | Path) -> str:
+        """Write the stored teacher targets (top-k and target log-probs) so a later evaluation can use the same reference."""
+        path = Path(path)
+        path.parent.mkdir(parents=True, exist_ok=True)
+        torch.save({"top_k": self.config.teacher_cache_top_k, "targets": self.teacher_cache, "target_logp": self.teacher_target_logp}, path)
+        return str(path)
+
+    def load_teacher_targets(self, path: str | Path) -> int:
+        """Replace the stored teacher targets with a saved set; returns the number of entries. Fails on a different top-k."""
+        data = torch.load(Path(path), map_location="cpu")
+        if data["top_k"] != self.config.teacher_cache_top_k:
+            raise ValueError(f"saved targets use top_k={data['top_k']}, this trainer uses {self.config.teacher_cache_top_k}")
+        self.teacher_cache = dict(data["targets"])
+        self.teacher_target_logp = dict(data["target_logp"])
+        return len(self.teacher_cache)
+
+    @staticmethod
+    def _grad_norm(parameters) -> float:
+        grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
+        return float(torch.norm(torch.stack([grad.norm() for grad in grads]))) if grads else 0.0
+
+    @staticmethod
+    def _with_gold(cached: tuple[torch.Tensor, torch.Tensor, torch.Tensor], targets: torch.Tensor, target_logp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
+        """The stored top-k with the gold token of every position guaranteed present: where it is absent, it replaces the k-th (least likely) entry."""
+        ids, logps, rests = (tensor.clone() for tensor in cached)
+        present = (ids == targets[:, None].to(ids.dtype)).any(dim=-1)
+        missing = ~present
+        if missing.any():
+            ids[missing, -1] = targets[missing].to(ids.dtype)
+            logps[missing, -1] = target_logp[missing].to(logps.dtype)
+            rests[missing] = torch.log(torch.clamp(1.0 - logps[missing].float().exp().sum(dim=-1), min=1e-12)).to(rests.dtype)
+        return ids, logps, rests
 
     @torch.no_grad()
     def _teacher_top_k(self, teacher_states: torch.Tensor, head: torch.nn.Module, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
@@ -744,69 +1315,67 @@ class ActivationContextTrainer:
         return torch.cat(ids), torch.cat(logps), torch.cat(rests)
 
     def _chunked_kl(self, teacher_states: torch.Tensor | None, student_states: torch.Tensor, head: torch.nn.Module,
-                    cached: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, float]:
+                    cached: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
+                    target_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, float, torch.Tensor]:
         """
-        Mean KL(teacher || student) and top-1 agreement over the positions, the vocabulary logits one chunk at a
-        time. With `cached` (the teacher's top-k ids, log-probs and log remainder mass, see `_teacher_top_k`) the
-        teacher states are not needed: the KL is between the k+1-way coarsened distributions (the top-k tokens
-        and "everything else"), a lower bound of the exact KL that is tight when the top-k carry the mass.
+        Mean KL(teacher || student), top-1 agreement over the positions, and the student's detached log-prob of each
+        target token ([C], empty when no target_ids), the vocabulary logits one chunk at a time. With `cached` (the
+        teacher's top-k ids, log-probs and log remainder mass, see `_teacher_top_k`) the teacher states are not needed:
+        the KL is between the k+1-way coarsened distributions (the top-k tokens and "everything else"), a lower bound
+        of the exact KL that is tight when the top-k carry the mass. Live teacher states are only used by the drift term.
         """
         chunk = getattr(self, "_logits_chunk_tokens", self.config.logits_chunk_tokens or 64)
         fp32 = self.config.fp32_head_matmul
         total = torch.zeros((), device=student_states.device, dtype=torch.float32)
         agree = 0.0
         num = student_states.shape[0]
+        logps = []
+
+        hits = []
 
-        def chunk_kl(student_chunk: torch.Tensor, teacher_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
+        def token_logp(student_logp: torch.Tensor, targets: torch.Tensor | None) -> torch.Tensor:
+            if targets is None:
+                return student_logp.new_zeros((0,))
+            hits.append((student_logp.argmax(dim=-1) == targets).detach())
+            return student_logp.gather(-1, targets[:, None])[:, 0].detach().float()
+
+        def chunk_kl(student_chunk: torch.Tensor, teacher_chunk: torch.Tensor, targets: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
             with torch.no_grad():
                 teacher_logp = F.log_softmax(head_logits(teacher_chunk, head, fp32), dim=-1)
             student_logp = F.log_softmax(head_logits(student_chunk, head, fp32), dim=-1)
             kl_sum = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1).sum()
             agreement = (teacher_logp.argmax(dim=-1) == student_logp.argmax(dim=-1)).float().sum()
-            return kl_sum, agreement.detach()
+            return kl_sum, agreement.detach(), token_logp(student_logp, targets)
 
-        def chunk_kl_cached(student_chunk: torch.Tensor, ids: torch.Tensor, teacher_logp: torch.Tensor, teacher_rest: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
+        def chunk_kl_cached(student_chunk: torch.Tensor, ids: torch.Tensor, teacher_logp: torch.Tensor, teacher_rest: torch.Tensor,
+                            targets: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
             student_logp = F.log_softmax(head_logits(student_chunk, head, fp32), dim=-1)
             student_top = student_logp.gather(-1, ids.long())
             student_rest = torch.log(torch.clamp(1.0 - student_top.exp().sum(dim=-1), min=1e-12))
             kl_sum = ((teacher_logp.exp() * (teacher_logp - student_top)).sum(dim=-1) + teacher_rest.exp() * (teacher_rest - student_rest)).sum()
             agreement = (ids[:, 0].long() == student_logp.argmax(dim=-1)).float().sum()
-            return kl_sum, agreement.detach()
+            return kl_sum, agreement.detach(), token_logp(student_logp, targets)
 
         for start in range(0, num, chunk):
             student_chunk = student_states[start:start + chunk]
+            targets = target_ids[start:start + chunk] if target_ids is not None else None
             if cached is not None:
-                function, arguments = chunk_kl_cached, (student_chunk, *(tensor[start:start + chunk] for tensor in cached))
+                function, arguments = chunk_kl_cached, (student_chunk, *(tensor[start:start + chunk] for tensor in cached), targets)
             else:
-                function, arguments = chunk_kl, (student_chunk, teacher_states[start:start + chunk])
+                function, arguments = chunk_kl, (student_chunk, teacher_states[start:start + chunk], targets)
             if torch.is_grad_enabled() and student_chunk.requires_grad:
-                kl_sum, agreement = checkpoint(function, *arguments, use_reentrant=False)
+                kl_sum, agreement, chunk_logp = checkpoint(function, *arguments, use_reentrant=False)
             else:
-                kl_sum, agreement = function(*arguments)
+                kl_sum, agreement, chunk_logp = function(*arguments)
             total = total + kl_sum
             agree += float(agreement)
-        return total / max(1, num), agree / max(1, num)
+            logps.append(chunk_logp)
+        if target_ids is not None:
+            self._last_hits = torch.cat(hits[:len(logps)]) if hits else target_ids.new_zeros((0,), dtype=torch.bool)
+        return total / max(1, num), agree / max(1, num), (torch.cat(logps) if logps else student_states.new_zeros((0,), dtype=torch.float32))
 
     def _autotuning_variables(self, ac_model: ActivationContextModel) -> dict:
         variables = get_ac_training_autotuning_variables(ac_model.side.model_config, ac_model.target.model_config,
                         self.config, ac_config=ac_model.config, device=TARGET_DEVICE)
         self._logits_chunk_tokens = variables["logits_chunk_tokens"]
         return variables
-
-
-
-@dataclass
-class _CheckpointConfig:
-    gradient_checkpointing: bool
-    gradient_checkpointing_min_tokens: int | None
-
-
-def _flatten(messages: list[dict]) -> list[dict]:
-    out = []
-    for message in messages:
-        message = dict(message)
-        content = message.get("content")
-        if isinstance(content, list):
-            message["content"] = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
-        out.append(message)
-    return out
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/ac_model/ac_model_utils.py">activation/ac_model/ac_model_utils.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/ac_model/ac_model_utils.py`

```diff
diff --git a/activation/ac_model/ac_model_utils.py b/activation/ac_model/ac_model_utils.py
index 31f34f2..586c797 100644
--- a/activation/ac_model/ac_model_utils.py
+++ b/activation/ac_model/ac_model_utils.py
@@ -11,6 +11,8 @@ import queue
 import threading
 import time
 import typing as t
+from copy import deepcopy
+from dataclasses import dataclass, field
 from collections import OrderedDict
 from concurrent.futures import Future
 
@@ -62,7 +64,8 @@ def tokenize_with_parts(
     renders one sentinel string per part; the text is split at the sentinels and the pieces tokenized
     (no special tokens, as the template's own tokenization). The caller writes rows over the spans.
     """
-    flat = flatten_with_sentinels(messages, parts="sentinel")
+    flat = [_training_template_message(tokenizer, message, reasoning_as_text=True)
+            for message in flatten_with_sentinels(messages, parts="sentinel")]
     index = sum(1 for _ in direct_parts(messages))
     assert index == len(part_lengths), f"{index} parts in the messages, {len(part_lengths)} lengths given"
     if tools and not template_takes_tools(tokenizer):
@@ -74,6 +77,230 @@ def tokenize_with_parts(
     return encode_with_part_sentinels(tokenizer, text, list(part_lengths), pad_id)
 
 
+def _training_template_message(tokenizer, message: dict, *, reasoning_as_text: bool = False) -> dict:
+    """Keep separate reasoning and structured calls visible under the selected template."""
+    from ..agent.agent_utils import ModelDialect
+    dialect = ModelDialect.for_tokenizer(tokenizer)
+    message = dict(message)
+    if reasoning_as_text and message.get("role") == "assistant" and isinstance(message.get("content"), str):
+        # Agent records may carry inline reasoning. Escape its delimiters so templates cannot strip earlier thoughts.
+        message["content"] = message["content"].replace("<think>", "[Reasoning]").replace("</think>", "[/Reasoning]")
+    reasoning = message.pop("reasoning", None) or message.get("reasoning_content")
+    if reasoning:
+        if not reasoning_as_text and "reasoning" in (tokenizer.chat_template or ""):
+            message["reasoning_content"] = reasoning
+        else:
+            message.pop("reasoning_content", None)
+            message["content"] = str(reasoning) + "\n\n" + str(message.get("content") or "")
+    if not template_takes_tools(tokenizer) and message.get("tool_calls"):
+        calls = [dict(call.get("function", call), id=call.get("id", f"call_{index}"))
+                 for index, call in enumerate(message["tool_calls"])]
+        message = dialect.rendering.assistant_message(message.get("content") or "", calls)
+    return message
+
+
+def assistant_target_ids(tokenizer, message: dict, tools: list[dict] | None = None,
+                         chat_template_kwargs: dict | None = None) -> list[int]:
+    """Render a complete public assistant output once, excluding the supplied generation prompt."""
+    message = _training_template_message(tokenizer, message)
+    prefix = [{"role": "user", "content": ""}]
+    if tools and not template_takes_tools(tokenizer):
+        prefix, tools = tools_in_system_text(prefix, tools), None
+    kwargs = {"enable_thinking": bool(message.get("reasoning_content"))} if chat_template_kwargs is None else chat_template_kwargs
+    opened = tokenizer.apply_chat_template(prefix, tools=tools, tokenize=False, add_generation_prompt=True, **kwargs)
+    completed = tokenizer.apply_chat_template(prefix + [message], tools=tools, tokenize=False, add_generation_prompt=False, **kwargs)
+    if not completed.startswith(opened):
+        raise ValueError("Assistant output does not extend this tokenizer's generation prompt")
+    output = completed[len(opened):]
+    from ..harness.hf_utils import canonical_eot_token
+    eot = canonical_eot_token(tokenizer) or tokenizer.eos_token
+    if eot and eot in output and not output.rsplit(eot, 1)[1].strip():
+        output = output[:output.rfind(eot) + len(eot)]
+    ids = tokenizer.encode(output, add_special_tokens=False)
+    if not ids:
+        raise ValueError("Assistant output has no target tokens")
+    return list(ids)
+
+
+def cached_vocabulary_size(tokenizer) -> int:
+    """len(tokenizer) walks the added-token table every call (~30 ms on Qwen3.5; it was 96 % of example building): computed once per tokenizer object."""
+    size = tokenizer.__dict__.get("_activation_vocabulary_size")
+    if size is None:
+        size = tokenizer.__dict__["_activation_vocabulary_size"] = len(tokenizer)
+    return size
+
+
+def prepare_training_history(tokenizer, messages: list[dict], start: int, assistant_ids: list[list[int]],
+                             part_lengths: list[int], *, tools: list[dict] | None = None,
+                             chat_template_kwargs: dict | None = None) -> tuple[list[int], list[tuple[int, int]], list[int]]:
+    """Incremental runtime-shaped tokens, AC spans and predicting positions for one history."""
+    from ..agent.agent_utils import ModelDialect
+    if not isinstance(start, int) or not 0 <= start < len(messages):
+        raise ValueError("Training history start lies outside its messages")
+    selected = [index for index in range(start, len(messages)) if messages[index].get("role") == "assistant"]
+    if len(selected) != len(assistant_ids) or not selected:
+        raise ValueError("Every retained assistant message needs exactly one target-token sequence")
+    vocabulary_size = cached_vocabulary_size(tokenizer)
+    for ids in assistant_ids:
+        if not ids or any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids):
+            raise ValueError("Assistant targets must be nonempty valid tokenizer IDs")
+    if any(direct_parts([message]) for message in messages if message.get("role") == "assistant"):
+        raise ValueError("Assistant outputs cannot contain AC input parts")
+    dialect = ModelDialect.for_tokenizer(tokenizer)
+    chat_template_kwargs = {"enable_thinking": False} if chat_template_kwargs is None else chat_template_kwargs
+    messages = list(messages)
+    if tools and not template_takes_tools(tokenizer):
+        previous_count = len(messages)
+        messages, tools = tools_in_system_text(messages, tools), None
+        start += len(messages) - previous_count
+    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
+    cursor = 0
+    def lengths(block):
+        nonlocal cursor
+        count = len(direct_parts(block))
+        values = part_lengths[cursor:cursor + count]
+        cursor += count
+        if len(values) != count:
+            raise ValueError("AC part lengths do not match the message history")
+        return values
+    first = next(index for index in range(start, len(messages)) if messages[index].get("role") == "assistant")
+    if first == 0:
+        raise ValueError("An assistant target needs preceding prompt context")
+    prefix = [_training_template_message(tokenizer, message, reasoning_as_text=True) for message in messages[:first]]
+    ids, spans = dialect.prompt_tokens(tokenizer, prefix, tools, chat_template_kwargs, lengths(messages[:first]), pad)
+    if not ids:
+        raise ValueError("An assistant target needs a nonempty token prefix")
+    positions = []
+    selected_ids = iter(assistant_ids)
+    index = first
+    while index < len(messages):
+        message = messages[index]
+        if message.get("role") != "assistant":
+            raise ValueError("Expected an assistant output after a generation prompt")
+        output = list(next(selected_ids))
+        positions.extend(range(len(ids) - 1, len(ids) + len(output) - 1))
+        ids.extend(output)
+        end = index + 1
+        while end < len(messages) and messages[end].get("role") != "assistant":
+            end += 1
+        if end > index + 1 or end < len(messages):
+            following = messages[index + 1:end]
+            wrapper, local_spans = dialect.continuation_tokens(tokenizer, messages[:index + 1], following,
+                                tools, chat_template_kwargs, lengths(following), pad)
+            joined = dialect.join_continuation(output, wrapper)
+            shift = len(wrapper) - len(joined)
+            spans.extend((len(ids) + a - shift, len(ids) + b - shift) for a, b in local_spans)
+            ids.extend(joined)
+        index = end
+    if cursor != len(part_lengths):
+        raise ValueError("Unused AC part lengths in the message history")
+    return ids, spans, positions
+
+
+def validate_paired_histories(teacher: list[dict], teacher_start: int, student: list[dict], student_start: int) -> None:
+    """Suffix roles/calls agree; tool evidence can differ, assistant outputs cannot."""
+    if not 0 <= teacher_start < len(teacher) or not 0 <= student_start < len(student):
+        raise ValueError("Paired history starts lie outside their messages")
+    left, right = teacher[teacher_start:], student[student_start:]
+    if len(left) != len(right):
+        raise ValueError("Paired histories have different retained message counts")
+    for a, b in zip(left, right):
+        if a.get("role") != b.get("role"):
+            raise ValueError("Paired histories have different retained role order")
+        for key in ("tool_calls", "tool_call_id", "name"):
+            if a.get(key) != b.get(key):
+                raise ValueError("Paired histories have different retained call identity/order")
+        if a.get("role") == "assistant" and any(a.get(key) != b.get(key) for key in ("content", "reasoning", "reasoning_content")):
+            raise ValueError("Paired histories have different assistant outputs")
+
+
+def validate_part_capacity(ac_model, request, capacity: int) -> None:
+    """Check every recursive side forward: content tokens plus summary rows."""
+    children = [ac_model._child_request(part, request.compression_ratio) for part in direct_parts(request.messages)]
+    lengths = [ac_model.part_view_rows(child.messages, child.compression_ratio, child.tools) for child in children]
+    tokenizer = ac_model.side.tokenizer
+    ids, _ = tokenize_with_parts(tokenizer, request.messages, lengths, tokenizer.pad_token_id or 0, tools=request.tools)
+    total = len(ids) + ac_model.num_view_rows(len(ids), request.compression_ratio)      # the content tokens plus the summary rows
+    if total > capacity:
+        raise ValueError(f"AC side forward needs {total} positions, exceeding supported capacity {capacity}")
+    for child in children:
+        validate_part_capacity(ac_model, child, capacity)
+
+
+def normalize_public_assistants(tokenizer, messages: list[dict], tools: list[dict] | None,
+                                max_tokens: int) -> tuple[list[dict], list[dict] | None]:
+    """Split only public text/reasoning into complete, budgeted Python continuation turns."""
+    from ..dataset.loaders.trajectory_utils import map_definition, TOOL_MAP
+    if max_tokens < 1:
+        raise ValueError("max_assistant_tokens must be positive")
+    messages, tools = deepcopy(messages), deepcopy(tools or [])
+    used = {call.get("id") for message in messages for call in message.get("tool_calls") or []}
+    output, counter = [], 0
+    for message in messages:
+        if message.get("role") != "assistant":
+            output.append(message)
+            continue
+        while len(assistant_target_ids(tokenizer, message, tools, {"enable_thinking": True})) > max_tokens:
+            field = next((key for key in ("reasoning", "reasoning_content", "content")
+                          if isinstance(message.get(key), str) and message[key]), None)
+            if field is None:
+                raise ValueError("Public assistant has an indivisible call exceeding its token budget")
+            counter += 1
+            while f"ac_continue_{counter}" in used:
+                counter += 1
+            call_id = f"ac_continue_{counter}"
+            used.add(call_id)
+            if not any(tool.get("function", tool).get("name") == "python" for tool in tools):
+                tools.append(map_definition({"name": "python"}, TOOL_MAP))
+            call = {"id": call_id, "type": "function", "function": {
+                "name": "python", "arguments": {"code": f"print('continue {counter}')"}}}
+            def turn(text):
+                return {"role": "assistant", "content": "", field: text, "tool_calls": [call]}
+            text = message[field]
+            low, high, best = 1, len(text), 0
+            while low <= high:
+                middle = (low + high) // 2
+                if len(assistant_target_ids(tokenizer, turn(text[:middle]), tools, {"enable_thinking": True})) <= max_tokens:
+                    best, low = middle, middle + 1
+                else:
+                    high = middle - 1
+            if not best:
+                raise ValueError("Public continuation call cannot fit the assistant token budget")
+            output.extend([turn(text[:best]), {"role": "tool", "name": "python", "tool_call_id": call_id,
+                                               "content": f"continue {counter}\n"}])
+            message[field] = text[best:]
+        output.append(message)
+    return output, tools or None
+
+
+def complete_public_turns(messages: list[dict]) -> list[list[dict]]:
+    """Assistant outputs with their complete replies and intervening input, without dangling calls."""
+    groups = []
+    index = 0
+    while index < len(messages):
+        if messages[index].get("role") != "assistant":
+            raise ValueError("Public continuation must begin with an assistant")
+        end = index + 1
+        while end < len(messages) and messages[end].get("role") != "assistant":
+            end += 1
+        group = messages[index:end]
+        calls = group[0].get("tool_calls") or []
+        replies = [message for message in group[1:] if message.get("role") == "tool"]
+        terminal_answer = (end == len(messages) and len(group) == 1 and len(calls) == 1
+                           and calls[0].get("function", calls[0]).get("name") == "submit_answer")
+        if len(calls) != len(replies) and not terminal_answer:
+            raise ValueError("Public trajectory has unmatched tool calls or replies")
+        for call, reply in zip(calls, replies):
+            if reply.get("tool_call_id") and reply["tool_call_id"] != call.get("id"):
+                raise ValueError("Public tool reply does not match its call")
+        if end == len(messages):
+            while len(group) > 1 and group[-1].get("role") in ("user", "system"):
+                group = group[:-1]
+        groups.append(group)
+        index = end
+    return groups
+
+
 def sinusoidal_positions(length: int, d: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
     """[length, d] fixed sinusoidal positions (the mixer is bidirectional and has no positions of its own)."""
     position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
@@ -203,7 +430,7 @@ def unit_rows(rows: torch.Tensor) -> torch.Tensor:
 
 
 class AdaptiveSummaryPooling(nn.Module):
-    """V content-derived mean queries, each attending its ordered adaptive bin; O(N + V) source rows."""
+    """V content-derived mean queries, each attending its ordered bin of the content (token-count bins); O(N + V) source rows."""
 
     def __init__(self, d: int, num_heads: int) -> None:
         super().__init__()
@@ -227,8 +454,7 @@ class AdaptiveSummaryPooling(nn.Module):
         weights = valid[..., None].to(rows.dtype)
         means = (bins * weights).sum(dim=1) / weights.sum(dim=1)
         normed = self.norm(bins)
-        update, _ = self.attention(self.norm(means)[:, None], normed, normed,
-                                   key_padding_mask=~valid, need_weights=False)
+        update, _ = self.attention(self.norm(means)[:, None], normed, normed, key_padding_mask=~valid, need_weights=False)
         summaries = means + self.residual_scale * update[:, 0]
         return summaries + self.residual_scale * self.ffn(self.norm(summaries))
 
@@ -236,30 +462,145 @@ class AdaptiveSummaryPooling(nn.Module):
 class RowHead(nn.Module):
     """FFN x4 into the destination width, each row RMS-normalized and scaled by a learned scalar (initialized to the destination embedding RMS)."""
 
-    def __init__(self, d_in: int, d_out: int, initial_scale: float = 1.0) -> None:
+    def __init__(self, d_in: int, d_out: int, initial_scale: float = 1.0, skip: bool = False) -> None:
         super().__init__()
         self.ffn = FeedForward(d_in, 4 * d_in, d_out)
         self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
+        self.skip = skip
+        if skip:
+            if d_in != d_out:
+                raise ValueError("RowHead skip needs equal widths")
+            nn.init.zeros_(self.ffn.down.weight); nn.init.zeros_(self.ffn.down.bias)   # identity at init: the row is the normalized input
 
     def forward(self, rows: torch.Tensor) -> torch.Tensor:
         out = self.ffn(rows)
+        if self.skip:
+            out = out + rows
         rms = out.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
         return out / rms * self.scale
 
 
+@dataclass
+class ACOutput:
+    """What the encoder emits for one part: the rows in the reader's input space plus, for a deep model, one residual delta per
+    intervention layer ([V, d_target], `layer_inputs`) that the reader adds to that layer's input at the row positions. Recursive
+    results carry an empty map and their pre-adapter final states (`states`)."""
+    input_embeds: torch.Tensor
+    layer_inputs: dict[int, torch.Tensor] = field(default_factory=dict)
+    states: torch.Tensor | None = None       # a recursive result's final V side states before the recursive adapter ([V, d_side]): the
+                                             # parent's read-outs attend them as extra keys when `recursive_keys` is on
+
+    def to(self, device: torch.device | str, dtype: torch.dtype) -> "ACOutput":
+        move = lambda tensor: tensor.to(device=device, dtype=dtype)
+        return ACOutput(move(self.input_embeds), {int(layer): move(rows) for layer, rows in self.layer_inputs.items()},
+                        None if self.states is None else move(self.states))
+
+    def detach_cpu(self, dtype: torch.dtype = torch.bfloat16) -> "ACOutput":
+        cpu = lambda tensor: tensor.detach().to(device="cpu", dtype=dtype).contiguous()
+        return ACOutput(cpu(self.input_embeds), {int(layer): cpu(rows) for layer, rows in self.layer_inputs.items()},
+                        None if self.states is None else cpu(self.states))
+
+    @property
+    def has_interventions(self) -> bool:
+        return bool(self.layer_inputs)
+
+    @property
+    def tensors(self) -> list[torch.Tensor]:
+        return [self.input_embeds, *self.layer_inputs.values(), *([self.states] if self.states is not None else [])]
+
+    @property
+    def nbytes(self) -> int:
+        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors)
+
+
+class DeltaHead(nn.Module):
+    """Bottleneck FFN (d_in -> r -> d_out, SiLU) into a reader layer's residual stream at the row positions. `down` is zero-initialized
+    (weight and bias), so the delta is exactly 0 at init - a deep model starts bit-identical to the input-only model - and the head owns
+    its magnitude from there (no normalization, no learned scale)."""
+
+    def __init__(self, d_in: int, d_out: int, r: int) -> None:
+        super().__init__()
+        self.up = nn.Linear(d_in, r)
+        self.down = nn.Linear(r, d_out)
+        nn.init.zeros_(self.down.weight); nn.init.zeros_(self.down.bias)
+
+    def forward(self, rows: torch.Tensor) -> torch.Tensor:
+        return self.down(F.silu(self.up(rows.float())))
+
+
+class PassageAttention(nn.Module):
+    """The read-out in front of every delta head: each memory row's final side state queries the passage tokens' final side states
+    through bottleneck projections (d -> r; `heads` attention heads of r / heads dims) and adds the projected result back:
+    `rows + out(attn)`. `out` is zero-initialized, so at init the block is the identity on the rows (the head behind it reads the
+    row's own state) and the attention path opens only as `out` leaves zero. One shared LayerNorm normalizes both inputs."""
+
+    def __init__(self, d: int, r: int, heads: int | None = None) -> None:
+        super().__init__()
+        self.heads = heads if heads is not None else (4 if r % 4 == 0 and r >= 64 else 1)
+        if r % self.heads:
+            raise ValueError("the bottleneck width must be a multiple of the head count")
+        self.norm = nn.LayerNorm(d)
+        self.query = nn.Linear(d, r)
+        self.key = nn.Linear(d, r)
+        self.value = nn.Linear(d, r)
+        self.out = nn.Linear(r, d)
+        nn.init.zeros_(self.out.weight)
+        nn.init.zeros_(self.out.bias)
+
+    def forward(self, rows: torch.Tensor, passage: torch.Tensor) -> torch.Tensor:
+        """rows [M, d] (queries), passage [N, d] (keys and values) -> [M, d]."""
+        rows, passage = rows.float(), passage.float()
+        if passage.shape[0] == 0:
+            return rows
+        normed = self.norm(passage)
+        split = lambda tensor, n: tensor.view(n, self.heads, -1).transpose(0, 1)                   # [heads, n, r / heads]
+        q, k, v = split(self.query(self.norm(rows)), rows.shape[0]), split(self.key(normed), passage.shape[0]), split(self.value(normed), passage.shape[0])
+        mixed = F.scaled_dot_product_attention(q, k, v).transpose(0, 1).reshape(rows.shape[0], -1)
+        return rows + self.out(mixed)
+
+
+def resolve_intervention_layers(num_layers: int, frequency: int, add_last: bool, full_attention_layers: t.Iterable[int] | None = None) -> list[int]:
+    """
+    Reader layers whose input receives a delta: `frequency` interior points floor(L*i/(frequency+1)), each snapped to a
+    full-attention block within two indices (the earlier one on a tie), plus L-1 when asked; 0 excluded, sorted,
+    deduplicated. Frequency 0 disables everything, the last-layer flag included. Frequency -1 = every full-attention
+    layer of the reader (0 excluded; the last-layer flag and the snapping are moot).
+    """
+    if frequency == 0:
+        return []                              # zero mode is unconditional, whatever the target description says
+    if frequency < -1 or num_layers < 1:
+        raise ValueError("intervention_frequency must be -1 (every full-attention layer), 0 or positive, and the reader must have layers")
+    full = sorted(set(int(index) for index in (full_attention_layers or [])))
+    if frequency == -1:
+        if not full:
+            raise ValueError("intervention_frequency -1 selects every full-attention layer, and the reader describes none")
+        return [layer for layer in full if 0 < layer < num_layers]
+    chosen = []
+    for i in range(1, frequency + 1):
+        layer = (num_layers * i) // (frequency + 1)
+        if full:
+            candidates = [index for index in full if abs(index - layer) <= 2]
+            if candidates:
+                layer = min(candidates, key=lambda index: (abs(index - layer), index))
+        chosen.append(layer)
+    if add_last:
+        chosen.append(num_layers - 1)
+    return sorted({layer for layer in chosen if 0 < layer < num_layers})
+
+
 # --------------------------------------------------------------------------------------- cache and queue
 class RowCache:
     """CPU cache of encoded rows keyed by row_cache_key, LRU by bytes."""
 
     def __init__(self, max_bytes: int) -> None:
         self.max_bytes = int(max_bytes)
-        self.entries: "OrderedDict[str, torch.Tensor]" = OrderedDict()
+        self.entries: "OrderedDict[str, ACOutput]" = OrderedDict()
         self.total_bytes = 0
         self.hits = 0
         self.misses = 0
         self.lock = threading.Lock()
 
-    def get(self, key: str) -> torch.Tensor | None:
+    def get(self, key: str) -> ACOutput | None:
         with self.lock:
             rows = self.entries.get(key)
             if rows is None:
@@ -269,20 +610,21 @@ class RowCache:
             self.hits += 1
             return rows
 
-    def put(self, key: str, rows: torch.Tensor) -> None:
+    def put(self, key: str, rows: torch.Tensor | ACOutput) -> None:
+        """Every tensor of the output (rows and deltas) is kept in bf16 on the CPU and counted."""
         if self.max_bytes <= 0:
             return
-        rows = rows.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
-        size = rows.numel() * rows.element_size()
+        output = (rows if isinstance(rows, ACOutput) else ACOutput(rows)).detach_cpu()
+        size = output.nbytes
         with self.lock:
             if key in self.entries:
-                self.total_bytes -= self.entries[key].numel() * self.entries[key].element_size()
-            self.entries[key] = rows
+                self.total_bytes -= self.entries[key].nbytes
+            self.entries[key] = output
             self.entries.move_to_end(key)
             self.total_bytes += size
             while self.total_bytes > self.max_bytes and self.entries:
                 _, evicted = self.entries.popitem(last=False)
-                self.total_bytes -= evicted.numel() * evicted.element_size()
+                self.total_bytes -= evicted.nbytes
 
     def clear(self) -> None:
         with self.lock:
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/agent/agent.py">activation/agent/agent.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/agent/agent.py`

```diff
diff --git a/activation/agent/agent.py b/activation/agent/agent.py
index df81869..108cd7d 100644
--- a/activation/agent/agent.py
+++ b/activation/agent/agent.py
@@ -27,7 +27,6 @@ import traceback
 import typing as t
 import uuid
 from copy import deepcopy
-from dataclasses import asdict
 
 import torch
 
@@ -37,7 +36,8 @@ from .agent_config import AgentConfig, AgentRunResult, TrajectoryStep
 from .agent_env import AgentEnv
 from .agent_tools import (DEFAULT_TOOLS, AgentTool, CompactionTool, SubagentTool, ToolCallResult, injected_content, segment_messages_with_parts,
                           tool_content, truncate_output, subagent_config_of)
-from .agent_utils import PARSE_ERROR_TOOL, ModelDialect, parameter_types_of
+from .agent_utils import (PARSE_ERROR_TOOL, ModelDialect, parameter_types_of, capture_ac_spans,
+                          trajectory_step_dict, load_ac_rows)
 
 MAX_CONSECUTIVE_NO_TOOL_TURNS = 3   # nudged after each; the run ends with no_tool_call at the third in a row
 MAX_COMPACTION_REFUSALS = 5         # turns that ignore a due compaction before the run ends with compaction_refused
@@ -275,7 +275,7 @@ class Agent:
         self.segment_start = len(ids)
         self.last_sampled = []
         results.prompt_token_ids, results.prompt_messages = list(ids), deepcopy(messages)
-        results.prompt_ac_spans = [{"start": start, "length": end - start} for start, end in spans]
+        results.prompt_ac_spans = capture_ac_spans(spans, rows)
         self._after_append()
 
     def compact(self, summary: str) -> None:
@@ -312,7 +312,7 @@ class Agent:
         self.messages.append(message)
         self.prefix += list(token_ids)
         self.last_sampled = list(token_ids)
-        self.run_results.trajectory.append(asdict(step))
+        self.run_results.trajectory.append(trajectory_step_dict(step))
 
     def append_user_message(self, text: str) -> None:
         message = {"role": "user", "content": text}
@@ -342,12 +342,12 @@ class Agent:
     def _append_step(self, step: TrajectoryStep, messages: list[dict], spans: list[tuple[int, int]], rows: list[torch.Tensor]) -> None:
         offset = len(self.prefix)
         assert [int(r.shape[0]) for r in rows] == [end - start for start, end in spans], "placeholder runs differ from the encoded rows"
-        step.ac_spans = [{"start": start, "length": end - start} for start, end in spans]
+        step.ac_spans = capture_ac_spans(spans, rows)
         self.messages.extend(messages)
         self.prefix += step.token_ids
         self.spans += [(offset + start, offset + end) for start, end in spans]
         self.rows += rows
-        self.run_results.trajectory.append(asdict(step))
+        self.run_results.trajectory.append(trajectory_step_dict(step))
         self._after_append()
 
     def _after_append(self) -> None:
@@ -641,31 +641,27 @@ class Agent:
     def resume_from_run_result(harness: "HarnessRuntime", run_result: AgentRunResult, reporter: "RolloutReporter | None" = None) -> "Agent":
         """
         An agent positioned exactly after the recorded steps of `run_result`'s current segment: messages,
-        prefix and spans from the record, rows re-encoded from the recorded parts (the record never
-        stores rows). Counters continue from the record.
+        prefix, spans and exact captured rows from the record. Counters continue from the record.
         """
         agent = Agent(harness, run_result.agent_config, reporter=reporter, seed=run_result.seed)
         agent.run_results = run_result
         agent.dialect = ModelDialect.for_tokenizer(agent.loaded_model.tokenizer)
         agent._prepare_tools()
-        if agent.ac_model is not None and run_result.ac_model_version is not None and agent.ac_model.version != run_result.ac_model_version:
-            print(f"Agent {agent.agent_id[:8]} - resuming rows with AC version {agent.ac_model.version}, recorded {run_result.ac_model_version}", flush=True)
         agent.messages = deepcopy(run_result.prompt_messages)
         agent.prefix = list(run_result.prompt_token_ids)
         agent.segment_start = len(agent.prefix)
         spans = [(span["start"], span["start"] + span["length"]) for span in run_result.prompt_ac_spans]
-        parts_messages: list[dict] = list(run_result.prompt_messages)
+        rows = [load_ac_rows(span) for span in run_result.prompt_ac_spans]
         for step in run_result.trajectory:
             offset = len(agent.prefix)
             agent.messages.extend(deepcopy(step["messages"]))
             agent.prefix += list(step["token_ids"])
             spans += [(offset + span["start"], offset + span["start"] + span["length"]) for span in step["ac_spans"]]
-            parts_messages += step["messages"]
+            rows += [load_ac_rows(span) for span in step.get("ac_spans", [])]
             if step["role"] == "assistant":
                 agent.last_sampled = list(step["token_ids"])
         agent.spans = spans
-        agent.rows = agent._encode_parts(parts_messages)
-        assert [int(r.shape[0]) for r in agent.rows] == [end - start for start, end in spans], "recorded spans differ from the re-encoded rows"
+        agent.rows = rows
         agent.started = True
         agent._after_append()
         return agent
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/agent/agent_config.py">activation/agent/agent_config.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/agent/agent_config.py`

```diff
diff --git a/activation/agent/agent_config.py b/activation/agent/agent_config.py
index 8c4d07a..ddebe4d 100644
--- a/activation/agent/agent_config.py
+++ b/activation/agent/agent_config.py
@@ -7,6 +7,7 @@ from __future__ import annotations
 import json
 import typing as t
 from dataclasses import dataclass, field, replace
+from pathlib import Path
 
 from activation.common.utils import class_spec_name, resolve_class_name
 
@@ -129,7 +130,7 @@ class TrajectoryStep:
                                                                # (recorded whether or not it was rendered), subagent_index. Empty on old records.
     messages: list[dict] = field(default_factory=list)         # the dialect messages this step appended, activation_context parts inline and in
                                                                # order: one assistant message, the tool messages, or the nudge
-    ac_spans: list[dict] = field(default_factory=list)         # [{"start", "length"}] placeholder runs inside token_ids, one per part of `messages`
+    ac_spans: list[dict] = field(default_factory=list)         # {start, length, rows (CPU) | row_path}, relative to this step's token_ids
     # Token segment (slice 2): the prompt the model read is AgentRunResult.prompt_token_ids + every step's token_ids in order.
     token_ids: list[int] = field(default_factory=list)         # assistant: the sampled tokens verbatim; tool/user: the template's wrapper up to the next generation prompt
     logprobs: list[float] = field(default_factory=list)        # assistant only: the engine's log-prob of each sampled token (pi_old); empty when not recorded
@@ -161,7 +162,7 @@ class AgentRunResult:
     seed: int = 0
     prompt_token_ids: list[int] = field(default_factory=list)  # the segment's first prompt (system + first messages + tools, templated once)
     prompt_messages: list[dict] = field(default_factory=list)  # the dialect messages of that prompt, parts inline (a compaction tree lives here)
-    prompt_ac_spans: list[dict] = field(default_factory=list)  # [{"start", "length"}] placeholder runs inside prompt_token_ids
+    prompt_ac_spans: list[dict] = field(default_factory=list)  # {start, length, rows (CPU) | row_path}, relative to prompt_token_ids
     ac_model_name: str | None = None                           # the encoder that produced the rows, and its version at run time
     ac_model_version: int | None = None
     source: str = "policy"                                     # "policy" | "hinted:<model>" | "oracle:<model>": who produced this run
@@ -201,20 +202,18 @@ class AgentRunResult:
                 total[key] = total.get(key, 0) + value
         return total
 
-    def serialize(self) -> dict:
-        data = {key: value for key, value in self.__dict__.items()
-                if key not in ("agent_config", "subagent_results", "compactions", "trajectory", "answer")}
+    def serialize(self, *, base_dir: Path | None = None) -> dict:
+        from .agent_utils import serialize_ac_rows
+        data = dict(self.__dict__)
         data["agent_config"] = self.agent_config.serialize()
-        data["answer"] = _jsonable(self.answer)
-        data["trajectory"] = _jsonable(self.trajectory)
-        data["subagent_results"] = [result.serialize() for result in self.subagent_results]
-        data["compactions"] = [result.serialize() for result in self.compactions]
-        return data
+        return _jsonable(serialize_ac_rows(data, base_dir=base_dir))
 
     @staticmethod
     def deserialize(data: dict, harness: "HarnessRuntime | None" = None, agent_config: AgentConfig | None = None,
-                    strict: bool = False) -> "AgentRunResult":
+                    strict: bool = False, *, base_dir: Path | None = None) -> "AgentRunResult":
         """With agent_config given (the caller's live config), the stored config is not rebuilt. `strict`: see AgentConfig.deserialize."""
+        from .agent_utils import deserialize_ac_rows
+        data = deserialize_ac_rows(data, base_dir=base_dir)
         fields = {field_.name for field_ in AgentRunResult.__dataclass_fields__.values()}
         data = {key: value for key, value in data.items() if key in fields}          # cache rows carry extra keys
         config_data = data.pop("agent_config")
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/agent/agent_programs.py">activation/agent/agent_programs.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/agent/agent_programs.py`

```diff
diff --git a/activation/agent/agent_programs.py b/activation/agent/agent_programs.py
deleted file mode 100644
index 545e1b8..0000000
--- a/activation/agent/agent_programs.py
+++ /dev/null
@@ -1,37 +0,0 @@
-"""
-(Not in the main code for now).
-Currently, we can only elicit certain behaviors through the prompt, and trust the agent, even in a 2B-parameter model, to respect it.
-I think that's unrealistic at this small scale of models.
-Ultimately, if we are doing harness-model co-design, this is the kind of programmatic control harness designers should have anyway.
-
-Agent Programs are the remedy. These are harness-written logic, easily extensible, to programmatically perform a given workflow.
-
-Here is how they operate:
-- Takes in the initialized but unstarted agent.
-- Execute their program.
-    - Can ultimately call agent.augment_context(messages={text, activation, etc}):
-        - Also a convenient run_result = agent.run_subagent(...) which appends to run results to save the program from boilerplate.
-        - Also a convenient agent.set_final_answer(...). If programmatically set, no need for a run.
-- There can be many such programs.
-- Those get appended to the initial message sent to the agent.
-
-So pure parallel solving + synthesis becomes:
-- Run N solvers programmatically.
-- Augment context. Ask the main agent to just synthesize.
-
-Now there is a problem (just like our tools I guess) around serializability. Pick any common sense/convenient approach.
-But double-check the following:
-- The harness designers can make adhoc tools and programs, and run result serde won't prevent harvesting.
-- As in, the env/tool/program class can live in the test script that run the agents; not necessarily in the main codebase?
-- If not, that's probably also fine: it just means the main code base has to have well-known extension points to avoid task-specific pollution.
-
-(Unrelated issue: I think score should be float|None = None).
-"""
-
-# @AI: AgentConfig gains agent_programs: dict[str, list[tuple[AgentProgram, dict]]].
-class AgentProgram:
-    def __init__(self, agent: 'Agent', otherargs):
-        pass
-
-    def execute(self):
-        pass
\ No newline at end of file
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/agent/agent_utils.py">activation/agent/agent_utils.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/agent/agent_utils.py`

```diff
diff --git a/activation/agent/agent_utils.py b/activation/agent/agent_utils.py
index 56098a4..5fe57a6 100644
--- a/activation/agent/agent_utils.py
+++ b/activation/agent/agent_utils.py
@@ -21,20 +21,175 @@ generation-prompt form the model actually read, which is what training must see.
 from __future__ import annotations
 
 import json
+import hashlib
+import os
 import re
+import tempfile
 import typing as t
 import uuid
-from dataclasses import dataclass, field
+from dataclasses import dataclass, field, fields
+from pathlib import Path
+
+import torch
 
 from activation.common.ac_parts import encode_with_part_sentinels, flatten_with_sentinels
 
 if t.TYPE_CHECKING:
     from .agent import Agent
     from .agent_config import AgentConfig
+    from .agent_config import AgentRunResult, TrajectoryStep
     from .agent_tools import ToolCallResult
     from activation.harness import HarnessRuntime
 
 PARSE_ERROR_TOOL = "parse_error"
+
+
+def capture_ac_spans(spans: list[tuple[int, int]], rows: list[torch.Tensor]) -> list[dict]:
+    """Keep the exact final-cast observation; already contiguous CPU rows are not copied."""
+    if len(spans) != len(rows):
+        raise ValueError("AC spans and captured rows differ in count")
+    captured = []
+    for (start, end), tensor in zip(spans, rows):
+        span = {"start": start, "length": end - start, "rows": tensor.detach().cpu().contiguous()}
+        load_ac_rows(span)
+        captured.append(span)
+    return captured
+
+
+def trajectory_step_dict(step: "TrajectoryStep") -> dict:
+    """Convert the existing record without asdict's recursive tensor deepcopy."""
+    return {entry.name: getattr(step, entry.name) for entry in fields(step)}
+
+
+def _map_ac_records(data, convert) -> dict:
+    """Visit only owned row fields; raw parts and arbitrary tool payloads are untouched."""
+    record = dict(data if isinstance(data, dict) else vars(data))
+    config = record.get("agent_config")
+    if config is not None and not isinstance(config, dict):
+        record["agent_config"] = config.serialize()
+    if "prompt_ac_spans" in record:
+        record["prompt_ac_spans"] = [convert(span) for span in record["prompt_ac_spans"]]
+    if "trajectory" in record:
+        record["trajectory"] = [dict(step, ac_spans=[convert(span) for span in step["ac_spans"]])
+                                if "ac_spans" in step else dict(step) for step in record["trajectory"]]
+    for key in ("compactions", "subagent_results"):
+        if key in record:
+            record[key] = [_map_ac_records(child, convert) for child in record[key]]
+    return record
+
+
+def _ac_rows_digest(rows: torch.Tensor) -> str:
+    digest = hashlib.sha256(f"{rows.dtype}:{tuple(rows.shape)}:".encode())
+    digest.update(rows.contiguous().view(torch.uint8).numpy().tobytes())
+    return digest.hexdigest()
+
+
+def load_ac_rows(span: dict) -> torch.Tensor:
+    """Read captured rows or a resolved tensor file, preserving the recorded CPU dtype."""
+    rows = span.get("rows")
+    path = None
+    if rows is None:
+        if not span.get("row_path"):
+            raise ValueError("AC span has no captured rows; regenerate the rollout")
+        path = Path(span["row_path"])
+        if not path.is_absolute():
+            raise ValueError("AC row path is unresolved; deserialize with the JSONL parent as base_dir")
+        try:
+            rows = torch.load(path, map_location="cpu", weights_only=True)
+        except FileNotFoundError:
+            raise
+        except Exception as exc:
+            raise ValueError(f"Cannot read AC rows from {path}: {exc}") from exc
+    if not isinstance(rows, torch.Tensor) or rows.ndim != 2 or not rows.is_floating_point():
+        raise ValueError("AC rows must be a floating-point matrix")
+    if rows.device.type != "cpu" or rows.requires_grad or not rows.is_contiguous():
+        raise ValueError("Captured AC rows must be detached contiguous CPU tensors")
+    if rows.shape[0] != span.get("length") or rows.shape[0] < 1 or rows.shape[1] < 1:
+        raise ValueError("AC row shape does not match its span")
+    if path is not None and len(path.stem) == 64 and _ac_rows_digest(rows) != path.stem:
+        raise ValueError(f"AC row content hash mismatch: {path}")
+    return rows
+
+
+def serialize_ac_rows(data: dict, *, base_dir: Path | None = None) -> dict:
+    """Publish folder-local tensor files and return relative references; retain input buffers."""
+    destination = None if base_dir is None else Path(base_dir).resolve()
+
+    def save(span: dict) -> dict:
+        if destination is None:
+            raise ValueError("Serializing AC rows requires the JSONL parent as base_dir")
+        source = dict(span)
+        if source.get("row_path") and not Path(source["row_path"]).is_absolute():
+            source["row_path"] = str(destination / source["row_path"])
+        rows = load_ac_rows(source)
+        digest = _ac_rows_digest(rows)
+        folder = destination / "saved_tensors"
+        folder.mkdir(parents=True, exist_ok=True)
+        target = folder / f"{digest}.pt"
+        if target.exists():
+            load_ac_rows({"length": span["length"], "row_path": str(target)})
+        else:
+            temporary = None
+            try:
+                with tempfile.NamedTemporaryFile(dir=folder, prefix=f".{digest}.", suffix=".tmp", delete=False) as handle:
+                    temporary = Path(handle.name)
+                    torch.save(rows, handle)
+                    handle.flush()
+                    os.fsync(handle.fileno())
+                os.replace(temporary, target)
+            finally:
+                if temporary is not None:
+                    temporary.unlink(missing_ok=True)
+        return {key: value for key, value in span.items() if key not in ("rows", "row_path")} | {
+            "row_path": f"saved_tensors/{digest}.pt"}
+
+    return _map_ac_records(data, save)
+
+
+def deserialize_ac_rows(data: dict, *, base_dir: Path | None = None) -> dict:
+    """Resolve references once, without loading tensor payloads."""
+    def resolve(span: dict) -> dict:
+        span = dict(span)
+        if span.get("row_path"):
+            path = Path(span["row_path"])
+            if not path.is_absolute():
+                if base_dir is None:
+                    raise ValueError("Relative AC row references require the JSONL parent as base_dir")
+                path = Path(base_dir) / path
+            span["row_path"] = str(path.resolve())
+        return span
+    return _map_ac_records(data, resolve)
+
+
+def release_ac_rows(result: "AgentRunResult", serialized: dict, *, base_dir: Path) -> None:
+    """After JSONL publication, replace tensors in the actual retained span dictionaries."""
+    def release(record, saved):
+        data = record if isinstance(record, dict) else vars(record)
+        pairs = [(data.get("prompt_ac_spans", []), saved.get("prompt_ac_spans", []))]
+        pairs += [(step.get("ac_spans", []), saved_step.get("ac_spans", []))
+                  for step, saved_step in zip(data.get("trajectory", []), saved.get("trajectory", []))]
+        for spans, saved_spans in pairs:
+            for span, saved_span in zip(spans, saved_spans):
+                span["row_path"] = str((Path(base_dir) / saved_span["row_path"]).resolve())
+                span.pop("rows", None)
+        for key in ("compactions", "subagent_results"):
+            for child, saved_child in zip(data.get(key, []), saved.get(key, [])):
+                release(child, saved_child)
+    release(result, serialized)
+
+
+def ac_spans_for_report(spans: list[dict]) -> list[dict]:
+    """Inspection metadata only; reports are not another tensor dataset."""
+    result = []
+    for span in spans:
+        item = {key: value for key, value in span.items() if key not in ("rows", "row_path")}
+        rows = span.get("rows")
+        if rows is not None:
+            item.update(shape=list(rows.shape), dtype=str(rows.dtype))
+        if span.get("row_path"):
+            item["saved_tensor"] = Path(span["row_path"]).name
+        result.append(item)
+    return result
 ASSISTANT_SENTINEL = "\u241f TRESSOIR_ASSISTANT_TURN \u241f"   # never in real text; marks where the sampled tokens sit in a rendering
 TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
 TOOL_CALL_OPEN = re.compile(r"<tool_call>(?!.*</tool_call>)(.*)$", re.S)   # an unterminated block (max_tokens hit)
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/agent/rollout_caching.py">activation/agent/rollout_caching.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/agent/rollout_caching.py`

```diff
diff --git a/activation/agent/rollout_caching.py b/activation/agent/rollout_caching.py
index 7c82758..ff59cd0 100644
--- a/activation/agent/rollout_caching.py
+++ b/activation/agent/rollout_caching.py
@@ -15,6 +15,7 @@ from dataclasses import dataclass
 from activation.common.data_syncing import resolve_path
 
 from .agent_config import AgentConfig, AgentRunResult, _class_spec
+from .agent_utils import release_ac_rows
 
 CACHE_FILENAME = "rollouts.jsonl"
 
@@ -100,9 +101,16 @@ class RolloutCache:
     def append(self, result: AgentRunResult) -> None:
         if self.path is None:
             return
-        row = result.serialize() | {"config_key": config_key(result.agent_config)}
+        row = result.serialize(base_dir=self.path.parent) | {"config_key": config_key(result.agent_config)}
         with self.lock:
-            self.rows[(row["config_key"], int(result.seed))] = row
             with open(self.path, "a") as handle:
-                handle.write(json.dumps(row, ensure_ascii=False, default=repr) + "\n")
-                handle.flush()
+                offset = handle.tell()
+                try:
+                    handle.write(json.dumps(row, ensure_ascii=False, default=repr) + "\n")
+                    handle.flush()
+                except Exception:
+                    handle.seek(offset)
+                    handle.truncate()
+                    raise
+            self.rows[(row["config_key"], int(result.seed))] = row
+            release_ac_rows(result, row, base_dir=self.path.parent)
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/agent/rollout_manager.py">activation/agent/rollout_manager.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/agent/rollout_manager.py`

```diff
diff --git a/activation/agent/rollout_manager.py b/activation/agent/rollout_manager.py
index 24a61f3..f1487d7 100644
--- a/activation/agent/rollout_manager.py
+++ b/activation/agent/rollout_manager.py
@@ -90,7 +90,7 @@ class RolloutManager:
         results: list[AgentRunResult | None] = []
         for config, seed in jobs:
             row = cache.get(config_key(config), seed)
-            results.append(None if row is None else AgentRunResult.deserialize(row, self.harness, agent_config=config))
+            results.append(None if row is None else AgentRunResult.deserialize(row, self.harness, agent_config=config, base_dir=cache.path.parent))
         pending = [index for index, result in enumerate(results) if result is None]
         skipped = 0
         if redo is not None and redo.only:                                       # the rows to redo alone; never-cached tasks stay None
@@ -177,7 +177,10 @@ class RolloutManager:
             try:
                 config, seed = jobs[index]
                 if deadline is None:
-                    return self._run_one(config, seed, perform_scoring, reporter)
+                    result = self._run_one(config, seed, perform_scoring, reporter)
+                    if result.finish_reason != "deadline":
+                        cache.append(result)
+                    return result
                 now = time.time()
                 with counts_lock:
                     counts["skipped" if now >= deadline else "started"] += 1
@@ -192,6 +195,8 @@ class RolloutManager:
                         result.finish_reason = "deadline"
                     if result.agent_config is run_config:
                         result.agent_config = config           # the clip is an execution detail: the result belongs to the job's own config
+                if result.finish_reason != "deadline":
+                    cache.append(result)
                 return result
             finally:
                 limiter.release()
@@ -207,8 +212,6 @@ class RolloutManager:
                             continue
                         results[index] = result
                         finished[index] = (time.time(), result)
-                        if result.finish_reason != "deadline":        # a cut run is not a result: the next launch redoes it
-                            cache.append(result)
                         if probe and on_probe is not None and len(finished) >= len(probe):   # the first N to finish: a straggler must not delay the tuning by its whole duration
                             now = time.time()
                             phases.append(self._phase_metrics(label, started, now, loaded_model, sampler.samples, finished))
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/agent/rollout_reporter.py">activation/agent/rollout_reporter.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/agent/rollout_reporter.py`

```diff
diff --git a/activation/agent/rollout_reporter.py b/activation/agent/rollout_reporter.py
index 94ddb22..a464e1e 100644
--- a/activation/agent/rollout_reporter.py
+++ b/activation/agent/rollout_reporter.py
@@ -16,6 +16,7 @@ from collections import deque
 from pathlib import Path
 
 from activation.common.reporting import HtmlReporter, _write_atomic, format_seconds
+from .agent_utils import ac_spans_for_report
 
 if t.TYPE_CHECKING:
     from .agent import Agent
@@ -201,8 +202,9 @@ def trajectory_item(agent: "Agent", state: str, folder: Path | None = None) -> d
         _write_atomic(folder / file, json.dumps({
             "agent_id": agent.agent_id, "state": state, "system_prompt": agent.agent_config.system_prompt,
             "user_prompt": agent.agent_config.user_prompt,
-            "trajectory": [{key: value for key, value in step.items() if key not in ("token_ids", "logprobs", "messages")} for step in results.trajectory],
-            "compactions": len(results.compactions), "prompt_ac_spans": results.prompt_ac_spans,
+            "trajectory": [{key: ac_spans_for_report(value) if key == "ac_spans" else value
+                            for key, value in step.items() if key not in ("token_ids", "logprobs", "messages")} for step in results.trajectory],
+            "compactions": len(results.compactions), "prompt_ac_spans": ac_spans_for_report(results.prompt_ac_spans),
             "answer": results.answer, "finish_reason": results.finish_reason, "score": results.score,
         }, indent=1, default=str))
     steps = []
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/agent_training/agent_trainer.py">activation/agent_training/agent_trainer.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/agent_training/agent_trainer.py`

```diff
diff --git a/activation/agent_training/agent_trainer.py b/activation/agent_training/agent_trainer.py
index 77560d3..db40087 100644
--- a/activation/agent_training/agent_trainer.py
+++ b/activation/agent_training/agent_trainer.py
@@ -55,6 +55,8 @@ from .agent_training_utils import (
     set_checkpointing,
     set_learning_rates,
     sampled_logprobs,
+    training_inputs_embeds,
+    validate_fixed_rows,
 )
 
 if t.TYPE_CHECKING:
@@ -114,6 +116,8 @@ class AgentTrainer:
         training_data: list[AgentTrainingItem],
         reporting_data: t.Sequence[AgentTrainingItem] = (),
         reporter: AgentTrainingReporter | None = None,
+        *,
+        reset_optimizer: bool = False,
     ) -> AgentTrainingStats:
         """
         One round on `lora_name`: engine to sleep, base + adapter on the GPU, one clipped-surrogate pass
@@ -162,6 +166,8 @@ class AgentTrainer:
                             "gradient_checkpointing_min_tokens": variables["gradient_checkpointing_min_tokens"]}
         module_manager.ensure_lora(lora_name)
         base = loaded_model.model
+        validate_fixed_rows(training_data, examples, loaded_model)
+        validate_fixed_rows(reporting_data, reporting_examples, loaded_model)
         device = next(base.parameters()).device
         base.train()
         previous_attn = base.config._attn_implementation
@@ -175,6 +181,8 @@ class AgentTrainer:
         disable_dropout(base)
         parameters = module_manager.lora_parameters(lora_name)
         assert parameters, f"LoRA {lora_name!r} has no trainable parameters"
+        if reset_optimizer:
+            self.optimizers.pop(lora_name, None)
         optimizer = persistent_optimizer(self.optimizers, lora_name, [{"params": parameters, "lr": config.learning_rate}], device,
                                          betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay)
         pad_token_id = loaded_model.tokenizer.pad_token_id
@@ -278,7 +286,7 @@ class AgentTrainer:
     # ----------------------------------------------------------------------------- internals
     def _hidden_and_logprobs(self, loaded_model, lora_name: str, batch: Collated) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
         base = loaded_model.model
-        inputs_embeds = base.get_input_embeddings()(batch.input_ids)
+        inputs_embeds = training_inputs_embeds(base, batch)
         kwargs = dict(self.config.decoder_kwargs)
         attention_mask = batch.model_attention_mask
         if batch.packed:
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/agent_training/agent_training_utils.py">activation/agent_training/agent_training_utils.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/agent_training/agent_training_utils.py`

```diff
diff --git a/activation/agent_training/agent_training_utils.py b/activation/agent_training/agent_training_utils.py
index 13a7590..ee9154c 100644
--- a/activation/agent_training/agent_training_utils.py
+++ b/activation/agent_training/agent_training_utils.py
@@ -7,10 +7,12 @@ sequence (16k positions x a 250k vocabulary would not fit).
 from __future__ import annotations
 
 import typing as t
-from dataclasses import dataclass
+from dataclasses import dataclass, field
 
+import math
 import torch
 import torch.utils.checkpoint
+from ..agent.agent_utils import load_ac_rows
 
 if t.TYPE_CHECKING:
     from .agent_trainer import AgentTrainingItem
@@ -25,6 +27,7 @@ class TrainingExample:
     weight: float                  # the advantage A of the whole trajectory
     item_index: int
     needs_old_logprobs: bool       # teacher / hinted items (or runs recorded without log-probs): pi_old comes from a no-grad pass
+    ac_spans: list[dict] = field(default_factory=list)  # absolute sequence offsets; captured rows or resolved file references
 
     @property
     def num_loss_tokens(self) -> int:
@@ -54,18 +57,23 @@ def build_example(item: "AgentTrainingItem", item_index: int, think_replacement:
     replaced by `think_replacement` (no loss on it), so the student never conditions on reasoning it will not have.
     """
     run = item.run_results
-    if run.prompt_ac_spans or any(step.get("ac_spans") for step in run.trajectory):
-        # Placeholder ids at part positions would be embedded as pad tokens: training on AC-bearing sequences
-        # needs the rows rebuilt from the recorded parts (deferred to the training-recipe slices).
-        raise ValueError("Training on AC-bearing runs awaits the fixed-row trainer integration")
     token_ids = list(run.prompt_token_ids)
     if not token_ids:
         return None
     loss_mask = [False] * len(token_ids)
     old_logprobs = [0.0] * len(token_ids)
+    ac_spans = [dict(span) for span in run.prompt_ac_spans]
+    if any(span["start"] < 0 or span["start"] + span["length"] > len(token_ids) for span in ac_spans):
+        raise ValueError("AC span lies outside its recorded prompt")
     any_recorded = False
     for step in run.trajectory:
         segment = list(step.get("token_ids") or [])
+        if step.get("ac_spans") and step.get("role") == "assistant":
+            raise ValueError("Sampled assistant tokens cannot contain AC input rows")
+        for span in step.get("ac_spans", []):
+            if span["start"] < 0 or span["start"] + span["length"] > len(segment):
+                raise ValueError("AC span lies outside its recorded token segment")
+            ac_spans.append(dict(span, start=len(token_ids) + span["start"]))
         if step.get("role") == "assistant":
             logprobs = [float(value) for value in (step.get("logprobs") or [])]
             think_end = step.get("think_end")
@@ -94,9 +102,16 @@ def build_example(item: "AgentTrainingItem", item_index: int, think_replacement:
     loss_mask[0] = False                                                   # the first token is never predicted
     if not any(loss_mask):
         return None
+    previous_end = 0
+    for span in ac_spans:
+        start, end = span["start"], span["start"] + span["length"]
+        if start < previous_end or end <= start or end > len(token_ids) or any(loss_mask[start:end]):
+            raise ValueError("AC spans must be ordered, nonoverlapping input-only positions within the trajectory")
+        previous_end = end
     return TrainingExample(
         token_ids=token_ids, loss_mask=loss_mask, old_logprobs=old_logprobs, weight=float(item.weight), item_index=item_index,
         needs_old_logprobs=item.ignore_logprobs or not any_recorded,
+        ac_spans=ac_spans,
     )
 
 
@@ -141,6 +156,7 @@ class Collated:
     segment_offsets: list[int]     # where each example starts in its row (0 unless packed)
     model_attention_mask: torch.Tensor | None = None   # what the decoder gets: None = causal only (right padding never reaches a real token)
     cu_seq_lens: torch.Tensor | None = None            # packed: int32 [E + 1] boundaries, for the linear-attention kernels (fla varlen)
+    ac_spans: list[tuple[int, dict]] = field(default_factory=list)  # batch row and span at its padded/packed offset
 
     @property
     def packed(self) -> bool:
@@ -191,6 +207,7 @@ def collate(examples: list[TrainingExample], indices: list[int], pad_token_id: i
         segment_ids=torch.arange(len(indices), device=device)[:, None].expand(len(indices), length).contiguous(),
         segment_offsets=[0] * len(indices),
         model_attention_mask=None if causal_only else attention_mask.to(device),
+        ac_spans=[(row, span) for row, index in enumerate(indices) for span in examples[index].ac_spans],
     )
 
 
@@ -236,9 +253,46 @@ def _collate_packed(examples: list[TrainingExample], indices: list[int], pad_tok
         segment_offsets=offsets,
         model_attention_mask=None,
         cu_seq_lens=torch.tensor(boundaries, dtype=torch.int32, device=device),
+        ac_spans=[(0, dict(span, start=offset + span["start"]))
+                  for index, offset in zip(indices, offsets) for span in examples[index].ac_spans],
     )
 
 
+def fixed_rows_for_model(span: dict, embedding: torch.nn.Module) -> torch.Tensor:
+    rows = load_ac_rows(span)
+    if rows.shape[1] != embedding.weight.shape[1] or rows.dtype != embedding.weight.dtype:
+        raise ValueError(f"Captured AC rows {tuple(rows.shape)}/{rows.dtype} do not match reader embeddings "
+                         f"{embedding.weight.shape[1]}/{embedding.weight.dtype}")
+    return rows
+
+
+def validate_fixed_rows(items, examples: list[TrainingExample], loaded_model) -> None:
+    """Preflight payloads one at a time without retaining a dataset of loaded tensors."""
+    for example in examples:
+        if not example.ac_spans:
+            continue
+        source_name = items[example.item_index].run_results.agent_config.model_name
+        if source_name != loaded_model.model_config.model_name:
+            source = loaded_model.harness.loaded_models.get(source_name)
+            if source is None or source.model_config.model_id != loaded_model.model_config.model_id:
+                raise ValueError("Captured AC rows belong to a different reader model")
+        for span in example.ac_spans:
+            fixed_rows_for_model(span, loaded_model.model.get_input_embeddings())
+
+
+def training_inputs_embeds(model, batch: Collated) -> torch.Tensor:
+    """Replay the recorded observation for both reference and policy passes; no encoder."""
+    embedding = model.get_input_embeddings()
+    inputs = embedding(batch.input_ids)
+    if batch.ac_spans:
+        inputs = inputs.clone()
+        for row, span in batch.ac_spans:
+            rows = fixed_rows_for_model(span, embedding)
+            start = span["start"]
+            inputs[row, start:start + span["length"]] = rows.to(inputs.device)
+    return inputs
+
+
 class _HeadMatmul(torch.autograd.Function):
     """
     logits = states @ weight.T with bf16 operands on the tensor cores and fp32 accumulation and output
@@ -421,9 +475,15 @@ def optimizer_state_to(optimizer: torch.optim.Optimizer, device) -> None:
                 state[key] = value.to(target)
 
 
-def set_learning_rates(optimizer: torch.optim.Optimizer, updates_done: int, warmup_updates: int) -> float:
-    """Linear warm-up: every group's lr = base_lr x min(1, (updates_done + 1) / warmup_updates); the configured rate once past it. Returns the factor."""
+def set_learning_rates(optimizer: torch.optim.Optimizer, updates_done: int, warmup_updates: int, *, total_updates: int | None = None,
+                       schedule: str = "constant", final_fraction: float = 0.1) -> float:
+    """Linear warm-up: every group's lr = base_lr x min(1, (updates_done + 1) / warmup_updates); the configured rate once past it.
+    schedule="cosine": after the warm-up the factor follows a half cosine from 1 to final_fraction over total_updates. Returns the factor."""
     factor = min(1.0, (updates_done + 1) / warmup_updates) if warmup_updates > 0 else 1.0
+    if schedule == "cosine" and total_updates:
+        span = max(1, total_updates - warmup_updates)
+        progress = min(1.0, max(0.0, (updates_done - warmup_updates) / span))
+        factor *= final_fraction + (1.0 - final_fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))
     for group in optimizer.param_groups:
         group["lr"] = group["base_lr"] * factor
     return factor
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/autotuners/fla.py">activation/autotuners/fla.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/autotuners/fla.py`

```diff
diff --git a/activation/autotuners/fla.py b/activation/autotuners/fla.py
index 9e4ea8a..b5d1e41 100644
--- a/activation/autotuners/fla.py
+++ b/activation/autotuners/fla.py
@@ -44,8 +44,8 @@ def validate_catalog(profile: dict) -> bool:
     tuners = inventory()
     try:
         records = profile["records"]
-        if {record["kernel"] for record in records} != set(tuners):
-            return False
+        if {record["kernel"] for record in records} != set(tuners) - set(profile.get("unused_kernels", [])):
+            return False        # every Triton tuner this platform runs has records; TileLang-routed ones (Hopper) are listed as unused
         return all(record["config"] in [config_values(c) for c in tuners[record["kernel"]].configs]
                    and record["semantic"] and record["key"] == digest(record["semantic"]) for record in records)
     except (KeyError, TypeError, ValueError):
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/autotuners/worker.py">activation/autotuners/worker.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/autotuners/worker.py`

```diff
diff --git a/activation/autotuners/worker.py b/activation/autotuners/worker.py
index 014d533..e89e54b 100644
--- a/activation/autotuners/worker.py
+++ b/activation/autotuners/worker.py
@@ -68,7 +68,11 @@ def validate(geometry, catalog):
               exercise(geometry, catalog, length=129, batch=7),
               exercise(geometry, catalog, length=129, packed=True),
               exercise(geometry, catalog, length=65, final_state=True)]
-    assert catalog.visited == set(inventory()), (catalog.visited, set(inventory()))
+    unused = set(inventory()) - catalog.visited
+    assert catalog.visited <= set(inventory()), (catalog.visited, set(inventory()))
+    if unused:      # FLA's backend dispatch routes some ops to TileLang on Hopper (e.g. chunk_bwd_dqkwg): those Triton tuners never run here
+        print(f"Autotuners: Triton kernels not used on this platform: {sorted(unused)}", flush=True)
+    catalog.unused_kernels = sorted(unused)
     return max(errors)
 
 
@@ -89,7 +93,7 @@ def main():
             exercise(geometry, catalog, length=length, reference=False)
         def save(phase, selected, error):
             seconds = time.monotonic() - started
-            profile = {"identity": identity, "coverage": MODES,
+            profile = {"identity": identity, "coverage": MODES, "unused_kernels": getattr(selected, "unused_kernels", []),
                        "records": sorted(selected.records.values(), key=lambda r: (r["kernel"], r["key"], r["bucket"])),
                        "validation": {"passed": True, "max_relative_rms_error": error, "batches": [1, 3, 7],
                                       "measured_lengths": [65, 129, 2048, 8192, 32768],
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/cloud/sky.py">activation/cloud/sky.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/cloud/sky.py`

```diff
diff --git a/activation/cloud/sky.py b/activation/cloud/sky.py
index ab58560..6d0b7f2 100644
--- a/activation/cloud/sky.py
+++ b/activation/cloud/sky.py
@@ -177,7 +177,7 @@ def _cluster_record(name: str, *, refresh: bool = False) -> dict | None:
     return next((record for record in records if record["name"] == name), None)
 
 
-def _labels(gpu_type: str, gpu_count: int) -> dict[str, str]:
+def _labels(gpu_type: str, gpu_count: int, use_spot: bool = False) -> dict[str, str]:
     return {
         f"{LABEL_PREFIX}-config": CONFIG_VERSION,
         f"{LABEL_PREFIX}-cloud": "aws",
@@ -185,7 +185,7 @@ def _labels(gpu_type: str, gpu_count: int) -> dict[str, str]:
         f"{LABEL_PREFIX}-gpu-count": str(gpu_count),
         f"{LABEL_PREFIX}-disk-gb": str(DISK_SIZE_GB),
         f"{LABEL_PREFIX}-autostop-minutes": str(AUTOSTOP_MINUTES),
-        f"{LABEL_PREFIX}-spot": "false",
+        f"{LABEL_PREFIX}-spot": "true" if use_spot else "false",
     }
 
 
@@ -195,7 +195,7 @@ def _require_matching_config(
     gpu_count: int,
 ) -> None:
     actual = record.get("labels") or {}
-    expected = _labels(gpu_type, gpu_count)
+    expected = _labels(gpu_type, gpu_count, actual.get(f"{LABEL_PREFIX}-spot") == "true")
     if all(actual.get(key) == value for key, value in expected.items()):
         return
 
@@ -417,10 +417,11 @@ def _task_yaml(
     *,
     image_id: str = IMAGE_ID,
     docker_secrets: dict[str, str] | None = None,
+    use_spot: bool = False,
 ) -> str:
     labels = "\n".join(
         f"    {key}: {json.dumps(value)}"
-        for key, value in _labels(gpu_type, gpu_count).items()
+        for key, value in _labels(gpu_type, gpu_count, use_spot).items()
     )
     secrets = docker_secrets if docker_secrets is not None else {}
     secrets_yaml = ""
@@ -435,7 +436,7 @@ def _task_yaml(
 resources:
   infra: aws
   accelerators: {GPU_TYPES[gpu_type]}:{gpu_count}
-  use_spot: false
+  use_spot: {"true" if use_spot else "false"}
   disk_size: {DISK_SIZE_GB}
   image_id: {image_id}
   labels:
@@ -464,6 +465,7 @@ def setup(
     gpu_type: str,
     gpu_count: int = 1,
     rebuild_image: bool = True,
+    use_spot: bool = False,
 ) -> int:
     gpu_type = gpu_type.lower()
     if gpu_type not in GPU_TYPES:
@@ -505,6 +507,7 @@ def setup(
                 gpu_count,
                 image_id=image_id,
                 docker_secrets=docker_secrets,
+                use_spot=use_spot,
             )
         )
         task_file.flush()
@@ -933,6 +936,11 @@ def _build_parser() -> argparse.ArgumentParser:
         default=1,
         help="number of GPUs (default: 1)",
     )
+    setup_parser.add_argument(
+        "--spot",
+        action="store_true",
+        help="request a spot instance (cheaper, may be preempted; runs must be restartable)",
+    )
     setup_parser.add_argument(
         "--no-image-rebuild",
         action="store_true",
@@ -1026,6 +1034,7 @@ def main() -> int:
                 args.gpu,
                 args.gpu_count,
                 rebuild_image=not args.no_image_rebuild,
+                use_spot=args.spot,
             )
         if args.action == "exec":
             return exec_cmd(args.name, args.command, sync=args.sync, watch_paths=[tuple(pair) for pair in args.watch] if args.watch else None,
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/common/ac_parts.py">activation/common/ac_parts.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/common/ac_parts.py`

```diff
diff --git a/activation/common/ac_parts.py b/activation/common/ac_parts.py
index bfc17c6..e80745d 100644
--- a/activation/common/ac_parts.py
+++ b/activation/common/ac_parts.py
@@ -104,6 +104,25 @@ def is_ac_part(part: object) -> bool:
     return isinstance(part, dict) and part.get("type") == AC_PART_TYPE
 
 
+def strip_ac_parts(messages: list[dict]) -> list[dict]:
+    """
+    The same messages with every activation_context part removed (descendants included): the no-context view of a
+    history. Text pieces of a content list are kept in order; a list left with only text becomes one string.
+    """
+    stripped = []
+    for message in messages:
+        content = message.get("content")
+        if isinstance(content, list):
+            pieces = [piece for piece in content if not is_ac_part(piece)]
+            if all(isinstance(piece, dict) and piece.get("type") == "text" for piece in pieces):
+                content = "".join(piece.get("text", "") for piece in pieces)
+            else:
+                content = pieces
+            message = {**message, "content": content}
+        stripped.append(message)
+    return stripped
+
+
 def direct_parts(messages: list[dict]) -> list[dict]:
     """The activation_context parts of these messages, in order of appearance (not their descendants)."""
     parts = []
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/common/reporting.py">activation/common/reporting.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/common/reporting.py`

```diff
diff --git a/activation/common/reporting.py b/activation/common/reporting.py
index 2a033fe..2fcd991 100644
--- a/activation/common/reporting.py
+++ b/activation/common/reporting.py
@@ -2,13 +2,18 @@
 Live HTML report for long-running jobs.
 
 One self-describing JSON document (status, named widgets, their points) rendered into a linear
-Tressoir page with Plotly.js. `report_data.json` is rewritten atomically on every render; the page
-polls it (every `poll_seconds`) and redraws in place, so the HTML file itself is rewritten only
-every `page_rewrite_seconds` (and at the end) and the viewer does not flicker on file changes. Every
-number the page shows is also in the embedded data block, which is the fallback when the poll cannot
-fetch (e.g. a file:// page), so the finished page still works alone. The page renders inside the
-Tressoir VS Code webview (which injects the markup after load, morphs it in place on file changes
-and fires `tressoir:render`) and in a plain browser (timed reload when polling fails).
+Tressoir page with Plotly.js. `report_data.json` is rewritten atomically on every render and is the
+live path: the page polls it (every `poll_seconds`) and updates its existing DOM in place, keeping
+sections by widget name, redrawing plots with Plotly.react (the viewer's zoom and legend state
+survive), rebuilding only table rows and changing text only when it changed. The HTML file itself is
+written on the first render, whenever a widget is added, removed or reordered, when the run finishes,
+and otherwise at most every `page_rewrite_seconds`. Every number the page shows is also in the
+embedded data block, which is what the page shows when the poll cannot fetch (a file:// page, or a
+viewer without the folder as a resource root): the failure is stated in the page's metadata line and
+the fetch is retried with a bounded backoff; the page never reloads itself. Inside the Tressoir
+VS Code webview (which injects the markup after load, morphs it in place on file changes and fires
+`tressoir:render`) repeated render events are harmless: they adopt the embedded data only when it is
+newer than what the page shows. There is one polling loop and one boot per page.
 
 `HtmlReporter` is job-agnostic: widgets are created by name (line plots, bar plots, tables, text
 blocks, rendered trajectories) and fed through `add_data_point`. Job-specific reporters subclass it and own the widget
@@ -54,6 +59,7 @@ _PAGE_TEMPLATE = """<!doctype html>
     .report-plot { width: 100%; height: 22rem; margin-block: var(--space-2) var(--space-3); }
     .report-plot.tall { height: 26rem; }
     .report-footer { color: var(--muted); font-size: 0.85rem; }
+    .report-live-failure { color: var(--warning); }
     .report-table td { font-variant-numeric: tabular-nums; white-space: nowrap; }
     .trajectory { border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); padding: var(--space-2) var(--space-3); margin-block: var(--space-2); }
     .trajectory > summary { cursor: pointer; font-weight: 600; }
@@ -116,9 +122,19 @@ _PAGE_TEMPLATE = """<!doctype html>
   <script defer src="https://cdn.plot.ly/plotly-basic-2.35.2.min.js"></script>
   <script>
   (function () {
+    // The page keeps one DOM and updates it in place: sections are kept by widget name, plots are
+    // redrawn with Plotly.react (zoom and legend state survive), tables rebuild only their rows,
+    // text blocks change only when their text changed. There is one polling loop and one boot;
+    // repeated tressoir:render events are harmless; a failed fetch is shown, never a reload.
+    if (window.__reportPageBooted) return;
+    window.__reportPageBooted = true;
     var report = null;
-    var plots = [];
+    var plots = {};            // widget name -> { div, widget }
+    var sections = {};         // widget name -> <section>
     var polling = false;
+    var pollTimer = null;
+    var pollDelay = 0;
+    var fetchFailure = null;   // { at, reason } while live data is unavailable
     var embedded = JSON.parse(document.getElementById("report-data").textContent);
 
     function cssColor(token, fallback) {
@@ -185,6 +201,7 @@ _PAGE_TEMPLATE = """<!doctype html>
         legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { color: t.muted } },
         hovermode: "closest",
         autosize: true,
+        uirevision: widget.name,   // a constant per widget: Plotly.react keeps the viewer's zoom and legend state
       };
       if (widget.log_y) { l.yaxis.type = "log"; l.yaxis.exponentformat = "e"; l.yaxis.showexponent = "all"; }
       if (widget.type === "bar") l.xaxis.type = "category";
@@ -193,7 +210,8 @@ _PAGE_TEMPLATE = """<!doctype html>
     function drawAll() {
       if (!window.Plotly) return;
       var t = theme();
-      plots.forEach(function (p) {
+      Object.keys(plots).forEach(function (name) {
+        var p = plots[name];
         if (!p.div.isConnected) return;
         Plotly.react(p.div, traces(p.widget, t), layout(p.widget, t), { displaylogo: false, responsive: false });
       });
@@ -204,6 +222,9 @@ _PAGE_TEMPLATE = """<!doctype html>
       if (text !== undefined) node.textContent = text;
       return node;
     }
+    function setText(node, text) {
+      if (node.textContent !== text) node.textContent = text;
+    }
     function codeBlock(text) {
       var pre = el("pre", "code-block");
       pre.appendChild(el("code", "", text));
@@ -244,94 +265,189 @@ _PAGE_TEMPLATE = """<!doctype html>
       });
       return details;
     }
-    function build() {
-      if (!report) report = embedded;
-      if (window.Plotly) plots.forEach(function (p) { try { Plotly.purge(p.div); } catch (_) {} });
-      plots = [];
-      document.title = report.title;
+    // Reorder `parent`'s children to match `wanted` (a list of nodes), moving only nodes that are out of place.
+    function reorder(parent, wanted) {
+      var cursor = parent.firstElementChild;
+      for (var i = 0; i < wanted.length; i++) {
+        var node = wanted[i];
+        if (cursor === node) { cursor = cursor.nextElementSibling; continue; }
+        parent.insertBefore(node, cursor);
+      }
+    }
+    function updateMeta() {
       var meta = document.getElementById("report-meta");
-      meta.replaceChildren();
-      ["Updated " + report.updated_at, (polling ? "Polling report_data.json every " + report.poll_seconds + " s" : "Refreshes every " + report.refresh_seconds + " s"), "Self-contained: data embedded, libraries from pinned HTTPS URLs"]
-        .forEach(function (text) { meta.appendChild(el("li", "", text)); });
+      var lines = ["Updated " + report.updated_at,
+                   (polling ? "Polling report_data.json every " + report.poll_seconds + " s" : "Showing the data embedded in the page"),
+                   "Self-contained: data embedded, libraries from pinned HTTPS URLs"];
+      if (fetchFailure) {
+        lines.push("Live data unavailable since " + fetchFailure.at + " (" + fetchFailure.reason + "); showing the last data this page has; retrying");
+      }
+      while (meta.children.length > lines.length) meta.removeChild(meta.lastChild);
+      lines.forEach(function (text, i) {
+        var li = meta.children[i];
+        if (!li) { li = el("li"); meta.appendChild(li); }
+        if (i === lines.length - 1 && fetchFailure) li.className = "report-live-failure"; else if (li.className) li.className = "";
+        setText(li, text);
+      });
+    }
+    function updateStatus() {
       var status = document.getElementById("report-status");
-      status.replaceChildren();
-      Object.keys(report.status).forEach(function (key) {
-        var item = el("div");
-        item.appendChild(el("dt", "", key));
-        item.appendChild(el("dd", "", report.status[key]));
-        status.appendChild(item);
+      var keys = Object.keys(report.status);
+      var byKey = {};
+      Array.prototype.forEach.call(status.children, function (item) { byKey[item.getAttribute("data-key")] = item; });
+      var wanted = keys.map(function (key) {
+        var item = byKey[key];
+        if (!item) {
+          item = el("div"); item.setAttribute("data-key", key);
+          item.appendChild(el("dt", "", key)); item.appendChild(el("dd", "", ""));
+          status.appendChild(item);
+        }
+        setText(item.lastChild, report.status[key]);
+        delete byKey[key];
+        return item;
       });
-      var root = document.getElementById("report-widgets");
-      root.replaceChildren();
-      report.widgets.forEach(function (w) {
-        var id = "widget-" + w.name;
-        var section = el("section", "section");
-        section.setAttribute("aria-labelledby", id);
-        var h = el("h2", "", w.title); h.id = id; section.appendChild(h);
-        if (w.description) section.appendChild(el("p", "", w.description));
-        if (w.type === "line" || w.type === "bar") {
+      Object.keys(byKey).forEach(function (key) { byKey[key].remove(); });
+      reorder(status, wanted);
+    }
+    function updateTable(section, w) {
+      var table = section.querySelector("table");
+      var columnsKey = JSON.stringify(w.columns);
+      if (!table || table.getAttribute("data-columns") !== columnsKey) {
+        var region = section.querySelector(".scroll-region");
+        if (region) region.remove();
+        region = el("div", "scroll-region");
+        table = el("table", "table report-table");
+        table.setAttribute("data-columns", columnsKey);
+        var head = table.createTHead().insertRow();
+        w.columns.forEach(function (c) { head.appendChild(el("th", "", c)); });
+        table.createTBody();
+        region.appendChild(table); section.appendChild(region);
+      }
+      var body = table.tBodies[0];
+      var rowsKey = JSON.stringify(w.rows);
+      if (body.getAttribute("data-rows") === rowsKey) return;
+      body.setAttribute("data-rows", rowsKey);
+      body.replaceChildren();
+      w.rows.forEach(function (row) {
+        var tr = body.insertRow();
+        w.columns.forEach(function (c) { tr.insertCell().textContent = row[c] === undefined || row[c] === null ? "" : row[c]; });
+      });
+    }
+    function updateTrajectories(section, w) {
+      var key = JSON.stringify(w.items);
+      if (section.getAttribute("data-items") === key) return;
+      section.setAttribute("data-items", key);
+      Array.prototype.slice.call(section.children).forEach(function (child) {
+        if (child.tagName !== "H2" && !child.classList.contains("widget-description")) child.remove();
+      });
+      w.items.forEach(function (item) { section.appendChild(trajectory(item)); });
+      if (!w.items.length) section.appendChild(el("p", "", "No trajectories yet."));
+    }
+    function updateWidget(w) {
+      var id = "widget-" + w.name;
+      var section = sections[w.name];
+      if (!section) {
+        section = el("section", "section");
+        section.id = id;
+        section.setAttribute("aria-labelledby", id + "-heading");
+        var h = el("h2", "", w.title); h.id = id + "-heading"; section.appendChild(h);
+        section.appendChild(el("p", "widget-description", ""));
+        sections[w.name] = section;
+        document.getElementById("report-widgets").appendChild(section);
+      }
+      setText(section.firstElementChild, w.title);
+      var description = section.querySelector(".widget-description");
+      setText(description, w.description || "");
+      description.hidden = !w.description;
+      if (w.type === "line" || w.type === "bar") {
+        var p = plots[w.name];
+        if (!p) {
           var div = el("div", "report-plot" + (w.smoothing_window ? " tall" : ""));
           section.appendChild(div);
-          plots.push({ div: div, widget: w });
-        } else if (w.type === "table") {
-          var region = el("div", "scroll-region");
-          var table = el("table", "table report-table");
-          var head = table.createTHead().insertRow();
-          w.columns.forEach(function (c) { head.appendChild(el("th", "", c)); });
-          var body = table.createTBody();
-          w.rows.forEach(function (row) {
-            var tr = body.insertRow();
-            w.columns.forEach(function (c) { tr.insertCell().textContent = row[c] === undefined || row[c] === null ? "" : row[c]; });
-          });
-          region.appendChild(table); section.appendChild(region);
-        } else if (w.type === "text") {
-          var pre = el("pre", "code-block");
-          pre.appendChild(el("code", "", w.text));
-          section.appendChild(pre);
-        } else if (w.type === "trajectories") {
-          w.items.forEach(function (item) { section.appendChild(trajectory(item)); });
-          if (!w.items.length) section.appendChild(el("p", "", "No trajectories yet."));
+          p = plots[w.name] = { div: div, widget: w };
         }
-        root.appendChild(section);
+        p.widget = w;
+      } else if (w.type === "table") {
+        updateTable(section, w);
+      } else if (w.type === "text") {
+        var pre = section.querySelector("pre.code-block");
+        if (!pre) { pre = codeBlock(""); section.appendChild(pre); }
+        setText(pre.firstChild, w.text);
+      } else if (w.type === "trajectories") {
+        updateTrajectories(section, w);
+      }
+      return section;
+    }
+    function update() {
+      if (!report) report = embedded;
+      if (document.title !== report.title) document.title = report.title;
+      setText(document.getElementById("report-title"), report.title);
+      setText(document.getElementById("report-description"), report.description);
+      updateMeta();
+      updateStatus();
+      var root = document.getElementById("report-widgets");
+      var seen = {};
+      var wanted = report.widgets.map(function (w) { seen[w.name] = true; return updateWidget(w); });
+      Object.keys(sections).forEach(function (name) {
+        if (seen[name]) return;
+        var gone = plots[name];
+        if (gone && window.Plotly) { try { Plotly.purge(gone.div); } catch (_) {} }
+        delete plots[name];
+        sections[name].remove();
+        delete sections[name];
       });
-      document.getElementById("report-footer").textContent =
-        "Updated " + report.updated_at + " · refreshes every " + report.refresh_seconds + " s · plots: Plotly.js basic 2.35.2 from cdn.plot.ly · page assets: Tressoir linear v0.1.7 and CodeMirror 5.65.16 from CDNs";
+      reorder(root, wanted);
+      setText(document.getElementById("report-footer"),
+        "Updated " + report.updated_at + " · plots: Plotly.js basic 2.35.2 from cdn.plot.ly · page assets: Tressoir linear v0.1.7 and CodeMirror 5.65.16 from CDNs");
     }
     function render() {
-      build();
+      update();
       var tries = 0;
       (function whenPlotly() {
         if (window.Plotly) return drawAll();
         if (tries++ < 600) setTimeout(whenPlotly, 50);   // up to 30 s for the network
       })();
     }
+    function schedulePoll(seconds) {
+      if (pollTimer !== null) clearTimeout(pollTimer);
+      pollTimer = setTimeout(poll, seconds * 1000);
+    }
     function poll() {
       // report_data.json beside the page is rewritten on every render; the page itself only rarely.
+      if (pollTimer !== null) { clearTimeout(pollTimer); pollTimer = null; }
       if (report && report.finished && polling) return;
+      var base = (report && report.poll_seconds) || 2;
       fetch("report_data.json?t=" + Date.now(), { cache: "no-store" }).then(function (response) {
-        if (!response.ok) throw new Error("status " + response.status);
+        if (!response.ok) throw new Error("HTTP " + response.status);
         return response.json();
       }).then(function (fresh) {
-        polling = true;
+        var recovered = fetchFailure !== null || !polling;
+        polling = true; fetchFailure = null; pollDelay = 0;
         if (JSON.stringify(fresh) !== JSON.stringify(report)) { report = fresh; render(); }
-        setTimeout(poll, (report.poll_seconds || 2) * 1000);
-      }).catch(function () {
-        // No fetch here (file://, or a viewer without the folder as a resource root): the embedded
-        // data and the occasional file rewrite carry the page instead.
-        if (!polling && /^(https?|file):$/.test(location.protocol) && !window.tressoirNotebook && !report.finished) {
-          setTimeout(function () { location.reload(); }, report.refresh_seconds * 1000);
-        }
+        else if (recovered) updateMeta();
+        schedulePoll(report.poll_seconds || 2);
+      }).catch(function (error) {
+        // No reload: the data already on the page stays, the failure is shown, and the fetch is
+        // retried with a bounded backoff (the extension delivers file changes through
+        // tressoir:render in the meantime).
+        if (!fetchFailure) fetchFailure = { at: new Date().toLocaleTimeString(), reason: (error && error.message) || String(error) };
+        pollDelay = Math.min(60, pollDelay ? pollDelay * 2 : base);
+        updateMeta();
+        schedulePoll(pollDelay);
       });
     }
+    function onRender() {
+      // Fired by the Tressoir extension after it injects the page and after each in-place morph:
+      // adopt the embedded data only when it is newer than what the page already shows.
+      var fresh = JSON.parse(document.getElementById("report-data").textContent);
+      if (!report || fresh.updated_at > report.updated_at) report = fresh;
+      render();
+    }
     function boot() {
       // The Tressoir extension injects this page after load and fires tressoir:render once the
       // scripts have run, and again after it morphs the file in place; a plain browser renders
-      // now. Either way the page then polls report_data.json.
-      document.addEventListener("tressoir:render", function () {
-        var fresh = JSON.parse(document.getElementById("report-data").textContent);
-        if (!report || fresh.updated_at > report.updated_at) report = fresh;
-        render();
-      });
+      // now. Either way the page then polls report_data.json. Everything here runs once.
+      document.addEventListener("tressoir:render", onRender);
       if (!window.tressoirNotebook) render();
       poll();
       if (window.matchMedia) {
@@ -342,7 +458,7 @@ _PAGE_TEMPLATE = """<!doctype html>
         new MutationObserver(drawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme-kind"] });
       }
       window.addEventListener("resize", function () {
-        if (window.Plotly) plots.forEach(function (p) { if (p.div.isConnected) Plotly.Plots.resize(p.div); });
+        if (window.Plotly) Object.keys(plots).forEach(function (name) { var p = plots[name]; if (p.div.isConnected) Plotly.Plots.resize(p.div); });
       });
     }
     if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
@@ -413,6 +529,7 @@ class HtmlReporter:
         self.finished = False
         self.last_render_time = 0.0
         self.last_page_time = 0.0
+        self.last_page_widgets: list[str] | None = None     # widget names as of the last page write
 
     # ----------------------------------------------------------------------------- widgets
     def set_status(self, **fields: str) -> None:
@@ -517,8 +634,10 @@ class HtmlReporter:
 
     def render(self, force: bool = False) -> None:
         """
-        Write report_data.json atomically (throttled unless forced); the page itself is written on the
-        first render, then at most every page_rewrite_seconds, and when the run finishes.
+        Write report_data.json atomically (throttled unless forced): that file is the live path the
+        page polls. The page itself is written on the first render, whenever the set or order of
+        widgets changed since it was last written, when the run finishes, and otherwise at most every
+        page_rewrite_seconds so a page opened without polling is not too stale.
         """
         now = time.time()
         if not force and now - self.last_render_time < self.min_render_interval_seconds:
@@ -527,9 +646,12 @@ class HtmlReporter:
         report = self.document()
         _write_atomic(self.folder / DATA_FILENAME, json.dumps(report, indent=1))
         page = self.folder / REPORT_FILENAME
-        if self.finished or not page.exists() or now - self.last_page_time >= self.page_rewrite_seconds:
+        widget_names = [widget["name"] for widget in report["widgets"]]
+        structure_changed = widget_names != self.last_page_widgets
+        if self.finished or not page.exists() or structure_changed or now - self.last_page_time >= self.page_rewrite_seconds:
             _write_atomic(page, render_report_page(report))
             self.last_page_time = now
+            self.last_page_widgets = widget_names
         self.last_render_time = now
 
     def finish(self) -> None:
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/dataset/dataset_manager.py">activation/dataset/dataset_manager.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/dataset/dataset_manager.py`

```diff
diff --git a/activation/dataset/dataset_manager.py b/activation/dataset/dataset_manager.py
index 030bd26..dcfc48a 100644
--- a/activation/dataset/dataset_manager.py
+++ b/activation/dataset/dataset_manager.py
@@ -60,6 +60,7 @@ class DatasetManager:
         study_context: str | None = None,
         caching_id: str | None = None,
         modality: DataModality | None = None,
+        *, progress: t.Callable[[dict], None] | None = None,
     ) -> list[DatasetQAExample]:
         """
         Synthesize study questions from the dataset's chunks (see DatasetStudyGenerator.generate_examples_qa).
@@ -67,7 +68,7 @@ class DatasetManager:
         """
         study_generator = self._get_or_create_study_generator(dataset_id)
         seed = self.harness.harness_config.dataset_study_seed if base_seed is None else base_seed
-        return study_generator.generate_examples_qa(num_samples, seed, study_context, caching_id, modality)
+        return study_generator.generate_examples_qa(num_samples, seed, study_context, caching_id, modality, progress=progress)
 
 
     def label_study_examples(
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/dataset/dataset_study.py">activation/dataset/dataset_study.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/dataset/dataset_study.py`

```diff
diff --git a/activation/dataset/dataset_study.py b/activation/dataset/dataset_study.py
index 51f8db6..12e15a7 100644
--- a/activation/dataset/dataset_study.py
+++ b/activation/dataset/dataset_study.py
@@ -202,6 +202,7 @@ class DatasetStudyGenerator:
         study_context: str | None = None,
         caching_id: str | None = None,
         modality: DataModality | None = None,
+        *, progress: t.Callable[[dict], None] | None = None,
     ) -> list[DatasetQAExample]:
         """
         One question per sampled chunk (num_samples chunks of the given modality, or of every modality
@@ -230,6 +231,12 @@ class DatasetStudyGenerator:
                 examples.append(self._example_from_row(row, kind, base_seed))
         stats.study_num_cached += len(cached_rows)
         pending = list(enumerate(study_samples))[len(cached_rows):]
+        def report(completed: int, phase: str) -> None:
+            if progress is not None:
+                progress({"phase": phase, "completed": completed, "total": len(study_samples),
+                          "accepted": len(examples), "cached": len(cached_rows),
+                          "rejected": stats.study_num_junk_skipped, "failed": stats.study_num_parse_failures})
+        report(len(cached_rows), "loading QA model" if pending else "QA complete")
         if not pending:
             return examples
         # vllm batches a whole conversation list inside one chat() call, so a
@@ -258,6 +265,7 @@ class DatasetStudyGenerator:
                     },
                 ])
             batch_start_time = time.time()
+            report(len(cached_rows) + batch_start, "generating QA")
             outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
             elapsed = time.time() - batch_start_time
             total_generation_time += elapsed
@@ -284,6 +292,7 @@ class DatasetStudyGenerator:
                 examples.append(self._example_from_row(row, kind, base_seed))
             if key is not None:
                 self.cache.append(key, new_rows)
+            report(len(cached_rows) + batch_start + len(batch), "generating QA")
             if reporting_interval <= 0:
                 print(
                     f"{self.dataset_id} - Study ({kind}): {batch_start + len(batch)}/{len(pending)} chunks, "
@@ -291,6 +300,7 @@ class DatasetStudyGenerator:
                 )
                 reporting_interval = max(1, len(pending) // 20)
             reporting_interval -= len(batch)
+        report(len(study_samples), "QA complete")
         return examples
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/dataset/dataset_utils.py">activation/dataset/dataset_utils.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/dataset/dataset_utils.py`

```diff
diff --git a/activation/dataset/dataset_utils.py b/activation/dataset/dataset_utils.py
index 40a34e0..5b2f233 100644
--- a/activation/dataset/dataset_utils.py
+++ b/activation/dataset/dataset_utils.py
@@ -69,19 +69,18 @@ def normalize_split_name(split: str) -> DataSplit:
 def message_text(message: dict) -> str:
     """The text of a message whose content is a string or a list of parts (activation_context parts render as a marker)."""
     content = message.get("content")
-    if content is None:
-        return ""
-    if isinstance(content, str):
-        return content
-    pieces = []
-    for part in content:
-        if not isinstance(part, dict):
-            pieces.append(str(part))
-        elif part.get("type") == "text":
-            pieces.append(part.get("text", ""))
-        elif part.get("type") == "activation_context":
-            pieces.append("[activation context]")
-    return "".join(pieces)
+    if isinstance(content, list):
+        pieces = []
+        for part in content:
+            if not isinstance(part, dict):
+                pieces.append(str(part))
+            elif part.get("type") == "text":
+                pieces.append(part.get("text", ""))
+            elif part.get("type") == "activation_context":
+                pieces.append("[activation context]")
+        content = "".join(pieces)
+    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
+    return "\n\n".join(piece for piece in (str(reasoning), str(content or "")) if piece)
 
 
 def render_message(message: dict) -> str:
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/dataset/loaders/trajectory_utils.py">activation/dataset/loaders/trajectory_utils.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/dataset/loaders/trajectory_utils.py`

```diff
diff --git a/activation/dataset/loaders/trajectory_utils.py b/activation/dataset/loaders/trajectory_utils.py
index 7e215da..f33a80a 100644
--- a/activation/dataset/loaders/trajectory_utils.py
+++ b/activation/dataset/loaders/trajectory_utils.py
@@ -14,7 +14,7 @@ and a list of tool definitions in the OpenAI function shape. `reformat_trajector
 into our messages: the source system prompt is dropped (a short domain prompt rides in
 `trajectory_kwargs["system_prompt"]`), tool names and argument keys are mapped onto our tools
 (`shell(script)`, `python(code)`; unknown tools keep their names and definitions), reasoning stays
-as plain text ahead of the reply (`reasoning="keep"`, since the math `cot` rows are nothing but
+in its separate field (`reasoning="keep"`, since the math `cot` rows are nothing but
 reasoning) or is dropped, calls become structured `tool_calls` (the shape `MessageRendering.assistant_message`
 produces), and every tool result is one `tool` message.
 """
@@ -25,6 +25,7 @@ import re
 import typing as t
 
 from ...agent.agent_utils import parse_tool_calls
+from ..dataset_utils import message_text
 
 TOOL_MAP: dict[str, tuple[str, dict[str, str]]] = {
     "bash": ("shell", {"command": "script"}),
@@ -206,9 +207,9 @@ def reformat_trajectory(
             calls = [map_call(call, tool_map) for call in message.get("tool_calls") or []]
             text = str(message.get("content") or "").strip()
             reasoning_text = (message.get("reasoning") or "").strip()
-            if reasoning == "keep" and reasoning_text:
-                text = f"{reasoning_text}\n\n{text}".strip() if text else reasoning_text
             out: dict = {"role": "assistant", "content": text}
+            if reasoning == "keep" and reasoning_text:
+                out["reasoning"] = reasoning_text
             if calls:
                 out["tool_calls"] = []
                 for call in calls:
@@ -253,7 +254,7 @@ def trajectory_chars(messages: list[dict]) -> int:
     """Characters of content and arguments, the cheap size measure loaders filter on."""
     total = 0
     for message in messages:
-        total += len(str(message.get("content") or ""))
+        total += len(message_text(message))
         for call in message.get("tool_calls") or []:
             total += len(json.dumps(call.get("function", call).get("arguments", {}), ensure_ascii=False))
     return total
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/harness/hf_utils.py">activation/harness/hf_utils.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/harness/hf_utils.py`

```diff
diff --git a/activation/harness/hf_utils.py b/activation/harness/hf_utils.py
index ab746aa..3171f76 100644
--- a/activation/harness/hf_utils.py
+++ b/activation/harness/hf_utils.py
@@ -3,6 +3,7 @@ HF-facing code for model configuration parsing, message formatting, etc.
 Stores overly detailed/specific logic to keep the rest of code cleaner.
 """
 from collections import Counter
+from contextlib import contextmanager
 import typing as t
 import torch
 import torch.nn.functional as F
@@ -43,6 +44,175 @@ _LINEAR_ATTENTION_NAMES = frozenset(
 _FFN_ONLY_NAMES = frozenset({"mlp", "moe"})
 
 
+@contextmanager
+def checkpoint_adapter_scope(model, adapter_context, enabled: bool | None = None):
+    """Bind each layer's recomputation to the adapter used for its original forward."""
+    previous = [(module, module.gradient_checkpointing, getattr(module, "_gradient_checkpointing_func", None))
+                for module in model.modules() if hasattr(module, "gradient_checkpointing")]
+    was_enabled = model.is_gradient_checkpointing
+    if enabled is not None and enabled != was_enabled:
+        if enabled:
+            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
+        else:
+            model.gradient_checkpointing_disable()
+    try:
+        for module in model.modules():
+            if not getattr(module, "gradient_checkpointing", False) or not hasattr(module, "_gradient_checkpointing_func"):
+                continue
+            original = module._gradient_checkpointing_func
+            def scoped(function, *args, _checkpoint=original, **kwargs):
+                return _checkpoint(function, *args, **(kwargs | {
+                    "use_reentrant": False, "context_fn": lambda: (adapter_context(), adapter_context())}))
+            module._gradient_checkpointing_func = scoped
+        yield
+    finally:
+        for module, checkpointed, original in previous:
+            module.gradient_checkpointing = checkpointed
+            if original is not None:
+                module._gradient_checkpointing_func = original
+            elif hasattr(module, "_gradient_checkpointing_func"):
+                del module._gradient_checkpointing_func
+
+
+
+# --------------------------------------------------------------------------------------- deep AC inputs (interventions)
+INTERVENTIONS_KWARG = "interventions"
+_HOOKS_ATTRIBUTE = "_activation_intervention_hooks"
+
+
+class Interventions(t.TypedDict, total=False):
+    """One logical sequence's deltas: `positions` index the unpadded physical row (prompt indices, sorted, unique); every
+    `layer_inputs[layer]` is [M, d_model], added to that decoder layer's input at those positions. None means no intervention."""
+    positions: list[int]
+    layer_inputs: dict[int, torch.Tensor]
+    row_rms: dict[int, torch.Tensor]     # written by the reader hooks: residual RMS at the row positions per intervened layer (0-d tensors; diagnostic)
+
+
+class BoundInterventions:
+    """The payloads of one decoder forward bound to that forward's cache offset (set once by the text-model pre-hook);
+    travels in the layer kwargs, so a checkpointed layer recomputes with exactly the same tensors."""
+
+    __slots__ = ("payloads", "offset")
+
+    def __init__(self, payloads: list[Interventions | None], offset: int) -> None:
+        self.payloads = payloads
+        self.offset = offset
+
+
+def text_decoder(model) -> torch.nn.Module:
+    """The text decoder that owns `.layers` (below a multimodal wrapper's `language_model`); never a vision tower."""
+    decoder = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, model.base_model_prefix)
+    for name in ("language_model", "text_model"):
+        inner = getattr(decoder, name, None)
+        if inner is not None and hasattr(inner, "layers"):
+            return inner
+    if not hasattr(decoder, "layers"):
+        raise ValueError(f"{type(decoder).__name__} has no decoder layers to intervene on")
+    return decoder
+
+
+def decoder_layers(model) -> torch.nn.ModuleList:
+    return text_decoder(model).layers
+
+
+def _cache_offset(kwargs: dict) -> int:
+    cache = kwargs.get("past_key_values")
+    if cache is None:
+        cache = kwargs.get("past_key_value")
+    if cache is None or not hasattr(cache, "get_seq_length"):
+        return 0
+    return int(cache.get_seq_length())
+
+
+def install_intervention_hooks(model) -> None:
+    """
+    Persistent, parameter-free hooks (installed once per model): the text model's pre-hook binds the forward's
+    `interventions=` list to the cache offset before any layer runs; every decoder layer's pre-hook takes the kwarg
+    out (so no attention or recurrence kernel ever sees it) and, when a payload names this layer, adds the delta
+    rows to the layer input out of place. A forward without the kwarg is untouched. The engine mapping is this same
+    operation: a per-request scatter-add on the hidden state entering the named layers, at the rows' prompt positions.
+    """
+    if getattr(model, _HOOKS_ATTRIBUTE, None) is not None:
+        return
+    text = text_decoder(model)
+    handles = []
+
+    def bind(module, args, kwargs):
+        payloads = kwargs.get(INTERVENTIONS_KWARG)
+        if payloads is None or isinstance(payloads, BoundInterventions):
+            return None
+        return args, {**kwargs, INTERVENTIONS_KWARG: BoundInterventions(list(payloads), _cache_offset(kwargs))}
+    handles.append(text.register_forward_pre_hook(bind, with_kwargs=True))
+
+    def layer_hook(index: int):
+        def hook(module, args, kwargs):
+            bound = kwargs.get(INTERVENTIONS_KWARG)
+            if bound is None:
+                return None
+            kwargs = {key: value for key, value in kwargs.items() if key != INTERVENTIONS_KWARG}
+            if not isinstance(bound, BoundInterventions):
+                raise TypeError("interventions must reach the decoder layers through the text model's forward")
+            hidden = args[0]
+            if hidden.shape[0] != len(bound.payloads):
+                raise ValueError(f"{len(bound.payloads)} intervention payloads for {hidden.shape[0]} physical rows")
+            length = hidden.shape[1]
+            output = None
+            for row, payload in enumerate(bound.payloads):
+                if not payload:
+                    continue
+                rows = payload["layer_inputs"].get(index)
+                if rows is None:
+                    continue
+                positions = torch.as_tensor(payload["positions"], device=hidden.device, dtype=torch.long)
+                if positions.numel() != rows.shape[0]:
+                    raise ValueError("positions and delta rows differ in count")
+                local = positions - bound.offset
+                keep = (local >= 0) & (local < length)
+                if not bool(keep.any()):
+                    continue
+                # Diagnostic for the producer: the residual RMS at this forward's row positions (the deltas' true denominator).
+                payload.setdefault("row_rms", {})[index] = hidden[row, local[keep]].detach().float().pow(2).mean(dim=-1).sqrt().mean()   # 0-d tensor: no host sync here
+                if output is None:
+                    output = hidden.clone()
+                output[row].index_add_(0, local[keep], rows[keep].to(device=hidden.device, dtype=hidden.dtype))
+            return ((output if output is not None else hidden, *args[1:]), kwargs)
+        return hook
+    for index, layer in enumerate(decoder_layers(model)):
+        handles.append(layer.register_forward_pre_hook(layer_hook(index), with_kwargs=True))
+    setattr(model, _HOOKS_ATTRIBUTE, handles)
+
+
+def generate_with_interventions(peft_model, *, inputs_embeds: torch.Tensor, interventions: list[Interventions | None], **generate_kwargs):
+    """
+    Greedy, single-sequence `generate` from embeddings whose prompt rows carry deltas. HF validates generate's kwargs
+    against the model signature (an `interventions=` kwarg is rejected), so the payload is bound for this call only by
+    wrapping the base model's `prepare_inputs_for_generation` (the one HF's loop calls every step under a PEFT wrapper
+    too; the wrapper keeps the original signature for HF's validation and is removed in `finally`). The layer hooks
+    map every step's cache offset, so the row deltas land on the prompt at prefill and on nothing afterwards.
+    """
+    import functools
+    if inputs_embeds.dim() != 3 or inputs_embeds.shape[0] != 1 or len(interventions) != 1:
+        raise ValueError("generate_with_interventions takes one sequence and one payload")
+    if generate_kwargs.get("num_beams", 1) != 1 or generate_kwargs.get("do_sample", False) or generate_kwargs.get("num_return_sequences", 1) != 1:
+        raise ValueError("generate_with_interventions supports greedy single-sequence decoding only")
+    base_model = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
+    install_intervention_hooks(base_model)
+    previous = base_model.__dict__.get("prepare_inputs_for_generation")
+    original = base_model.prepare_inputs_for_generation
+    @functools.wraps(original)
+    def bound(*args, **kwargs):
+        model_inputs = original(*args, **kwargs)
+        model_inputs[INTERVENTIONS_KWARG] = interventions
+        return model_inputs
+    base_model.prepare_inputs_for_generation = bound
+    try:
+        return peft_model.generate(inputs_embeds=inputs_embeds, **generate_kwargs)
+    finally:
+        if previous is None:
+            del base_model.__dict__["prepare_inputs_for_generation"]
+        else:
+            base_model.prepare_inputs_for_generation = previous
+
 
 def canonical_layer_type(raw_type: object) -> LayerType:
     normalized = str(raw_type).casefold()
@@ -159,10 +329,14 @@ def model_description_and_tokenizer_from_hf(
     else:
         eos_token = canonical_eos_token(hf_config, tokenizer)
     # Finalize.
+    num_heads = getattr(text_config, "num_attention_heads", None)
+    head_dim = getattr(text_config, "head_dim", None) or (text_config.hidden_size // num_heads if num_heads else None)
+    kv_heads = getattr(text_config, "num_key_value_heads", None) or num_heads
     description = ModelDescription(
         d_model=text_config.hidden_size,
         d_ff=getattr(text_config, "intermediate_size", None) or 4 * text_config.hidden_size,
         is_multimodal=is_multimodal,
+        d_kv=int(kv_heads * head_dim) if kv_heads and head_dim else None,
         dtype=dtype,
         layer_descriptions=layers,
         eos_token=eos_token,
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/harness/loaded_model.py">activation/harness/loaded_model.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/harness/loaded_model.py`

```diff
diff --git a/activation/harness/loaded_model.py b/activation/harness/loaded_model.py
index f497587..9a9b9a6 100644
--- a/activation/harness/loaded_model.py
+++ b/activation/harness/loaded_model.py
@@ -29,6 +29,7 @@ from .hf_utils import (
 from .vllm_wrapper import VLLMWrapper
 
 if t.TYPE_CHECKING:
+    from .hf_utils import Interventions
     from .runtime import HarnessRuntime
 
 @dataclass
@@ -487,11 +488,15 @@ class LoadedModel:
         attention_mask: torch.Tensor | None,
         position_ids: torch.Tensor|None = None,
         lora_name: str|None = None,
+        gradient_checkpointing: bool | None = None,
+        interventions: "list[Interventions | None] | None" = None,
         **kwargs,
     ) -> torch.Tensor:
         """
         Last hidden state [B, S, d_model] of the causal decoder without the language-model head.
         Extra keywords reach every decoder layer (e.g. `cu_seq_lens_q` for packed rows).
+        `interventions`: one payload (or None) per physical row of `inputs_embeds`, positions already in that
+        row's coordinates; the deltas are added to the named layers' inputs by the persistent hooks.
         With a LoRA name the module manager activates that adapter for this pass (PEFT injects the
         adapters into the base's own linear layers, so self.model is the LoRA'd module tree; the
         wrapper only routes and manages adapters); with None any injected adapters are disabled.
@@ -500,7 +505,14 @@ class LoadedModel:
         assert self.model is not None, f"{self.model_config.model_name} - Model is not loaded."
         model = self.model
         decoder = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, model.base_model_prefix)
-        with self.harness.module_manager.lora_context(self.model_config.model_name, lora_name):
+        from .hf_utils import INTERVENTIONS_KWARG, checkpoint_adapter_scope, install_intervention_hooks
+        if interventions is not None:
+            if len(interventions) != inputs_embeds.shape[0]:
+                raise ValueError(f"{len(interventions)} intervention payloads for {inputs_embeds.shape[0]} physical rows")
+            install_intervention_hooks(model)
+            kwargs = {**kwargs, INTERVENTIONS_KWARG: list(interventions)}
+        adapter_context = lambda: self.harness.module_manager.lora_context(self.model_config.model_name, lora_name)
+        with adapter_context(), checkpoint_adapter_scope(model, adapter_context, gradient_checkpointing):
             outputs = decoder(
                 inputs_embeds=inputs_embeds,
                 attention_mask=attention_mask,
@@ -532,4 +544,4 @@ class LoadedModel:
 
     def simple_vector_embed(self, text: str) -> torch.Tensor:
         """Simple function to test embeddings."""
-        return self.simple_vector_embed_many([text])[0]
\ No newline at end of file
+        return self.simple_vector_embed_many([text])[0]
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/harness/model_config.py">activation/harness/model_config.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/harness/model_config.py`

```diff
diff --git a/activation/harness/model_config.py b/activation/harness/model_config.py
index 9fc6f37..8a6c8e7 100644
--- a/activation/harness/model_config.py
+++ b/activation/harness/model_config.py
@@ -51,6 +51,8 @@ class ModelDescription:
     d_ff: int
     """Feed-forward width, for batch sizing."""
     is_multimodal: bool
+    d_kv: int | None = None
+    """num_key_value_heads x head_dim of the full-attention layers (the width of one token's K, and of its V); None when unknown."""
     layer_descriptions: list[LayerDescription] = field(default_factory=list)
     dtype: torch.dtype = torch.bfloat16
     eos_token: str | None = None
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/harness/module_manager.py">activation/harness/module_manager.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/harness/module_manager.py`

```diff
diff --git a/activation/harness/module_manager.py b/activation/harness/module_manager.py
index 587e4a0..463befc 100644
--- a/activation/harness/module_manager.py
+++ b/activation/harness/module_manager.py
@@ -194,28 +194,45 @@ class ModuleManager:
         """
         if lora_name is None:
             peft_model = self.peft_models.get(model_name)
-            if peft_model is None:
-                yield
-            else:
-                with peft_model.disable_adapter():
-                    yield
+        else:
+            lora_config = self.get_lora_config(lora_name)
+            assert lora_config.model_name == model_name, f"LoRA {lora_name!r} belongs to {lora_config.model_name!r}, not {model_name!r}."
+            peft_model = self.ensure_lora(lora_name)
+        if peft_model is None:
+            yield
             return
-        lora_config = self.get_lora_config(lora_name)
-        assert lora_config.model_name == model_name, f"LoRA {lora_name!r} belongs to {lora_config.model_name!r}, not {model_name!r}."
-        peft_model = self.ensure_lora(lora_name)
-        peft_model.set_adapter(lora_config.adapter_name)
-        yield
+        previous_adapter = peft_model.active_adapter
+        trainability = [(parameter, parameter.requires_grad) for parameter in peft_model.parameters()]
+        from peft.tuners.tuners_utils import BaseTunerLayer
+        layers = [(module, module.disable_adapters) for module in peft_model.modules() if isinstance(module, BaseTunerLayer)]
+        try:
+            if lora_name is not None:
+                peft_model.set_adapter(lora_config.adapter_name)
+            for layer, _ in layers:
+                layer.enable_adapters(lora_name is not None)
+            # PEFT routing changes trainability. Optimizer ownership, not the active adapter, controls it here.
+            for parameter, trainable in trainability:
+                parameter.requires_grad_(trainable)
+            yield
+        finally:
+            peft_model.set_adapter(previous_adapter)
+            for layer, disabled in layers:
+                layer.enable_adapters(not disabled)
+            for parameter, trainable in trainability:
+                parameter.requires_grad_(trainable)
 
     def lora_parameters(self, lora_name: str) -> list[nn.Parameter]:
         """The trainable parameters of that adapter only (injecting it if needed)."""
         peft_model = self.ensure_lora(lora_name)
         adapter_name = self.get_lora_config(lora_name).adapter_name
-        peft_model.set_adapter(adapter_name)
-        return [
+        parameters = [
             parameter
             for name, parameter in peft_model.named_parameters()
-            if f".{adapter_name}." in name and parameter.requires_grad
+            if f".{adapter_name}." in name
         ]
+        for parameter in parameters:
+            parameter.requires_grad_(True)
+        return parameters
 
     def free_lora(self, model_name: str, lora_name: str, checkpoint_path: str | None = None) -> None:
         """
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/tests/test_basic_ac_self_distillation.py">activation/tests/test_basic_ac_self_distillation.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/tests/test_basic_ac_self_distillation.py`

```diff
diff --git a/activation/tests/test_basic_ac_self_distillation.py b/activation/tests/test_basic_ac_self_distillation.py
new file mode 100644
index 0000000..2ec3f1e
--- /dev/null
+++ b/activation/tests/test_basic_ac_self_distillation.py
@@ -0,0 +1,290 @@
+"""Four real HotpotQA problems through rollout, AC self-distillation and AC rollout.
+
+The custom program retrieves evidence, asks one real solver subagent, then lets the
+main agent answer. No mocks, synthetic trajectories or gold answers enter the program.
+Each loss mode runs one epoch with two optimizer updates. Scores are printed as
+same-problem diagnostics; this small test does not assert a quality improvement.
+
+    uv run pytest activation/tests/test_basic_ac_self_distillation.py --gpu --slow -s
+
+Both loss modes use the paired-history, loss_kind and folder-local row APIs.
+"""
+import json
+import math
+import uuid
+from pathlib import Path
+
+import pytest
+import torch
+
+from activation.ac_model import (
+    ActivationContextModelConfig,
+    ActivationContextStudyGenerator,
+    ActivationContextTrainer,
+    ActivationContextTrainingConfig,
+    ActivationContextTrainingReporter,
+)
+from activation.agent import AgentConfig, AgenticProgram, AgentRunResult, RolloutReporter, SemanticSearchTool
+from activation.agent.agent_utils import load_ac_rows
+from activation.common.data_syncing import resolve_path
+from activation.dataset.loaders import HotpotQADataset
+from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
+
+SIDE_NAME, SIDE_ID = "self_distill_side", "Qwen/Qwen3.5-0.8B"
+READER_NAME, READER_ID = "self_distill_reader", "Qwen/Qwen3.5-4B"
+PROBLEMS = 4
+
+
+class RetrieveThenSolveProgram(AgenticProgram):
+    """Actual retrieval and a solver precede the main agent's own recorded answer."""
+    name = "retrieve_then_solve"
+
+    def execute(self) -> AgentRunResult:
+        question = self.agent.agent_config.dataset_task.task_datum["question"]
+        evidence = self.agent.run_tool("semantic_search", query=question, top_k=2)
+        if evidence.is_error:
+            raise RuntimeError(evidence.output)
+        solver = self.agent.run_subagent(
+            f"Find evidence for this question using semantic_search, then submit a short answer: {question}",
+            max_duration=60.0,
+        )
+        self.agent.augment_context([
+            evidence,
+            solver,
+            "Use the retrieved evidence and check the solver's suggestion. You may search again. "
+            "Submit only the short answer to the original question.",
+        ])
+        return self.agent.run()
+
+
+def _harness(ac_name: str, side_lora: str, reader_lora: str) -> HarnessRuntime:
+    harness = HarnessRuntime(HarnessRuntimeConfig(
+        model_configs={
+            SIDE_NAME: ModelConfig(SIDE_NAME, SIDE_ID),
+            READER_NAME: ModelConfig(READER_NAME, READER_ID),
+        },
+        agent_max_concurrent=2,
+    ))
+    harness.module_manager.register_lora(reader_lora, READER_NAME, rank=16)
+    harness.module_manager.register_ac_model(ActivationContextModelConfig(
+        ac_name, SIDE_NAME, side_lora, READER_NAME, reader_lora,
+        side_lora_rank=16,
+    ))
+    return harness
+
+
+def _tasks(harness: HarnessRuntime):
+    # Select by source difficulty and question length, never by model success.
+    # The bounded source prefix supplies ordinary distractor documents as well.
+    dataset = HotpotQADataset.load(harness, max_examples=128, split="train", seed=0)
+    harness.dataset_manager.register_dataset(dataset)
+    harness.dataset_manager.build_bm25_indexes()
+    easy = [task for task in dataset.scorable_tasks.values() if task.task_datum.get("level") == "easy"]
+    assert len(easy) >= PROBLEMS, f"expected four easy questions in the source prefix, found {len(easy)}"
+    tasks = sorted(easy, key=lambda task: (len(task.task_datum["question"]), task.task_id))[:PROBLEMS]
+    for task in tasks:
+        print(f"HotpotQA {task.task_id}: {task.task_datum['question']}")
+    return tasks
+
+
+def _configs(tasks, ac_name: str | None, reader_lora: str | None) -> list[AgentConfig]:
+    return [AgentConfig(
+        agent_name=f"hotpot_{index}",
+        system_prompt=(
+            "Answer the question from the Wikipedia corpus using semantic_search. "
+            "Check the evidence and call submit_answer with the short answer only."
+        ),
+        user_prompt=task.agent_prompt,
+        dataset_task=task,
+        model_name=READER_NAME,
+        lora_name=reader_lora,
+        ac_model_name=ac_name,
+        tools={"semantic_search": (SemanticSearchTool, {"dataset_id": task.dataset_id, "max_chars": 1200}), "subagent": None},
+        agentic_program=(RetrieveThenSolveProgram, {}),
+        metadata={"program": RetrieveThenSolveProgram.name, "benchmark": "hotpotqa"},
+        call_kwargs={"sampling_params": {"max_tokens": 1024, "temperature": 0.0}},
+        max_turns=4,
+        max_duration=120.0,
+        max_tool_errors=2,
+        compaction_threshold_tokens=16_384,
+        absolute_trajectory_cap=32_000,
+    ) for index, task in enumerate(tasks)]
+
+
+def _rollouts(harness, configs, caching_id: str, report_folder: Path, seed: int) -> list[AgentRunResult]:
+    reporter = RolloutReporter(str(report_folder), title=f"HotpotQA: {caching_id}")
+    try:
+        runs = harness.rollout_manager.perform_single_rollouts(
+            configs, seed=seed, caching_id=caching_id, perform_scoring=True, reporter=reporter,
+        )
+    finally:
+        reporter.finish()
+    assert len(runs) == PROBLEMS
+    for run in runs:
+        assert run.finish_reason != "error", run.score_feedback
+        assert len(run.subagent_results) == 1
+        assert run.subagent_results[0].finish_reason != "error", run.subagent_results[0].score_feedback
+        assert run.score is not None and math.isfinite(run.score)
+        print(f"{run.agent_config.agent_name}: score={run.score:.3f}, answer={run.answer!r}, "
+              f"finish={run.finish_reason}, rows={run.num_ac_rows}")
+    return runs
+
+
+def _segments(run: AgentRunResult):
+    for segment in [*run.compactions, run]:
+        yield segment
+        for child in segment.subagent_results:
+            yield from _segments(child)
+
+
+def _spans(runs: list[AgentRunResult]):
+    for root in runs:
+        for segment in _segments(root):
+            yield from segment.prompt_ac_spans
+            for step in segment.trajectory:
+                yield from step.get("ac_spans", [])
+
+
+def _assert_coverage(runs, items) -> None:
+    recorded = sum(
+        len(step.get("token_ids") or [])
+        for root in runs for segment in _segments(root) for step in segment.trajectory
+        if step["role"] == "assistant"
+    )
+    selected = sum(len(ids) for item in items for ids in item.assistant_token_ids)
+    assert selected == recorded > 0, (selected, recorded)
+    assert all(item.weight == 1.0 for item in items)
+
+
+def _assert_eval(summary, count: int) -> None:
+    assert summary["items"] == count > 0
+    assert summary["dropped_too_long"] == 0
+    assert summary["loss"] is not None and math.isfinite(summary["loss"])
+
+
+def _check_saved_rows(runs, caching_id: str, width: int) -> None:
+    folder = resolve_path(f"ROLLOUTS/{caching_id}", create=False)
+    assert (folder / "rollouts.jsonl").is_file()
+    assert (folder / "saved_tensors").is_dir()
+    spans = list(_spans(runs))
+    assert spans, "the AC rollouts must produce actual captured rows"
+    for span in spans:
+        assert "rows" not in span, "successfully saved results must release their tensor fields"
+        path = Path(span["row_path"])
+        assert path.is_absolute() and path.is_file()
+        assert path.parent == (folder / "saved_tensors").resolve()
+    rows = load_ac_rows(spans[0])
+    assert rows.device.type == "cpu" and not rows.requires_grad
+    assert rows.shape == (spans[0]["length"], width)
+    assert torch.isfinite(rows).all()
+
+
+def _release(harness, ac_name: str) -> None:
+    for loaded in harness.loaded_models.values():
+        loaded.engine_to_device(FREE_DEVICE)
+    ac_model = harness.module_manager.get_ac_model(ac_name)
+    ac_model.release()
+    for model_name, lora_name in (
+        (READER_NAME, ac_model.config.target_model_lora_name),
+        (SIDE_NAME, ac_model.config.base_side_model_lora_name),
+    ):
+        harness.module_manager.free_lora(model_name, lora_name)
+    for loaded in harness.loaded_models.values():
+        loaded.model_to_device(FREE_DEVICE)
+
+
+@pytest.mark.gpu
+@pytest.mark.slow
+@pytest.mark.parametrize("loss_kind", ["kl", "sft"])
+def test_basic_ac_self_distillation(loss_kind: str) -> None:
+    # A new namespace prevents cached rollouts/checkpoints from hiding this run.
+    run_id = f"hotpot_ac_{loss_kind}_{uuid.uuid4().hex[:10]}"
+    ac_name, side_lora, reader_lora = run_id, f"{run_id}_side", f"{run_id}_reader"
+    folder = resolve_path(f"AC_SELF_DISTILLATION_TEST/{run_id}")
+    harness = _harness(ac_name, side_lora, reader_lora)
+    training_reporter = None
+    try:
+        tasks = _tasks(harness)
+        before_configs = _configs(tasks, ac_name=None, reader_lora=None)
+        before_runs = _rollouts(harness, before_configs, f"{run_id}_text", folder / "text_rollouts", seed=0)
+        assert all(run.num_ac_rows == 0 for run in before_runs)
+        assert all(len(run.injected_input) >= 2 for run in before_runs)
+
+        config = ActivationContextTrainingConfig(
+            loss_kind=loss_kind,
+            num_epochs=1,
+            updates_per_epoch=2,
+            reporting_interval=1.0,
+            completion_samples=1,
+            completion_max_new_tokens=32,
+            logits_chunk_tokens=128,
+            seed=0,
+        )
+        study = ActivationContextStudyGenerator(harness, ac_name)
+        # Explicitly include all four for this execution example in both modes.
+        # The SFT default score_1 selection is tested privately; scores stay real.
+        items = study.items_from_run_results(before_runs, loss_kind=loss_kind, selection="all")
+        _assert_coverage(before_runs, items)
+        assert len(items) >= PROBLEMS
+        trainer = ActivationContextTrainer(harness, config)
+        training_reporter = ActivationContextTrainingReporter(str(folder / "training"), f"HotpotQA self-distillation: {loss_kind}")
+        # Before training the reader is the teacher train() will store, so this evaluation takes its targets from it.
+        before = trainer.eval(ac_name, items, reporter=training_reporter, reporting_name="before", teacher="current")
+        _assert_eval(before, len(items))
+        stats = trainer.train(ac_name, items, reporting_data=items, reporter=training_reporter)
+        after = stats.reporting
+        _assert_eval(after, len(items))
+        assert stats.steps == 2 and len(stats.epochs) == 1
+        assert stats.loss_kind == loss_kind and all(math.isfinite(value) for value in stats.loss)
+        assert stats.checkpoint_path and (Path(stats.checkpoint_path) / "ac_modules.pt").is_file()
+        assert stats.target_lora_checkpoint_path and Path(stats.target_lora_checkpoint_path).is_dir()
+        if loss_kind == "sft":
+            assert stats.teacher_tokens == 0 and all(row["teacher_tokens"] == 0 for row in after["per_item"])
+            assert after["kl"] is None and after["agreement"] is None
+        print(json.dumps(stats.summarize(), indent=2))
+
+        ac_model = harness.module_manager.get_ac_model(ac_name)
+        completed = ac_model.load_completed_epoch()
+        assert resolve_path(completed["ac_checkpoint"], create=False).resolve() == Path(stats.checkpoint_path).resolve()
+        reloaded_trainer = ActivationContextTrainer(harness, config)
+        if loss_kind == "kl":   # a fresh trainer scores against the saved training-start targets, not against the trained reader
+            assert reloaded_trainer.load_teacher_targets(Path(stats.checkpoint_path) / "teacher_targets.pt") == len(items)
+        reloaded = reloaded_trainer.eval(ac_name, items)
+        _assert_eval(reloaded, len(items))
+        assert math.isclose(reloaded["loss"], after["loss"], rel_tol=5e-3, abs_tol=5e-3)
+        harness.module_manager.exchange_lora(READER_NAME, reader_lora)
+
+        ac_configs = _configs(tasks, ac_name=ac_name, reader_lora=reader_lora)
+        ac_cache = f"{run_id}_ac"
+        ac_runs = _rollouts(harness, ac_configs, ac_cache, folder / "ac_rollouts", seed=1)
+        assert all(run.ac_model_name == ac_name and run.lora_name == reader_lora for run in ac_runs)
+        assert all(len(run.prompt_ac_spans) >= 2 for run in ac_runs), "program should inject multiple actual AC parts"
+        _check_saved_rows(ac_runs, ac_cache, ac_model.d_target)
+
+        cached = harness.rollout_manager.perform_single_rollouts(ac_configs, seed=1, caching_id=ac_cache, perform_scoring=True)
+        assert len(cached) == PROBLEMS
+        assert [run.answer for run in cached] == [run.answer for run in ac_runs]
+        _check_saved_rows(cached, ac_cache, ac_model.d_target)
+        # Native AC histories also convert to differentiable training data.
+        ac_items = study.items_from_run_results(cached, loss_kind=loss_kind, selection="all")
+        _assert_coverage(cached, ac_items)
+
+        summary = {
+            "loss_kind": loss_kind,
+            "task_ids": [task.task_id for task in tasks],
+            "text_scores": [run.score for run in before_runs],
+            "ac_scores": [run.score for run in ac_runs],
+            "training_items": len(items),
+            "ac_history_items": len(ac_items),
+            "before_loss": before["loss"],
+            "after_loss": after["loss"],
+            "reloaded_loss": reloaded["loss"],
+            "checkpoint": completed,
+            "quality_scope": "same four training questions; diagnostic only",
+        }
+        (folder / "summary.json").write_text(json.dumps(summary, indent=2))
+        print(json.dumps(summary, indent=2))
+    finally:
+        if training_reporter is not None:
+            training_reporter.finish()
+        _release(harness, ac_name)
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/tests/test_basic_agent_ac.py">activation/tests/test_basic_agent_ac.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/tests/test_basic_agent_ac.py`

```diff
diff --git a/activation/tests/test_basic_agent_ac.py b/activation/tests/test_basic_agent_ac.py
index 743dcb0..fcc98ec 100644
--- a/activation/tests/test_basic_agent_ac.py
+++ b/activation/tests/test_basic_agent_ac.py
@@ -175,8 +175,9 @@ def test_ac_agent_channels_simulated():
         _probe_engine(agent, "search")
 
         # --- resume: the serialized record rebuilds the same prefix, spans and rows.
-        data = json.loads(json.dumps(agent.run_results.serialize()))
-        resumed = Agent.resume_from_run_result(harness, AgentRunResult.deserialize(data, harness))
+        base_dir = resolve_path("AGENT_AC_TEST/roundtrip")
+        data = json.loads(json.dumps(agent.run_results.serialize(base_dir=base_dir)))
+        resumed = Agent.resume_from_run_result(harness, AgentRunResult.deserialize(data, harness, base_dir=base_dir))
         assert resumed.prefix == agent.prefix and resumed.spans == agent.spans
         assert [tuple(r.shape) for r in resumed.rows] == [tuple(r.shape) for r in agent.rows]
         assert all(torch.equal(a, b) for a, b in zip(resumed.rows, agent.rows))
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="/workspace/activation/tests/test_basic_agent_ac_training.py">activation/tests/test_basic_agent_ac_training.py</a></span><span class="card-oneliner">Current workspace versus /source</span><span class="card-badge">Snapshot</span></summary>

`activation/tests/test_basic_agent_ac_training.py`

```diff
diff --git a/activation/tests/test_basic_agent_ac_training.py b/activation/tests/test_basic_agent_ac_training.py
index 5408b9d..7c196b7 100644
--- a/activation/tests/test_basic_agent_ac_training.py
+++ b/activation/tests/test_basic_agent_ac_training.py
@@ -32,7 +32,7 @@ TARGET_NAME, TARGET_ID = "qwen3.5-4b", "Qwen/Qwen3.5-4B"
 AC_NAME, SIDE_LORA, TARGET_LORA = "ac_dev", "ac_side", "ac_target"
 QA_MODEL_NAME = "RedHatAI/Qwen3.5-9B-FP8-dynamic"
 ITEMS, HELD = 48, 8
-CACHING_ID = "ac_training_test_3a1_9b"
+CACHING_ID = "ac_training_test_codesign_v3_9b"
 
 
 def _harness(with_study: bool) -> HarnessRuntime:
@@ -46,7 +46,16 @@ def _harness(with_study: bool) -> HarnessRuntime:
 
 
 def _trainer(harness: HarnessRuntime) -> ActivationContextTrainer:
-    return ActivationContextTrainer(harness, ActivationContextTrainingConfig(examples_per_update=8, num_epochs=2))
+    return ActivationContextTrainer(harness, ActivationContextTrainingConfig(loss_kind="kl", examples_per_update=8, num_epochs=2))
+
+
+def _print_item(item: ActivationContextTrainingItem, tokenizer) -> None:
+    messages = item.in_context_messages or item.ac_messages
+    question = next((message.get("content", "") for message in reversed(messages) if message.get("role") == "user"), "")
+    if isinstance(question, list):
+        question = "".join(piece.get("text", "") for piece in question if isinstance(piece, dict))
+    answer = "\n".join(tokenizer.decode(ids, skip_special_tokens=False) for ids in item.assistant_token_ids)
+    print(f"  [{item.item_id}] Q: {str(question)[-160:]!r} A: {answer[:80]!r}; assistant spans={len(item.assistant_token_ids)}")
 
 
 def _run(harness: HarnessRuntime, items: list[ActivationContextTrainingItem], name: str) -> None:
@@ -59,9 +68,12 @@ def _run(harness: HarnessRuntime, items: list[ActivationContextTrainingItem], na
     trainer = _trainer(harness)
     reporter = ActivationContextTrainingReporter(str(resolve_path(f"AC_TRAINING_TEST/{name}")), f"AC training test: {name}")
     try:
-        before = trainer.eval(AC_NAME, held, release=False, reporter=reporter, reference_name="untrained_ac")
-        no_context = trainer.eval(AC_NAME, references, release=False, reporter=reporter, reporting_name="initial_no_context", reference_name="no_context")
-        stats = trainer.train(AC_NAME, train, reporting_data=held, reporter=reporter)
+        # Before training the reader is the teacher train() will store, so these two evaluations may take targets from it.
+        before = trainer.eval(AC_NAME, held, release=False, reporter=reporter, reference_name="untrained_ac", teacher="current")
+        no_context = trainer.eval(AC_NAME, references, release=False, reporter=reporter, reporting_name="initial_no_context", reference_name="no_context", teacher="current")
+        # The no-context items ride along as validation data so the call stores their targets; the evaluation after
+        # training then compares the no-context student to the same training-start teacher as everything else.
+        stats = trainer.train(AC_NAME, train, reporting_data=held, validation_data=references, reporter=reporter)
         after = stats.reporting
         matched_no_context = trainer.eval(AC_NAME, references, reporter=reporter, reporting_name="final_no_context")
         assert len(stats.epochs) == 2
@@ -98,7 +110,10 @@ def test_ac_compaction() -> None:
     math = NemotronMathDataset.load(harness, max_examples=ITEMS, subset="tir", excluded_problems=aime_problem_statements(), min_chars=40_000, min_tool_calls=1)
     generator = ActivationContextStudyGenerator(harness, AC_NAME)
     items = generator.generate_compaction_samples(swe.dataset_id, ITEMS // 2, seed=0) + generator.generate_compaction_samples(math.dataset_id, ITEMS // 2, seed=0)
-    print(f"\n{len(items)} compaction items; depths {sorted({item.info['depth'] for item in items})}; mid-turn cuts {sum(bool(item.teacher_partial_text) for item in items)}")
+    print(f"\n{len(items)} compaction items; depths {sorted({item.info['depth'] for item in items})}; "
+          f"assistant spans {sum(len(item.assistant_token_ids) for item in items)}; "
+          f"assistant targets {sum(len(ids) for item in items for ids in item.assistant_token_ids)}")
+    assert all(item.assistant_token_ids and all(item.assistant_token_ids) for item in items)
     _run(harness, items, "compaction")
 
 
@@ -115,7 +130,7 @@ def test_ac_trajectory_qa() -> None:
     harness.loaded_models[QA_MODEL_NAME].engine_to_device(FREE_DEVICE)
     print(f"\n{len(items)} traj_qa items; distractors {sorted({item.info['distractors'] for item in items})}")
     for item in items[:3]:
-        print(f"  [{item.item_id}] Q: {item.ac_prefix[-1]['content'][-1]['text'][-160:]!r} A: {item.completion_text[:80]!r}")
+        _print_item(item, harness.loaded_models[TARGET_NAME].tokenizer)
     _run(harness, items, "traj_qa")
 
 
@@ -130,5 +145,5 @@ def test_ac_rag_qa() -> None:
     harness.loaded_models[QA_MODEL_NAME].engine_to_device(FREE_DEVICE)
     print(f"\n{len(items)} rag_qa items; passages per item {sorted({item.info['passages'] for item in items})}")
     for item in items[:3]:
-        print(f"  [{item.item_id}] Q: {item.in_context_prefix[-1]['content'][-160:]!r} A: {item.completion_text[:80]!r}")
+        _print_item(item, harness.loaded_models[TARGET_NAME].tokenizer)
     _run(harness, items, "rag_qa")
```

</details>
