"""
HF-facing code for model configuration parsing, message formatting, etc.
Stores overly detailed/specific logic to keep the rest of code cleaner.
"""
from collections import Counter
from contextlib import contextmanager
import typing as t
import torch
import torch.nn.functional as F
from .model_config import (
    LayerType,
    ModelDescription,
    LayerDescription,
    ModelConfig,
    EmbeddingReadoutType,
)
from transformers import (
    AutoConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    PreTrainedConfig,
    AutoTokenizer,
    AutoProcessor,
    ProcessorMixin,
)

if t.TYPE_CHECKING:
    import peft
    from .module_manager import LoraConfig


TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SOURCE_DEVICE = "cpu"
FREE_DEVICE = "disk"
SUPPORTS_FP4 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)

_FULL_ATTENTION_NAMES = frozenset({"attention", "full_attention", "global_attention"})
_WINDOW_ATTENTION_NAMES = frozenset(
    {"local_attention", "sliding_attention", "window_attention"}
)
_LINEAR_ATTENTION_NAMES = frozenset(
    {"gated_delta_net", "linear_attention", "mamba", "recurrent", "ssm"}
)
_FFN_ONLY_NAMES = frozenset({"mlp", "moe"})


@contextmanager
def checkpoint_adapter_scope(model, adapter_context, enabled: bool | None = None):
    """Bind each layer's recomputation to the adapter used for its original forward."""
    previous = [(module, module.gradient_checkpointing, getattr(module, "_gradient_checkpointing_func", None))
                for module in model.modules() if hasattr(module, "gradient_checkpointing")]
    was_enabled = model.is_gradient_checkpointing
    if enabled is not None and enabled != was_enabled:
        if enabled:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
    try:
        for module in model.modules():
            if not getattr(module, "gradient_checkpointing", False) or not hasattr(module, "_gradient_checkpointing_func"):
                continue
            original = module._gradient_checkpointing_func
            def scoped(function, *args, _checkpoint=original, **kwargs):
                return _checkpoint(function, *args, **(kwargs | {
                    "use_reentrant": False, "context_fn": lambda: (adapter_context(), adapter_context())}))
            module._gradient_checkpointing_func = scoped
        yield
    finally:
        for module, checkpointed, original in previous:
            module.gradient_checkpointing = checkpointed
            if original is not None:
                module._gradient_checkpointing_func = original
            elif hasattr(module, "_gradient_checkpointing_func"):
                del module._gradient_checkpointing_func



# --------------------------------------------------------------------------------------- deep AC inputs (interventions)
INTERVENTIONS_KWARG = "interventions"
_HOOKS_ATTRIBUTE = "_activation_intervention_hooks"


class Interventions(t.TypedDict, total=False):
    """One logical sequence's deltas: `positions` index the unpadded physical row (prompt indices, sorted, unique);
    every `layer_inputs[layer]` is [M, d_model] added to that decoder layer's input; every `kv_inputs[layer]` is a
    (dk, dv) pair of [M, num_kv_heads x head_dim] added to that full-attention layer's keys (after k_norm, before RoPE)
    and values (after v_proj) at the same positions; every `kv_slots[layer]` is (k, v, keep_k, keep_v): the row slices of
    that layer's keys and values are REPLACED by keep x the reader's own value + the side-computed k / v ([M, num_kv_heads x
    head_dim] each; the keeps are 0-d; the K/V transfer targets send the side model's own K/V this way). `cross_inputs` (the cross-attention ceiling) carries the side model's final states of
    every passage token (`passage` [N, d_side]), the reader position from which each token may be read (`visible_from`, N
    entries: the start of its part's row run) and the per-layer blocks; every target position from the first visible one on
    receives `block(hidden[position], passage)` added to that layer's input. Producers' scales are already applied; None
    means no intervention."""
    positions: list[int]
    layer_inputs: dict[int, torch.Tensor]
    kv_inputs: dict[int, tuple[torch.Tensor, torch.Tensor]]
    kv_slots: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    cross_inputs: dict                   # {"passage": Tensor [N, d_side], "visible_from": list[int], "blocks": {layer: module}}
    row_rms: dict[int, torch.Tensor]     # written by the reader hooks: residual RMS at the row positions per intervened layer (0-d tensors; diagnostic)
    row_kv_rms: dict[int, list[torch.Tensor]]   # K/V payloads: [K RMS, V RMS] of the reader's own K/V at the row positions per layer (0-d tensors; diagnostic)
    slot_kv_rms: dict[int, list[torch.Tensor]]  # K/V slots: [K RMS, V RMS] of the written (replacement) K/V at the row positions per layer (0-d tensors; diagnostic)
    cross_rms: dict[int, list[torch.Tensor]]    # cross-attention: [delta RMS, residual RMS] over the target positions written per layer (0-d tensors; diagnostic)


_KV_BOUND_ATTRIBUTE = "_activation_kv_bound"


class BoundInterventions:
    """The payloads of one decoder forward bound to that forward's cache offset (set once by the text-model pre-hook);
    travels in the layer kwargs, so a checkpointed layer recomputes with exactly the same tensors."""

    __slots__ = ("payloads", "offset")

    def __init__(self, payloads: list[Interventions | None], offset: int) -> None:
        self.payloads = payloads
        self.offset = offset


def text_decoder(model) -> torch.nn.Module:
    """The text decoder that owns `.layers` (below a multimodal wrapper's `language_model`); never a vision tower."""
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, model.base_model_prefix)
    for name in ("language_model", "text_model"):
        inner = getattr(decoder, name, None)
        if inner is not None and hasattr(inner, "layers"):
            return inner
    if not hasattr(decoder, "layers"):
        raise ValueError(f"{type(decoder).__name__} has no decoder layers to intervene on")
    return decoder


def decoder_layers(model) -> torch.nn.ModuleList:
    return text_decoder(model).layers


def _cache_offset(kwargs: dict) -> int:
    cache = kwargs.get("past_key_values")
    if cache is None:
        cache = kwargs.get("past_key_value")
    if cache is None or not hasattr(cache, "get_seq_length"):
        return 0
    return int(cache.get_seq_length())


def install_intervention_hooks(model) -> None:
    """
    Persistent, parameter-free hooks (installed once per model): the text model's pre-hook binds the forward's
    `interventions=` list to the cache offset before any layer runs; every decoder layer's pre-hook takes the kwarg
    out (so no attention or recurrence kernel ever sees it) and, when a payload names this layer, adds the delta
    rows to the layer input out of place. K/V payloads (`kv_inputs`) are stashed on the layer for the duration of
    its forward (the recompute of a checkpointed layer re-runs the pre-hook, so it stashes the same tensors again)
    and added, out of place, by forward hooks on `self_attn.k_norm` (keys after the norm, before RoPE) and
    `self_attn.v_proj` (values) at the same positions; K/V slot payloads (`kv_slots`) take the same route but REPLACE the
    row slices (keep x the reader's own value + the side-computed content). Cross-attention payloads (`cross_inputs`) run
    the layer's block on the target positions inside the pre-hook (the recompute of a checkpointed layer runs it again on the
    same tensors). A forward without the kwarg is untouched.
    """
    if getattr(model, _HOOKS_ATTRIBUTE, None) is not None:
        return
    text = text_decoder(model)
    handles = []

    def add_rows(output: torch.Tensor, bound: "BoundInterventions", index: int, part: int) -> torch.Tensor | None:
        """`output` [B, S, ...] with the rows' K (part 0) or V (part 1) deltas of layer `index` added at positions - offset, or, for a
        K/V slot payload, the row slices replaced by keep x own + contribution; None when nothing lands."""
        length = output.shape[1]
        result = None
        rms_of = lambda values: values.reshape(values.shape[0], -1).pow(2).mean(dim=-1).sqrt().mean()   # 0-d tensor: no host sync here
        for row, payload in enumerate(bound.payloads):
            if not payload:
                continue
            pair = (payload.get("kv_inputs") or {}).get(index)
            slot = (payload.get("kv_slots") or {}).get(index)
            if pair is None and slot is None:
                continue
            if pair is not None and slot is not None:
                raise ValueError("a payload carries both K/V deltas and K/V slots for one layer")
            rows = (pair if pair is not None else slot)[part]
            positions = torch.as_tensor(payload["positions"], device=output.device, dtype=torch.long)
            if positions.numel() != rows.shape[0]:
                raise ValueError("positions and K/V delta rows differ in count")
            local = positions - bound.offset
            keep = (local >= 0) & (local < length)
            if not bool(keep.any()):
                continue
            # Diagnostic for the producer: the reader's own K (part 0) or V (part 1) RMS at this forward's row positions (the true denominator).
            payload.setdefault("row_kv_rms", {}).setdefault(index, [None, None])[part] = rms_of(output[row, local[keep]].detach().float())
            if result is None:
                result = output.clone()
            contribution = rows[keep].to(device=output.device, dtype=output.dtype).reshape(-1, *output.shape[2:])
            if pair is not None:
                result[row].index_add_(0, local[keep], contribution)
            else:                                                             # replacement: keep x the reader's own value + the side-computed slot content
                # Mixed in fp32 and cast once: in bf16 a keep of 1 - takeover rounds to 1.0 below a takeover of 0.4 % and then moves in
                # 0.8 % steps while the takeover's gradient is the exact linearisation; at init (keep 1, content 0) either precision is exact.
                own = output[row, local[keep]]
                written = (own.float() * slot[2 + part].to(device=output.device, dtype=torch.float32) + contribution.float()).to(output.dtype)
                result[row, local[keep]] = written
                payload.setdefault("slot_kv_rms", {}).setdefault(index, [None, None])[part] = rms_of(written.detach().float())
        return result

    def bind(module, args, kwargs):
        payloads = kwargs.get(INTERVENTIONS_KWARG)
        if payloads is None or isinstance(payloads, BoundInterventions):
            return None
        return args, {**kwargs, INTERVENTIONS_KWARG: BoundInterventions(list(payloads), _cache_offset(kwargs))}
    handles.append(text.register_forward_pre_hook(bind, with_kwargs=True))

    def layer_hook(index: int):
        def hook(module, args, kwargs):
            bound = kwargs.get(INTERVENTIONS_KWARG)
            if hasattr(module, _KV_BOUND_ATTRIBUTE):
                setattr(module, _KV_BOUND_ATTRIBUTE, None)      # every layer forward starts clean: an early-stopped checkpoint recompute never reaches the clearing hook
            if bound is None:
                return None
            kwargs = {key: value for key, value in kwargs.items() if key != INTERVENTIONS_KWARG}
            if not isinstance(bound, BoundInterventions):
                raise TypeError("interventions must reach the decoder layers through the text model's forward")
            hidden = args[0]
            if hidden.shape[0] != len(bound.payloads):
                raise ValueError(f"{len(bound.payloads)} intervention payloads for {hidden.shape[0]} physical rows")
            length = hidden.shape[1]
            output = None
            kv_here = False
            for row, payload in enumerate(bound.payloads):
                if not payload:
                    continue
                rows = payload["layer_inputs"].get(index)
                has_kv = (payload.get("kv_inputs") or {}).get(index) is not None or (payload.get("kv_slots") or {}).get(index) is not None
                cross = payload.get("cross_inputs")
                block = (cross or {}).get("blocks", {}).get(index)
                if rows is None and not has_kv and block is None:
                    continue
                kv_here = kv_here or has_kv
                positions = torch.as_tensor(payload["positions"], device=hidden.device, dtype=torch.long)
                if rows is not None and positions.numel() != rows.shape[0]:
                    raise ValueError("positions and delta rows differ in count")
                local = positions - bound.offset
                keep = (local >= 0) & (local < length)
                if bool(keep.any()):
                    # Diagnostic for the producer: the residual RMS at this forward's row positions (the deltas' true denominator).
                    payload.setdefault("row_rms", {})[index] = hidden[row, local[keep]].detach().float().pow(2).mean(dim=-1).sqrt().mean()   # 0-d tensor: no host sync here
                    if rows is not None:
                        if output is None:
                            output = hidden.clone()
                        output[row].index_add_(0, local[keep], rows[keep].to(device=hidden.device, dtype=hidden.dtype))
                if block is not None:
                    # Cross-attention ceiling: every target position from the first visible passage token on (this chunk's share of them,
                    # decode steps included) queries the passage states it may see; the block's projected attention is added to the input.
                    visible_from = list(cross["visible_from"])                         # plain ints: the first visible position needs no device sync
                    passage = cross["passage"]
                    if len(visible_from) != passage.shape[0]:
                        raise ValueError("visible_from and passage states differ in count")
                    visible = torch.as_tensor(visible_from, device=hidden.device, dtype=torch.long)
                    start = max(0, min(visible_from) - bound.offset) if visible_from else length
                    if start < length:
                        absolute = torch.arange(start, length, device=hidden.device) + bound.offset
                        queries = hidden[row, start:length]
                        delta = block(queries, passage.to(device=hidden.device), absolute[:, None] >= visible[None, :])
                        payload.setdefault("cross_rms", {})[index] = [delta.detach().float().pow(2).mean(dim=-1).sqrt().mean(),
                                                                      queries.detach().float().pow(2).mean(dim=-1).sqrt().mean()]
                        if output is None:
                            output = hidden.clone()
                        output[row, start:length] = output[row, start:length] + delta.to(dtype=hidden.dtype)
            if kv_here:
                if getattr(module, "self_attn", None) is None or not hasattr(module.self_attn, "k_norm"):
                    raise ValueError(f"decoder layer {index} has no full attention (k_norm) to receive K/V deltas")
                setattr(module, _KV_BOUND_ATTRIBUTE, bound)
            return ((output if output is not None else hidden, *args[1:]), kwargs)
        return hook

    def clear_kv(module, args, kwargs, output):
        if getattr(module, _KV_BOUND_ATTRIBUTE, None) is not None:
            setattr(module, _KV_BOUND_ATTRIBUTE, None)

    def kv_hook(layer, index: int, part: int):
        def hook(module, args, output):
            bound = getattr(layer, _KV_BOUND_ATTRIBUTE, None)
            if bound is None:
                return None
            return add_rows(output, bound, index, part)
        return hook
    for index, layer in enumerate(decoder_layers(model)):
        handles.append(layer.register_forward_pre_hook(layer_hook(index), with_kwargs=True))
        attention = getattr(layer, "self_attn", None)
        if attention is not None and hasattr(attention, "k_norm") and hasattr(attention, "v_proj"):
            setattr(layer, _KV_BOUND_ATTRIBUTE, None)
            handles.append(layer.register_forward_hook(clear_kv, with_kwargs=True))
            handles.append(attention.k_norm.register_forward_hook(kv_hook(layer, index, 0)))
            handles.append(attention.v_proj.register_forward_hook(kv_hook(layer, index, 1)))
    setattr(model, _HOOKS_ATTRIBUTE, handles)


@contextmanager
def capture_layer_input_rms(model, layers: t.Iterable[int], positions: t.Sequence[int] | None = None):
    """Records, per listed layer, the residual RMS of every forward's layer input into the yielded dict: the median over
    positions (all rows), the first token excluded when there is more than one (its residual is an outlier that would
    dominate a mean and is never a row position). With `positions` (row positions of every row of the forward), the mean
    over those positions of the per-position RMS instead: the same quantity the intervention hooks write as `row_rms`."""
    record: dict[int, list[float]] = {int(layer): [] for layer in layers}
    handles = []
    index_positions = None if positions is None else [int(position) for position in positions]
    def hook(index: int):
        def capture(module, args, kwargs):
            hidden = args[0].detach().float()
            if index_positions is not None:
                record[index].append(float(hidden[:, index_positions].pow(2).mean(dim=-1).sqrt().mean()))
                return
            per_position = hidden.pow(2).mean(dim=-1).sqrt()      # [B, S]
            if per_position.shape[1] > 1:
                per_position = per_position[:, 1:]
            record[index].append(float(per_position.reshape(-1).median()))
        return capture
    all_layers = decoder_layers(model)
    for index in record:
        handles.append(all_layers[index].register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_kv_rms(model, layers: t.Iterable[int], positions: t.Sequence[int] | None = None):
    """Records, per listed full-attention layer, (K RMS, V RMS) of every forward — K after k_norm and before RoPE, V after v_proj,
    each flattened to num_kv_heads x head_dim per position, averaged over rows and positions — into the yielded dict.
    With `positions`, averaged over those positions only (the row positions: what the K/V hooks write as `row_kv_rms`)."""
    record: dict[int, list[tuple[float, float]]] = {int(layer): [] for layer in layers}
    pending: dict[int, list[float | None]] = {index: [None, None] for index in record}
    handles = []
    index_positions = None if positions is None else [int(position) for position in positions]
    def hook(index: int, part: int):
        def capture(module, args, output):
            flat = output.detach().float().reshape(*output.shape[:2], -1)
            if index_positions is not None:
                flat = flat[:, index_positions]
            pending[index][part] = float(flat.pow(2).mean(dim=-1).sqrt().mean())
            if pending[index][0] is not None and pending[index][1] is not None:
                record[index].append((pending[index][0], pending[index][1]))
                pending[index] = [None, None]
        return capture
    all_layers = decoder_layers(model)
    for index in record:
        attention = getattr(all_layers[index], "self_attn", None)
        if attention is None or not hasattr(attention, "k_norm"):
            raise ValueError(f"decoder layer {index} has no full attention to calibrate K/V deltas against")
        handles.append(attention.k_norm.register_forward_hook(hook(index, 0)))
        handles.append(attention.v_proj.register_forward_hook(hook(index, 1)))
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_kv_rows(model, layers: t.Iterable[int], spans: list[tuple[int, int]]):
    """Captures, per listed full-attention decoder layer, that layer's pre-RoPE keys (the output of `self_attn.k_norm`) and values (the
    output of `self_attn.v_proj`) at `spans[row] = (start, end)` of every physical row of the next forward, each flattened to
    [end - start, num_kv_heads x head_dim] in the model's dtype (graph-attached slices, not copies): the yielded dict maps layer ->
    (keys per row, values per row). One capture per forward; a non-reentrant checkpoint recompute re-runs the layer without these hooks,
    which save no tensors of their own, so the captured slices' backward finds the recomputed values in place."""
    record: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {int(layer): ([], []) for layer in layers}
    handles = []
    def hook(index: int, part: int):
        def capture(module, args, output):
            store = record[index][part]
            if store:
                return
            store.extend(output[row, start:end].reshape(end - start, -1) for row, (start, end) in enumerate(spans))
        return capture
    all_layers = decoder_layers(model)
    for index in record:
        attention = getattr(all_layers[index], "self_attn", None)
        if attention is None or not hasattr(attention, "k_norm") or not hasattr(attention, "v_proj"):
            raise ValueError(f"decoder layer {index} has no full attention (k_norm / v_proj) to capture K/V from")
        handles.append(attention.k_norm.register_forward_hook(hook(index, 0)))
        handles.append(attention.v_proj.register_forward_hook(hook(index, 1)))
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_layer_input_rows(model, layers: t.Iterable[int], spans: list[tuple[int, int]]):
    """Captures, per listed decoder layer, that layer's input hidden state at `spans[row] = (start, end)` of every physical
    row of the next forward (a graph-attached slice, not a copy): the yielded dict maps layer -> list of [end - start, d] per row.
    Cheap and flat: only the row slices are held, never the whole hidden-state tuple."""
    record: dict[int, list[torch.Tensor]] = {int(layer): [] for layer in layers}
    handles = []
    def hook(index: int):
        def capture(module, args, kwargs):
            hidden = args[0]
            if record[index]:
                return                                                             # one capture per forward (a checkpoint recompute never re-runs here: handles are gone)
            record[index].extend(hidden[row, start:end] for row, (start, end) in enumerate(spans))
        return capture
    all_layers = decoder_layers(model)
    for index in record:
        handles.append(all_layers[index].register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()


def contiguous_runs(positions: t.Sequence[int]) -> list[tuple[int, int]]:
    """Sorted, unique positions as maximal [start, end) runs: [5, 6, 7, 12] -> [(5, 8), (12, 13)]. Runs let a capture hook take basic slices
    (views that save no tensors for backward) instead of an index gather (which would save its index tensor and change the saved-tensor
    sequence a checkpoint recompute has to reproduce)."""
    runs: list[tuple[int, int]] = []
    for position in sorted({int(position) for position in positions}):
        if runs and runs[-1][1] == position:
            runs[-1] = (runs[-1][0], position + 1)
        else:
            runs.append((position, position + 1))
    return runs


DISTILL_SOURCES = ("attn_out",)      # "hidden" (the layer's residual stream) was dropped at design review: the residual's per-dimension scale spread makes a whole-tensor RMS normalisation misleading


@contextmanager
def capture_layer_outputs(model, layers: t.Iterable[int], runs: list[list[tuple[int, int]]], source: str = "attn_out"):
    """Captures, per listed full-attention decoder layer, the attention block's output (`self_attn`'s first output, i.e. after `o_proj`: the
    residual contribution of attention) at `runs[row]` (the (start, end) runs of every physical row of the next forward), each run as a
    [end - start, d] clone: the yielded dict maps layer -> per row the list of run tensors, in order. The slice and the clone are graph-attached
    (their backward reaches the block) and save no tensor of their own, so a non-reentrant checkpoint recompute of the layer - which runs
    without these hooks - re-creates exactly the saved-tensor sequence of the original forward; the clone also stops the captured rows from
    pinning the block's whole [B, S, d] output. One capture per forward."""
    if source not in DISTILL_SOURCES:
        raise ValueError(f"source must be one of {DISTILL_SOURCES}")
    record: dict[int, list[list[torch.Tensor]]] = {int(layer): [] for layer in layers}
    handles = []
    def hook(index: int):
        def capture(module, args, output):
            if record[index]:
                return
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            record[index].extend([tensor[row, start:end].clone() for start, end in row_runs] for row, row_runs in enumerate(runs))
        return capture
    all_layers = decoder_layers(model)
    for index in record:
        if not 0 <= index < len(all_layers):
            raise ValueError(f"decoder layer {index} does not exist (the reader has {len(all_layers)} layers)")
        attention = getattr(all_layers[index], "self_attn", None)
        if attention is None:
            raise ValueError(f"decoder layer {index} has no full attention (self_attn) to capture the attention output from")
        handles.append(attention.register_forward_hook(hook(index)))
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_attention_qk(model, layers: t.Iterable[int]):
    """Diagnostic (no grad): per listed full-attention layer, the queries after `q_norm` and the keys after `k_norm` (both [B, S, heads, head_dim],
    pre-RoPE, the K including any K/V intervention written by the payload hooks, which run first) of the next forward, plus the forward's rotary
    (cos, sin) under "rope"; `attention_shares` turns them into attention weights the way the model does."""
    record: dict = {int(layer): {} for layer in layers}
    handles = []
    install_intervention_hooks(model)        # the payload hooks (persistent, once per model) must run before these: the K read here is the K the payload wrote
    def hook(index: int, name: str):
        def capture(module, args, output):
            record[index].setdefault(name, output.detach())
        return capture
    all_layers = decoder_layers(model)
    for index in record:
        attention = getattr(all_layers[index], "self_attn", None)
        if attention is None or not hasattr(attention, "q_norm") or not hasattr(attention, "k_norm"):
            raise ValueError(f"decoder layer {index} has no full attention (q_norm / k_norm) to read attention weights from")
        handles.append(attention.q_norm.register_forward_hook(hook(index, "q")))
        handles.append(attention.k_norm.register_forward_hook(hook(index, "k")))
    def rope(module, args, output):
        record.setdefault("rope", tuple(tensor.detach() for tensor in output))      # returns None: the module's output stays its own
    handles.append(text_decoder(model).rotary_emb.register_forward_hook(rope))
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def attention_shares(model, captured: dict, layer: int, queries: t.Sequence[int], keys: t.Sequence[int]) -> torch.Tensor:
    """From a `capture_attention_qk` record of a single-row forward: for `layer`, the attention weights of the positions `queries` (causal softmax
    over every earlier position, the model's RoPE, GQA grouping and scaling) summed over the key positions `keys`, per head and query
    position ([heads, len(queries)]); the mean is the share of attention mass those keys receive."""
    import sys
    text = text_decoder(model)
    rotary = sys.modules[type(text).__module__].apply_rotary_pos_emb
    attention = decoder_layers(model)[layer].self_attn
    q = captured[layer]["q"].transpose(1, 2).float()          # [1, H, S, D]
    k = captured[layer]["k"].transpose(1, 2).float()          # [1, Hkv, S, D]
    cos, sin = (tensor.float() for tensor in captured["rope"])
    q, k = rotary(q, k, cos, sin)
    repeats = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeats, dim=1) if repeats > 1 else k
    query_index = torch.as_tensor(list(queries), device=q.device, dtype=torch.long)
    scores = torch.matmul(q[0, :, query_index], k[0].transpose(-1, -2)) * attention.scaling          # [H, C, S]
    causal = torch.arange(k.shape[2], device=q.device)[None, :] > query_index[:, None]                # key after the query
    scores = scores.masked_fill(causal[None], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    key_index = torch.as_tensor(list(keys), device=q.device, dtype=torch.long)
    return weights[:, :, key_index].sum(dim=-1)


def generate_with_interventions(peft_model, *, inputs_embeds: torch.Tensor, interventions: list[Interventions | None], **generate_kwargs):
    """
    Greedy, single-sequence `generate` from embeddings whose prompt rows carry deltas. HF validates generate's kwargs
    against the model signature (an `interventions=` kwarg is rejected), so the payload is bound for this call only by
    wrapping the base model's `prepare_inputs_for_generation` (the one HF's loop calls every step under a PEFT wrapper
    too; the wrapper keeps the original signature for HF's validation and is removed in `finally`). The layer hooks
    map every step's cache offset, so the row deltas and K/V slots land on the prompt at prefill and on nothing afterwards
    (the cache keeps the written K/V); a cross-attention payload also reaches every generated position.
    """
    import functools
    if inputs_embeds.dim() != 3 or inputs_embeds.shape[0] != 1 or len(interventions) != 1:
        raise ValueError("generate_with_interventions takes one sequence and one payload")
    if generate_kwargs.get("num_beams", 1) != 1 or generate_kwargs.get("do_sample", False) or generate_kwargs.get("num_return_sequences", 1) != 1:
        raise ValueError("generate_with_interventions supports greedy single-sequence decoding only")
    base_model = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    install_intervention_hooks(base_model)
    previous = base_model.__dict__.get("prepare_inputs_for_generation")
    original = base_model.prepare_inputs_for_generation
    @functools.wraps(original)
    def bound(*args, **kwargs):
        model_inputs = original(*args, **kwargs)
        model_inputs[INTERVENTIONS_KWARG] = interventions
        return model_inputs
    base_model.prepare_inputs_for_generation = bound
    try:
        return peft_model.generate(inputs_embeds=inputs_embeds, **generate_kwargs)
    finally:
        if previous is None:
            del base_model.__dict__["prepare_inputs_for_generation"]
        else:
            base_model.prepare_inputs_for_generation = previous


def canonical_layer_type(raw_type: object) -> LayerType:
    normalized = str(raw_type).casefold()
    if normalized in _FULL_ATTENTION_NAMES:
        return LayerType.FULL_ATTENTION
    if normalized in _WINDOW_ATTENTION_NAMES:
        return LayerType.WINDOW_ATTENTION
    if normalized in _LINEAR_ATTENTION_NAMES:
        return LayerType.LINEAR_ATTENTION
    if normalized in _FFN_ONLY_NAMES:
        return LayerType.FFN_ONLY
    raise ValueError(f"Unsupported model layer type: {raw_type!r}")


def canonical_eot_token(tokenizer: PreTrainedTokenizerBase) -> str | None:
    candidates = list(getattr(tokenizer, "additional_special_tokens", None) or ())
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token:
        candidates.append(eos_token)
    for token in candidates:
        normalized = token.casefold()
        if "eot" in normalized or "im_end" in normalized or "end_of_turn" in normalized:
            return token
    return None

def compute_last_token(tokenizer: PreTrainedTokenizerBase) -> str|None:
    # Tokenize empty string and return last special token.
    return tokenizer.convert_ids_to_tokens(
        tokenizer("", add_special_tokens=True)["input_ids"][-1]
    )


def canonical_eos_token(hf_config: PreTrainedConfig, tokenizer: PreTrainedTokenizerBase) -> str | None:
    """Resolve the model config's EOS ID to its tokenizer representation."""
    text_config = hf_config.get_text_config()
    raw_token_ids = getattr(
        text_config,
        "eos_token_id",
        getattr(hf_config, "eos_token_id", None),
    )
    token_ids = (
        [raw_token_ids]
        if isinstance(raw_token_ids, int)
        else list(raw_token_ids or ())
    )
    for token_id in token_ids:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token is not None:
            return token
    return tokenizer.eos_token


def canonical_embedding_readout_type(model_id: str) -> EmbeddingReadoutType|None:
    normalized_id = model_id.lower()
    if "embed" not in normalized_id:
        return None
    if "qwen" in normalized_id:
        return EmbeddingReadoutType.EOS_TOKEN
    raise ValueError(f"Unknown embedding model: {model_id}")


def model_description_and_tokenizer_from_hf(
    model_id: str,
    dtype: torch.dtype,
    trust_remote_code: bool = False,
) -> tuple['ModelDescription', PreTrainedTokenizerBase, PreTrainedTokenizerBase|ProcessorMixin]:
    # Get basic configs.
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    text_config = hf_config.get_text_config()
    # Load tokenizer and processor based on modality.
    is_multimodal = text_config is not hf_config
    if is_multimodal:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        assert isinstance(processor, ProcessorMixin)
        tokenizer = processor.tokenizer
    else:
        processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tokenizer = processor

    # Compute layer descriptions.
    num_layers = int(text_config.num_hidden_layers)
    raw_layer_types = getattr(text_config, "layer_types", None)

    if raw_layer_types is None:
        raw_layer_types = ["full_attention"] * num_layers

    if len(raw_layer_types) != num_layers:
        raise ValueError(
            f"Config reports {num_layers} layers but supplies "
            f"{len(raw_layer_types)} layer types"
        )

    window_size = getattr(text_config, "sliding_window", None)
    layers = []

    for index, raw_type in enumerate(raw_layer_types):
        layer_type = canonical_layer_type(raw_type)
        layers.append(
            LayerDescription(
                layer_type=layer_type,
                layer_idx=index,
                window_attention_size=(
                    window_size
                    if layer_type is LayerType.WINDOW_ATTENTION
                    else None
                ),
            )
        )

    # Get readout type.
    embedding_readout_type = canonical_embedding_readout_type(model_id)
    if embedding_readout_type is EmbeddingReadoutType.EOS_TOKEN:
        eos_token = compute_last_token(tokenizer)
    else:
        eos_token = canonical_eos_token(hf_config, tokenizer)
    # Finalize.
    num_heads = getattr(text_config, "num_attention_heads", None)
    head_dim = getattr(text_config, "head_dim", None) or (text_config.hidden_size // num_heads if num_heads else None)
    kv_heads = getattr(text_config, "num_key_value_heads", None) or num_heads
    description = ModelDescription(
        d_model=text_config.hidden_size,
        d_ff=getattr(text_config, "intermediate_size", None) or 4 * text_config.hidden_size,
        is_multimodal=is_multimodal,
        d_kv=int(kv_heads * head_dim) if kv_heads and head_dim else None,
        dtype=dtype,
        layer_descriptions=layers,
        eos_token=eos_token,
        eot_token=canonical_eot_token(tokenizer),
        is_embedding_model=embedding_readout_type is not None,
        embedding_readout_type=embedding_readout_type,
    )
    return description, tokenizer, processor


def adapt_message_format(model_config: ModelConfig, messages: list[dict]):
    """
    Adapt the message.
    Assumes the input is strongly typed e.g., {type: "text", text: ...}.
    """
    if model_config.model_description.is_multimodal:
        return messages
    for msg in messages:
        if "content" in msg:
            if isinstance(msg["content"], str):
                continue
            if isinstance(msg["content"], list):
                assert len(msg["content"]) == 1
                if msg["content"][0]["type"] == "text":
                    msg["content"] = msg["content"][0]["text"]
                else:
                    raise RuntimeError("Non-multimodal model only supports direct content, texts.")



def readout_embedding(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    readout_type: EmbeddingReadoutType,
) -> torch.Tensor:
    """Readout the embedding from the last hidden state of a model."""
    if readout_type in [EmbeddingReadoutType.EOS_TOKEN, EmbeddingReadoutType.EOT_TOKEN]:
        # Left padded (or no padding) if right-most point is always active.
        # So directly readout.
        left_padded = bool(attention_mask[:, -1].all())
        if left_padded:
            pooled = last_hidden_state[:, -1]
        else:
            # Right-padded: select the last active index (equivalent to sum of 1s up to that point).
            padding_end_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(padding_end_indices.shape[0], device=last_hidden_state.device)
            pooled = last_hidden_state[batch_indices, padding_end_indices]
    else:
        # Avg: just average all active positions.
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return F.normalize(pooled, p=2, dim=1)


def make_peft_lora_config(lora_config: "LoraConfig") -> "peft.LoraConfig":
    """PEFT config for the given targets; bias untouched; task type causal LM."""
    import peft
    return peft.LoraConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=list(lora_config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def pretty_format_model_description(model_config: ModelConfig) -> str:
    """Return a compact human-readable description of a loaded model."""

    description = model_config.model_description
    if description is None:
        return f"{model_config.model_name}: not loaded"

    counts = Counter(layer.layer_type.value for layer in description.layer_descriptions)
    layer_summary = ", ".join(
        f"{count} {layer_type}" for layer_type, count in sorted(counts.items())
    )
    lines = [
        f"model_name: {model_config.model_name}",
        f"model_id: {model_config.model_id}",
        f"d_model: {description.d_model}",
        f"d_ff: {description.d_ff}",
        f"dtype: {description.dtype}",
        f"layers: {len(description.layer_descriptions)} ({layer_summary})",
        f"multimodal: {description.is_multimodal}",
        f"embedding_model: {description.is_embedding_model}",
        f"eos_token: {description.eos_token}",
        f"eot_token: {description.eot_token}",
    ]
    if description.embedding_readout_type is not None:
        lines.append(f"embedding_readout: {description.embedding_readout_type.value}")
    return "\n".join(lines)
