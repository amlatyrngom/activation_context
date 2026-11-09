"""
Key/value transfer: the keys and values an encoder computes at chosen positions (its summary tokens) become entries of the
reader's attention, at every layer. The reader never computes those positions: it attends them as if they were tokens it had
already read.

Encoder side (`LayerInputCapture`, used through `LoadedModel.decoder_forward(capture=...)`): pre-hooks take the hidden state
entering every decoder layer; after the forward, each layer's own input norm and key/value projections (the active adapter
included) are applied to the chosen positions only. That is exactly what the layer's attention computed there (before the
rotary position is applied to the keys), with gradients, and it works under gradient checkpointing (which turns HF's own cache
off).

Reader side (`bind_kv_prefix`, used through `decoder_forward(interventions=...)` when a payload carries `kv_values`): the
reader's sequence keeps one inert placeholder per transferred entry (zero embedding, attended by nobody), so every index the
caller holds stays valid. The transferred keys are rotated to their positions (the placeholder run's start plus the entry's
position in the encoder's sequence), the tokens after a run are numbered as if the encoder's whole sequence stood in its place,
and a boolean mask lets exactly those tokens attend the run's entries. `PrefixKV` hands every layer "the entries + this
forward's keys/values" without storing anything, so a checkpointed layer recomputes the same result. It travels as the
`kv_prefix=` keyword and a pre-hook on every attention module turns it into that module's `past_key_values`: HF drops a
`past_key_values` argument from a checkpointed layer's call, the keyword passes.

Generation (`generate_with_kv_prefix`): the same prefill, then a plain greedy loop over a growing copy of the cache.

The above needs a full-attention decoder (every layer has `self_attn` with `k_proj` / `v_proj`); the per-layer residual
deltas (`layer_inputs`) and key/value transfer are not combined in one payload.

Overwrite mode (`kv_overwrite` in the payload; `bind_kv_overwrite`, `OverwriteKV`), for hybrid decoders whose other layers are
recurrent: the row positions are real positions of the reader again (the caller gives them an input embedding) and everything
propagates as usual, except that at every full-attention layer the keys and values AT the row positions are replaced by the
transferred ones (rotated to the rows' own positions in the reader). The recurrent layers run over the rows untouched and
carry forward whatever the rows picked up from attention. The encoder side captures at its attention layers only.
"""
from __future__ import annotations

import sys
import typing as t

import torch

KV_VALUES, KV_POSITIONS, KV_RUNS = "kv_values", "kv_positions", "kv_runs"
KV_BIAS = "kv_bias"                                        # optional: {layer: [M, kv_heads]} added to the entries' attention scores
KV_LAYER_COUNTS = "kv_layer_counts"                        # optional: {layer: [entries per run]}: that layer keeps only the first ones of each run
                                                           # (kv_values / kv_bias of the layer hold exactly those; needs MASKLESS)
KV_OVERWRITE = "kv_overwrite"
KV_PREFIX_KWARG = "kv_prefix"
_HOOKS_ATTRIBUTE = "_activation_kv_prefix_hooks"


def _layers(model) -> torch.nn.ModuleList:
    from .hf_utils import decoder_layers
    return decoder_layers(model)


def kv_layers(model) -> list[int]:
    """The decoder layers that have attention keys and values (all of them in a full-attention model; the full-attention blocks of a hybrid)."""
    def has(layer) -> bool:
        attention = getattr(layer, "self_attn", None)
        return attention is not None and all(hasattr(attention, name) for name in ("k_proj", "v_proj", "head_dim")) and hasattr(layer, "input_layernorm")
    return [index for index, layer in enumerate(_layers(model)) if has(layer)]


def check_kv_model(model, every_layer: bool = True) -> tuple[int, int, int]:
    """(layers, key/value heads, head width) of a decoder that can take transferred keys and values: at every layer (the
    attention-entry mode), or at its attention layers at least (the overwrite mode)."""
    layers = _layers(model)
    with_kv = kv_layers(model)
    if not with_kv or (every_layer and len(with_kv) != len(layers)):
        missing = next((index for index in range(len(layers)) if index not in with_kv), None)
        raise ValueError(f"key/value transfer as attention entries needs a full-attention decoder: layer {missing} ({type(layers[missing]).__name__ if missing is not None else 'none'}) "
                         "has no self_attn key/value projections (a hybrid decoder takes the overwrite mode)")
    attention = layers[with_kv[0]].self_attn
    head_dim = int(attention.head_dim)
    return len(layers), int(attention.k_proj.out_features) // head_dim, head_dim


class LayerInputCapture:
    """For one decoder forward: the keys (not yet rotated) and values of every layer at `positions[row]` of each physical row.
    After the forward, `keys[layer][row]` and `values[layer][row]` are [M_row, kv_heads, head_dim]."""

    def __init__(self, positions: list[torch.Tensor]) -> None:
        self.positions = positions
        self.inputs: dict[int, torch.Tensor | list[torch.Tensor]] = {}
        self.keys: dict[int, list[torch.Tensor]] = {}
        self.values: dict[int, list[torch.Tensor]] = {}

    def _select(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        return [hidden[row].index_select(0, positions.to(hidden.device)) for row, positions in enumerate(self.positions)]

    def install(self, model) -> list:
        check_kv_model(model, every_layer=False)
        def hook(index: int):
            def take(module, args, kwargs):
                if index in self.inputs:                    # a checkpointed layer's recomputation: the first call's tensor stands
                    return None
                hidden = args[0] if args else kwargs["hidden_states"]
                if hidden.shape[0] != len(self.positions):
                    raise ValueError(f"{len(self.positions)} capture position lists for {hidden.shape[0]} physical rows")
                # With gradients the layer input is held as it is (autograd keeps it alive anyway) and indexed after the forward,
                # outside any checkpointed region; without, only the chosen rows are kept.
                self.inputs[index] = hidden if hidden.requires_grad else self._select(hidden)
                return None
            return take
        layers = _layers(model)
        return [layers[index].register_forward_pre_hook(hook(index), with_kwargs=True) for index in kv_layers(model)]

    def finish(self, model) -> None:
        """The layers' own norm and projections on the chosen rows; call inside the forward's adapter scope."""
        counts = [int(positions.shape[0]) for positions in self.positions]
        layers = _layers(model)
        for index in kv_layers(model):
            layer = layers[index]
            held = self.inputs.pop(index)
            rows = torch.cat(held if isinstance(held, list) else self._select(held), dim=0)
            attention = layer.self_attn
            normed = layer.input_layernorm(rows)
            keys = attention.k_proj(normed).view(rows.shape[0], -1, attention.head_dim)
            if hasattr(attention, "k_norm"):
                keys = attention.k_norm(keys)
            values = attention.v_proj(normed).view(rows.shape[0], -1, attention.head_dim)
            self.keys[index] = list(keys.split(counts))
            self.values[index] = list(values.split(counts))


def rotate_keys(model, keys: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Keys [B, heads, P, head_dim] rotated to `positions` [B, P] with the decoder's own rotary embedding."""
    from .hf_utils import text_decoder
    text = text_decoder(model)
    cos, sin = text.rotary_emb(keys, positions)
    return sys.modules[type(text).__module__].apply_rotary_pos_emb(keys, keys, cos, sin)[1]


class PrefixKV:
    """The transferred entries of every layer in front of the forward's own keys/values. `grow=False`: nothing is stored (a
    layer run twice returns the same tensors); `grow=True`: the returned tensors become the stored ones (generation).
    `biases` (optional, per layer [B, kv_heads, P]): a score added to every query head's attention to each entry, e.g. the
    log-weight β of a compacted cache (Attention Matching); the attention pre-hook turns the layer's boolean mask additive."""

    def __init__(self, keys: list[torch.Tensor], values: list[torch.Tensor], grow: bool = False, biases: list[torch.Tensor] | None = None,
                 maskless: "MasklessLayout | list[MasklessLayout] | None" = None) -> None:
        self.keys, self.values, self.grow, self.biases, self.maskless = keys, values, grow, biases, maskless
        # Maskless generation: each layer's [entries, tokens] keys/values, already widened by the bias dimensions, in a buffer with
        # room for `capacity` more tokens (_maskless_attention fills it); update() then hands over the new keys/values only.
        self.capacity = 0
        self.store: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(keys)
        self.filled = [0] * len(keys)

    @property
    def stores(self) -> bool:
        return self.grow and self.maskless is not None

    def biased_mask(self, mask: torch.Tensor | None, attention) -> torch.Tensor:
        """The additive float mask [B, heads, S, P + S] of `attention`'s layer: the boolean `mask` [B, 1, S, P + S] (True =
        attend) as 0 / dtype minimum, plus this layer's bias on the P entry columns, repeated over each key/value head's query heads."""
        if mask is None or mask.dtype != torch.bool or mask.ndim != 4:
            raise ValueError("entry biases need the boolean 4D mask of bind_kv_prefix")
        bias = self.biases[attention.layer_idx]
        groups = int(getattr(attention, "num_key_value_groups", 1))
        additive = torch.zeros(mask.shape, dtype=bias.dtype, device=mask.device).masked_fill(~mask, torch.finfo(bias.dtype).min)
        heads = bias.repeat_interleave(groups, dim=1)[:, :, None, :]                       # [B, heads, 1, P]
        entries = bias.shape[-1]
        return torch.cat([additive[..., :entries] + heads, additive[..., entries:].expand(-1, heads.shape[1], -1, -1)], dim=-1)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if self.stores:
            return key_states, value_states                # _maskless_attention appends them to the layer's buffer
        keys = torch.cat([self.keys[layer_idx].to(key_states.dtype), key_states], dim=2)
        values = torch.cat([self.values[layer_idx].to(value_states.dtype), value_states], dim=2)
        if self.grow:
            self.keys[layer_idx], self.values[layer_idx] = keys, values
        return keys, values

    def get_seq_length(self, *args, **kwargs) -> int:
        return 0                                           # the forward's tokens keep their physical indices (positions are passed explicitly)


class OverwriteKV:
    """Overwrite mode: the keys and values at the row positions of this forward's tokens are replaced by the transferred ones
    (`entries[layer]` = per physical row None or (positions [M], keys [heads, M, head_dim] rotated, values [heads, M, head_dim])).
    Nothing is stored here; with `inner` (the cache of a generation prefill) the replaced tensors go on into that cache."""

    def __init__(self, entries: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]], inner=None) -> None:
        self.entries, self.inner = entries, inner

    def bound(self, inner) -> "OverwriteKV":
        return OverwriteKV(self.entries, inner)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.entries.get(layer_idx)
        if rows is not None:
            if len(rows) != key_states.shape[0]:
                raise ValueError(f"{len(rows)} overwrite rows for {key_states.shape[0]} physical rows")
            key_states = torch.stack([key_states[row] if entry is None else key_states[row].index_copy(1, entry[0], entry[1].to(key_states.dtype)) for row, entry in enumerate(rows)])
            value_states = torch.stack([value_states[row] if entry is None else value_states[row].index_copy(1, entry[0], entry[2].to(value_states.dtype)) for row, entry in enumerate(rows)])
        if self.inner is not None:
            return self.inner.update(key_states, value_states, layer_idx, *args, **kwargs)
        return key_states, value_states

    def get_seq_length(self, *args, **kwargs) -> int:
        return 0 if self.inner is None else self.inner.get_seq_length(*args, **kwargs)


ENTRY_BIAS_KWARG = "kv_entry_bias"
MASKLESS_KWARG = "kv_maskless"
MASKLESS_STORE_KWARG = "kv_maskless_store"
_ENTRY_BIAS_DIMS = 8                                      # extra head dimensions carrying the entry bias (a multiple of 8 keeps the fused kernels)
_MASKED_SCORE = -1.0e4                                    # exp(-1e4) = 0: a key that takes no attention


class MasklessLayout(t.NamedTuple):
    """A prefix forward's boolean mask, rewritten as what the fused lower-right causal kernel cannot see by itself: keys never
    visible (pads, placeholders, a row's missing entries) and, per group of runs ending at the same row index, which tokens
    come before that end (they may not see the group's entries). bind_kv_prefix builds it only when it reproduces the mask on
    every row that is attended."""
    entries: int                     # P
    prompt: int                      # S of the bound forward
    key_bias: torch.Tensor           # [B, P + S] fp32: 0 or _MASKED_SCORE
    early: torch.Tensor              # [B, S, G]: 1 where the token comes before group g's end
    groups: torch.Tensor             # [B, P, G]: 1 where the entry belongs to group g


def _maskless_attention(module, query, key, value, bias, layout: MasklessLayout, dropout: float = 0.0, scaling: float | None = None,
                        store: "tuple[PrefixKV, int] | None" = None, **unused):
    """Attention over [entries, tokens] without a mask: SDPA's lower-right causal bias (entries visible to every token by
    position, tokens causal), the never-visible keys and the entry bias β in two extra head dimensions ([1, 1] on the query,
    [hi, lo] = score / scaling on the key), and per group one more dimension (the query's "before the end" flag times
    _MASKED_SCORE / scaling on the group's entry keys). Keys and values keep their kv heads (native GQA).
    `store` (generation, PrefixKV.stores): `key` / `value` are this forward's own; the prefill widens [entries, prompt] once
    into the layer's buffer, a decode step writes its key/value (visible, no bias) into the next slot and attends the filled part."""
    import torch.nn.functional as F
    from torch.nn.attention.bias import causal_lower_right
    scaling = scaling or query.shape[-1] ** -0.5
    head_dim = value.shape[-1]
    queries = query.shape[2]
    if store is not None:
        prefix, layer = store
        if prefix.store[layer] is None:
            key = torch.cat([prefix.keys[layer].to(key.dtype), key], dim=2)
            value = torch.cat([prefix.values[layer].to(value.dtype), value], dim=2)
            if key.shape[2] != layout.entries + layout.prompt:
                raise ValueError(f"maskless generation: the prefill has {key.shape[2]} keys, bound for {layout.entries + layout.prompt}")
            key, value = _widened(key, value, bias, layout, scaling)
            total = key.shape[2]
            keys_buffer = key.new_empty(*key.shape[:2], total + prefix.capacity, key.shape[3])
            values_buffer = value.new_empty(*value.shape[:2], total + prefix.capacity, value.shape[3])
            keys_buffer[:, :, :total], values_buffer[:, :, :total] = key, value
            prefix.store[layer], prefix.filled[layer] = (keys_buffer, values_buffer), total
        else:
            if queries != 1:
                raise ValueError(f"maskless generation: a decode step of {queries} queries")
            keys_buffer, values_buffer = prefix.store[layer]
            at = prefix.filled[layer]
            if at + 1 > keys_buffer.shape[2]:                                         # past the capacity: double the buffer
                keys_buffer = torch.cat([keys_buffer, torch.empty_like(keys_buffer)], dim=2)
                values_buffer = torch.cat([values_buffer, torch.empty_like(values_buffer)], dim=2)
                prefix.store[layer] = (keys_buffer, values_buffer)
            keys_buffer[:, :, at:at + 1] = torch.cat([key, key.new_zeros(*key.shape[:-1], _ENTRY_BIAS_DIMS)], dim=-1)
            values_buffer[:, :, at:at + 1] = torch.cat([value, value.new_zeros(*value.shape[:-1], _ENTRY_BIAS_DIMS)], dim=-1)
            prefix.filled[layer] = at + 1
        length = prefix.filled[layer]
        key, value = keys_buffer[:, :, :length], values_buffer[:, :, :length]
    else:
        length = key.shape[2]
        key, value = _widened(key, value, bias, layout, scaling)
    query = _widened_query(query, layout, length)
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=causal_lower_right(queries, length), dropout_p=dropout,
                                            scale=scaling, enable_gqa=query.shape[1] != key.shape[1])
    return output.transpose(1, 2).contiguous()[..., :head_dim], None


def _widened_query(query: torch.Tensor, layout: MasklessLayout, length: int) -> torch.Tensor:
    """Queries [B, heads, q, d] + [1, 1, early flags per group, 0...] (the flags only in the bound forward itself)."""
    batch, _, queries, _ = query.shape
    group_count = layout.groups.shape[-1]
    if length == layout.entries + layout.prompt and queries == layout.prompt:
        early = layout.early.to(query.dtype)                                      # the bound forward (or its checkpointed rerun)
    elif queries == 1:
        early = query.new_zeros(batch, 1, group_count)                            # a decode step comes after every run
    else:
        raise ValueError(f"maskless prefix attention: {queries} queries over {length} keys, bound for {layout.prompt} over {layout.entries + layout.prompt}")
    rest = _ENTRY_BIAS_DIMS - 2 - group_count
    return torch.cat([query, query.new_ones(*query.shape[:-1], 2), early[:, None].expand(batch, query.shape[1], queries, group_count),
                      query.new_zeros(*query.shape[:-1], rest)], dim=-1)


def _widened(key: torch.Tensor, value: torch.Tensor, bias: torch.Tensor | None, layout: MasklessLayout, scaling: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Keys [B, kv_heads, L, d] + [hi, lo, group columns, 0...] (L = the entries, the bound tokens and any generated ones) and
    values + zeros."""
    import torch.nn.functional as F
    batch, heads, length, _ = key.shape
    entries = layout.entries
    score = layout.key_bias
    if length > score.shape[1]:                                                   # decode steps: the generated tokens are visible
        score = F.pad(score, (0, length - score.shape[1]))
    score = score[:, None, :].expand(batch, heads, length)
    if bias is not None:
        score = score + F.pad(bias.float(), (0, length - entries))
    target = score / scaling
    high = target.to(key.dtype)
    low = (target - high.float()).to(key.dtype)
    group_count = layout.groups.shape[-1]
    group_keys = F.pad(layout.groups.to(key.dtype) * (_MASKED_SCORE / scaling), (0, 0, 0, length - entries))   # [B, P + ..., G]
    rest = _ENTRY_BIAS_DIMS - 2 - group_count
    key = torch.cat([key, high[..., None], low[..., None], group_keys[:, None].expand(batch, heads, length, group_count),
                     key.new_zeros(batch, heads, length, rest)], dim=-1)
    value = torch.cat([value, value.new_zeros(*value.shape[:-1], _ENTRY_BIAS_DIMS)], dim=-1)
    return key, value


def _sdpa_with_entry_bias(module, query, key, value, attention_mask, *args, **kwargs):
    """HF's sdpa attention, plus (when the attention pre-hook passed `kv_entry_bias` [B, kv_heads, P]) a score bias on the
    first P key columns, carried in extra head dimensions instead of a float mask: queries gain [1, 1, 0, ...], the entry
    keys [hi, lo, 0, ...] with hi + lo = bias / scaling split into two bf16 parts, so q.k * scaling gains exactly the bias
    (accumulated in fp32) while the mask stays boolean, needs no gradient and keeps the fused kernels. With `kv_maskless`
    (a MasklessLayout) the mask is not used at all (_maskless_attention)."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    bias = kwargs.pop(ENTRY_BIAS_KWARG, None)
    layout = kwargs.pop(MASKLESS_KWARG, None)
    store = kwargs.pop(MASKLESS_STORE_KWARG, None)
    if layout is not None:
        return _maskless_attention(module, query, key, value, bias, layout, *args, store=store, **kwargs)
    if bias is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask, *args, **kwargs)
    head_dim = value.shape[-1]
    scaling = kwargs.get("scaling") or query.shape[-1] ** -0.5
    kwargs["scaling"] = scaling
    target = bias.float() / scaling                                                           # [B, kv_heads, P]
    high = target.to(key.dtype)
    low = (target - high.float()).to(key.dtype)
    batch, heads, length = key.shape[:3]
    entries = target.shape[-1]
    entry_columns = torch.cat([high[..., None], low[..., None], key.new_zeros(batch, heads, entries, _ENTRY_BIAS_DIMS - 2)], dim=-1)
    key = torch.cat([key, torch.cat([entry_columns, key.new_zeros(batch, heads, length - entries, _ENTRY_BIAS_DIMS)], dim=2)], dim=-1)
    query = torch.cat([query, query.new_ones(*query.shape[:-1], 2), query.new_zeros(*query.shape[:-1], _ENTRY_BIAS_DIMS - 2)], dim=-1)
    value = torch.cat([value, value.new_zeros(*value.shape[:-1], _ENTRY_BIAS_DIMS)], dim=-1)
    output, weights = sdpa_attention_forward(module, query, key, value, attention_mask, *args, **kwargs)
    return output[..., :head_dim], weights


def _register_entry_bias_sdpa() -> None:
    from transformers import AttentionInterface
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if ALL_ATTENTION_FUNCTIONS.get("sdpa") is not _sdpa_with_entry_bias:
        AttentionInterface.register("sdpa", _sdpa_with_entry_bias)        # identical to HF's sdpa unless `kv_entry_bias` is passed


def install_kv_hooks(model) -> None:
    """Persistent, parameter-free pre-hooks (once per model): on every attention module a `kv_prefix=` keyword is taken out and
    handed over as `past_key_values` (an OverwriteKV first bound to the cache the call came with); on a recurrent block's mixer
    the keyword is only taken out. A forward without the keyword is untouched."""
    if getattr(model, _HOOKS_ATTRIBUTE, None) is not None:
        return
    check_kv_model(model, every_layer=False)
    _register_entry_bias_sdpa()
    def hook(module, args, kwargs):
        if KV_PREFIX_KWARG not in kwargs:
            return None
        kwargs = dict(kwargs)
        prefix = kwargs.pop(KV_PREFIX_KWARG)
        kwargs["past_key_values"] = prefix.bound(kwargs.get("past_key_values")) if isinstance(prefix, OverwriteKV) else prefix
        if isinstance(prefix, PrefixKV) and (prefix.biases is not None or prefix.maskless is not None):
            if getattr(module.config, "_attn_implementation", None) == "sdpa":
                if prefix.biases is not None:
                    kwargs[ENTRY_BIAS_KWARG] = prefix.biases[module.layer_idx]     # carried in extra head dimensions (_sdpa_with_entry_bias)
                if prefix.maskless is not None:
                    kwargs[MASKLESS_KWARG] = prefix.maskless[module.layer_idx] if isinstance(prefix.maskless, list) else prefix.maskless
                    if prefix.stores:
                        kwargs[MASKLESS_STORE_KWARG] = (prefix, module.layer_idx)
            elif prefix.biases is not None:
                kwargs["attention_mask"] = prefix.biased_mask(kwargs.get("attention_mask"), module)
        return args, kwargs
    def drop(module, args, kwargs):
        if KV_PREFIX_KWARG not in kwargs:
            return None
        return args, {key: value for key, value in kwargs.items() if key != KV_PREFIX_KWARG}
    with_kv = set(kv_layers(model))
    handles = []
    for index, layer in enumerate(_layers(model)):
        if index in with_kv:
            handles.append(layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True))
        else:
            handles.extend(child.register_forward_pre_hook(drop, with_kwargs=True) for name, child in layer.named_children() if name.endswith("_attn"))
    setattr(model, _HOOKS_ATTRIBUTE, handles)


def is_overwrite(payloads: t.Sequence[dict | None] | None) -> bool:
    return bool(payloads) and any(payload and payload.get(KV_VALUES) and payload.get(KV_OVERWRITE) for payload in payloads)


def bind_kv_overwrite(model, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None, payloads: t.Sequence[dict | None]) -> OverwriteKV:
    """The OverwriteKV of a reader forward of `inputs_embeds` [B, S, d]: per payload (None: nothing for that row) `positions`
    (the row positions in the physical row) and `kv_values` {attention layer: (keys, values)}, each [M, kv_heads, head_dim], keys
    not rotated. The keys are rotated to the rows' own positions (real tokens counted from 0; `attention_mask` [B, S] marks pads)."""
    check_kv_model(model, every_layer=False)
    install_kv_hooks(model)
    batch, length = inputs_embeds.shape[:2]
    device = inputs_embeds.device
    if len(payloads) != batch:
        raise ValueError(f"{len(payloads)} payloads for {batch} physical rows")
    real = torch.ones(batch, length, dtype=torch.bool, device=device) if attention_mask is None else attention_mask.to(device).bool()
    position_ids = (real.long().cumsum(dim=1) - 1).clamp_min(0)
    with_kv = kv_layers(model)
    entries: dict[int, list] = {layer: [None] * batch for layer in with_kv}
    for row, payload in enumerate(payloads):
        if not (payload and payload.get(KV_VALUES)):
            continue
        if payload.get("layer_inputs"):
            raise ValueError("a payload carries either residual deltas or transferred keys/values, not both")
        if sorted(payload[KV_VALUES]) != with_kv:
            raise ValueError(f"transferred keys/values must cover the decoder's attention layers {with_kv}")
        positions = torch.as_tensor(payload["positions"], dtype=torch.long, device=device)
        if positions.numel() and int(positions.max()) >= length:
            raise ValueError("a row position lies beyond the forward's tokens")
        for layer in with_kv:
            keys, values = payload[KV_VALUES][layer]
            if keys.shape[0] != positions.shape[0]:
                raise ValueError("row positions and transferred entries differ in count")
            rotated = rotate_keys(model, keys.to(inputs_embeds).transpose(0, 1)[None], position_ids[row, positions][None])[0]
            entries[layer][row] = (positions, rotated, values.to(inputs_embeds).transpose(0, 1))
    return OverwriteKV(entries)


DROP_PLACEHOLDERS = False     # opt-in: reader forwards drop the inert placeholder rows (drop_placeholders) instead of carrying them


def drop_placeholders(inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None, payloads: t.Sequence[dict | None]
                      ) -> tuple[torch.Tensor, torch.Tensor, list[dict | None], torch.Tensor]:
    """The same reader forward without its placeholder rows (they attend only themselves and nobody attends them, so every
    other row's result is unchanged): (inputs_embeds [B, S', d] left-padded, attention_mask [B, S'], payloads whose runs are
    empty and name their entry count (start, start, span, entries), kept [B, S'] = each new row's index in the old row, -1 on
    new padding). `restore_rows` puts outputs back at the old indices."""
    batch, length = inputs_embeds.shape[:2]
    device = inputs_embeds.device
    real = torch.ones(batch, length, dtype=torch.bool, device=device) if attention_mask is None else attention_mask.to(device).bool()
    keep = real.clone()
    for row, payload in enumerate(payloads):
        if payload and payload.get(KV_VALUES):
            for start, end, *_ in payload[KV_RUNS]:
                keep[row, start:end] = False
    kept_rows = [keep[row].nonzero()[:, 0] for row in range(batch)]
    width = max(int(rows.shape[0]) for rows in kept_rows)
    kept = torch.full((batch, width), -1, dtype=torch.long, device=device)
    new_payloads: list[dict | None] = []
    for row, rows in enumerate(kept_rows):
        offset = width - int(rows.shape[0])
        kept[row, offset:] = rows
        payload = payloads[row]
        if payload and payload.get(KV_VALUES):
            new_index = torch.full((length + 1,), -1, dtype=torch.long, device=device)
            new_index[rows] = torch.arange(offset, width, device=device)
            new_index[length] = width
            def first_kept_at_or_after(index: int) -> int:                    # where the run's entries now sit (before the next kept row)
                after = rows[rows >= index]
                return int(new_index[after[0]]) if after.numel() else width
            runs = []
            for start, end, span, *count in payload[KV_RUNS]:
                at = first_kept_at_or_after(end)
                runs.append((at, at, span, count[0] if count else end - start))
            payload = {**payload, KV_RUNS: runs}
        new_payloads.append(payload)
    index = kept.clamp_min(0)
    embeds = torch.gather(inputs_embeds, 1, index[..., None].expand(-1, -1, inputs_embeds.shape[-1]))
    return embeds, kept >= 0, new_payloads, kept


def restore_rows(hidden: torch.Tensor, kept: torch.Tensor, length: int) -> torch.Tensor:
    """[B, S', ...] outputs of a forward on drop_placeholders' rows, back at their old indices of a [B, length, ...] tensor (zero elsewhere)."""
    out = hidden.new_zeros(hidden.shape[0], length + 1, *hidden.shape[2:])
    target = torch.where(kept >= 0, kept, torch.full_like(kept, length))
    out = out.index_put((torch.arange(hidden.shape[0], device=hidden.device)[:, None].expand_as(target), target), hidden)
    return out[:, :length]


def has_kv(payloads: t.Sequence[dict | None] | None) -> bool:
    return bool(payloads) and any(payload and payload.get(KV_VALUES) for payload in payloads)


def bind_kv_prefix(model, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None, payloads: t.Sequence[dict | None],
                   grow: bool = False) -> tuple[dict | torch.Tensor, torch.Tensor, PrefixKV, torch.Tensor]:
    """
    (attention mask mapping per layer type, or the one 4D mask for a model without layer types, position ids [B, S], PrefixKV, column visibility [B, P + S]) for a reader forward of
    `inputs_embeds` [B, S, d] whose rows carry one payload each (None: no transfer for that row):

        kv_values     {layer: (keys, values)}, each [M, kv_heads, head_dim], keys not rotated, every decoder layer present
        kv_positions  [M] long: each entry's position in its encoder sequence
        kv_runs       [(start, end, length)] per part, in order: the placeholder run in this row's physical coordinates
                      (end - start entries) and the length of the encoder sequence it stands for
        kv_bias       optional {layer: [M, kv_heads]}: added to the entries' attention scores (0 where absent)

    `attention_mask` [B, S] (0 on pads) is honoured for the token columns, and positions count real tokens only. The last
    return value says which columns a token after the prompt may attend (the decode steps' mask).
    """
    num_layers, heads, head_dim = check_kv_model(model)
    install_kv_hooks(model)
    batch, length = inputs_embeds.shape[:2]
    device = inputs_embeds.device
    if len(payloads) != batch:
        raise ValueError(f"{len(payloads)} payloads for {batch} physical rows")
    counts = [int(payload[KV_POSITIONS].shape[0]) if payload and payload.get(KV_VALUES) else 0 for payload in payloads]
    prefix = max(counts)
    real = torch.ones(batch, length, dtype=torch.bool, device=device) if attention_mask is None else attention_mask.to(device).bool()
    position_ids = (real.long().cumsum(dim=1) - 1).clamp_min(0)
    token_columns = real.clone()                                             # columns a later token may attend: real tokens that are no placeholders
    prefix_visible = torch.zeros(batch, length, prefix, dtype=torch.bool, device=device)   # [row token, entry]
    key_positions = torch.zeros(batch, prefix, dtype=torch.long, device=device)
    run_ends: list[list[tuple[int, int, int]]] = [[] for _ in payloads]       # per row (end, first entry, last entry + 1)
    run_starts: list[list[torch.Tensor]] = [[] for _ in payloads]             # per row: each run's first position
    relatives: list[torch.Tensor | None] = [None] * batch
    per_layer = any(counts[row] and payload.get(KV_LAYER_COUNTS) for row, payload in enumerate(payloads))
    if per_layer and not MASKLESS:
        raise ValueError("per-layer entry counts (kv_layer_counts) need the maskless attention (kv_transfer.MASKLESS)")
    for row, payload in enumerate(payloads):
        if not counts[row]:
            continue
        if payload.get("layer_inputs"):
            raise ValueError("a payload carries either residual deltas or transferred keys/values, not both")
        if sorted(payload[KV_VALUES]) != list(range(num_layers)):
            raise ValueError(f"transferred keys/values must cover every decoder layer 0..{num_layers - 1}")
        relative = relatives[row] = payload[KV_POSITIONS].to(device)
        cursor = 0
        for start, end, span, *count in payload[KV_RUNS]:
            entries = count[0] if count else end - start                   # (start, end, span, entries): fewer placeholders than entries (drop_placeholders)
            if not 0 <= start <= end <= length or span < end - start or (count and end - start not in (0, entries)):
                raise ValueError(f"placeholder run ({start}, {end}) standing for {span} positions does not fit a row of {length}")
            run_position = position_ids[row, start] if start < length else position_ids[row, length - 1] + 1
            run_starts[row].append(run_position.clone())                       # a copy: the shift below moves position_ids[row, end:] in place
            key_positions[row, cursor:cursor + entries] = run_position + relative[cursor:cursor + entries]
            position_ids[row, end:] += span - (end - start)
            prefix_visible[row, end:, cursor:cursor + entries] = True
            token_columns[row, start:end] = False
            run_ends[row].append((end, cursor, cursor + entries))
            cursor += entries
        if cursor != counts[row]:
            raise ValueError("placeholder runs and transferred entries differ in count")
    causal = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    eye = torch.eye(length, dtype=torch.bool, device=device)
    tokens = (causal[None] & token_columns[:, None, :]) | eye[None]         # every token attends itself (a placeholder: only itself)
    mask = torch.cat([prefix_visible, tokens], dim=2)[:, None]              # [B, 1, S, P + S], True = attend
    entry_columns = torch.arange(prefix, device=device)[None] < torch.tensor(counts, device=device)[:, None]
    columns = torch.cat([entry_columns, token_columns], dim=1)
    maskless = _maskless_layout(mask, columns, token_columns, run_ends, prefix) if MASKLESS and prefix else None
    if per_layer and maskless is None:
        raise ValueError("per-layer entry counts (kv_layer_counts) on a layout the maskless attention cannot reproduce")

    def layer_entries(layer: int) -> tuple[list[int], torch.Tensor, list[list[tuple[int, int, int]]]]:
        """This layer's entry count per row, their key positions [B, P_l] and runs (end, first, last) in the layer's entries:
        each run keeps the first kv_layer_counts[layer][run] of its entries (all of them without kv_layer_counts)."""
        totals, ranges = [0] * batch, [[] for _ in range(batch)]
        spans = []
        for row, payload in enumerate(payloads):
            layer_counts = payload.get(KV_LAYER_COUNTS) if counts[row] else None
            taken = 0
            for index, (end, first, last) in enumerate(run_ends[row]):
                count = int(layer_counts[layer][index]) if layer_counts else last - first
                if not 0 <= count <= last - first:
                    raise ValueError(f"layer {layer}: {count} entries for a run of {last - first}")
                spans.append((row, taken, first, count, run_starts[row][index]))
                ranges[row].append((end, taken, taken + count))
                taken += count
            totals[row] = taken
        positions = torch.zeros(batch, max(totals), dtype=torch.long, device=device)
        for row, taken, first, count, start in spans:
            positions[row, taken:taken + count] = start + relatives[row][first:first + count]
        return totals, positions, ranges

    keys, values, layer_layouts = [], [], []
    biased = any(counts[row] and payload.get(KV_BIAS) for row, payload in enumerate(payloads))
    biases = [] if biased else None
    for layer in range(num_layers):
        totals, positions, ranges = layer_entries(layer) if per_layer else (counts, key_positions, run_ends)
        width = max(totals)
        layer_keys = inputs_embeds.new_zeros(batch, width, heads, head_dim)
        layer_values = inputs_embeds.new_zeros(batch, width, heads, head_dim)
        for row, payload in enumerate(payloads):
            if totals[row]:
                row_keys, row_values = payload[KV_VALUES][layer]
                if row_keys.shape[0] != totals[row]:
                    raise ValueError(f"layer {layer}: {row_keys.shape[0]} transferred entries for {totals[row]}")
                layer_keys[row, :totals[row]] = row_keys.to(layer_keys)
                layer_values[row, :totals[row]] = row_values.to(layer_values)
        keys.append(rotate_keys(model, layer_keys.transpose(1, 2), positions))
        values.append(layer_values.transpose(1, 2))
        if biased:
            layer_bias = inputs_embeds.new_zeros(batch, heads, width, dtype=torch.float32)       # fp32: β of ±10 needs more than bf16
            for row, payload in enumerate(payloads):
                if totals[row] and payload.get(KV_BIAS):
                    layer_bias[row, :, :totals[row]] = payload[KV_BIAS][layer].to(layer_bias).transpose(0, 1)
            biases.append(layer_bias)
        if per_layer:
            layer_layouts.append(_layer_layout(maskless, totals, ranges, run_ends, token_columns))
    cache = PrefixKV(keys, values, grow=grow, biases=biases, maskless=layer_layouts if per_layer else maskless)
    from .hf_utils import text_decoder
    layer_types = getattr(text_decoder(model).config, "layer_types", None)
    if not layer_types:                                                      # e.g. Llama: one mask, taken as-is (4D) by the model
        return mask, position_ids, cache, columns
    kinds = set(layer_types) | {"full_attention"}
    return {kind: mask for kind in kinds}, position_ids, cache, columns


MASKLESS = False      # opt-in: prefix forwards through the sdpa wrapper run the fused lower-right causal kernel without the mask (MasklessLayout)


def _layer_layout(full: MasklessLayout, totals: list[int], ranges: list[list[tuple[int, int, int]]], run_ends: list[list[tuple[int, int, int]]],
                  token_columns: torch.Tensor) -> MasklessLayout:
    """One layer's MasklessLayout when its runs keep fewer entries (kv_layer_counts): the same groups (runs ending at the same
    index) and early tokens as the verified full layout, over this layer's entries."""
    batch, length = token_columns.shape
    width = max(totals)
    device = token_columns.device
    groups = torch.zeros(batch, width, full.groups.shape[-1], device=device)
    for row, runs in enumerate(ranges):
        ends = sorted({end for end, _, _ in run_ends[row]})
        for end, first, last in runs:
            groups[row, first:last, ends.index(end)] = 1.0
    entry_columns = torch.arange(width, device=device)[None] < torch.tensor(totals, device=device)[:, None]
    key_bias = torch.zeros(batch, width + length, device=device).masked_fill(~torch.cat([entry_columns, token_columns], dim=1), _MASKED_SCORE)
    return MasklessLayout(width, length, key_bias, full.early, groups)


def _maskless_layout(mask: torch.Tensor, columns: torch.Tensor, token_columns: torch.Tensor, run_ends: list[list[tuple[int, int, int]]],
                     prefix: int) -> MasklessLayout | None:
    """The MasklessLayout of a bound forward, or None when it would not reproduce `mask` [B, 1, S, P + S] on every attended row
    (more groups of run ends than spare head dimensions, or a visibility the kernel's causality does not give)."""
    batch, length = token_columns.shape
    device = mask.device
    ends = [sorted({end for end, _, _ in runs}) for runs in run_ends]
    group_count = max(1, max(len(row) for row in ends))
    if group_count > _ENTRY_BIAS_DIMS - 2:
        return None
    early = torch.zeros(batch, length, group_count, device=device)
    groups = torch.zeros(batch, prefix, group_count, device=device)
    for row, runs in enumerate(run_ends):
        for end, first, last in runs:
            group = ends[row].index(end)
            early[row, :end, group] = 1.0
            groups[row, first:last, group] = 1.0
    key_bias = torch.zeros(batch, prefix + length, device=device).masked_fill(~columns, _MASKED_SCORE)
    blocked = torch.einsum("bsg,bpg->bsp", early, groups) > 0
    causal = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    implied = columns[:, None, :] & torch.cat([~blocked, causal[None].expand(batch, -1, -1)], dim=2)
    attended = token_columns[:, :, None]                                     # rows a later computation reads: real tokens, no placeholders
    if bool(((implied != mask[:, 0]) & attended).any()):
        return None
    return MasklessLayout(prefix, length, key_bias, early, groups)


def _next_token(logits: torch.Tensor, do_sample: bool, temperature: float, top_k: int | None, top_p: float,
                generator: torch.Generator | None) -> torch.Tensor:
    if not do_sample:
        return logits.argmax(dim=-1)
    logits = logits.float() / max(temperature, 1e-6)
    if top_k:
        kth = logits.topk(min(int(top_k), logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        ordered, order = logits.sort(dim=-1, descending=True)
        probs = ordered.softmax(dim=-1)
        drop = probs.cumsum(dim=-1) - probs > top_p                      # keep the smallest prefix whose mass reaches top_p
        logits = logits.scatter(-1, order, ordered.masked_fill(drop, float("-inf")))
    return torch.multinomial(logits.softmax(dim=-1), 1, generator=generator)[:, 0]


@torch.no_grad()
def generate_with_kv_prefix(peft_model, *, inputs_embeds: torch.Tensor, interventions: t.Sequence[dict | None],
                            attention_mask: torch.Tensor | None = None, max_new_tokens: int = 20,
                            eos_token_id: int | list[int] | None = None, pad_token_id: int | None = None,
                            do_sample: bool = False, temperature: float = 1.0, top_k: int | None = None, top_p: float = 1.0,
                            generator: torch.Generator | None = None, **unused) -> torch.Tensor:
    """
    Greedy (default) or sampled decoding of B prompts [B, L, d] whose payloads carry transferred keys/values (left-padded with
    `attention_mask` when B > 1; every row ends on a real token). Sampling: `temperature`, then top-k, then nucleus top-p, as
    HF's warpers do. Returns the new token ids [B, T] (as `generate` does for embeddings-only input), padded with
    `pad_token_id` after a row's end token. The caller holds the adapter scope.
    """
    if unused.get("num_beams", 1) != 1:
        raise ValueError("generate_with_kv_prefix supports greedy or sampled decoding only")
    from .hf_utils import text_decoder
    base_model = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    decoder = text_decoder(base_model)
    head = base_model.get_output_embeddings()
    embed = base_model.get_input_embeddings()
    batch = inputs_embeds.shape[0]
    if attention_mask is not None and not bool(attention_mask[:, -1].all()):
        raise ValueError("generate_with_kv_prefix takes left-padded rows (every row ends on a real token)")
    if DROP_PLACEHOLDERS:
        inputs_embeds, attention_mask, interventions, _ = drop_placeholders(inputs_embeds, attention_mask, interventions)
    mask, position_ids, cache, columns = bind_kv_prefix(base_model, inputs_embeds, attention_mask, interventions, grow=True)
    cache.capacity = max_new_tokens                                         # maskless: room in each layer's buffer for every generated token
    hidden = decoder(inputs_embeds=inputs_embeds, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True,
                     **{KV_PREFIX_KWARG: cache}).last_hidden_state[:, -1]
    ends = [] if eos_token_id is None else ([eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id))
    pad = pad_token_id if pad_token_id is not None else (ends[0] if ends else 0)
    finished = torch.zeros(batch, dtype=torch.bool, device=inputs_embeds.device)
    position = position_ids[:, -1]
    kinds = list(mask) if isinstance(mask, dict) else None
    tokens = []
    for step in range(max_new_tokens):
        token = _next_token(head(hidden), do_sample, temperature, top_k, top_p, generator)
        token = torch.where(finished, torch.full_like(token, pad), token)
        tokens.append(token)
        for end in ends:
            finished = finished | (token == end)
        if bool(finished.all()) or step == max_new_tokens - 1:
            break
        position = position + 1
        columns = torch.cat([columns, torch.ones(batch, 1, dtype=torch.bool, device=columns.device)], dim=1)
        step_mask = columns[:, None, None, :]
        hidden = decoder(inputs_embeds=embed(token)[:, None].to(inputs_embeds.dtype), attention_mask=step_mask if kinds is None else {kind: step_mask for kind in kinds},
                         position_ids=position[:, None], use_cache=False, return_dict=True, **{KV_PREFIX_KWARG: cache}).last_hidden_state[:, -1]
    return torch.stack(tokens, dim=1)


def generate_with_kv_overwrite(peft_model, *, inputs_embeds: torch.Tensor, interventions: t.Sequence[dict | None],
                               attention_mask: torch.Tensor | None = None, **generate_kwargs):
    """
    Greedy `generate` for prompts whose payloads carry keys/values in overwrite mode (left-padded with `attention_mask` when
    B > 1). HF's own loop and cache: the overwrite is bound for the prefill only (the one forward that holds the row positions),
    through the base model's `prepare_inputs_for_generation`, as hf_utils.generate_with_interventions binds its payloads; the
    replaced keys and values go into the cache, so the decode steps attend them without further work.
    """
    import functools
    if generate_kwargs.get("num_beams", 1) != 1 or generate_kwargs.get("do_sample", False) or generate_kwargs.get("num_return_sequences", 1) != 1:
        raise ValueError("generate_with_kv_overwrite supports greedy decoding only")
    base_model = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    overwrite = bind_kv_overwrite(base_model, inputs_embeds, attention_mask, interventions)
    holder = getattr(base_model, base_model.base_model_prefix, None)
    restore_deltas = hasattr(holder, "rope_deltas")
    deltas = holder.rope_deltas if restore_deltas else None
    previous = base_model.__dict__.get("prepare_inputs_for_generation")
    original = base_model.prepare_inputs_for_generation
    @functools.wraps(original)
    def bound(*args, **kwargs):
        model_inputs = original(*args, **kwargs)
        cache = model_inputs.get("past_key_values")
        if cache is None or not hasattr(cache, "get_seq_length") or int(cache.get_seq_length()) == 0:    # the prefill
            model_inputs[KV_PREFIX_KWARG] = overwrite
        return model_inputs
    base_model.prepare_inputs_for_generation = bound
    try:
        extra = {} if attention_mask is None else {"attention_mask": attention_mask}
        return peft_model.generate(inputs_embeds=inputs_embeds, **extra, **generate_kwargs)
    finally:
        if previous is None:
            del base_model.__dict__["prepare_inputs_for_generation"]
        else:
            base_model.prepare_inputs_for_generation = previous
        if restore_deltas:
            holder.rope_deltas = deltas
