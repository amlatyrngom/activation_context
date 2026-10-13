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
    layer run twice returns the same tensors); `grow=True`: the returned tensors become the stored ones (generation)."""

    def __init__(self, keys: list[torch.Tensor], values: list[torch.Tensor], grow: bool = False) -> None:
        self.keys, self.values, self.grow = keys, values, grow

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
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


def install_kv_hooks(model) -> None:
    """Persistent, parameter-free pre-hooks (once per model): on every attention module a `kv_prefix=` keyword is taken out and
    handed over as `past_key_values` (an OverwriteKV first bound to the cache the call came with); on a recurrent block's mixer
    the keyword is only taken out. A forward without the keyword is untouched."""
    if getattr(model, _HOOKS_ATTRIBUTE, None) is not None:
        return
    check_kv_model(model, every_layer=False)
    def hook(module, args, kwargs):
        if KV_PREFIX_KWARG not in kwargs:
            return None
        kwargs = dict(kwargs)
        prefix = kwargs.pop(KV_PREFIX_KWARG)
        kwargs["past_key_values"] = prefix.bound(kwargs.get("past_key_values")) if isinstance(prefix, OverwriteKV) else prefix
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


def has_kv(payloads: t.Sequence[dict | None] | None) -> bool:
    return bool(payloads) and any(payload and payload.get(KV_VALUES) for payload in payloads)


def bind_kv_prefix(model, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None, payloads: t.Sequence[dict | None],
                   grow: bool = False) -> tuple[dict, torch.Tensor, PrefixKV, torch.Tensor]:
    """
    (attention mask mapping, position ids [B, S], PrefixKV, column visibility [B, P + S]) for a reader forward of
    `inputs_embeds` [B, S, d] whose rows carry one payload each (None: no transfer for that row):

        kv_values     {layer: (keys, values)}, each [M, kv_heads, head_dim], keys not rotated, every decoder layer present
        kv_positions  [M] long: each entry's position in its encoder sequence
        kv_runs       [(start, end, length)] per part, in order: the placeholder run in this row's physical coordinates
                      (end - start entries) and the length of the encoder sequence it stands for

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
    for row, payload in enumerate(payloads):
        if not counts[row]:
            continue
        if payload.get("layer_inputs"):
            raise ValueError("a payload carries either residual deltas or transferred keys/values, not both")
        if sorted(payload[KV_VALUES]) != list(range(num_layers)):
            raise ValueError(f"transferred keys/values must cover every decoder layer 0..{num_layers - 1}")
        relative = payload[KV_POSITIONS].to(device)
        cursor = 0
        for start, end, span in payload[KV_RUNS]:
            entries = end - start
            if not 0 <= start <= end <= length or span < entries:
                raise ValueError(f"placeholder run ({start}, {end}) standing for {span} positions does not fit a row of {length}")
            key_positions[row, cursor:cursor + entries] = position_ids[row, start] + relative[cursor:cursor + entries]
            position_ids[row, end:] += span - entries
            prefix_visible[row, end:, cursor:cursor + entries] = True
            token_columns[row, start:end] = False
            cursor += entries
        if cursor != counts[row]:
            raise ValueError("placeholder runs and transferred entries differ in count")
    causal = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    eye = torch.eye(length, dtype=torch.bool, device=device)
    tokens = (causal[None] & token_columns[:, None, :]) | eye[None]         # every token attends itself (a placeholder: only itself)
    mask = torch.cat([prefix_visible, tokens], dim=2)[:, None]              # [B, 1, S, P + S], True = attend
    keys, values = [], []
    for layer in range(num_layers):
        layer_keys = inputs_embeds.new_zeros(batch, prefix, heads, head_dim)
        layer_values = inputs_embeds.new_zeros(batch, prefix, heads, head_dim)
        for row, payload in enumerate(payloads):
            if counts[row]:
                row_keys, row_values = payload[KV_VALUES][layer]
                layer_keys[row, :counts[row]] = row_keys.to(layer_keys)
                layer_values[row, :counts[row]] = row_values.to(layer_values)
        keys.append(rotate_keys(model, layer_keys.transpose(1, 2), key_positions))
        values.append(layer_values.transpose(1, 2))
    entry_columns = torch.arange(prefix, device=device)[None] < torch.tensor(counts, device=device)[:, None]
    columns = torch.cat([entry_columns, token_columns], dim=1)
    from .hf_utils import text_decoder
    kinds = set(getattr(text_decoder(model).config, "layer_types", None) or ["full_attention"]) | {"full_attention"}
    return {kind: mask for kind in kinds}, position_ids, PrefixKV(keys, values, grow=grow), columns


@torch.no_grad()
def generate_with_kv_prefix(peft_model, *, inputs_embeds: torch.Tensor, interventions: t.Sequence[dict | None],
                            attention_mask: torch.Tensor | None = None, max_new_tokens: int = 20,
                            eos_token_id: int | list[int] | None = None, pad_token_id: int | None = None, **unused) -> torch.Tensor:
    """
    Greedy decoding of B prompts [B, L, d] whose payloads carry transferred keys/values (left-padded with `attention_mask`
    when B > 1; every row ends on a real token). Returns the new token ids [B, T] (as `generate` does for embeddings-only
    input), padded with `pad_token_id` after a row's end token. The caller holds the adapter scope.
    """
    if unused.get("do_sample") or unused.get("num_beams", 1) != 1:
        raise ValueError("generate_with_kv_prefix supports greedy decoding only")
    from .hf_utils import text_decoder
    base_model = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    decoder = text_decoder(base_model)
    head = base_model.get_output_embeddings()
    embed = base_model.get_input_embeddings()
    batch = inputs_embeds.shape[0]
    if attention_mask is not None and not bool(attention_mask[:, -1].all()):
        raise ValueError("generate_with_kv_prefix takes left-padded rows (every row ends on a real token)")
    mask, position_ids, cache, columns = bind_kv_prefix(base_model, inputs_embeds, attention_mask, interventions, grow=True)
    hidden = decoder(inputs_embeds=inputs_embeds, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True,
                     **{KV_PREFIX_KWARG: cache}).last_hidden_state[:, -1]
    ends = [] if eos_token_id is None else ([eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id))
    pad = pad_token_id if pad_token_id is not None else (ends[0] if ends else 0)
    finished = torch.zeros(batch, dtype=torch.bool, device=inputs_embeds.device)
    position = position_ids[:, -1]
    kinds = list(mask)
    tokens = []
    for step in range(max_new_tokens):
        token = head(hidden).argmax(dim=-1)
        token = torch.where(finished, torch.full_like(token, pad), token)
        tokens.append(token)
        for end in ends:
            finished = finished | (token == end)
        if bool(finished.all()) or step == max_new_tokens - 1:
            break
        position = position + 1
        columns = torch.cat([columns, torch.ones(batch, 1, dtype=torch.bool, device=columns.device)], dim=1)
        step_mask = columns[:, None, None, :]
        hidden = decoder(inputs_embeds=embed(token)[:, None].to(inputs_embeds.dtype), attention_mask={kind: step_mask for kind in kinds},
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
