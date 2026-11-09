"""
HF-facing code for model configuration parsing, message formatting, etc.
Stores overly detailed/specific logic to keep the rest of code cleaner.
"""
from collections import Counter
from contextlib import contextmanager, nullcontext
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
    """Bind recomputation to the adapter already scoped around the original decoder forward."""
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
                    "use_reentrant": False, "context_fn": lambda: (nullcontext(), adapter_context())}))
            module._gradient_checkpointing_func = scoped
        yield
    finally:
        for module, checkpointed, original in previous:
            module.gradient_checkpointing = checkpointed
            if original is not None:
                module._gradient_checkpointing_func = original
            elif hasattr(module, "_gradient_checkpointing_func"):
                del module._gradient_checkpointing_func


@contextmanager
def peft_adapter_scope(model, adapter_name: str | None):
    """Route an existing PEFT model without resetting an already-correct ordinary LoRA.

    Trainability belongs to the optimizer: snapshot it live on every entry and restore
    it even when routing was unchanged. No parameter/layer references survive this scope.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils.other import AuxiliaryTrainingWrapper

    previous_adapter = model.active_adapter
    trainability = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    modules = list(model.modules())
    layers = [(module, module.disable_adapters) for module in modules if isinstance(module, BaseTunerLayer)]
    ordinary = not any(isinstance(module, AuxiliaryTrainingWrapper) for module in modules)
    ordinary = ordinary and not any(layer.merged for layer, _ in layers)
    adapters = tuple(model.peft_config)

    def route(name):
        names = [name] if isinstance(name, str) else list(name)
        if (not ordinary or tuple(model.peft_config) != adapters or model.active_adapter != name
                or model.base_model.active_adapter != name
                or any(layer.active_adapters != names or layer.merged for layer, _ in layers)):
            model.set_adapter(name)

    try:
        if adapter_name is not None:
            route(adapter_name)
        for layer, _ in layers:
            if not ordinary or layer.disable_adapters != (adapter_name is None):
                layer.enable_adapters(adapter_name is not None)
        for parameter, trainable in trainability:
            parameter.requires_grad_(trainable)
        yield
    finally:
        route(previous_adapter)
        for layer, disabled in layers:
            if not ordinary or layer.disable_adapters != disabled:
                layer.enable_adapters(not disabled)
        for parameter, trainable in trainability:
            parameter.requires_grad_(trainable)



# --------------------------------------------------------------------------------------- deep AC inputs (interventions)
INTERVENTIONS_KWARG = "interventions"
_HOOKS_ATTRIBUTE = "_activation_intervention_hooks"


class Interventions(t.TypedDict, total=False):
    """One logical sequence's deltas: `positions` index the unpadded physical row (prompt indices, sorted, unique); every
    `layer_inputs[layer]` is [M, d_model], added to that decoder layer's input at those positions. None means no intervention."""
    positions: list[int]
    layer_inputs: dict[int, torch.Tensor]
    row_rms: dict[int, torch.Tensor]     # written by the reader hooks: residual RMS at the row positions per intervened layer (0-d tensors; diagnostic)
    # Key/value transfer (kv_transfer.bind_kv_prefix; instead of `layer_inputs`): the row positions stay as inert placeholders and
    # every decoder layer attends the transferred entries.
    kv_values: dict[int, tuple[torch.Tensor, torch.Tensor]]   # layer -> (keys, values), each [M, kv_heads, head_dim], keys not rotated
    kv_positions: torch.Tensor           # [M] long: each entry's position in its encoder sequence
    kv_runs: list[tuple[int, int, int]]  # per part: (start, end) of its placeholder run and the length of the encoder sequence it stands for
    kv_overwrite: bool                   # overwrite mode: the rows are real positions and their own keys/values are replaced at the attention layers


class BoundInterventions:
    """The payloads of one decoder forward bound to that forward's cache offset (set once by the text-model pre-hook);
    travels in the layer kwargs, so a checkpointed layer recomputes with exactly the same tensors."""

    __slots__ = ("payloads", "offset", "_indices")

    def __init__(self, payloads: list[Interventions | None], offset: int) -> None:
        self.payloads = payloads
        self.offset = offset
        self._indices = {}

    def indices(self, row: int, length: int, device: torch.device):
        """Position metadata is fixed for this forward, including its checkpoint recomputation."""
        source_positions = tuple(self.payloads[row]["positions"])
        key = (source_positions, self.offset, length, device)
        if key not in self._indices:
            local = [position - self.offset for position in source_positions]
            selected = [index for index, position in enumerate(local) if 0 <= position < length]
            positions = torch.tensor([local[index] for index in selected], device=device, dtype=torch.long)
            rows = (None if len(selected) == len(local)
                    else torch.tensor(selected, device=device, dtype=torch.long))
            self._indices[key] = positions, rows
        return self._indices[key]


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
    rows to the layer input out of place. A forward without the kwarg is untouched. The engine mapping is this same
    operation: a per-request scatter-add on the hidden state entering the named layers, at the rows' prompt positions.
    """
    if getattr(model, _HOOKS_ATTRIBUTE, None) is not None:
        return
    text = text_decoder(model)
    handles = []

    def bind(module, args, kwargs):
        payloads = kwargs.get(INTERVENTIONS_KWARG)
        if payloads is None or isinstance(payloads, BoundInterventions):
            return None
        return args, {**kwargs, INTERVENTIONS_KWARG: BoundInterventions(list(payloads), _cache_offset(kwargs))}
    handles.append(text.register_forward_pre_hook(bind, with_kwargs=True))

    def layer_hook(index: int):
        def hook(module, args, kwargs):
            bound = kwargs.get(INTERVENTIONS_KWARG)
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
            for row, payload in enumerate(bound.payloads):
                if not payload:
                    continue
                rows = payload["layer_inputs"].get(index)
                if rows is None:
                    continue
                if len(payload["positions"]) != rows.shape[0]:
                    raise ValueError("positions and delta rows differ in count")
                local, selected = bound.indices(row, length, hidden.device)
                if local.numel() == 0:
                    continue
                # Diagnostic for the producer: the residual RMS at this forward's row positions (the deltas' true denominator).
                payload.setdefault("row_rms", {})[index] = hidden[row, local].detach().float().pow(2).mean(dim=-1).sqrt().mean()   # 0-d tensor: no host sync here
                if output is None:
                    output = hidden.clone()
                rows = rows if selected is None else rows.index_select(0, selected)
                output[row].index_add_(0, local, rows.to(device=hidden.device, dtype=hidden.dtype))
            return ((output if output is not None else hidden, *args[1:]), kwargs)
        return hook
    for index, layer in enumerate(decoder_layers(model)):
        handles.append(layer.register_forward_pre_hook(layer_hook(index), with_kwargs=True))
    setattr(model, _HOOKS_ATTRIBUTE, handles)


def generate_with_interventions(peft_model, *, inputs_embeds: torch.Tensor, interventions: list[Interventions | None], **generate_kwargs):
    """
    Greedy, single-sequence `generate` from embeddings whose prompt rows carry deltas. HF validates generate's kwargs
    against the model signature (an `interventions=` kwarg is rejected), so the payload is bound for this call only by
    wrapping the base model's `prepare_inputs_for_generation` (the one HF's loop calls every step under a PEFT wrapper
    too; the wrapper keeps the original signature for HF's validation and is removed in `finally`). The layer hooks
    map every step's cache offset, so the row deltas land on the prompt at prefill and on nothing afterwards.
    """
    import functools
    if inputs_embeds.dim() != 3 or inputs_embeds.shape[0] != 1 or len(interventions) != 1:
        raise ValueError("generate_with_interventions takes one sequence and one payload")
    from .kv_transfer import generate_with_kv_overwrite, generate_with_kv_prefix, has_kv, is_overwrite
    if is_overwrite(interventions):          # key/value transfer, overwrite mode: HF's own loop, the overwrite bound at the prefill
        return generate_with_kv_overwrite(peft_model, inputs_embeds=inputs_embeds, interventions=interventions, **generate_kwargs)
    if has_kv(interventions):                # key/value transfer: its own prefill and greedy loop
        return generate_with_kv_prefix(peft_model, inputs_embeds=inputs_embeds, interventions=interventions, **generate_kwargs)
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


def generate_batch_with_interventions(peft_model, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
                                      interventions: list[Interventions | None], **generate_kwargs):
    """
    Greedy or sampled `generate` (one sequence per row) for B rows at once: LEFT-padded embeddings [B, L, d] with their mask [B, L] (0 on the pads) and one payload
    (or None) per row whose positions are already in the padded row's coordinates (the unpadded position plus the row's pad
    count). Left padding is what keeps every row equal to its single-sequence run: HF derives the positions from the mask (a
    row's first real token is position 0), the full-attention layers mask the pads, and a linear-attention layer zeroes the
    pads' hidden states (`create_recurrent_attention_mask`), so its bias-free projections, convolution window and recurrent
    state are still exactly zero at the row's first real token. Right padding would run the recurrence over the pads before
    the first new token. The payload is bound only while the forward can still reach a row position (the prefill): the decode
    steps would add nothing, and run without the per-layer hook work.
    """
    import functools
    if inputs_embeds.dim() != 3 or attention_mask.shape != inputs_embeds.shape[:2] or len(interventions) != inputs_embeds.shape[0]:
        raise ValueError("generate_batch_with_interventions takes [B, L, d] embeddings, their [B, L] mask and one payload per row")
    if not bool(attention_mask[:, -1].all()):
        raise ValueError("generate_batch_with_interventions takes left-padded rows (every row ends on a real token)")
    from .kv_transfer import generate_with_kv_overwrite, generate_with_kv_prefix, has_kv, is_overwrite
    if is_overwrite(interventions):          # key/value transfer, overwrite mode: HF's own loop, the overwrite bound at the prefill
        return generate_with_kv_overwrite(peft_model, inputs_embeds=inputs_embeds, attention_mask=attention_mask, interventions=interventions, **generate_kwargs)
    if has_kv(interventions):                # key/value transfer: its own prefill and greedy loop
        return generate_with_kv_prefix(peft_model, inputs_embeds=inputs_embeds, attention_mask=attention_mask, interventions=interventions, **generate_kwargs)
    last = max((max(payload["positions"]) for payload in interventions if payload and payload["positions"]), default=-1)
    # Sampling keeps each row in place, so the payload rows still line up; beams and several sequences per row would not.
    if last >= 0 and (generate_kwargs.get("num_beams", 1) != 1 or generate_kwargs.get("num_return_sequences", 1) != 1):
        raise ValueError("generate_batch_with_interventions supports one sequence per row (greedy or sampled) when rows carry payloads (without any: HF's own generate)")
    base_model = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    # A multimodal wrapper keeps `rope_deltas` of the last generate's batch size, and a later embeddings-only forward of another
    # batch size would read them: put back what was there.
    holder = getattr(base_model, base_model.base_model_prefix, None)
    restore_deltas = hasattr(holder, "rope_deltas")
    deltas = holder.rope_deltas if restore_deltas else None
    try:
        if last < 0:
            return peft_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs)
        install_intervention_hooks(base_model)
        previous = base_model.__dict__.get("prepare_inputs_for_generation")
        original = base_model.prepare_inputs_for_generation
        @functools.wraps(original)
        def bound(*args, **kwargs):
            model_inputs = original(*args, **kwargs)
            if _cache_offset(model_inputs) <= last:
                model_inputs[INTERVENTIONS_KWARG] = interventions
            return model_inputs
        base_model.prepare_inputs_for_generation = bound
        try:
            return peft_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs)
        finally:
            if previous is None:
                del base_model.__dict__["prepare_inputs_for_generation"]
            else:
                base_model.prepare_inputs_for_generation = previous
    finally:
        if restore_deltas:
            holder.rope_deltas = deltas


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
