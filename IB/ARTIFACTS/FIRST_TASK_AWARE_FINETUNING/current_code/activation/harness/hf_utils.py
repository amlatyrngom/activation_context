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
    """One logical sequence's deltas: `positions` index the unpadded physical row (prompt indices, sorted, unique); every
    `layer_inputs[layer]` is [M, d_model], added to that decoder layer's input at those positions. None means no intervention."""
    positions: list[int]
    layer_inputs: dict[int, torch.Tensor]
    row_rms: dict[int, torch.Tensor]     # written by the reader hooks: residual RMS at the row positions per intervened layer (0-d tensors; diagnostic)


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
                positions = torch.as_tensor(payload["positions"], device=hidden.device, dtype=torch.long)
                if positions.numel() != rows.shape[0]:
                    raise ValueError("positions and delta rows differ in count")
                local = positions - bound.offset
                keep = (local >= 0) & (local < length)
                if not bool(keep.any()):
                    continue
                # Diagnostic for the producer: the residual RMS at this forward's row positions (the deltas' true denominator).
                payload.setdefault("row_rms", {})[index] = hidden[row, local[keep]].detach().float().pow(2).mean(dim=-1).sqrt().mean()   # 0-d tensor: no host sync here
                if output is None:
                    output = hidden.clone()
                output[row].index_add_(0, local[keep], rows[keep].to(device=hidden.device, dtype=hidden.dtype))
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
