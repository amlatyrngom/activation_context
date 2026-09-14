"""
The messy parts of agent training, kept out of the trainer: turning a trajectory into one training
sequence with its loss mask and stored log-probs, grouping sequences into micro-batches (one each,
padded, or packed into one row), padding, and the chunked log-prob computation that avoids materialising the full logits of a long
sequence (16k positions x a 250k vocabulary would not fit).
"""
from __future__ import annotations

import typing as t
from dataclasses import dataclass

import torch
import torch.utils.checkpoint

if t.TYPE_CHECKING:
    from .agent_trainer import AgentTrainingItem


@dataclass
class TrainingExample:
    """One trajectory as the model read it: the first prompt, then every step's token segment."""
    token_ids: list[int]
    loss_mask: list[bool]          # True at positions whose token carries loss (predicted from the previous position)
    old_logprobs: list[float]      # aligned with token_ids; the engine's log-prob where loss_mask is True, 0.0 elsewhere
    weight: float                  # the advantage A of the whole trajectory
    item_index: int
    needs_old_logprobs: bool       # teacher / hinted items (or runs recorded without log-probs): pi_old comes from a no-grad pass

    @property
    def num_loss_tokens(self) -> int:
        return sum(self.loss_mask)


def build_example(item: "AgentTrainingItem", item_index: int) -> TrainingExample | None:
    """
    The run's prompt_token_ids followed by each trajectory step's token_ids; assistant segments carry
    loss on their sampled tokens (the ones with a recorded log-prob; all of them for ignore_logprobs
    items). None when the run has no token segments or no sampled tokens.
    """
    run = item.run_results
    token_ids = list(run.prompt_token_ids)
    if not token_ids:
        return None
    loss_mask = [False] * len(token_ids)
    old_logprobs = [0.0] * len(token_ids)
    any_recorded = False
    for step in run.trajectory:
        segment = list(step.get("token_ids") or [])
        if step.get("role") == "assistant":
            logprobs = [float(value) for value in (step.get("logprobs") or [])]
            recorded = min(len(logprobs), len(segment))
            any_recorded = any_recorded or recorded > 0
            sampled = len(segment) if item.ignore_logprobs or recorded == 0 else recorded
            loss_mask += [True] * sampled + [False] * (len(segment) - sampled)
            if item.ignore_logprobs or recorded == 0:
                old_logprobs += [0.0] * len(segment)
            else:
                old_logprobs += logprobs[:sampled] + [0.0] * (len(segment) - sampled)
        else:
            loss_mask += [False] * len(segment)
            old_logprobs += [0.0] * len(segment)
        token_ids += segment
    loss_mask[0] = False                                                   # the first token is never predicted
    if not any(loss_mask):
        return None
    return TrainingExample(
        token_ids=token_ids, loss_mask=loss_mask, old_logprobs=old_logprobs, weight=float(item.weight), item_index=item_index,
        needs_old_logprobs=item.ignore_logprobs or not any_recorded,
    )


def micro_batches(examples: list[TrainingExample], token_budget: int, pad_free: bool = True, packed: bool = False) -> list[list[int]]:
    """
    Indices grouped into micro-batches, longest first. Packed: examples concatenated into one row
    under `token_budget` tokens (no padding, per-sequence attention; see `collate(packed=True)`).
    Pad-free: one example each (no padding, so the attention layers run their causal fast path and
    no token is wasted). Otherwise examples are padded to the longest under `token_budget` padded
    tokens per micro-batch (longest-first bins keep the padding small).
    """
    order = sorted(range(len(examples)), key=lambda index: -len(examples[index].token_ids))
    if pad_free and not packed:
        return [[index] for index in order]
    groups: list[list[int]] = []
    for index in order:
        length = len(examples[index].token_ids)
        for group in groups:
            if packed:
                fits = sum(len(examples[i].token_ids) for i in group) + length <= token_budget
            else:
                width = max(len(examples[i].token_ids) for i in group)
                fits = max(width, length) * (len(group) + 1) <= token_budget
            if fits:
                group.append(index)
                break
        else:
            groups.append([index])
    return groups


@dataclass
class Collated:
    input_ids: torch.Tensor        # [B, S]
    attention_mask: torch.Tensor   # [B, S] real tokens (statistics)
    position_ids: torch.Tensor     # [B, S]; packed: restarts at 0 for every example
    loss_mask: torch.Tensor        # [B, S] bool
    old_logprobs: torch.Tensor     # [B, S] float32
    weights: torch.Tensor          # [E] float32, one per example in batch order
    example_indices: list[int]     # into the examples list, batch order
    segment_ids: torch.Tensor      # [B, S] long: which example (batch order) each position belongs to
    segment_offsets: list[int]     # where each example starts in its row (0 unless packed)
    model_attention_mask: torch.Tensor | None = None   # what the decoder gets: None = causal only (right padding never reaches a real token)
    cu_seq_lens: torch.Tensor | None = None            # packed: int32 [E + 1] boundaries, for the linear-attention kernels (fla varlen)

    @property
    def packed(self) -> bool:
        return self.cu_seq_lens is not None


def collate(examples: list[TrainingExample], indices: list[int], pad_token_id: int, device, length_multiple: int = 1,
            causal_only: bool | None = None, packed: bool = False, spacer_tokens: int = 0) -> Collated:
    """
    Right-padded tensors on `device`. `attention_mask` marks the real tokens (statistics); the decoder
    gets `model_attention_mask`: None when nothing but tail padding is present (`causal_only`, the
    default for a single example), since under causal attention a real token never sees a later pad and
    the pads carry no loss. `length_multiple` rounds the length up so the compiled kernels see few shapes.

    Packed: every example concatenated into one row with no padding and its positions restarting at 0.
    transformers detects the resets (attention_mask None) and gives the attention layers a per-sequence
    block mask (flex: a BlockMask; sdpa: a dense one, slow; flash: cu_seqlens), and `cu_seq_lens` goes
    to the gated-delta layers so their state restarts too (the `fla` kernels honour it; the torch fallback
    would not). Their short causal conv would still see the previous example's last tokens at a boundary
    (measured: 5x the bf16 noise), so `spacer_tokens` (the conv width minus one, see `packing_spacer_tokens`)
    of pad go between examples as their own tiny sequence; `packed_attention_masks` hands the gated-delta
    layers a mask that zeroes them, which is exactly the zero padding a fresh sequence's conv sees.
    """
    if packed:
        return _collate_packed(examples, indices, pad_token_id, device, length_multiple, spacer_tokens)
    length = max(len(examples[index].token_ids) for index in indices)
    length = -(-length // max(1, length_multiple)) * max(1, length_multiple)
    if causal_only is None:
        causal_only = len(indices) == 1
    rows, attention, loss, old, weights = [], [], [], [], []
    for index in indices:
        example = examples[index]
        pad = length - len(example.token_ids)
        rows.append(example.token_ids + [pad_token_id] * pad)
        attention.append([1] * len(example.token_ids) + [0] * pad)
        loss.append(example.loss_mask + [False] * pad)
        old.append(example.old_logprobs + [0.0] * pad)
        weights.append(example.weight)
    attention_mask = torch.tensor(attention, dtype=torch.long)
    return Collated(
        input_ids=torch.tensor(rows, dtype=torch.long, device=device),
        attention_mask=attention_mask.to(device),
        position_ids=(attention_mask.cumsum(dim=1) - 1).clamp(min=0).to(device),
        loss_mask=torch.tensor(loss, dtype=torch.bool, device=device),
        old_logprobs=torch.tensor(old, dtype=torch.float32, device=device),
        weights=torch.tensor(weights, dtype=torch.float32, device=device),
        example_indices=list(indices),
        segment_ids=torch.arange(len(indices), device=device)[:, None].expand(len(indices), length).contiguous(),
        segment_offsets=[0] * len(indices),
        model_attention_mask=None if causal_only else attention_mask.to(device),
    )


def _collate_packed(examples: list[TrainingExample], indices: list[int], pad_token_id: int, device, length_multiple: int = 1,
                    spacer_tokens: int = 0) -> Collated:
    """One row; spacers between examples and a `length_multiple` tail are pad segments (their own sequence, no loss, masked)."""
    tokens, positions, loss, old, segments, offsets, boundaries, real = [], [], [], [], [], [], [0], []

    def pad_segment(count: int):
        tokens.extend([pad_token_id] * count)
        positions.extend(range(count))
        loss.extend([False] * count)
        old.extend([0.0] * count)
        segments.extend([len(indices)] * count)
        real.extend([0] * count)
        boundaries.append(len(tokens))

    for ordinal, index in enumerate(indices):
        if ordinal and spacer_tokens:
            pad_segment(spacer_tokens)
        example = examples[index]
        length = len(example.token_ids)
        offsets.append(len(tokens))
        tokens += example.token_ids
        positions += list(range(length))
        loss += example.loss_mask
        old += example.old_logprobs
        segments += [ordinal] * length
        real += [1] * length
        boundaries.append(len(tokens))
    pad = -(-len(tokens) // max(1, length_multiple)) * max(1, length_multiple) - len(tokens)
    if pad:
        pad_segment(pad)
    return Collated(
        input_ids=torch.tensor([tokens], dtype=torch.long, device=device),
        attention_mask=torch.tensor([real], dtype=torch.long, device=device),
        position_ids=torch.tensor([positions], dtype=torch.long, device=device),
        loss_mask=torch.tensor([loss], dtype=torch.bool, device=device),
        old_logprobs=torch.tensor([old], dtype=torch.float32, device=device),
        weights=torch.tensor([examples[index].weight for index in indices], dtype=torch.float32, device=device),
        example_indices=list(indices),
        segment_ids=torch.tensor([segments], dtype=torch.long, device=device),
        segment_offsets=offsets,
        model_attention_mask=None,
        cu_seq_lens=torch.tensor(boundaries, dtype=torch.int32, device=device),
    )


class _HeadMatmul(torch.autograd.Function):
    """
    logits = states @ weight.T with bf16 operands on the tensor cores and fp32 accumulation and output
    (`torch.mm(out_dtype=float32)`): the products of two bf16 values are exact in fp32, so this equals the
    fp32 GEMM of the upcast operands up to summation order, at a fraction of the SIMT fp32 kernel's time
    and without the fp32 copies of the states and the 2.5 GB weight. The head is frozen: only the gradient
    to the states is formed, from the fp32 gradient split into a high and a low bf16 part (two GEMMs,
    ~2^-16 relative error instead of bf16's 2^-8).
    """

    @staticmethod
    def forward(ctx, states, weight):
        ctx.save_for_backward(weight)
        ctx.states_dtype = states.dtype
        return torch.mm(states, weight.t(), out_dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_logits):
        (weight,) = ctx.saved_tensors
        high = grad_logits.to(weight.dtype)
        low = (grad_logits - high.float()).to(weight.dtype)
        grad_states = torch.mm(high, weight, out_dtype=torch.float32) + torch.mm(low, weight, out_dtype=torch.float32)
        return grad_states.to(ctx.states_dtype), None


def head_logits(states: torch.Tensor, head: torch.nn.Module, fp32_matmul: bool = False) -> torch.Tensor:
    """fp32 logits of `states` through the frozen output head (see _HeadMatmul; the fp32 GEMM on CPU or on request)."""
    weight = head.weight
    bias = getattr(head, "bias", None)
    if fp32_matmul or not states.is_cuda or weight.dtype not in (torch.bfloat16, torch.float16) or weight.dtype != states.dtype:
        logits = torch.nn.functional.linear(states.float(), weight.float(), bias.float() if bias is not None else None)
    else:
        logits = _HeadMatmul.apply(states, weight)
        if bias is not None:
            logits = logits + bias.float()
    return logits


def sampled_logprobs(hidden: torch.Tensor, head: torch.nn.Module, batch: Collated, chunk_tokens: int,
                     fp32_head_matmul: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    log pi_theta of every loss token: the decoder's hidden state at the previous position through the
    output head, in chunks, with fp32 logits (`head_logits`): bf16 logits of magnitude 20-30 are quantised
    in steps of 0.125-0.25, which alone would put several percent of the tokens outside a 0.2 ratio clip at
    the first step. Returns (logprobs [N], example of each in batch order [N], target ids [N]). The previous
    position is always inside the same example: an example's first token never carries loss (build_example).
    """
    rows, positions = batch.loss_mask.nonzero(as_tuple=True)
    states = hidden[rows, positions - 1]                                                   # [N, d]
    targets = batch.input_ids[rows, positions]                                             # [N]
    segments = batch.segment_ids[rows, positions]                                          # [N]

    def chunk_logprobs(chunk_states, chunk_targets):
        logits = head_logits(chunk_states, head, fp32_head_matmul)                          # [n, V] fp32
        return torch.log_softmax(logits, dim=-1).gather(-1, chunk_targets[:, None])[:, 0]

    pieces = []
    for start in range(0, states.shape[0], chunk_tokens):
        # Checkpointed: autograd would otherwise keep every chunk's [n, V] fp32 log-softmax (2 GB per 2,048
        # positions) until backward; recomputing the chunk there keeps the head at a constant footprint.
        pieces.append(torch.utils.checkpoint.checkpoint(
            chunk_logprobs, states[start:start + chunk_tokens], targets[start:start + chunk_tokens], use_reentrant=False,
        ))
    return torch.cat(pieces) if pieces else states.new_zeros(0, dtype=torch.float32), segments, targets


def clipped_surrogate(logprobs: torch.Tensor, old_logprobs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Per token: min(s A, clip(s, 1-eps, 1+eps) A) with s = exp(logp - old), A the sample's weight.
    Correct for negative weights (the PPO form). Returns the token terms and the ratio statistics.
    """
    ratio = torch.exp(logprobs - old_logprobs)
    surrogate = torch.minimum(ratio * weights, ratio.clamp(1 - epsilon, 1 + epsilon) * weights)
    with torch.no_grad():
        stats = {
            "mean_ratio": float(ratio.mean()) if ratio.numel() else 1.0,
            "clip_fraction": float(((ratio - 1).abs() > epsilon).float().mean()) if ratio.numel() else 0.0,
            "mean_logprob": float(logprobs.mean()) if logprobs.numel() else 0.0,
        }
    return surrogate, stats


def set_attention_implementation(model, name: str) -> None:
    """
    The base model's attention implementation for training. Flex is forced on models that do not declare
    it (Qwen3.5 does not; its text-only path works: the block mask comes from transformers' generic
    masking and the attention layers dispatch through the same interface as sdpa).
    """
    if name == "flex_attention" and not getattr(type(model), "_supports_flex_attn", False):
        print(f"{type(model).__name__} does not declare flex attention support; forcing it for the text-only training path", flush=True)
        for module in model.modules():
            cls = type(module)
            if hasattr(cls, "_supports_flex_attn") and not getattr(cls, "_supports_flex_attn"):
                cls._supports_flex_attn = True
    if model.config._attn_implementation != name:
        model.set_attn_implementation(name)


CHECKPOINT_FREE_TOKENS_ON_LARGE_CARDS = 6_144      # micro-batches below this keep their activations on a card with >= 80 GB


def checkpointing_min_tokens(config, device) -> int:
    """The configured threshold, or the default for the card (see AgentTrainingConfig.gradient_checkpointing_min_tokens)."""
    if config.gradient_checkpointing_min_tokens is not None:
        return int(config.gradient_checkpointing_min_tokens)
    if getattr(device, "type", None) == "cuda" and torch.cuda.get_device_properties(device).total_memory >= 80e9:
        return CHECKPOINT_FREE_TOKENS_ON_LARGE_CARDS
    return 0


def set_checkpointing(model, config, num_tokens: int, min_tokens: int = 0) -> bool:
    """Gradient checkpointing on for this micro-batch when configured and the batch is long enough; returns the state."""
    wanted = bool(config.gradient_checkpointing) and num_tokens >= min_tokens
    if wanted != bool(model.is_gradient_checkpointing):
        if wanted:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
    return wanted


def _text_config(model):
    return model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config


def packing_spacer_tokens(model) -> int:
    """Pad tokens between packed examples: the gated-delta conv width minus one (0 for a plain attention model)."""
    config = _text_config(model)
    if "linear_attention" not in (getattr(config, "layer_types", None) or []):
        return 0
    return max(0, int(getattr(config, "linear_conv_kernel_dim", 4)) - 1)


def packed_attention_masks(model, inputs_embeds: torch.Tensor, batch: Collated):
    """
    The decoder's attention_mask argument for a packed row: None for a plain attention model (transformers
    finds the sequences from the position resets), else the per-layer-type dict the hybrid model builds
    itself, with the packed causal mask for the attention layers and the real-token mask for the
    gated-delta layers (which zero their input where it is 0: spacers and tail pads).
    """
    config = _text_config(model)
    layer_types = getattr(config, "layer_types", None) or []
    if "linear_attention" not in layer_types:
        return None
    from transformers.masking_utils import create_causal_mask
    full = create_causal_mask(config, inputs_embeds, None, None, batch.position_ids)
    masks = {layer_type: full for layer_type in set(layer_types)}
    masks["linear_attention"] = batch.attention_mask.bool()
    return masks


def check_packing_support(model) -> None:
    """
    Packed rows need the gated-delta kernels that take `cu_seqlens` (transformers routes to the `fla`
    package or a hub kernel; its pure torch fallback ignores the boundaries and would carry state across examples).
    """
    if "linear_attention" not in (getattr(_text_config(model), "layer_types", None) or []):
        return
    try:
        import fla  # noqa: F401
    except ImportError as error:
        raise RuntimeError("pack_micro_batches needs the flash-linear-attention package (fla) for per-example recurrent state") from error


def disable_dropout(module: torch.nn.Module) -> int:
    """Dropout layers to eval while the rest trains (checkpointing needs training mode). Returns how many."""
    count = 0
    for child in module.modules():
        if isinstance(child, torch.nn.Dropout):
            child.eval()
            count += 1
    return count
