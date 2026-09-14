"""
The messy parts of agent training, kept out of the trainer: turning a trajectory into one training
sequence with its loss mask and stored log-probs, packing sequences into token-budgeted micro-batches,
padding, and the chunked log-prob computation that avoids materialising the full logits of a long
sequence (16k positions x a 250k vocabulary would not fit).
"""
from __future__ import annotations

import typing as t
from dataclasses import dataclass

import torch

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


def micro_batches(examples: list[TrainingExample], token_budget: int, pad_free: bool = True) -> list[list[int]]:
    """
    Indices grouped into micro-batches, longest first. Pad-free: one example each (no padding, so the
    attention layers run their causal fast path and no token is wasted). Otherwise examples are packed
    under `token_budget` padded tokens per micro-batch (longest-first bins keep the padding small).
    """
    order = sorted(range(len(examples)), key=lambda index: -len(examples[index].token_ids))
    if pad_free:
        return [[index] for index in order]
    groups: list[list[int]] = []
    for index in order:
        length = len(examples[index].token_ids)
        for group in groups:
            width = max(len(examples[i].token_ids) for i in group)
            if max(width, length) * (len(group) + 1) <= token_budget:
                group.append(index)
                break
        else:
            groups.append([index])
    return groups


@dataclass
class Collated:
    input_ids: torch.Tensor        # [B, S]
    attention_mask: torch.Tensor   # [B, S]
    position_ids: torch.Tensor     # [B, S]
    loss_mask: torch.Tensor        # [B, S] bool
    old_logprobs: torch.Tensor     # [B, S] float32
    weights: torch.Tensor          # [B] float32
    example_indices: list[int]     # into the examples list, batch order
    model_attention_mask: torch.Tensor | None = None   # what the decoder gets: None = causal only (right padding never reaches a real token)


def collate(examples: list[TrainingExample], indices: list[int], pad_token_id: int, device, length_multiple: int = 1,
            causal_only: bool | None = None) -> Collated:
    """
    Right-padded tensors on `device`. `attention_mask` marks the real tokens (statistics); the decoder
    gets `model_attention_mask`: None when nothing but tail padding is present (`causal_only`, the
    default for a single example), since under causal attention a real token never sees a later pad and
    the pads carry no loss. `length_multiple` rounds the length up so the compiled kernels see few shapes.
    """
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
        model_attention_mask=None if causal_only else attention_mask.to(device),
    )


def sampled_logprobs(hidden: torch.Tensor, head: torch.nn.Module, batch: Collated, chunk_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    log pi_theta of every loss token: the decoder's hidden state at the previous position through the
    output head, in chunks. The head runs in fp32 (its weight cast once per call; it is frozen, so no
    gradient flows through the cast): bf16 logits of magnitude 20-30 are quantised in steps of 0.125-0.25,
    which alone would put several percent of the tokens outside a 0.2 ratio clip at the first step.
    Returns (logprobs [N], batch row of each [N], target ids [N]).
    """
    rows, positions = batch.loss_mask.nonzero(as_tuple=True)
    states = hidden[rows, positions - 1]                                                   # [N, d]
    targets = batch.input_ids[rows, positions]                                             # [N]
    with torch.no_grad():
        weight = head.weight.float()
        bias = head.bias.float() if getattr(head, "bias", None) is not None else None
    pieces = []
    for start in range(0, states.shape[0], chunk_tokens):
        logits = torch.nn.functional.linear(states[start:start + chunk_tokens].float(), weight, bias)   # [n, V] fp32
        pieces.append(torch.log_softmax(logits, dim=-1).gather(-1, targets[start:start + chunk_tokens, None])[:, 0])
    return torch.cat(pieces) if pieces else states.new_zeros(0, dtype=torch.float32), rows, targets


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


def disable_dropout(module: torch.nn.Module) -> int:
    """Dropout layers to eval while the rest trains (checkpointing needs training mode). Returns how many."""
    count = 0
    for child in module.modules():
        if isinstance(child, torch.nn.Dropout):
            child.eval()
            count += 1
    return count
