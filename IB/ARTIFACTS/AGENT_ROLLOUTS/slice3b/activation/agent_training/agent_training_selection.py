"""
Selection functions: one group of rollouts (the same task, several seeds) in, weighted training items
out. The trainer only sees the weights, so the learning rule lives here:

- `group_mean_advantage` (default): weight = reward - mean(group); uniform groups carry no
  within-task information and yield nothing. GRPO's advantage without the std division (Dr. GRPO,
  RLOO); pairs with group size 8 and oversampling to fill a batch with mixed groups (DAPO's dynamic sampling).
- `raft_positive`: correct runs at +1, the rest dropped (RAFT / RAFT++ with the clipped ratio).
- `weighted_positive_negative`: correct +0.1, incorrect -1 (W-REINFORCE's lambda = 0.1).
- `reinforce_reject`: +1 / -1 on mixed groups only.

Every function unrolls subagent runs with the parent's weight (a subagent's trajectory is judged by the
answer it helped produce), always includes a run's compacted segments (the same agent's earlier
sequences), and marks non-policy sources (`run.source != "policy"`) as `ignore_logprobs`.
"""
from __future__ import annotations

import typing as t

from activation.agent import AgentRunResult

from .agent_trainer import AgentTrainingItem, unroll

SelectionFunction = t.Callable[[list[AgentRunResult]], list[AgentTrainingItem]]


def items_for(run: AgentRunResult, weight: float, model_name: str, lora_name: str, include_subagents: bool = True) -> list[AgentTrainingItem]:
    """One item per (sub)agent run under `run`, all at `weight`; zero weights produce nothing."""
    if weight == 0.0:
        return []
    runs = unroll(run) if include_subagents else list(run.compactions) + [run]   # compacted segments are the same agent: always included
    return [
        AgentTrainingItem(run_results=r, model_name=model_name, lora_name=lora_name, weight=float(weight), ignore_logprobs=r.source != "policy")
        for r in runs
    ]


def group_mean_advantage(group: list[AgentRunResult], *, model_name: str, lora_name: str, include_subagents: bool = True) -> list[AgentTrainingItem]:
    if not group:
        return []
    mean = sum(run.score for run in group) / len(group)
    return [item for run in group for item in items_for(run, run.score - mean, model_name, lora_name, include_subagents)]


def raft_positive(group: list[AgentRunResult], *, model_name: str, lora_name: str, include_subagents: bool = True) -> list[AgentTrainingItem]:
    return [item for run in group if run.score >= 1.0 for item in items_for(run, 1.0, model_name, lora_name, include_subagents)]


def weighted_positive_negative(
    group: list[AgentRunResult], *, model_name: str, lora_name: str, positive_weight: float = 0.1, negative_weight: float = -1.0,
    include_subagents: bool = True,
) -> list[AgentTrainingItem]:
    return [
        item for run in group
        for item in items_for(run, positive_weight if run.score >= 1.0 else negative_weight, model_name, lora_name, include_subagents)
    ]


def reinforce_reject(group: list[AgentRunResult], *, model_name: str, lora_name: str, include_subagents: bool = True) -> list[AgentTrainingItem]:
    scores = {run.score >= 1.0 for run in group}
    if len(scores) < 2:
        return []
    return weighted_positive_negative(group, model_name=model_name, lora_name=lora_name, positive_weight=1.0, negative_weight=-1.0,
                                      include_subagents=include_subagents)


SELECTION_FUNCTIONS = {
    "group_mean": group_mean_advantage,
    "raft": raft_positive,
    "weighted": weighted_positive_negative,
    "reinforce_rej": reinforce_reject,
}
