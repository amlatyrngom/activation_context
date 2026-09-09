"""Agent-owned execution configs, with shared caching and bounded synthetic tuning.

Trainers pass existing configs and apply the returned values. This package owns
hardware detection, compatibility, tuning and small version-controlled presets.
Explicit execution settings always win; objectives and model weights stay caller-owned.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import torch
    from ..harness.model_config import ModelConfig
    from ..agent_training.agent_training_config import AgentTrainingConfig
    from ..ac_model.ac_model import ActivationContextModelConfig
    from ..ac_model.ac_model_training import ActivationContextTrainingConfig


class TrainingAutotuningVariables(TypedDict):
    fla_config: dict | None
    logits_chunk_tokens: int
    provenance: dict


class AgentTrainingAutotuningVariables(TrainingAutotuningVariables):
    gradient_checkpointing_min_tokens: int


class ACTrainingAutotuningVariables(TrainingAutotuningVariables):
    side_gradient_checkpointing_min_tokens: int
    target_gradient_checkpointing_min_tokens: int


def get_agent_training_autotuning_variables(
    model_config: ModelConfig,
    training_config: AgentTrainingConfig,
    *, device: torch.device | None = None,
    memory_budget_bytes: int | None = None,
) -> AgentTrainingAutotuningVariables:
    from .core import resolve
    values, threshold = resolve([model_config], training_config, device, memory_budget_bytes)
    return {**values, "gradient_checkpointing_min_tokens": threshold}


def get_ac_training_autotuning_variables(
    side_model_config: ModelConfig,
    target_model_config: ModelConfig,
    training_config: ActivationContextTrainingConfig,
    *, ac_config: ActivationContextModelConfig | None = None,
    device: torch.device | None = None,
    memory_budget_bytes: int | None = None,
) -> ACTrainingAutotuningVariables:
    from .core import resolve
    values, threshold = resolve([side_model_config, target_model_config], training_config, device, memory_budget_bytes)
    # No whole-model AC memory measurement is inferred from a kernel benchmark.
    # Each caller retains its independent checkpointing enable/disable flag.
    return {**values, "side_gradient_checkpointing_min_tokens": threshold,
            "target_gradient_checkpointing_min_tokens": threshold}


def configure_fla_runtime(bundle: dict | None):
    """Apply the returned catalog for a trainer's model-call scope; restore on exit."""
    from .fla import configure_fla_runtime as configure
    return configure(bundle)
