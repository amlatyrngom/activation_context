"""
Registry of LoRA adapters and retrieval activation-context (AC) models.

Each adapter and each AC model is bound to one loaded base model by name. The manager owns one
PEFT wrapper per base model: the wrap happens the first time an adapter of that model is needed
after the base is loaded, further adapters of the same base are added by name on that wrapper
(the base weights are present once; each adapter only adds its own A/B matrices inside the
wrapped linear layers), and the base weights stay frozen throughout. A forward pass names the
adapter it wants (`lora_context`); with no name the adapters are disabled for that pass. Freeing
a base with adapters attached is refused until `free_lora` has dropped them (checkpointing is not
implemented yet). Engine and training use of a model remain mutually exclusive.
"""
import re
import typing as t
from contextlib import contextmanager
from dataclasses import dataclass

from torch import nn

from .hf_utils import TARGET_DEVICE, make_peft_lora_config

if t.TYPE_CHECKING:
    from peft import PeftModel
    from ..retrieval.retrieval_ac import StandardRetrievalACModel
    from .runtime import HarnessRuntime


DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
"""The seven projections of every transformer block: the usual full-coverage choice. The output head is never targeted."""


@dataclass
class LoraConfig:
    lora_name: str
    model_name: str
    rank: int
    alpha: int
    """2 x rank by default, the common LoRA convention."""
    dropout: float
    """0.05 by default, the common PEFT default."""
    target_modules: list[str]
    adapter_name: str = ""
    """The name PEFT knows the adapter by: lora_name with every character outside [0-9A-Za-z_] replaced, since PEFT
    uses it as a module name (no dots)."""

    def __post_init__(self):
        self.adapter_name = self.adapter_name or re.sub(r"[^0-9A-Za-z_]", "_", self.lora_name)


class ModuleManager:
    """Registry of LoRA adapters and retrieval AC models, each bound to one loaded base model."""

    def __init__(self, harness: "HarnessRuntime"):
        self.harness = harness
        self.lora_configs: dict[str, LoraConfig] = dict()
        self.retrieval_acs: dict[str, "StandardRetrievalACModel"] = dict()
        self.peft_models: dict[str, "PeftModel"] = dict()
        """One PEFT wrapper per base model name, present while at least one adapter is injected."""

    # ---------------------------------------------------------------- registration

    def register_lora(
        self,
        lora_name: str,
        model_name: str,
        rank: int = 128,
        alpha: int | None = None,
        dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ) -> LoraConfig:
        """
        Record the adapter config; the adapter itself is injected on first use.
        alpha None means 2 x rank; target_modules None means the seven block projections.
        """
        assert model_name in self.harness.loaded_models, f"Unknown model {model_name!r}."
        assert lora_name not in self.lora_configs, f"LoRA {lora_name!r} is already registered."
        max_rank = self.harness.harness_config.max_lora_rank
        assert 1 <= rank <= max_rank, f"LoRA rank {rank} outside [1, {max_rank}] (harness max_lora_rank)."
        lora_config = LoraConfig(
            lora_name=lora_name,
            model_name=model_name,
            rank=rank,
            alpha=alpha if alpha is not None else 2 * rank,
            dropout=dropout,
            target_modules=list(target_modules or DEFAULT_LORA_TARGET_MODULES),
        )
        assert all(other.adapter_name != lora_config.adapter_name for other in self.lora_configs.values()), (
            f"LoRA {lora_name!r} maps to adapter name {lora_config.adapter_name!r}, which another LoRA already uses."
        )
        self.lora_configs[lora_name] = lora_config
        return lora_config

    def register_retrieval_ac(self, ac_name: str, model_name: str, ac_model: "StandardRetrievalACModel") -> None:
        """The AC model must have been built for model_name (it produces rows in that model's input space)."""
        assert model_name in self.harness.loaded_models, f"Unknown model {model_name!r}."
        assert ac_name not in self.retrieval_acs, f"AC model {ac_name!r} is already registered."
        assert ac_model.base_model_name == model_name, (
            f"AC model {ac_name!r} was built for {ac_model.base_model_name!r}, not {model_name!r}."
        )
        self.retrieval_acs[ac_name] = ac_model

    def get_lora_config(self, lora_name: str) -> LoraConfig:
        assert lora_name in self.lora_configs, f"Unknown LoRA {lora_name!r}."
        return self.lora_configs[lora_name]

    def get_retrieval_ac(self, ac_name: str) -> "StandardRetrievalACModel":
        assert ac_name in self.retrieval_acs, f"Unknown AC model {ac_name!r}."
        return self.retrieval_acs[ac_name]

    # ---------------------------------------------------------------- adapters on the base

    def has_loras(self, model_name: str) -> bool:
        """True while adapters are injected into that base (its weights must not be freed)."""
        return model_name in self.peft_models

    def ensure_lora(self, lora_name: str) -> "PeftModel":
        """
        Load the base on the target device if needed, wrap it once with PEFT, add the adapter by
        name if missing, and return the wrapper. Base weights stay frozen. Which adapter a forward
        pass uses is decided per pass by `lora_context`.
        """
        from peft import get_peft_model
        lora_config = self.get_lora_config(lora_name)
        loaded_model = self.harness.loaded_models[lora_config.model_name]
        loaded_model.model_to_device(TARGET_DEVICE)
        peft_model = self.peft_models.get(lora_config.model_name)
        if peft_model is None:
            loaded_model.model.requires_grad_(False)
            peft_model = get_peft_model(
                loaded_model.model, make_peft_lora_config(lora_config), adapter_name=lora_config.adapter_name,
            )
            self.peft_models[lora_config.model_name] = peft_model
        elif lora_config.adapter_name not in peft_model.peft_config:
            peft_model.add_adapter(lora_config.adapter_name, make_peft_lora_config(lora_config))
        return peft_model

    @contextmanager
    def lora_context(self, model_name: str, lora_name: str | None):
        """
        Forward-pass scope on a base: with a LoRA name, that adapter (injected if needed) is the
        active one; with None, every injected adapter is disabled so the plain base runs.
        """
        if lora_name is None:
            peft_model = self.peft_models.get(model_name)
            if peft_model is None:
                yield
            else:
                with peft_model.disable_adapter():
                    yield
            return
        lora_config = self.get_lora_config(lora_name)
        assert lora_config.model_name == model_name, f"LoRA {lora_name!r} belongs to {lora_config.model_name!r}, not {model_name!r}."
        peft_model = self.ensure_lora(lora_name)
        peft_model.set_adapter(lora_config.adapter_name)
        yield

    def lora_parameters(self, lora_name: str) -> list[nn.Parameter]:
        """The trainable parameters of that adapter only (injecting it if needed)."""
        peft_model = self.ensure_lora(lora_name)
        adapter_name = self.get_lora_config(lora_name).adapter_name
        return [
            parameter
            for name, parameter in peft_model.named_parameters()
            if f".{adapter_name}." in name and parameter.requires_grad
        ]

    def free_lora(self, model_name: str, lora_name: str, checkpoint_path: str | None = None) -> None:
        """
        Drop the adapter's weights from the base. When it was the last adapter of that base, the
        PEFT wrapper is removed and the plain base modules are restored, so the base can be freed.
        Checkpointing before the drop is not implemented yet (checkpoint_path must be None); the
        registered config stays, so the adapter can be re-injected fresh.
        """
        assert checkpoint_path is None, "LoRA checkpointing is not implemented yet."
        lora_config = self.get_lora_config(lora_name)
        assert lora_config.model_name == model_name, f"LoRA {lora_name!r} belongs to {lora_config.model_name!r}, not {model_name!r}."
        peft_model = self.peft_models.get(model_name)
        if peft_model is None or lora_config.adapter_name not in peft_model.peft_config:
            return                                                                      # never injected
        if len(peft_model.peft_config) > 1:
            peft_model.delete_adapter(lora_config.adapter_name)
            return
        loaded_model = self.harness.loaded_models[model_name]
        loaded_model.model = peft_model.base_model.unload()                                # LoRA layers replaced back in place
        del self.peft_models[model_name]
