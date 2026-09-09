"""
Registry of LoRA adapters and retrieval activation-context (AC) models.

Each adapter and each AC model is bound to one loaded base model by name. The manager owns one
PEFT wrapper per base model: the wrap happens the first time an adapter of that model is needed
after the base is loaded, further adapters of the same base are added by name on that wrapper
(the base weights are present once; each adapter only adds its own A/B matrices inside the
wrapped linear layers), and the base weights stay frozen throughout. A forward pass names the
adapter it wants (`lora_context`); with no name the adapters are disabled for that pass. Freeing
a base with adapters attached is refused until `free_lora` has dropped them.

Adapters travel between training and serving as checkpoints under the synced folder
(`LORAS/<lora_name>/round_<n>` plus a `latest` copy): `save_lora` writes one, `exchange_lora` makes the
engine load it on its next request naming the adapter (a fresh `LoRARequest` id per exchange, no engine
restart), `branch_lora` copies one adapter's latest checkpoint into a new name, and `register_lora`'s
`checkpoint_path` starts an adapter from a saved one. A freshly registered adapter is the identity
(PEFT zero-initialises the B matrices), so round 0 of a training run samples the base model exactly.
"""
import re
import shutil
import typing as t
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from activation.common.data_syncing import resolve_path

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
    checkpoint_path: str | None = None
    """Where the trainable adapter is loaded from when injected (None: fresh, zero-init B). Updated by save_lora."""
    engine_checkpoint_path: str | None = None
    """What the serving engine applies for this name (None: nothing yet, requests run the base). Set by exchange_lora."""
    engine_version: int = 0
    """Bumped by every exchange; the engine caches adapters by integer id, so a new version is a new LoRARequest."""
    saved_rounds: int = 0
    """Checkpoints written so far (round_000, round_001, ...)."""

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
        self._engine_version_counter = 0
        self._engine_int_ids: dict[str, int] = dict()
        """Integer ids of the adapters' current engine views, unique across adapters and exchanges (vLLM caches by id)."""

    # ---------------------------------------------------------------- registration

    def register_lora(
        self,
        lora_name: str,
        model_name: str,
        rank: int = 128,
        alpha: int | None = None,
        dropout: float = 0.05,
        target_modules: list[str] | None = None,
        checkpoint_path: str | None = None, # @AI: Make this dire
    ) -> LoraConfig:
        """
        Record the adapter config; the adapter itself is injected on first use, fresh (zero-init B: the
        identity) or from `checkpoint_path` (a PEFT adapter folder, absolute or relative to the synced
        folder, e.g. "LORAS/<name>/latest"). alpha None means 2 x rank; target_modules None means the
        seven block projections. A checkpoint is also what the engine serves after the first exchange.
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
            checkpoint_path=None if checkpoint_path is None else str(_checkpoint_folder(checkpoint_path)),
        )
        if lora_config.checkpoint_path is not None:
            assert Path(lora_config.checkpoint_path, "adapter_config.json").exists(), f"No adapter at {lora_config.checkpoint_path}."
        assert all(other.adapter_name != lora_config.adapter_name for other in self.lora_configs.values()), (
            f"LoRA {lora_name!r} maps to adapter name {lora_config.adapter_name!r}, which another LoRA already uses."
        )
        self.lora_configs[lora_name] = lora_config
        return lora_config

    # @AI: Remove me and related logic.
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
        from peft import PeftModel, get_peft_model
        lora_config = self.get_lora_config(lora_name)
        loaded_model = self.harness.loaded_models[lora_config.model_name]
        loaded_model.model_to_device(TARGET_DEVICE)
        peft_model = self.peft_models.get(lora_config.model_name)
        if peft_model is None:
            loaded_model.model.requires_grad_(False)
            if lora_config.checkpoint_path is not None:
                peft_model = PeftModel.from_pretrained(
                    loaded_model.model, lora_config.checkpoint_path, adapter_name=lora_config.adapter_name, is_trainable=True,
                )
            else:
                peft_model = get_peft_model(
                    loaded_model.model, make_peft_lora_config(lora_config), adapter_name=lora_config.adapter_name,
                )
            self.peft_models[lora_config.model_name] = peft_model
        elif lora_config.adapter_name not in peft_model.peft_config:
            if lora_config.checkpoint_path is not None:
                peft_model.load_adapter(lora_config.checkpoint_path, adapter_name=lora_config.adapter_name, is_trainable=True)
            else:
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
        Drop the adapter's weights from the base, saving them first when `checkpoint_path` is given
        (see save_lora). When it was the last adapter of that base, the PEFT wrapper is removed and the
        plain base modules are restored, so the base can be freed. The registered config stays, with
        `checkpoint_path` pointing at the last save, so the adapter re-injects from there.
        """
        lora_config = self.get_lora_config(lora_name)
        assert lora_config.model_name == model_name, f"LoRA {lora_name!r} belongs to {lora_config.model_name!r}, not {model_name!r}."
        peft_model = self.peft_models.get(model_name)
        if peft_model is None or lora_config.adapter_name not in peft_model.peft_config:
            return                                                                      # never injected
        if checkpoint_path is not None:
            self.save_lora(model_name, lora_name, checkpoint_path)
        if len(peft_model.peft_config) > 1:
            peft_model.delete_adapter(lora_config.adapter_name)
            return
        loaded_model = self.harness.loaded_models[model_name]
        loaded_model.model = peft_model.base_model.unload()                                # LoRA layers replaced back in place
        del self.peft_models[model_name]

    # ---------------------------------------------------------------- checkpoints and the engine view

    def checkpoint_folder(self, lora_name: str, round_index: int | None = None) -> Path:
        """`LORAS/<lora_name>/round_<n>` under the synced folder (`latest` when round_index is None)."""
        leaf = "latest" if round_index is None else f"round_{round_index:03d}"
        return resolve_path(f"LORAS/{lora_name}/{leaf}", create=False)

    def save_lora(self, model_name: str, lora_name: str, checkpoint_path: str | None = None) -> str:
        """
        Write the adapter (only this one) as a PEFT adapter folder that both `ensure_lora` and the
        serving engine can load: `checkpoint_path`, or the next `LORAS/<lora_name>/round_<n>`; a
        `latest` copy is refreshed next to the numbered folders. The registered config's
        `checkpoint_path` now points at the save, so a re-injection continues from it.
        """
        lora_config = self.get_lora_config(lora_name)
        assert lora_config.model_name == model_name, f"LoRA {lora_name!r} belongs to {lora_config.model_name!r}, not {model_name!r}."
        peft_model = self.peft_models.get(model_name)
        assert peft_model is not None and lora_config.adapter_name in peft_model.peft_config, f"LoRA {lora_name!r} is not injected."
        folder = Path(checkpoint_path) if checkpoint_path is not None else self.checkpoint_folder(lora_name, lora_config.saved_rounds)
        if checkpoint_path is not None and not folder.is_absolute():
            folder = resolve_path(checkpoint_path, create=False)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(folder), selected_adapters=[lora_config.adapter_name])
        nested = folder / lora_config.adapter_name                                          # PEFT nests non-default adapters
        if nested.is_dir():
            for item in nested.iterdir():
                shutil.move(str(item), str(folder / item.name))
            nested.rmdir()
        assert (folder / "adapter_config.json").exists(), f"PEFT did not write an adapter at {folder}"
        if checkpoint_path is None:
            lora_config.saved_rounds += 1
            latest = self.checkpoint_folder(lora_name)
            if latest.exists():
                shutil.rmtree(latest)
            shutil.copytree(folder, latest)
        lora_config.checkpoint_path = str(folder)
        return str(folder)

    def exchange_lora(self, model_name: str, lora_name: str, checkpoint_path: str | None = None) -> None:
        """
        Make the serving engine see the adapter: from `checkpoint_path`, or from its last save. The
        next request naming `lora_name` carries a new LoRARequest (new integer id), which vLLM loads
        from the folder without a restart; requests before the first exchange run the base model.
        """
        lora_config = self.get_lora_config(lora_name)
        assert lora_config.model_name == model_name, f"LoRA {lora_name!r} belongs to {lora_config.model_name!r}, not {model_name!r}."
        path = checkpoint_path if checkpoint_path is not None else lora_config.checkpoint_path
        assert path is not None, f"LoRA {lora_name!r} has no checkpoint to exchange; call save_lora (or train) first."
        folder = _checkpoint_folder(path)
        assert (folder / "adapter_config.json").exists(), f"No adapter at {folder}."
        lora_config.engine_checkpoint_path = str(folder)
        lora_config.engine_version += 1
        self._engine_version_counter += 1
        self._engine_int_ids[lora_name] = self._engine_version_counter

    def branch_lora(self, model_name: str, src_lora_name: str, dest_lora_name: str) -> LoraConfig:
        """
        A new adapter that starts from another's latest save: the checkpoint is copied to
        `LORAS/<dest>/round_000` (and `latest`), the config registered with the same shape, and the
        engine view exchanged when the source had one.
        """
        src = self.get_lora_config(src_lora_name)
        assert src.model_name == model_name, f"LoRA {src_lora_name!r} belongs to {src.model_name!r}, not {model_name!r}."
        assert src.checkpoint_path is not None, f"LoRA {src_lora_name!r} has no checkpoint to branch from; save it first."
        dest_round = self.checkpoint_folder(dest_lora_name, 0)
        for target in (dest_round, self.checkpoint_folder(dest_lora_name)):
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src.checkpoint_path, target)
        dest = self.register_lora(
            dest_lora_name, model_name, rank=src.rank, alpha=src.alpha, dropout=src.dropout,
            target_modules=list(src.target_modules), checkpoint_path=str(dest_round),
        )
        dest.saved_rounds = 1
        if src.engine_checkpoint_path is not None:
            self.exchange_lora(model_name, dest_lora_name, str(dest_round))
        return dest

    def engine_lora_request(self, lora_name: str):
        """The vLLM LoRARequest for the adapter's current engine view, or None before the first exchange."""
        lora_config = self.get_lora_config(lora_name)
        if lora_config.engine_checkpoint_path is None:
            return None
        from vllm.lora.request import LoRARequest
        return LoRARequest(
            lora_name=f"{lora_config.adapter_name}_v{lora_config.engine_version}",
            lora_int_id=self._engine_int_ids[lora_name],
            lora_path=lora_config.engine_checkpoint_path,
        )


def _checkpoint_folder(path: str) -> Path:
    folder = Path(path)
    return folder if folder.is_absolute() else resolve_path(path, create=False)
