"""
Qwen3.5 (vLLM 0.28) whose prompt-embedding rows carry per-layer residual deltas; see the package docstring for the layout.

Only two things differ from upstream `Qwen3_5ForConditionalGeneration`:
  - `embed_input_ids` pads token embeddings with zero deltas to the runner's wide rows;
  - the decoder (`Qwen3_5ResModel.forward`, upstream `Qwen3NextModel.forward` plus the marked lines) splits the wide rows and,
    before each named layer, adds its delta to the layer input. vLLM's layers pass (hidden_states, residual) and fuse the
    add into the next norm, so the layer input HF hooks is hidden_states + residual: the delta goes onto `residual` (onto
    `hidden_states` at layer 0, where residual is None). A zero delta leaves every value bit for bit as upstream computes it.
The decoder stays under upstream's `support_torch_compile` (inherited); its forward keeps the compiled argument names.
"""
from itertools import islice
from unittest import mock

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.models import qwen3_5
from vllm.sequence import IntermediateTensors

from . import LAYERS_KEY


class Qwen3_5ResModel(qwen3_5.Qwen3_5Model):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        layers = [int(layer) for layer in getattr(self.config, LAYERS_KEY)]
        if get_pp_group().world_size != 1:
            raise NotImplementedError("residual deltas ride the first stage's input rows: pipeline parallelism is not supported")
        if self.use_sequence_parallel:
            raise NotImplementedError("residual deltas with sequence-parallel MoE layers are not supported")
        self.delta_slot = {layer: slot for slot, layer in enumerate(layers)}   # layer -> its column block after the row

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        width = self.config.hidden_size
        deltas = None
        if inputs_embeds is not None:                                         # [N, (1 + k) * d]: the row, then one delta per layer
            deltas = inputs_embeds[:, width:]
            hidden_states = inputs_embeds[:, :width].contiguous()
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            slot = self.delta_slot.get(layer_idx)
            if deltas is not None and slot is not None:                       # the delta joins this layer's input
                delta = deltas[:, slot * width:(slot + 1) * width]
                if residual is None:
                    hidden_states = hidden_states + delta
                else:
                    residual = residual + delta
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class Qwen3_5ResForConditionalGeneration(qwen3_5.Qwen3_5ForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # Upstream builds its decoder by the module-level name Qwen3_5Model (inside Qwen3_5ForCausalLM): build ours there.
        with mock.patch.object(qwen3_5, "Qwen3_5Model", Qwen3_5ResModel):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        assert type(self.language_model.model) is Qwen3_5ResModel
        self.delta_width = vllm_config.model_config.get_inputs_embeds_size() - self.config.text_config.hidden_size
        if self.delta_width != len(self.language_model.model.delta_slot) * self.config.text_config.hidden_size:
            raise ValueError(f"text_config.projection_size gives {self.delta_width} delta columns for layers "
                             f"{sorted(self.language_model.model.delta_slot)} (vllm_residuals.engine_overrides sets both)")

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None) -> torch.Tensor:
        embeds = super().embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=is_multimodal)
        return F.pad(embeds, (0, self.delta_width))                           # token positions: zero deltas
