"""
Retrieval model: a frozen base LLM with an optional LoRA and an optional activation-context (AC)
model, plus a projection head, embedding queries and documents through the same path.

Sequence layout: text tokens, the EOS token, then the V view rows from the AC model, all
left-padded so the readout is always the last V positions (the last position alone without an AC
model). Scores between a query and a candidate use MaxSim over their V vectors, which is the plain
dot product when V = 1.
"""
import typing as t
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from ..dataset.dataset_utils import safe_truncate_embedding_chunk
from ..harness.hf_utils import TARGET_DEVICE

if t.TYPE_CHECKING:
    from ..harness import HarnessRuntime
    from .retrieval_ac import StandardRetrievalACModel


DEFAULT_QUERY_INSTRUCTION = "Instruct: Retrieve passages that answer the question\nQuery: "


class RetrievalModel(nn.Module):
    """Frozen base + optional LoRA + optional AC model + projection head, embedding either side."""

    def __init__(
        self,
        harness: "HarnessRuntime",
        base_model_name: str,
        d_embedding_result: int | None,     # None: d_model, no head
        ac_name: str | None,                # None: no AC, V = 1 EOS readout
        lora_name: str | None,              # None: no LoRA (base frozen; only AC + head train)
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ):
        super().__init__()
        self.harness = harness
        self.base_model_name = base_model_name
        self.loaded_model = harness.loaded_models[base_model_name]
        description = self.loaded_model.model_config.model_description
        self.d_model = description.d_model
        self.ac_name = ac_name
        self.ac_model: "StandardRetrievalACModel|None" = harness.module_manager.get_retrieval_ac(ac_name) if ac_name else None
        if self.ac_model is not None:
            assert self.ac_model.base_model_name == base_model_name, "The AC model was built for another base model."
        self.lora_name = lora_name
        if lora_name:
            assert harness.module_manager.get_lora_config(lora_name).model_name == base_model_name
        self.num_vectors = self.ac_model.num_view_tokens if self.ac_model is not None else 1
        self.d_embedding_result = d_embedding_result or self.d_model
        self.head = nn.Linear(self.d_model, d_embedding_result, bias=False) if d_embedding_result else None
        self.query_instruction = query_instruction
        self.input_limit_chars = harness.harness_config.doc_embedding_input_limit_chars
        self.eos_token_id = self.loaded_model.tokenizer.convert_tokens_to_ids(description.eos_token)
        self.gradient_checkpointing = False
        # Token counters the trainer reads for its throughput stats.
        self.real_tokens_embedded = 0
        self.padded_tokens_embedded = 0
        self.forwards_embedded = 0
        self.view_scale: float | None = None                                              # set on first embed_batch

    @property
    def device(self) -> torch.device:
        return torch.device(self.loaded_model.current_model_device)

    def ensure_resident(self) -> None:
        """
        Get the base with the LoRA active (loading it on the target device), place the embedding
        layer copy, the AC model and the head on the same device.
        """
        if self.lora_name:
            self.harness.module_manager.ensure_lora(self.lora_name)
        else:
            self.loaded_model.model_to_device(TARGET_DEVICE)
            self.loaded_model.model.requires_grad_(False)
        self.loaded_model.embedding_layer_to_device(TARGET_DEVICE)
        device = self.device
        if self.ac_model is not None:
            self.ac_model.to(device)
        if self.head is not None:
            self.head.to(device)

    def set_training_mode(self, training: bool, gradient_checkpointing: bool = False) -> None:
        """
        Train or eval on the base, the AC model and the head. Hugging Face activation checkpointing
        only runs in training mode; the non-reentrant variant is used so frozen embedding inputs work.
        """
        base = self.loaded_model.model
        base.train(training)
        if training and gradient_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        elif base.is_gradient_checkpointing:
            base.gradient_checkpointing_disable()
        self.gradient_checkpointing = training and gradient_checkpointing
        if self.ac_model is not None:
            self.ac_model.gradient_checkpointing = self.gradient_checkpointing
        self.train(training)

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()

    def embed_batch(self, texts: list[str], is_query: bool) -> torch.Tensor:
        """
        [B, V, d_embedding_result], each vector L2-normalized. Queries get the instruction
        prefix, documents are raw. Gradients flow in training mode.
        """
        texts = [safe_truncate_embedding_chunk(text, self.input_limit_chars) for text in texts]
        if is_query:
            texts = [self.query_instruction + text for text in texts]
        tokenizer = self.loaded_model.tokenizer
        token_rows = tokenizer(texts, add_special_tokens=False, padding=False)["input_ids"]
        token_rows = [row + [self.eos_token_id] for row in token_rows]
        device = self.device
        embedding_layer = self.loaded_model.embedding_layer
        base_dtype = embedding_layer.weight.dtype
        num_views = self.num_vectors if self.ac_model is not None else 0
        length = max(len(row) for row in token_rows) + num_views
        batch_size = len(texts)
        if self.view_scale is None:                                                     # RMS of a token embedding row
            self.view_scale = float(embedding_layer.weight.detach().float().pow(2).mean().sqrt().item())
        padded_ids = torch.full((batch_size, length - num_views), self.eos_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, length, dtype=torch.long)
        for index, row in enumerate(token_rows):
            padded_ids[index, length - num_views - len(row):] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, length - num_views - len(row):] = 1
        padded_ids, attention_mask = padded_ids.to(device), attention_mask.to(device)

        with self._autocast():
            view_rows = None
            if self.ac_model is not None:                                               # [B, V, d_model] at embedding scale
                view_rows = (self.ac_model(texts) * self.view_scale).to(base_dtype)
            inputs_embeds = embedding_layer(padded_ids)                                 # padding rows are masked out
            if view_rows is not None:
                inputs_embeds = torch.cat([inputs_embeds[:, :length - num_views], view_rows], dim=1)
            position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min(0)
            hidden = self.loaded_model.decoder_forward(inputs_embeds, attention_mask, position_ids, self.lora_name or None)  # [B, S, d_model]
            readout = hidden[:, -self.num_vectors:, :]
        readout = readout.float()
        if self.head is not None:
            readout = self.head(readout)
        self.real_tokens_embedded += sum(len(row) for row in token_rows) + batch_size * num_views   # = mask sum, no GPU sync
        self.padded_tokens_embedded += batch_size * length
        self.forwards_embedded += 1
        return F.normalize(readout, p=2, dim=-1)

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """{"lora": [...], "ac": [...], "head": [...]} with missing groups omitted."""
        groups: dict[str, list[nn.Parameter]] = {}
        if self.lora_name:
            groups["lora"] = self.harness.module_manager.lora_parameters(self.lora_name)
        if self.ac_model is not None:
            groups["ac"] = list(self.ac_model.parameters())
        if self.head is not None:
            groups["head"] = list(self.head.parameters())
        return groups

    @staticmethod
    def similarity(query_embeddings: torch.Tensor, candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """
        MaxSim [B, N] from [B, V, d] and [N, V', d]: mean_i max_j <q_i, c_j>, so the score stays in
        [-1, 1] for any V and the temperature keeps its single-cosine meaning (review finding B1:
        the sum over V=8 views multiplied the logit scale by 8).
        """
        scores = torch.einsum("bvd,nwd->bnvw", query_embeddings, candidate_embeddings)
        return scores.max(dim=-1).values.mean(dim=-1)
