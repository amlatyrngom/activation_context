"""
Reference embedders for the in-batch metrics: the same validation batches scored by a well-trained
single-vector embedder (Qwen/Qwen3-Embedding-0.6B by default) and by the frozen base alone, so a
training curve has a trained reference and a floor even at small scales. Nothing beyond the batch
is embedded; corpus-level comparisons belong to the slice 2 index.
"""
import typing as t

import torch

from ..dataset.dataset_utils import safe_truncate_embedding_chunk
from ..harness.hf_utils import FREE_DEVICE, TARGET_DEVICE
from .retrieval_model import DEFAULT_QUERY_INSTRUCTION, RetrievalModel

if t.TYPE_CHECKING:
    from ..harness import HarnessRuntime


class BaselineEmbedder:
    """
    A harness embedding model behind the scoring interface of RetrievalModel (embed_batch [B, 1, d],
    similarity, num_vectors, device, query_instruction, document_instruction, input_limit_chars, ac_model), so the trainer
    evaluates it with the same batches, loss and metrics as the model under training. Queries get
    the same instruction prefix (the Qwen3-Embedding "Instruct: ...\\nQuery: " format).
    """
    num_vectors = 1
    ac_model = None

    def __init__(self, harness: "HarnessRuntime", model_name: str, query_instruction: str = DEFAULT_QUERY_INSTRUCTION):
        self.harness = harness
        self.model_name = model_name
        self.loaded_model = harness.loaded_models[model_name]
        assert self.loaded_model.model_config.model_description.is_embedding_model, f"{model_name} is not an embedding model"
        self.query_instruction = query_instruction
        self.document_instruction = ""                                                # Qwen3-Embedding: no instruction on documents
        self.input_limit_chars = harness.harness_config.doc_embedding_input_limit_chars
        self.device = torch.device(TARGET_DEVICE)
        self.real_tokens_embedded = self.padded_tokens_embedded = self.forwards_embedded = 0

    def embed_batch(self, texts: list[str], is_query: bool) -> torch.Tensor:
        """[B, 1, d], L2-normalized, on the CPU (the harness embedding path returns there)."""
        texts = [safe_truncate_embedding_chunk(text, self.input_limit_chars) for text in texts]
        if is_query:
            texts = [self.query_instruction + text for text in texts]
        embeddings = self.loaded_model.simple_vector_embed_many(texts)                 # [B, d] normalized
        self.forwards_embedded += 1
        return embeddings.float().unsqueeze(1)

    similarity = staticmethod(RetrievalModel.similarity)

    def free(self) -> None:
        self.loaded_model.model_to_device(FREE_DEVICE)


def frozen_base_reference(harness: "HarnessRuntime", retrieval_model: RetrievalModel) -> RetrievalModel:
    """
    The base of the model under training with no LoRA, no AC model and no head: the EOS readout of
    the frozen language model, L2-normalized. It shares the resident base, so it costs no memory.
    """
    reference = RetrievalModel(
        harness, retrieval_model.base_model_name, d_embedding_result=None, ac_name=None, lora_name=None,
        query_instruction=retrieval_model.query_instruction,
    )
    reference.set_training_mode(False)
    return reference
