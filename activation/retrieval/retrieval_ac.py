"""
Contains a ac model for the retrieval.
An AC model places new latent input in the model. If the model also has lora enabled, this new latent input can live in a subspace with new properties.
Required for Slice 1.
"""
import torch

class StandardRetrievalACModel(torch.nn.Module):
    """
    A retrieval activation context model.
    Appends retrieval_ac_num_view_tokens to the model's input embeddings.
    The architecture is generally as follows:
    - Takes in the query.
    - Byte Embedding Table: 256xd_ac_model.
    - Positional Encoding.
    - Windowed self-attention+pooling reduces the length to 1/4th.
        - Self-attention on 8-byte wide windows with stride of 4.
        - Pool to one vector per window.
        - The standard ffn with 4 as ffn mult.
    - Standard bidirectional transformer with ac_model_num_layers layers.
        - As standard as it gets with normalizations, positional encodings, etc all in place without anything fancy at all.
    - Query with up-projection to the main d_model with the number of view tokens.

    If possible:
    - Should accept bog-standard training hyperparams if these happen to differ from lora.
    - E.g., lr, clipping, etc.
    """
    def __init__(
        self,
        harness: object,
        base_model_name: str,
        ac_model_name: str,
        lora_rank: int = 64,
        d_ac_model: int|None = None, # None = 1/4th of main model's d_model.
        num_view_tokens: int = 8,
        num_ac_layers: int|None = None, # None = 1/4th of main model's d_model
    ):
        pass

    def batch_compute_view_inputs(self, args) -> torch.Tensor:
        pass

    @staticmethod
    def checkpoint(self):
        raise NotImplementedError # Do not implement yet.

    @staticmethod
    def load_from_checkpoint(self):
        raise NotImplementedError # Do not implement yet.
