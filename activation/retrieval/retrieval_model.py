import torch

class RetrievalModel:
    """
    Retrieval Model, with an optionally attached lora and activation context model.
    """
    def __init__(
        self,
        harness: object,
        base_model_name: str,
        d_embedding_result: int,
        ac_name: str|None, # None=no ac.
        lora_name: str|None, # None=no lora (no full fine-tuning either btw).
    ):
        # @
        pass

    def embed_batch(self, queries: list[str]) -> torch.Tensor:
        pass 


    