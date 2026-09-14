"""
A passthrough AC Model.
An AC Model has two components:
- A Lora
"""


class ACModelConfig:
    lora_rank: int = 16
    """Rank of the lora."""

    d_model_ratio: float = 1.0 / 4.0
    """Ratio of d_model."""