"""CPU smoke of activation.vllm_residuals: imports, override shape, packing order, registration (no engine)."""
import torch

import activation.vllm_residuals as r

print(r.engine_overrides(4096, [31, 3, 7]))
packed = r.pack_rows(torch.zeros(2, 4), {7: torch.ones(2, 4), 3: 2 * torch.ones(2, 4)})
assert packed.shape == (2, 12) and packed[0, 4:8].eq(2).all() and packed[0, 8:].eq(1).all(), packed
from activation.vllm_residuals import qwen3_5 as q  # noqa: E402

print([c.__name__ for c in q.Qwen3_5ResModel.__mro__[:4]], hasattr(q.Qwen3_5ResForConditionalGeneration, "_processor_factory"))
r.register()
from vllm import ModelRegistry  # noqa: E402

print("registered:", r.ARCHITECTURE in ModelRegistry.get_supported_archs())
from vllm.model_executor.models.config import MODELS_CONFIG_MAP  # noqa: E402

print("config hook:", MODELS_CONFIG_MAP[r.ARCHITECTURE].__name__)
