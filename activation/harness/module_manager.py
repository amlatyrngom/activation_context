"""
Manages loras and AC models.
"""

import torch
class ModuleManager:
    def register_retrieval_ac(self, ac_name: str, model_name: str, ac_model: object):
        pass

    def register_lora(self, lora_name: str, model_name: str, rank: int =128):
        pass