from dataclasses import dataclass, field
from enum import StrEnum, auto
import os
import typing as t
import torch
from dotenv import load_dotenv

from .model_config import ModelConfig

@dataclass
class HarnessRuntimeConfig:
    # Map from model_name --> model config.
    model_configs: dict[str, ModelConfig] = field(default_factory=dict)

    # Model configs
    max_lora_rank: int = 128
    max_loras: int = 4

    # Document chunking
    doc_chunk_size_chars: int = 4096
    doc_chunk_overlap_chars: int = 256
    doc_chunk_enable_extension: bool = False
    doc_embedding_model_name: str|None = None
    doc_embedding_batch_size: int = 32
    doc_embedding_input_limit_chars: int|None = None # Absolute limit on embedding inputs. Used for fast testing.


    # Dataset Study
    dataset_study_qa_batch_size: int|None = None # None = recommended value.
    dataset_study_qa_model_name: str|None = None
    dataset_study_chunk_input_limit: int|None = None # Limit for fast testing.
    dataset_study_chat_kwargs: dict|None = None
    dataset_study_seed: int = 0 # Seeds chunk sampling and generation sampling.
    dataset_study_label_top_k: int = 10 # bm25 candidates per question.
    dataset_study_label_pool_max_chars: int = 32768 # Char budget of the snippet pool shown to the labeler.
    dataset_study_label_batch_size: int|None = None # None = recommended value.
    dataset_study_label_model_name: str|None = None

    # Agents
    agent_max_concurrent: int = 64 # Thread pool bound for rollouts; the engine batches across them.
    agent_env_default_image: str = "docker.io/library/python:3.12-slim" # Sandbox image when a config names no dockerfile.
    agent_env_memory_limit_mb: int | None = 8192 # Address space per sandboxed process (ulimit -v); None = unlimited.

@dataclass
class HarnessStats:
    # (model_name, seconds) records for actual state changes (no-ops unrecorded).
    model_initialization_times: list[tuple[str, float]] = field(default_factory=list)
    model_loading_times: list[tuple[str, float]] = field(default_factory=list)
    model_freeing_times: list[tuple[str, float]] = field(default_factory=list)

    def summarize(self) -> dict:
        """Per-model totals, grouped by model name."""
        summary: dict[str, dict] = {}
        for key, records in [
            ("initialization", self.model_initialization_times),
            ("load", self.model_loading_times),
            ("free", self.model_freeing_times),
        ]:
            for model_name, elapsed in records:
                model_summary = summary.setdefault(model_name, {})
                model_summary[f"num_{key}s"] = model_summary.get(f"num_{key}s", 0) + 1
                model_summary[f"total_{key}_time"] = round(
                    model_summary.get(f"total_{key}_time", 0.0) + elapsed, 3
                )
        return summary

# Load the right env vars
load_dotenv()
# Bound the NVFP4/JIT kernel build: unbounded nvcc fan-out exhausts a 62 GB host.
os.environ.setdefault("MAX_JOBS", "3")
os.environ.setdefault("NVCC_THREADS", "1")
