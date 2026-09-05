"""
Batches, mechanical batching optimizations, GPU batch sizing and the out-of-memory probe.

Everything here reorders or deduplicates work without changing what the loss sees: length-grouped
forwards, candidate deduplication, data-measured sizing and seeded validation groups.
"""
import gc
import math
import random
import typing as t
from dataclasses import dataclass, field

import torch

from ..dataset.dataset import DatasetDocumentChunk, LabeledRetrievalQAExample
from ..dataset.dataset_utils import safe_truncate_embedding_chunk

if t.TYPE_CHECKING:
    from ..dataset import DatasetIndex
    from ..harness import HarnessRuntime
    from .retrieval_model import RetrievalModel


GB = 1024 ** 3
CHARS_PER_TOKEN = 4
CONTEXT_BYTES = 1 * GB                  # CUDA context and workspace.
BYTES_PER_TRAINABLE_PARAM = 16          # fp32 weight + grad + two Adam states.
CPU_BATCH_SIZE = 8


@dataclass
class RetrievalBatch:
    examples: list[LabeledRetrievalQAExample]     # the queries
    candidates: list[list[DatasetDocumentChunk]]  # per example: its positives first, then hard negatives
    num_positives: list[int]                      # per example: how many leading candidates are positives


DatasetIndexes = t.Union["DatasetIndex", dict[str, "DatasetIndex"]]


def _index_for(example: LabeledRetrievalQAExample, dataset_index: DatasetIndexes) -> "DatasetIndex":
    if isinstance(dataset_index, dict):
        return dataset_index[example.dataset_id]
    return dataset_index


def _example_candidates(example: LabeledRetrievalQAExample, dataset_index: DatasetIndexes) -> tuple[list[DatasetDocumentChunk], int]:
    index = _index_for(example, dataset_index)
    positives = [index.chunks[chunk_id] for chunk_id in (example.positive_chunk_ids or [])]
    negatives = [index.chunks[chunk_id] for chunk_id in (example.hard_negative_chunk_ids or [])]
    assert positives, f"Example {example.example_id} has no positive chunk; select_training_data drops those."
    return positives + negatives, len(positives)


def _batch_from(examples: list[LabeledRetrievalQAExample], dataset_index: DatasetIndexes) -> RetrievalBatch:
    candidates, num_positives = [], []
    for example in examples:
        chunks, count = _example_candidates(example, dataset_index)
        candidates.append(chunks)
        num_positives.append(count)
    return RetrievalBatch(examples=list(examples), candidates=candidates, num_positives=num_positives)


def make_batches(
    examples: list[LabeledRetrievalQAExample],
    dataset_index: DatasetIndexes,
    batch_size: int,
    rng: random.Random,
) -> t.Iterator[RetrievalBatch]:
    """One epoch: shuffle, group by batch_size, resolve every labeled chunk id to its chunk."""
    order = list(examples)
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield _batch_from(order[start:start + batch_size], dataset_index)


def fixed_batches(
    examples: list[LabeledRetrievalQAExample],
    dataset_index: DatasetIndexes,
    batch_size: int | None,
    seed: int,
) -> list[RetrievalBatch]:
    """Reporting (batch_size None: one batch) and validation (seeded groups drawn once)."""
    order = list(examples)
    random.Random(seed).shuffle(order)
    if not order:
        return []
    batch_size = batch_size or len(order)
    return [_batch_from(order[start:start + batch_size], dataset_index) for start in range(0, len(order), batch_size)]


def flatten_candidates(batch: RetrievalBatch) -> tuple[list[DatasetDocumentChunk], list[list[int]]]:
    """Distinct chunks in first-seen order and, per example, the indices of its candidates."""
    chunks: list[DatasetDocumentChunk] = []
    positions: dict[str, int] = {}
    per_example: list[list[int]] = []
    for candidates in batch.candidates:
        indices = []
        for chunk in candidates:
            if chunk.chunk_id not in positions:
                positions[chunk.chunk_id] = len(chunks)
                chunks.append(chunk)
            indices.append(positions[chunk.chunk_id])
        per_example.append(indices)
    return chunks, per_example


def length_groups(lengths: list[int], tolerance: float, min_group: int) -> list[list[int]]:
    """
    Indices sorted by length and cut into consecutive runs whose lengths lie within tolerance of the
    run's shortest member; a run only closes once it holds min_group texts.
    """
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    groups: list[list[int]] = []
    for index in order:
        if groups and (len(groups[-1]) < min_group or lengths[index] <= lengths[groups[-1][0]] * (1 + tolerance)):
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def embed_in_length_groups(
    retrieval_model: "RetrievalModel",
    texts: list[str],
    is_query: bool,
    tolerance: float = 0.15,
    min_group: int = 8,
) -> torch.Tensor:
    """
    embed_batch over length-sorted groups whose lengths lie within tolerance of each other,
    concatenated back in the original order. Same graph, same loss, less padding.
    """
    groups = length_groups([len(text) for text in texts], tolerance, min_group)
    embedded = [retrieval_model.embed_batch([texts[index] for index in group], is_query) for group in groups]
    order = torch.tensor([index for group in groups for index in group], device=embedded[0].device)
    restored = torch.empty_like(torch.cat(embedded))
    restored[order] = torch.cat(embedded)
    return restored


def observed_shape(
    training_data: list[LabeledRetrievalQAExample],
    dataset_index: DatasetIndexes,
    retrieval_model: "RetrievalModel",
) -> tuple[str, int]:
    """The longest text (query with its instruction, or candidate chunk) and the largest candidate count."""
    limit = retrieval_model.input_limit_chars
    longest, max_candidates = "", 1
    for example in training_data:
        query = retrieval_model.query_instruction + safe_truncate_embedding_chunk(example.query, limit)
        if len(query) > len(longest):
            longest = query
        chunks, _ = _example_candidates(example, dataset_index)
        max_candidates = max(max_candidates, len(chunks))
        for chunk in chunks:
            text = safe_truncate_embedding_chunk(chunk.chunk_text, limit)
            if len(text) > len(longest):
                longest = text
    return longest, max_candidates


def _per_token_layer_internals(d_model: int, d_ff: int) -> int:
    # Saved tensors per token per layer in bf16: about six d-sized (norms, q, k, v, attention out,
    # residual) and four d_ff-sized (gate, up, activation, down input).
    return (6 * d_model + 4 * d_ff) * 2


def recommended_batch_size(
    harness: "HarnessRuntime",
    retrieval_model: "RetrievalModel",
    training_data: list[LabeledRetrievalQAExample],
    dataset_index: DatasetIndexes,
    gradient_checkpointing: bool,
    headroom_fraction: float,
    device: torch.device,
    cap: int = 128,
) -> tuple[int, str]:
    """
    Examples per step for this GPU from the observed longest text and largest candidate count;
    returns the size and the arithmetic as printed and stored in the stats.
    """
    longest, max_candidates = observed_shape(training_data, dataset_index, retrieval_model)
    if device.type != "cuda":
        return CPU_BATCH_SIZE, f"{device.type}: no memory sizing; batch_size = {CPU_BATCH_SIZE}"
    loaded_model = retrieval_model.loaded_model
    description = loaded_model.model_config.model_description
    properties = torch.cuda.get_device_properties(device)
    total = properties.total_memory
    headroom = total * headroom_fraction
    base_weights = sum(p.numel() * p.element_size() for n, p in loaded_model.model.named_parameters() if ".lora_" not in n)
    embedding_copy = sum(p.numel() * p.element_size() for p in loaded_model.embedding_layer.parameters())
    groups = retrieval_model.trainable_parameter_groups()
    group_counts = {name: sum(p.numel() for p in params) for name, params in groups.items()}
    trainable = sum(group_counts.values())
    trainable_bytes = trainable * BYTES_PER_TRAINABLE_PARAM
    usable = total - headroom - CONTEXT_BYTES - base_weights - embedding_copy - trainable_bytes

    num_views = retrieval_model.num_vectors if retrieval_model.ac_model is not None else 0
    tokens = len(longest) // CHARS_PER_TOKEN + 1 + num_views
    num_layers = len(description.layer_descriptions)
    internals = _per_token_layer_internals(description.d_model, description.d_ff)
    per_token = (num_layers * description.d_model * 2 + internals) if gradient_checkpointing else num_layers * internals
    ac_bytes = 0
    if retrieval_model.ac_model is not None:
        ac = retrieval_model.ac_model
        ac_internals = _per_token_layer_internals(ac.d_ac_model, 4 * ac.d_ac_model)
        ac_per_position = (ac.num_ac_layers * ac.d_ac_model * 2 + ac_internals) if gradient_checkpointing else ac.num_ac_layers * ac_internals
        byte_count = len(longest.encode("utf-8"))
        ac_bytes = ac_per_position * (byte_count // 4 + 1) + byte_count * ac.d_ac_model * 2 * 6
    per_sequence = per_token * tokens + ac_bytes
    sequences = 1 + max_candidates
    per_example = per_sequence * sequences
    recommended = int(usable // per_example) if usable > 0 else 0
    if recommended < 1:
        raise RuntimeError(
            f"GPU {properties.name} cannot fit one example: usable {usable / GB:.1f} GB, per example {per_example / GB:.2f} GB."
        )
    batch_size = min(recommended, cap)
    explanation = "\n".join([
        f"GPU {properties.name}: {total / GB:.1f} GB total, {headroom / GB:.1f} GB headroom, {CONTEXT_BYTES / GB:.1f} GB context",
        f"base {loaded_model.model_config.model_id} {str(description.dtype).replace('torch.', '')}: "
        f"{base_weights / GB:.1f} GB weights + {embedding_copy / GB:.1f} GB embedding copy",
        f"trainable {trainable / 1e6:.1f}M params ({', '.join(f'{name} {count / 1e6:.1f}M' for name, count in group_counts.items())})"
        f" x {BYTES_PER_TRAINABLE_PARAM} B = {trainable_bytes / GB:.1f} GB",
        f"usable for activations: {usable / GB:.1f} GB",
        f"observed longest text {len(longest):,} chars (~{tokens - num_views} tokens + {num_views} view rows); max {max_candidates} candidates per example",
        f"per token {per_token / 2**20:.2f} MB {'with' if gradient_checkpointing else 'without'} checkpointing"
        f" -> {per_example / GB:.2f} GB per example ({sequences} sequences)",
        f"recommended batch_size = min({recommended}, cap {cap}) = {batch_size}",
    ])
    return batch_size, explanation


def probe_batch_size(
    step_fn: t.Callable[[RetrievalBatch], None],
    harness: "HarnessRuntime",
    retrieval_model: "RetrievalModel",
    batch_size: int,
    longest_text: str,
    max_candidates: int,
    attempts: int = 3,
) -> tuple[int, int]:
    """
    One synthetic forward/backward at batch_size with every sequence at the observed longest;
    halve on out-of-memory. Returns (size that passed, attempts used). Skipped on CPU.
    """
    if retrieval_model.device.type != "cuda":
        return batch_size, 0
    for attempt in range(1, attempts + 1):
        examples, candidates, num_positives = [], [], []
        for example_index in range(batch_size):
            chunks = [
                DatasetDocumentChunk(
                    chunk_id=f"probe:{example_index}:{chunk_index}:0", doc_id=f"probe:{example_index}:{chunk_index}",
                    dataset_id="probe", chunk_text=longest_text, chunk_start=0,
                )
                for chunk_index in range(max_candidates)
            ]
            examples.append(LabeledRetrievalQAExample(
                example_id=f"probe:{example_index}", dataset_id="probe", query=longest_text,
                positive_doc_ids=[chunks[0].doc_id], positive_chunk_ids=[chunks[0].chunk_id],
                hard_negative_chunk_ids=[chunk.chunk_id for chunk in chunks[1:]],
            ))
            candidates.append(chunks)
            num_positives.append(1)
        batch = RetrievalBatch(examples=examples, candidates=candidates, num_positives=num_positives)
        failed = False
        try:
            step_fn(batch)
            torch.cuda.synchronize()
            return batch_size, attempt
        except torch.OutOfMemoryError:
            failed = True                                                                # release the traceback first
        if failed:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Batch probe: out of memory at batch_size={batch_size} (attempt {attempt}/{attempts}).")
            if batch_size == 1 or attempt == attempts:
                raise RuntimeError(f"Batch probe failed after {attempt} attempts; last batch_size {batch_size}.")
            batch_size //= 2
    return batch_size, attempts
