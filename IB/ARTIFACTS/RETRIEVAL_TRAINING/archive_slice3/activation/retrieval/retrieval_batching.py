"""
Batches, mechanical batching optimizations, GPU batch sizing and the out-of-memory probe.

Everything here reorders or deduplicates work without changing what the loss sees: forwards cut by
a padded-token budget, candidate deduplication, sizing from the heaviest real batch the data can
produce, a probe on that real batch, and seeded validation groups.
"""
import gc
import math
import random
import typing as t
from dataclasses import dataclass, field

import torch

from ..dataset.dataset import DatasetDocumentChunk, LabeledRetrievalQAExample

if t.TYPE_CHECKING:
    from ..dataset import DatasetIndex
    from .retrieval_model import RetrievalModel


GB = 1024 ** 3
CHARS_PER_TOKEN = 4
CONTEXT_BYTES = 1 * GB                  # CUDA context and workspace.
BYTES_PER_TRAINABLE_PARAM = 16          # fp32 weight + grad + two Adam states.
CPU_BATCH_SIZE = 8
FORWARD_TOKEN_BUDGET = 8192             # padded tokens per forward on a GPU: a launches-versus-padding knob, not memory
CPU_FORWARD_TOKEN_BUDGET = 512


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


def estimated_tokens(text_length: int, num_views: int) -> int:
    """Padded-length estimate shared by grouping and sizing: characters / 4, plus EOS and the view rows."""
    return text_length // CHARS_PER_TOKEN + 1 + num_views


def token_budget_groups(lengths: list[int], budget_tokens: int, num_views: int) -> list[list[int]]:
    """
    Indices sorted by length and cut into consecutive runs whose padded size (count x longest
    member) stays within budget_tokens. A single text over the budget forms its own run: the next
    text is at least as long, so it can never join. Sorting keeps padding low inside a run; the
    budget bounds the number of forwards (15 tolerance-cut forwards per 32-query step were
    launch-bound at 16 % of the GPU's compute peak).
    """
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    groups: list[list[int]] = []
    for index in order:
        longest = estimated_tokens(lengths[index], num_views)                          # ascending: the newest is the longest
        if groups and (len(groups[-1]) + 1) * longest <= budget_tokens:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def embedded_text_length(retrieval_model: "RetrievalModel", text: str, is_query: bool) -> int:
    """Characters embed_batch will actually see: cut to the input limit, plus the query instruction."""
    prefix = len(retrieval_model.query_instruction) if is_query else 0
    limit = retrieval_model.input_limit_chars
    return (len(text) if limit is None else min(len(text), limit)) + prefix


def embed_in_length_groups(
    retrieval_model: "RetrievalModel",
    texts: list[str],
    is_query: bool,
    budget_tokens: int | None = None,
) -> torch.Tensor:
    """
    embed_batch over length-sorted groups cut by a padded-token budget, concatenated back in the
    original order. Same graph, same loss; only the number and shape of the forwards change.
    """
    if budget_tokens is None:
        budget_tokens = FORWARD_TOKEN_BUDGET if retrieval_model.device.type == "cuda" else CPU_FORWARD_TOKEN_BUDGET
    num_views = retrieval_model.num_vectors if retrieval_model.ac_model is not None else 0
    lengths = [embedded_text_length(retrieval_model, text, is_query) for text in texts]
    groups = token_budget_groups(lengths, budget_tokens, num_views)
    embedded = [retrieval_model.embed_batch([texts[index] for index in group], is_query) for group in groups]
    order = torch.tensor([index for group in groups for index in group], device=embedded[0].device)
    restored = torch.empty_like(torch.cat(embedded))
    restored[order] = torch.cat(embedded)
    return restored


def example_token_counts(
    training_data: list[LabeledRetrievalQAExample],
    dataset_index: DatasetIndexes,
    retrieval_model: "RetrievalModel",
) -> list[int]:
    """Per example: estimated padded tokens of its query plus all its candidates, as embed_batch will see them."""
    num_views = retrieval_model.num_vectors if retrieval_model.ac_model is not None else 0
    counts = []
    for example in training_data:
        total = estimated_tokens(embedded_text_length(retrieval_model, example.query, True), num_views)
        chunks, _ = _example_candidates(example, dataset_index)
        total += sum(estimated_tokens(embedded_text_length(retrieval_model, chunk.chunk_text, False), num_views) for chunk in chunks)
        counts.append(total)
    return counts


def largest_fitting_batch(counts_desc: list[int], usable_bytes: int, per_token_bytes: int, cap: int) -> int:
    """
    Largest B with per_token_bytes x sum(top-B counts) <= usable_bytes, at most cap; 0 when even
    the heaviest example does not fit. With counts sorted descending this is the heaviest batch of
    B examples the data can produce, so no shuffle of the epoch exceeds it (dedup only lightens).
    """
    total = 0
    for size, count in enumerate(counts_desc, start=1):
        total += count
        if total * per_token_bytes > usable_bytes:
            return size - 1
        if size >= cap:
            return cap
    return len(counts_desc)


def _per_token_layer_internals(d_model: int, d_ff: int, with_lora: bool) -> int:
    # Saved tensors per token per layer in bf16: about six d-sized (norms, q, k, v, attention out,
    # residual) and four d_ff-sized (gate, up, activation, down input). A LoRA on the seven
    # projections keeps, per projection, its fp32 input and the dropout output (4 B each) on top:
    # six inputs of d_model and one of d_ff. Measured on Qwen3-0.6B + LoRA r128 (RTX PRO 6000):
    # 3.0 MB per token without checkpointing, which this formula gives; without the LoRA term it
    # said 1.0 MB.
    internals = (6 * d_model + 4 * d_ff) * 2
    if with_lora:
        internals += (6 * d_model + d_ff) * 8
    return internals


@dataclass
class BatchSizing:
    batch_size: int
    explanation: str                                    # the arithmetic as printed and stored in the stats
    probe_examples: list[LabeledRetrievalQAExample]     # the batch_size heaviest examples: the probe batch


def _heaviest(training_data: list[LabeledRetrievalQAExample], counts: list[int], size: int) -> list[LabeledRetrievalQAExample]:
    order = sorted(range(len(counts)), key=lambda index: -counts[index])
    return [training_data[index] for index in order[:size]]


def configured_batch_sizing(
    batch_size: int,
    training_data: list[LabeledRetrievalQAExample],
    dataset_index: DatasetIndexes,
    retrieval_model: "RetrievalModel",
) -> BatchSizing:
    """A batch size from the config, still probed with the heaviest real batch of that size."""
    counts = example_token_counts(training_data, dataset_index, retrieval_model)
    return BatchSizing(batch_size, f"batch_size = {batch_size} (from the config)", _heaviest(training_data, counts, batch_size))


def recommended_batch_size(
    retrieval_model: "RetrievalModel",
    training_data: list[LabeledRetrievalQAExample],
    dataset_index: DatasetIndexes,
    gradient_checkpointing: bool,
    headroom_fraction: float,
    device: torch.device,
    cap: int = 128,
) -> BatchSizing:
    """
    Examples per step for this GPU: the largest batch whose heaviest possible members (per-example
    token totals, descending) fit the usable memory at the configured checkpointing setting. The
    explanation prints the batch for both settings.
    """
    counts = example_token_counts(training_data, dataset_index, retrieval_model)
    if device.type != "cuda":
        return BatchSizing(CPU_BATCH_SIZE, f"{device.type}: no memory sizing; batch_size = {CPU_BATCH_SIZE}",
                           _heaviest(training_data, counts, CPU_BATCH_SIZE))
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

    num_layers = len(description.layer_descriptions)
    internals = _per_token_layer_internals(description.d_model, description.d_ff, with_lora=bool(retrieval_model.lora_name))
    per_token_on = num_layers * description.d_model * 2 + internals                  # layer inputs kept + one layer recomputed
    per_token_off = num_layers * internals
    if retrieval_model.ac_model is not None:                                          # one pooled window per token, plus the byte stage
        ac = retrieval_model.ac_model
        ac_internals = _per_token_layer_internals(ac.d_ac_model, 4 * ac.d_ac_model, with_lora=False)
        byte_stage = CHARS_PER_TOKEN * ac.d_ac_model * 2 * 6
        per_token_on += ac.num_ac_layers * ac.d_ac_model * 2 + ac_internals + byte_stage
        per_token_off += ac.num_ac_layers * ac_internals + byte_stage
    counts_desc = sorted(counts, reverse=True)
    batch_on = largest_fitting_batch(counts_desc, usable, per_token_on, cap)
    batch_off = largest_fitting_batch(counts_desc, usable, per_token_off, cap)
    batch_size = batch_on if gradient_checkpointing else batch_off
    if batch_size < 1:
        raise RuntimeError(
            f"GPU {properties.name} cannot fit the heaviest example: usable {usable / GB:.1f} GB, "
            f"{counts_desc[0]:,} est. tokens x {(per_token_on if gradient_checkpointing else per_token_off) / 2**20:.2f} MB."
        )
    heaviest_total = sum(counts_desc[:batch_size])
    explanation = "\n".join([
        f"GPU {properties.name}: {total / GB:.1f} GB total, {headroom / GB:.1f} GB headroom, {CONTEXT_BYTES / GB:.1f} GB context",
        f"base {loaded_model.model_config.model_id} {str(description.dtype).replace('torch.', '')}: "
        f"{base_weights / GB:.1f} GB weights + {embedding_copy / GB:.1f} GB embedding copy",
        f"trainable {trainable / 1e6:.1f}M params ({', '.join(f'{name} {count / 1e6:.1f}M' for name, count in group_counts.items())})"
        f" x {BYTES_PER_TRAINABLE_PARAM} B = {trainable_bytes / GB:.1f} GB",
        f"usable for activations: {usable / GB:.1f} GB",
        f"{len(counts):,} examples: {sum(counts) / len(counts):,.0f} est. tokens on average, {counts_desc[0]:,} heaviest",
        f"per token {per_token_on / 2**20:.2f} MB with checkpointing, {per_token_off / 2**20:.2f} MB without",
        f"batch with checkpointing {batch_on}, without {batch_off} (cap {cap}); using {'with' if gradient_checkpointing else 'without'}",
        f"batch_size = {batch_size}: heaviest {batch_size} examples total {heaviest_total:,} est. tokens"
        f" = {heaviest_total * (per_token_on if gradient_checkpointing else per_token_off) / GB:.1f} GB",
    ])
    return BatchSizing(batch_size, explanation, _heaviest(training_data, counts, batch_size))


def probe_batch_size(
    step_fn: t.Callable[[RetrievalBatch], None],
    retrieval_model: "RetrievalModel",
    probe_examples: list[LabeledRetrievalQAExample],
    dataset_index: DatasetIndexes,
    attempts: int = 3,
) -> tuple[int, int]:
    """
    One real forward/backward on the heaviest examples (their real queries and candidates, through
    the real grouping, AC model and dedup); halve the batch on out-of-memory. Returns (size that
    passed, attempts used). Skipped on CPU.
    """
    batch_size = len(probe_examples)
    if retrieval_model.device.type != "cuda":
        return batch_size, 0
    for attempt in range(1, attempts + 1):
        batch = _batch_from(probe_examples[:batch_size], dataset_index)
        distinct = len(flatten_candidates(batch)[0])
        listed = sum(len(candidates) for candidates in batch.candidates)
        if distinct < 0.9 * listed:
            print(f"Batch probe: the heaviest {batch_size} examples share candidates ({distinct} distinct of {listed}); "
                  "the probe batch is lighter than the sizing bound.")
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
