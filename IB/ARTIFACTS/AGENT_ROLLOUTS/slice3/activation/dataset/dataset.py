from dataclasses import dataclass, field
from enum import StrEnum, auto
import typing as t
import numpy as np


NATIVE, EXTERNAL, SYNTHETIC = "native", "external", "synthetic"
"""Conventional `origin` values of documents and examples (who authored them); loaders may use others, e.g. "rollout:<caching_id>"."""


class DataModality(StrEnum):
    TEXT = auto()
    TRAJECTORY = auto()   # chat messages; chunked at message boundaries (dataset_index)


class DataSplit(StrEnum):
    TRAIN = auto()
    TRAIN_HELDOUT = auto() # Small set used to approximate test-time trends.
    VAL = auto()
    TEST = auto()


class DatasetTaskMetricsKind(StrEnum):
    """
    Built-in scoring policies for corpus tasks.
    """
    EXACT_MATCH = auto()
    F1 = auto()
    NUMERIC_APPROX = auto()
    NUMERIC_EXACT = auto()
    FINQA = auto() # custom.
    JUDGE = auto()


@dataclass
class DatasetDocumentChunk:
    chunk_id: str
    doc_id: str
    dataset_id: str
    chunk_text: str
    chunk_start: int
    """Offset of chunk_text in the document text (for trajectories: in the rendering of the whole trajectory)."""
    chunk_messages: list[dict] | None = None
    """Trajectory chunks: the message slice chunk_text renders (an over-long message is split by characters)."""

@dataclass
class DatasetDocument:
    doc_id: str
    """ID of this document."""

    dataset_id: str
    """ID of the dataset object this belongs to."""

    text: str|None = None
    """Actual text"""

    trajectory: list[dict]|None = None
    """Chat messages in our dialect (assistant `tool_calls` as structured fields, one `tool` message per result)."""

    trajectory_kwargs: dict|None = None
    """{"tools": [definitions], "system_prompt": str, "answer": str | None, ...}: what the trajectory ran with."""

    modality: DataModality = DataModality.TEXT
    """Modality"""

    origin: str = NATIVE
    """Source of the doc (NATIVE / EXTERNAL / SYNTHETIC or a loader-specific string)."""

    split: DataSplit = DataSplit.TRAIN
    """Split of the doc"""

    alias_doc_ids: list[str] = field(default_factory=list)
    """Any aliases this doc has."""

    source_datum: dict = field(default_factory=dict)
    """The raw datum from HF/source."""

    chunks: dict[str, DatasetDocumentChunk] = field(default_factory=dict)
    """The chunks of this document."""


@dataclass
class DatasetQAExample:
    """A question with reference answers and (optional) document / chunk labels; the study generator makes synthetic ones."""
    example_id: str
    dataset_id: str
    query: str
    gold_answers: list[str] | None = None
    origin: str = NATIVE
    split: DataSplit = DataSplit.TRAIN
    positive_doc_ids: list[str] | None = None
    positive_chunk_ids: list[str] | None = None
    hard_negative_doc_ids: list[str] | None = None
    hard_negative_chunk_ids: list[str] | None = None
    excluded_doc_ids: list[str] | None = None
    oracle_labeled: bool = False


@dataclass
class DatasetTask:
    """
    A runnable agent task that owns its reference datum and scoring policy.
    """
    task_id: str
    dataset_id: str
    task_datum: dict # source data from hf.
    reference_metrics_kind: DatasetTaskMetricsKind
    gold_answer: str = ""
    gold_answer_aliases: list[str] = field(default_factory=list)
    agent_prompt: str = "" # What an agent is asked to do; the loader fills it.

    def score(self, run_result: t.Any, force_metric: DatasetTaskMetricsKind | None = None) -> float:
        """
        Unwrap an agent result and dispatch to the configured shared metric.
        """
        from . import scoring
        metric = force_metric or self.reference_metrics_kind
        answer = scoring.unwrap_submit_answer(run_result)
        golds = [self.gold_answer, *self.gold_answer_aliases]
        if metric == DatasetTaskMetricsKind.EXACT_MATCH:
            return scoring.exact_match(answer, golds)
        if metric == DatasetTaskMetricsKind.F1:
            return scoring.token_f1(answer, golds)
        if metric == DatasetTaskMetricsKind.NUMERIC_APPROX:
            return scoring.numeric_approx(answer, golds)
        if metric == DatasetTaskMetricsKind.NUMERIC_EXACT:
            return scoring.numeric_exact(answer, golds)
        if metric == DatasetTaskMetricsKind.FINQA:
            return scoring.finqa_match(answer, golds)
        if metric == DatasetTaskMetricsKind.JUDGE:
            raise NotImplementedError("LLM-as-judge not yet implemented!")



@dataclass
class DatasetStats:
    # Load time.
    initial_load_latency: float = 0.0
    total_document_chars: int = 0
    total_num_documents: int = 0
    avg_document_chars: int = 0
    num_decontaminated: int = 0
    """Rows a loader dropped for overlapping an evaluation set (see loaders/nemotron_math.py)."""
    # Index.
    num_chunks: int = 0
    bm25_build_latency: float = 0.0
    bm25_query_search_latencies: list[float] = field(default_factory=list)
    query_batch_sizes: list[int] = field(default_factory=list)
    # Study.
    study_prompt_tokens: list[int] = field(default_factory=list)
    study_output_tokens: list[int] = field(default_factory=list)
    study_batch_latencies: list[float] = field(default_factory=list)
    study_num_parse_failures: int = 0
    study_num_junk_skipped: int = 0
    study_num_cached: int = 0
    study_label_prompt_tokens: list[int] = field(default_factory=list)
    study_label_output_tokens: list[int] = field(default_factory=list)
    study_label_batch_latencies: list[float] = field(default_factory=list)
    study_label_pool_sizes: list[int] = field(default_factory=list)
    study_num_label_positives: int = 0
    study_num_label_negatives: int = 0
    study_num_label_ambiguous: int = 0
    study_num_label_parse_failures: int = 0
    study_num_label_inherited_positives: int = 0
    study_num_label_inherited_negatives: int = 0
    study_num_label_inherit_skipped: int = 0
    study_num_labels_cached: int = 0

    def summarize(self) -> dict:
        # Returns average statistics.
        def average(values: list) -> float:
            return float(np.average(values)) if values else 0.0
        def total(values: list) -> float:
            return float(np.sum(values)) if values else 0.0
        return {
            "initial_load_latency": self.initial_load_latency,
            "total_document_chars": self.total_document_chars,
            "total_num_documents": self.total_num_documents,
            "avg_document_chars": self.avg_document_chars,
            "num_decontaminated": self.num_decontaminated,
            "num_chunks": self.num_chunks,
            "bm25_build_latency": self.bm25_build_latency,
            "num_query_batches": len(self.query_batch_sizes),
            "avg_query_batch_size": average(self.query_batch_sizes),
            "avg_bm25_query_search_latency": average(self.bm25_query_search_latencies),
            "total_bm25_query_search_latency": total(self.bm25_query_search_latencies),
            "num_study_requests": len(self.study_prompt_tokens),
            "avg_study_prompt_tokens": average(self.study_prompt_tokens),
            "avg_study_output_tokens": average(self.study_output_tokens),
            "total_study_prompt_tokens": total(self.study_prompt_tokens),
            "total_study_output_tokens": total(self.study_output_tokens),
            "total_study_latency": total(self.study_batch_latencies),
            "study_output_tokens_per_s": (
                total(self.study_output_tokens) / total(self.study_batch_latencies)
                if self.study_batch_latencies else 0.0
            ),
            "study_num_parse_failures": self.study_num_parse_failures,
            "study_num_junk_skipped": self.study_num_junk_skipped,
            "study_num_cached": self.study_num_cached,
            "num_study_label_requests": len(self.study_label_prompt_tokens),
            "avg_study_label_prompt_tokens": average(self.study_label_prompt_tokens),
            "avg_study_label_output_tokens": average(self.study_label_output_tokens),
            "total_study_label_latency": total(self.study_label_batch_latencies),
            "avg_study_label_pool_size": average(self.study_label_pool_sizes),
            "study_num_label_positives": self.study_num_label_positives,
            "study_num_label_negatives": self.study_num_label_negatives,
            "study_num_label_ambiguous": self.study_num_label_ambiguous,
            "study_num_label_parse_failures": self.study_num_label_parse_failures,
            "study_num_label_inherited_positives": self.study_num_label_inherited_positives,
            "study_num_label_inherited_negatives": self.study_num_label_inherited_negatives,
            "study_num_label_inherit_skipped": self.study_num_label_inherit_skipped,
            "study_num_labels_cached": self.study_num_labels_cached,
        }


@dataclass
class LoadedDataset:
    """
    One registered dataset.
    """
    dataset_id: str
    """Uniquely identifies a loaded dataset. Must include relevant load arguments."""

    documents: dict[str, DatasetDocument] = field(default_factory=dict)
    """List of documents."""

    labeled_qa_examples: dict[str, DatasetQAExample] = field(default_factory=dict)
    """Questions with reference answers and labels, keyed by example id (native ones from the loader)."""
    
    scorable_tasks: dict[str, DatasetTask] = field(default_factory=dict)
    """Scorable tasks. Supported later."""

    stats: DatasetStats|None = None
    """Dataset stats"""
