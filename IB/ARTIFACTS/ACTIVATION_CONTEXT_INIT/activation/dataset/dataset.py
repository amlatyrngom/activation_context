from dataclasses import dataclass, field
from enum import StrEnum, auto
import typing as t
import numpy as np

class DataOrigin(StrEnum):
    """Who authored one registerd labeled example"""
    NATIVE = auto()
    EXTERNAL = auto()
    SYNTHETIC = auto()


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
    FINQA = auto() # custom.
    JUDGE = auto()


@dataclass
class DatasetDocument:
    doc_id: str
    """ID of this document."""

    dataset_id: str
    """ID of the dataset object this belongs to."""

    text: str
    """Actual text"""
    
    atomic: bool = False
    """Whether document should be chunked using a much larger chunk size."""

    origin: DataOrigin = DataOrigin.NATIVE
    """Source of the doc"""

    split: DataSplit = DataSplit.TRAIN
    """Split of the doc"""

    alias_doc_ids: list[str] = field(default_factory=list)
    """Any aliases this doc has."""

    source_datum: dict = field(default_factory=dict)
    """The raw datum from HF/source."""


@dataclass
class LabeledRetrievalQAExample:
    example_id: str
    dataset_id: str
    query: str
    gold_answers: list[str] | None = None
    origin: DataOrigin = DataOrigin.NATIVE
    split: DataSplit = DataSplit.TRAIN
    positive_doc_ids: list[str] | None = None
    positive_chunk_ids: list[str] | None = None
    hard_negative_doc_ids: list[str] | None = None
    hard_negative_chunk_ids: list[str] | None = None
    excluded_doc_ids: list[str] | None = None # @AI: This seems more appropriate than excluded chunk ids.
    scope: t.Literal["whole", "multi"] = "multi"


@dataclass
class LabeledMessagesExample:
    """
    One correct, e2e conversation.
    """
    example_id: str
    dataset_id: str
    messages: list[dict]
    origin: DataOrigin = DataOrigin.NATIVE
    split: DataSplit = DataSplit.TRAIN


@dataclass
class DatasetTask:
    """
    A runnable agent task that owns its reference datum and scoring policy.
    """

    task_id: str
    corpus_id: str
    task_datum: dict # source data from hf.
    reference_metrics_kind: DatasetTaskMetricsKind
    gold_answer: str = ""
    gold_answer_aliases: list[str] = field(default_factory=list)
    run_config: t.Any = None # TODO: Add me later when doing rollouts.

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
        if metric == DatasetTaskMetricsKind.FINQA:
            return scoring.finqa_match(answer, golds)
        if metric == DatasetTaskMetricsKind.JUDGE:
            raise NotImplementedError("LLM-as-judge not yet implemented!")



@dataclass
class DatasetStats:
    # Computed at load time.
    initial_load_latency: float = 0.0
    total_document_chars: int = 0
    total_num_documents: int = 0
    avg_document_chars: int = 0
    # Computed at index build time.
    index_size_mb: float = 0.0
    build_embedding_latencies: list[float] = field(default_factory=list)
    build_embedding_input_chars: list[int] = field(default_factory=list)
    # Computed/Updated at query time.
    query_embedding_latencies: list[float] = field(default_factory=list)
    query_embedding_input_chars: list[int] = field(default_factory=list)
    query_search_latencies: list[float] = field(default_factory=list)
    query_batch_sizes: list[int] = field(default_factory=list)


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
            "index_size_mb": self.index_size_mb,
            "num_build_embedding_batches": len(self.build_embedding_latencies),
            "avg_build_embedding_latency": average(self.build_embedding_latencies),
            "total_build_embedding_latency": total(self.build_embedding_latencies),
            "avg_build_embedding_input_chars": average(self.build_embedding_input_chars),
            "total_build_embedding_input_chars": total(self.build_embedding_input_chars),
            "num_query_batches": len(self.query_batch_sizes),
            "avg_query_batch_size": average(self.query_batch_sizes),
            "avg_query_embedding_latency": average(self.query_embedding_latencies),
            "total_query_embedding_latency": total(self.query_embedding_latencies),
            "avg_query_embedding_input_chars": average(self.query_embedding_input_chars),
            "total_query_embedding_input_chars": total(self.query_embedding_input_chars),
            "avg_query_search_latency": average(self.query_search_latencies),
            "total_query_search_latency": total(self.query_search_latencies),
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

    labeled_retrieval_examples: dict[str, LabeledRetrievalQAExample] = field(default_factory=dict)
    """List of labeled retrieval examples"""

    labeled_messages_examples: dict[str, LabeledMessagesExample] = field(default_factory=dict)
    """List of labeled messages."""
    
    scorable_tasks: dict[str, DatasetTask] = field(default_factory=dict)
    """Scorable tasks. Supported later."""

    stats: DatasetStats|None = None
    """Dataset stats"""

    self_study_context: str|None = None
    """When generating synthetic questions with an oracle, what kind of questions to emphasize."""
