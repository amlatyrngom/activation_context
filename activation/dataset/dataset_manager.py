import random
import typing as t
from .dataset import LoadedDataset, LabeledRetrievalQAExample, DataOrigin, DataSplit
from .dataset_index import DatasetIndex
from .dataset_utils import initialize_dataset_stats
from .dataset_study import DatasetStudyGenerator

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

class DatasetManager:
    def __init__(self, harness: "HarnessRuntime"):
        self.harness = harness
        self.loaded_datasets: dict[str, LoadedDataset] = dict()
        self.dataset_indexes: dict[str, DatasetIndex] = dict()
        self.dataset_study_generators: dict[str, DatasetStudyGenerator] = dict()

    def register_dataset(self, loaded_dataset: LoadedDataset):
        """Register a specific dataset."""
        if loaded_dataset.stats is None:
            loaded_dataset.stats = initialize_dataset_stats(loaded_dataset, load_time=0.0)
        print(f"{loaded_dataset.dataset_id} - Loaded. Load Time = {loaded_dataset.stats.initial_load_latency: .2f}.")
        self.loaded_datasets[loaded_dataset.dataset_id] = loaded_dataset
        return

    def _get_or_create_index(self, dataset_id: str) -> DatasetIndex:
        """Chunk a dataset once; the bm25 and dense indexes are built on top explicitly."""
        if dataset_id not in self.dataset_indexes:
            self.dataset_indexes[dataset_id] = DatasetIndex(
                self.harness, self.loaded_datasets[dataset_id],
            )
        return self.dataset_indexes[dataset_id]

    def _get_or_create_study_generator(self, dataset_id: str) -> DatasetStudyGenerator:
        """Get a dataset study generator."""
        if dataset_id not in self.dataset_study_generators:
            self.dataset_study_generators[dataset_id] = DatasetStudyGenerator(
                self.harness, self.loaded_datasets[dataset_id],
            )
        return self.dataset_study_generators[dataset_id]


    def build_bm25_indexes(self):
        """Build the bm25 index of every loaded dataset (CPU only, no model needed)."""
        print(f"Building bm25 indexes: {list(self.loaded_datasets.keys())}")
        for dataset_id in self.loaded_datasets:
            self._get_or_create_index(dataset_id).build_bm25_index()
        return

    def build_dense_indexes(self):
        """Build the dense (embedding) index of every loaded dataset."""
        print(f"Building dense indexes: {list(self.loaded_datasets.keys())}")
        for dataset_id in self.loaded_datasets:
            self._get_or_create_index(dataset_id).build_dense_index()
        return

    
    def synthesize_study_examples_qa(self, dataset_id: str, num_samples: int):
        """Synthesize study examples"""
        study_generator = self._get_or_create_study_generator(dataset_id)
        return study_generator.generate_examples_qa(num_samples)

    def label_study_examples(
        self,
        dataset_id: str,
        num_samples: int,
        synthetic_only: bool = False,
    ):
        study_generator = self._get_or_create_study_generator(dataset_id)
        return study_generator.generate_examples_labels(num_samples, synthetic_only)


    def select_training_data(
        self,
        dataset_id: str,
        num_samples: int,
        synthetic_only: bool = False,
        oracle_labeled_only: bool = True,
        val_ratio: float = 0.1,
        max_reporting_size: int = 50,
        force_partition: bool = True, # Most datasets only have training. This forces a val set.
        seed: int = 0,
    ) -> tuple[list[LabeledRetrievalQAExample], list[LabeledRetrievalQAExample], list[LabeledRetrievalQAExample]]:
        """
        Select training data. Returns tuples with the following:
        - training data: actually used to update the gradients.
        - validation: small amount of data for validation (~10% in general).
        - reporting: trivial amount of data (subset of the validation set). Used for plotting.
        There is a general min of 10 for training and validation, and 1 for reporting regardless of the fractions.

        num_samples examples are selected before the split; validation is carved from them by
        val_ratio unless force_partition is False and the dataset carries native validation-split
        examples. Oracle-labeled examples are used as they are. Without the oracle, an example
        inherits chunk labels from its document-level labels: one chunk per positive document and,
        where the dataset carries native hard negatives, one chunk per hard-negative document.
        Examples without a positive chunk are dropped and counted in the dataset stats.
        """
        loaded_dataset = self.loaded_datasets[dataset_id]
        stats = loaded_dataset.stats
        candidates = [
            example
            for example in loaded_dataset.labeled_retrieval_examples.values()
            if (not synthetic_only or example.origin == DataOrigin.SYNTHETIC)
            and (not oracle_labeled_only or example.oracle_labeled)
        ]
        study_generator = self._get_or_create_study_generator(dataset_id)
        selected: list[LabeledRetrievalQAExample] = []
        for example in candidates:
            if not example.oracle_labeled and not example.positive_chunk_ids:
                study_generator._inherit_labels(example, pool=[])
            if example.positive_chunk_ids:
                selected.append(example)
            else:
                stats.training_select_num_dropped_no_positive += 1
        rng = random.Random(seed)
        native_validation = [example for example in selected if example.split == DataSplit.VAL]
        if not force_partition and native_validation:
            training_pool = [example for example in selected if example.split != DataSplit.VAL]
            rng.shuffle(training_pool)
            rng.shuffle(native_validation)
            training_data = training_pool[:num_samples]
            validation_data = native_validation[:max(10, round(num_samples * val_ratio))]
        else:
            rng.shuffle(selected)
            chosen = selected[:num_samples]
            num_validation = max(10, round(len(chosen) * val_ratio))
            validation_data = chosen[:num_validation]
            training_data = chosen[num_validation:]
        assert len(training_data) >= 10, (
            f"{dataset_id} - Only {len(training_data)} training examples after the split; need at least 10 "
            f"({len(selected)} selectable, {stats.training_select_num_dropped_no_positive} dropped without a positive chunk)."
        )
        assert len(validation_data) >= 10, f"{dataset_id} - Only {len(validation_data)} validation examples; need at least 10."
        reporting_data = validation_data[:max(1, min(max_reporting_size, len(validation_data)))]
        print(
            f"{dataset_id} - Selected {len(training_data)} training / {len(validation_data)} validation / "
            f"{len(reporting_data)} reporting examples."
        )
        return training_data, validation_data, reporting_data
