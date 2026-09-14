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

    
    def synthesize_study_examples_qa(self, dataset_id: str, num_samples: int, caching_id: str|None=None):
        """Synthesize study questions (one hard question with its answer per sampled chunk)."""
        study_generator = self._get_or_create_study_generator(dataset_id)
        return study_generator.generate_examples_qa(num_samples, caching_id)

    def synthesize_study_examples_description(self, dataset_id: str, num_samples: int, caching_id: str|None=None):
        """Synthesize study descriptions (one search-friendly description per sampled chunk that needs one)."""
        study_generator = self._get_or_create_study_generator(dataset_id)
        return study_generator.generate_examples_descriptions(num_samples, caching_id)

    def synthesize_programmatic_examples(self, dataset_id: str, num_samples: int):
        """Programmatic examples: a span of a chunk as the query, the chunk as its positive; no model."""
        study_generator = self._get_or_create_study_generator(dataset_id)
        return study_generator.generate_programmatic_examples(num_samples)

    def label_study_examples(
        self,
        dataset_id: str,
        num_samples: int,
        origins: list[DataOrigin]|None = None, # None = all origins.
        caching_id: str|None = None,
        label_model_name: str|None = None,
    ):
        study_generator = self._get_or_create_study_generator(dataset_id)
        return study_generator.generate_examples_labels(num_samples, origins, caching_id, label_model_name)

    def select_training_data(
        self,
        dataset_id: str,
        num_samples: int,
        origins: list[DataOrigin]|None = None, # None = all origins, programmatic included.
        oracle_labeled_only: bool = True,
        val_ratio: float = 0.1,
        max_reporting_size: int = 50,
        force_partition: bool = True, # Most datasets only have training. This forces a val set.
        seed: int = 0,
    ) -> tuple[list[LabeledRetrievalQAExample], list[LabeledRetrievalQAExample], list[LabeledRetrievalQAExample]]:
        """Training / validation / reporting examples; see DatasetStudyGenerator.select_training_data."""
        study_generator = self._get_or_create_study_generator(dataset_id)
        return study_generator.select_training_data(
            num_samples, origins, oracle_labeled_only, val_ratio, max_reporting_size, force_partition, seed,
        )

    def select_testing_data(
        self,
        dataset_id: str,
        num_samples: int|None = None, # None=all
        seed: int = 0,
    ) -> list[LabeledRetrievalQAExample]:
        """The native test-split examples with inherited chunk labels; see DatasetStudyGenerator.select_testing_data."""
        study_generator = self._get_or_create_study_generator(dataset_id)
        return study_generator.select_testing_data(num_samples, seed)
