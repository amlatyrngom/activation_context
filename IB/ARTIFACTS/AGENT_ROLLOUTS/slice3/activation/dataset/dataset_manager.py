import typing as t
from .dataset import LoadedDataset, DatasetQAExample, DataModality
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
        """Chunk a dataset once; the bm25 index is built on top explicitly."""
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


    def synthesize_study_examples_qa(
        self,
        dataset_id: str,
        num_samples: int,
        base_seed: int | None = None,
        study_context: str | None = None,
        caching_id: str | None = None,
        modality: DataModality | None = None,
    ) -> list[DatasetQAExample]:
        """
        Synthesize study questions from the dataset's chunks (see DatasetStudyGenerator.generate_examples_qa).
        Returns the examples; nothing is written into the dataset. base_seed None: the harness study seed.
        """
        study_generator = self._get_or_create_study_generator(dataset_id)
        seed = self.harness.harness_config.dataset_study_seed if base_seed is None else base_seed
        return study_generator.generate_examples_qa(num_samples, seed, study_context, caching_id, modality)


    def label_study_examples(
        self,
        dataset_id: str,
        examples: list[DatasetQAExample],
        caching_id: str | None = None,
    ) -> list[DatasetQAExample]:
        """Oracle-label exactly these examples (the caller selects them, e.g. by origin); returns the labeled ones."""
        study_generator = self._get_or_create_study_generator(dataset_id)
        return study_generator.generate_examples_labels(examples, caching_id)
