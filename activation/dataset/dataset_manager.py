import typing as t
from .dataset import LoadedDataset
from .document_index import DocumentIndex
from .dataset_utils import initialize_dataset_stats

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

class DatasetManager:
    def __init__(self, harness: "HarnessRuntime"):
        self.harness = harness
        self.loaded_datasets: dict[str, LoadedDataset] = dict()
        self.document_indexes: dict[str, DocumentIndex] = dict()

    def register_dataset(self, loaded_dataset: LoadedDataset):
        """Register a specific dataset."""
        if loaded_dataset.stats is None:
            loaded_dataset.stats = initialize_dataset_stats(loaded_dataset, load_time=0.0)
        print(f"{loaded_dataset.dataset_id} - Loaded. Load Time = {loaded_dataset.stats.initial_load_latency: .2f}.")
        self.loaded_datasets[loaded_dataset.dataset_id] = loaded_dataset
        return

    def build_document_indexes(self):
        """Build all document indexes"""
        print(f"Building Indexes: {list(self.loaded_datasets.keys())}")
        for loaded_dataset in self.loaded_datasets.values():
            self.document_indexes[loaded_dataset.dataset_id] = DocumentIndex(
                self.harness, loaded_dataset,
            )
        return

