import json

import pytest

from activation.dataset import (
    DatasetManager,
)
from activation.harness import (
    HarnessRuntime,
    HarnessRuntimeConfig,
    ModelConfig,
    FREE_DEVICE,
    SUPPORTS_FP4,
)


class DatasetStudyGenerator:
    def __init__(self, harness: HarnessRuntime):
        self.harness = harness

    def study_dataset(self, harness: HarnessRuntime):
        self.harness = harness
