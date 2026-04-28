from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from inference_lab.types import RuntimeBenchmarkResult, RuntimeConfig


class BaseRunner(ABC):
    runtime_name: str

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    @abstractmethod
    def input_shape(self) -> tuple[int, int, int, int]:
        raise NotImplementedError

    @abstractmethod
    def benchmark(self, inputs: list[np.ndarray], image_paths: list[Path]) -> RuntimeBenchmarkResult:
        raise NotImplementedError
