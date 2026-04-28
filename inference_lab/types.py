from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class InferenceOutput:
    output_vector: list[float]
    top_index: int
    raw_shape: list[int]


@dataclass(slots=True)
class RuntimeConfig:
    model_path: Path
    warmup: int
    iterations: int
    batch_size: int = 1
    device: str = "CPU"


@dataclass(slots=True)
class RuntimeBenchmarkResult:
    runtime_name: str
    device: str
    provider: str
    warmup_iterations: int
    benchmark_iterations: int
    latency_ms: list[float]
    throughput_images_per_second: float
    output_samples: list[InferenceOutput] = field(default_factory=list)
    model_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConsistencyResult:
    compared_runtimes: list[str]
    is_consistent: bool
    average_cosine_similarity: float | None
    average_max_abs_diff: float | None
    top_index_match_rate: float | None
    tolerance: float
    note: str


@dataclass(slots=True)
class BenchmarkRunResult:
    model_name: str
    model_path: str
    image_count: int
    runtimes: list[RuntimeBenchmarkResult]
    consistency: ConsistencyResult | None
    environment: dict[str, str]
