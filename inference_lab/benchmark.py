from __future__ import annotations

from pathlib import Path

import numpy as np

from inference_lab.preprocess import load_image
from inference_lab.runtimes.onnx_runner import OnnxRuntimeRunner
from inference_lab.runtimes.openvino_runner import OpenVinoRunner
from inference_lab.types import (
    BenchmarkRunResult,
    ConsistencyResult,
    RuntimeBenchmarkResult,
    RuntimeConfig,
)
from inference_lab.utils import collect_environment_info, resolve_image_paths, validate_positive_int


def run_benchmark(
    model_path: Path,
    image_path: Path,
    runtime_name: str,
    iterations: int,
    warmup: int,
    device: str = "CPU",
) -> BenchmarkRunResult:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Place a public ONNX model in the models/ directory."
        )
    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    validate_positive_int("iterations", iterations)
    validate_positive_int("warmup", warmup)

    runners = _build_runners(model_path, runtime_name, iterations, warmup, device)
    reference_shape = runners[0].input_shape()
    image_size = (reference_shape[3], reference_shape[2])
    image_paths = resolve_image_paths(image_path)
    inputs = [load_image(path, image_size=image_size) for path in image_paths]

    results = [runner.benchmark(inputs, image_paths) for runner in runners]
    consistency = compare_runtime_outputs(results) if len(results) > 1 else None

    return BenchmarkRunResult(
        model_name=model_path.stem,
        model_path=str(model_path),
        image_count=len(image_paths),
        runtimes=results,
        consistency=consistency,
        environment=collect_environment_info(),
    )


def _build_runners(
    model_path: Path,
    runtime_name: str,
    iterations: int,
    warmup: int,
    device: str,
) -> list[OnnxRuntimeRunner | OpenVinoRunner]:
    config = RuntimeConfig(
        model_path=model_path,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    normalized = runtime_name.lower()
    if normalized == "onnxruntime":
        return [OnnxRuntimeRunner(config)]
    if normalized == "openvino":
        return [OpenVinoRunner(config)]
    if normalized == "both":
        return [OnnxRuntimeRunner(config), OpenVinoRunner(config)]
    raise ValueError("runtime must be one of: onnxruntime, openvino, both")


def compare_runtime_outputs(results: list[RuntimeBenchmarkResult], tolerance: float = 1e-3) -> ConsistencyResult:
    if len(results) < 2:
        raise ValueError("At least two runtime results are required for comparison.")

    baseline = results[0].output_samples
    candidate = results[1].output_samples
    pair_count = min(len(baseline), len(candidate))
    if pair_count == 0:
        raise ValueError("No runtime outputs available to compare.")

    cosine_scores: list[float] = []
    max_abs_diffs: list[float] = []
    top_index_matches = 0

    for index in range(pair_count):
        left = np.asarray(baseline[index].output_vector, dtype=np.float32)
        right = np.asarray(candidate[index].output_vector, dtype=np.float32)

        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        cosine = float(np.dot(left, right) / denom) if denom > 0 else 0.0
        cosine_scores.append(cosine)
        max_abs_diffs.append(float(np.max(np.abs(left - right))))
        if baseline[index].top_index == candidate[index].top_index:
            top_index_matches += 1

    average_cosine = float(np.mean(cosine_scores))
    average_max_abs_diff = float(np.mean(max_abs_diffs))
    match_rate = top_index_matches / pair_count
    is_consistent = match_rate == 1.0 and average_max_abs_diff <= tolerance

    return ConsistencyResult(
        compared_runtimes=[results[0].runtime_name, results[1].runtime_name],
        is_consistent=is_consistent,
        average_cosine_similarity=average_cosine,
        average_max_abs_diff=average_max_abs_diff,
        top_index_match_rate=match_rate,
        tolerance=tolerance,
        note=(
            "This is a lightweight functional consistency check across runtime outputs. "
            "It is not a substitute for a full accuracy evaluation on a labeled dataset."
        ),
    )
