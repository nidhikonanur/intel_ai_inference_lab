from inference_lab.benchmark import compare_runtime_outputs
from inference_lab.types import InferenceOutput, RuntimeBenchmarkResult


def test_compare_runtime_outputs_reports_consistency() -> None:
    left = RuntimeBenchmarkResult(
        runtime_name="onnxruntime",
        device="CPU",
        provider="CPUExecutionProvider",
        warmup_iterations=1,
        benchmark_iterations=2,
        latency_ms=[1.0, 1.2],
        throughput_images_per_second=900.0,
        output_samples=[
            InferenceOutput([0.1, 0.9, 0.0], 1, [1, 3]),
            InferenceOutput([0.2, 0.7, 0.1], 1, [1, 3]),
        ],
    )
    right = RuntimeBenchmarkResult(
        runtime_name="openvino",
        device="CPU",
        provider="CPU",
        warmup_iterations=1,
        benchmark_iterations=2,
        latency_ms=[0.8, 1.0],
        throughput_images_per_second=1000.0,
        output_samples=[
            InferenceOutput([0.10001, 0.89999, 0.0], 1, [1, 3]),
            InferenceOutput([0.19999, 0.70001, 0.1], 1, [1, 3]),
        ],
    )

    consistency = compare_runtime_outputs([left, right], tolerance=1e-3)

    assert consistency.is_consistent is True
    assert consistency.top_index_match_rate == 1.0
    assert consistency.average_max_abs_diff is not None
