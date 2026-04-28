import json
from pathlib import Path

from inference_lab.report import build_report_payload, render_markdown_report
from inference_lab.report import write_reports
from inference_lab.types import BenchmarkRunResult, ConsistencyResult, InferenceOutput, RuntimeBenchmarkResult


def test_report_payload_contains_runtime_and_consistency_data() -> None:
    result = BenchmarkRunResult(
        model_name="mobilenetv2",
        model_path="models/mobilenetv2.onnx",
        image_count=2,
        runtimes=[
            RuntimeBenchmarkResult(
                runtime_name="onnxruntime",
                device="CPU",
                provider="CPUExecutionProvider",
                warmup_iterations=2,
                benchmark_iterations=10,
                latency_ms=[1.0, 2.0, 3.0],
                throughput_images_per_second=500.0,
                output_samples=[InferenceOutput([0.1, 0.9], 1, [1, 2])],
            )
        ],
        consistency=ConsistencyResult(
            compared_runtimes=["onnxruntime", "openvino"],
            is_consistent=True,
            average_cosine_similarity=0.999,
            average_max_abs_diff=0.0002,
            top_index_match_rate=1.0,
            tolerance=0.001,
            note="Lightweight check.",
        ),
        environment={"python_version": "3.11.0", "platform": "test-platform"},
    )

    payload = build_report_payload(result)
    markdown = render_markdown_report(payload)

    assert payload["model_name"] == "mobilenetv2"
    assert payload["runtimes"][0]["latency_ms"]["mean"] == 2.0
    assert payload["consistency"]["is_consistent"] is True
    assert "Intel AI Inference Lab Benchmark Report" in markdown
    assert "Lightweight check." in markdown


def test_write_reports_creates_json_and_markdown_files(tmp_path: Path) -> None:
    result = BenchmarkRunResult(
        model_name="mobilenetv2",
        model_path="models/mobilenetv2.onnx",
        image_count=1,
        runtimes=[
            RuntimeBenchmarkResult(
                runtime_name="onnxruntime",
                device="CPU",
                provider="CPUExecutionProvider",
                warmup_iterations=2,
                benchmark_iterations=10,
                latency_ms=[1.0, 2.0, 3.0],
                throughput_images_per_second=500.0,
                output_samples=[InferenceOutput([0.1, 0.9], 1, [1, 2])],
            )
        ],
        consistency=None,
        environment={"python_version": "3.11.0", "platform": "test-platform"},
    )

    json_path, markdown_path = write_reports(result, tmp_path)

    assert json_path.exists()
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["model_path"] == "models/mobilenetv2.onnx"
    assert "Only one runtime was benchmarked" in markdown_path.read_text(encoding="utf-8")
