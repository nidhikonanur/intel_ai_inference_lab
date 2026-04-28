from __future__ import annotations

import argparse
from pathlib import Path

from inference_lab.benchmark import run_benchmark
from inference_lab.metrics import summarize_latency
from inference_lab.report import write_reports
from inference_lab.utils import ensure_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inference_lab",
        description="Benchmark ONNX computer vision inference across ONNX Runtime and OpenVINO.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run a local inference benchmark.")
    benchmark_parser.add_argument("--model", type=Path, required=True, help="Path to an ONNX model.")
    benchmark_parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to an image file or directory of images.",
    )
    benchmark_parser.add_argument(
        "--runtime",
        default="both",
        choices=["onnxruntime", "openvino", "both"],
        help="Runtime backend selection.",
    )
    benchmark_parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations.")
    benchmark_parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    benchmark_parser.add_argument("--device", default="CPU", help="OpenVINO device target.")
    benchmark_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for generated JSON and Markdown reports.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "benchmark":
        try:
            ensure_directory(args.output_dir)
            result = run_benchmark(
                model_path=args.model,
                image_path=args.images,
                runtime_name=args.runtime,
                iterations=args.iterations,
                warmup=args.warmup,
                device=args.device,
            )
            json_path, markdown_path = write_reports(result, args.output_dir)
            print_summary(result, json_path, markdown_path)
            return 0
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.exit(status=1, message=f"error: {exc}\n")

    parser.error(f"Unknown command: {args.command}")
    return 2


def print_summary(result, json_path: Path, markdown_path: Path) -> None:
    print(f"\nBenchmark complete for model: {result.model_name}")
    for runtime in result.runtimes:
        stats = summarize_latency(runtime.latency_ms)
        print(
            f"- {runtime.runtime_name}: mean={stats['mean']:.3f} ms, "
            f"median={stats['median']:.3f} ms, p95={stats['p95']:.3f} ms, "
            f"throughput={runtime.throughput_images_per_second:.3f} img/s"
        )
    if result.consistency is not None:
        print(
            "- consistency: "
            f"match_rate={result.consistency.top_index_match_rate:.3f}, "
            f"avg_max_abs_diff={result.consistency.average_max_abs_diff:.6f}, "
            f"consistent={result.consistency.is_consistent}"
        )
    print(f"- JSON report: {json_path}")
    print(f"- Markdown report: {markdown_path}\n")


if __name__ == "__main__":
    raise SystemExit(main())
