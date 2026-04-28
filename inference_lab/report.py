from __future__ import annotations

import json
from pathlib import Path

from inference_lab.metrics import summarize_latency
from inference_lab.types import BenchmarkRunResult
from inference_lab.utils import ensure_directory, slugify, utc_timestamp


def build_report_payload(result: BenchmarkRunResult) -> dict:
    runtime_payloads = []
    for runtime in result.runtimes:
        runtime_payloads.append(
            {
                "runtime_name": runtime.runtime_name,
                "device": runtime.device,
                "provider": runtime.provider,
                "warmup_iterations": runtime.warmup_iterations,
                "benchmark_iterations": runtime.benchmark_iterations,
                "latency_ms": summarize_latency(runtime.latency_ms),
                "throughput_images_per_second": runtime.throughput_images_per_second,
                "sample_top_indices": [sample.top_index for sample in runtime.output_samples],
                "sample_output_shapes": [sample.raw_shape for sample in runtime.output_samples],
                "extra": runtime.extra,
            }
        )

    return {
        "project": "Intel AI Inference Lab",
        "generated_at_utc": utc_timestamp(),
        "model_name": result.model_name,
        "model_path": result.model_path,
        "image_count": result.image_count,
        "environment": result.environment,
        "runtimes": runtime_payloads,
        "consistency": None
        if result.consistency is None
        else {
            "compared_runtimes": result.consistency.compared_runtimes,
            "is_consistent": result.consistency.is_consistent,
            "average_cosine_similarity": result.consistency.average_cosine_similarity,
            "average_max_abs_diff": result.consistency.average_max_abs_diff,
            "top_index_match_rate": result.consistency.top_index_match_rate,
            "tolerance": result.consistency.tolerance,
            "note": result.consistency.note,
        },
    }


def render_markdown_report(payload: dict) -> str:
    lines = [
        "# Intel AI Inference Lab Benchmark Report",
        "",
        f"- Generated at (UTC): `{payload['generated_at_utc']}`",
        f"- Model: `{payload['model_name']}`",
        f"- Model path: `{payload['model_path']}`",
        f"- Image count: `{payload['image_count']}`",
        "",
        "## Environment",
        "",
    ]

    for key, value in payload["environment"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Runtime Results", ""])
    for runtime in payload["runtimes"]:
        latency = runtime["latency_ms"]
        lines.extend(
            [
                f"### {runtime['runtime_name']}",
                "",
                f"- Device: `{runtime['device']}`",
                f"- Provider: `{runtime['provider']}`",
                f"- Warmup iterations: `{runtime['warmup_iterations']}`",
                f"- Benchmark iterations: `{runtime['benchmark_iterations']}`",
                f"- Mean latency (ms): `{latency['mean']:.3f}`",
                f"- Median latency (ms): `{latency['median']:.3f}`",
                f"- P95 latency (ms): `{latency['p95']:.3f}`",
                f"- Min latency (ms): `{latency['min']:.3f}`",
                f"- Max latency (ms): `{latency['max']:.3f}`",
                f"- Throughput (images/sec): `{runtime['throughput_images_per_second']:.3f}`",
                f"- Sample top indices: `{runtime['sample_top_indices']}`",
                "",
            ]
        )

    lines.extend(["## Consistency Check", ""])
    consistency = payload["consistency"]
    if consistency is None:
        lines.append(
            "Only one runtime was benchmarked, so no cross-runtime consistency check was generated."
        )
    else:
        lines.extend(
            [
                f"- Compared runtimes: `{consistency['compared_runtimes']}`",
                f"- Consistent within tolerance: `{consistency['is_consistent']}`",
                f"- Average cosine similarity: `{consistency['average_cosine_similarity']}`",
                f"- Average max absolute difference: `{consistency['average_max_abs_diff']}`",
                f"- Top-1 index match rate: `{consistency['top_index_match_rate']}`",
                f"- Tolerance: `{consistency['tolerance']}`",
                "",
                consistency["note"],
            ]
        )

    return "\n".join(lines) + "\n"


def write_reports(result: BenchmarkRunResult, output_dir: Path) -> tuple[Path, Path]:
    ensure_directory(output_dir)
    payload = build_report_payload(result)
    base_name = f"{slugify(result.model_name)}-{slugify(payload['generated_at_utc'])}"
    json_path = output_dir / f"{base_name}.json"
    markdown_path = output_dir / f"{base_name}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown_report(payload), encoding="utf-8")
    return json_path, markdown_path
