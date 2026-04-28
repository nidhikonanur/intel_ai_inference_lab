from __future__ import annotations

import math
import statistics


def summarize_latency(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        raise ValueError("Latency list cannot be empty.")

    sorted_values = sorted(latencies_ms)
    return {
        "mean": statistics.fmean(sorted_values),
        "median": statistics.median(sorted_values),
        "p95": percentile(sorted_values, 95.0),
        "min": sorted_values[0],
        "max": sorted_values[-1],
    }


def percentile(sorted_values: list[float], percentile_value: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty list.")
    if percentile_value < 0 or percentile_value > 100:
        raise ValueError("Percentile must be between 0 and 100.")

    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (percentile_value / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def throughput_from_latency(latencies_ms: list[float], batch_size: int = 1) -> float:
    mean_ms = summarize_latency(latencies_ms)["mean"]
    return 0.0 if mean_ms <= 0 else (1000.0 / mean_ms) * batch_size
