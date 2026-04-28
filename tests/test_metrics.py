from inference_lab.metrics import percentile, summarize_latency, throughput_from_latency


def test_summarize_latency_returns_expected_values() -> None:
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = summarize_latency(latencies)

    assert stats["mean"] == 3.0
    assert stats["median"] == 3.0
    assert stats["p95"] == 4.8
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0


def test_percentile_rejects_invalid_values() -> None:
    try:
        percentile([1.0], 101.0)
    except ValueError as exc:
        assert "between 0 and 100" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid percentile.")


def test_throughput_from_latency_uses_mean_latency() -> None:
    throughput = throughput_from_latency([10.0, 10.0, 10.0], batch_size=2)
    assert throughput == 200.0
