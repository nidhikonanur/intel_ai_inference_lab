from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from inference_lab.metrics import throughput_from_latency
from inference_lab.types import InferenceOutput, RuntimeBenchmarkResult, RuntimeConfig
from inference_lab.runtimes.base import BaseRunner
from inference_lab.utils import resolve_nchw_shape


class OnnxRuntimeRunner(BaseRunner):
    runtime_name = "onnxruntime"

    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__(config)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install dependencies from requirements.txt."
            ) from exc

        self._session = ort.InferenceSession(str(config.model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = tuple(self._session.get_inputs()[0].shape)

    def input_shape(self) -> tuple[int, int, int, int]:
        return resolve_nchw_shape(self._input_shape)

    def benchmark(self, inputs: list[np.ndarray], image_paths: list[Path]) -> RuntimeBenchmarkResult:
        latencies_ms: list[float] = []
        output_samples: list[InferenceOutput] = []

        for _ in range(self.config.warmup):
            for batch in inputs:
                self._session.run(None, {self._input_name: batch})

        for _ in range(self.config.iterations):
            for batch in inputs:
                start = time.perf_counter()
                outputs = self._session.run(None, {self._input_name: batch})
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latencies_ms.append(elapsed_ms)
                if len(output_samples) < len(image_paths):
                    output_samples.append(_extract_output(outputs[0]))

        return RuntimeBenchmarkResult(
            runtime_name=self.runtime_name,
            device=self.config.device,
            provider="CPUExecutionProvider",
            warmup_iterations=self.config.warmup,
            benchmark_iterations=self.config.iterations,
            latency_ms=latencies_ms,
            throughput_images_per_second=throughput_from_latency(latencies_ms, self.config.batch_size),
            output_samples=output_samples,
            model_path=str(self.config.model_path),
            extra={"session_providers": self._session.get_providers()},
        )


def _extract_output(output: np.ndarray) -> InferenceOutput:
    vector = np.asarray(output).reshape(-1).astype(np.float32)
    return InferenceOutput(
        output_vector=vector.tolist(),
        top_index=int(np.argmax(vector)),
        raw_shape=list(np.asarray(output).shape),
    )
