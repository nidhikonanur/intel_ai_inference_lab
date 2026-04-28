from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from inference_lab.metrics import throughput_from_latency
from inference_lab.runtimes.base import BaseRunner
from inference_lab.types import InferenceOutput, RuntimeBenchmarkResult, RuntimeConfig
from inference_lab.utils import resolve_nchw_shape


class OpenVinoRunner(BaseRunner):
    runtime_name = "openvino"

    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__(config)
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError(
                "openvino is not installed. Install dependencies from requirements.txt."
            ) from exc

        self._core = ov.Core()
        self._model = self._core.read_model(model=str(config.model_path))
        self._compiled_model = self._core.compile_model(self._model, config.device)
        self._infer_request = self._compiled_model.create_infer_request()
        self._input_port = self._compiled_model.input(0)
        self._output_port = self._compiled_model.output(0)
        self._shape = tuple(self._input_port.partial_shape)

    def input_shape(self) -> tuple[int, int, int, int]:
        return resolve_nchw_shape(self._shape)

    def benchmark(self, inputs: list[np.ndarray], image_paths: list[Path]) -> RuntimeBenchmarkResult:
        latencies_ms: list[float] = []
        output_samples: list[InferenceOutput] = []

        for _ in range(self.config.warmup):
            for batch in inputs:
                self._infer_request.infer({self._input_port.any_name: batch})

        for _ in range(self.config.iterations):
            for batch in inputs:
                start = time.perf_counter()
                outputs = self._infer_request.infer({self._input_port.any_name: batch})
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latencies_ms.append(elapsed_ms)
                if len(output_samples) < len(image_paths):
                    output = outputs[self._output_port]
                    output_samples.append(_extract_output(output))

        return RuntimeBenchmarkResult(
            runtime_name=self.runtime_name,
            device=self.config.device,
            provider=self.config.device,
            warmup_iterations=self.config.warmup,
            benchmark_iterations=self.config.iterations,
            latency_ms=latencies_ms,
            throughput_images_per_second=throughput_from_latency(latencies_ms, self.config.batch_size),
            output_samples=output_samples,
            model_path=str(self.config.model_path),
            extra={"available_devices": self._core.available_devices},
        )


def _extract_output(output: np.ndarray) -> InferenceOutput:
    vector = np.asarray(output).reshape(-1).astype(np.float32)
    return InferenceOutput(
        output_vector=vector.tolist(),
        top_index=int(np.argmax(vector)),
        raw_shape=list(np.asarray(output).shape),
    )
