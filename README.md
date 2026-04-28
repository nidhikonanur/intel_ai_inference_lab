# Intel AI Inference Lab

Intel AI Inference Lab is a portfolio project tailored for Intel AI Software Engineering internship preparation. It is an original, local benchmarking toolkit for comparing computer vision inference across ONNX Runtime and OpenVINO with a consistent preprocessing pipeline, repeatable latency measurement, and honest output consistency checks.

This project does not use Intel branding, does not claim affiliation, and does not claim hardware-specific optimization unless you measure it on your own machine. The goal is to demonstrate strong AI software engineering habits: clean Python design, inference workflows, runtime comparison, reporting, testing, and reproducibility.

## Why this project is relevant

This project is designed to showcase the skills commonly expected in AI software engineering roles:

- Python AI software development
- AI solution design, implementation, and optimization
- Computer vision inference workflows
- ONNX Runtime integration
- OpenVINO integration
- Model performance benchmarking
- Lightweight functional validation across backends
- System-minded engineering, documentation, and reproducibility

## Features

- Loads an ONNX computer vision model from a local `models/` directory
- Runs inference with `onnxruntime`
- Runs inference with `openvino`
- Uses one shared preprocessing pipeline for both runtimes
- Supports warmup and benchmark iterations
- Measures mean, median, p95, min, max latency
- Computes approximate throughput in images per second
- Compares runtime outputs with a lightweight consistency check
- Generates benchmark reports in both Markdown and JSON
- Exposes a simple CLI for repeatable local experiments
- Includes unit tests for metrics, preprocessing, and report generation
- Includes an optional PyTorch to ONNX export helper

## Submission fit

This project is a good internship submission because it demonstrates more than just model usage. It shows how to:

- integrate multiple inference backends behind a clean Python interface
- reason about latency, throughput, and output consistency
- design reproducible CLI workflows for local experimentation
- document tradeoffs honestly without overstating optimization claims
- connect model execution to a hardware and systems mindset through OpenVINO device selection and runtime comparison

## Project structure

```text
intel_ai_inference_lab/
├── inference_lab/
│   ├── __init__.py
│   ├── __main__.py
│   ├── benchmark.py
│   ├── cli.py
│   ├── metrics.py
│   ├── preprocess.py
│   ├── report.py
│   ├── types.py
│   ├── utils.py
│   └── runtimes/
│       ├── base.py
│       ├── onnx_runner.py
│       └── openvino_runner.py
├── models/
├── reports/
├── sample_images/
├── scripts/
├── tests/
├── Makefile
├── PROMPTS.md
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Architecture

1. CLI parses benchmark settings such as model path, runtime choice, warmup, and iterations.
2. A runtime adapter is created for ONNX Runtime, OpenVINO, or both.
3. Input images are resized, normalized, and converted to NCHW tensors.
4. Each backend runs warmup passes followed by timed benchmark passes.
5. Latency statistics and throughput are computed from the timing data.
6. Output vectors are compared across runtimes with a lightweight consistency check.
7. A Markdown report and a JSON report are written to `reports/`.

## Tech stack

- Python 3.10+
- NumPy
- Pillow
- ONNX Runtime
- OpenVINO
- Pytest
- Optional: PyTorch and Torchvision for ONNX export

## Setup

```bash
cd /Users/nidhikonanur/Documents/Playground/intel_ai_inference_lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you want the lightest validation path first, install only the unit test dependencies:

```bash
python -m pip install pytest numpy pillow
```

Optional export dependencies:

```bash
python -m pip install "torch>=2.2" "torchvision>=0.17"
```

## Model setup

Place a public ONNX classification model in `models/`.

Recommended starting point:

- `mobilenetv2-7.onnx`

Example path:

```text
models/mobilenetv2-7.onnx
```

If you want to generate your own ONNX model from Torchvision, use:

```bash
python scripts/export_torchvision_to_onnx.py --model-name mobilenet_v3_small --output models/mobilenet_v3_small.onnx
```

## Runtime notes

ONNX Runtime path:

- implemented in [inference_lab/runtimes/onnx_runner.py](/Users/nidhikonanur/Documents/Playground/intel_ai_inference_lab/inference_lab/runtimes/onnx_runner.py)
- expects a local ONNX model file
- currently uses the `CPUExecutionProvider`
- if `onnxruntime` is not installed, the CLI exits with a clear error message

OpenVINO path:

- implemented in [inference_lab/runtimes/openvino_runner.py](/Users/nidhikonanur/Documents/Playground/intel_ai_inference_lab/inference_lab/runtimes/openvino_runner.py)
- reads the same ONNX model and compiles it for the selected device
- default device is `CPU`, but the CLI exposes `--device` for future experiments
- if `openvino` is not installed, the CLI exits with a clear error message

This project is intended to make both inference paths easy to verify locally without claiming that every machine will produce identical performance or numerical output.

## How to run ONNX Runtime benchmark

```bash
python -m inference_lab benchmark \
  --model models/mobilenetv2-7.onnx \
  --images sample_images \
  --runtime onnxruntime \
  --iterations 50 \
  --warmup 5
```

## How to run OpenVINO benchmark

```bash
python -m inference_lab benchmark \
  --model models/mobilenetv2-7.onnx \
  --images sample_images \
  --runtime openvino \
  --iterations 50 \
  --warmup 5 \
  --device CPU
```

## How to run both runtimes

```bash
python -m inference_lab benchmark \
  --model models/mobilenetv2-7.onnx \
  --images sample_images \
  --runtime both \
  --iterations 50 \
  --warmup 5 \
  --output-dir reports
```

## CLI options

```bash
python -m inference_lab benchmark --help
```

Supported options:

- `--model`: path to an ONNX model
- `--images`: image file or directory
- `--runtime`: `onnxruntime`, `openvino`, or `both`
- `--iterations`: number of timed benchmark iterations
- `--warmup`: number of warmup iterations
- `--device`: OpenVINO target device, default `CPU`
- `--output-dir`: report directory

## How to run tests

```bash
python -m pytest -q
```

Or with `make`:

```bash
make test
```

## Sample output report

Example summary from a generated Markdown report:

```md
# Intel AI Inference Lab Benchmark Report

- Model: `mobilenetv2-7`
- Image count: `2`

## Runtime Results

### onnxruntime
- Mean latency (ms): `4.812`
- P95 latency (ms): `5.102`
- Throughput (images/sec): `207.814`

### openvino
- Mean latency (ms): `3.954`
- P95 latency (ms): `4.380`
- Throughput (images/sec): `252.908`

## Consistency Check
- Consistent within tolerance: `True`
- Top-1 index match rate: `1.0`
```

## Engineering notes

- The consistency check is intentionally lightweight. It compares output vectors and top-1 predictions, but it is not a full accuracy benchmark.
- Throughput is derived from mean latency for a batch size of 1 in the current implementation.
- The project is structured so unit tests do not require model downloads.
- OpenVINO device selection is exposed through the CLI so future experiments can target additional devices when available.
- Benchmark commands are intentionally short and reproducible so they are easy to rerun during interview demos.

## Validation checklist

Before submitting or demoing the project:

- run `python -m pytest -q`
- confirm `python -m inference_lab benchmark --help` works
- place a real ONNX model in `models/`
- run one ONNX Runtime benchmark
- run one OpenVINO benchmark
- inspect the generated Markdown and JSON files in `reports/`

## Limitations

- The repository does not bundle a pretrained ONNX model to keep the project lightweight.
- Current benchmarking assumes batch size 1 and focuses on single-process local execution.
- Sample images are tiny synthetic images intended to keep tests simple, not to represent a real evaluation dataset.
- No labeled dataset accuracy benchmark is included.
- OpenVINO and ONNX Runtime results can differ slightly due to backend-specific graph execution and numerical behavior.
- The README references Intel only as internship preparation context and does not claim affiliation, endorsement, or official Intel optimization results.

## Future improvements

- Add optional batch benchmarking and richer throughput experiments
- Add support for more model families and object detection workflows
- Add CSV export and result visualization with Matplotlib
- Add optional provider selection for ONNX Runtime
- Add environment snapshotting for deeper reproducibility
- Extend OpenVINO device exploration for future Intel hardware acceleration experiments

## Reproducibility checklist

- Use a Python virtual environment
- Record exact package versions
- Keep the same ONNX model across runs
- Use the same input images and iteration counts
- Save generated reports under `reports/`
