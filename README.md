# Intel AI Inference Lab

Intel AI Inference Lab is a local benchmarking toolkit for comparing computer vision inference across ONNX Runtime and OpenVINO. It uses a shared preprocessing pipeline, repeatable latency measurement, lightweight output consistency checks, and Markdown/JSON reporting to make inference experiments easier to run, inspect, and reproduce.

I built this project to strengthen and demonstrate my AI software engineering skills, especially around model inference, runtime comparison, benchmarking, testing, and clean Python tooling.

This project does not use Intel branding, does not claim affiliation, and does not claim hardware-specific optimization unless measured locally.

ONNX Runtime is a performance-focused scoring engine for ONNX models, while OpenVINO is an open-source toolkit for optimizing and deploying AI inference across supported hardware. This project uses both to compare local inference behavior in a simple, reproducible workflow.

## Why I Built This

I wanted to build a project that goes beyond calling a model once and instead shows the engineering around AI inference:

- loading and running a computer vision model
- using multiple inference runtimes
- measuring latency and throughput
- checking output consistency
- generating reproducible reports
- writing unit tests around the tooling
- documenting limitations honestly

The goal is to show practical AI software engineering habits: clean interfaces, reproducible experiments, performance measurement, and careful documentation.

## Features

- Loads an ONNX computer vision model from a local `models/` directory
- Runs inference with `onnxruntime`
- Runs inference with `openvino`
- Uses one shared preprocessing pipeline for both runtimes
- Supports warmup and benchmark iterations
- Measures mean, median, p95, min, and max latency
- Computes approximate throughput in images per second
- Compares runtime outputs with a lightweight consistency check
- Generates benchmark reports in both Markdown and JSON
- Exposes a CLI for repeatable local experiments
- Includes unit tests for metrics, preprocessing, and report generation
- Includes an optional PyTorch-to-ONNX export helper

## Engineering Focus

This project demonstrates how to:

- integrate multiple inference backends behind a clean Python interface
- reason about latency, throughput, and output consistency
- design reproducible CLI workflows for local experimentation
- separate preprocessing, runtime execution, metrics, and reporting logic
- document tradeoffs without overstating benchmark results
- expose OpenVINO device selection for future hardware experiments

## Project Structure

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

1. The CLI parses benchmark settings such as model path, runtime choice, warmup iterations, benchmark iterations, and output directory.
2. A runtime adapter is created for ONNX Runtime, OpenVINO, or both.
3. Input images are resized, normalized, and converted to NCHW tensors.
4. Each backend runs warmup passes followed by timed benchmark passes.
5. Latency statistics and throughput are computed from timing data.
6. Output vectors are compared across runtimes with a lightweight consistency check.
7. Markdown and JSON reports are written to `reports/`.

## Tech Stack

- Python 3.10+
- NumPy
- Pillow
- ONNX Runtime
- OpenVINO
- Pytest
- Optional: PyTorch and Torchvision for ONNX export

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/YOUR_USERNAME/intel_ai_inference_lab.git
cd intel_ai_inference_lab
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

## Model Setup

Place a public ONNX classification model in `models/`.

Recommended starting point:

```text
mobilenetv2-7.onnx
```

Example path:

```text
models/mobilenetv2-7.onnx
```

If you want to generate your own ONNX model from Torchvision, use:

```bash
python scripts/export_torchvision_to_onnx.py --model-name mobilenet_v3_small --output models/mobilenet_v3_small.onnx
```

## Runtime Notes

### ONNX Runtime

- implemented in `inference_lab/runtimes/onnx_runner.py`
- expects a local ONNX model file
- currently uses the `CPUExecutionProvider`
- exits with a clear error message if `onnxruntime` is not installed

### OpenVINO

- implemented in `inference_lab/runtimes/openvino_runner.py`
- reads the same ONNX model and compiles it for the selected device
- defaults to `CPU`
- exposes `--device` for future experiments
- exits with a clear error message if `openvino` is not installed

This project is intended to make both inference paths easy to verify locally without claiming that every machine will produce identical performance or numerical output.

## Run an ONNX Runtime Benchmark

```bash
python -m inference_lab benchmark \
  --model models/mobilenetv2-7.onnx \
  --images sample_images \
  --runtime onnxruntime \
  --iterations 50 \
  --warmup 5
```

## Run an OpenVINO Benchmark

```bash
python -m inference_lab benchmark \
  --model models/mobilenetv2-7.onnx \
  --images sample_images \
  --runtime openvino \
  --iterations 50 \
  --warmup 5 \
  --device CPU
```

## Run Both Runtimes

```bash
python -m inference_lab benchmark \
  --model models/mobilenetv2-7.onnx \
  --images sample_images \
  --runtime both \
  --iterations 50 \
  --warmup 5 \
  --output-dir reports
```

## CLI Options

View all options:

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

## Run Tests

```bash
python -m pytest -q
```

Or with Make:

```bash
make test
```

## Sample Output Report

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

## Engineering Notes

- The consistency check is intentionally lightweight. It compares output vectors and top-1 predictions, but it is not a full accuracy benchmark.
- Throughput is derived from mean latency for a batch size of 1 in the current implementation.
- Unit tests do not require model downloads.
- OpenVINO device selection is exposed through the CLI so future experiments can target additional devices when available.
- Benchmark commands are intentionally short and reproducible so they are easy to rerun during demos or interviews.

## Validation Checklist

Before demoing the project:

```bash
python -m pytest -q
python -m inference_lab benchmark --help
```

Then:

1. Place a real ONNX model in `models/`.
2. Run one ONNX Runtime benchmark.
3. Run one OpenVINO benchmark.
4. Inspect the generated Markdown and JSON files in `reports/`.

## Limitations

- The repository does not bundle a pretrained ONNX model to keep the project lightweight.
- Current benchmarking assumes batch size 1 and focuses on single-process local execution.
- Sample images are small synthetic images intended to keep tests simple, not to represent a real evaluation dataset.
- No labeled dataset accuracy benchmark is included.
- OpenVINO and ONNX Runtime results can differ slightly due to backend-specific graph execution and numerical behavior.
- The project does not claim official Intel optimization results.

## Future Improvements

- Add optional batch benchmarking and richer throughput experiments
- Add support for more model families and object detection workflows
- Add CSV export and result visualization with Matplotlib
- Add optional provider selection for ONNX Runtime
- Add environment snapshotting for deeper reproducibility
- Extend OpenVINO device exploration for future hardware acceleration experiments
- Add optional profiling hooks for CPU utilization and memory usage

## Reproducibility Checklist

- Use a Python virtual environment
- Record exact package versions
- Keep the same ONNX model across runs
- Use the same input images and iteration counts
- Save generated reports under `reports/`
