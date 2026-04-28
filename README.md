# Intel AI Inference Lab

Intel AI Inference Lab is a local benchmarking toolkit for comparing computer vision inference across ONNX Runtime and OpenVINO. It uses a shared preprocessing pipeline, repeatable latency measurement, lightweight output consistency checks, and Markdown/JSON reporting to make inference experiments easier to run, inspect, and reproduce.

I built this project to strengthen and demonstrate my AI software engineering skills, especially around model inference, runtime comparison, benchmarking, testing, and clean Python tooling.

This project does not use Intel branding, does not claim affiliation, and does not claim hardware-specific optimization unless measured locally.

ONNX Runtime is a performance-focused scoring engine for ONNX models, while OpenVINO is an open-source toolkit for optimizing and deploying AI inference across supported hardware. This project uses both to compare local inference behavior in a simple, reproducible workflow. :contentReference[oaicite:0]{index=0}

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
