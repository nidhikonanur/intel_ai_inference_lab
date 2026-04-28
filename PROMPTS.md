# PROMPTS.md

## Build prompt

The project was initially created from the following prompt:

> You are helping me build an impressive AI software engineering portfolio project tailored for Intel’s AI Software Engineering Internship.
>
> Project name:
> intel_ai_inference_lab
>
> Goal:
> Build a polished, original AI inference optimization and benchmarking project that demonstrates Python AI software engineering, computer vision inference, ONNX Runtime, OpenVINO, model benchmarking, performance/accuracy tradeoff analysis, and clear engineering documentation.
>
> Target role requirements to optimize for:
> - Python AI software development
> - AI software solution design, development, and optimization
> - Computer vision, machine learning, and deep learning workflows
> - ONNX Runtime
> - OpenVINO
> - Optional PyTorch if feasible
> - Model performance and accuracy benchmarking
> - Hardware/software integration mindset
> - System-level deployment thinking
> - Clean API design
> - Testing, documentation, and reproducibility
>
> Project concept:
> Build an AI inference benchmarking toolkit called Intel AI Inference Lab. The tool should compare computer vision model inference across ONNX Runtime and OpenVINO on local hardware. It should run a small image classification or object detection model on sample images, measure latency and throughput, validate output consistency, and generate a clear benchmark report.

The full prompt also specified CLI, report generation, testing, documentation, structure, and reliability constraints.

## Follow-up prompts used during implementation

- Build the project as a clean standalone folder inside the shared workspace.
- Keep runtime-heavy dependencies optional for unit tests when possible.
- Use honest language around local benchmarking and output consistency checks.
- Add packaging metadata, a Makefile, and an optional Torchvision-to-ONNX export helper.
- Polish the project for an Intel AI Software Engineering internship submission.
- Review the project against code quality, runtime clarity, testing, documentation, and claim-safety criteria.
- Fix weak areas without adding fragile dependencies and summarize the exact commands needed to run it.

## AI-assisted coding disclosure

AI-assisted coding was used during scaffolding and implementation.

All generated code and documentation were reviewed, adapted, and organized for clarity, reproducibility, and honest technical claims. The project should still be manually validated on the target machine with the chosen ONNX model and installed runtimes before being presented as a finished benchmark result.
