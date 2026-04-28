from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a lightweight torchvision classification model to ONNX."
    )
    parser.add_argument(
        "--model-name",
        default="mobilenet_v3_small",
        help="Torchvision model function name, for example mobilenet_v3_small or resnet18.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/torchvision-export.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument("--height", type=int, default=224, help="Input height.")
    parser.add_argument("--width", type=int, default=224, help="Input width.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        import torchvision.models as models
    except ImportError as exc:
        raise SystemExit(
            "torch and torchvision are required for export. Install them manually before running this script."
        ) from exc

    if not hasattr(models, args.model_name):
        available = sorted(name for name in dir(models) if not name.startswith("_"))
        raise SystemExit(f"Unknown model '{args.model_name}'. Available examples: {available[:20]}")

    model_builder = getattr(models, args.model_name)
    model = model_builder(weights="DEFAULT")
    model.eval()

    dummy = torch.randn(1, 3, args.height, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
