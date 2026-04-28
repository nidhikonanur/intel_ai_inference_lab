from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}


def resolve_image_paths(image_path: Path) -> list[Path]:
    if image_path.is_file():
        return [image_path]

    paths = sorted(
        candidate
        for candidate in image_path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found in: {image_path}")
    return paths


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def collect_environment_info() -> dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python_implementation": platform.python_implementation(),
    }


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0. Received: {value}")


def resolve_nchw_shape(shape: Iterable[object], default_spatial: int = 224) -> tuple[int, int, int, int]:
    resolved: list[int] = []
    for index, value in enumerate(shape):
        if isinstance(value, int) and value > 0:
            resolved.append(value)
        elif index == 0:
            resolved.append(1)
        elif index == 1:
            resolved.append(3)
        else:
            resolved.append(default_spatial)
    if len(resolved) != 4:
        raise ValueError(f"Expected a 4D NCHW input shape, received: {list(shape)}")
    return tuple(resolved)  # type: ignore[return-value]


def slugify(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "-" for char in value)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-")
