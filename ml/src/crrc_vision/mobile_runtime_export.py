"""Deterministic command contracts for ncnn and MNN model conversion."""

from __future__ import annotations

import subprocess
from pathlib import Path


def normalize_captured_output(payload: bytes | None) -> bytes:
    """Return captured subprocess output without applying a locale codec."""

    return payload or b""


def _required_file(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def build_mnn_command(
    converter: Path,
    model: Path,
    output: Path,
    *,
    optimize_level: int = 1,
) -> list[str]:
    """Build an argument-safe ONNX-to-MNN conversion command."""

    if optimize_level not in (0, 1, 2):
        raise ValueError("INVALID_MNN_OPTIMIZE_LEVEL")
    executable = _required_file(converter)
    source = _required_file(model)
    command = [
        str(executable),
        "-f",
        "ONNX",
        "--modelFile",
        str(source),
        "--MNNModel",
        str(output.resolve()),
        "--bizCode",
        "crrc-fastener",
    ]
    if optimize_level != 1:
        command.extend(["--optimizeLevel", str(optimize_level)])
    return command


def build_pnnx_command(
    converter: Path,
    model: Path,
    output: Path,
    *,
    fp16: bool,
) -> list[str]:
    """Build a fixed-shape pnnx command with every artifact path explicit."""

    executable = _required_file(converter)
    source = _required_file(model)
    root = output.resolve()
    return [
        str(executable),
        str(source),
        "inputshape=[1,3,640,640]",
        f"fp16={1 if fp16 else 0}",
        f"pnnxparam={root / 'model.pnnx.param'}",
        f"pnnxbin={root / 'model.pnnx.bin'}",
        f"pnnxpy={root / 'model_pnnx.py'}",
        f"pnnxonnx={root / 'model.pnnx.onnx'}",
        f"ncnnparam={root / 'model.ncnn.param'}",
        f"ncnnbin={root / 'model.ncnn.bin'}",
        f"ncnnpy={root / 'model_ncnn.py'}",
    ]


def build_cmake_configure_command(
    *,
    cmake: Path,
    source: Path,
    build: Path,
    definitions: dict[str, str],
) -> list[str]:
    """Build a deterministic Ninja configure command."""

    executable = _required_file(cmake)
    source_root = source.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    command = [
        str(executable),
        "-S",
        str(source_root),
        "-B",
        str(build.resolve()),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    command.extend(f"-D{name}={definitions[name]}" for name in sorted(definitions))
    return command


def build_cmake_build_command(cmake: Path, build: Path, target: str) -> list[str]:
    """Build one named release target without using a shell."""

    if not target.strip():
        raise ValueError("CMAKE_TARGET_REQUIRED")
    return [
        str(_required_file(cmake)),
        "--build",
        str(build.resolve()),
        "--target",
        target,
        "--config",
        "Release",
    ]


def validate_checkout(checkout: Path, expected_revision: str) -> str:
    """Return HEAD only when the checkout is exactly the pinned revision."""

    root = checkout.resolve()
    if not (root / ".git").exists():
        raise FileNotFoundError(root / ".git")
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if revision != expected_revision:
        raise RuntimeError(
            f"RUNTIME_REVISION_MISMATCH:{revision} != {expected_revision}"
        )
    return revision
