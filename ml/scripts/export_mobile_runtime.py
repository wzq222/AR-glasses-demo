"""Convert one model with a pinned mobile runtime and preserve audit evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.mobile_benchmark import (
    PINNED_RUNTIME_REVISIONS,
    sha256_file,
)
from crrc_vision.mobile_runtime_export import (
    build_mnn_command,
    build_pnnx_command,
    normalize_captured_output,
    validate_checkout,
)


EXPECTED_TRUTH_SHA256 = "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"path escapes CRRC_VISION_DATA_ROOT: {relative}")
    return path


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("ncnn", "mnn"), required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--mnn-optimize-level", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--ncnn-fp32", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = asset_root().resolve()
    checkout = _below(root, args.checkout)
    tool = _below(root, args.tool)
    model = _below(root, args.model)
    truth = _below(root, args.truth)
    run = _below(root, args.run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("RUNTIME_EXPORT_RUN_NOT_EMPTY")
    run.mkdir(parents=True, exist_ok=True)
    revision = validate_checkout(checkout, PINNED_RUNTIME_REVISIONS[args.runtime])
    truth_before = sha256_file(truth)
    if truth_before != EXPECTED_TRUTH_SHA256:
        raise ValueError("FORMAL_TRUTH_HASH_MISMATCH")

    if args.runtime == "mnn":
        output_root = run / "mnn"
        output_root.mkdir(parents=True, exist_ok=True)
        expected_outputs = [output_root / "model.mnn"]
        command = build_mnn_command(
            tool,
            model,
            expected_outputs[0],
            optimize_level=args.mnn_optimize_level,
        )
    else:
        output_root = run / "ncnn"
        output_root.mkdir(parents=True, exist_ok=True)
        expected_outputs = [
            output_root / "model.ncnn.param",
            output_root / "model.ncnn.bin",
        ]
        command = build_pnnx_command(
            tool,
            model,
            output_root,
            fp16=not args.ncnn_fp32,
        )

    report: dict[str, object] = {
        "schema_version": "mobile-runtime-export-v1",
        "runtime": args.runtime,
        "runtime_revision": revision,
        "source_model": str(model),
        "source_model_sha256": sha256_file(model),
        "formal_truth_sha256_before": truth_before,
        "command": command,
        "execute_requested": args.execute,
        "status": "prepared",
    }
    if args.execute:
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True)
        (run / "stdout.log").write_bytes(normalize_captured_output(completed.stdout))
        (run / "stderr.log").write_bytes(normalize_captured_output(completed.stderr))
        artifacts = {
            str(path.relative_to(run)).replace("\\", "/"): sha256_file(path)
            for path in expected_outputs
            if path.is_file() and path.stat().st_size > 0
        }
        report.update(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "exit_code": completed.returncode,
                "artifacts": artifacts,
                "status": (
                    "converted"
                    if completed.returncode == 0 and len(artifacts) == len(expected_outputs)
                    else "conversion_failed"
                ),
            }
        )
    truth_after = sha256_file(truth)
    report["formal_truth_sha256_after"] = truth_after
    if truth_after != truth_before:
        raise RuntimeError("FORMAL_TRUTH_CHANGED")
    _atomic_json(run / "export-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] != "conversion_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
