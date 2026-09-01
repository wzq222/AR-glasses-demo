from __future__ import annotations

import re


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_RUN_TOKEN = r"[A-Za-z0-9_-]{1,64}"
_BOX_PATTERN = re.compile(
    rf"\bbox image=(?P<file_name>\S+) index=(?P<index>\d+) "
    rf"class=(?P<class_id>-?\d+) score=(?P<score>{_FLOAT}) "
    rf"left=(?P<left>{_FLOAT}) top=(?P<top>{_FLOAT}) "
    rf"right=(?P<right>{_FLOAT}) bottom=(?P<bottom>{_FLOAT}) "
    rf"run_token=(?P<run_token>{_RUN_TOKEN})(?:\s|$)"
)
_SUMMARY_PATTERN = re.compile(
    rf"\bimage=(?P<file_name>\S+) run=(?P<run>\d+) "
    rf"detections=(?P<detections>\d+) total=(?P<total>{_FLOAT})ms "
    rf"preprocess=(?P<preprocess>{_FLOAT})ms "
    rf"inference=(?P<inference>{_FLOAT})ms "
    rf"postprocess=(?P<postprocess>{_FLOAT})ms "
    rf"run_token=(?P<run_token>{_RUN_TOKEN})(?:\s|$)"
)
_COMPLETE_PATTERN = re.compile(
    rf"\bcomplete images=(?P<images>\d+) "
    rf"run_token=(?P<run_token>{_RUN_TOKEN})(?:\s|$)"
)


def parse_box_line(line: str) -> dict[str, object] | None:
    """Parse one detailed Android benchmark box without trusting other log text."""
    match = _BOX_PATTERN.search(line)
    if match is None:
        return None
    left = float(match.group("left"))
    top = float(match.group("top"))
    right = float(match.group("right"))
    bottom = float(match.group("bottom"))
    if right <= left or bottom <= top:
        raise ValueError("ANDROID_BENCHMARK_BOX_INVALID")
    return {
        "file_name": match.group("file_name"),
        "index": int(match.group("index")),
        "class_id": int(match.group("class_id")),
        "score": float(match.group("score")),
        "bbox": [left, top, right - left, bottom - top],
        "run_token": match.group("run_token"),
    }


def parse_summary_line(line: str) -> dict[str, object] | None:
    """Parse one benchmark timing summary."""
    match = _SUMMARY_PATTERN.search(line)
    if match is None:
        return None
    return {
        "file_name": match.group("file_name"),
        "run": int(match.group("run")),
        "detections": int(match.group("detections")),
        "total_ms": float(match.group("total")),
        "preprocess_ms": float(match.group("preprocess")),
        "inference_ms": float(match.group("inference")),
        "postprocess_ms": float(match.group("postprocess")),
        "run_token": match.group("run_token"),
    }


def parse_complete_line(line: str) -> dict[str, object] | None:
    """Parse the terminal marker for one isolated benchmark run."""
    match = _COMPLETE_PATTERN.search(line)
    if match is None:
        return None
    return {
        "images": int(match.group("images")),
        "run_token": match.group("run_token"),
    }
