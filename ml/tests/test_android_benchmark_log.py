from __future__ import annotations

from crrc_vision.android_benchmark_log import (
    parse_box_line,
    parse_complete_line,
    parse_summary_line,
)


def test_parse_box_line_extracts_phone_detection() -> None:
    line = (
        "09-01 15:03:37.265 3329 3388 I DetectorBenchmark: "
        "box image=scene.jpg index=2 class=0 score=0.8125 "
        "left=10.25 top=20.5 right=40.75 bottom=60.0 run_token=run-abc_123"
    )

    assert parse_box_line(line) == {
        "file_name": "scene.jpg",
        "index": 2,
        "class_id": 0,
        "score": 0.8125,
        "bbox": [10.25, 20.5, 30.5, 39.5],
        "run_token": "run-abc_123",
    }


def test_parse_box_line_ignores_non_box_log() -> None:
    assert parse_box_line("DetectorBenchmark: image=scene.jpg run=0 detections=4") is None


def test_parse_summary_line_extracts_stage_timings() -> None:
    line = (
        "DetectorBenchmark: image=scene.jpg run=0 detections=4 total=123.5ms "
        "preprocess=20.0ms inference=100.0ms postprocess=3.5ms "
        "run_token=run-abc_123"
    )

    assert parse_summary_line(line) == {
        "file_name": "scene.jpg",
        "run": 0,
        "detections": 4,
        "total_ms": 123.5,
        "preprocess_ms": 20.0,
        "inference_ms": 100.0,
        "postprocess_ms": 3.5,
        "run_token": "run-abc_123",
    }


def test_parse_complete_line_extracts_current_run_contract() -> None:
    line = "DetectorBenchmark: complete images=17 run_token=run-abc_123"

    assert parse_complete_line(line) == {
        "images": 17,
        "run_token": "run-abc_123",
    }


def test_lines_without_run_token_are_rejected() -> None:
    assert parse_box_line(
        "DetectorBenchmark: box image=scene.jpg index=0 class=0 score=0.8 "
        "left=0 top=0 right=1 bottom=1"
    ) is None
    assert parse_summary_line(
        "DetectorBenchmark: image=scene.jpg run=0 detections=1 total=1ms "
        "preprocess=0ms inference=1ms postprocess=0ms"
    ) is None
