from crrc_vision.reference_teacher import (
    TeacherPrediction,
    build_run_manifest,
    ensure_complete_selection,
    map_teacher_category,
    validate_checkpoint_globals,
    validate_ultralytics_version,
    xyxy_to_xywh,
)


def test_checkpoint_globals_reject_non_framework_types():
    assert validate_checkpoint_globals(
        [
            "torch.nn.modules.conv.Conv2d",
            "ultralytics.nn.tasks.DetectionModel",
        ]
    ) == ()
    assert validate_checkpoint_globals(["os.system"]) == (
        "UNSAFE_CHECKPOINT_GLOBAL",
    )


def test_teacher_mapping_is_explicitly_unconfirmed():
    category, status = map_teacher_category(1)

    assert category == "pipe_joint"
    assert status == "inferred_unconfirmed"


def test_prediction_id_is_stable_and_preserves_teacher_class():
    item = TeacherPrediction("a.jpg", 2, "class_2", (1, 2, 3, 4), 0.9)

    assert item.stable_id == item.stable_id
    assert item.to_dict()["teacher_class_id"] == 2


def test_selection_coverage_rejects_missing_and_extra_images():
    assert ensure_complete_selection(["a.jpg", "b.jpg"], ["a.jpg", "b.jpg"]) == ()
    assert ensure_complete_selection(["a.jpg", "b.jpg"], ["a.jpg"]) == (
        "INCOMPLETE_SELECTION_COVERAGE",
    )


def test_run_manifest_records_integrity_and_research_boundary():
    value = build_run_manifest(
        checkpoint_sha256="A" * 64,
        selection_sha256="B" * 64,
        truth_sha256="C" * 64,
        images=100,
        predictions=700,
    )

    assert value["safe_load"] == "weights_only"
    assert value["research_only"] is True
    assert value["truth_sha256_before"] == value["truth_sha256_after"]


def test_reference_runtime_version_is_pinned():
    assert validate_ultralytics_version("8.2.40") == ()
    assert validate_ultralytics_version("8.4.0") == (
        "INCOMPATIBLE_ULTRALYTICS_VERSION",
    )


def test_xyxy_conversion_clips_to_image_bounds():
    assert xyxy_to_xywh((-2.0, 3.0, 12.0, 14.0), width=10, height=10) == (
        0.0,
        3.0,
        10.0,
        7.0,
    )
