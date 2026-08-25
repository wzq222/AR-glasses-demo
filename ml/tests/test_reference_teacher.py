from crrc_vision.reference_teacher import (
    TeacherPrediction,
    ensure_complete_selection,
    map_teacher_category,
    validate_checkpoint_globals,
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
