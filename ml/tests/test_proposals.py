from crrc_vision.proposals import (
    MODEL_REVISION,
    TEXT_PROMPT,
    TRANSFORMERS_VERSION,
    PilotAudit,
    Proposal,
    clip_box,
    pilot_can_expand,
    select_pilot_items,
    validate_loading_info,
    validate_transformers_version,
)


def test_clip_box_stays_inside_image():
    assert clip_box((-2.0, 3.0, 15.0, 12.0), width=10, height=10) == (
        0.0,
        3.0,
        10.0,
        7.0,
    )


def test_pilot_requires_precision_and_coverage():
    assert pilot_can_expand(PilotAudit(7, 5, 3, 12)) is True
    assert pilot_can_expand(PilotAudit(5, 7, 3, 12)) is False
    assert pilot_can_expand(PilotAudit(8, 2, 4, 12)) is False


def test_proposal_id_is_stable_and_content_sensitive():
    first = Proposal(
        "a.jpg", "fastener", (1.0, 2.0, 3.0, 4.0), 0.8, "grounding-dino"
    )
    same = Proposal(
        "a.jpg", "fastener", (1.0, 2.0, 3.0, 4.0), 0.8, "grounding-dino"
    )
    different = Proposal(
        "a.jpg", "pipe_joint", (1.0, 2.0, 3.0, 4.0), 0.8, "grounding-dino"
    )

    assert first.stable_id == same.stable_id
    assert first.stable_id != different.stable_id


def test_pilot_selection_covers_splits_and_density_buckets():
    items = [
        {
            "relative_path": f"{split}-{bucket}-{index}.jpg",
            "scene_group": f"{split}-{bucket}-{index}",
            "split": split,
            "focus_score": float(index),
            "candidate_count": count,
        }
        for split in ("train", "val")
        for bucket, count in enumerate((0, 1, 4, 20))
        for index in range(2)
    ]

    selected = select_pilot_items(items, count=12)

    assert len(selected) == 12
    assert {item["split"] for item in selected} == {"train", "val"}
    assert {item["candidate_count"] for item in selected} == {0, 1, 4, 20}
    assert selected == select_pilot_items(list(reversed(items)), count=12)


def test_model_revision_is_the_verified_clean_checkpoint():
    assert MODEL_REVISION == "a2bb814dd30d776dcf7e30523b00659f4f141c71"
    assert TEXT_PROMPT == "bolt. nut. screw. fastener. pipe joint."
    assert TRANSFORMERS_VERSION == "4.40.2"


def test_transformers_version_is_pinned_for_checkpoint_compatibility():
    assert validate_transformers_version("4.40.2") == ()
    assert validate_transformers_version("4.57.6") == ("INCOMPATIBLE_TRANSFORMERS_VERSION",)


def test_loading_info_rejects_partial_checkpoint_load():
    assert validate_loading_info(
        {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}
    ) == ()
    assert validate_loading_info(
        {"missing_keys": ["layer.weight"], "unexpected_keys": [], "mismatched_keys": []}
    ) == ("MISSING_MODEL_WEIGHTS",)
