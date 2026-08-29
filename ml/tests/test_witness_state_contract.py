from crrc_vision.witness_state_contract import validate_h1_record


def test_h1_record_requires_physical_state_contract() -> None:
    record = {
        "sample_id": "h1a-0001",
        "intent": "SUBTLE_DISPLACED",
        "output_state": "DISPLACED",
        "topology": "nut_plate",
        "mark_role": "bridges_moving_fixed",
        "has_marked_point": True,
        "source_split": "train",
        "eligible_split": "train",
        "source_scene_id": "scene-001",
        "source_reference_sha256": "A" * 64,
        "prompt_sha256": "B" * 64,
    }
    assert validate_h1_record(record) == ()


def test_lookalike_must_not_be_a_marked_point() -> None:
    record = {
        "sample_id": "h1a-0002",
        "intent": "LOOKALIKE",
        "output_state": None,
        "topology": "unknown",
        "mark_role": "ambiguous",
        "has_marked_point": True,
        "source_split": "train",
        "eligible_split": "train",
        "source_scene_id": "scene-002",
        "source_reference_sha256": "A" * 64,
        "prompt_sha256": "B" * 64,
    }
    assert "LOOKALIKE_MARKED_POINT_CONFLICT" in validate_h1_record(record)


def test_decidable_state_requires_bridge_across_moving_and_fixed_surfaces() -> None:
    record = {
        "sample_id": "h1a-0003",
        "intent": "SUBTLE_DISPLACED",
        "output_state": "DISPLACED",
        "topology": "fitting_pipe",
        "mark_role": "moving_only",
        "has_marked_point": True,
        "source_split": "train",
        "eligible_split": "train",
        "source_scene_id": "scene-003",
        "source_reference_sha256": "A" * 64,
        "prompt_sha256": "B" * 64,
    }
    assert "DECIDABLE_STATE_REQUIRES_BRIDGE" in validate_h1_record(record)


def test_decidable_state_rejects_unknown_connection_topology() -> None:
    record = {
        "sample_id": "h1a-0004",
        "intent": "ALIGNED",
        "output_state": "ALIGNED",
        "topology": "unknown",
        "mark_role": "bridges_moving_fixed",
        "has_marked_point": True,
        "source_split": "train",
        "eligible_split": "train",
        "source_scene_id": "scene-004",
        "source_reference_sha256": "A" * 64,
        "prompt_sha256": "B" * 64,
    }
    assert "DECIDABLE_STATE_REQUIRES_KNOWN_TOPOLOGY" in validate_h1_record(record)
