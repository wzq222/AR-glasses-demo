from crrc_vision.synthetic_audit import audit_records


def _record(index: int, state: str) -> dict:
    return {
        "sample_id": f"sample-{state.lower()}-{index}",
        "synthetic": True,
        "eligible_split": "train",
        "source_split": "train",
        "source_scene_id": f"scene-{index:02d}",
        "source_reference_sha256": f"{index + 1:064x}",
        "state": state,
        "review_status": "APPROVED",
    }


def test_audit_rejects_validation_lineage() -> None:
    record = _record(0, "NORMAL")
    record["source_split"] = "val"
    result = audit_records([record])
    assert result.passed is False
    assert any("source_split" in error for error in result.errors)


def test_audit_requires_balanced_approved_states() -> None:
    records = [_record(index, "NORMAL") for index in range(8)]
    result = audit_records(records)
    assert result.passed is False
    assert result.approved_by_state["NORMAL"] == 8
    assert result.approved_by_state["SLIGHT_LOOSE"] == 0


def test_audit_passes_balanced_pilot() -> None:
    records = []
    for state_offset, state in enumerate(("NORMAL", "SLIGHT_LOOSE", "OBVIOUS_LOOSE")):
        records.extend(_record(state_offset * 20 + index, state) for index in range(8))
    result = audit_records(records)
    assert result.passed is True
    assert result.approved_total == 24
