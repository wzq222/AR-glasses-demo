from __future__ import annotations

from collections.abc import Mapping


H1_INTENTS = frozenset(
    {
        "ALIGNED",
        "SUBTLE_DISPLACED",
        "OBVIOUS_DISPLACED",
        "DAMAGED_MARK",
        "INSUFFICIENT",
        "LOOKALIKE",
    }
)
OUTPUT_STATES = frozenset(
    {"ALIGNED", "DISPLACED", "DAMAGED_MARK", "INSUFFICIENT"}
)
TOPOLOGIES = frozenset(
    {
        "bolt_head_plate",
        "nut_stud",
        "nut_plate",
        "double_nut",
        "fitting_pipe",
        "clamp_pipe",
        "unknown",
    }
)
MARK_ROLES = frozenset(
    {"bridges_moving_fixed", "moving_only", "fixed_only", "ambiguous"}
)
INTENT_TO_STATE = {
    "ALIGNED": "ALIGNED",
    "SUBTLE_DISPLACED": "DISPLACED",
    "OBVIOUS_DISPLACED": "DISPLACED",
    "DAMAGED_MARK": "DAMAGED_MARK",
    "INSUFFICIENT": "INSUFFICIENT",
    "LOOKALIKE": None,
}
DECIDABLE_INTENTS = frozenset(
    {"ALIGNED", "SUBTLE_DISPLACED", "OBVIOUS_DISPLACED", "DAMAGED_MARK"}
)


def _digest(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in text
    )


def validate_h1_record(record: Mapping[str, object]) -> tuple[str, ...]:
    errors: list[str] = []
    intent = str(record.get("intent", ""))
    if intent not in H1_INTENTS:
        errors.append("INVALID_INTENT")
    if record.get("output_state") != INTENT_TO_STATE.get(intent):
        errors.append("INTENT_STATE_MISMATCH")
    if str(record.get("topology", "")) not in TOPOLOGIES:
        errors.append("INVALID_TOPOLOGY")
    if str(record.get("mark_role", "")) not in MARK_ROLES:
        errors.append("INVALID_MARK_ROLE")
    if intent in DECIDABLE_INTENTS and record.get("mark_role") != "bridges_moving_fixed":
        errors.append("DECIDABLE_STATE_REQUIRES_BRIDGE")
    if intent in DECIDABLE_INTENTS and record.get("topology") == "unknown":
        errors.append("DECIDABLE_STATE_REQUIRES_KNOWN_TOPOLOGY")
    if intent == "LOOKALIKE" and record.get("has_marked_point") is not False:
        errors.append("LOOKALIKE_MARKED_POINT_CONFLICT")
    if intent != "LOOKALIKE" and record.get("has_marked_point") is not True:
        errors.append("POSITIVE_MARKED_POINT_REQUIRED")
    if record.get("source_split") != "train" or record.get("eligible_split") != "train":
        errors.append("SYNTHETIC_MUST_BE_TRAIN_ONLY")
    for key in ("source_reference_sha256", "prompt_sha256"):
        if not _digest(record.get(key)):
            errors.append(f"INVALID_{key.upper()}")
    if not str(record.get("sample_id", "")).strip() or not str(
        record.get("source_scene_id", "")
    ).strip():
        errors.append("MISSING_IDENTITY")
    return tuple(errors)
