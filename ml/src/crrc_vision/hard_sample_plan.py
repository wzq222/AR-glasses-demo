from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
from itertools import combinations

from .witness_state_contract import (
    H1_INTENTS,
    INTENT_TO_STATE,
    MARK_ROLES,
    TOPOLOGIES,
    validate_h1_record,
)


H1A_QUOTAS = {
    "ALIGNED": 4,
    "SUBTLE_DISPLACED": 6,
    "OBVIOUS_DISPLACED": 4,
    "DAMAGED_MARK": 4,
    "INSUFFICIENT": 3,
    "LOOKALIKE": 3,
}

COMMON_PROMPT = (
    "Create one photorealistic close-up phone inspection photo using the supplied real "
    "rail-equipment crop as the structural reference. Preserve the connection topology, "
    "camera viewpoint, metal geometry, grime, rust, oil, shadows and surrounding industrial "
    "context. No illustration, CGI, text, watermark, duplicate fasteners, melted geometry "
    "or floating components. Change only the state requested below."
)

STATE_PROMPTS = {
    "ALIGNED": (
        "Keep the moving component physically fixed; the two paint fragments bridge both "
        "components and remain aligned, but add one realistic difficulty: glare, mild blur, "
        "partial shadow or worn paint."
    ),
    "SUBTLE_DISPLACED": (
        "Rotate only the moving component and its attached paint fragment by 2 to 8 degrees "
        "around the true joint axis; keep the fixed component and fixed-side paint unchanged; "
        "the result must be subtle but physically consistent."
    ),
    "OBVIOUS_DISPLACED": (
        "Rotate only the moving component and its attached paint fragment by 18 to 35 degrees "
        "around the true joint axis; keep the fixed component unchanged and preserve mechanically "
        "valid contact."
    ),
    "DAMAGED_MARK": (
        "Do not move either mechanical component; create irregular cured-paint chipping, cracking, "
        "fading or contamination that breaks visual continuity without a rigid relative rotation."
    ),
    "INSUFFICIENT": (
        "Keep the physical state ambiguous because one paint fragment is occluded, out of focus, "
        "overexposed or only one side is marked; do not invent a visible displacement."
    ),
    "LOOKALIKE": (
        "Remove any valid paint bridge and show a realistic red or yellow lookalike such as "
        "heat-shrink tubing, terminal sleeve, warning paint, rust or reflection; it must not be "
        "a marked inspection point."
    ),
}

_PHYSICALLY_DECIDABLE = frozenset(
    {"ALIGNED", "SUBTLE_DISPLACED", "OBVIOUS_DISPLACED", "DAMAGED_MARK"}
)


def _topology_record(value: object) -> tuple[str, str]:
    if isinstance(value, str):
        topology, mark_role = value, "bridges_moving_fixed"
    elif isinstance(value, Mapping):
        topology = str(value.get("topology", ""))
        mark_role = str(value.get("mark_role", ""))
        if value.get("decision", "APPROVED") != "APPROVED":
            raise ValueError("topology decision must be APPROVED")
    else:
        raise ValueError("topology decision must be a string or mapping")
    if topology not in TOPOLOGIES:
        raise ValueError(f"invalid topology: {topology}")
    if mark_role not in MARK_ROLES:
        raise ValueError(f"invalid mark role: {mark_role}")
    return topology, mark_role


def _allowed_intents(mark_role: str) -> tuple[str, ...]:
    if mark_role == "bridges_moving_fixed":
        return tuple(H1A_QUOTAS)
    return ("INSUFFICIENT", "LOOKALIKE")


def _assign_intent_pairs(
    reference_rows: Sequence[tuple[str, str]], seed: int
) -> dict[str, tuple[str, str]]:
    rng = random.Random(seed)
    remaining = Counter(H1A_QUOTAS)
    ordered = sorted(
        reference_rows,
        key=lambda row: (len(_allowed_intents(row[1])), row[0]),
    )
    candidates: dict[str, list[tuple[str, str]]] = {}
    for reference_id, mark_role in ordered:
        pairs = list(combinations(_allowed_intents(mark_role), 2))
        rng.shuffle(pairs)
        candidates[reference_id] = pairs

    assignment: dict[str, tuple[str, str]] = {}

    def search(index: int) -> bool:
        if index == len(ordered):
            return not any(remaining.values())
        reference_id, _ = ordered[index]
        for pair in candidates[reference_id]:
            if all(remaining[intent] > 0 for intent in pair):
                for intent in pair:
                    remaining[intent] -= 1
                assignment[reference_id] = pair
                if search(index + 1):
                    return True
                assignment.pop(reference_id, None)
                for intent in pair:
                    remaining[intent] += 1
        return False

    if not search(0):
        raise ValueError("topology decisions cannot satisfy the frozen H1a quota")
    return assignment


def build_hard_sample_prompt(job: Mapping[str, object]) -> str:
    intent = str(job["intent"])
    if intent not in STATE_PROMPTS:
        raise ValueError(f"unknown H1 intent: {intent}")
    topology = str(job["topology"])
    role = str(job["mark_role"])
    return (
        f"{COMMON_PROMPT} The physical connection topology is {topology}; the observed paint "
        f"role is {role}. {STATE_PROMPTS[intent]} Keep the same single inspection checkpoint "
        "and enough surrounding fixed and moving surfaces to judge the requested evidence."
    )


def build_h1a_jobs(
    references: Sequence[Mapping[str, object]],
    topology_by_reference: Mapping[str, object],
    seed: int = 20260829,
) -> list[dict[str, object]]:
    if len(references) != 12:
        raise ValueError("H1a requires exactly 12 references")
    reference_ids = [str(row.get("reference_id", "")) for row in references]
    if any(not value for value in reference_ids) or len(set(reference_ids)) != len(reference_ids):
        raise ValueError("reference_id values must be non-empty and unique")
    if set(reference_ids) != set(topology_by_reference):
        raise ValueError("topology decisions must match the 12 references exactly")

    normalized_topology: dict[str, tuple[str, str]] = {}
    for reference_id in reference_ids:
        normalized_topology[reference_id] = _topology_record(
            topology_by_reference[reference_id]
        )
    assignment = _assign_intent_pairs(
        [(reference_id, normalized_topology[reference_id][1]) for reference_id in reference_ids],
        seed,
    )

    jobs: list[dict[str, object]] = []
    for reference in references:
        reference_id = str(reference["reference_id"])
        if reference.get("source_split") != "train":
            raise ValueError(f"reference must come from train: {reference_id}")
        topology, mark_role = normalized_topology[reference_id]
        for intent in assignment[reference_id]:
            sample_id = f"h1a-{len(jobs) + 1:04d}"
            job: dict[str, object] = {
                "sample_id": sample_id,
                "reference_id": reference_id,
                "intent": intent,
                "output_state": INTENT_TO_STATE[intent],
                "topology": topology,
                "mark_role": mark_role,
                "has_marked_point": intent != "LOOKALIKE",
                "source_split": "train",
                "eligible_split": "train",
                "source_scene_id": str(reference.get("source_scene_id", "")),
                "source_reference_sha256": str(
                    reference.get("source_reference_sha256", "")
                ).upper(),
                "crop_path": str(reference.get("crop_path", "")),
            }
            prompt = build_hard_sample_prompt(job)
            job["prompt"] = prompt
            job["prompt_sha256"] = sha256(prompt.encode("utf-8")).hexdigest().upper()
            errors = validate_h1_record(job)
            if errors:
                raise ValueError(f"invalid H1 job {sample_id}: {', '.join(errors)}")
            jobs.append(job)

    counts = Counter(str(job["intent"]) for job in jobs)
    if counts != Counter(H1A_QUOTAS):
        raise RuntimeError(f"internal H1a quota mismatch: {dict(counts)}")
    return jobs
