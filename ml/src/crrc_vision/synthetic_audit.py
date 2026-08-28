from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .synthetic_contract import SYNTHETIC_STATES, assert_formal_truth_unchanged


@dataclass(frozen=True)
class SyntheticAuditResult:
    passed: bool
    approved_total: int
    approved_by_state: dict[str, int]
    rejected_total: int
    uncertain_total: int
    errors: tuple[str, ...]


def audit_records(
    records: Iterable[Mapping[str, object]],
    *,
    minimum_per_state: int = 8,
    minimum_approval_rate: float = 0.75,
) -> SyntheticAuditResult:
    items = list(records)
    errors: list[str] = []
    approved_counts: Counter[str] = Counter()
    rejected_total = 0
    uncertain_total = 0
    seen_ids: set[str] = set()

    for index, record in enumerate(items):
        prefix = f"record[{index}]"
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in seen_ids:
            errors.append(f"{prefix}.sample_id missing or duplicated")
        seen_ids.add(sample_id)
        if record.get("synthetic") is not True:
            errors.append(f"{prefix}.synthetic must be true")
        if record.get("eligible_split") != "train":
            errors.append(f"{prefix}.eligible_split must be train")
        if record.get("source_split") != "train":
            errors.append(f"{prefix}.source_split must be train")
        state = str(record.get("state", ""))
        if state not in SYNTHETIC_STATES:
            errors.append(f"{prefix}.state is invalid")
        digest = str(record.get("source_reference_sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            errors.append(f"{prefix}.source_reference_sha256 is invalid")

        review = str(record.get("review_status", ""))
        if review == "APPROVED" and state in SYNTHETIC_STATES:
            approved_counts[state] += 1
        elif review == "REJECTED":
            rejected_total += 1
        elif review == "UNCERTAIN":
            uncertain_total += 1
        else:
            errors.append(f"{prefix}.review_status is invalid")

    approved_total = sum(approved_counts.values())
    for state in sorted(SYNTHETIC_STATES):
        if approved_counts[state] < minimum_per_state:
            errors.append(
                f"approved {state} {approved_counts[state]} < required {minimum_per_state}"
            )
    approval_rate = approved_total / len(items) if items else 0.0
    if approval_rate < minimum_approval_rate:
        errors.append(
            f"approval rate {approval_rate:.4f} < required {minimum_approval_rate:.4f}"
        )
    return SyntheticAuditResult(
        passed=not errors,
        approved_total=approved_total,
        approved_by_state={state: approved_counts[state] for state in sorted(SYNTHETIC_STATES)},
        rejected_total=rejected_total,
        uncertain_total=uncertain_total,
        errors=tuple(errors),
    )


def audit_manifest(
    records: Iterable[Mapping[str, object]],
    formal_truth: Path,
) -> SyntheticAuditResult:
    assert_formal_truth_unchanged(formal_truth)
    return audit_records(records)
