"""Training readiness policy shared by detector training entrypoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingReadiness:
    images: int
    accepted: int
    rejected: int
    unreviewed: int
    minimum_precision: float = 0.80

    @property
    def reviewed_precision(self) -> float:
        reviewed = self.accepted + self.rejected
        return self.accepted / reviewed if reviewed else 0.0

    @property
    def reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.unreviewed:
            reasons.append(f"{self.unreviewed} annotations remain unreviewed")
        if self.accepted + self.rejected == 0:
            reasons.append("no reviewed annotations are available")
        elif self.reviewed_precision < self.minimum_precision:
            reasons.append(
                f"reviewed precision {self.reviewed_precision:.3f} is below {self.minimum_precision:.3f}"
            )
        if self.accepted == 0:
            reasons.append("no accepted annotations are available")
        return reasons

    @property
    def can_train(self) -> bool:
        return not self.reasons

    def to_dict(self) -> dict[str, object]:
        return {
            "images": self.images,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "unreviewed": self.unreviewed,
            "reviewed_precision": self.reviewed_precision,
            "minimum_precision": self.minimum_precision,
            "can_train": self.can_train,
            "reasons": self.reasons,
        }
