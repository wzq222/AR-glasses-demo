from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


Role = Literal["admin", "inspector", "reviewer"]
StepType = Literal["PHOTO", "QR", "FASTENER_MARK", "METER", "HUMAN_CONFIRM"]


class LoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=8, max_length=256)


class UserCreate(BaseModel):
    username: str = Field(pattern=r"^[A-Za-z0-9_.-]{2,64}$")
    display_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=12, max_length=256)
    role: Role = "inspector"


class StepDefinition(BaseModel):
    key: str = Field(pattern=r"^[A-Z0-9_]{2,64}$")
    type: StepType
    title: str = Field(min_length=1, max_length=120)
    instruction: str = Field(min_length=1, max_length=500)
    required: bool = True
    require_evidence: bool = True
    require_human_confirmation: bool = False
    config: dict[str, Any] = Field(default_factory=dict)


class TemplateCreate(BaseModel):
    code: str = Field(pattern=r"^[A-Z0-9_-]{2,64}$")
    title: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    steps: list[StepDefinition] = Field(min_length=1, max_length=100)

    @field_validator("steps")
    @classmethod
    def unique_steps(cls, value: list[StepDefinition]) -> list[StepDefinition]:
        keys = [step.key for step in value]
        if len(keys) != len(set(keys)):
            raise ValueError("step keys must be unique")
        return value


class AssignmentCreate(BaseModel):
    template_id: int
    assignee_id: int
    asset_code: str = Field(min_length=1, max_length=128)
    due_at: datetime | None = None


class RunCreate(BaseModel):
    assignment_id: int
    device: dict[str, Any] = Field(default_factory=dict)


class StepResultUpsert(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=128)
    status: Literal["succeeded", "uncertain", "failed", "skipped"]
    value: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0, le=1)
    requires_human_review: bool
    human_decision: str | None = Field(default=None, max_length=200)
    analyzer_version: str = Field(min_length=1, max_length=128)
    error_code: str | None = Field(default=None, max_length=64)
    captured_at: datetime


class ReviewRequest(BaseModel):
    decision: Literal["reviewed", "rejected"]
    note: str = Field(default="", max_length=1000)
