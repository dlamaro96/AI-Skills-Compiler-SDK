"""Request and response models used by the SDK."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agent_skill_compiler.models.domain import ActionKind, RunRecord, RunStatus, TraceEvent


class StartRunRequest(BaseModel):
    """Payload used to create a new run."""

    task_name: str
    input_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecordEventRequest(BaseModel):
    """Payload used to append a trace event to a run."""

    timestamp: datetime | None = None
    event_type: str = "trace"
    agent_name: str
    step_index: int | None = None
    action_name: str
    action_kind: ActionKind
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    tool_metadata: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str | None = None
    latency_ms: int = 0
    success: bool = True
    parent_event_id: str | None = None


class FinishRunRequest(BaseModel):
    """Payload used to complete a run."""

    status: RunStatus
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunDetailResponse(BaseModel):
    """Full run detail returned by the API."""

    run: RunRecord
    events: list[TraceEvent]
