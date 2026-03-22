"""Lightweight models returned by the SDK and remote ASC backend."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class RunStatus(StrEnum):
    """Terminal status for a workflow run."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class EventType(StrEnum):
    """High-level event family."""

    EXECUTION = "execution"
    TRACE = "trace"


class ActionKind(StrEnum):
    """Normalized event action type."""

    ROUTE = "route"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_DECISION = "agent_decision"
    MESSAGE = "message"
    ERROR = "error"
    FINAL_OUTPUT = "final_output"


class RunRecord(BaseModel):
    """Structured metadata for one workflow execution."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str | None = None
    task_name: str
    input_text: str
    status: RunStatus
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    total_latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    """Single trace event emitted during agent execution."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    project_id: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)
    event_type: EventType = EventType.TRACE
    agent_name: str
    step_index: int
    action_name: str
    action_kind: ActionKind
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    tool_metadata: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str | None = None
    latency_ms: int = 0
    success: bool = True
    parent_event_id: str | None = None


class CandidateSkill(BaseModel):
    """Suggested skill derived from repeated execution patterns."""

    skill_id: str = Field(default_factory=lambda: str(uuid4()))
    suggested_name: str
    sequence_signature: str
    frequency: int
    success_rate: float
    avg_latency_ms: float
    involved_agents: list[str]
    natural_language_description: str
    score: float = 0.0


class AnalysisSummary(BaseModel):
    """Top-level output produced by the analysis pipeline."""

    total_runs: int
    successful_runs: int
    candidate_skills: list[CandidateSkill] = Field(default_factory=list)
