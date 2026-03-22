"""Async HTTP client used by framework adapters and async applications."""

from __future__ import annotations

from typing import Any

import httpx

from agent_skill_compiler.models.api import FinishRunRequest, RecordEventRequest, StartRunRequest
from agent_skill_compiler.models.domain import AnalysisSummary, RunRecord, RunStatus, TraceEvent
from agent_skill_compiler.sdk.config import SkillCompilerConnection
from agent_skill_compiler.sdk.noop import NoopAsyncSkillCompilerClient


class AsyncSkillCompilerClient:
    """Async API client for recording runs from async agent runtimes."""

    def __init__(self, base_url: str, public_key: str, secret_key: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.enabled = True
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "X-ASC-Public-Key": public_key,
                "X-ASC-Secret-Key": secret_key,
            },
        )

    @classmethod
    def from_env(
        cls,
        *,
        timeout: float = 30.0,
        optional: bool = False,
    ) -> "AsyncSkillCompilerClient | NoopAsyncSkillCompilerClient":
        """Create an async client from environment variables."""

        connection = SkillCompilerConnection.from_env()
        if connection.is_configured:
            return cls(
                base_url=connection.base_url,
                public_key=connection.public_key,
                secret_key=connection.secret_key,
                timeout=timeout,
            )
        if optional:
            return NoopAsyncSkillCompilerClient()
        missing = [
            name
            for name, value in (
                ("ASC_BASE_URL / SKILL_COMPILER_HOST", connection.base_url),
                ("ASC_PUBLIC_KEY / SKILL_COMPILER_PUBLIC_KEY", connection.public_key),
                ("ASC_SECRET_KEY / SKILL_COMPILER_SECRET_KEY", connection.secret_key),
            )
            if not value
        ]
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing Skill Compiler environment configuration: {missing_text}.")

    async def close(self) -> None:
        """Close the underlying HTTP client."""

        await self._client.aclose()

    async def start_run(self, task_name: str, input_text: str, metadata: dict[str, Any] | None = None) -> RunRecord:
        """Start a run on the remote service."""

        payload = StartRunRequest(task_name=task_name, input_text=input_text, metadata=metadata or {})
        response = await self._client.post("/api/runs/start", json=payload.model_dump(mode="json"))
        response.raise_for_status()
        return RunRecord.model_validate(response.json())

    async def record_event(
        self,
        *,
        run_id: str,
        agent_name: str,
        action_name: str,
        action_kind: str,
        input_payload: dict[str, Any] | None = None,
        output_payload: dict[str, Any] | None = None,
        tool_metadata: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        latency_ms: int = 0,
        success: bool = True,
        parent_event_id: str | None = None,
        step_index: int | None = None,
    ) -> TraceEvent:
        """Record one event against a run."""

        payload = RecordEventRequest(
            agent_name=agent_name,
            action_name=action_name,
            action_kind=action_kind,
            input_payload=input_payload or {},
            output_payload=output_payload or {},
            tool_metadata=tool_metadata or {},
            tool_call_id=tool_call_id,
            latency_ms=latency_ms,
            success=success,
            parent_event_id=parent_event_id,
            step_index=step_index,
        )
        response = await self._client.post(f"/api/runs/{run_id}/events", json=payload.model_dump(mode="json"))
        response.raise_for_status()
        return TraceEvent.model_validate(response.json())

    async def finish_run(
        self,
        run_id: str,
        *,
        status: str = RunStatus.SUCCESS,
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord:
        """Finish a run on the remote service."""

        payload = FinishRunRequest(status=status, metadata=metadata or {})
        response = await self._client.post(f"/api/runs/{run_id}/finish", json=payload.model_dump(mode="json"))
        response.raise_for_status()
        return RunRecord.model_validate(response.json())

    async def analyze(self) -> AnalysisSummary:
        """Trigger analysis and fetch the latest skill summary."""

        response = await self._client.post("/api/analyze")
        response.raise_for_status()
        return AnalysisSummary.model_validate(response.json())

    async def __aenter__(self) -> "AsyncSkillCompilerClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
