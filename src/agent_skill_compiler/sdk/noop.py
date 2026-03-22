"""No-op SDK clients used when tracing is optional."""

from __future__ import annotations

from typing import Any

from agent_skill_compiler.models.domain import AnalysisSummary, RunRecord, RunStatus, TraceEvent


class NoopSkillCompilerClient:
    """Synchronous no-op client that safely disables tracing."""

    enabled = False

    def close(self) -> None:
        """Close the no-op client."""

    def start_run(self, task_name: str, input_text: str, metadata: dict[str, Any] | None = None) -> RunRecord:
        """Return a local placeholder run."""

        return RunRecord(
            task_name=task_name,
            input_text=input_text,
            status=RunStatus.RUNNING,
            metadata=metadata or {},
        )

    def record_event(
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
        """Return a local placeholder event."""

        del run_id, step_index
        return TraceEvent(
            run_id="noop-run",
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
            step_index=1,
        )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str = RunStatus.SUCCESS,
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord:
        """Return a local placeholder completed run."""

        del run_id
        return RunRecord(
            task_name="noop",
            input_text="",
            status=status,
            metadata=metadata or {},
        )

    def analyze(self) -> AnalysisSummary:
        """Return an empty analysis response."""

        return AnalysisSummary(total_runs=0, successful_runs=0, candidate_skills=[])

    def __enter__(self) -> "NoopSkillCompilerClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class NoopAsyncSkillCompilerClient:
    """Async no-op client that safely disables tracing."""

    enabled = False

    async def close(self) -> None:
        """Close the async no-op client."""

    async def start_run(self, task_name: str, input_text: str, metadata: dict[str, Any] | None = None) -> RunRecord:
        """Return a local placeholder run."""

        return RunRecord(
            task_name=task_name,
            input_text=input_text,
            status=RunStatus.RUNNING,
            metadata=metadata or {},
        )

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
        """Return a local placeholder event."""

        del run_id, step_index
        return TraceEvent(
            run_id="noop-run",
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
            step_index=1,
        )

    async def finish_run(
        self,
        run_id: str,
        *,
        status: str = RunStatus.SUCCESS,
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord:
        """Return a local placeholder completed run."""

        del run_id
        return RunRecord(
            task_name="noop",
            input_text="",
            status=status,
            metadata=metadata or {},
        )

    async def analyze(self) -> AnalysisSummary:
        """Return an empty analysis response."""

        return AnalysisSummary(total_runs=0, successful_runs=0, candidate_skills=[])

    async def __aenter__(self) -> "NoopAsyncSkillCompilerClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
