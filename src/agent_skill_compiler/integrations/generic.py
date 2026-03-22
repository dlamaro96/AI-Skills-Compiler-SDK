"""Framework-agnostic tracing helpers for custom agent runtimes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from time import perf_counter
from typing import Any

from agent_skill_compiler.models.domain import ActionKind, RunRecord, RunStatus, TraceEvent
from agent_skill_compiler.sdk.async_client import AsyncSkillCompilerClient
from agent_skill_compiler.sdk.client import SkillCompilerClient
from agent_skill_compiler.sdk.noop import NoopAsyncSkillCompilerClient, NoopSkillCompilerClient


def serialize_for_trace(value: Any) -> Any:
    """Serialize arbitrary Python values into JSON-safe structures."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): serialize_for_trace(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_for_trace(item) for item in value]
    if isinstance(value, BaseException):
        return {
            "error": str(value),
            "type": type(value).__name__,
        }
    if hasattr(value, "model_dump"):
        try:
            return serialize_for_trace(value.model_dump(mode="json"))
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return serialize_for_trace(value.dict())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return serialize_for_trace(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "as_dict"):
        try:
            return serialize_for_trace(value.as_dict())
        except Exception:
            pass
    if hasattr(value, "model_dump_json"):
        try:
            return serialize_for_trace(json.loads(value.model_dump_json()))
        except Exception:
            pass
    if hasattr(value, "json"):
        try:
            json_result = value.json()
            if isinstance(json_result, str):
                return serialize_for_trace(json.loads(json_result))
            return serialize_for_trace(json_result)
        except Exception:
            pass
    if is_dataclass(value):
        return serialize_for_trace(asdict(value))
    if hasattr(value, "__dict__"):
        try:
            return {
                str(key): serialize_for_trace(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        except Exception:
            pass
    return str(value)


def normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Normalize tool arguments into a consistent dictionary payload."""

    if arguments is None:
        return {}
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"raw": arguments}
        if isinstance(parsed, dict):
            return serialize_for_trace(parsed)
        return {"value": serialize_for_trace(parsed)}
    serialized = serialize_for_trace(arguments)
    if isinstance(serialized, dict):
        return serialized
    return {"value": serialized}


def get_first_attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first non-null attribute or dict key from a list of names."""

    for name in names:
        if isinstance(obj, dict) and name in obj and obj[name] is not None:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _normalize_payload(value: Any) -> dict[str, Any]:
    serialized = serialize_for_trace(value)
    if serialized is None:
        return {}
    if isinstance(serialized, dict):
        return serialized
    return {"value": serialized}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    for attr in ("text", "content", "output"):
        attr_value = getattr(value, attr, None)
        if isinstance(attr_value, str) and attr_value:
            return attr_value
    return str(serialize_for_trace(value))


def _annotate_payload(payload: dict[str, Any], *, semantic_name: str | None = None) -> dict[str, Any]:
    if not semantic_name:
        return payload
    enriched = dict(payload)
    enriched["_asc"] = {"semantic_name": semantic_name}
    return enriched


_TOOL_METADATA_FIELDS = (
    "tool_call_id",
    "tool_name",
    "semantic_name",
    "tool_args",
    "tool_call_error",
    "result",
    "metrics",
    "usage",
    "token_usage",
    "child_run_id",
    "created_at",
    "requires_confirmation",
    "confirmed",
    "confirmation_note",
    "requires_user_input",
    "user_input_schema",
    "answered",
    "external_execution_required",
)
_TOOL_METADATA_NESTED_FIELDS = (
    "tool",
    "result",
    "usage",
    "metrics",
    "token_usage",
    "response_metadata",
    "additional_kwargs",
)


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="python")
            if isinstance(dumped, Mapping):
                return {str(key): item for key, item in dumped.items()}
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            dumped = value.dict()
            if isinstance(dumped, Mapping):
                return {str(key): item for key, item in dumped.items()}
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
            if isinstance(dumped, Mapping):
                return {str(key): item for key, item in dumped.items()}
        except Exception:
            pass
    if hasattr(value, "as_dict"):
        try:
            dumped = value.as_dict()
            if isinstance(dumped, Mapping):
                return {str(key): item for key, item in dumped.items()}
        except Exception:
            pass
    if is_dataclass(value):
        dumped = asdict(value)
        if isinstance(dumped, Mapping):
            return {str(key): item for key, item in dumped.items()}
    if hasattr(value, "__dict__"):
        try:
            return {
                str(key): item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        except Exception:
            pass
    return None


def _extract_tool_metadata(value: Any, *, _depth: int = 0, _visited: set[int] | None = None) -> dict[str, Any]:
    if value is None or _depth > 4:
        return {}
    if isinstance(value, (str, int, float, bool)):
        return {}

    visited = _visited or set()
    value_id = id(value)
    if value_id in visited:
        return {}
    visited.add(value_id)

    metadata: dict[str, Any] = {}
    mapping = _as_mapping(value)
    if mapping is not None:
        for field in _TOOL_METADATA_FIELDS:
            field_value = mapping.get(field)
            if field_value is not None:
                metadata[field] = serialize_for_trace(field_value)

        if "tool_name" not in metadata:
            tool_name = mapping.get("name")
            if tool_name is not None:
                metadata["tool_name"] = serialize_for_trace(tool_name)

        for nested_field in _TOOL_METADATA_NESTED_FIELDS:
            nested_value = mapping.get(nested_field)
            if nested_value is None:
                continue
            nested_metadata = _extract_tool_metadata(
                nested_value,
                _depth=_depth + 1,
                _visited=visited,
            )
            for key, item in nested_metadata.items():
                metadata.setdefault(key, item)
        return metadata

    for field in _TOOL_METADATA_FIELDS:
        field_value = getattr(value, field, None)
        if field_value is not None:
            metadata[field] = serialize_for_trace(field_value)

    if "tool_name" not in metadata:
        tool_name = getattr(value, "name", None)
        if tool_name is not None:
            metadata["tool_name"] = serialize_for_trace(tool_name)

    for nested_field in _TOOL_METADATA_NESTED_FIELDS:
        nested_value = getattr(value, nested_field, None)
        if nested_value is None:
            continue
        nested_metadata = _extract_tool_metadata(
            nested_value,
            _depth=_depth + 1,
            _visited=visited,
        )
        for key, item in nested_metadata.items():
            metadata.setdefault(key, item)

    return metadata


def _resolved_tool_call_id(
    explicit_tool_call_id: str | None,
    extracted_metadata: dict[str, Any],
    fallback: str | None = None,
) -> str | None:
    if explicit_tool_call_id:
        return explicit_tool_call_id
    extracted_tool_call_id = extracted_metadata.get("tool_call_id")
    if isinstance(extracted_tool_call_id, str) and extracted_tool_call_id:
        return extracted_tool_call_id
    return fallback


def _build_tool_metadata(
    *,
    action_name: str,
    semantic_name: str | None = None,
    tool_metadata: dict[str, Any] | None = None,
    source: Any | None = None,
) -> dict[str, Any]:
    metadata = _extract_tool_metadata(source)
    explicit_metadata = serialize_for_trace(tool_metadata or {})
    if isinstance(explicit_metadata, dict):
        metadata.update(explicit_metadata)
    metadata.setdefault("tool_name", action_name)
    if semantic_name:
        metadata["semantic_name"] = semantic_name
    return metadata


class NoopTraceEvent:
    """Placeholder event returned when tracing is disabled."""

    def __init__(self, event_id: str | None = None) -> None:
        self.event_id = event_id


class NoopTracedRun:
    """Trace helper that safely ignores all calls."""

    def __init__(self) -> None:
        self.run = RunRecord(task_name="noop", input_text="", status=RunStatus.RUNNING)
        self.default_agent_name = "application"
        self.is_finished = False

    def event(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    def tool_call(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    def tool_result(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    def decision(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    def route(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    def message(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    def error(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    def final_output(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    def update_metadata(self, **kwargs: Any) -> None:
        del kwargs

    def finish(self, **kwargs: Any) -> RunRecord:
        del kwargs
        self.is_finished = True
        self.run.status = RunStatus.SUCCESS
        return self.run

    def fail(self, **kwargs: Any) -> RunRecord:
        del kwargs
        self.is_finished = True
        self.run.status = RunStatus.FAILED
        return self.run

    def time_tool_call(self, **kwargs: Any) -> "_NoopTimedToolCall":
        del kwargs
        return _NoopTimedToolCall()

    def __enter__(self) -> "NoopTracedRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb


class NoopAsyncTracedRun(NoopTracedRun):
    """Async trace helper that safely ignores all calls."""

    async def event(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    async def tool_call(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    async def tool_result(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    async def decision(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    async def route(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    async def message(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    async def error(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    async def final_output(self, **kwargs: Any) -> NoopTraceEvent:
        del kwargs
        return NoopTraceEvent()

    async def finish(self, **kwargs: Any) -> RunRecord:
        return super().finish(**kwargs)

    async def fail(self, **kwargs: Any) -> RunRecord:
        return super().fail(**kwargs)

    async def __aenter__(self) -> "NoopAsyncTracedRun":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb


class TracedRun:
    """Synchronous helper for recording a run and normalized events."""

    def __init__(
        self,
        client: SkillCompilerClient | NoopSkillCompilerClient,
        run: RunRecord,
        default_agent_name: str = "application",
    ) -> None:
        self.client = client
        self.run = run
        self.default_agent_name = default_agent_name
        self.is_finished = False
        self.pending_metadata: dict[str, Any] = {}

    @classmethod
    def start(
        cls,
        client: SkillCompilerClient | NoopSkillCompilerClient,
        *,
        task_name: str,
        input_text: str,
        metadata: dict[str, Any] | None = None,
        default_agent_name: str = "application",
    ) -> "TracedRun | NoopTracedRun":
        if isinstance(client, NoopSkillCompilerClient):
            return NoopTracedRun()
        run = client.start_run(task_name=task_name, input_text=input_text, metadata=metadata or {})
        return cls(client=client, run=run, default_agent_name=default_agent_name)

    def update_metadata(self, **metadata: Any) -> None:
        """Stage additional run metadata to be persisted at finish time."""

        self.pending_metadata.update(serialize_for_trace(metadata))

    def event(
        self,
        *,
        agent_name: str | None = None,
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
        return self.client.record_event(
            run_id=self.run.run_id,
            agent_name=agent_name or self.default_agent_name,
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

    def tool_call(
        self,
        *,
        action_name: str,
        arguments: Any | None = None,
        agent_name: str | None = None,
        parent_event_id: str | None = None,
        semantic_name: str | None = None,
        tool_metadata: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> TraceEvent:
        resolved_tool_metadata = _build_tool_metadata(
            action_name=action_name,
            semantic_name=semantic_name,
            tool_metadata=tool_metadata,
            source=arguments,
        )
        return self.event(
            agent_name=agent_name,
            action_name=action_name,
            action_kind=ActionKind.TOOL_CALL,
            input_payload=_annotate_payload(normalize_tool_arguments(arguments), semantic_name=semantic_name),
            tool_metadata=resolved_tool_metadata,
            tool_call_id=_resolved_tool_call_id(tool_call_id, resolved_tool_metadata),
            parent_event_id=parent_event_id,
        )

    def tool_result(
        self,
        *,
        action_name: str,
        result: Any | None = None,
        agent_name: str | None = None,
        latency_ms: int = 0,
        success: bool = True,
        parent_event_id: str | None = None,
        semantic_name: str | None = None,
        tool_metadata: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> TraceEvent:
        resolved_tool_metadata = _build_tool_metadata(
            action_name=action_name,
            semantic_name=semantic_name,
            tool_metadata=tool_metadata,
            source=result,
        )
        return self.event(
            agent_name=agent_name,
            action_name=action_name,
            action_kind=ActionKind.TOOL_RESULT,
            output_payload=_annotate_payload(_normalize_payload(result), semantic_name=semantic_name),
            tool_metadata=resolved_tool_metadata,
            tool_call_id=_resolved_tool_call_id(tool_call_id, resolved_tool_metadata, parent_event_id),
            latency_ms=latency_ms,
            success=success,
            parent_event_id=parent_event_id,
        )

    def decision(
        self,
        *,
        action_name: str,
        decision: Any | None = None,
        reason: str | None = None,
        agent_name: str | None = None,
        parent_event_id: str | None = None,
    ) -> TraceEvent:
        input_payload = {"reason": reason} if reason else {}
        return self.event(
            agent_name=agent_name,
            action_name=action_name,
            action_kind=ActionKind.AGENT_DECISION,
            input_payload=input_payload,
            output_payload=_normalize_payload(decision),
            parent_event_id=parent_event_id,
        )

    def route(
        self,
        *,
        destination_agent: str,
        agent_name: str | None = None,
        reason: str | None = None,
        parent_event_id: str | None = None,
    ) -> TraceEvent:
        return self.event(
            agent_name=agent_name,
            action_name=destination_agent,
            action_kind=ActionKind.ROUTE,
            input_payload={"reason": reason} if reason else {},
            output_payload={"destination_agent": destination_agent},
            parent_event_id=parent_event_id,
        )

    def message(
        self,
        *,
        content: Any,
        role: str = "assistant",
        agent_name: str | None = None,
        parent_event_id: str | None = None,
    ) -> TraceEvent:
        return self.event(
            agent_name=agent_name,
            action_name=role,
            action_kind=ActionKind.MESSAGE,
            output_payload={"role": role, "content": _normalize_text(content)},
            parent_event_id=parent_event_id,
        )

    def error(
        self,
        *,
        error: Any,
        agent_name: str | None = None,
        parent_event_id: str | None = None,
    ) -> TraceEvent:
        return self.event(
            agent_name=agent_name,
            action_name="error",
            action_kind=ActionKind.ERROR,
            output_payload=_normalize_payload(error),
            success=False,
            parent_event_id=parent_event_id,
        )

    def final_output(
        self,
        *,
        output: Any,
        agent_name: str | None = None,
        latency_ms: int = 0,
        parent_event_id: str | None = None,
    ) -> TraceEvent:
        return self.event(
            agent_name=agent_name,
            action_name="final_output",
            action_kind=ActionKind.FINAL_OUTPUT,
            output_payload={
                "response": _normalize_text(output),
                "output": serialize_for_trace(output),
            },
            latency_ms=latency_ms,
            parent_event_id=parent_event_id,
        )

    def finish(self, *, metadata: dict[str, Any] | None = None, status: str = RunStatus.SUCCESS) -> RunRecord:
        self.is_finished = True
        merged_metadata = dict(self.pending_metadata)
        if metadata:
            merged_metadata.update(serialize_for_trace(metadata))
        self.run = self.client.finish_run(run_id=self.run.run_id, status=status, metadata=merged_metadata)
        return self.run

    def fail(self, *, error: Any | None = None, metadata: dict[str, Any] | None = None) -> RunRecord:
        if error is not None:
            self.error(error=error)
        merged_metadata = {"error": _normalize_text(error)} if error is not None else {}
        if metadata:
            merged_metadata.update(metadata)
        return self.finish(metadata=merged_metadata, status=RunStatus.FAILED)

    def time_tool_call(
        self,
        *,
        action_name: str,
        arguments: Any | None = None,
        agent_name: str | None = None,
        semantic_name: str | None = None,
        tool_metadata: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> "_TimedToolCall":
        return _TimedToolCall(
            self,
            action_name=action_name,
            arguments=arguments,
            agent_name=agent_name,
            semantic_name=semantic_name,
            tool_metadata=tool_metadata,
            tool_call_id=tool_call_id,
        )

    def __enter__(self) -> "TracedRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.is_finished:
            return
        if exc is not None:
            self.fail(error=exc)
        else:
            self.finish()


class AsyncTracedRun:
    """Async helper for recording a run and normalized events."""

    def __init__(
        self,
        client: AsyncSkillCompilerClient | NoopAsyncSkillCompilerClient,
        run: RunRecord,
        default_agent_name: str = "application",
    ) -> None:
        self.client = client
        self.run = run
        self.default_agent_name = default_agent_name
        self.is_finished = False
        self.pending_metadata: dict[str, Any] = {}

    @classmethod
    async def start(
        cls,
        client: AsyncSkillCompilerClient | NoopAsyncSkillCompilerClient,
        *,
        task_name: str,
        input_text: str,
        metadata: dict[str, Any] | None = None,
        default_agent_name: str = "application",
    ) -> "AsyncTracedRun | NoopAsyncTracedRun":
        if isinstance(client, NoopAsyncSkillCompilerClient):
            return NoopAsyncTracedRun()
        run = await client.start_run(task_name=task_name, input_text=input_text, metadata=metadata or {})
        return cls(client=client, run=run, default_agent_name=default_agent_name)

    def update_metadata(self, **metadata: Any) -> None:
        """Stage additional run metadata to be persisted at finish time."""

        self.pending_metadata.update(serialize_for_trace(metadata))

    async def event(
        self,
        *,
        agent_name: str | None = None,
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
        return await self.client.record_event(
            run_id=self.run.run_id,
            agent_name=agent_name or self.default_agent_name,
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

    async def tool_call(
        self,
        *,
        action_name: str,
        arguments: Any | None = None,
        agent_name: str | None = None,
        parent_event_id: str | None = None,
        semantic_name: str | None = None,
        tool_metadata: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> TraceEvent:
        resolved_tool_metadata = _build_tool_metadata(
            action_name=action_name,
            semantic_name=semantic_name,
            tool_metadata=tool_metadata,
            source=arguments,
        )
        return await self.event(
            agent_name=agent_name,
            action_name=action_name,
            action_kind=ActionKind.TOOL_CALL,
            input_payload=_annotate_payload(normalize_tool_arguments(arguments), semantic_name=semantic_name),
            tool_metadata=resolved_tool_metadata,
            tool_call_id=_resolved_tool_call_id(tool_call_id, resolved_tool_metadata),
            parent_event_id=parent_event_id,
        )

    async def tool_result(
        self,
        *,
        action_name: str,
        result: Any | None = None,
        agent_name: str | None = None,
        latency_ms: int = 0,
        success: bool = True,
        parent_event_id: str | None = None,
        semantic_name: str | None = None,
        tool_metadata: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> TraceEvent:
        resolved_tool_metadata = _build_tool_metadata(
            action_name=action_name,
            semantic_name=semantic_name,
            tool_metadata=tool_metadata,
            source=result,
        )
        return await self.event(
            agent_name=agent_name,
            action_name=action_name,
            action_kind=ActionKind.TOOL_RESULT,
            output_payload=_annotate_payload(_normalize_payload(result), semantic_name=semantic_name),
            tool_metadata=resolved_tool_metadata,
            tool_call_id=_resolved_tool_call_id(tool_call_id, resolved_tool_metadata, parent_event_id),
            latency_ms=latency_ms,
            success=success,
            parent_event_id=parent_event_id,
        )

    async def decision(
        self,
        *,
        action_name: str,
        decision: Any | None = None,
        reason: str | None = None,
        agent_name: str | None = None,
        parent_event_id: str | None = None,
    ) -> TraceEvent:
        input_payload = {"reason": reason} if reason else {}
        return await self.event(
            agent_name=agent_name,
            action_name=action_name,
            action_kind=ActionKind.AGENT_DECISION,
            input_payload=input_payload,
            output_payload=_normalize_payload(decision),
            parent_event_id=parent_event_id,
        )

    async def route(
        self,
        *,
        destination_agent: str,
        agent_name: str | None = None,
        reason: str | None = None,
        parent_event_id: str | None = None,
    ) -> TraceEvent:
        return await self.event(
            agent_name=agent_name,
            action_name=destination_agent,
            action_kind=ActionKind.ROUTE,
            input_payload={"reason": reason} if reason else {},
            output_payload={"destination_agent": destination_agent},
            parent_event_id=parent_event_id,
        )

    async def message(
        self,
        *,
        content: Any,
        role: str = "assistant",
        agent_name: str | None = None,
        parent_event_id: str | None = None,
    ) -> TraceEvent:
        return await self.event(
            agent_name=agent_name,
            action_name=role,
            action_kind=ActionKind.MESSAGE,
            output_payload={"role": role, "content": _normalize_text(content)},
            parent_event_id=parent_event_id,
        )

    async def error(
        self,
        *,
        error: Any,
        agent_name: str | None = None,
        parent_event_id: str | None = None,
    ) -> TraceEvent:
        return await self.event(
            agent_name=agent_name,
            action_name="error",
            action_kind=ActionKind.ERROR,
            output_payload=_normalize_payload(error),
            success=False,
            parent_event_id=parent_event_id,
        )

    async def final_output(
        self,
        *,
        output: Any,
        agent_name: str | None = None,
        latency_ms: int = 0,
        parent_event_id: str | None = None,
    ) -> TraceEvent:
        return await self.event(
            agent_name=agent_name,
            action_name="final_output",
            action_kind=ActionKind.FINAL_OUTPUT,
            output_payload={
                "response": _normalize_text(output),
                "output": serialize_for_trace(output),
            },
            latency_ms=latency_ms,
            parent_event_id=parent_event_id,
        )

    async def finish(self, *, metadata: dict[str, Any] | None = None, status: str = RunStatus.SUCCESS) -> RunRecord:
        self.is_finished = True
        merged_metadata = dict(self.pending_metadata)
        if metadata:
            merged_metadata.update(serialize_for_trace(metadata))
        self.run = await self.client.finish_run(run_id=self.run.run_id, status=status, metadata=merged_metadata)
        return self.run

    async def fail(self, *, error: Any | None = None, metadata: dict[str, Any] | None = None) -> RunRecord:
        if error is not None:
            await self.error(error=error)
        merged_metadata = {"error": _normalize_text(error)} if error is not None else {}
        if metadata:
            merged_metadata.update(metadata)
        return await self.finish(metadata=merged_metadata, status=RunStatus.FAILED)

    async def __aenter__(self) -> "AsyncTracedRun":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.is_finished:
            return
        if exc is not None:
            await self.fail(error=exc)
        else:
            await self.finish()


class SkillCompilerTracer:
    """High-level synchronous tracer with env bootstrap and default metadata."""

    def __init__(
        self,
        client: SkillCompilerClient | NoopSkillCompilerClient,
        *,
        default_agent_name: str = "application",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.default_agent_name = default_agent_name
        self.metadata = metadata or {}

    @classmethod
    def from_env(
        cls,
        *,
        optional: bool = True,
        timeout: float = 30.0,
        default_agent_name: str = "application",
        metadata: dict[str, Any] | None = None,
    ) -> "SkillCompilerTracer":
        """Build a tracer from environment variables."""

        client = SkillCompilerClient.from_env(optional=optional, timeout=timeout)
        return cls(client=client, default_agent_name=default_agent_name, metadata=metadata)

    def trace(
        self,
        *,
        task_name: str,
        input_text: str,
        metadata: dict[str, Any] | None = None,
        agent_name: str | None = None,
    ) -> TracedRun | NoopTracedRun:
        """Start a run with shared default metadata."""

        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(serialize_for_trace(metadata))
        return trace_run(
            self.client,
            task_name=task_name,
            input_text=input_text,
            metadata=merged_metadata,
            default_agent_name=agent_name or self.default_agent_name,
        )


class AsyncSkillCompilerTracer:
    """High-level async tracer with env bootstrap and default metadata."""

    def __init__(
        self,
        client: AsyncSkillCompilerClient | NoopAsyncSkillCompilerClient,
        *,
        default_agent_name: str = "application",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.default_agent_name = default_agent_name
        self.metadata = metadata or {}

    @classmethod
    def from_env(
        cls,
        *,
        optional: bool = True,
        timeout: float = 30.0,
        default_agent_name: str = "application",
        metadata: dict[str, Any] | None = None,
    ) -> "AsyncSkillCompilerTracer":
        """Build an async tracer from environment variables."""

        client = AsyncSkillCompilerClient.from_env(optional=optional, timeout=timeout)
        return cls(client=client, default_agent_name=default_agent_name, metadata=metadata)

    async def trace(
        self,
        *,
        task_name: str,
        input_text: str,
        metadata: dict[str, Any] | None = None,
        agent_name: str | None = None,
    ) -> AsyncTracedRun | NoopAsyncTracedRun:
        """Start a run with shared default metadata."""

        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(serialize_for_trace(metadata))
        return await trace_run_async(
            self.client,
            task_name=task_name,
            input_text=input_text,
            metadata=merged_metadata,
            default_agent_name=agent_name or self.default_agent_name,
        )


class _TimedToolCall:
    """Convenience timer for sync tool call instrumentation."""

    def __init__(
        self,
        traced_run: TracedRun,
        *,
        action_name: str,
        arguments: Any | None,
        agent_name: str | None,
        semantic_name: str | None,
        tool_metadata: dict[str, Any] | None,
        tool_call_id: str | None,
    ) -> None:
        self.traced_run = traced_run
        self.action_name = action_name
        self.arguments = arguments
        self.agent_name = agent_name
        self.semantic_name = semantic_name
        self.tool_metadata = tool_metadata
        self.tool_call_id = tool_call_id
        self.started_at: float | None = None
        self.tool_call_event: TraceEvent | None = None

    def __enter__(self) -> "_TimedToolCall":
        self.started_at = perf_counter()
        self.tool_call_event = self.traced_run.tool_call(
            action_name=self.action_name,
            arguments=self.arguments,
            agent_name=self.agent_name,
            semantic_name=self.semantic_name,
            tool_metadata=self.tool_metadata,
            tool_call_id=self.tool_call_id,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, tb
        if self.started_at is None or self.tool_call_event is None:
            return
        latency_ms = int((perf_counter() - self.started_at) * 1000)
        self.traced_run.tool_result(
            action_name=self.action_name,
            result=exc if exc is not None else {"status": "ok"},
            agent_name=self.agent_name,
            latency_ms=latency_ms,
            success=exc is None,
            parent_event_id=self.tool_call_event.event_id,
            semantic_name=self.semantic_name,
            tool_metadata=self.tool_metadata,
            tool_call_id=self.tool_call_event.tool_call_id or self.tool_call_event.event_id,
        )


class _NoopTimedToolCall:
    """Convenience timer for disabled tracing."""

    def __enter__(self) -> "_NoopTimedToolCall":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb


def trace_run(
    client: SkillCompilerClient | NoopSkillCompilerClient | None,
    *,
    task_name: str,
    input_text: str,
    metadata: dict[str, Any] | None = None,
    default_agent_name: str = "application",
) -> TracedRun | NoopTracedRun:
    """Start and return a synchronous traced run helper."""

    if client is None:
        return NoopTracedRun()
    return TracedRun.start(
        client,
        task_name=task_name,
        input_text=input_text,
        metadata=metadata,
        default_agent_name=default_agent_name,
    )


async def trace_run_async(
    client: AsyncSkillCompilerClient | NoopAsyncSkillCompilerClient | None,
    *,
    task_name: str,
    input_text: str,
    metadata: dict[str, Any] | None = None,
    default_agent_name: str = "application",
) -> AsyncTracedRun | NoopAsyncTracedRun:
    """Start and return an async traced run helper."""

    if client is None:
        return NoopAsyncTracedRun()
    return await AsyncTracedRun.start(
        client,
        task_name=task_name,
        input_text=input_text,
        metadata=metadata,
        default_agent_name=default_agent_name,
    )
