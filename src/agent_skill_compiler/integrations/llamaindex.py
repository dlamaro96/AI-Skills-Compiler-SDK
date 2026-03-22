"""Callback handler helpers for LlamaIndex."""

from __future__ import annotations

from typing import Any

from agent_skill_compiler.integrations.generic import (
    SkillCompilerTracer,
    get_first_attr,
    normalize_tool_arguments,
    serialize_for_trace,
)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(serialize_for_trace(value))


def create_llamaindex_callback_handler(
    tracer: SkillCompilerTracer,
    *,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    default_agent_name: str = "LlamaIndex",
):
    """Create a callback handler compatible with LlamaIndex callback managers."""

    try:
        from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    except ImportError as exc:
        raise ImportError(
            "LlamaIndex support requires `llama-index` to be installed."
        ) from exc

    class _SkillCompilerLlamaIndexHandler(BaseCallbackHandler):
        def __init__(self) -> None:
            super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
            self._trace_run = None
            self._trace_id: str | None = None
            self._tool_events: dict[str, dict[str, str | None]] = {}
            self._finalized = False

        def _ensure_run(self, input_text: Any = "") -> Any:
            if self._trace_run is None:
                self._trace_run = tracer.trace(
                    task_name=task_name or self._trace_id or "llamaindex_run",
                    input_text=_normalize_text(input_text),
                    metadata={
                        "framework": "llamaindex",
                        "adapter": "agent_skill_compiler.integrations.llamaindex",
                        **(metadata or {}),
                    },
                    agent_name=default_agent_name,
                )
            return self._trace_run

        def start_trace(self, trace_id: str | None = None) -> None:
            self._trace_id = trace_id

        def end_trace(self, trace_id: str | None = None, trace_map: dict[str, list[str]] | None = None) -> None:
            del trace_id
            if self._trace_run is None or self._finalized:
                return
            self._finalized = True
            self._trace_run.finish(
                metadata={
                    "framework": "llamaindex",
                    "trace_map": serialize_for_trace(trace_map or {}),
                }
            )

        def on_event_start(
            self,
            event_type: Any,
            payload: dict[str, Any] | None = None,
            event_id: str = "",
            parent_id: str = "",
            **kwargs: Any,
        ) -> str:
            del parent_id, kwargs
            event_name = str(event_type)
            payload = payload or {}

            if "QUERY" in event_name.upper():
                self._ensure_run(input_text=get_first_attr(payload, "query_str", default=payload))

            if "TOOL" in event_name.upper():
                trace_run = self._ensure_run()
                tool_name = str(get_first_attr(payload, "tool_name", "name", default="tool"))
                tool_call = trace_run.tool_call(
                    action_name=tool_name,
                    arguments=normalize_tool_arguments(payload),
                    agent_name=default_agent_name,
                    semantic_name=tool_name,
                    tool_metadata={
                        "framework": "llamaindex",
                        "tool_name": tool_name,
                    },
                    tool_call_id=event_id or None,
                )
                self._tool_events[event_id] = {
                    "event_id": tool_call.event_id,
                    "action_name": tool_name,
                }

            if "LLM" in event_name.upper():
                trace_run = self._ensure_run()
                trace_run.message(
                    role="user",
                    content=serialize_for_trace(payload),
                    agent_name=default_agent_name,
                )

            return event_id

        def on_event_end(
            self,
            event_type: Any,
            payload: dict[str, Any] | None = None,
            event_id: str = "",
            **kwargs: Any,
        ) -> None:
            del kwargs
            if self._trace_run is None:
                return

            event_name = str(event_type)
            payload = payload or {}

            if "TOOL" in event_name.upper():
                tool_event = self._tool_events.get(event_id, {})
                action_name = tool_event.get("action_name") or str(
                    get_first_attr(payload, "tool_name", "name", default="tool")
                )
                self._trace_run.tool_result(
                    action_name=action_name,
                    result=payload,
                    agent_name=default_agent_name,
                    semantic_name=action_name,
                    parent_event_id=tool_event.get("event_id"),
                    tool_call_id=event_id or None,
                    tool_metadata={
                        "framework": "llamaindex",
                        "tool_name": action_name,
                    },
                )
                return

            if "LLM" in event_name.upper():
                self._trace_run.message(
                    role="assistant",
                    content=serialize_for_trace(payload),
                    agent_name=default_agent_name,
                )
                return

            if "QUERY" in event_name.upper():
                self._trace_run.final_output(output=payload, agent_name=default_agent_name)

    return _SkillCompilerLlamaIndexHandler()
