"""Callback helpers for Google ADK."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from agent_skill_compiler.integrations.generic import (
    SkillCompilerTracer,
    get_first_attr,
    normalize_tool_arguments,
    serialize_for_trace,
)

_TRACE_STATE_KEY = "__asc_trace_state__"


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(serialize_for_trace(value))


def _state(context: Any) -> dict[str, Any]:
    state = getattr(context, _TRACE_STATE_KEY, None)
    if state is None:
        state = {"tool_events": {}}
        setattr(context, _TRACE_STATE_KEY, state)
    return state


def _agent_name(context: Any, fallback: str) -> str:
    return str(get_first_attr(context, "agent_name", default=fallback))


@dataclass(frozen=True)
class GoogleADKCallbacks:
    """Container of ADK callback functions ready to attach to an agent."""

    before_agent: Callable[..., Any]
    after_agent: Callable[..., Any]
    before_model: Callable[..., Any]
    after_model: Callable[..., Any]
    before_tool: Callable[..., Any]
    after_tool: Callable[..., Any]


def create_google_adk_callbacks(
    tracer: SkillCompilerTracer,
    *,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    default_agent_name: str = "GoogleADK",
) -> GoogleADKCallbacks:
    """Create a callback bundle compatible with Google ADK agents."""

    def before_agent(callback_context: Any, *args: Any, **kwargs: Any) -> None:
        del kwargs
        state = _state(callback_context)
        if state.get("trace_run") is not None:
            return None
        agent_name = _agent_name(callback_context, default_agent_name)
        input_value = args[0] if args else ""
        state["trace_run"] = tracer.trace(
            task_name=task_name or agent_name,
            input_text=_normalize_text(input_value),
            metadata={
                "framework": "google-adk",
                "adapter": "agent_skill_compiler.integrations.google_adk",
                **(metadata or {}),
            },
            agent_name=agent_name,
        )
        state["agent_name"] = agent_name
        return None

    def after_agent(callback_context: Any, result: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        state = _state(callback_context)
        trace_run = state.get("trace_run")
        if trace_run is None:
            return None
        trace_run.final_output(output=result, agent_name=state.get("agent_name", default_agent_name))
        trace_run.finish(
            metadata={
                "framework": "google-adk",
                "response": serialize_for_trace(result),
            }
        )
        state["trace_run"] = None
        return None

    def before_model(callback_context: Any, llm_request: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        state = _state(callback_context)
        trace_run = state.get("trace_run")
        if trace_run is None:
            return None
        trace_run.message(
            role="user",
            content=serialize_for_trace(llm_request),
            agent_name=state.get("agent_name", default_agent_name),
        )
        return None

    def after_model(callback_context: Any, llm_response: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        state = _state(callback_context)
        trace_run = state.get("trace_run")
        if trace_run is None:
            return None
        trace_run.message(
            role="assistant",
            content=serialize_for_trace(llm_response),
            agent_name=state.get("agent_name", default_agent_name),
        )
        return None

    def before_tool(callback_context: Any, tool: Any, tool_args: Any | None = None, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        state = _state(callback_context)
        trace_run = state.get("trace_run")
        if trace_run is None:
            return None
        tool_name = str(get_first_attr(tool, "name", default="tool"))
        started_at = perf_counter()
        event = trace_run.tool_call(
            action_name=tool_name,
            arguments=normalize_tool_arguments(tool_args),
            agent_name=state.get("agent_name", default_agent_name),
            semantic_name=tool_name,
            tool_metadata={
                "framework": "google-adk",
                "tool_name": tool_name,
            },
        )
        state["tool_events"][tool_name] = {
            "event_id": event.event_id,
            "started_at": started_at,
        }
        return None

    def after_tool(callback_context: Any, tool: Any, result: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        state = _state(callback_context)
        trace_run = state.get("trace_run")
        if trace_run is None:
            return None
        tool_name = str(get_first_attr(tool, "name", default="tool"))
        tool_event = state["tool_events"].get(tool_name, {})
        started_at = float(tool_event.get("started_at") or perf_counter())
        trace_run.tool_result(
            action_name=tool_name,
            result=result,
            agent_name=state.get("agent_name", default_agent_name),
            latency_ms=int((perf_counter() - started_at) * 1000),
            parent_event_id=tool_event.get("event_id"),
            semantic_name=tool_name,
            tool_metadata={
                "framework": "google-adk",
                "tool_name": tool_name,
            },
        )
        return None

    return GoogleADKCallbacks(
        before_agent=before_agent,
        after_agent=after_agent,
        before_model=before_model,
        after_model=after_model,
        before_tool=before_tool,
        after_tool=after_tool,
    )
