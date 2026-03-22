"""Event listener helpers for CrewAI."""

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


def _event_key(source: Any, event: Any) -> str:
    for value in (
        get_first_attr(event, "execution_id", "run_id", "crew_id", "flow_id", "task_id"),
        get_first_attr(source, "id"),
    ):
        if value:
            return str(value)
    return str(id(source))


def create_crewai_event_listener(
    tracer: SkillCompilerTracer,
    *,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    default_agent_name: str = "CrewAI",
):
    """Create a CrewAI event listener that forwards runs into Skill Compiler."""

    try:
        from crewai.events import (
            AgentExecutionCompletedEvent,
            AgentExecutionErrorEvent,
            BaseEventListener,
            CrewKickoffCompletedEvent,
            CrewKickoffFailedEvent,
            CrewKickoffStartedEvent,
            ToolUsageErrorEvent,
            ToolUsageFinishedEvent,
            ToolUsageStartedEvent,
        )
    except ImportError as exc:
        raise ImportError(
            "CrewAI support requires `crewai` to be installed."
        ) from exc

    class _SkillCompilerCrewAIListener(BaseEventListener):
        def __init__(self) -> None:
            self._runs: dict[str, Any] = {}
            self._tool_events: dict[str, dict[str, dict[str, str | None]]] = {}
            super().__init__()

        def setup_listeners(self, crewai_event_bus) -> None:
            @crewai_event_bus.on(CrewKickoffStartedEvent)
            def _on_started(source: Any, event: Any) -> None:
                key = _event_key(source, event)
                self._runs[key] = tracer.trace(
                    task_name=task_name or get_first_attr(event, "crew_name", default="crewai_run"),
                    input_text=_normalize_text(get_first_attr(event, "inputs", default="")),
                    metadata={
                        "framework": "crewai",
                        "adapter": "agent_skill_compiler.integrations.crewai",
                        **(metadata or {}),
                    },
                    agent_name=default_agent_name,
                )
                self._tool_events[key] = {}

            @crewai_event_bus.on(CrewKickoffCompletedEvent)
            def _on_completed(source: Any, event: Any) -> None:
                key = _event_key(source, event)
                trace_run = self._runs.get(key)
                if trace_run is None:
                    return
                trace_run.final_output(output=get_first_attr(event, "output"), agent_name=default_agent_name)
                trace_run.finish(
                    metadata={
                        "framework": "crewai",
                        "response": serialize_for_trace(get_first_attr(event, "output")),
                    }
                )

            @crewai_event_bus.on(CrewKickoffFailedEvent)
            def _on_failed(source: Any, event: Any) -> None:
                key = _event_key(source, event)
                trace_run = self._runs.get(key)
                if trace_run is None:
                    return
                trace_run.fail(error=get_first_attr(event, "error"))

            @crewai_event_bus.on(ToolUsageStartedEvent)
            def _on_tool_started(source: Any, event: Any) -> None:
                key = _event_key(source, event)
                trace_run = self._runs.get(key)
                if trace_run is None:
                    return
                tool_name = str(
                    get_first_attr(
                        event,
                        "tool_name",
                        default=get_first_attr(get_first_attr(event, "tool"), "name", default="tool"),
                    )
                )
                tool_call = trace_run.tool_call(
                    action_name=tool_name,
                    arguments=normalize_tool_arguments(get_first_attr(event, "arguments", default={})),
                    agent_name=default_agent_name,
                    semantic_name=tool_name,
                    tool_metadata={
                        "framework": "crewai",
                        "tool_name": tool_name,
                    },
                )
                self._tool_events[key][tool_name] = {
                    "event_id": tool_call.event_id,
                    "action_name": tool_name,
                }

            @crewai_event_bus.on(ToolUsageFinishedEvent)
            def _on_tool_finished(source: Any, event: Any) -> None:
                key = _event_key(source, event)
                trace_run = self._runs.get(key)
                if trace_run is None:
                    return
                tool_name = str(get_first_attr(event, "tool_name", default="tool"))
                tool_event = self._tool_events.get(key, {}).get(tool_name, {})
                trace_run.tool_result(
                    action_name=tool_event.get("action_name") or tool_name,
                    result=get_first_attr(event, "result"),
                    agent_name=default_agent_name,
                    semantic_name=tool_name,
                    parent_event_id=tool_event.get("event_id"),
                    tool_metadata={
                        "framework": "crewai",
                        "tool_name": tool_name,
                    },
                )

            @crewai_event_bus.on(ToolUsageErrorEvent)
            def _on_tool_error(source: Any, event: Any) -> None:
                key = _event_key(source, event)
                trace_run = self._runs.get(key)
                if trace_run is None:
                    return
                tool_name = str(get_first_attr(event, "tool_name", default="tool"))
                tool_event = self._tool_events.get(key, {}).get(tool_name, {})
                trace_run.tool_result(
                    action_name=tool_event.get("action_name") or tool_name,
                    result=get_first_attr(event, "error"),
                    agent_name=default_agent_name,
                    semantic_name=tool_name,
                    parent_event_id=tool_event.get("event_id"),
                    success=False,
                    tool_metadata={
                        "framework": "crewai",
                        "tool_name": tool_name,
                    },
                )

            @crewai_event_bus.on(AgentExecutionCompletedEvent)
            def _on_agent_completed(source: Any, event: Any) -> None:
                key = _event_key(source, event)
                trace_run = self._runs.get(key)
                if trace_run is None:
                    return
                trace_run.message(
                    role="assistant",
                    content=get_first_attr(event, "output"),
                    agent_name=str(get_first_attr(get_first_attr(event, "agent"), "role", default=default_agent_name)),
                )

            @crewai_event_bus.on(AgentExecutionErrorEvent)
            def _on_agent_error(source: Any, event: Any) -> None:
                key = _event_key(source, event)
                trace_run = self._runs.get(key)
                if trace_run is None:
                    return
                trace_run.error(
                    error=get_first_attr(event, "error"),
                    agent_name=str(get_first_attr(get_first_attr(event, "agent"), "role", default=default_agent_name)),
                )

    return _SkillCompilerCrewAIListener()
