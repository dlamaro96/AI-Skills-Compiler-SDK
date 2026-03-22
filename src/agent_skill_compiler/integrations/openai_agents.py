"""Helpers for tracing OpenAI Agents SDK runs."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from agent_skill_compiler.integrations.generic import (
    AsyncSkillCompilerTracer,
    get_first_attr,
    normalize_tool_arguments,
    serialize_for_trace,
)

_TOOL_CALLED_EVENTS = {"tool_called", "tool_search_called"}
_TOOL_OUTPUT_EVENTS = {"tool_output", "tool_search_output_created"}


def _agent_name(agent: Any) -> str:
    return get_first_attr(agent, "name", "agent_name", default="OpenAIAgent")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    for attr in ("text", "content", "output", "final_output"):
        attr_value = getattr(value, attr, None)
        if isinstance(attr_value, str) and attr_value:
            return attr_value
    return str(serialize_for_trace(value))


def _tool_source(value: Any) -> Any:
    return get_first_attr(value, "raw_item", "tool", default=value)


def _tool_name(value: Any) -> str:
    source = _tool_source(value)
    return str(
        get_first_attr(
            source,
            "name",
            "tool_name",
            "action_name",
            default="tool",
        )
    )


def _tool_call_id(value: Any) -> str | None:
    source = _tool_source(value)
    tool_call_id = get_first_attr(
        value,
        "tool_call_id",
        "call_id",
        "id",
    )
    if tool_call_id is None:
        tool_call_id = get_first_attr(source, "tool_call_id", "call_id", "id")
    return str(tool_call_id) if tool_call_id else None


def _tool_arguments(value: Any) -> dict[str, Any]:
    source = _tool_source(value)
    arguments = get_first_attr(
        source,
        "arguments",
        "tool_args",
        "input",
        "input_payload",
        default={},
    )
    return normalize_tool_arguments(arguments)


def _tool_output(value: Any) -> Any:
    return get_first_attr(
        value,
        "output",
        "result",
        default=value,
    )


def _message_text(value: Any) -> str:
    text = get_first_attr(value, "text", "content")
    if isinstance(text, str) and text:
        return text
    parts = get_first_attr(value, "parts")
    if isinstance(parts, list):
        collected: list[str] = []
        for part in parts:
            part_text = get_first_attr(part, "text")
            if isinstance(part_text, str) and part_text:
                collected.append(part_text)
        if collected:
            return "".join(collected)
    return _normalize_text(value)


class TracedOpenAIAgentsStream:
    """Proxy around a streamed OpenAI Agents run that traces as events are consumed."""

    def __init__(
        self,
        result: Any,
        tracer: AsyncSkillCompilerTracer,
        *,
        agent: Any,
        input: Any,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        default_agent_name: str | None = None,
    ) -> None:
        self._result = result
        self._tracer = tracer
        self._input = input
        self._task_name = task_name
        self._metadata = metadata or {}
        self._agent_name = default_agent_name or _agent_name(agent)
        self._trace_run: Any | None = None
        self._tool_events: dict[str, dict[str, str | None]] = {}
        self._response_parts: list[str] = []
        self._closed = False

    async def _ensure_run(self) -> Any:
        if self._trace_run is None:
            self._trace_run = await self._tracer.trace(
                task_name=self._task_name or self._agent_name,
                input_text=_normalize_text(self._input),
                metadata={
                    "framework": "openai-agents",
                    "adapter": "agent_skill_compiler.integrations.openai_agents",
                    **self._metadata,
                },
                agent_name=self._agent_name,
            )
        return self._trace_run

    async def _handle_event(self, event: Any) -> None:
        trace_run = await self._ensure_run()
        event_type = getattr(event, "type", None)

        if event_type == "raw_response_event":
            delta = get_first_attr(getattr(event, "data", None), "delta")
            if isinstance(delta, str) and delta:
                self._response_parts.append(delta)
            return

        if event_type == "agent_updated_stream_event":
            new_agent = getattr(event, "new_agent", None)
            new_agent_name = _agent_name(new_agent)
            if new_agent_name != self._agent_name:
                await trace_run.route(
                    destination_agent=new_agent_name,
                    agent_name=self._agent_name,
                    reason="OpenAI Agents handoff",
                )
                self._agent_name = new_agent_name
            return

        if event_type != "run_item_stream_event":
            return

        event_name = getattr(event, "name", None)
        item = getattr(event, "item", None)

        if event_name in _TOOL_CALLED_EVENTS:
            started_at = perf_counter()
            action_name = _tool_name(item)
            tool_call_id = _tool_call_id(item)
            tool_call_event = await trace_run.tool_call(
                action_name=action_name,
                arguments=_tool_arguments(item),
                agent_name=self._agent_name,
                semantic_name=action_name,
                tool_metadata={
                    "framework": "openai-agents",
                    "tool_name": action_name,
                    "source": serialize_for_trace(item),
                },
                tool_call_id=tool_call_id,
            )
            if tool_call_id:
                self._tool_events[tool_call_id] = {
                    "event_id": tool_call_event.event_id,
                    "action_name": action_name,
                    "started_at": str(started_at),
                }
            return

        if event_name in _TOOL_OUTPUT_EVENTS:
            action_name = _tool_name(item)
            tool_call_id = _tool_call_id(item)
            tool_event = self._tool_events.get(tool_call_id or "")
            started_at = float((tool_event or {}).get("started_at") or perf_counter())
            latency_ms = int((perf_counter() - started_at) * 1000)
            await trace_run.tool_result(
                action_name=(tool_event or {}).get("action_name") or action_name,
                result=_tool_output(item),
                agent_name=self._agent_name,
                latency_ms=latency_ms,
                parent_event_id=(tool_event or {}).get("event_id"),
                semantic_name=(tool_event or {}).get("action_name") or action_name,
                tool_metadata={
                    "framework": "openai-agents",
                    "tool_name": action_name,
                    "source": serialize_for_trace(item),
                },
                tool_call_id=tool_call_id,
            )
            return

        if event_name == "message_output_created":
            text = _message_text(item)
            if text:
                self._response_parts.append(text)
            return

        if event_name in {"handoff_requested", "handoff_occured"}:
            destination = _normalize_text(item) or "handoff"
            await trace_run.route(
                destination_agent=destination,
                agent_name=self._agent_name,
                reason="OpenAI Agents handoff event",
            )
            return

        if event_name == "reasoning_item_created":
            await trace_run.decision(
                action_name="reasoning",
                decision=serialize_for_trace(item),
                agent_name=self._agent_name,
                reason="OpenAI Agents reasoning item",
            )

    async def _finalize(self, *, status: str = "success", error: Any = None) -> None:
        if self._closed:
            return
        self._closed = True
        trace_run = await self._ensure_run()

        final_output = get_first_attr(self._result, "final_output", "text", "output")
        if final_output is None:
            final_output = "".join(self._response_parts).strip() or "[completed]"

        if error is not None:
            await trace_run.final_output(output=f"[error] {_normalize_text(error)}")
            await trace_run.fail(
                error=error,
                metadata={
                    "framework": "openai-agents",
                    "response": serialize_for_trace(final_output),
                },
            )
            return

        await trace_run.final_output(output=final_output, agent_name=self._agent_name)
        await trace_run.finish(
            status=status,
            metadata={
                "framework": "openai-agents",
                "response": serialize_for_trace(final_output),
                "interruptions": serialize_for_trace(
                    get_first_attr(self._result, "interruptions", default=[])
                ),
            },
        )

    async def stream_events(self):
        """Yield underlying events while recording them into Skill Compiler."""

        try:
            async for event in self._result.stream_events():
                await self._handle_event(event)
                yield event
        except Exception as exc:
            await self._finalize(error=exc)
            raise
        else:
            await self._finalize()

    def __aiter__(self):
        return self.stream_events()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)


async def run_openai_agent(
    agent: Any,
    tracer: AsyncSkillCompilerTracer,
    *,
    input: Any,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    default_agent_name: str | None = None,
    runner: Any | None = None,
    **run_kwargs: Any,
) -> Any:
    """Run an OpenAI Agents workflow and capture final output metadata."""

    if runner is None:
        from agents import Runner

        runner = Runner

    agent_name = default_agent_name or _agent_name(agent)
    trace_run = await tracer.trace(
        task_name=task_name or agent_name,
        input_text=_normalize_text(input),
        metadata={
            "framework": "openai-agents",
            "adapter": "agent_skill_compiler.integrations.openai_agents",
            **(metadata or {}),
        },
        agent_name=agent_name,
    )
    try:
        result = await runner.run(agent, input, **run_kwargs)
        final_output = get_first_attr(result, "final_output", "text", "output", default=result)
        await trace_run.final_output(output=final_output, agent_name=agent_name)
        await trace_run.finish(
            metadata={
                "framework": "openai-agents",
                "response": serialize_for_trace(result),
            }
        )
        return result
    except Exception as exc:
        await trace_run.fail(error=exc)
        raise


def run_openai_agent_streamed(
    agent: Any,
    tracer: AsyncSkillCompilerTracer,
    *,
    input: Any,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    default_agent_name: str | None = None,
    runner: Any | None = None,
    **run_kwargs: Any,
) -> TracedOpenAIAgentsStream:
    """Run an OpenAI Agents workflow in streaming mode with automatic tracing."""

    if runner is None:
        from agents import Runner

        runner = Runner

    result = runner.run_streamed(agent, input, **run_kwargs)
    return TracedOpenAIAgentsStream(
        result,
        tracer,
        agent=agent,
        input=input,
        task_name=task_name,
        metadata=metadata,
        default_agent_name=default_agent_name,
    )
