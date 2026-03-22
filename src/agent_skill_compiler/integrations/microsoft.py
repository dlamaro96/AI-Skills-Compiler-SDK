"""Helpers for tracing Microsoft Agent Framework runs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any

from agent_skill_compiler.integrations.generic import AsyncTracedRun, _normalize_payload, _normalize_text
from agent_skill_compiler.sdk.async_client import AsyncSkillCompilerClient

_TRACE_RUN_KEY = "__asc_traced_run__"
_TRACE_AGENT_NAME_KEY = "__asc_agent_name__"


def _stringify_messages(messages: list[Any] | None) -> str:
    if not messages:
        return ""
    parts: list[str] = []
    for message in messages:
        for attr in ("text", "content"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value:
                parts.append(value)
                break
        else:
            parts.append(repr(message))
    return "\n".join(parts)


def _result_payload(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    text = getattr(result, "text", None)
    if isinstance(text, str) and text:
        return {"response": text}
    messages = getattr(result, "messages", None)
    if messages:
        return {"response": _stringify_messages(list(messages))}
    return _normalize_payload(result)


def create_agent_framework_middleware(
    client: AsyncSkillCompilerClient,
    *,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    default_agent_name: str | None = None,
) -> list[Any]:
    """Return middleware objects that trace Microsoft Agent Framework runs."""

    try:
        from agent_framework import agent_middleware, function_middleware
    except ImportError as exc:
        raise ImportError(
            "Microsoft Agent Framework support requires `pip install agent-framework --pre`."
        ) from exc

    @agent_middleware
    async def traced_agent_middleware(context: Any, call_next: Callable[[], Awaitable[None]]) -> None:
        resolved_agent_name = default_agent_name or getattr(context.agent, "name", None) or "MicrosoftAgentFramework"
        if getattr(context, "function_invocation_kwargs", None) is None:
            context.function_invocation_kwargs = {}
        traced_run = await AsyncTracedRun.start(
            client,
            task_name=task_name or resolved_agent_name,
            input_text=_stringify_messages(list(getattr(context, "messages", []) or [])),
            metadata={
                "framework": "microsoft-agent-framework",
                "adapter": "agent_skill_compiler.integrations.microsoft",
                **(metadata or {}),
            },
            default_agent_name=resolved_agent_name,
        )
        context.function_invocation_kwargs[_TRACE_RUN_KEY] = traced_run
        context.function_invocation_kwargs[_TRACE_AGENT_NAME_KEY] = resolved_agent_name

        try:
            await call_next()
        except Exception as exc:
            await traced_run.fail(error=exc)
            raise

        if getattr(context, "result", None) is not None:
            await traced_run.final_output(
                output=_result_payload(context.result),
                agent_name=resolved_agent_name,
            )
        await traced_run.finish(
            metadata={
                "response": _result_payload(getattr(context, "result", None)),
            }
        )

    @function_middleware
    async def traced_function_middleware(context: Any, call_next: Callable[[], Awaitable[None]]) -> None:
        if getattr(context, "kwargs", None) is None:
            context.kwargs = {}
        traced_run = context.kwargs.pop(_TRACE_RUN_KEY, None)
        resolved_agent_name = context.kwargs.pop(_TRACE_AGENT_NAME_KEY, default_agent_name or "MicrosoftAgentFramework")
        if traced_run is None:
            await call_next()
            return

        function_name = getattr(context.function, "name", None) or "tool"
        semantic_name = function_name
        tool_metadata = {
            "framework": "microsoft-agent-framework",
            "tool_name": function_name,
            "agent_name": resolved_agent_name,
        }
        started_at = perf_counter()
        tool_call_event = await traced_run.tool_call(
            action_name=function_name,
            arguments=getattr(context, "arguments", {}),
            agent_name=resolved_agent_name,
            semantic_name=semantic_name,
            tool_metadata=tool_metadata,
        )
        try:
            await call_next()
        except Exception as exc:
            await traced_run.tool_result(
                action_name=function_name,
                result=exc,
                agent_name=resolved_agent_name,
                latency_ms=int((perf_counter() - started_at) * 1000),
                success=False,
                parent_event_id=tool_call_event.event_id,
                tool_call_id=tool_call_event.event_id,
                semantic_name=semantic_name,
                tool_metadata=tool_metadata,
            )
            raise

        await traced_run.tool_result(
            action_name=function_name,
            result=getattr(context, "result", None),
            agent_name=resolved_agent_name,
            latency_ms=int((perf_counter() - started_at) * 1000),
            parent_event_id=tool_call_event.event_id,
            tool_call_id=tool_call_event.event_id,
            semantic_name=semantic_name,
            tool_metadata=tool_metadata,
        )

    return [traced_agent_middleware, traced_function_middleware]
