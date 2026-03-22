"""Helpers for tracing Agno agent runs."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from agent_skill_compiler.integrations.generic import TracedRun, _normalize_payload, _normalize_text
from agent_skill_compiler.sdk.client import SkillCompilerClient


def _agent_name(agent: Any) -> str:
    return getattr(agent, "name", None) or getattr(agent, "agent_name", None) or "AgnoAgent"


def _coerce_input_text(value: Any) -> str:
    return value if isinstance(value, str) else _normalize_text(value)


def _append_tool_hook(agent: Any, hook: Callable[..., Any]) -> Callable[[], None]:
    original_hooks = list(getattr(agent, "tool_hooks", []) or [])
    setattr(agent, "tool_hooks", [*original_hooks, hook])

    def restore() -> None:
        setattr(agent, "tool_hooks", original_hooks)

    return restore


def _build_tool_hook(traced_run: TracedRun, *, agent_name: str) -> Callable[..., Any]:
    def hook(function_name: str, function_call: Callable[..., Any], arguments: dict[str, Any], **_: Any) -> Any:
        started_at = perf_counter()
        semantic_name = function_name
        tool_metadata = {
            "framework": "agno",
            "tool_name": function_name,
            "agent_name": agent_name,
        }
        tool_call_event = traced_run.tool_call(
            action_name=function_name,
            arguments=arguments,
            agent_name=agent_name,
            semantic_name=semantic_name,
            tool_metadata=tool_metadata,
        )
        try:
            result = function_call(**arguments)
        except Exception as exc:
            traced_run.tool_result(
                action_name=function_name,
                result=exc,
                agent_name=agent_name,
                latency_ms=int((perf_counter() - started_at) * 1000),
                success=False,
                parent_event_id=tool_call_event.event_id,
                tool_call_id=tool_call_event.event_id,
                semantic_name=semantic_name,
                tool_metadata=tool_metadata,
            )
            raise
        traced_run.tool_result(
            action_name=function_name,
            result=result,
            agent_name=agent_name,
            latency_ms=int((perf_counter() - started_at) * 1000),
            parent_event_id=tool_call_event.event_id,
            tool_call_id=tool_call_event.event_id,
            semantic_name=semantic_name,
            tool_metadata=tool_metadata,
        )
        return result

    return hook


def run_agno_agent(
    agent: Any,
    client: SkillCompilerClient,
    *,
    input: Any,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    default_agent_name: str | None = None,
    **run_kwargs: Any,
) -> Any:
    """Run an Agno agent with Skill Compiler tracing attached."""

    resolved_agent_name = default_agent_name or _agent_name(agent)
    traced_run = TracedRun.start(
        client,
        task_name=task_name or resolved_agent_name,
        input_text=_coerce_input_text(input),
        metadata={
            "framework": "agno",
            "adapter": "agent_skill_compiler.integrations.agno",
            **(metadata or {}),
        },
        default_agent_name=resolved_agent_name,
    )
    restore_hooks = _append_tool_hook(agent, _build_tool_hook(traced_run, agent_name=resolved_agent_name))
    try:
        response = agent.run(input=input, **run_kwargs)
        traced_run.final_output(output=response, agent_name=resolved_agent_name)
        traced_run.finish(
            metadata={
                "response": _normalize_payload(response),
            }
        )
        return response
    except Exception as exc:
        traced_run.fail(error=exc)
        raise
    finally:
        restore_hooks()
