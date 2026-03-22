"""Callback handlers for LangChain and LangGraph style runtimes."""

from __future__ import annotations

from typing import Any
from uuid import UUID

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


def _coerce_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return str(value)
    return str(value)


def create_langchain_callback_handler(
    tracer: SkillCompilerTracer,
    *,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    default_agent_name: str = "LangChain",
):
    """Create a callback handler compatible with LangChain and LangGraph."""

    try:
        from langchain_core.callbacks.base import BaseCallbackHandler
    except ImportError as exc:
        raise ImportError(
            "LangChain/LangGraph support requires `langchain-core` or `langchain` to be installed."
        ) from exc

    class _SkillCompilerLangChainHandler(BaseCallbackHandler):
        raise_error = False
        run_inline = True

        def __init__(self) -> None:
            super().__init__()
            self._trace_run = None
            self._tool_events: dict[str, dict[str, str | None]] = {}
            self._root_run_id: str | None = None
            self._finalized = False

        def _ensure_run(
            self,
            *,
            inputs: Any = None,
            serialized: Any = None,
            extra_metadata: dict[str, Any] | None = None,
        ) -> Any:
            if self._trace_run is None:
                resolved_name = task_name or get_first_attr(
                    serialized,
                    "name",
                    "id",
                    default=default_agent_name,
                )
                resolved_input = _normalize_text(inputs)
                self._trace_run = tracer.trace(
                    task_name=resolved_name,
                    input_text=resolved_input,
                    metadata={
                        "framework": "langchain",
                        "adapter": "agent_skill_compiler.integrations.langchain",
                        **(metadata or {}),
                        **serialize_for_trace(extra_metadata or {}),
                    },
                    agent_name=default_agent_name,
                )
            return self._trace_run

        def _finish(self, *, output: Any = None, error: Any = None) -> None:
            if self._trace_run is None or self._finalized:
                return
            self._finalized = True
            if error is not None:
                self._trace_run.fail(error=error)
                return
            if output is not None:
                self._trace_run.final_output(output=output, agent_name=default_agent_name)
            self._trace_run.finish(
                metadata={
                    "framework": "langchain",
                    "response": serialize_for_trace(output),
                }
            )

        def on_chain_start(
            self,
            serialized: dict[str, Any],
            inputs: dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            del kwargs
            if parent_run_id is None and self._root_run_id is None:
                self._root_run_id = _coerce_id(run_id)
                self._ensure_run(
                    inputs=inputs,
                    serialized=serialized,
                    extra_metadata={
                        "tags": tags or [],
                        "langchain_metadata": metadata or {},
                    },
                )

        def on_chain_end(
            self,
            outputs: dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> Any:
            del parent_run_id, kwargs
            if _coerce_id(run_id) == self._root_run_id:
                self._finish(output=outputs)

        def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> Any:
            del parent_run_id, kwargs
            if _coerce_id(run_id) == self._root_run_id:
                self._finish(error=error)

        def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> Any:
            del parent_run_id, kwargs
            trace_run = self._ensure_run(serialized=serialized)
            action_name = str(
                get_first_attr(serialized, "name", default="tool")
            )
            tool_call_id = _coerce_id(run_id)
            event = trace_run.tool_call(
                action_name=action_name,
                arguments=normalize_tool_arguments(input_str),
                agent_name=default_agent_name,
                semantic_name=action_name,
                tool_metadata={
                    "framework": "langchain",
                    "tool_name": action_name,
                    "serialized": serialize_for_trace(serialized),
                },
                tool_call_id=tool_call_id,
            )
            if tool_call_id:
                self._tool_events[tool_call_id] = {
                    "event_id": event.event_id,
                    "action_name": action_name,
                }

        def on_tool_end(
            self,
            output: Any,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> Any:
            del parent_run_id, kwargs
            if self._trace_run is None:
                return
            tool_call_id = _coerce_id(run_id)
            tool_event = self._tool_events.get(tool_call_id or "")
            action_name = (tool_event or {}).get("action_name") or "tool"
            self._trace_run.tool_result(
                action_name=action_name,
                result=output,
                agent_name=default_agent_name,
                semantic_name=action_name,
                parent_event_id=(tool_event or {}).get("event_id"),
                tool_call_id=tool_call_id,
                tool_metadata={
                    "framework": "langchain",
                    "tool_name": action_name,
                },
            )

        def on_tool_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> Any:
            del parent_run_id, kwargs
            if self._trace_run is None:
                return
            tool_call_id = _coerce_id(run_id)
            tool_event = self._tool_events.get(tool_call_id or "")
            action_name = (tool_event or {}).get("action_name") or "tool"
            self._trace_run.tool_result(
                action_name=action_name,
                result=error,
                agent_name=default_agent_name,
                semantic_name=action_name,
                parent_event_id=(tool_event or {}).get("event_id"),
                tool_call_id=tool_call_id,
                success=False,
                tool_metadata={
                    "framework": "langchain",
                    "tool_name": action_name,
                },
            )

        def on_agent_finish(self, finish: Any, **kwargs: Any) -> Any:
            del kwargs
            trace_run = self._ensure_run()
            trace_run.final_output(output=serialize_for_trace(finish), agent_name=default_agent_name)

        def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
            del kwargs
            if self._trace_run is None:
                return
            llm_output = get_first_attr(response, "llm_output", default={})
            if llm_output:
                self._trace_run.update_metadata(llm_output=serialize_for_trace(llm_output))

        def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
            del kwargs
            if self._trace_run is None:
                return
            self._trace_run.error(error=error, agent_name=default_agent_name)

    return _SkillCompilerLangChainHandler()


def create_langgraph_callback_handler(
    tracer: SkillCompilerTracer,
    *,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    default_agent_name: str = "LangGraph",
):
    """Alias LangGraph support to the LangChain callback handler."""

    return create_langchain_callback_handler(
        tracer,
        task_name=task_name,
        metadata={
            "framework": "langgraph",
            **(metadata or {}),
        },
        default_agent_name=default_agent_name,
    )
