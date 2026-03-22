"""Framework integration helpers."""

from agent_skill_compiler.integrations.agno import run_agno_agent
from agent_skill_compiler.integrations.crewai import create_crewai_event_listener
from agent_skill_compiler.integrations.generic import (
    AsyncSkillCompilerTracer,
    AsyncTracedRun,
    NoopAsyncTracedRun,
    NoopTraceEvent,
    NoopTracedRun,
    SkillCompilerTracer,
    TracedRun,
    get_first_attr,
    normalize_tool_arguments,
    serialize_for_trace,
    trace_run,
    trace_run_async,
)
from agent_skill_compiler.integrations.google_adk import (
    GoogleADKCallbacks,
    create_google_adk_callbacks,
)
from agent_skill_compiler.integrations.langchain import (
    create_langchain_callback_handler,
    create_langgraph_callback_handler,
)
from agent_skill_compiler.integrations.llamaindex import create_llamaindex_callback_handler
from agent_skill_compiler.integrations.microsoft import create_agent_framework_middleware
from agent_skill_compiler.integrations.openai_agents import (
    TracedOpenAIAgentsStream,
    run_openai_agent,
    run_openai_agent_streamed,
)

__all__ = [
    "AsyncSkillCompilerTracer",
    "AsyncTracedRun",
    "NoopAsyncTracedRun",
    "NoopTraceEvent",
    "NoopTracedRun",
    "SkillCompilerTracer",
    "TracedRun",
    "GoogleADKCallbacks",
    "create_agent_framework_middleware",
    "create_crewai_event_listener",
    "create_google_adk_callbacks",
    "create_langchain_callback_handler",
    "create_langgraph_callback_handler",
    "create_llamaindex_callback_handler",
    "get_first_attr",
    "normalize_tool_arguments",
    "run_openai_agent",
    "run_openai_agent_streamed",
    "run_agno_agent",
    "serialize_for_trace",
    "trace_run",
    "trace_run_async",
    "TracedOpenAIAgentsStream",
]
