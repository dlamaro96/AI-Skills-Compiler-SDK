"""Unified Skill Compiler facade for framework integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_skill_compiler.integrations.agno import run_agno_agent
from agent_skill_compiler.integrations.crewai import create_crewai_event_listener
from agent_skill_compiler.integrations.generic import AsyncSkillCompilerTracer, SkillCompilerTracer
from agent_skill_compiler.integrations.google_adk import create_google_adk_callbacks
from agent_skill_compiler.integrations.langchain import (
    create_langchain_callback_handler,
    create_langgraph_callback_handler,
)
from agent_skill_compiler.integrations.llamaindex import create_llamaindex_callback_handler
from agent_skill_compiler.integrations.microsoft import create_agent_framework_middleware
from agent_skill_compiler.integrations.openai_agents import (
    run_openai_agent,
    run_openai_agent_streamed,
)


@dataclass(frozen=True)
class FrameworkSupport:
    """Human-readable support metadata for one framework."""

    framework: str
    support_level: str
    entrypoint: str
    notes: str


_FRAMEWORK_SUPPORT: tuple[FrameworkSupport, ...] = (
    FrameworkSupport(
        framework="openai-agents",
        support_level="native-runner",
        entrypoint="SkillCompiler.openai_agents.run_streamed(...)",
        notes="Best support path for OpenAI Agents. Streaming captures tool calls, handoffs, and final output automatically.",
    ),
    FrameworkSupport(
        framework="agno",
        support_level="native-runner",
        entrypoint="SkillCompiler.agno.run(...)",
        notes="Runs the Agno agent with automatic tool hooks and final output tracing.",
    ),
    FrameworkSupport(
        framework="microsoft-agent-framework",
        support_level="middleware",
        entrypoint="SkillCompiler.microsoft.middleware(...)",
        notes="Returns middleware objects to attach at agent creation time.",
    ),
    FrameworkSupport(
        framework="google-adk",
        support_level="callbacks",
        entrypoint="SkillCompiler.google_adk.callbacks(...)",
        notes="Returns callback functions to attach to ADK agents.",
    ),
    FrameworkSupport(
        framework="crewai",
        support_level="event-listener",
        entrypoint="SkillCompiler.crewai.listener(...)",
        notes="Returns a CrewAI listener instance. Import and instantiate it where the crew or flow starts.",
    ),
    FrameworkSupport(
        framework="langchain",
        support_level="callback-handler",
        entrypoint="SkillCompiler.langchain.callback_handler(...)",
        notes="Returns a LangChain callback handler. Works for custom agents and chains that expose callbacks.",
    ),
    FrameworkSupport(
        framework="langgraph",
        support_level="callback-handler",
        entrypoint="SkillCompiler.langgraph.callback_handler(...)",
        notes="Uses the LangChain callback surface that LangGraph already supports.",
    ),
    FrameworkSupport(
        framework="llamaindex",
        support_level="callback-handler",
        entrypoint="SkillCompiler.llamaindex.callback_handler(...)",
        notes="Returns a LlamaIndex callback handler for callback managers and settings-based registration.",
    ),
    FrameworkSupport(
        framework="pydantic-ai",
        support_level="generic-trace",
        entrypoint="SkillCompiler.trace(...)",
        notes="No native adapter yet. Use the generic tracer around your agent calls or bridge through your own OpenTelemetry pipeline.",
    ),
    FrameworkSupport(
        framework="custom",
        support_level="generic-trace",
        entrypoint="SkillCompiler.trace(...)",
        notes="Use the generic tracer when the framework is not listed or when you need custom event mapping.",
    ),
)


def get_framework_support() -> tuple[FrameworkSupport, ...]:
    """Return the current framework support matrix."""

    return _FRAMEWORK_SUPPORT


class _AgnoIntegration:
    def __init__(self, compiler: "SkillCompiler") -> None:
        self._compiler = compiler

    def run(
        self,
        agent: Any,
        *,
        input: Any,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        default_agent_name: str | None = None,
        **run_kwargs: Any,
    ) -> Any:
        return run_agno_agent(
            agent,
            self._compiler.sync_tracer.client,
            input=input,
            task_name=task_name,
            metadata=metadata,
            default_agent_name=default_agent_name,
            **run_kwargs,
        )


class _OpenAIAgentsIntegration:
    def __init__(self, compiler: "SkillCompiler") -> None:
        self._compiler = compiler

    async def run(
        self,
        agent: Any,
        *,
        input: Any,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        default_agent_name: str | None = None,
        runner: Any | None = None,
        **run_kwargs: Any,
    ) -> Any:
        return await run_openai_agent(
            agent,
            self._compiler.async_tracer,
            input=input,
            task_name=task_name,
            metadata=metadata,
            default_agent_name=default_agent_name,
            runner=runner,
            **run_kwargs,
        )

    def run_streamed(
        self,
        agent: Any,
        *,
        input: Any,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        default_agent_name: str | None = None,
        runner: Any | None = None,
        **run_kwargs: Any,
    ) -> Any:
        return run_openai_agent_streamed(
            agent,
            self._compiler.async_tracer,
            input=input,
            task_name=task_name,
            metadata=metadata,
            default_agent_name=default_agent_name,
            runner=runner,
            **run_kwargs,
        )


class _MicrosoftIntegration:
    def __init__(self, compiler: "SkillCompiler") -> None:
        self._compiler = compiler

    def middleware(
        self,
        *,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        default_agent_name: str | None = None,
    ) -> list[Any]:
        return create_agent_framework_middleware(
            self._compiler.async_tracer.client,
            task_name=task_name,
            metadata=metadata,
            default_agent_name=default_agent_name,
        )


class _GoogleADKIntegration:
    def __init__(self, compiler: "SkillCompiler") -> None:
        self._compiler = compiler

    def callbacks(
        self,
        *,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        default_agent_name: str = "GoogleADK",
    ) -> Any:
        return create_google_adk_callbacks(
            self._compiler.sync_tracer,
            task_name=task_name,
            metadata=metadata,
            default_agent_name=default_agent_name,
        )


class _CrewAIIntegration:
    def __init__(self, compiler: "SkillCompiler") -> None:
        self._compiler = compiler

    def listener(
        self,
        *,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        default_agent_name: str = "CrewAI",
    ) -> Any:
        return create_crewai_event_listener(
            self._compiler.sync_tracer,
            task_name=task_name,
            metadata=metadata,
            default_agent_name=default_agent_name,
        )


class _LangChainIntegration:
    def __init__(self, compiler: "SkillCompiler") -> None:
        self._compiler = compiler

    def callback_handler(
        self,
        *,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        default_agent_name: str = "LangChain",
    ) -> Any:
        return create_langchain_callback_handler(
            self._compiler.sync_tracer,
            task_name=task_name,
            metadata=metadata,
            default_agent_name=default_agent_name,
        )


class _LangGraphIntegration:
    def __init__(self, compiler: "SkillCompiler") -> None:
        self._compiler = compiler

    def callback_handler(
        self,
        *,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        default_agent_name: str = "LangGraph",
    ) -> Any:
        return create_langgraph_callback_handler(
            self._compiler.sync_tracer,
            task_name=task_name,
            metadata=metadata,
            default_agent_name=default_agent_name,
        )


class _LlamaIndexIntegration:
    def __init__(self, compiler: "SkillCompiler") -> None:
        self._compiler = compiler

    def callback_handler(
        self,
        *,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        default_agent_name: str = "LlamaIndex",
    ) -> Any:
        return create_llamaindex_callback_handler(
            self._compiler.sync_tracer,
            task_name=task_name,
            metadata=metadata,
            default_agent_name=default_agent_name,
        )


class SkillCompiler:
    """Unified facade for generic tracing and framework-specific integrations."""

    def __init__(
        self,
        *,
        sync_tracer: SkillCompilerTracer,
        async_tracer: AsyncSkillCompilerTracer,
    ) -> None:
        self.sync_tracer = sync_tracer
        self.async_tracer = async_tracer
        self.agno = _AgnoIntegration(self)
        self.openai_agents = _OpenAIAgentsIntegration(self)
        self.microsoft = _MicrosoftIntegration(self)
        self.google_adk = _GoogleADKIntegration(self)
        self.crewai = _CrewAIIntegration(self)
        self.langchain = _LangChainIntegration(self)
        self.langgraph = _LangGraphIntegration(self)
        self.llamaindex = _LlamaIndexIntegration(self)

    @classmethod
    def from_env(
        cls,
        *,
        optional: bool = True,
        timeout: float = 30.0,
        default_agent_name: str = "application",
        service: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SkillCompiler":
        """Build the unified facade from environment variables."""

        merged_metadata = dict(metadata or {})
        if service:
            merged_metadata.setdefault("service", service)

        return cls(
            sync_tracer=SkillCompilerTracer.from_env(
                optional=optional,
                timeout=timeout,
                default_agent_name=default_agent_name,
                metadata=merged_metadata,
            ),
            async_tracer=AsyncSkillCompilerTracer.from_env(
                optional=optional,
                timeout=timeout,
                default_agent_name=default_agent_name,
                metadata=merged_metadata,
            ),
        )

    def trace(self, **kwargs: Any) -> Any:
        """Start a generic synchronous trace run."""

        return self.sync_tracer.trace(**kwargs)

    async def trace_async(self, **kwargs: Any) -> Any:
        """Start a generic asynchronous trace run."""

        return await self.async_tracer.trace(**kwargs)

    def support(self) -> tuple[FrameworkSupport, ...]:
        """Return the SDK support matrix."""

        return get_framework_support()

    def detect_framework(self, target: Any) -> str:
        """Best-effort detection of a framework from an object."""

        module = getattr(type(target), "__module__", "") or getattr(target, "__module__", "")
        module = module.lower()

        if module.startswith("agno.") or ".agno." in module:
            return "agno"
        if module.startswith("agents.") or ".agents." in module:
            return "openai-agents"
        if module.startswith("agent_framework.") or ".agent_framework." in module:
            return "microsoft-agent-framework"
        if module.startswith("google.adk.") or ".google.adk." in module:
            return "google-adk"
        if module.startswith("crewai.") or ".crewai." in module:
            return "crewai"
        if module.startswith("langchain.") or ".langchain." in module or module.startswith("langchain_core."):
            return "langchain"
        if module.startswith("langgraph.") or ".langgraph." in module:
            return "langgraph"
        if module.startswith("llama_index.") or ".llama_index." in module:
            return "llamaindex"
        if module.startswith("pydantic_ai.") or ".pydantic_ai." in module:
            return "pydantic-ai"
        return "custom"

    def instrument(
        self,
        target: Any | None = None,
        *,
        framework: str = "auto",
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        default_agent_name: str | None = None,
    ) -> Any:
        """Return the most convenient integration object for the requested framework."""

        resolved_framework = framework
        if resolved_framework == "auto":
            if target is None:
                raise ValueError("`target` is required when framework='auto'.")
            resolved_framework = self.detect_framework(target)

        if resolved_framework == "agno":
            if target is None:
                return self.agno

            compiler = self

            class _BoundAgnoAgent:
                def run(self, *, input: Any, **run_kwargs: Any) -> Any:
                    return compiler.agno.run(
                        target,
                        input=input,
                        task_name=task_name,
                        metadata=metadata,
                        default_agent_name=default_agent_name,
                        **run_kwargs,
                    )

            return _BoundAgnoAgent()

        if resolved_framework == "openai-agents":
            if target is None:
                return self.openai_agents

            compiler = self

            class _BoundOpenAIAgent:
                async def run(self, *, input: Any, **run_kwargs: Any) -> Any:
                    return await compiler.openai_agents.run(
                        target,
                        input=input,
                        task_name=task_name,
                        metadata=metadata,
                        default_agent_name=default_agent_name,
                        **run_kwargs,
                    )

                def run_streamed(self, *, input: Any, **run_kwargs: Any) -> Any:
                    return compiler.openai_agents.run_streamed(
                        target,
                        input=input,
                        task_name=task_name,
                        metadata=metadata,
                        default_agent_name=default_agent_name,
                        **run_kwargs,
                    )

            return _BoundOpenAIAgent()

        if resolved_framework == "microsoft-agent-framework":
            return self.microsoft.middleware(
                task_name=task_name,
                metadata=metadata,
                default_agent_name=default_agent_name,
            )

        if resolved_framework == "google-adk":
            return self.google_adk.callbacks(
                task_name=task_name,
                metadata=metadata,
                default_agent_name=default_agent_name or "GoogleADK",
            )

        if resolved_framework == "crewai":
            return self.crewai.listener(
                task_name=task_name,
                metadata=metadata,
                default_agent_name=default_agent_name or "CrewAI",
            )

        if resolved_framework == "langchain":
            return self.langchain.callback_handler(
                task_name=task_name,
                metadata=metadata,
                default_agent_name=default_agent_name or "LangChain",
            )

        if resolved_framework == "langgraph":
            return self.langgraph.callback_handler(
                task_name=task_name,
                metadata=metadata,
                default_agent_name=default_agent_name or "LangGraph",
            )

        if resolved_framework == "llamaindex":
            return self.llamaindex.callback_handler(
                task_name=task_name,
                metadata=metadata,
                default_agent_name=default_agent_name or "LlamaIndex",
            )

        if resolved_framework in {"pydantic-ai", "custom"}:
            return self.sync_tracer

        raise ValueError(f"Unsupported framework '{resolved_framework}'.")
