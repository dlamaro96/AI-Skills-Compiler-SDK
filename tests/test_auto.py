from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from agent_skill_compiler.auto import SkillCompiler, get_framework_support
from agent_skill_compiler.integrations.openai_agents import run_openai_agent_streamed


class FakeSyncTracer:
    def trace(self, **kwargs):
        return kwargs


class FakeAsyncRun:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._event_index = 0

    async def tool_call(self, **kwargs):
        self.calls.append(("tool_call", kwargs))
        self._event_index += 1
        return SimpleNamespace(event_id=f"evt-{self._event_index}")

    async def tool_result(self, **kwargs):
        self.calls.append(("tool_result", kwargs))
        return SimpleNamespace()

    async def route(self, **kwargs):
        self.calls.append(("route", kwargs))
        return SimpleNamespace()

    async def decision(self, **kwargs):
        self.calls.append(("decision", kwargs))
        return SimpleNamespace()

    async def final_output(self, **kwargs):
        self.calls.append(("final_output", kwargs))
        return SimpleNamespace()

    async def finish(self, **kwargs):
        self.calls.append(("finish", kwargs))
        return SimpleNamespace()

    async def fail(self, **kwargs):
        self.calls.append(("fail", kwargs))
        return SimpleNamespace()


class FakeAsyncTracer:
    def __init__(self) -> None:
        self.trace_kwargs = None
        self.client = SimpleNamespace()
        self.run = FakeAsyncRun()

    async def trace(self, **kwargs):
        self.trace_kwargs = kwargs
        return self.run


class FakeStreamResult:
    def __init__(self) -> None:
        self.final_output = "Final answer"
        self.interruptions = []

    async def stream_events(self):
        yield SimpleNamespace(
            type="raw_response_event",
            data=SimpleNamespace(delta="Hello "),
        )
        yield SimpleNamespace(
            type="run_item_stream_event",
            name="tool_called",
            item=SimpleNamespace(
                raw_item=SimpleNamespace(name="search_docs", arguments={"query": "refund"}),
                call_id="tool-1",
            ),
        )
        yield SimpleNamespace(
            type="run_item_stream_event",
            name="tool_output",
            item=SimpleNamespace(
                name="search_docs",
                output={"documents": ["refund-policy"]},
                call_id="tool-1",
            ),
        )


class FakeRunner:
    @staticmethod
    def run_streamed(agent, input, **kwargs):
        del agent, input, kwargs
        return FakeStreamResult()


class OpenAIAgent:
    pass


OpenAIAgent.__module__ = "agents.core"


class AgnoAgent:
    pass


AgnoAgent.__module__ = "agno.agent"


class LlamaIndexAgent:
    pass


LlamaIndexAgent.__module__ = "llama_index.core.agent"


class SkillCompilerAutoTests(unittest.TestCase):
    def test_support_matrix_lists_major_frameworks(self):
        frameworks = {item.framework for item in get_framework_support()}
        self.assertIn("openai-agents", frameworks)
        self.assertIn("agno", frameworks)
        self.assertIn("microsoft-agent-framework", frameworks)
        self.assertIn("google-adk", frameworks)
        self.assertIn("crewai", frameworks)

    def test_detect_framework_uses_module_names(self):
        compiler = SkillCompiler(sync_tracer=FakeSyncTracer(), async_tracer=FakeAsyncTracer())
        self.assertEqual(compiler.detect_framework(OpenAIAgent()), "openai-agents")
        self.assertEqual(compiler.detect_framework(AgnoAgent()), "agno")
        self.assertEqual(compiler.detect_framework(LlamaIndexAgent()), "llamaindex")
        self.assertEqual(compiler.detect_framework(object()), "custom")

    def test_openai_stream_wrapper_records_trace_events(self):
        tracer = FakeAsyncTracer()
        stream = run_openai_agent_streamed(
            OpenAIAgent(),
            tracer,
            input="hello",
            task_name="support_chat",
            metadata={"service": "tests"},
            runner=FakeRunner,
        )

        async def consume():
            seen = []
            async for event in stream.stream_events():
                seen.append(event)
            return seen

        events = asyncio.run(consume())
        self.assertEqual(len(events), 3)
        self.assertEqual(tracer.trace_kwargs["task_name"], "support_chat")
        self.assertEqual(tracer.trace_kwargs["metadata"]["framework"], "openai-agents")

        call_types = [name for name, _ in tracer.run.calls]
        self.assertIn("tool_call", call_types)
        self.assertIn("tool_result", call_types)
        self.assertIn("final_output", call_types)
        self.assertIn("finish", call_types)


if __name__ == "__main__":
    unittest.main()
