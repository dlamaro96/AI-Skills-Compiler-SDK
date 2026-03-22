# Agent Skill Compiler

`agent-skill-compiler` is the Python ingestion SDK for Agent Skill Compiler.

It helps backend developers send agent runs, tool calls, outputs, routing decisions, and final responses to a remote `skills-compiler-be` / ASC backend with as little framework-specific code as possible.

This package is for `pip install` only. It does not run the ASC backend, database, or frontend.

## Project Status

- Status: beta
- Stability promise: public APIs documented in this README are intended to stay backward-compatible across minor releases whenever possible
- Local test command: `PYTHONPATH=src python -m unittest discover -s tests`
- Changelog: [`CHANGELOG.md`](./CHANGELOG.md)
- Security policy: [`SECURITY.md`](./SECURITY.md)
- Roadmap: [`ROADMAP.md`](./ROADMAP.md)

## Repository Layout

- SDK repository: [AI-Skills-Compiler-SDK](https://github.com/dlamaro96/AI-Skills-Compiler-SDK)
- Full platform repository: [AI-Skills-Compiler](https://github.com/dlamaro96/AI-Skills-Compiler)

If you want the backend, frontend, dashboards, and the wider platform, use the platform repository. This repository is only the Python SDK published to PyPI.

## What This SDK Optimizes For

- A single package for many Python agent frameworks
- Safe backend-only ingestion to ASC
- Clear support levels per framework
- Good default behavior with optional no-op mode
- A generic fallback when your framework is not supported natively yet

## Install

Base package:

```bash
pip install agent-skill-compiler
```

Optional framework extras:

```bash
pip install "agent-skill-compiler[agno]"
pip install "agent-skill-compiler[openai-agents]"
pip install "agent-skill-compiler[microsoft]"
pip install "agent-skill-compiler[google-adk]"
pip install "agent-skill-compiler[crewai]"
pip install "agent-skill-compiler[langchain]"
pip install "agent-skill-compiler[llamaindex]"
pip install "agent-skill-compiler[pydantic-ai]"
```

You can also install your framework directly and keep `agent-skill-compiler` as a separate dependency.

## Required Environment Variables

Your application backend needs:

```bash
ASC_BASE_URL=http://your-asc-backend
ASC_PUBLIC_KEY=asc_pk_...
ASC_SECRET_KEY=asc_sk_...
```

Legacy aliases are also supported:

```bash
SKILL_COMPILER_HOST=http://your-asc-backend
SKILL_COMPILER_PUBLIC_KEY=asc_pk_...
SKILL_COMPILER_SECRET_KEY=asc_sk_...
```

If the environment variables are missing and you use `optional=True`, the SDK becomes a safe no-op and will not break your app.

## Fastest Way To Start

Use the unified facade:

```python
from agent_skill_compiler import SkillCompiler

asc = SkillCompiler.from_env(
    optional=True,
    service="app-backend",
    default_agent_name="application",
)
```

From there, pick the integration that matches your runtime.

## Support Matrix

| Framework | Support level | Best entrypoint | What you get |
| --- | --- | --- | --- |
| OpenAI Agents SDK | Native runner | `asc.openai_agents.run_streamed(...)` | Run tracing, tool calls, tool outputs, handoffs, final output |
| Agno | Native runner | `asc.agno.run(...)` | Run tracing, tool hooks, final output |
| Microsoft Agent Framework | Middleware | `asc.microsoft.middleware(...)` | Agent and function tracing through middleware |
| Google ADK | Callbacks | `asc.google_adk.callbacks(...)` | Agent, model, and tool callback tracing |
| CrewAI | Event listener | `asc.crewai.listener(...)` | Crew lifecycle and tool usage tracing |
| LangChain | Callback handler | `asc.langchain.callback_handler(...)` | Tool and chain tracing through callbacks |
| LangGraph | Callback handler | `asc.langgraph.callback_handler(...)` | Graph tracing via LangChain callback surface |
| LlamaIndex | Callback handler | `asc.llamaindex.callback_handler(...)` | Query, LLM, and tool tracing through callbacks |
| PydanticAI | Generic fallback | `asc.trace(...)` | Manual tracing around your agent call |
| Custom runtimes | Generic fallback | `asc.trace(...)` | Full manual event mapping |

## What Ships To PyPI

The published package is intentionally small.

Included:

- `src/agent_skill_compiler`
- `README.md`
- `LICENSE`
- package metadata from `pyproject.toml`

Not included:

- local databases such as `.data/`
- local virtual environments such as `.venv/`
- local install checks such as `.install-check/`, `.pypi-install/`, `.release-check/`
- build outputs such as `dist/`
- frontend and backend application code from the main platform repo

This means `.data/agent_skill_compiler.db` is not required for the open-source SDK and should not be part of the published package.

## GitHub To PyPI Publishing

This repository is configured for GitHub-based publishing to PyPI using trusted publishing.

Included workflows:

- `.github/workflows/ci.yml`
- `.github/workflows/release.yml`

How publishing works:

1. Push changes to `main`.
2. Create a GitHub release.
3. The `release.yml` workflow builds the package and publishes it to PyPI.

One-time PyPI setup is still required:

1. Go to the PyPI project settings for `agent-skill-compiler`.
2. Open the trusted publishing section.
3. Add this GitHub repository as a trusted publisher:
   - Owner: `dlamaro96`
   - Repository: `AI-Skills-Compiler-SDK`
   - Workflow: `release.yml`
   - Environment: `pypi`

If the trusted publisher is not configured on PyPI, the GitHub workflow will build successfully but publishing will fail.

## Quick Start By Framework

### Generic Fallback

Use this when your framework is custom, partially supported, or you need full control.

```python
from agent_skill_compiler import SkillCompiler

asc = SkillCompiler.from_env(optional=True, service="support-api")

with asc.trace(
    task_name="support_triage",
    input_text="Investigate this issue.",
    metadata={
        "framework": "custom",
        "workflow": "support_triage",
        "session_id": "sess_123",
        "user_id": "user_123",
    },
) as run:
    tool_call = run.tool_call(
        action_name="search_docs",
        semantic_name="knowledge_search",
        arguments={"query": "refund policy"},
    )

    run.tool_result(
        action_name="search_docs",
        semantic_name="knowledge_search",
        result={"documents": ["refund-policy-v2"]},
        tool_call_id=tool_call.event_id,
        parent_event_id=tool_call.event_id,
    )

    run.final_output(output="Escalate to billing.")
```

### OpenAI Agents SDK

This is the preferred path when you use `Runner.run_streamed(...)`.

```python
from agent_skill_compiler import SkillCompiler

asc = SkillCompiler.from_env(optional=True, service="chat-api")

stream = asc.openai_agents.run_streamed(
    agent,
    input=conversation_messages,
    task_name="customer_support_chat",
    metadata={
        "workflow": "customer_support_chat",
        "session_id": session_id,
        "user_id": user_id,
    },
)

async for event in stream.stream_events():
    # forward to your UI, websocket, SSE, etc.
    pass
```

Notes:

- Streaming mode is the best-supported path because it exposes tool and handoff events.
- Non-streaming `asc.openai_agents.run(...)` records the run and final output, but not as much fine-grained tool detail as the streamed integration.

### Agno

```python
from agent_skill_compiler import SkillCompiler

asc = SkillCompiler.from_env(optional=True, service="agno-api")

response = asc.agno.run(
    agent,
    input="Summarize the latest billing guidance.",
    task_name="billing_guidance",
    metadata={"workflow": "billing_guidance"},
)
```

### Microsoft Agent Framework

Attach middleware when building the agent:

```python
from agent_skill_compiler import SkillCompiler

asc = SkillCompiler.from_env(optional=True, service="weather-api")

middleware = asc.microsoft.middleware(
    task_name="weather_assistant",
    metadata={"workflow": "weather_assistant"},
)
```

### Google ADK

Attach the callback bundle during agent creation:

```python
from agent_skill_compiler import SkillCompiler

asc = SkillCompiler.from_env(optional=True, service="adk-api")

callbacks = asc.google_adk.callbacks(
    task_name="travel_assistant",
    metadata={"workflow": "travel_assistant"},
)

agent = LlmAgent(
    name="TravelAssistant",
    model="gemini-2.5-flash",
    instruction="Be helpful.",
    before_agent_callback=callbacks.before_agent,
    after_agent_callback=callbacks.after_agent,
    before_model_callback=callbacks.before_model,
    after_model_callback=callbacks.after_model,
    before_tool_callback=callbacks.before_tool,
    after_tool_callback=callbacks.after_tool,
)
```

### CrewAI

Create a listener and import it where your crew or flow starts:

```python
from agent_skill_compiler import SkillCompiler

asc = SkillCompiler.from_env(optional=True, service="crewai-api")

skill_compiler_listener = asc.crewai.listener(
    task_name="research_crew",
    metadata={"workflow": "research_crew"},
)
```

### LangChain

```python
from agent_skill_compiler import SkillCompiler

asc = SkillCompiler.from_env(optional=True, service="langchain-api")

handler = asc.langchain.callback_handler(
    task_name="support_chain",
    metadata={"workflow": "support_chain"},
)

chain.invoke(
    {"question": "What is our refund policy?"},
    config={"callbacks": [handler]},
)
```

### LangGraph

Use the same callback style through the LangChain-compatible callback surface:

```python
from agent_skill_compiler import SkillCompiler

asc = SkillCompiler.from_env(optional=True, service="langgraph-api")

handler = asc.langgraph.callback_handler(
    task_name="support_graph",
    metadata={"workflow": "support_graph"},
)

graph.invoke(
    {"messages": [{"role": "user", "content": "Help me with my subscription"}]},
    config={"callbacks": [handler]},
)
```

### LlamaIndex

```python
from agent_skill_compiler import SkillCompiler

asc = SkillCompiler.from_env(optional=True, service="llamaindex-api")

handler = asc.llamaindex.callback_handler(
    task_name="retrieval_agent",
    metadata={"workflow": "retrieval_agent"},
)
```

### PydanticAI

There is no native adapter yet. Use the generic tracer for now:

```python
from agent_skill_compiler import SkillCompiler

asc = SkillCompiler.from_env(optional=True, service="pydantic-ai-api")

with asc.trace(
    task_name="support_agent",
    input_text=user_prompt,
    metadata={"framework": "pydantic-ai"},
) as run:
    result = agent.run_sync(user_prompt)
    run.final_output(output=result)
```

## One-Line Detection Helper

If you want one entrypoint and are willing to let the SDK pick the integration shape, use `instrument(...)`:

```python
from agent_skill_compiler import SkillCompiler

asc = SkillCompiler.from_env(optional=True, service="chat-api")

instrumented_agent = asc.instrument(agent, framework="auto")
```

What `instrument(...)` returns depends on the framework:

- Agno: a bound object with `.run(...)`
- OpenAI Agents: a bound object with `.run(...)` and `.run_streamed(...)`
- Microsoft Agent Framework: middleware list
- Google ADK: callback bundle
- CrewAI: event listener
- LangChain / LangGraph / LlamaIndex: callback handler
- Custom / unsupported: the generic tracer

## What Is Supported vs Not Yet Native

Fully or strongly supported:

- OpenAI Agents streamed runs
- Agno runs
- Microsoft Agent Framework middleware
- Google ADK callback wiring
- CrewAI listener wiring
- LangChain / LangGraph callback wiring
- LlamaIndex callback wiring

Supported with manual fallback:

- PydanticAI
- Any custom runtime
- Any framework where you only want to trace selected parts of execution

Not included in this package:

- ASC project creation
- ASC key creation
- Running the ASC backend
- Running the ASC frontend
- A server-side OpenTelemetry ingest pipeline

## Recommended Workarounds

If your framework is not fully native yet:

1. Wrap the outer agent call with `asc.trace(...)`.
2. Record tool calls with `run.tool_call(...)` and `run.tool_result(...)`.
3. Use stable `semantic_name` values so the backend groups equivalent tools together.
4. Pass framework metadata like `framework`, `workflow`, `session_id`, and `user_id`.

If your framework already emits callbacks or events:

1. Use the closest callback-based integration in this SDK.
2. Add framework-native metadata to your handler registration.
3. Fall back to generic tracing only for missing event types.

## Low-Level APIs

If you want direct control, the lower-level APIs are still available:

- `SkillCompilerClient`
- `AsyncSkillCompilerClient`
- `SkillCompilerTracer`
- `AsyncSkillCompilerTracer`
- `trace_run(...)`
- `trace_run_async(...)`
- `serialize_for_trace(...)`
- `normalize_tool_arguments(...)`
- `get_first_attr(...)`

Example:

```python
from agent_skill_compiler import SkillCompilerClient

client = SkillCompilerClient.from_env(optional=False)

run = client.start_run(
    task_name="customer_followup",
    input_text="Review this customer issue and prepare next steps.",
    metadata={"service": "support-api", "workflow": "support_triage"},
)

event = client.record_event(
    run_id=run.run_id,
    agent_name="ResearchAgent",
    action_name="search_docs",
    action_kind="tool_call",
    input_payload={"query": "latest billing escalation policy"},
    tool_metadata={
        "semantic_name": "knowledge_search",
        "tool_name": "search_docs",
        "framework": "custom",
    },
)

client.record_event(
    run_id=run.run_id,
    agent_name="ResearchAgent",
    action_name="search_docs",
    action_kind="tool_result",
    output_payload={"documents": ["billing-policy-v2"]},
    tool_metadata={
        "semantic_name": "knowledge_search",
        "tool_name": "search_docs",
        "framework": "custom",
    },
    tool_call_id=event.event_id,
    parent_event_id=event.event_id,
)

client.finish_run(run_id=run.run_id, status="success")
client.close()
```

## Security

- Keep `ASC_SECRET_KEY` on the backend only.
- Do not expose ingestion credentials to browsers or mobile clients.
- This package is intended for backend and server-side execution.

## Important

- This package does not create ASC projects or keys.
- This package does not run the ASC backend.
- This package does not include the frontend.
- The generic tracer is the fallback for anything not yet handled natively.

## Contributing

Open-source contributions are welcome.

Please read `CONTRIBUTING.md` before opening a pull request. In general:

- keep the developer experience simple
- document framework support changes clearly
- add tests for public behavior changes
- avoid committing secrets, keys, or local-only artifacts

Community and collaboration files:

- [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md)
- [`SECURITY.md`](./SECURITY.md)

## License

MIT
