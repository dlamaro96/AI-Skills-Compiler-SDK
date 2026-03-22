# Contributing

Contributions are welcome.

## Scope

This repository contains the Python SDK published to PyPI as `agent-skill-compiler`.

The full product platform, including the backend and frontend, lives in:

- [AI-Skills-Compiler](https://github.com/dlamaro96/AI-Skills-Compiler)

## Development Setup

1. Create a virtual environment.
2. Install the package in editable mode with the extras you need.
3. Run tests before opening a pull request.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[openai-agents,agno,langchain]"
PYTHONPATH=src python -m unittest discover -s tests
```

## Contribution Guidelines

- Keep the public API simple for application developers.
- Prefer framework-native integrations when the framework exposes callbacks, middleware, or event streams.
- Preserve the generic tracer as the fallback for unsupported runtimes.
- Update `README.md` whenever framework support changes.
- Add or update tests for public behavior changes.

## Pull Requests

Please include:

- what changed
- why it changed
- how it was tested
- any compatibility notes for developers using the SDK

## Versioning

Use semantic versioning:

- patch: bug fixes and documentation-only changes
- minor: new backward-compatible integrations or capabilities
- major: breaking API changes

## Security

- Never commit real ASC keys or secrets.
- Keep `ASC_SECRET_KEY` server-side only.
