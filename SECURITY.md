# Security Policy

## Supported Versions

Security fixes are applied to the latest released version of `agent-skill-compiler`.

If you report an issue against an older version, the fix may be shipped only in the latest release.

## Reporting A Vulnerability

Please do not open a public GitHub issue for a suspected security vulnerability.

Instead:

1. Email the maintainer privately.
2. Include a clear description of the issue.
3. Include reproduction steps, affected versions, and impact if known.
4. Share whether the issue involves credential exposure, unauthorized ingestion, dependency risk, or data leakage.

## Scope

This repository contains the Python SDK only.

If the issue affects the wider product platform, backend, frontend, or deployment setup, also review the main platform repository:

- [AI-Skills-Compiler](https://github.com/dlamaro96/AI-Skills-Compiler)

## Security Expectations

- Never commit real ASC keys or secrets.
- Keep `ASC_SECRET_KEY` on backend systems only.
- Do not expose SDK ingestion credentials in browser or mobile clients.
- Prefer trusted publishing for PyPI releases.
