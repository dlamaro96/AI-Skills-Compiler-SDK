"""Configuration helpers for Skill Compiler SDK clients."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


@dataclass(frozen=True)
class SkillCompilerConnection:
    """Resolved connection settings for the Skill Compiler SDK."""

    base_url: str
    public_key: str
    secret_key: str

    @property
    def is_configured(self) -> bool:
        """Return whether the connection has all required fields."""

        return bool(self.base_url and self.public_key and self.secret_key)

    @classmethod
    def from_env(cls) -> "SkillCompilerConnection":
        """Resolve connection settings from supported environment variable names."""

        return cls(
            base_url=_first_env(
                "ASC_BASE_URL",
                "SKILL_COMPILER_BASE_URL",
                "SKILL_COMPILER_HOST",
            ),
            public_key=_first_env(
                "ASC_PUBLIC_KEY",
                "SKILL_COMPILER_PUBLIC_KEY",
            ),
            secret_key=_first_env(
                "ASC_SECRET_KEY",
                "SKILL_COMPILER_SECRET_KEY",
            ),
        )
