"""Typed project settings loaded from YAML."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Settings:
    values: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Settings":
        with Path(path).open(encoding="utf-8") as stream:
            return cls(yaml.safe_load(stream))

    def section(self, name: str) -> dict[str, Any]:
        return self.values[name]
