"""Retry policy for incomplete S2G provider responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ControllerIncompletePolicy:
    allowed_stages: tuple[str, ...]
    maximum_completion_reattempts: int = 1

    def historical_attempts(self, stage: str) -> tuple[dict[str, Any], ...]:
        """The product always starts a controller call fresh; no old attempt is replayed."""

        if stage not in self.allowed_stages:
            raise ValueError(f"Controller stage is outside S2G policy: {stage}")
        return ()
