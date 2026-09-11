"""Local semantic-validation policy for S2G structured outputs."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .output_policy import CONTROLLER_STAGES


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "config/s2g_local_validation_policy.json"


class LocalSemanticValidationError(ValueError):
    """A schema-valid value violates an S2G semantic invariant."""


SemanticValidator = Callable[[Any], None]


@dataclass(frozen=True)
class _ValidatorBinding:
    stage: str
    validator: SemanticValidator


_VALIDATOR: ContextVar[_ValidatorBinding | None] = ContextVar(
    "s2g_local_semantic_validator", default=None
)


@contextmanager
def local_semantic_validator(stage: str, validator: SemanticValidator) -> Iterator[None]:
    token = _VALIDATOR.set(_ValidatorBinding(stage, validator))
    try:
        yield
    finally:
        _VALIDATOR.reset(token)


@dataclass(frozen=True)
class LocalSemanticValidationPolicy:
    policy_version: str
    allowed_stages: tuple[str, ...]
    maximum_reattempts: int

    @classmethod
    def load(cls, path: Path = POLICY_PATH) -> "LocalSemanticValidationPolicy":
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != "s2g-local-validation-policy-v1":
            raise ValueError("Unexpected S2G local-validation policy schema")
        if value.get("status") != "ACTIVE":
            raise ValueError("S2G local-validation policy is not active")
        if value.get("maximum_identical_reattempts") != 1:
            raise ValueError("Exactly one local-semantic re-attempt is required")
        stages = tuple(value.get("applies_to_stages", ()))
        if stages != CONTROLLER_STAGES:
            raise ValueError("Local-validation policy stages differ from S2G runtime stages")
        if value.get("silent_output_repair") != "PROHIBITED":
            raise ValueError("Silent output repair must be prohibited")
        return cls(str(value["policy_version"]), stages, 1)
