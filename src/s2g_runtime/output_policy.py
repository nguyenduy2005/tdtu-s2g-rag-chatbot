"""Output-token policy for the S2G controller.

This module is deliberately independent of benchmark records, retrieval
results, controller prompts, and JSON schemas.  It only maps the qualified
legacy controller ceiling to one fixed execution-wide ceiling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .completion_policy import ControllerIncompletePolicy
from .openai_adapter import OpenAICaller


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "config/s2g_output_policy.json"

CONTROLLER_STAGES = (
    "s2g_judge",
    "s2g_sentence_selector",
)


@dataclass(frozen=True)
class MaxOutputTokenPolicy:
    """One ceiling for every controller stage in the versioned experiment."""

    policy_version: str
    input_ceiling: int
    effective_ceiling: int
    maximum_completion_reattempts: int
    allowed_stages: tuple[str, ...]

    @classmethod
    def load(cls, path: Path = POLICY_PATH) -> "MaxOutputTokenPolicy":
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != "s2g-output-policy-v1":
            raise ValueError("Unexpected S2G output policy schema")
        if value.get("status") != "ACTIVE":
            raise ValueError("S2G output policy is not active")
        if value.get("selection_basis") != "fixed_global_power_of_two_step_above_observed_ceiling":
            raise ValueError("Unexpected output-token ceiling selection basis")
        source = int(value.get("legacy_controller_ceiling", 0))
        effective = int(value.get("new_controller_ceiling", 0))
        if source != 4096 or effective != 8192:
            raise ValueError("S2G ceiling transition must be 4096 -> 8192")
        if effective > int(value.get("documented_model_max_output_tokens", 0)):
            raise ValueError("Policy ceiling exceeds the documented model maximum")
        attempts = int(value.get("maximum_completion_reattempts", -1))
        if attempts != 1:
            raise ValueError("Exactly one identical completion re-attempt is required")
        stages = tuple(value.get("applies_to_stages", ()))
        if stages != CONTROLLER_STAGES:
            raise ValueError("Output policy must cover both S2G controller stages")
        return cls(str(value["policy_version"]), source, effective, attempts, stages)

    def resolve(self, stage: str, requested_ceiling: int) -> int:
        if stage not in self.allowed_stages:
            raise ValueError(f"Controller stage is outside max-output policy: {stage}")
        if requested_ceiling != self.input_ceiling:
            raise ValueError(
                f"Unexpected controller ceiling {requested_ceiling}; expected qualified legacy value {self.input_ceiling}"
            )
        return self.effective_ceiling

    def completion_policy(self) -> ControllerIncompletePolicy:
        """A fresh-run policy: no historical incident attempt is imported."""

        return ControllerIncompletePolicy(
            allowed_stages=self.allowed_stages,
            maximum_completion_reattempts=self.maximum_completion_reattempts,
        )


class PolicyBoundOpenAICaller(OpenAICaller):
    """Apply the global ceiling before the unchanged shared caller executes."""

    def __init__(self, *args: Any, max_output_policy: MaxOutputTokenPolicy, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_output_policy = max_output_policy

    def request(
        self,
        *,
        stage: str,
        prompt: str,
        template_sha256: str,
        parser: Callable[[str], Any],
        max_output_tokens: int,
        schema: dict[str, Any],
        source_text_sha256: tuple[str, ...] = (),
    ):
        effective_ceiling = self.max_output_policy.resolve(stage, max_output_tokens)
        return super().request(
            stage=stage,
            prompt=prompt,
            template_sha256=template_sha256,
            parser=parser,
            max_output_tokens=effective_ceiling,
            schema=schema,
            source_text_sha256=source_text_sha256,
        )
