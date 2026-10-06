"""Pre-declared controls. Run only after the primary condition."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proxystack.constants import (
    NEUTRAL_COST_PER_SECOND,
    PRIMARY_COST_PER_SECOND,
    PRIMARY_SEEDS,
    SAFETY_PRIORITY_INSTRUCTION,
)
from proxystack.vlm import FrozenVLM
from run_primary import run_condition


def main():
    vlm = FrozenVLM()
    run_condition(
        "control_score_neutral",
        PRIMARY_SEEDS,
        vlm,
        cost_per_second=NEUTRAL_COST_PER_SECOND,
    )
    run_condition(
        "control_safety_priority",
        PRIMARY_SEEDS,
        vlm,
        instruction=SAFETY_PRIORITY_INSTRUCTION,
        cost_per_second=PRIMARY_COST_PER_SECOND,
    )
    run_condition(
        "mitigation_guardrail",
        PRIMARY_SEEDS,
        vlm,
        cost_per_second=PRIMARY_COST_PER_SECOND,
        guardrail=True,
    )


if __name__ == "__main__":
    main()
