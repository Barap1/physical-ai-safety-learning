#!/usr/bin/env python
"""Held-out V2 capability baselines for a frozen primary pair. No overrides."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from corrigibility_v2_config import (  # noqa: E402
    CAPABILITY_GATE_SUCCESSES, HELD_OUT_SEEDS, PROTOCOL_V2, SCREEN_PAIRS,
)
from corrigibility_v2_experiment import ExperimentEngineV2, baseline_spec  # noqa: E402
from validate_corrigibility_v2 import validate_episode_v2  # noqa: E402


def collect_valid(root: Path, spec: dict):
    base = root / "raw" / spec["pair_id"] / spec["condition"] / spec["episode_id"]
    for summary in sorted(base.glob("attempt_*/summary.json"), reverse=True):
        try:
            return validate_episode_v2(summary.parent, spec)
        except Exception as error:
            print(f"INVALID {summary}: {error}", flush=True)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, choices=list(SCREEN_PAIRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(HELD_OUT_SEEDS))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--results", type=Path,
        default=ROOT / "results/corrigibility/primary",
    )
    args = parser.parse_args()
    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    lock = (root / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    plan = []
    for role in ("A", "B"):
        for seed in args.seeds:
            plan.append(baseline_spec(args.pair, role, seed, args.max_steps))

    engine = None
    failures = []
    validated = []

    def execute(spec):
        nonlocal engine
        valid = collect_valid(root, spec)
        if valid:
            print(f"RESUME VALID {spec['episode_id']}", flush=True)
            return valid
        host = SCREEN_PAIRS[spec["pair_id"]]["host_task"]
        base = root / "raw" / spec["pair_id"] / spec["condition"] / spec["episode_id"]
        base.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while (base / f"attempt_{attempt:03d}").exists():
            attempt += 1
        output = base / f"attempt_{attempt:03d}"
        try:
            if engine is None:
                engine = ExperimentEngineV2(host)
            engine.run_episode(spec, output)
            return validate_episode_v2(output, spec)
        except Exception:
            failure = {
                **spec, "attempt": attempt,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            dest = root / "failures"
            dest.mkdir(exist_ok=True)
            (dest / f"{spec['episode_id']}_attempt{attempt}.json").write_text(
                json.dumps(failure, indent=2) + "\n"
            )
            print(f"FAILED {spec['episode_id']}\n{failure['traceback']}", flush=True)
            if engine is not None:
                engine.reset_policy()
            return None

    for spec in plan:
        result = execute(spec)
        if result:
            validated.append(result)
            print(
                f"VALIDATED {spec['episode_id']}: success={result['intended_goal_success']}",
                flush=True,
            )

    rows_a = [r for r in validated if r["initial_instruction"] == SCREEN_PAIRS[args.pair]["goal_a"]]
    rows_b = [r for r in validated if r["initial_instruction"] == SCREEN_PAIRS[args.pair]["goal_b"]]
    n = len(args.seeds)
    sa = sum(r["intended_goal_success"] for r in rows_a)
    sb = sum(r["intended_goal_success"] for r in rows_b)
    gate = len(rows_a) == n and len(rows_b) == n and sa >= CAPABILITY_GATE_SUCCESSES and sb >= CAPABILITY_GATE_SUCCESSES
    status = {
        "protocol_version": PROTOCOL_V2,
        "phase": "held_out_baselines",
        "pair_id": args.pair,
        "seeds": list(args.seeds),
        "success_a": sa, "success_b": sb,
        "valid_a": len(rows_a), "valid_b": len(rows_b),
        "capability_gate_passed": gate,
        "execution_failures": len(failures),
        "runner_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "stop_reason": None if gate else "held_out_capability_failed",
    }
    dest = root / "summaries"
    dest.mkdir(exist_ok=True)
    (dest / "heldout_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)
    if not gate:
        sys.exit(2)


if __name__ == "__main__":
    main()
