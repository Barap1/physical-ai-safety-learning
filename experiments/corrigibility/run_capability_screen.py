#!/usr/bin/env python
"""Sequential V2 capability screen. No mid-execution overrides. Gate cannot be bypassed."""
from __future__ import annotations

import argparse
import csv
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
    CAPABILITY_GATE_SUCCESSES, DISCOVERY_SEEDS, PROTOCOL_V2, SCREEN_PAIRS,
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


def write_progress(root: Path, rows: list[dict], status: dict):
    dest = root / "summaries"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "capability_screen_status.json").write_text(json.dumps(status, indent=2) + "\n")
    if not rows:
        return
    fields = [
        "pair_id", "goal_role", "condition", "seed", "episode_id", "host_task",
        "initial_instruction", "intended_target", "valid", "grasp_occurred",
        "grasp_timestep", "intended_goal_success", "behavioral_outcome",
        "failure_reason", "final_one", "final_five", "v2_true_count",
        "end_streak", "metric_disagreement_v1_style",
    ]
    table = []
    for row in rows:
        target = row["intended_target"]
        stats = row["window_stats"][target]
        table.append({
            "pair_id": row["pair_id"],
            "goal_role": row.get("goal_role"),
            "condition": row["condition"],
            "seed": row["seed"],
            "episode_id": row["episode_id"],
            "host_task": row["host_task"],
            "initial_instruction": row["initial_instruction"],
            "intended_target": target,
            "valid": True,
            "grasp_occurred": row["grasp_occurred"],
            "grasp_timestep": row["grasp_timestep"],
            "intended_goal_success": row["intended_goal_success"],
            "behavioral_outcome": row["behavioral_outcome"],
            "failure_reason": row["failure_reason"],
            "final_one": stats["final_one"],
            "final_five": stats["final_five"],
            "v2_true_count": stats["true_count"],
            "end_streak": stats["end_streak"],
            "metric_disagreement_v1_style": row["metric_disagreement_v1_style"],
        })
    with (dest / "capability_screen_episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    (dest / "capability_screen_episodes.json").write_text(json.dumps(table, indent=2) + "\n")


def rank_pairs(pair_stats: dict) -> list[dict]:
    ranked = []
    for pair_id, stats in pair_stats.items():
        if not stats["qualifies"]:
            continue
        ranked.append({
            "pair_id": pair_id,
            "min_success_rate": stats["min_success_rate"],
            "success_a": stats["success_a"],
            "success_b": stats["success_b"],
            "predicate_family": "On/On",
            "rationale": SCREEN_PAIRS[pair_id]["rationale"],
        })
    # PROTOCOL_V2 ranking: highest min success, then simpler / clearer pairs, then ID.
    order = {"P1": 0, "P2": 1, "P4": 2, "P3": 3}
    ranked.sort(key=lambda r: (-r["min_success_rate"], order.get(r["pair_id"], 9), r["pair_id"]))
    return ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=list(SCREEN_PAIRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DISCOVERY_SEEDS))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--results", type=Path,
        default=ROOT / "results/corrigibility/capability_screen",
    )
    parser.add_argument("--dry-plan", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 100:
        parser.error("Protocol requires at least 100 actions")

    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    lock = (root / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    plan = []
    for pair_id in args.pairs:
        if pair_id not in SCREEN_PAIRS:
            parser.error(f"Unknown pair {pair_id}")
        for role in ("A", "B"):
            for seed in args.seeds:
                plan.append(baseline_spec(pair_id, role, seed, args.max_steps))
    if args.dry_plan:
        print(json.dumps({"n": len(plan), "plan": plan}, indent=2))
        return

    engine = None
    current_host = None
    failures = []
    validated = []

    def execute(spec):
        nonlocal engine, current_host
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
                current_host = host
            elif host != current_host:
                engine.set_host(host)
                current_host = host
            engine.run_episode(spec, output)
            result = validate_episode_v2(output, spec)
            print(
                f"VALIDATED {spec['episode_id']}: "
                f"success={result['intended_goal_success']} outcome={result['behavioral_outcome']}",
                flush=True,
            )
            return result
        except Exception:
            failure = {
                **spec,
                "attempt": attempt,
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

    status = {
        "protocol_version": PROTOCOL_V2,
        "phase": "capability_screen_discovery",
        "max_steps": args.max_steps,
        "seeds": list(args.seeds),
        "pairs": list(args.pairs),
        "gate": f">= {CAPABILITY_GATE_SUCCESSES}/{len(args.seeds)} V2 successes for BOTH goals",
        "runner_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    for spec in plan:
        result = execute(spec)
        if result:
            validated.append(result)
        write_progress(root, validated, {**status, "completed": len(validated), "failures": len(failures)})

    pair_stats = {}
    for pair_id in args.pairs:
        rows_a = [r for r in validated if r["pair_id"] == pair_id and r.get("goal_role") == "A"]
        rows_b = [r for r in validated if r["pair_id"] == pair_id and r.get("goal_role") == "B"]
        # goal_role may be missing in older resumes; infer from condition.
        if not rows_a:
            rows_a = [r for r in validated if r["pair_id"] == pair_id and r["condition"].endswith(f"a_{pair_id.lower()}")]
        if not rows_b:
            rows_b = [r for r in validated if r["pair_id"] == pair_id and r["condition"].endswith(f"b_{pair_id.lower()}")]
        # Simpler: match by instruction
        rows_a = [r for r in validated if r["pair_id"] == pair_id and r["initial_instruction"] == SCREEN_PAIRS[pair_id]["goal_a"]]
        rows_b = [r for r in validated if r["pair_id"] == pair_id and r["initial_instruction"] == SCREEN_PAIRS[pair_id]["goal_b"]]
        sa = sum(r["intended_goal_success"] for r in rows_a)
        sb = sum(r["intended_goal_success"] for r in rows_b)
        n = len(args.seeds)
        qualifies = len(rows_a) == n and len(rows_b) == n and sa >= CAPABILITY_GATE_SUCCESSES and sb >= CAPABILITY_GATE_SUCCESSES
        pair_stats[pair_id] = {
            "valid_a": len(rows_a), "valid_b": len(rows_b),
            "success_a": sa, "success_b": sb,
            "rate_a": sa / n, "rate_b": sb / n,
            "min_success_rate": min(sa, sb) / n,
            "qualifies": qualifies,
        }
        print(f"PAIR {pair_id}: A {sa}/{n}, B {sb}/{n}, qualifies={qualifies}", flush=True)

    ranked = rank_pairs(pair_stats)
    status.update({
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_plan": len(plan),
        "n_validated": len(validated),
        "execution_failures": len(failures),
        "pair_stats": pair_stats,
        "qualified_ranked": ranked,
        "selected_pair": ranked[0]["pair_id"] if ranked else None,
        "stop_reason": None if ranked else "no_pair_met_capability_gate",
    })
    write_progress(root, validated, status)
    (root / "summaries" / "capability_screen_ranking.json").write_text(
        json.dumps({"pair_stats": pair_stats, "ranked": ranked}, indent=2) + "\n"
    )
    print(json.dumps(status, indent=2), flush=True)
    if not ranked:
        sys.exit(2)


if __name__ == "__main__":
    main()
