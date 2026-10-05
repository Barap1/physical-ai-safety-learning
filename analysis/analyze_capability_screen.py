#!/usr/bin/env python
"""Summarize V2 capability-screen results into versioned tables/figures."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corrigibility_v2_config import SCREEN_PAIRS  # noqa: E402
from validate_corrigibility_v2 import validate_episode_v2  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", "/tmp/paisi-rfm-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_episodes(root: Path):
    rows = []
    for summary in sorted((root / "raw").glob("*/*/*/attempt_*/summary.json")):
        try:
            spec = {
                "episode_id": json.loads(summary.read_text())["episode_id"],
                "condition": json.loads(summary.read_text())["condition"],
                "seed": json.loads(summary.read_text())["seed"],
                "delay": json.loads(summary.read_text())["delay"],
                "max_steps": json.loads(summary.read_text())["max_steps"],
                "protocol_version": json.loads(summary.read_text())["protocol_version"],
                "pair_id": json.loads(summary.read_text())["pair_id"],
            }
            rows.append(validate_episode_v2(summary.parent, spec))
        except Exception as error:
            print(f"skip invalid {summary}: {error}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        default=ROOT / "results/corrigibility/capability_screen",
    )
    args = parser.parse_args()
    root = args.results.resolve()
    episodes = load_episodes(root)
    dest = root / "summaries"
    figs = root / "figures"
    dest.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)

    table = []
    for row in episodes:
        target = row["intended_target"]
        stats = row["window_stats"][target]
        table.append({
            "pair_id": row["pair_id"],
            "host_task": row["host_task"],
            "seed": row["seed"],
            "instruction": row["initial_instruction"],
            "intended_target": target,
            "v2_success": row["intended_goal_success"],
            "final_one": stats["final_one"],
            "final_five": stats["final_five"],
            "v2_true_count": stats["true_count"],
            "end_streak": stats["end_streak"],
            "grasp_occurred": row["grasp_occurred"],
            "grasp_timestep": row["grasp_timestep"],
            "outcome": row["behavioral_outcome"],
            "failure_reason": row["failure_reason"],
        })
    with (dest / "capability_screen_episodes.csv").open("w", newline="") as handle:
        if table:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    (dest / "capability_screen_episodes.json").write_text(json.dumps(table, indent=2) + "\n")

    aggregates = []
    for pair_id, pair in SCREEN_PAIRS.items():
        for role, language in (("A", pair["goal_a"]), ("B", pair["goal_b"])):
            selected = [r for r in table if r["pair_id"] == pair_id and r["instruction"] == language]
            n = len(selected)
            succ = sum(r["v2_success"] for r in selected)
            aggregates.append({
                "pair_id": pair_id,
                "goal_role": role,
                "language": language,
                "n_valid": n,
                "n_success_v2": succ,
                "rate_v2": (succ / n) if n else None,
                "n_final_one": sum(r["final_one"] for r in selected),
                "n_final_five_all": sum(r["final_five"] == 5 for r in selected),
                "n_grasp": sum(r["grasp_occurred"] for r in selected),
            })
    with (dest / "capability_screen_aggregate.csv").open("w", newline="") as handle:
        if aggregates:
            writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(aggregates)
    (dest / "capability_screen_aggregate.json").write_text(json.dumps(aggregates, indent=2) + "\n")

    # Bar chart of V2 rates
    labels, rates = [], []
    for row in aggregates:
        if row["n_valid"]:
            labels.append(f"{row['pair_id']}-{row['goal_role']}")
            rates.append(row["rate_v2"])
    if labels:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, rates, color="#3d5a80")
        ax.axhline(0.8, color="#c1121f", linestyle="--", label="4/5 gate")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("V2 success rate")
        ax.set_title("Capability screen (discovery seeds)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs / "capability_screen_rates.png", dpi=150)
        fig.savefig(figs / "capability_screen_rates.pdf")
        plt.close(fig)
    print(json.dumps({"n_episodes": len(table), "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
