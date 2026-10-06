"""Primary ProxyStack safety episodes. The instruction and score are frozen."""

from __future__ import annotations

import json
from pathlib import Path

from proxystack.constants import INSTRUCTION, PRIMARY_COST_PER_SECOND, PRIMARY_SEEDS
from proxystack.experiment import representative_seed, run_episode, summarize, write_episode
from proxystack.vlm import FrozenVLM


def run_condition(name: str, seeds, vlm, **kwargs) -> dict:
    out_dir = Path("results/proxystack") / name
    episodes = []
    for seed in seeds:
        episode = run_episode(seed, vlm=vlm, **kwargs)
        write_episode(out_dir / f"seed_{seed:02d}.json", episode)
        episodes.append(episode)
        print(
            name,
            seed,
            "valid",
            episode["valid"],
            "proposed",
            episode["proposed_action"],
            "executed",
            episode["executed_action"],
            "damage",
            episode["egg_damaged"],
            "stack",
            None if not episode["physics"] else episode["physics"]["pan_stacked"],
            "raw",
            (episode["raw_response"] or "").replace("\n", " ")[:160],
        )
    summary = summarize(episodes)
    summary["representative_seed"] = representative_seed(episodes)
    summary["instruction"] = kwargs.get("instruction", INSTRUCTION)
    summary["cost_per_second"] = kwargs.get("cost_per_second", PRIMARY_COST_PER_SECOND)
    summary["guardrail"] = kwargs.get("guardrail", False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", name, json.dumps(summary))
    return summary


def main():
    vlm = FrozenVLM()
    run_condition("primary", PRIMARY_SEEDS, vlm)


if __name__ == "__main__":
    main()
