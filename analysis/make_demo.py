#!/usr/bin/env python
"""Small representative figure from actual video frames, not generated imagery."""
import argparse
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from validate_corrigibility import validate_episode
import imageio.v2 as imageio
os.environ.setdefault("MPLCONFIGDIR", "/tmp/paisi-rfm-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=ROOT / "results/corrigibility/raw/condition_A/condition_A_delay5_seed0/attempt_001")
    parser.add_argument("--output", type=Path, default=ROOT / "results/corrigibility/demo/representative_switch.png")
    args = parser.parse_args()
    summary = validate_episode(args.episode)
    if not summary["override_occurred"]:
        raise ValueError("Demo requires a triggered intervention")
    reader = imageio.get_reader(args.episode / "rollout.mp4")
    try:
        steps = [max(0, summary["override_timestep"] - 10), summary["override_timestep"], summary["steps"] - 1]
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        for ax, step, label in zip(axes, steps, ("Original goal", "Operator override", "Final state")):
            ax.imshow(reader.get_data(step))
            ax.set_title(f"{label} · step {step}", fontsize=11)
            ax.axis("off")
        fig.suptitle(f"Actual SmolVLA rollout · {summary['condition']} · seed {summary['seed']} · "
                     f"plate {summary['final_bowl_to_plate_distance']:.3f} m / "
                     f"stove {summary['final_bowl_to_stove_distance']:.3f} m", fontsize=11)
        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        plt.close(fig)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
