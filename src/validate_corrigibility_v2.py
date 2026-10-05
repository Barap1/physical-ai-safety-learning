"""Validate V2 episode artifacts without mutating them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from corrigibility_v2_metrics import episode_metrics_v2


def validate_episode_v2(directory: Path, spec: dict | None = None) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    if spec is not None:
        for key in ("episode_id", "condition", "seed", "delay", "max_steps", "protocol_version", "pair_id"):
            assert summary[key] == spec[key], f"Protocol/spec mismatch: {key}"
    raw = (directory / "telemetry.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == summary["telemetry_sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == summary["steps"] == summary["max_steps"]
    required = {
        "episode_id", "seed", "pair_id", "suite", "host_task", "timestep",
        "simulation_time", "initial_instruction", "active_instruction",
        "condition", "grasp_occurred", "grasp_timestep", "override_occurred",
        "override_timestep", "action", "end_effector_position", "gripper_state",
        "object_position", "target_predicates", "reward", "done",
        "generation_epoch", "inference_id", "instruction_generation_epoch",
    }
    for t, row in enumerate(rows):
        assert required <= row.keys()
        assert row["timestep"] == t
        assert row["episode_id"] == summary["episode_id"]
        assert len(row["action"]) == 7 and np.isfinite(row["action"]).all()
        assert row["generation_epoch"] == int(row["override_occurred"])
        expected_instruction = (
            summary["override_text"] if row["override_occurred"] else summary["initial_instruction"]
        )
        assert row["active_instruction"] == expected_instruction
        if t:
            assert row["simulation_time"] > rows[t - 1]["simulation_time"]
    triggers = [r for r in rows if r["override_this_step"]]
    if summary["override_occurred"]:
        assert len(triggers) == len(summary["queue_audit"]) == 1
        audit = summary["queue_audit"][0]
        assert audit["deque_replaced"] and audit["simulation_state_unchanged"]
        assert audit["fresh_observation"] and audit["fresh_inference"]
        assert audit["queued_actions_after_reset"] == 0
    else:
        assert not triggers and not summary["queue_audit"]
    recomputed = episode_metrics_v2(rows, summary["intended_target"], summary["override_state"])
    for key, value in recomputed.items():
        assert summary[key] == value, f"Metric mismatch for {key}"
    reader = imageio.get_reader(directory / "rollout.mp4")
    try:
        assert reader.count_frames() == len(rows)
        assert reader.get_meta_data()["fps"] == 20
        frame = reader.get_data(0)
        assert frame.shape[0] >= 800 and frame.shape[1] == 640
        reader.get_data(len(rows) - 1)
    finally:
        reader.close()
    return summary
