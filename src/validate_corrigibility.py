"""Validate complete episode artifacts, including queue/inference evidence."""
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from corrigibility_metrics import episode_metrics


def validate_episode(directory: Path, spec: dict | None = None) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    if spec is not None:
        for key, value in spec.items():
            assert summary[key] == value, f"Protocol/spec mismatch: {key}"
    raw = (directory / "telemetry.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == summary["telemetry_sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == summary["steps"] == summary["max_steps"]
    required = {"episode_id", "seed", "simulation_time", "initial_instruction", "active_instruction",
                "override_text", "override_condition", "override_timing_setting", "grasp_occurred",
                "grasp_timestep", "override_occurred", "override_timestep", "action",
                "end_effector_position", "gripper_state", "bowl_position", "plate_position",
                "stove_position", "bowl_to_plate_distance", "bowl_to_stove_distance", "reward", "done",
                "action_inference_phase", "generation_epoch", "inference_id"}
    for t, row in enumerate(rows):
        assert required <= row.keys()
        assert row["timestep"] == t and row["episode_id"] == summary["episode_id"]
        assert row["seed"] == summary["seed"]
        assert len(row["action"]) == 7 and np.isfinite(row["action"]).all()
        assert row["generation_epoch"] == int(row["override_occurred"]), "Stale generation"
        assert row["action_inference_phase"] == ("after_override" if row["override_occurred"] else "before_override")
        assert row["active_instruction"] == (summary["override_text"] if row["override_occurred"] else summary["initial_instruction"])
        for target in ("stove", "plate"):
            distance = np.linalg.norm(np.array(row["bowl_position"]) - row[f"{target}_position"])
            assert abs(distance - row[f"bowl_to_{target}_distance"]) < 1e-9
        if t:
            assert row["simulation_time"] > rows[t - 1]["simulation_time"]
    triggers = [r for r in rows if r["override_this_step"]]
    if summary["override_occurred"]:
        assert len(triggers) == len(summary["queue_audit"]) == 1
        audit = summary["queue_audit"][0]
        assert audit["deque_replaced"] and audit["simulation_state_unchanged"]
        assert audit["fresh_observation"] and audit["fresh_inference"]
        assert audit["queued_actions_after_reset"] == 0
        assert audit["inference_count_after"] == audit["inference_count_before"] + 1
        assert triggers[0]["inference_this_step"]
        assert summary["override_timestep"] == summary["grasp_timestep"] + summary["delay"]
    else:
        assert not triggers and not summary["queue_audit"]
    recomputed = episode_metrics(rows, summary["override_state"])
    assert all(summary[k] == v for k, v in recomputed.items()), "Metric mismatch"
    reader = imageio.get_reader(directory / "rollout.mp4")
    try:
        assert reader.count_frames() == len(rows)
        assert reader.get_meta_data()["fps"] == 20
        assert reader.get_data(0).shape == (800, 640, 3)
        reader.get_data(len(rows) - 1)
    finally:
        reader.close()
    return summary
