"""Generalized V2 corrigibility engine for multi-goal LIBERO-Goal hosts.

Preserves v1 engine/files. Capability screening uses baselines only (no override).
Override support is included for the later primary freeze, with the same audit
invariants as v1.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import random
import textwrap
import time

from huggingface_hub import hf_hub_download

from corrigibility_rollout import (
    MODEL_ID, LiberoConfig, OffScreenRenderEnv, SmolVLAConfig, SmolVLAPolicy,
    benchmark, get_libero_dummy_action, get_libero_path, get_task_init_states,
    imageio, make_env_pre_post_processors, make_pre_post_processors,
    np, policy_observation, torch, Image, ImageDraw,
)
from corrigibility_experiment import AuditedSmolVLA
from corrigibility_v2_config import (
    GOAL_SPECS, OBJECT_SPECS, PROTOCOL_V2, SCREEN_PAIRS, SUITE, TARGET_SPECS, goal_keys,
)
from corrigibility_v2_metrics import episode_metrics_v2


def annotated_frame(obs, spec, instruction, timestep, grasp, override, predicates):
    scene = Image.fromarray(obs["agentview_image"][::-1, ::-1].copy()).resize((640, 640))
    frame = Image.new("RGB", (640, 820), "black")
    frame.paste(scene, (0, 180))
    draw = ImageDraw.Draw(frame)
    heading = "OPERATOR OVERRIDE" if override is not None else "ORIGINAL GOAL"
    color = "#ffb347" if override is not None else "white"
    pred_txt = " ".join(f"{k}={int(v)}" for k, v in predicates.items())
    lines = [
        heading + ":",
        *textwrap.wrap(instruction, width=85),
        f"pair={spec.get('pair_id')} seed={spec['seed']} condition={spec['condition']} t={timestep}",
        f"grasp={grasp is not None} after={grasp} override={override is not None} at={override}",
        f"predicates: {pred_txt}",
    ]
    for i, line in enumerate(lines):
        draw.text((12, 8 + i * 20), line, fill=color)
    if timestep == override:
        draw.rectangle((0, 0, 639, 819), outline=color, width=6)
    return np.asarray(frame)


class ExperimentEngineV2:
    def __init__(self, host_task: str):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.set_num_threads(4)
        config = SmolVLAConfig.from_pretrained(MODEL_ID)
        config.device = "cuda"
        self.policy = AuditedSmolVLA.from_pretrained(MODEL_ID, config=config).eval()
        self.policy.inference_count = 0
        self.policy.instruction_epoch = 0
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=config, pretrained_path=MODEL_ID,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(
            LiberoConfig(task=SUITE), config,
        )
        self.suite = benchmark.get_benchmark_dict()[SUITE]()
        self.set_host(host_task)
        self.provenance = {
            "model_id": MODEL_ID,
            "suite": SUITE,
            "cuda_device": torch.cuda.get_device_name(),
            "torch_threads": torch.get_num_threads(),
            "package_versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "lerobot", "robosuite", "mujoco", "transformers")
            },
            "policy_chunk_size": config.chunk_size,
            "policy_n_action_steps": config.n_action_steps,
            "model_revision": Path(
                hf_hub_download(MODEL_ID, "config.json", local_files_only=True)
            ).parent.name,
            "source_sha256": {
                "smolvla": hashlib.sha256(Path(inspect.getfile(SmolVLAPolicy)).read_bytes()).hexdigest(),
                "experiment_v2": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "metrics_v2": hashlib.sha256(
                    (Path(__file__).parent / "corrigibility_v2_metrics.py").read_bytes()
                ).hexdigest(),
            },
        }

    def set_host(self, host_task: str):
        self.host_task_name = host_task
        self.task_id = next(
            i for i, task in enumerate(self.suite.tasks) if task.name == host_task
        )
        self.task = self.suite.get_task(self.task_id)
        self.bddl = Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file
        self.init_states = get_task_init_states(self.suite, self.task_id)

    def reset_policy(self):
        self.policy.reset()
        for pipeline in (self.pre, self.post, self.env_pre, self.env_post):
            pipeline.reset()
        assert len(self.policy._queues["action"]) == 0

    def simulator_state(self, env, obs, object_key: str, tracked_targets: list[str]):
        domain = env.env
        states = domain.object_states_dict
        obj_name = OBJECT_SPECS[object_key]["body"]
        positions = {}
        for label, body in (
            ("object", obj_name),
            ("plate", "plate_1"),
            ("stove", "flat_stove_1"),
            ("cabinet", "wooden_cabinet_1"),
            ("wine", "wine_bottle_1"),
            ("rack", "wine_rack_1"),
        ):
            positions[label] = np.array(env.sim.data.body_xpos[domain.obj_body_id[body]])
        grasped = bool(domain._check_grasp(
            env.robots[0].gripper,
            domain.objects_dict[obj_name].contact_geoms,
        ))
        manipulated = states[OBJECT_SPECS[object_key]["state"]]
        predicates = {}
        for key in tracked_targets:
            spec = TARGET_SPECS[key]
            host = states[spec["predicate_state"]]
            predicates[key] = bool(host.check_ontop(manipulated))
        return {
            "end_effector_position": np.asarray(obs["robot0_eef_pos"]).tolist(),
            "object_position": positions["object"].tolist(),
            "plate_position": positions["plate"].tolist(),
            "stove_position": positions["stove"].tolist(),
            "cabinet_position": positions["cabinet"].tolist(),
            "wine_position": positions["wine"].tolist(),
            "rack_position": positions["rack"].tolist(),
            "gripper_state": {
                "qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist(),
                "qvel": np.asarray(obs["robot0_gripper_qvel"]).tolist(),
                "object_grasped": grasped,
                # Compatibility alias used by some overlays/tests.
                "bowl_grasped": grasped if object_key == "bowl" else False,
            },
            "target_predicates": predicates,
            "object_to_plate_distance": float(np.linalg.norm(positions["object"] - positions["plate"])),
            "object_to_stove_distance": float(np.linalg.norm(positions["object"] - positions["stove"])),
            "object_to_cabinet_distance": float(np.linalg.norm(positions["object"] - positions["cabinet"])),
            "object_to_rack_distance": float(np.linalg.norm(positions["object"] - positions["rack"])),
            # Secondary body-origin distances; stove root is offset from cook region.
            "distance_metric_note": "body_origin_euclidean_secondary_only",
        }

    def run_episode(self, spec: dict, output: Path) -> dict:
        output.mkdir(parents=True, exist_ok=False)
        pair = SCREEN_PAIRS[spec["pair_id"]]
        if pair["host_task"] != self.host_task_name:
            self.set_host(pair["host_task"])
        seed = spec["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.reset_policy()
        self.policy.instruction_epoch = 0

        initial = spec["initial_instruction"]
        override_text = spec.get("override_text")
        # Intended endpoint language for metrics.
        if spec["condition"].startswith("baseline"):
            intended_language = initial
        elif spec["condition"] == "control":
            intended_language = pair["goal_a"]
        elif spec["condition"] in (
            "positive_switch", "explicit_revocation", "emergency_revocation",
        ):
            intended_language = pair["goal_b"]
        else:
            intended_language = spec.get("intended_instruction") or initial
        intended_meta = goal_keys(intended_language)
        object_key = intended_meta["object"]
        # Always track both pair targets.
        tracked = [goal_keys(pair["goal_a"])["target"], goal_keys(pair["goal_b"])["target"]]
        # Ensure unique and stable order.
        tracked = list(dict.fromkeys(tracked))

        env = OffScreenRenderEnv(
            bddl_file_name=str(self.bddl), camera_heights=256, camera_widths=256,
            control_freq=20, ignore_done=True,
        )
        rows, audits = [], []
        instruction = initial
        grasp = override = None
        override_state = None
        try:
            env.seed(seed)
            env.reset()
            obs = env.set_init_state(self.init_states[seed % len(self.init_states)])
            for _ in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action())
            state = self.simulator_state(env, obs, object_key, tracked)
            initial_state = state
            if state["gripper_state"]["object_grasped"]:
                grasp = 0
            with (output / "telemetry.jsonl").open("x") as telemetry, imageio.get_writer(
                output / "rollout.mp4", fps=20, codec="libx264", macro_block_size=1,
            ) as video, torch.inference_mode():
                for t in range(spec["max_steps"]):
                    trigger = (
                        override_text is not None and override is None
                        and grasp is not None and t >= grasp + spec["delay"]
                    )
                    if trigger:
                        override_state = state.copy()
                        old_queue = self.policy._queues["action"]
                        sim_before = env.get_sim_state().copy()
                        count_before = self.policy.inference_count
                        self.reset_policy()
                        assert self.policy._queues["action"] is not old_queue
                        assert len(self.policy._queues["action"]) == 0
                        instruction = override_text
                        override = t
                        self.policy.instruction_epoch = 1
                        obs = env.env._get_observations(force_update=True)
                        assert np.array_equal(sim_before, env.get_sim_state())
                        audits.append({
                            "timestep": t,
                            "queued_actions_before": len(old_queue),
                            "queued_actions_after_reset": 0,
                            "deque_replaced": True,
                            "simulation_state_unchanged": True,
                            "fresh_observation": True,
                            "camera_sha256": hashlib.sha256(obs["agentview_image"].tobytes()).hexdigest(),
                            "inference_count_before": count_before,
                        })
                    before_queue = len(self.policy._queues["action"])
                    count_before_action = self.policy.inference_count
                    batch = self.pre(self.env_pre(policy_observation(obs, instruction)))
                    action = self.env_post({"action": self.post(self.policy.select_action(batch))})["action"]
                    action = action.detach().cpu().numpy()[0]
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError("Invalid action")
                    assert self.policy.generation_epoch == int(override is not None), "Stale action generation"
                    if trigger:
                        assert self.policy.inference_count == count_before_action + 1, "Fresh inference missing"
                        audits[-1].update({
                            "fresh_inference": True,
                            "inference_count_after": self.policy.inference_count,
                            "new_token_sha256": self.policy.token_digest,
                        })
                        (output / "override_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
                    obs, reward, simulator_done, _ = env.step(action)
                    state = self.simulator_state(env, obs, object_key, tracked)
                    if grasp is None and state["gripper_state"]["object_grasped"]:
                        grasp = t + 1
                    row = {
                        "episode_id": spec["episode_id"],
                        "seed": seed,
                        "pair_id": spec["pair_id"],
                        "suite": SUITE,
                        "host_task": self.host_task_name,
                        "timestep": t,
                        "simulation_time": float(env.sim.data.time),
                        "initial_instruction": initial,
                        "active_instruction": instruction,
                        "override_text": override_text,
                        "condition": spec["condition"],
                        "override_condition": spec["condition"],
                        "override_timing_setting": spec["delay"],
                        "intended_target": intended_meta["target"],
                        "manipulated_object": OBJECT_SPECS[object_key]["state"],
                        "grasp_occurred": grasp is not None,
                        "grasp_timestep": grasp,
                        "override_occurred": override is not None,
                        "override_timestep": override,
                        "override_this_step": bool(trigger),
                        "action": action.tolist(),
                        **state,
                        "reward": float(reward),
                        "done": bool(simulator_done),
                        "protocol_done": t + 1 == spec["max_steps"],
                        "action_inference_phase": "after_override" if self.policy.generation_epoch else "before_override",
                        "generation_epoch": self.policy.generation_epoch,
                        "inference_id": self.policy.inference_count,
                        "inference_timestamp": self.policy.inference_timestamp,
                        "inference_this_step": self.policy.inference_count > count_before_action,
                        "action_chunk_size": self.policy.generated_chunk_size,
                        "queued_actions_before_selection": before_queue,
                        "queued_actions_after_selection": len(self.policy._queues["action"]),
                        "language_token_sha256": self.policy.token_digest,
                        "instruction_generation_epoch": self.policy.instruction_epoch,
                    }
                    rows.append(row)
                    telemetry.write(json.dumps(row, allow_nan=False) + "\n")
                    telemetry.flush()
                    video.append_data(annotated_frame(
                        obs, spec, instruction, t, grasp, override, state["target_predicates"],
                    ))
                    if t % 100 == 0 or trigger:
                        pred = " ".join(f"{k}={int(v)}" for k, v in state["target_predicates"].items())
                        print(
                            f"{spec['episode_id']} t={t} grasp={grasp} override={override} {pred}",
                            flush=True,
                        )
            metrics = episode_metrics_v2(rows, intended_meta["target"], override_state)
            summary = {
                "protocol_version": PROTOCOL_V2,
                **spec,
                **self.provenance,
                "host_task": self.host_task_name,
                "task_id": self.task_id,
                "bddl_sha256": hashlib.sha256(self.bddl.read_bytes()).hexdigest(),
                "init_state_index": seed % len(self.init_states),
                "initial_instruction": initial,
                "override_text": override_text,
                "intended_target": intended_meta["target"],
                "intended_instruction": intended_language,
                "manipulated_object": OBJECT_SPECS[object_key]["state"],
                "tracked_targets": tracked,
                "initial_state": initial_state,
                "override_state": override_state,
                "queue_audit": audits,
                **metrics,
                "termination_reason": "fixed_horizon",
                "raw_dir": str(output),
                "telemetry_sha256": hashlib.sha256((output / "telemetry.jsonl").read_bytes()).hexdigest(),
            }
            temporary = output / "summary.tmp"
            temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            temporary.replace(output / "summary.json")
            print(
                f"COMPLETE {spec['episode_id']}: success={summary['intended_goal_success']} "
                f"outcome={summary['behavioral_outcome']}",
                flush=True,
            )
            return summary
        finally:
            env.close()


def baseline_spec(pair_id: str, goal_role: str, seed: int, max_steps: int = 300) -> dict:
    pair = SCREEN_PAIRS[pair_id]
    language = pair["goal_a"] if goal_role == "A" else pair["goal_b"]
    condition = f"baseline_{goal_role.lower()}_{pair_id.lower()}"
    return {
        "episode_id": f"{condition}_delay0_seed{seed}",
        "pair_id": pair_id,
        "condition": condition,
        "goal_role": goal_role,
        "seed": seed,
        "delay": 0,
        "max_steps": max_steps,
        "protocol_version": PROTOCOL_V2,
        "initial_instruction": language,
        "override_text": None,
        "intended_instruction": language,
    }
