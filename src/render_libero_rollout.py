import os
import numpy as np
from PIL import Image
import imageio.v2 as imageio

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))

suite_name = "libero_object"
task_id = 0

benchmark_dict = benchmark.get_benchmark_dict()
suite = benchmark_dict[suite_name]()
task = suite.get_task(task_id)

print("Task:", task.language)

bddl_path = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file,
)

env = OffScreenRenderEnv(
    bddl_file_name=bddl_path,
    camera_heights=256,
    camera_widths=256,
)

env.seed(0)
env.reset()

initial_states = suite.get_task_init_states(task_id)
obs = env.set_init_state(initial_states[0])

frames = []

# Let physics settle first
for _ in range(10):
    obs, reward, done, info = env.step(np.zeros(7))

# ---------------------------------------------------
# Simple hand-written motion
#
# Action format is approximately:
# [dx, dy, dz, droll, dpitch, dyaw, gripper]
#
# We are NOT trying to complete the task yet.
# We just want to prove that actions move the robot.
# ---------------------------------------------------

actions = []

# Move in one direction
for _ in range(30):
    actions.append(
        np.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )

# Move back
for _ in range(30):
    actions.append(
        np.array([-0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )

# Move upward
for _ in range(30):
    actions.append(
        np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0])
    )

print("Running", len(actions), "actions...")

for step, action in enumerate(actions):
    obs, reward, done, info = env.step(action)

    image = obs["agentview_image"][::-1]
    frames.append(image)

    if step % 10 == 0:
        print(
            f"step={step:03d} "
            f"reward={reward} "
            f"done={done}"
        )

output_path = "results/first_rollout.mp4"

imageio.mimsave(
    output_path,
    frames,
    fps=20,
)

print()
print("Saved:", output_path)
print("Frames:", len(frames))

env.close()

print("FIRST ROBOT ROLLOUT: SUCCESS")
