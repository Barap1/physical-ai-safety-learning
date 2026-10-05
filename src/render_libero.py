import os
import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))

suite_name = "libero_object"
task_id = 0

benchmark_dict = benchmark.get_benchmark_dict()
suite = benchmark_dict[suite_name]()
task = suite.get_task(task_id)

print("Suite:", suite_name)
print("Task ID:", task_id)
print("Task name:", task.name)
print("Language instruction:", task.language)

bddl_path = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file,
)

print("BDDL file:", bddl_path)

env = OffScreenRenderEnv(
    bddl_file_name=bddl_path,
    camera_heights=256,
    camera_widths=256,
)

env.seed(0)
env.reset()

initial_states = suite.get_task_init_states(task_id)
obs = env.set_init_state(initial_states[0])

for _ in range(5):
    obs, reward, done, info = env.step(np.zeros(7))

print("Observation keys:")
for key in obs.keys():
    print(" ", key)

image = obs["agentview_image"]
print("Camera image shape:", image.shape)

# MuJoCo offscreen images are usually upside down
image = image[::-1]

output_path = "results/first_libero_frame.png"
Image.fromarray(image).save(output_path)

print("Saved:", output_path)

env.close()
print("FIRST LIBERO RENDER: SUCCESS")
