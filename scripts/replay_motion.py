import time
import mujoco
import mujoco.viewer
import numpy as np
from my_amp.envs.g1_env import G1Env


env = G1Env("my_amp/envs/unitree_g1/scene.xml")
motion = np.load("data/motions/sfu_walking001.npz")


with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    for i in range(len(motion["joint_pos"])):
        env.data.qpos[:3] = motion["root_pos"][i]
        env.data.qpos[3:7] = motion["root_quat"][i]
        env.data.qpos[7:] = motion["joint_pos"][i]
        env.data.qvel[6:] = motion["joint_vel"][i]
        mujoco.mj_forward(env.model, env.data)
        viewer.sync()
        time.sleep(env.dt)