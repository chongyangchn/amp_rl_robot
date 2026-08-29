import time
import mujoco
import mujoco.viewer
import numpy as np
from my_amp.envs.g1_env import G1Env


def load_root_state(motion):
    if "root_pos" in motion:
        return (
            motion["root_pos"],
            motion["root_quat"],
            motion["root_lin_vel"],
            motion["root_ang_vel"],
        )

    return (
        motion["body_pos_w"][:, 0],
        motion["body_quat_w"][:, 0],
        motion["body_lin_vel_w"][:, 0],
        motion["body_ang_vel_w"][:, 0],
    )

env = G1Env("my_amp/envs/unitree_g1/scene.xml")
# motion = np.load("data/motions/sfu_walking001.npz")
# motion = np.load("data/motions_walk_250/walk_forward_loop_002__A024_250.npz")
motion = np.load("data/motions_walk/walk_forward_loop_002__A024.npz")

root_pos, root_quat, root_lin_vel, root_ang_vel = load_root_state(motion)


with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    for i in range(len(motion["joint_pos"])):
        env.data.qpos[:3] = root_pos[i]
        env.data.qpos[3:7] = root_quat[i]
        env.data.qpos[3:7] /= np.linalg.norm(env.data.qpos[3:7])
        env.data.qpos[7:] = motion["joint_pos"][i]

        env.data.qvel[:3] = root_lin_vel[i]
        env.data.qvel[3:6] = root_ang_vel[i]
        env.data.qvel[6:] = motion["joint_vel"][i]

        mujoco.mj_forward(env.model, env.data)
        viewer.sync()

        if not viewer.is_running():
            break

        time.sleep(env.dt)