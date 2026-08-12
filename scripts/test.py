import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from my_amp.envs.g1_env import G1Env
from my_amp.motion.loader import MotionLoader

def test1():
    env = G1Env("my_amp/envs/unitree_g1/scene.xml")
    motion = np.load("data/motions/walk_forward_loop_002__A024.npz")

    li = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    ri = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    n = len(motion["joint_pos"])
    lz = np.empty(n)
    rz = np.empty(n)

    for i in range(n):
        env.data.qpos[:3] = motion["body_pos_w"][i, 0]
        env.data.qpos[3:7] = motion["body_quat_w"][i, 0]
        env.data.qpos[7:] = motion["joint_pos"][i]
        mujoco.mj_forward(env.model, env.data)
        lz[i] = env.data.site_xpos[li][2]
        rz[i] = env.data.site_xpos[ri][2]

    print("left  foot z: min", round(lz.min(), 3), "median", round(np.median(lz), 3))
    print("right foot z: min", round(rz.min(), 3), "median", round(np.median(rz), 3))


def test2():
    loader = MotionLoader("data/motions")
    env = G1Env("my_amp/envs/unitree_g1/scene.xml")
    rng = np.random.default_rng(0)

    for trial in range(50):
        qpos, qvel = loader.get_reset_state(rng)
        obs = env.reset_to_ref(qpos, qvel)
        assert np.isfinite(obs["policy"]).all(), f"trial {trial} has NaN"
        for _ in range(10):
            obs, reward, done, _ = env.step(np.zeros(29))
            assert np.isfinite(obs["policy"]).all(), f"trial {trial} step NaN"
    print("50 random reference resets OK")



def test3():
    DEFAULT_MOTION = "data/motions/walk_forward_loop_002__A024.npz"
    motion_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MOTION

    motion, root_pos, root_quat, root_lin_vel, root_ang_vel = load_motion(motion_path)
    env = G1Env("my_amp/envs/unitree_g1/scene.xml")

    n = len(motion["joint_pos"])
    fps = float(motion["fps"][0]) if "fps" in motion else 50.0
    dt = 1.0 / fps
    print(f"replay {n} frames, fps={fps:.0f}, dt={dt * 1000:.1f}ms")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            for i in range(n):
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
                time.sleep(dt)

            if i % 100 == 0:
                pass  # 循环播放，关掉窗口退出

    print("replay finished")

def load_motion(path):
    motion = np.load(path)
    if "root_pos" in motion:
        root_pos = motion["root_pos"]
        root_quat = motion["root_quat"]
        root_lin_vel = motion["root_lin_vel"]
        root_ang_vel = motion["root_ang_vel"]
    else:
        # AMP_mjlab 的 npz 没有 root_*，根部位姿存在 body[0] 里
        root_pos = motion["body_pos_w"][:, 0]
        root_quat = motion["body_quat_w"][:, 0]
        root_lin_vel = motion["body_lin_vel_w"][:, 0]
        root_ang_vel = motion["body_ang_vel_w"][:, 0]
    return motion, root_pos, root_quat, root_lin_vel, root_ang_vel

if __name__ == "__main__":
    test3()