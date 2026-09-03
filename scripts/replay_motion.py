import time
from pathlib import Path

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


def get_fps(motion):
    if "fps" not in motion:
        return 50.0
    fps = np.asarray(motion["fps"]).ravel()
    return float(fps[0]) if len(fps) > 0 else 50.0


def main():
    motion_dir = Path("data/motions_walk_forward")
    motion_files = sorted(motion_dir.glob("*.npz"))
    if not motion_files:
        raise FileNotFoundError(f"no npz files found in {motion_dir}")

    env = G1Env("my_amp/envs/unitree_g1/scene.xml")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15

        for motion_file in motion_files:
            motion = np.load(motion_file)
            root_pos, root_quat, root_lin_vel, root_ang_vel = load_root_state(motion)
            fps = get_fps(motion)
            frame_dt = 1.0 / fps

            print(
                f"playing {motion_file.name} | frames={len(motion['joint_pos'])} | fps={fps}"
            )

            for i in range(len(motion["joint_pos"])):
                if not viewer.is_running():
                    return

                env.data.qpos[:3] = root_pos[i]
                env.data.qpos[3:7] = root_quat[i]
                env.data.qpos[3:7] /= np.linalg.norm(env.data.qpos[3:7])
                env.data.qpos[7:] = motion["joint_pos"][i]

                env.data.qvel[:3] = root_lin_vel[i]
                env.data.qvel[3:6] = root_ang_vel[i]
                env.data.qvel[6:] = motion["joint_vel"][i]

                mujoco.mj_forward(env.model, env.data)
                viewer.sync()
                time.sleep(frame_dt)

            time.sleep(0.3)


if __name__ == "__main__":
    main()
