from pathlib import Path
import numpy as np
from my_amp.amp.amp_obs import quat_to_rotmat

class MotionLoader:
    def __init__(self, motion_dir="data/motions"):
        self.motions = {}
        for p in Path(motion_dir).glob("*.npz"):
            self.motions[p.stem] = np.load(p)
        self.names = list(self.motions)
        if not self.names:
            raise FileNotFoundError(f"no npz found in {motion_dir}")

    def sample(self, rng):
        rng = rng if rng is not None else np.random.default_rng()
        name = rng.choice(self.names)
        motion = self.motions[name]
        idx = int(rng.integers(0, len(motion["joint_pos"])))
        return name, motion, idx

    def _root_state(self, motion, idx):
        if "root_pos" in motion:
            # 自己转换的动作源
            return (
                motion["root_pos"][idx],
                motion["root_quat"][idx],
                motion["root_lin_vel"][idx],
                motion["root_ang_vel"][idx],
            )
        return (
            # 这里是AMP_mjlab的动作源
            motion["body_pos_w"][idx, 0],
            motion["body_quat_w"][idx, 0],
            motion["body_lin_vel_w"][idx, 0],
            motion["body_ang_vel_w"][idx, 0],
        )

    def get_reset_state(self, rng=None):
        qpos, qvel, command = self.get_reset_state_with_command(rng)
        return qpos, qvel

    def get_reset_state_with_command(self, rng=None):
        name, motion, idx = self.sample(rng)
        root_pos, root_quat, root_lin_vel, root_ang_vel = self._root_state(motion, idx)

        # 把 root 速度转到 root 局部系，作为前向速度和转向速度命令
        R = quat_to_rotmat(root_quat)
        local_lin_vel = R.T @ root_lin_vel
        local_ang_vel = R.T @ root_ang_vel
        command = np.array([local_lin_vel[0], local_ang_vel[2]], dtype=np.float32)

        qpos = np.zeros(36, dtype=np.float64)
        qpos[:3] = root_pos
        qpos[3:7] = root_quat
        qpos[3:7] /= np.linalg.norm(qpos[3:7])
        qpos[7:] = motion["joint_pos"][idx]

        qvel = np.zeros(35, dtype=np.float64)
        qvel[:3] = root_lin_vel
        qvel[3:6] = root_ang_vel
        qvel[6:] = motion["joint_vel"][idx]

        return qpos, qvel, command

