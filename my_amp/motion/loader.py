from pathlib import Path
import numpy as np


class MotionLoader:
    def __init__(self, motion_dir="data/motions"):
        self.motions = {}
        for p in Path(motion_dir).glob("*.npz"):
            self.motions[p.stem] = np.load(p)
        self.names = list(self.motions)

    def sample(self):
        rng = rng if rng is not None else np.random.default_rng()
        name = rng.choice(self.names)
        motion = self.motions[name]
        idx = int(rng.integers(0, len(motion["joint_pos"])))
        return name, motion, idx

    def get_reset_state(self, rng=None):
        _, motion, idx = self.sample(rng)
        qpos = np.zeros(36, dtype=np.float64)
        qpos[:3] = motion["root_pos"][idx]
        qpos[3:7] = motion["root_quat"][idx]
        qpos[3:7] /= np.linalg.norm(qpos[3:7])
        qpos[7:] = motion["joint_pos"][idx]

        qvel = np.zeros(35, dtype=np.float64)
        qvel[:3] = motion["root_lin_vel"][idx]
        qvel[3:6] = motion["root_ang_vel"][idx]
        qvel[6:] = motion["joint_vel"][idx]

        return qpos, qvel

