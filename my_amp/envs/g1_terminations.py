import numpy as np

from my_amp.envs.g1_obs import quat_to_rotmat


def check_termination(env):
    height = env.data.qpos[2]
    if height < 0.4 or height > 1.4:
        return True, "bad_base_height"

    rot = quat_to_rotmat(env.data.qpos[3:7])
    projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0])
    if projected_gravity[2] > -0.342:
        return True, "bad_orientation"

    return False, ""
