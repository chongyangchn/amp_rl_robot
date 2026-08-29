import numpy as np


def quat_to_rotmat(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def _get_rot(env):
    return quat_to_rotmat(env.data.qpos[3:7])


def get_policy_obs(env):
    rot = _get_rot(env)
    local_ang_vel = rot.T @ env.data.qvel[3:6]
    projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0])
    joint_pos_norm = 2.0 * (
        (env.data.qpos[7:] - env.joint_low)
        / (env.joint_high - env.joint_low)
    ) - 1.0
    joint_vel = env.data.qvel[6:] * 0.05

    return np.concatenate([
        local_ang_vel,
        projected_gravity,
        env.command,
        joint_pos_norm,
        joint_vel,
        env.prev_action,
    ])


def get_critic_obs(env):
    policy = get_policy_obs(env)
    rot = _get_rot(env)
    local_lin_vel = rot.T @ env.data.qvel[:3]
    return np.concatenate([policy, local_lin_vel])
