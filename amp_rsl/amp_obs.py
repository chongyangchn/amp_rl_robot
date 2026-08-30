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


def _get_gait_phase(env):
    phase = 2.0 * np.pi * (env.step_count * env.dt / env.gait_period)
    return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)


def get_policy_obs(env):
    rot = _get_rot(env)

    local_lin_vel = rot.T @ env.data.qvel[:3]
    local_ang_vel = rot.T @ env.data.qvel[3:6]
    projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0])

    joint_pos_norm = (
        2.0 * (env.data.qpos[7:] - env.joint_low)
        / (env.joint_high - env.joint_low)
        - 1.0
    )
    joint_vel = env.data.qvel[6:] * 0.05
    foot_contact = env._get_foot_contacts()

    height = env.data.qpos[2]
    height_deviation = height - env.target_height
    torso_xmat = env.data.site_xmat[env.torso_site].reshape(3, 3)
    upright = torso_xmat[2, 2]

    gait_phase = _get_gait_phase(env)

    return np.concatenate([
        local_lin_vel,          # 3
        local_ang_vel,          # 3
        projected_gravity,      # 3
        env.command,            # 2
        joint_pos_norm,         # 29
        joint_vel,              # 29
        env.prev_action,        # 29
        foot_contact,           # 2
        [height_deviation, upright],  # 2
        gait_phase,             # 2
    ])


def get_critic_obs(env):
    return get_policy_obs(env)