import numpy as np

from my_amp.envs.g1_terminations import check_termination


def compute_task_reward(env, action):
    rot = env._cached_rot

    local_lin_vel = rot.T @ env.data.qvel[:3]
    local_ang_vel = rot.T @ env.data.qvel[3:6]
    projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0])

    # command = [forward_velocity, yaw_rate]
    lin_vel_error = (env.command[0] - local_lin_vel[0]) ** 2
    ang_vel_error = (
        (env.command[1] - local_ang_vel[2]) ** 2
        + local_ang_vel[0] ** 2
        + local_ang_vel[1] ** 2
    )

    track_lin_vel = np.exp(-lin_vel_error / 1.0)
    track_ang_vel = np.exp(-ang_vel_error / (np.pi ** 2))

    root_height = env.data.qpos[2]
    height_reward = np.exp(
        -((root_height - env.default_root_height) ** 2) / 0.3
    )

    upright_reward = np.exp(
        -np.sum(projected_gravity[:2] ** 2) / 0.25
    )
    body_ang_vel_reward = np.exp(
        -(local_ang_vel[0] ** 2 + local_ang_vel[1] ** 2) / (np.pi ** 2)
    )
    survival_reward = 1.0

    action_rate = np.mean((action - env.prev_action) ** 2)
    action_magnitude = float(np.mean(np.square(action)))
    joint_acc = float(np.mean(np.square(env.data.qacc[6:])))

    joint_pos = env.data.qpos[7:]
    joint_delta = joint_pos - env.default_joint_pos
    joint_pos_limits = float(
        np.mean(
            np.maximum(env.joint_low - joint_pos, 0.0)
            + np.maximum(joint_pos - env.joint_high, 0.0)
        )
    )

    terminated, _ = check_termination(env)
    is_terminated = 1.0 if terminated else 0.0

    foot_contact = env._get_foot_contacts()
    foot_slip = 0.0

    for foot_id, contact in zip(
        (env._left_foot_body, env._right_foot_body),
        foot_contact,
    ):
        if contact > 0:
            foot_vel_local = rot.T @ env.data.cvel[foot_id, :3]
            foot_slip += float(foot_vel_local[0] ** 2 + foot_vel_local[1] ** 2)

    foot_tilt = 0.0
    for foot_id in (env._left_foot_body, env._right_foot_body):
        foot_xmat = env.data.xmat[foot_id].reshape(3, 3)
        foot_up = foot_xmat[:, 2]
        foot_tilt += float(foot_up[0] ** 2 + foot_up[1] ** 2)
    foot_flat_reward = float(np.exp(-foot_tilt / 0.1))

    contacts, heel, toe, forces = env._get_foot_contact_details()
    heel_toe_penalty = float(np.mean(heel + toe) * 0.5)
    foot_center_reward = 1.0 - heel_toe_penalty

    foot_contact = env._get_foot_contacts()
    phase_val = (env.step_count * env.dt) % env.gait_period
    left_should_contact = 1.0 if np.sin(2.0 * np.pi * phase_val / env.gait_period) > 0.0 else 0.0
    right_should_contact = 1.0 - left_should_contact
    gait_reward = (
        1.0
        - 0.5 * abs(foot_contact[0] - left_should_contact)
        - 0.5 * abs(foot_contact[1] - right_should_contact)
    )

    left_hip = joint_delta[0]
    right_hip = joint_delta[6]
    left_knee = joint_delta[3]
    right_knee = joint_delta[9]
    left_ankle_pitch = joint_delta[4]
    right_ankle_pitch = joint_delta[10]
    left_ankle_roll = joint_delta[5]
    right_ankle_roll = joint_delta[11]
    left_shoulder = joint_delta[15]
    right_shoulder = joint_delta[22]

    leg_motion = abs(left_hip) + abs(right_hip) + abs(left_knee) + abs(right_knee)
    leg_motion_reward = float(np.clip(leg_motion / 1.2, 0.0, 1.0))

    ankle_motion = abs(left_ankle_pitch) + abs(right_ankle_pitch) + abs(left_ankle_roll) + abs(right_ankle_roll)
    ankle_use_reward = float(np.clip(ankle_motion / 0.5, 0.0, 1.0))

    arm_swing_reward = (
        float(np.exp(-((left_shoulder - right_hip * 0.6) ** 2) / 0.5 ** 2))
        + float(np.exp(-((right_shoulder - left_hip * 0.6) ** 2) / 0.5 ** 2))
    ) * 0.5

    total = (
        1.0 * track_lin_vel
        + 1.0 * track_ang_vel
        + 1.0 * height_reward
        + 1.0 * upright_reward
        + 1.0 * body_ang_vel_reward
        + 0.5 * gait_reward
        + 0.3 * leg_motion_reward
        + 0.3 * ankle_use_reward
        + 0.3 * arm_swing_reward
        + 0.5 * foot_flat_reward
        + 0.5 * foot_center_reward
        - 0.01 * action_rate
        - 0.10 * action_magnitude
        - 0.25 * foot_slip
        - 2.5e-7 * joint_acc
        - 10.0 * joint_pos_limits
        - 200.0 * is_terminated
    )

    reward_terms = {
        "track_lin_vel": track_lin_vel,
        "track_ang_vel": track_ang_vel,
        "height": height_reward,
        "upright": upright_reward,
        "body_ang_vel": body_ang_vel_reward,
        "survival": survival_reward,
        "gait": gait_reward,
        "leg_motion": leg_motion_reward,
        "ankle_use": ankle_use_reward,
        "arm_swing": arm_swing_reward,
        "foot_flat": foot_flat_reward,
        "foot_center": foot_center_reward,
        "action_rate": action_rate,
        "action_magnitude": action_magnitude,
        "foot_slip": foot_slip,
        "joint_acc": joint_acc,
        "joint_pos_limits": joint_pos_limits,
        "is_terminated": is_terminated,
    }

    return total * 0.05, reward_terms
