import numpy as np





def compute_task_reward(env, action):
    rot = env._cached_rot

    local_lin_vel = rot.T @ env.data.qvel[:3]
    local_ang_vel = rot.T @ env.data.qvel[3:6]

    lin_vel_error = (
        (env.command[0] - local_lin_vel[0]) ** 2
        + (env.command[1] - local_lin_vel[1]) ** 2
    )
    ang_vel_error = (
        (env.command[1] - local_ang_vel[2]) ** 2
        + local_ang_vel[0] ** 2
        + local_ang_vel[1] ** 2
    )

    track_lin_vel = np.exp(-lin_vel_error / 1.0)
    track_ang_vel = np.exp(-ang_vel_error / 1.0)

    root_height = env.data.qpos[2]
    desired_height = env.default_root_height
    height_reward = np.exp(-((root_height - desired_height) ** 2) / 0.3)

    action_rate = np.mean((action - env.prev_action) ** 2)

    foot_contact = env._get_foot_contacts()

    # 不要奖励 mean(foot_contact)
    foot_slip = 0.0
    for foot_id, contact in zip(
        (env._left_foot_body, env._right_foot_body),
        foot_contact,
    ):
        if contact > 0:
            foot_vel_xy = env.data.cvel[foot_id, 3:5]
            foot_slip += float(np.linalg.norm(foot_vel_xy))

    total = (
        1.0 * track_lin_vel
        + 1.0 * track_ang_vel
        + 1.0 * height_reward
        - 0.01 * action_rate
        - 0.25 * foot_slip
    )

    return total * 0.05


# def compute_task_reward(env, action):
#     total = (
#         1.0 * track_lin_vel(env)
#         + 1.0 * track_ang_vel(env)
#         + 1.0 * root_height_reward(env)
#         - 0.01 * action_rate_cost(env, action)
#         - 0.25 * foot_slip_cost(env)
#     )
#     return total