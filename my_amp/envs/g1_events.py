import numpy as np


def resample_command(env):
    lin_range = getattr(env, "command_lin_range", (-0.2, 0.6))
    yaw_range = getattr(env, "command_yaw_range", (-0.5, 0.5))
    env.command = np.array(
        [
            np.random.uniform(*lin_range),
            np.random.uniform(*yaw_range),
        ],
        dtype=np.float32,
    )


def resample_command_from_ref(env, command):
    env.command = np.array(command, dtype=np.float32)
