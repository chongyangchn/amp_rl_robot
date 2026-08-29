TRAIN_CFG = {
    "algorithm": {
        "class_name": "PPO",
        "rnd_cfg": None,               # 必须有
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.01,
        "learning_rate": 3e-4,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": (512, 256, 128),
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "BetaDistribution",
            "action_range": (-1.0, 1.0),
        },
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": (512, 256, 128),
        "activation": "elu",
        "obs_normalization": True,
    },
    "amp": {
        # "motion_dir": "data/motions",
        "motion_dir": "data/motions_walk_250",
        "body_names": [
            "pelvis", "torso_link",
            "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link",
            "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link",
            "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_pitch_link",
            "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_pitch_link",
        ],
        "anchor_name": "pelvis", 
        "amp_obs_dim": 210,  # 14 * 15 = 210  14个关节，每个关节15维
        "amp_reward_coef": 0.1,    
        "task_reward_lerp": 0.5,
        "disc_learning_rate": 5e-6,
        "disc_hidden_dims": (128, 64),
        "replay_buffer_size": 1_000_000,
        "disc_batch_size": 512,
        "disc_updates_per_iter": 1,
        "reset_from_ref_prob": 1, # 环境 reset 时，有 80% 的概率从参考动作的某一帧开始，而不是从默认站姿开始。
    },
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "use_ref_command": True,
        "num_steps_per_env": 24,
        "save_interval": 200,
        "logger": "tensorboard",
        "max_iterations": 10000,
        "num_envs": 64,
        "seed": 42,
        "multi_gpu": None,

    "motion": {
        "motion_dir": "data/motions",
        "reset_from_ref_prob": 0.8,
        "use_ref_command": True,
    }
}

