import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from rsl_rl.runners import OnPolicyRunner
from my_amp.envs.vec_env import G1VecEnv
from my_amp.configs.train_cfg import TRAIN_CFG

env = G1VecEnv(num_envs=1)
runner = OnPolicyRunner(env, TRAIN_CFG, device="cpu")
runner.load("logs/ppo_baseline/model_500.pt")

policy = runner.get_inference_policy("cpu")
obs = env.get_observations()

for t in range(300):
    with torch.no_grad():
        action = policy(obs)

    obs, rew, dones, info = env.step(action)
    if dones.any():
        print(f"fell at step {t}")
        break
else:
    print("survived full episode")