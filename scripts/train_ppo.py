import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from rsl_rl.runners import OnPolicyRunner
from my_amp.envs.vec_env import G1VecEnv
from my_amp.configs.train_cfg import TRAIN_CFG

def main():
    env = G1VecEnv(num_envs=TRAIN_CFG["num_envs"])
    runner = OnPolicyRunner(env, TRAIN_CFG, log_dir="logs/ppo_baseline", device="cpu")
    runner.learn(
        num_learning_iterations=TRAIN_CFG["max_iterations"],
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
