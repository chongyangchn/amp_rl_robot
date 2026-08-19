import torch

from my_amp.envs.vec_env import G1VecEnv
from my_amp.configs.train_cfg import TRAIN_CFG
from my_amp.amp.amp_runner import AMPRunner



def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = G1VecEnv(
        num_envs=TRAIN_CFG["num_envs"],
        amp_body_names=TRAIN_CFG["amp"]["body_names"],
        amp_anchor_name=TRAIN_CFG["amp"]["anchor_name"],
    )
    runner = AMPRunner(
        env,
        TRAIN_CFG,
        log_dir="logs/amp_baseline",
        device=device,
    )
    runner.learn(TRAIN_CFG["max_iterations"])


if __name__ == "__main__":
    main()


