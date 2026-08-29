import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import copy
import torch

from my_amp.envs.vec_env import G1VecEnv
from my_amp.motion.motion_loader import MotionLoader
from my_amp.configs.train_cfg import TRAIN_CFG
from amp_rsl.amp_runner import AMPRunner


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = copy.deepcopy(TRAIN_CFG)

    motion_loader = MotionLoader(cfg["amp"]["motion_dir"])

    env = G1VecEnv(
        num_envs=cfg["num_envs"],
        amp_body_names=cfg["amp"]["body_names"],
        amp_anchor_name=cfg["amp"]["anchor_name"],
        motion_loader=motion_loader,
        reset_from_ref_prob=cfg["amp"]["reset_from_ref_prob"],
    )

    runner = AMPRunner(
        env,
        cfg,
        log_dir="logs/amp_rsl_beta",
        device=device,
    )

    runner.learn(cfg["max_iterations"])


if __name__ == "__main__":
    main()