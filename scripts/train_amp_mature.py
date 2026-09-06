import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import torch

from my_amp.envs.vec_env import G1VecEnv
from my_amp.motion.motion_loader import MotionLoader
from my_amp.configs.train_cfg_mature import TRAIN_CFG_MATURE
from mature_rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = copy.deepcopy(TRAIN_CFG_MATURE)

    motion_loader = MotionLoader(cfg["amp_motion_files"])
    env = G1VecEnv(
        num_envs=cfg["num_envs"],
        max_episode_length=cfg["max_episode_length"],
        amp_body_names=cfg["amp_body_names"],
        amp_anchor_name=cfg["amp_anchor_name"],
        motion_loader=motion_loader,
        reset_from_ref_prob=0.6,
    )

    runner = AmpOnPolicyRunner(
        env,
        cfg,
        log_dir="logs/amp_mature_walk_forward_v6",
        device=device,
    )
    runner.learn(cfg["max_iterations"])


if __name__ == "__main__":
    main()
