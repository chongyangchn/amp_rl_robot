import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time
import torch
import mujoco
import mujoco.viewer

from rsl_rl.algorithms import PPO
from my_amp.envs.vec_env import G1VecEnv
from my_amp.configs.train_cfg import TRAIN_CFG
from my_amp.motion.motion_loader import MotionLoader


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # checkpoint_path = "logs/amp_rsl_v1/model_2000.pt"
    checkpoint_path = "logs/amp_rsl_beta_v2/model_600.pt"

    cfg = copy.deepcopy(TRAIN_CFG)
    motion_loader = MotionLoader(cfg["amp"]["motion_dir"])

    # 创建单环境
    env = G1VecEnv(
        num_envs=1,
        amp_body_names=cfg["amp"]["body_names"],
        amp_anchor_name=cfg["amp"]["anchor_name"],
        motion_loader=motion_loader,
        reset_from_ref_prob=1.0,
    )

    # 构建和训练时一样的 PPO
    obs = env.get_observations().to(device)
    ppo = PPO.construct_algorithm(obs, env, cfg, device)

    # 加载训练好的 actor
    ckpt = torch.load(checkpoint_path, map_location=device)
    ppo.actor.load_state_dict(ckpt["actor_state_dict"])
    ppo.actor.eval()

    single_env = env.envs[0]

    with mujoco.viewer.launch_passive(single_env.model, single_env.data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15

        obs = env.reset_all_to_ref().to(device)

        while viewer.is_running():
            with torch.no_grad():
                action = ppo.actor(obs)

            obs, rewards, dones, extras = env.step(action.to(env.device))
            obs = obs.to(device)

            viewer.sync()
            time.sleep(single_env.dt)

    print("eval finished")


if __name__ == "__main__":
    main()
