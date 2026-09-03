import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time

import torch
import mujoco
import mujoco.viewer

from my_amp.envs.vec_env import G1VecEnv
from my_amp.motion.motion_loader import MotionLoader
from my_amp.configs.train_cfg_mature import TRAIN_CFG_MATURE
from mature_rsl_rl.modules import ActorCritic, EmpiricalNormalization


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = copy.deepcopy(TRAIN_CFG_MATURE)
    checkpoint_path = "logs/amp_mature_walk_forward_v1/model_800.pt"

    motion_loader = MotionLoader(cfg["amp_motion_files"])
    env = G1VecEnv(
        num_envs=1,
        max_episode_length=cfg["max_episode_length"],
        amp_body_names=cfg["amp_body_names"],
        amp_anchor_name=cfg["amp_anchor_name"],
        motion_loader=motion_loader,
        reset_from_ref_prob=0.6,
    )

    obs = env.get_observations()
    num_actor_obs = obs["policy"].shape[1]
    num_critic_obs = obs["critic"].shape[1]

    policy_cfg = cfg["policy"]
    model = ActorCritic(
        num_actor_obs,
        num_critic_obs,
        env.num_actions,
        actor_hidden_dims=policy_cfg["actor_hidden_dims"],
        critic_hidden_dims=policy_cfg["critic_hidden_dims"],
        activation=policy_cfg["activation"],
        init_noise_std=policy_cfg["init_noise_std"],
        noise_std_type=policy_cfg["noise_std_type"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    obs_normalizer = EmpiricalNormalization(shape=[num_actor_obs]).to(device)
    if "obs_norm_state_dict" in ckpt:
        obs_normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
    obs_normalizer.eval()

    single_env = env.envs[0]
    with mujoco.viewer.launch_passive(single_env.model, single_env.data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15

        obs = env.reset_all_to_ref()
        while viewer.is_running():
            policy_obs = obs["policy"].to(device)
            with torch.no_grad():
                policy_obs = obs_normalizer(policy_obs)
                action = model.act_inference(policy_obs)
                action = torch.clamp(action, -1.0, 1.0)

            obs, rewards, dones, extras = env.step(action.to(env.device))
            viewer.sync()
            time.sleep(single_env.dt)

    print("eval finished")


if __name__ == "__main__":
    main()
