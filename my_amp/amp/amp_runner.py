import os
import time
import torch
import mujoco
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rsl_rl.algorithms import PPO
from my_amp.amp.amp_ppo import AMPPPO
from my_amp.amp.discriminator import Discriminator
from my_amp.motion.amp_loader import AMPLoader
from my_amp.storage.replay_buffer import ReplayBuffer
from my_amp.motion.motion_loader import MotionLoader


class AMPRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.cfg = train_cfg
        self.amp_cfg = train_cfg["amp"]
        self.device = device
        self.log_dir = log_dir

        # 普通 PPO（用你现有 config 构造）
        obs = self.env.get_observations().to(self.device)
        self.ppo = PPO.construct_algorithm(obs, self.env, self.cfg, self.device)

        # AMP 组件
        amp_obs_dim = self.amp_cfg["amp_obs_dim"]
        self.discriminator = Discriminator(
            amp_obs_dim=amp_obs_dim,
            amp_reward_coef=self.amp_cfg["amp_reward_coef"],
            task_reward_lerp=self.amp_cfg["task_reward_lerp"],
            hidden_dims=self.amp_cfg["disc_hidden_dims"],
        ).to(self.device)

        # all_body_names 从 MuJoCo 模型生成，顺序必须和 npz 一致
        model = mujoco.MjModel.from_xml_path("my_amp/envs/unitree_g1/scene.xml")
        all_body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(1, model.nbody)
        ]

        self.amp_loader = AMPLoader(
            self.amp_cfg["motion_dir"],
            self.amp_cfg["body_names"],
            self.amp_cfg["anchor_name"],
            all_body_names,
        )

        self.replay_buffer = ReplayBuffer(
            self.amp_cfg["replay_buffer_size"],
            amp_obs_dim,
            self.device,
        )

        self.alg = AMPPPO(
            self.ppo,
            self.discriminator,
            self.amp_loader,
            self.replay_buffer,
            self.amp_cfg,
            self.device,
        )

        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None
        self.current_learning_iteration = 0

        # self.motion_loader = MotionLoader(self.amp_cfg["motion_dir"])

    def learn(self, num_learning_iterations):
        obs = self.env.get_observations().to(self.device)
        if hasattr(self.env, "reset_all_to_ref"):
            obs = self.env.reset_all_to_ref().to(self.device)
        else:
            obs = self.env.get_observations().to(self.device)

        self.alg.train_mode()

        for it in tqdm(range(num_learning_iterations), desc="AMP training", dynamic_ncols=True):
            task_rewards = []
            style_rewards = []
            final_rewards = []
            action_list = []
            done_count = 0
            done_lengths = []
            start = time.time()

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    prev_episode_lengths = self.env.episode_length_buf.clone()

                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    step_metrics = self.alg.process_env_step(obs, rewards, dones, extras)

                    task_rewards.append(rewards.mean().item())
                    style_rewards.append(step_metrics["style_reward"].mean().item())
                    final_rewards.append(step_metrics["final_reward"].mean().item())
                    action_list.append(actions)

                    dones_cpu = dones.cpu()
                    if dones_cpu.any():
                        done_count += dones_cpu.sum().item()
                        done_lengths.extend(
                            (prev_episode_lengths[dones_cpu] + 1).tolist()
                        )

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            actions_t = torch.cat(action_list, dim=0)
            total_steps = self.cfg["num_steps_per_env"] * self.env.num_envs
            done_rate = done_count / total_steps
            avg_episode_length = sum(done_lengths) / len(done_lengths) if done_lengths else 0.0
            self.current_learning_iteration = it

            if self.writer is not None:
                tqdm.write(
                    f"Iter {it:05d} | style {sum(style_rewards) / len(style_rewards):.3f} "
                    f"| final {sum(final_rewards) / len(final_rewards):.3f} "
                    f"| ep_len {avg_episode_length:.0f} "
                    f"| disc_acc {self.alg.last_disc_acc:.3f}"
                )

            if self.writer is not None:
                self.writer.add_scalar("Loss/discriminator", self.alg.last_disc_loss, it)
                self.writer.add_scalar("Train/discriminator_acc", self.alg.last_disc_acc, it)

                self.writer.add_scalar("Train/task_reward", sum(task_rewards) / len(task_rewards), it)
                self.writer.add_scalar("Train/style_reward", sum(style_rewards) / len(style_rewards), it)
                self.writer.add_scalar("Train/final_reward", sum(final_rewards) / len(final_rewards), it)
                self.writer.add_scalar("Train/done_rate", done_rate, it)
                self.writer.add_scalar("Train/episode_length", avg_episode_length, it)

                self.writer.add_scalar("Action/mean", actions_t.mean().item(), it)
                self.writer.add_scalar("Action/std", actions_t.std().item(), it)
                self.writer.add_scalar("Action/min", actions_t.min().item(), it)
                self.writer.add_scalar("Action/max", actions_t.max().item(), it)

                self.writer.add_scalar("Disc/expert_logit", self.alg.last_expert_logit, it)
                self.writer.add_scalar("Disc/policy_logit", self.alg.last_policy_logit, it)
                
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"Loss/{key}", value, it)

            if self.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        if self.writer is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path):
        state = self.alg.save()
        state["iter"] = self.current_learning_iteration
        state["discriminator_state_dict"] = self.discriminator.state_dict()
        state["disc_optimizer_state_dict"] = self.alg.disc_optimizer.state_dict()
        torch.save(state, path)
