import time
import torch
import mujoco

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rsl_rl.algorithms import PPO

from amp_rsl.amp_ppo import AMPPPO
from amp_rsl.discriminator import AMPDiscriminator
from amp_rsl.replay_buffer import AMPReplayBuffer
from amp_rsl.motion_loader import AMPLoader


class AMPRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.cfg = train_cfg
        self.amp_cfg = train_cfg["amp"]
        self.device = device
        self.log_dir = log_dir

        obs = self.env.get_observations().to(self.device)
        self.ppo = PPO.construct_algorithm(obs, self.env, self.cfg, self.device)

        self.discriminator = AMPDiscriminator(
            amp_obs_dim=self.amp_cfg["amp_obs_dim"],
            hidden_dims=self.amp_cfg["disc_hidden_dims"],
        ).to(self.device)

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

        self.replay_buffer = AMPReplayBuffer(
            self.amp_cfg["replay_buffer_size"],
            self.amp_cfg["amp_obs_dim"],
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

    def learn(self, num_learning_iterations):
        obs = self.env.reset_all_to_ref().to(self.device)
        self.alg.train_mode()
        # Pre-fill AMP replay buffer before discriminator update.
        for _ in range(200):
            actions = self.alg.act(obs)

            obs, rewards, dones, extras = self.env.step(
                actions.to(self.env.device)
            )
            obs = obs.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)

            self.alg.process_env_step(obs, rewards, dones, extras)
            # 预填充只需要 AMP replay buffer，
            # 不要让 PPO RolloutStorage 堆积。
            self.alg.ppo.storage.clear()

        pbar = tqdm(range(num_learning_iterations), desc="AMP training", dynamic_ncols=True)

        for it in pbar:
            start = time.time()

            style_rewards = []
            final_rewards = []
            task_rewards = []
            action_list = []
            done_count = 0
            done_lengths = []

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    prev_lengths = self.env.episode_length_buf.clone()

                    actions = self.alg.act(obs)

                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )
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
                            (prev_lengths[dones_cpu] + 1).tolist()
                        )

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            actions_t = torch.cat(action_list, dim=0)
            total_steps = self.cfg["num_steps_per_env"] * self.env.num_envs
            done_rate = done_count / total_steps
            avg_episode_length = (
                sum(done_lengths) / len(done_lengths) if done_lengths else 0.0
            )

            metrics = {
                "Train/task_reward": sum(task_rewards) / len(task_rewards),
                "Train/style_reward": sum(style_rewards) / len(style_rewards),
                "Train/final_reward": sum(final_rewards) / len(final_rewards),
                "Train/done_rate": done_rate,
                "Train/episode_length": avg_episode_length,
                "Loss/discriminator": self.alg.last_disc_loss,
                "Train/discriminator_acc": self.alg.last_disc_acc,
                "Disc/expert_logit": self.alg.last_expert_logit,
                "Disc/policy_logit": self.alg.last_policy_logit,
                "Action/mean": actions_t.mean().item(),
                "Action/std": actions_t.std().item(),
                "Action/min": actions_t.min().item(),
                "Action/max": actions_t.max().item(),
                "Perf/fps": total_steps / (time.time() - start + 1e-8),
            }

            for key, value in loss_dict.items():
                metrics[f"Loss/{key}"] = value

            if self.writer is not None:
                for key, value in metrics.items():
                    self.writer.add_scalar(key, value, it)

            pbar.set_postfix(
                style=f"{metrics['Train/style_reward']:.3f}",
                ep_len=f"{avg_episode_length:.0f}",
                disc_acc=f"{self.alg.last_disc_acc:.3f}",
            )

            if self.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(self.log_dir + f"/model_{it}.pt")

        if self.writer is not None:
            self.save(self.log_dir + f"/model_{num_learning_iterations - 1}.pt")

    def save(self, path):
        state = self.alg.save()
        state["iter"] = self.current_learning_iteration
        state["discriminator_state_dict"] = self.discriminator.state_dict()
        state["disc_optimizer_state_dict"] = self.alg.disc_optimizer.state_dict()
        torch.save(state, path)