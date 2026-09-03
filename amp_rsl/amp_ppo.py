import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO
from amp_rsl.amp_reward import style_reward_from_logit, mix_amp_reward


class AMPPPO:
    def __init__(
        self,
        ppo: PPO,
        discriminator,
        amp_loader,
        replay_buffer,
        amp_normalizer,
        amp_cfg,
        device,
    ):
        self.ppo = ppo
        self.discriminator = discriminator.to(device)
        self.amp_loader = amp_loader
        self.replay_buffer = replay_buffer
        self.amp_normalizer = amp_normalizer
        self.amp_cfg = amp_cfg
        self.device = device

        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=amp_cfg["disc_learning_rate"],
            weight_decay=amp_cfg.get("disc_weight_decay", 1e-4),
        )

        self.disc_batch_size = amp_cfg["disc_batch_size"]
        self.disc_updates = amp_cfg.get("disc_updates_per_iter", 1)

        self._prev_amp_obs = None

        self.last_disc_loss = 0.0
        self.last_disc_acc = 0.0
        self.last_expert_logit = 0.0
        self.last_policy_logit = 0.0

    def act(self, obs):
        self._prev_amp_obs = obs["amp"].clone()
        actions = self.ppo.act(obs)
        return torch.clamp(actions, -1.0, 1.0)

    def process_env_step(self, obs, rewards, dones, extras):
        next_amp = obs["amp"].clone()
        terminal = dones.bool()

        next_amp_term = next_amp.clone()
        next_amp_term[terminal] = self._prev_amp_obs[terminal]

        prev_amp_norm = self.amp_normalizer.normalize(self._prev_amp_obs)
        next_amp_norm = self.amp_normalizer.normalize(next_amp_term)

        d = self.discriminator.logits(prev_amp_norm, next_amp_norm)
        style_reward = style_reward_from_logit(
            d,
            self.amp_cfg["amp_reward_coef"],
        )
        final_reward = mix_amp_reward(
            style_reward,
            rewards,
            self.amp_cfg["task_reward_lerp"],
        ).squeeze(-1)

        self.replay_buffer.insert(self._prev_amp_obs, next_amp_term)
        self._prev_amp_obs = next_amp.clone()

        self.ppo.process_env_step(obs, final_reward, dones, extras)

        return {
            "style_reward": style_reward.squeeze(-1),
            "final_reward": final_reward,
            "policy_logit": d,
        }

    def compute_returns(self, obs):
        self.ppo.compute_returns(obs)

    def update(self):
        self.update_discriminator()
        loss_dict = self.ppo.update()
        self._apply_min_std()
        return loss_dict

    def _apply_min_std(self):
        min_std_values = self.amp_cfg.get("min_normalized_std")
        if not min_std_values:
            return

        distribution = getattr(self.ppo.actor, "distribution", None)
        if distribution is None:
            return

        min_std = torch.as_tensor(
            min_std_values,
            device=self.device,
            dtype=torch.float32,
        )
        if min_std.numel() == 1:
            min_std = min_std.expand_as(distribution.std_param)

        if hasattr(distribution, "std_param"):
            with torch.no_grad():
                distribution.std_param.clamp_min_(min_std)
        elif hasattr(distribution, "log_std_param"):
            with torch.no_grad():
                distribution.log_std_param.clamp_min_(torch.log(min_std.clamp_min(1e-6)))

    def update_discriminator(self):
        if self.replay_buffer.size < self.disc_batch_size:
            return

        self.discriminator.train()
        mse = nn.MSELoss()

        total_loss = 0.0
        total_acc = 0.0
        total_expert_logit = 0.0
        total_policy_logit = 0.0

        for _ in range(self.disc_updates):
            e_s, e_n = self.amp_loader.sample_batch(self.disc_batch_size)
            p_s, p_n = self.replay_buffer.sample(self.disc_batch_size)

            e_s = torch.from_numpy(e_s).float().to(self.device)
            e_n = torch.from_numpy(e_n).float().to(self.device)
            e_s = e_s + torch.randn_like(e_s) * 0.01
            e_n = e_n + torch.randn_like(e_n) * 0.01
            p_s = p_s.float().to(self.device)
            p_n = p_n.float().to(self.device)

            self.amp_normalizer.update(torch.cat([p_s, e_s], dim=0))

            e_s = self.amp_normalizer.normalize(e_s)
            e_n = self.amp_normalizer.normalize(e_n)
            p_s = self.amp_normalizer.normalize(p_s)
            p_n = self.amp_normalizer.normalize(p_n)

            d_e = self.discriminator.logits(e_s, e_n)
            d_p = self.discriminator.logits(p_s, p_n)

            expert_loss = mse(d_e, 0.9 * torch.ones_like(d_e))
            policy_loss = mse(d_p, -0.9 * torch.ones_like(d_p))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen = self.discriminator.compute_gradient_penalty(e_s, e_n)
            loss = amp_loss + grad_pen

            self.disc_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), max_norm=1.0
            )
            self.disc_optimizer.step()

            total_loss += loss.item()
            total_acc += (
                (d_e > 0).float().mean().item()
                + (d_p < 0).float().mean().item()
            ) / 2.0
            total_expert_logit += d_e.detach().mean().item()
            total_policy_logit += d_p.detach().mean().item()

        n = self.disc_updates
        self.last_disc_loss = total_loss / n
        self.last_disc_acc = total_acc / n
        self.last_expert_logit = total_expert_logit / n
        self.last_policy_logit = total_policy_logit / n

    def train_mode(self):
        self.ppo.train_mode()

    def eval_mode(self):
        self.ppo.eval_mode()

    def get_policy(self):
        return self.ppo.get_policy()

    def save(self):
        return self.ppo.save()

    def load(self, state, strict=True):
        return self.ppo.load(state, strict)
