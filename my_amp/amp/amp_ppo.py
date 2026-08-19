import torch
import torch.nn as nn

class AMPPPO:
    def __init__(self, ppo, discriminator, amp_loader, replay_buffer, amp_cfg, device):
        self.ppo = ppo
        self.discriminator = discriminator.to(device)
        self.amp_loader = amp_loader
        self.replay_buffer = replay_buffer
        self.device = device
        self.amp_cfg = amp_cfg

        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=amp_cfg["disc_learning_rate"],
        )

        self.disc_batch_size = amp_cfg["disc_batch_size"]
        self.disc_updates = amp_cfg.get("disc_updates_per_iter", 1)

        self._prev_amp_obs = None
        self.last_disc_loss = 0.0
        self.last_disc_acc = 0.0

    def act(self, obs):
        self._prev_amp_obs = obs["amp"].clone()
        return self.ppo.act(obs)  

    def process_env_step(self, obs, rewards, dones, extras):
        next_amp = obs["amp"].clone()

        # 如果 episode 结束，下一帧是新 episode 的起点，
        # 不能用它当 next，否则会产生“跨 episode”的假 transition
        terminal = dones.bool()
        next_amp_term = next_amp.clone()
        next_amp_term[terminal] = self._prev_amp_obs[terminal]

        # 风格奖励 + 任务奖励混合（内部已完成）
        final_reward, d = self.discriminator.predict_amp_reward(
            self._prev_amp_obs,
            next_amp_term,
            rewards,
        )

        # 策略 transition 存进回放缓冲（负样本）
        self.replay_buffer.insert(self._prev_amp_obs, next_amp_term)
        self._prev_amp_obs = next_amp.clone()

        # 交给普通 PPO 存 rollout、更新 normalizer
        self.ppo.process_env_step(obs, final_reward, dones, extras)

    def compute_returns(self, obs):
        self.ppo.compute_returns(obs)

    def update(self):
        self.update_discriminator()
        return self.ppo.update()

    def update_discriminator(self):
        if self.replay_buffer.size < self.disc_batch_size:
            return

        self.discriminator.train()
        mse = nn.MSELoss()
        total_loss = 0.0
        total_acc = 0.0

        for _ in range(self.disc_updates):
            e_s, e_n = self.amp_loader.sample_batch(self.disc_batch_size)
            p_s, p_n = self.replay_buffer.sample(self.disc_batch_size)

            e_s = torch.from_numpy(e_s).float().to(self.device)
            e_n = torch.from_numpy(e_n).float().to(self.device)
            p_s = p_s.float().to(self.device)
            p_n = p_n.float().to(self.device)

            d_e = self.discriminator(torch.cat([e_s, e_n], dim=-1))
            d_p = self.discriminator(torch.cat([p_s, p_n], dim=-1))

            expert_loss = mse(d_e, torch.ones_like(d_e))
            policy_loss = mse(d_p, -1.0 * torch.ones_like(d_p))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen = self.discriminator.compute_grad_pen(e_s, e_n)
            loss = amp_loss + grad_pen

            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()

            acc = ((d_e > 0).float().mean() + (d_p < 0).float().mean()) / 2
            total_loss += loss.item()
            total_acc += acc.item()

        self.last_disc_loss = total_loss / self.disc_updates
        self.last_disc_acc = total_acc / self.disc_updates

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

     