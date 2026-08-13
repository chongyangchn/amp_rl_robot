import torch
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
    def __init__(
        self,
        amp_obs_dim,
        amp_reward_coef=0.1,
        task_reward_lerp=0.75,
        hidden_dims=(512, 256, 128),
    ):
        super().__init__()
        input_dim = amp_obs_dim * 2  # 当前帧 + 下一帧

        layers = []
        curr = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr, h))
            layers.append(nn.ReLU())
            curr = h
        layers.append(nn.Linear(curr, 1))

        self.net = nn.Sequential(*layers)
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        return self.net(x)

    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        logit = self.net(expert_data)
        ones = torch.ones_like(logit)
        grad = autograd.grad(
            outputs=logit,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        return lambda_ * (grad.norm(2, dim=1) - 0.0).pow(2).mean()


    def predict_amp_reward(self, state, next_state, task_reward):
        with torch.no_grad():
            self.eval()
            x = torch.cat([state, next_state], dim=-1)
            d = self.net(x)
            style_reward = self.amp_reward_coef * torch.clamp(
                1 - 0.25 * (d - 1).square(), min=0.0
            )
            if self.task_reward_lerp > 0:
                reward = (1 - self.task_reward_lerp) * style_reward \
                       + self.task_reward_lerp * task_reward.unsqueeze(-1)
            else:
                reward = style_reward
            self.train()
        return reward.squeeze(-1), d


    