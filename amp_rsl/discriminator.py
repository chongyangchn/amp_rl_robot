import torch
import torch.nn as nn
from torch import autograd

class AMPDiscriminator(nn.Module):
    def __init__(self, amp_obs_dim, hidden_dims=(128, 64)):
        super().__init__()
        input_dim = amp_obs_dim * 2

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def compute_gradient_penalty(self, expert_state, expert_next_state, lambda_=20.0):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad_(True)

        logit = self.forward(expert_data)
        grad = autograd.grad(
            outputs=logit,
            inputs=expert_data,
            grad_outputs=torch.ones_like(logit),
            create_graph=True,
            retain_graph=True,
        )[0]

        # return lambda_ * (grad.norm(2, dim=1) - 1.0).pow(2).mean()

        return lambda_ * (grad.norm(2, dim=1) - 0.0).pow(2).mean()


    def logits(self, state, next_state):
        return self.forward(torch.cat([state, next_state], dim=-1))
