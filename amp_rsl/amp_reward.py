import torch
import torch.nn.functional as F


def style_reward_from_logit(d, amp_reward_coef=1.0):
    return amp_reward_coef * F.softplus(-d)

def mix_amp_reward(style_reward, task_reward, task_reward_lerp):
    if task_reward_lerp > 0.0:
        return (
            (1.0 - task_reward_lerp) * style_reward
            + task_reward_lerp * task_reward.unsqueeze(-1)
        )
    return style_reward


