import torch
import numpy as np
from tensordict import TensorDict
from rsl_rl.rsl_rl.env import VecEnv
from my_amp.envs.g1_env import G1Env


class G1VecEnv(VecEnv):
    def __init__(
        self,
        num_envs=16,
        xml_path="my_amp/envs/unitree_g1/scene.xml",
        max_episode_length=300,
        amp_body_names=None,
        amp_anchor_name=None,
        motion_loader=None,
        reset_from_ref_prob=0.8,
    ):
        self.num_envs = num_envs
        # self.num_actions = self.envs[0].action_dim
        self.num_actions = 29
        self.max_episode_length = max_episode_length
        self.device = "cpu"
        self.cfg = {"env_name" : "g1_ppo", "num_envs" : num_envs}
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.envs = [
            G1Env(
                xml_path,
                amp_body_names=amp_body_names,
                amp_anchor_name=amp_anchor_name,
            )
            for _ in range(num_envs)
        ]
        # super().__init__()

        self.motion_loader = motion_loader
        self.reset_from_ref_prob = reset_from_ref_prob
        self.rng = np.random.default_rng(0)
    

    def _make_td(self, obs_list):
        policy = torch.from_numpy(np.stack([o["policy"] for o in obs_list])).float()
        critic = torch.from_numpy(np.stack([o["critic"] for o in obs_list])).float()
        amp = torch.from_numpy(np.stack([o["amp"] for o in obs_list])).float()
        return TensorDict(
            {"policy" : policy, "critic" : critic, "amp" : amp},
            batch_size=[self.num_envs],
        )

    def get_observations(self):
        return self._make_td([e.get_obs_vec() for e in self.envs])

    def step(self, actions):
        obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)

        # 每个环境本轮都推进一次 episode length
        self.episode_length_buf += 1

        # 先判断哪些环境这一轮达到最大长度
        time_outs = self.episode_length_buf >= self.max_episode_length

        for i, env in enumerate(self.envs):
            a = actions[i].cpu().numpy()
            o, r, d, info = env.step(a)

            # 物理 done 或 timeout 都结束
            d = bool(d) or bool(time_outs[i])
            if d:
                o = env.reset()

            obs_list.append(o)
            rewards[i] = r
            dones[i] = d

        dones_t = torch.from_numpy(dones)
        self.episode_length_buf[dones_t] = 0

        return (
            self._make_td(obs_list),
            torch.from_numpy(rewards),
            dones_t,
            {"time_outs": time_outs},
        )
        
