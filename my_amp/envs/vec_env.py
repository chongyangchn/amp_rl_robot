import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tensordict import TensorDict
from rsl_rl.env import VecEnv
from my_amp.envs.g1_env import G1Env


class G1VecEnv(VecEnv):
    def __init__(
        self,
        num_envs=16,
        xml_path="my_amp/envs/unitree_g1/scene.xml",
        max_episode_length=5000,
        amp_body_names=None,
        amp_anchor_name=None,
        motion_loader=None,
        reset_from_ref_prob=0.8,
    ):
        self.num_envs = num_envs
        self.num_actions = 29
        self.max_episode_length = max_episode_length
        self.device = "cpu"
        self.cfg = {"env_name": "g1_ppo", "num_envs": num_envs}
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.envs = [
            G1Env(
                xml_path,
                amp_body_names=amp_body_names,
                amp_anchor_name=amp_anchor_name,
            )
            for _ in range(num_envs)
        ]

        self.motion_loader = motion_loader
        self.reset_from_ref_prob = reset_from_ref_prob
        self.rng = np.random.default_rng(0)
        self._executor = ThreadPoolExecutor(max_workers=min(16, num_envs))

    def _make_td(self, obs_list):
        policy = torch.from_numpy(np.stack([o["policy"] for o in obs_list])).float()
        critic = torch.from_numpy(np.stack([o["critic"] for o in obs_list])).float()
        amp = torch.from_numpy(np.stack([o["amp"] for o in obs_list])).float()
        return TensorDict(
            {"policy": policy, "critic": critic, "amp": amp},
            batch_size=[self.num_envs],
        )

    def get_observations(self):
        return self._make_td([env.get_obs_vec() for env in self.envs])

    def reset_all_to_ref(self):
        if self.motion_loader is None:
            return self.get_observations()

        obs_list = []
        for env in self.envs:
            qpos, qvel, command = self.motion_loader.get_reset_state_with_command(self.rng)
            obs_list.append(env.reset_to_ref(qpos, qvel, command))
        return self._make_td(obs_list)

    def step(self, actions):
        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length

        actions_np = actions.cpu().numpy()
        futures = [
            self._executor.submit(self.envs[i].step, actions_np[i])
            for i in range(self.num_envs)
        ]
        step_results = [future.result() for future in futures]

        obs_list = [None] * self.num_envs
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        reward_terms_list = []
        termination_counts = {}

        for i, (obs, reward, done, info) in enumerate(step_results):
            reward_terms_list.append(info.get("rewards", {}))

            done = bool(done) or bool(time_outs[i])
            reason = info.get("termination", "")
            if done:
                if reason == "":
                    reason = "time_out" if time_outs[i] else "unknown"
                termination_counts[reason] = termination_counts.get(reason, 0) + 1

                if self.motion_loader is not None and self.rng.random() < self.reset_from_ref_prob:
                    qpos, qvel, command = self.motion_loader.get_reset_state_with_command(self.rng)
                    obs = self.envs[i].reset_to_ref(qpos, qvel, command)
                else:
                    obs = self.envs[i].reset()

            obs_list[i] = obs
            rewards[i] = reward
            dones[i] = done

        dones_t = torch.from_numpy(dones)
        self.episode_length_buf[dones_t] = 0

        if reward_terms_list:
            reward_means = {}
            for key in reward_terms_list[0]:
                reward_means[key] = float(
                    np.mean([term.get(key, 0.0) for term in reward_terms_list])
                )
            self.last_reward_terms = reward_means
        else:
            self.last_reward_terms = None

        self.last_termination_reasons = termination_counts

        return (
            self._make_td(obs_list),
            torch.from_numpy(rewards),
            dones_t,
            {
                "time_outs": time_outs,
                "termination_reasons": termination_counts,
                "reward_terms": reward_terms_list,
            },
        )
