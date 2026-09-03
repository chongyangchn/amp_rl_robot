from pathlib import Path
import numpy as np
import torch

from my_amp.amp.amp_obs import compute_amp_features


class AMPLoader:
    # 从动作文件中进行加载和采样，提供 AMP 判别器的观测数据
    def __init__(self, motion_dir, body_names, anchor_name, all_body_names):
        self.body_idx = [all_body_names.index(n) for n in body_names]
        self.anchor_idx = all_body_names.index(anchor_name)
        self.features = []

        for path in sorted(Path(motion_dir).glob("*.npz")):
            d = np.load(path)

            # 让 AMPLoader 跳过不合格文件
            required = [
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ]
            if not all(k in d for k in required):
                print(f"skip {path.name}: missing AMP body fields")
                continue

            pos_w = d["body_pos_w"]
            quat_w = d["body_quat_w"]
            lin_vel_w = d["body_lin_vel_w"]
            ang_vel_w = d["body_ang_vel_w"]
            T = pos_w.shape[0]

            feats = np.stack([
                compute_amp_features(
                    pos_w[i], quat_w[i], lin_vel_w[i], ang_vel_w[i],
                    self.body_idx, self.anchor_idx,
                )
                for i in range(T)
            ])
            self.features.append(feats)
            print(f"loaded {path.name}: {T} frames, dim={feats.shape[1]}")

        if not self.features:
            raise FileNotFoundError(f"no npz in {motion_dir}")

    @property
    def observation_dim(self):
        return self.features[0].shape[1]

    def sample_batch(self, batch_size, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        s_list = []
        s_next_list = []

        for _ in range(batch_size):
            motion_idx = int(rng.integers(0, len(self.features)))
            T = len(self.features[motion_idx])
            idx = int(rng.integers(0, T - 1))
            s_list.append(self.features[motion_idx][idx])
            s_next_list.append(self.features[motion_idx][idx + 1])

        return np.stack(s_list), np.stack(s_next_list)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            s, s_next = self.sample_batch(mini_batch_size)
            yield torch.from_numpy(s).float(), torch.from_numpy(s_next).float()

    
