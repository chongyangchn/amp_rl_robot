import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.pos = 0
        self.size = 0

    def insert(self, obs, next_obs):
        n = obs.shape[0]
        idx = torch.arange(self.pos, self.pos + n, device=self.device) % self.capacity
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError(f"replay size {self.size} < batch {batch_size}")
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[idx], self.next_obs[idx]
