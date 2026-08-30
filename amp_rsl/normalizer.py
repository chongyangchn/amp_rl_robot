import torch


class AMPNormalizer:
    def __init__(self, shape, device="cpu", eps=1e-5):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 0
        self.eps = eps

    def update(self, x):
        x = x.detach().float().to(self.mean.device)

        batch_count = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        total = self.count + batch_count
        delta = batch_mean - self.mean

        self.mean = self.mean + (batch_count / total) * delta
        self.var = (
            self.count * self.var
            + batch_count * batch_var
            + (self.count * batch_count / total) * delta * delta
        ) / total
        self.count = total

    def normalize(self, x):
        x = x.float().to(self.mean.device)
        return (x - self.mean) / (self.var.sqrt() + self.eps)

    def state_dict(self):
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state):
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]