import torch

from my_amp.motion.amp_loader import AMPLoader as _AMPLoader


class AMPLoader(_AMPLoader):
    def __init__(self, motion_file, body_names, anchor_name, all_body_names, device="cpu"):
        super().__init__(motion_file, body_names, anchor_name, all_body_names)
        self._device = torch.device(device)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for s, s_next in super().feed_forward_generator(num_mini_batch, mini_batch_size):
            yield s.to(self._device), s_next.to(self._device)

__all__ = ["AMPLoader"]
