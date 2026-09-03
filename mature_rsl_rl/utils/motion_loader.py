from my_amp.motion.amp_loader import AMPLoader as _AMPLoader


class AMPLoader(_AMPLoader):
    def __init__(self, motion_file, body_names, anchor_name, all_body_names, device="cpu"):
        super().__init__(motion_file, body_names, anchor_name, all_body_names)

__all__ = ["AMPLoader"]
