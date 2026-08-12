from my_amp.motion.sfu_to_g1 import convert_one

convert_one(
    "data/sfu_raw/SFU/0008/0008_Walking001_poses.npz",
    "data/motions/sfu_walking001.npz",
    max_frames=200,
)

# if __name__ == "__main__"
