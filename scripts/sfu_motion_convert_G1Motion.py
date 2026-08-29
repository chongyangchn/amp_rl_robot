from my_amp.motion.sfu_to_g1 import convert_one


path_sfu_raw = "data/sfu_raw/SFU/0008/0008_Walking001_poses.npz"
path_output = "data/motions/sfu_walking001.npz"

convert_one(
    path_sfu_raw,
    path_output,
    max_frames=200,
)

 