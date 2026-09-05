from my_amp.motion.sfu_to_g1 import convert_one


def main():
    convert_one(
        "data/sfu_raw/SFU/0018/0018_TraditionalChineseDance001_poses.npz",
        "data/motions_dance/traditional_chinese_dance001.npz",
        max_frames=400,
    )


if __name__ == "__main__":
    main()
