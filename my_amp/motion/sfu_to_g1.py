# 第 1 层：SMPL-X 解码，得到人体关节 3D 位置
# 需要先：pip install smplx，并从 SMPL-X 官方页面下载模型文件
from smplx import SMPLX
import numpy as np
import mujoco

smplx_model = SMPLX(
    model_path="data/smplx",      # 放官方下载的模型文件
    gender = "female",
    num_betas=16,
    use_pca=False,
    flat_hand_mean=True,
)

# poses 拆分：全局3 + 身体63 + 左手45 + 右手45
global_orient = poses[:, :3]
body_pose = poses[:, 3:66]
left_hand_pose = poses[:, 66:111]
right_hand_pose = poses[:, 111:156]

# 第 2 层：把 SMPL-X 关节位置映射到 G1 的关键点
# 例如：SMPL 的 left_hip / left_knee / left_ankle
#       -> G1 的 hip / knee / ankle body 位置
TARGET_JOINTS = {
    "left_hip":    "left_hip_yaw_link",
    "left_knee":   "left_knee_link",
    "left_ankle":  "left_ankle_pitch_link",
    "right_hip":   "right_hip_yaw_link",
    "right_knee":  "right_knee_link",
    "right_ankle": "right_ankle_pitch_link",
    "left_shoulder": "left_shoulder_pitch_link",
    "left_elbow":  "left_elbow_link",
    "left_wrist":  "left_wrist_roll_link",
    "right_shoulder": "right_shoulder_pitch_link",
    "right_elbow": "right_elbow_link",
    "right_wrist": "right_wrist_roll_link",
}


# 第 3 层：对每一帧解 G1 的 29 个关节角
# 用 MuJoCo FK + scipy 优化：让 G1 关键点尽量贴近 SMPL 目标点
from scipy.optimize import least_squares

def solve_g1_frame(targets: dict[str, np.ndarray], q0: np.ndarray) -> np.ndarray:
    def residual(q):
        env.data.qpos[7:] = env.default_joint_pos + q
        mujoco.mj_forward(env.model, env.data)
        errs = []
        for body_nam, target_pos in targets.items():
            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            errs.append((env.data.xpos[body_id] - target_pos) / 0.1)

        return np.concatenate(errs)

    res = least_squares(residual, q0, bounds=(-1.5, 1.5), max_nfev=50)
    return res.x