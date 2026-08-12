# 第 1 层：SMPL-X 解码，得到人体关节 3D 位置
# 需要先：pip install smplx，并从 SMPL-X 官方页面下载模型文件
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from my_amp.envs.g1_env import G1Env
from my_amp.motion.schema import save_g1_motion
from my_amp.motion.smplx_loader import SMPLX_JOINT_NAMES, build_smplx, decode_smplx_frame


# SMPL-X 关节名 -> G1 body 名
G1_TARGETS = [
    ("pelvis", "pelvis"),
    ("left_hip", "left_hip_pitch_link"),
    ("left_knee", "left_knee_link"),
    ("left_ankle", "left_ankle_pitch_link"),
    ("left_foot", "left_ankle_roll_link"),
    ("right_hip", "right_hip_pitch_link"),
    ("right_knee", "right_knee_link"),
    ("right_ankle", "right_ankle_pitch_link"),
    ("right_foot", "right_ankle_roll_link"),
    ("spine2", "torso_link"),
    ("left_shoulder", "left_shoulder_pitch_link"),
    ("left_elbow", "left_elbow_link"),
    ("left_wrist", "left_wrist_roll_link"),
    ("right_shoulder", "right_shoulder_pitch_link"),
    ("right_elbow", "right_elbow_link"),
    ("right_wrist", "right_wrist_roll_link"),
]


def axis_angle_to_wxyz(rotvec):
    xyzw = Rotation.from_rotvec(rotvec).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def solve_g1_frame(env, targets, q_prev=None):
    q0 = q_prev if q_prev is not None else np.zeros(29)
    low = env.model.jnt_range[1:, 0] - env.default_joint_pos
    high = env.model.jnt_range[1:, 1] - env.default_joint_pos

    def residual(q):
        env.data.qpos[7:] = env.default_joint_pos + q
        mujoco.mj_forward(env.model, env.data)
        errs = []
        for body_name, target in targets.items():
            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            errs.append((env.data.xpos[body_id] - target) / 0.1)
        if q_prev is not None:
            errs.append((q - q_prev) * 0.05)  # 时序平滑正则
        return np.concatenate(errs)

    res = least_squares(residual, q0, bounds=(low, high), max_nfev=50)
    return res.x


def resample_motion(joint_pos, root_pos, root_quat, src_fps, dst_fps):
    t_src = np.arange(len(joint_pos)) / src_fps
    t_dst = np.arange(0.0, t_src[-1], 1.0 / dst_fps)

    joint_new = np.column_stack([
        np.interp(t_dst, t_src, joint_pos[:, j]) for j in range(29)
    ])
    root_pos_new = np.column_stack([
        np.interp(t_dst, t_src, root_pos[:, j]) for j in range(3)
    ])
    # 四元数先用线性插值再归一化，能跑通后再换 slerp
    root_quat_new = np.column_stack([
        np.interp(t_dst, t_src, root_quat[:, j]) for j in range(4)
    ])
    norm = np.linalg.norm(root_quat_new, axis=1, keepdims=True)
    root_quat_new = root_quat_new / norm
    return joint_new, root_pos_new, root_quat_new


def convert_one(sfu_path, output_path, max_frames=None):
    d = np.load(sfu_path, allow_pickle=True)
    poses, trans, betas = d["poses"], d["trans"], d["betas"]
    gender = str(d["gender"])
    src_fps = float(d["mocap_framerate"])

    model = build_smplx(gender)
    env = G1Env("my_amp/envs/unitree_g1/scene.xml")

    n = len(poses) if max_frames is None else min(max_frames, len(poses))

    joint_pos = np.zeros((n, 29), dtype=np.float32)
    root_pos = np.zeros((n, 3), dtype=np.float32)
    root_quat = np.zeros((n, 4), dtype=np.float32)
    q_prev = None

    for i in range(n):
        joints = decode_smplx_frame(model, poses[i], trans[i], betas)
        targets = {}
        for smpl_name, g1_body in G1_TARGETS:
            idx = SMPLX_JOINT_NAMES.index(smpl_name)
            targets[g1_body] = joints[idx]

        env.data.qpos[:3] = trans[i]
        env.data.qpos[3:7] = axis_angle_to_wxyz(poses[i, :3])
        q = solve_g1_frame(env, targets, q_prev)
        q_prev = q
        mujoco.mj_forward(env.model, env.data)

        joint_pos[i] = env.default_joint_pos + q
        root_pos[i] = trans[i]
        root_quat[i] = env.data.qpos[3:7]

        if i % 200 == 0:
            print(f"frame {i}/{n}")

    dst_fps = 1.0 / env.dt  # 250 Hz，和 env 控制频率一致
    joint_pos, root_pos, root_quat = resample_motion(
        joint_pos, root_pos, root_quat, src_fps, dst_fps
    )

    joint_vel = np.gradient(joint_pos, axis=0) / env.dt
    root_lin_vel = np.gradient(root_pos, axis=0) / env.dt
    root_ang_vel = np.zeros_like(root_lin_vel)  # 先填 0，跑通后再精算

    save_g1_motion(
        output_path,
        fps=dst_fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos=root_pos,
        root_quat=root_quat,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
    )
    print("saved", output_path, joint_pos.shape)