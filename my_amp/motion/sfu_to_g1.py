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
# 因为 SMPL-X 给出的 joints 是“空间中的点”，而 MuJoCo 的 body xpos 也是“空间中的点”。
# 我们让 G1 的 body 位置尽量接近 SMPL-X 的对应关节点。


def axis_angle_to_wxyz(rotvec):
    xyzw = Rotation.from_rotvec(rotvec).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def solve_g1_frame(env, targets, q_prev=None, target_up=None):
    q0 = q_prev if q_prev is not None else np.zeros(29)
    low = env.model.jnt_range[1:, 0] - env.default_joint_pos
    high = env.model.jnt_range[1:, 1] - env.default_joint_pos

    def residual(q):
        env.data.qpos[7:] = env.default_joint_pos + q
        mujoco.mj_forward(env.model, env.data)

        errs = []

        # 位置误差
        for body_name, target in targets.items():
            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            errs.append((env.data.xpos[body_id] - target) / 0.1)

        # torso 朝上误差
        if target_up is not None:
            torso_xmat = env.data.site_xmat[env.torso_site].reshape(3, 3)
            current_up = torso_xmat[:, 2]
            errs.append((current_up - target_up) * 10.0)

            # 躯干横轴尽量水平
            current_right = torso_xmat[:, 1]
            errs.append(np.array([(current_right[2] - 0.0) * 40.0]))

        # 脚底高度和左右对称误差
        left_foot_z = env.data.site_xpos[env.left_foot_site][2]
        right_foot_z = env.data.site_xpos[env.right_foot_site][2]

        errs.append(np.array([max(0.0, 0.02 - left_foot_z) * 30.0]))
        errs.append(np.array([max(0.0, 0.02 - right_foot_z) * 30.0]))


        if left_foot_z < 0.08 and right_foot_z < 0.14:
            errs.append(np.array([(left_foot_z - right_foot_z) * 40.0]))

        if q_prev is not None:
            errs.append((q - q_prev) * 0.05)

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

def rotvec_to_yaw_wxyz(rotvec):
    euler = Rotation.from_rotvec(rotvec).as_euler("ZYX", degrees=False)
    yaw = euler[0]
    q_xyzw = Rotation.from_euler("Z", yaw).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def convert_one(sfu_path, output_path, max_frames=None):
    d = np.load(sfu_path, allow_pickle=True)
    poses, trans, betas = d["poses"], d["trans"], d["betas"]
    gender = str(d["gender"])
    src_fps = float(d["mocap_framerate"])

    model = build_smplx(gender)
    env = G1Env("my_amp/envs/unitree_g1/scene.xml")

    n = len(poses) if max_frames is None else min(max_frames, len(poses))

    joint_pos = np.zeros((n, 29), dtype=np.float32)
    # 初始化body数组
    nbody = env.model.nbody - 1
    body_pos_w = np.zeros((n, nbody, 3), dtype=np.float32)
    body_quat_w = np.zeros((n, nbody, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((n, nbody, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((n, nbody, 3), dtype=np.float32)
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
        # env.data.qpos[3:7] = axis_angle_to_wxyz(poses[i, :3])

        yaw_quat = rotvec_to_yaw_wxyz(poses[i, :3])
        env.data.qpos[3:7] = yaw_quat

        q = solve_g1_frame(
                env,
                targets,
                q_prev,
                target_up=np.array([0.0, 0.0, 1.0]),
            )
        q_prev = q

        # 用 G1 的脚底 site 修正 root 高度，避免悬空
        mujoco.mj_forward(env.model, env.data)
        left_z = env.data.site_xpos[env.left_foot_site][2]
        right_z = env.data.site_xpos[env.right_foot_site][2]
        ground_offset = min(left_z, right_z) - 0.02

        root_pos[i] = trans[i].copy()
        root_pos[i, 2] -= ground_offset
        root_quat[i] = env.data.qpos[3:7].copy()

        # 用修正后的 root 重新 forward，然后采集 body 特征
        env.data.qpos[:3] = root_pos[i]
        mujoco.mj_forward(env.model, env.data)

        joint_pos[i] = env.default_joint_pos + q
        body_pos_w[i] = env.data.xpos[1:]
        body_quat_w[i] = env.data.xquat[1:]
        body_lin_vel_w[i] = env.data.cvel[1:, 3:]
        body_ang_vel_w[i] = env.data.cvel[1:, :3]

        if i % 200 == 0:
            print(f"frame {i}/{n}")

    dst_fps = 1.0 / env.dt  # 250 Hz，和 env 控制频率一致
    joint_pos, root_pos, root_quat = resample_motion(
        joint_pos, root_pos, root_quat, src_fps, dst_fps
    )

    joint_vel = np.gradient(joint_pos, axis=0) / env.dt
    root_lin_vel = np.gradient(root_pos, axis=0) / env.dt

    # 先求四元数的局部角速度；够训练用，后续可以换成标准 quaternion derivative
    rotvec = Rotation.from_quat(
        root_quat[:, [1, 2, 3, 0]]
    ).as_rotvec()
    root_ang_vel = np.gradient(rotvec, axis=0) / env.dt

    T = len(joint_pos)
    body_pos_w = np.zeros((T, nbody, 3), dtype=np.float32)
    body_quat_w = np.zeros((T, nbody, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((T, nbody, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((T, nbody, 3), dtype=np.float32)

    for i in range(T):
        env.data.qpos[:3] = root_pos[i]
        env.data.qpos[3:7] = root_quat[i]
        env.data.qpos[7:] = joint_pos[i]

        env.data.qvel[:3] = root_lin_vel[i]
        env.data.qvel[3:6] = root_ang_vel[i]
        env.data.qvel[6:] = joint_vel[i]

        mujoco.mj_forward(env.model, env.data)

        body_pos_w[i] = env.data.xpos[1:]
        body_quat_w[i] = env.data.xquat[1:]
        body_lin_vel_w[i] = env.data.cvel[1:, 3:]
        body_ang_vel_w[i] = env.data.cvel[1:, :3]

    save_g1_motion(
            output_path,
            fps=dst_fps,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            root_pos=root_pos,
            root_quat=root_quat,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
        )  
    print("saved", output_path, joint_pos.shape)