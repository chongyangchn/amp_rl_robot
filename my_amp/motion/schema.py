import numpy as np

G1_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

def save_g1_motion(
        path, 
        fps,
        joint_pos,          # (T, 29)
        joint_vel,          # (T, 29)
        root_pos,           # (T, 3)
        root_quat,          # (T, 4) wxyz
        root_lin_vel,       # (T, 3)
        root_ang_vel,       # (T, 3)
        body_pos_w=None,    # (T, 30, 3)
        body_quat_w=None,
        body_lin_vel_w=None,
        body_ang_vel_w=None,
):
    np.savez(
        path,
        fps=np.asarray([fps], dtype=np.float64),
        joint_pos=np.asarray(joint_pos, dtype=np.float32),
        joint_vel=np.asarray(joint_vel, dtype=np.float32),
        root_pos=np.asarray(root_pos, dtype=np.float32),
        root_quat=np.asarray(root_quat, dtype=np.float32),
        root_lin_vel=np.asarray(root_lin_vel, dtype=np.float32),
        root_ang_vel=np.asarray(root_ang_vel, dtype=np.float32),
        body_pos_w=np.asarray(body_pos_w, dtype=np.float32),
        body_quat_w=np.asarray(body_quat_w, dtype=np.float32),
        body_lin_vel_w=np.asarray(body_lin_vel_w, dtype=np.float32),
        body_ang_vel_w=np.asarray(body_ang_vel_w, dtype=np.float32),
    )