import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


G1_XML = "my_amp/envs/unitree_g1/scene.xml"


def quat_to_wxyz(quat_xyzw):
    """Convert xyzw quaternion to MuJoCo's wxyz order."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def slerp_quats(quat_xyzw, src_times, dst_times):
    """Slerp xyzw quaternions to destination times."""
    quat_wxyz = np.stack([quat_to_wxyz(q) for q in quat_xyzw])
    rot = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    slerp = Slerp(src_times, rot)
    out_xyzw = slerp(dst_times).as_quat()
    out_wxyz = np.stack([
        out_xyzw[:, 3],
        out_xyzw[:, 0],
        out_xyzw[:, 1],
        out_xyzw[:, 2],
    ], axis=-1)
    return out_wxyz


def resample_1d(values, src_times, dst_times):
    return np.column_stack([
        np.interp(dst_times, src_times, values[:, i])
        for i in range(values.shape[1])
    ])


def convert_csv_to_npz(input_file, output_file, input_fps, output_fps=250.0):
    data = np.loadtxt(input_file, delimiter=",", dtype=np.float64)

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] != 36:
        raise ValueError(f"Expected 36 columns, got {data.shape[1]}")

    src_times = np.arange(data.shape[0], dtype=np.float64) / input_fps
    duration = src_times[-1]
    dst_times = np.arange(0.0, duration, 1.0 / output_fps, dtype=np.float64)

    root_pos_src = data[:, 0:3]
    root_quat_src = data[:, 3:7]
    joint_pos_src = data[:, 7:36]

    root_pos = resample_1d(root_pos_src, src_times, dst_times)
    joint_pos = resample_1d(joint_pos_src, src_times, dst_times)
    root_quat = slerp_quats(root_quat_src, src_times, dst_times)

    dt = 1.0 / output_fps
    root_lin_vel = np.gradient(root_pos, axis=0) / dt
    joint_vel = np.gradient(joint_pos, axis=0) / dt

    root_rot = Rotation.from_quat(root_quat[:, [1, 2, 3, 0]])
    root_ang_vel = np.zeros_like(root_lin_vel)
    for i in range(len(dst_times)):
        if i + 1 < len(dst_times):
            dq = (root_rot[i + 1] * root_rot[i].inv()).as_rotvec()
            root_ang_vel[i] = dq / dt
        else:
            root_ang_vel[i] = root_ang_vel[i - 1]

    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)
    nbody = model.nbody - 1

    body_pos_w = np.zeros((len(dst_times), nbody, 3), dtype=np.float32)
    body_quat_w = np.zeros((len(dst_times), nbody, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((len(dst_times), nbody, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((len(dst_times), nbody, 3), dtype=np.float32)

    qpos = np.zeros(36, dtype=np.float64)
    qvel = np.zeros(35, dtype=np.float64)

    for i in range(len(dst_times)):
        qpos[0:3] = root_pos[i]
        qpos[3:7] = root_quat[i]
        qpos[7:36] = joint_pos[i]

        qvel[0:3] = root_lin_vel[i]
        qvel[3:6] = root_ang_vel[i]
        qvel[6:35] = joint_vel[i]

        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)

        body_pos_w[i] = data.xpos[1:]
        body_quat_w[i] = data.xquat[1:]
        body_lin_vel_w[i] = data.cvel[1:, 3:]
        body_ang_vel_w[i] = data.cvel[1:, :3]

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_file,
        fps=np.asarray([output_fps], dtype=np.float64),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        root_pos=root_pos.astype(np.float32),
        root_quat=root_quat.astype(np.float32),
        root_lin_vel=root_lin_vel.astype(np.float32),
        root_ang_vel=root_ang_vel.astype(np.float32),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )

    print(f"Saved {output_file} with {len(dst_times)} frames at {output_fps} fps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--input-fps", type=float, required=True)
    parser.add_argument("--output-fps", type=float, default=250.0)
    args = parser.parse_args()

    convert_csv_to_npz(
        args.input_file,
        args.output_file,
        args.input_fps,
        args.output_fps,
    )


if __name__ == "__main__":
    main()
