import numpy as np
from my_amp.amp.amp_obs import quat_mul, quat_conjugate, quat_to_rotmat
import mujoco
import torch
from my_amp.motion.amp_loader import AMPLoader
from my_amp.envs.g1_env import G1Env
from my_amp.amp.discriminator import Discriminator
from my_amp.motion.motion_loader import MotionLoader


def test1():
    # 验证 1：四元数工具函数单测
    q_identity = np.array([1.0, 0, 0, 0])
    assert np.allclose(quat_conjugate(q_identity), q_identity)

    q1 = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])
    q2 = np.array([np.sqrt(2)/2, 0, np.sqrt(2)/2, 0])
    assert np.allclose(np.linalg.norm(quat_mul(q1, q2)), 1.0)

    R = quat_to_rotmat(q1)
    assert np.allclose(R @ R.T, np.eye(3))
    print("quat helpers OK")


def test2():
    # 验证 2：AMPLoader 特征维度和数值
    model = mujoco.MjModel.from_xml_path("my_amp/envs/unitree_g1/scene.xml")
    all_body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(1, model.nbody)
    ]

    body_names = [
        "pelvis", "torso_link",
        "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link",
        "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link",
        "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_pitch_link",
        "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_pitch_link",
    ]

    loader = AMPLoader("data/motions", body_names, "pelvis", all_body_names)
    assert loader.observation_dim == len(body_names) * 15    
    print("amp dim =", loader.observation_dim)
    s, s_next = loader.sample_batch(64, np.random.default_rng(0))
    print("batch shapes:", s.shape, s_next.shape)
    assert np.isfinite(s).all() and np.isfinite(s_next).all()
    print("AMP features finite OK")


def test3():
    # 环境侧 AMP 观测
    env = G1Env("my_amp/envs/unitree_g1/scene.xml")
    model = mujoco.MjModel.from_xml_path("my_amp/envs/unitree_g1/scene.xml")
    all_body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(1, model.nbody)
    ]
    body_names = [
        "pelvis", "torso_link",
        "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link",
        "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link",
        "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_pitch_link",
        "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_pitch_link",
    ]

    body_idx = [all_body_names.index(n) for n in body_names]
    anchor_idx = all_body_names.index("pelvis")

    amp_obs = env.get_amp_obs(body_idx, anchor_idx)
    print("env amp obs:", amp_obs.shape)
    assert amp_obs.shape == (210,)
    assert np.isfinite(amp_obs).all()
    print("env amp obs OK")

def test4():
    # 判别器冒烟训练
    model = mujoco.MjModel.from_xml_path("my_amp/envs/unitree_g1/scene.xml")
    all_body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(1, model.nbody)
    ]

    body_names = [
        "pelvis", "torso_link",
        "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link",
        "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link",
        "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_pitch_link",
        "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_pitch_link",
    ]

    loader = AMPLoader("data/motions", body_names, "pelvis", all_body_names)
    assert loader.observation_dim == len(body_names) * 15    
    print("amp dim =", loader.observation_dim)
    s, s_next = loader.sample_batch(64, np.random.default_rng(0))
    print("batch shapes:", s.shape, s_next.shape)
    assert np.isfinite(s).all() and np.isfinite(s_next).all()
    print("AMP features finite OK")


    disc = Discriminator(amp_obs_dim=loader.observation_dim)
    opt = torch.optim.Adam(disc.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for step in range(200):
        e_s, e_n = loader.sample_batch(64)
        e_s = torch.from_numpy(e_s).float()
        e_n = torch.from_numpy(e_n).float()

        p_s = torch.randn_like(e_s) * 0.5
        p_n = torch.randn_like(e_n) * 0.5

        d_e = disc(torch.cat([e_s, e_n], dim=-1))
        d_p = disc(torch.cat([p_s, p_n], dim=-1))

        loss = (
            loss_fn(d_e, torch.ones_like(d_e))
            + loss_fn(d_p, torch.zeros_like(d_p))
            + 0.1 * disc.compute_grad_pen(e_s, e_n)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            acc = (
                (d_e.sigmoid() > 0.5).float().mean()
                + (d_p.sigmoid() < 0.5).float().mean()
            ) / 2
            print(f"step {step:3d} loss {loss.item():.3f} acc {acc.item():.3f}")

def test5():
    # 真实策略数据（最接近第 5 步）
    motion_loader = MotionLoader("data/motions")
    env = G1Env("my_amp/envs/unitree_g1/scene.xml")


    model = mujoco.MjModel.from_xml_path("my_amp/envs/unitree_g1/scene.xml")
    all_body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(1, model.nbody)
    ]
    body_names = [
        "pelvis", "torso_link",
        "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link",
        "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link",
        "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_pitch_link",
        "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_pitch_link",
    ]

    body_idx = [all_body_names.index(n) for n in body_names]
    anchor_idx = all_body_names.index("pelvis")

    amp_obs = env.get_amp_obs(body_idx, anchor_idx)
    print("env amp obs:", amp_obs.shape)
    assert amp_obs.shape == (210,)
    assert np.isfinite(amp_obs).all()
    print("env amp obs OK")


    policy_pairs = []
    rng = np.random.default_rng(0)

    for _ in range(10):
        qpos, qvel = motion_loader.get_reset_state(rng)
        env.reset_to_ref(qpos, qvel)
        s = env.get_amp_obs(body_idx, anchor_idx)

        for _ in range(100):
            action = rng.uniform(-1.0, 1.0, 29)
            obs, reward, done, _ = env.step(action)
            s_next = env.get_amp_obs(body_idx, anchor_idx)
            policy_pairs.append((s, s_next))
            s = s_next
            if done:
                break

    print("collected policy pairs:", len(policy_pairs))

# def test6():


if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
