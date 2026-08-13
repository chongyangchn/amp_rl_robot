import numpy as np
import torch
import mujoco

from my_amp.amp.amp_obs import (
    quat_conjugate,
    quat_mul,
    quat_to_rotmat,
)
from my_amp.amp.discriminator import Discriminator
from my_amp.motion.amp_loader import AMPLoader
from my_amp.motion.loader import MotionLoader
from my_amp.envs.g1_env import G1Env

BODY_NAMES = [
    "pelvis", "torso_link",
    "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link",
    "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link",
    "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_pitch_link",
    "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_pitch_link",
]

ANCHOR_NAME = "pelvis"

MOTION_DIR = "data/motions"
XML_PATH = "my_amp/envs/unitree_g1/scene.xml"


def get_all_body_names(model):
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(1, model.nbody)
    ]


def test_quat_helpers():
    print("=== 1. quaternion helpers ===")
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quat_conjugate(q_identity), q_identity)

    q1 = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])
    q2 = np.array([np.sqrt(2)/2, 0, np.sqrt(2)/2, 0])
    assert np.allclose(np.linalg.norm(quat_mul(q1, q2)), 1.0)

    R = quat_to_rotmat(q1)
    assert np.allclose(R @ R.T, np.eye(3))
    print("quat helpers OK")


def test_amp_loader():
    print("=== 2. reference AMP features ===")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    all_body_names = get_all_body_names(model)

    loader = AMPLoader(MOTION_DIR, BODY_NAMES, ANCHOR_NAME, all_body_names)
    expected_dim = len(BODY_NAMES) * 15
    assert loader.observation_dim == expected_dim, (
        loader.observation_dim, expected_dim
    )
    print("amp dim =", loader.observation_dim)

    s, s_next = loader.sample_batch(64, np.random.default_rng(0))
    print("batch shapes:", s.shape, s_next.shape)
    assert np.isfinite(s).all()
    assert np.isfinite(s_next).all()
    print("AMP features finite OK")
    return loader, all_body_names

def test_env_amp_obs(all_body_names):
    print("=== 3. env AMP observation ===")
    env = G1Env(XML_PATH)
    body_idx = [all_body_names.index(n) for n in BODY_NAMES]
    anchor_idx = all_body_names.index(ANCHOR_NAME)

    amp_obs = env.get_amp_obs(body_idx, anchor_idx)
    print("env amp obs:", amp_obs.shape)
    assert amp_obs.shape == (len(BODY_NAMES) * 15,)
    assert np.isfinite(amp_obs).all()
    print("env amp obs OK")
    return env, body_idx, anchor_idx


def collect_policy_pairs(env, body_idx, anchor_idx, num_episodes=10, steps=100):
    print("=== 4. collect policy transitions ===")
    motion_loader = MotionLoader(MOTION_DIR)
    rng = np.random.default_rng(0)
    pairs = []

    for _ in range(num_episodes):
        qpos, qvel = motion_loader.get_reset_state(rng)
        env.reset_to_ref(qpos, qvel)
        s = env.get_amp_obs(body_idx, anchor_idx)

        for _ in range(steps):
            action = rng.uniform(-1.0, 1.0, 29)
            _, reward, done, _ = env.step(action)
            s_next = env.get_amp_obs(body_idx, anchor_idx)
            pairs.append((s, s_next))
            s = s_next
            if done:
                break

    print("collected policy pairs:", len(pairs))
    assert len(pairs) > 0
    return pairs

def train_discriminator(loader, policy_pairs, steps=300, batch_size=64):
    print("=== 5. discriminator: reference vs policy ===")
    policy_s = np.stack([p[0] for p in policy_pairs])
    policy_n = np.stack([p[1] for p in policy_pairs])

    disc = Discriminator(amp_obs_dim=loader.observation_dim)
    opt = torch.optim.Adam(disc.parameters(), lr=1e-3)
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)

    for step in range(steps):
        e_s, e_n = loader.sample_batch(batch_size, rng)
        e_s = torch.from_numpy(e_s).float()
        e_n = torch.from_numpy(e_n).float()

        idx = rng.integers(0, len(policy_pairs), size=batch_size)
        p_s = torch.from_numpy(policy_s[idx]).float()
        p_n = torch.from_numpy(policy_n[idx]).float()

        d_e = disc(torch.cat([e_s, e_n], dim=-1))
        d_p = disc(torch.cat([p_s, p_n], dim=-1))

        expert_loss = torch.nn.MSELoss()(d_e, torch.ones_like(d_e))
        policy_loss = torch.nn.MSELoss()(d_p, -1.0 * torch.ones_like(d_p))
        amp_loss = 0.5 * (expert_loss + policy_loss)
        loss = amp_loss + disc.compute_grad_pen(e_s, e_n)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            acc = (
                (d_e > 0).float().mean()
                + (d_p < 0).float().mean()
            ) / 2
            print(f"step {step:3d} loss {loss.item():.3f} acc {acc.item():.3f}")

    return disc, e_s, e_n, p_s, p_n



def test_style_reward(disc, e_s, e_n, p_s, p_n):
    print("=== 6. style reward sanity ===")
    with torch.no_grad():
        r_e, _ = disc.predict_amp_reward(e_s, e_n, torch.zeros(e_s.shape[0]))
        r_p, _ = disc.predict_amp_reward(p_s, p_n, torch.zeros(p_s.shape[0]))
    print("expert style reward mean:", round(r_e.mean().item(), 4))
    print("policy style reward mean:", round(r_p.mean().item(), 4))
    assert r_e.mean() > r_p.mean()
    print("style reward OK")

def main():
    test_quat_helpers()
    loader, all_body_names = test_amp_loader()
    env, body_idx, anchor_idx = test_env_amp_obs(all_body_names)
    policy_pairs = collect_policy_pairs(env, body_idx, anchor_idx)
    disc, e_s, e_n, p_s, p_n = train_discriminator(loader, policy_pairs)
    test_style_reward(disc, e_s, e_n, p_s, p_n)

    print()
    print("=== STEP 4 VERIFICATION PASSED ===")


if __name__ == "__main__":
    main()






