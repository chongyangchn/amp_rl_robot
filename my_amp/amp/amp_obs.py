import numpy as np

def quat_conjugate(q):
    """wxyz 四元数取共轭"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1, q2):
    """wxyz 四元数乘法 q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])

def quat_to_rotmat(q):
    """wxyz -> 3x3 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),         1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),         2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def compute_amp_features(pos_w, quat_w, lin_vel_w, ang_vel_w, body_idx, anchor_idx):
    """
    pos_w:      (nbody, 3)  世界系 body 位置
    quat_w:     (nbody, 4)  世界系 body 姿态 wxyz
    lin_vel_w:  (nbody, 3)  世界系 body 线速度
    ang_vel_w:  (nbody, 3)  世界系 body 角速度
    body_idx:   要参与判别的 body 索引列表
    anchor_idx: 根/锚点 body 索引（一般是 pelvis）
    """
    anchor_pos = pos_w[anchor_idx]
    anchor_quat = quat_w[anchor_idx]
    R_anchor = quat_to_rotmat(anchor_quat)

    feats = []

    for bi in body_idx:
        p_rel = R_anchor.T @ (pos_w[bi] - anchor_pos)

        q_rel = quat_mul(quat_conjugate(anchor_quat), quat_w[bi])
        R_rel = quat_to_rotmat(q_rel)
        ori6 = R_rel[:, :2].reshape(6)

        R_body = quat_to_rotmat(quat_w[bi])
        lin_b = R_body.T @ lin_vel_w[bi]
        ang_b = R_body.T @ ang_vel_w[bi]

        feats.append(np.concatenate([p_rel, ori6, lin_b, ang_b]))

    return np.concatenate(feats)