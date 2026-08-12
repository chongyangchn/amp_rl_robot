
from pathlib import Path
import numpy as np
import torch
from smplx import SMPLX
from smplx.joint_names import JOINT_NAMES

SMPLX_JOINT_NAMES = JOINT_NAMES[:55]  # 前 55 个是 SMPL-X 真正关节
PATCHED_DIR = Path("data/smplx_patched")


def build_smplx(gender: str) -> SMPLX:
    PATCHED_DIR.mkdir(exist_ok=True)
    patched_path = PATCHED_DIR / f"SMPLX_{gender.upper()}.npz"

    if not patched_path.exists():
        src = Path("data/smplx") / f"SMPLX_{gender.upper()}.npz"
        data = dict(np.load(src, allow_pickle=True))
        # 这套模型缺手部 PCA 和面部 landmark 字段，补占位即可
        data["hands_componentsl"] = np.zeros((45, 1), dtype=np.float32)
        data["hands_componentsr"] = np.zeros((45, 1), dtype=np.float32)
        data["hands_meanl"] = np.zeros((45,), dtype=np.float32)
        data["hands_meanr"] = np.zeros((45,), dtype=np.float32)
        data["lmk_faces_idx"] = np.zeros((68,), dtype=np.int32)
        data["lmk_bary_coords"] = np.zeros((68, 3), dtype=np.float32)
        np.savez(patched_path, **data)

    return SMPLX(
        model_path=str(PATCHED_DIR),
        gender=gender,
        num_betas=16,
        use_pca=False,
        num_pca_comps=0,
        flat_hand_mean=True,
        create_expression=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
        use_face_contour=False,
    )

def decode_smplx_frame(model, pose, trans, betas):
    pose = torch.from_numpy(pose[None]).float()
    out = model(
        global_orient=pose[:, :3],
        body_pose=pose[:, 3:66],
        left_hand_pose=pose[:, 66:111],
        right_hand_pose=pose[:, 111:156],
        betas=torch.from_numpy(betas[None]).float(),
        transl=torch.from_numpy(trans[None]).float(),
        expression=torch.zeros(1, 10),
        jaw_pose=torch.zeros(1, 3),
        leye_pose=torch.zeros(1, 3),
        reye_pose=torch.zeros(1, 3),
    )

    return out.joints[0].detach().cpu().numpy()[:55]

    