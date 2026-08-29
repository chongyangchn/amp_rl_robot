from smplx.joint_names import JOINT_NAMES
import time
import mujoco
import mujoco.viewer
import torch
import numpy as np
from my_amp.envs.g1_env import G1Env
from my_amp.envs.vec_env import G1VecEnv
from my_amp.motion.motion_loader import MotionLoader
from my_amp.configs.train_cfg import TRAIN_CFG


def name_of_jSMPL_X ():
    '''
    SMPL-X 的关节名
    ['pelvis', 
    'left_hip',     'right_hip', 
    'spine1', 
    'left_knee',     'right_knee', 
    'spine2', 
    'left_ankle',     'right_ankle', 
    'spine3', 
    'left_foot',     'right_foot', 
    'neck', 
    'left_collar',     'right_collar', 
    'head', 
    'left_shoulder',     'right_shoulder', 
    'left_elbow',     'right_elbow', 
    'left_wrist',     'right_wrist', 
    'jaw', 
    'left_eye_smplhf',     'right_eye_smplhf', 
    'left_index1',     'left_index2',     'left_index3',     
    'left_middle1',    'left_middle2',    'left_middle3',    
    'left_pinky1',     'left_pinky2',     'left_pinky3', 
    'left_ring1',     'left_ring2',     'left_ring3',     
    'left_thumb1',     'left_thumb2',    'left_thumb3', 
    'right_index1',     'right_index2',     'right_index3', 
    'right_middle1',     'right_middle2',     'right_middle3', 
    'right_pinky1',     'right_pinky2',     'right_pinky3', 
    'right_ring1',     'right_ring2',     'right_ring3', 
    'right_thumb1',     'right_thumb2',     'right_thumb3']
    '''
    SMPLX_JOINT_NAMES = JOINT_NAMES[:55]  # 前 55 个是 SMPL-X 真正关节
    print("SMPL-X joint names:", SMPLX_JOINT_NAMES)


def name_mujoco_g1():
    model = mujoco.MjModel.from_xml_path("my_amp/envs/unitree_g1/scene.xml")
    # 关节
    print("=== JOINTS ===")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"Joint {i}: {joint_name}")
    # body
    print("\n=== BODIES ===")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"Body {i}: {name}")  
    # site
    print("\n=== SITES ===")
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        print(f"Site {i}: {name}")


'''
Mujoco中的关节名
# 0 floating_base_joint：自由基座，占 qpos 7 维、qvel 6 维
floating_base_joint 比较特殊，它负责机器人整机在空间中的移动和旋转，所以它占用：
qpos[0:7] = 3 位置 + 4 四元数
qvel[0:6] = 3 线速度 + 3 角速度
# 1~29：G1 的 29 个旋转关节，占 qpos[7:36]、qvel[6:35]
Joint 0: floating_base_joint  自由基座关节
Joint 1: left_hip_pitch_joint 左髋俯仰关节
Joint 2: left_hip_roll_joint 左髋滚转关节
Joint 3: left_hip_yaw_joint 左髋偏航关节
Joint 4: left_knee_joint 左膝关节
Joint 5: left_ankle_pitch_joint 左踝俯仰关节
Joint 6: left_ankle_roll_joint 左踝滚转关节
Joint 7: right_hip_pitch_joint 右髋俯仰关节
Joint 8: right_hip_roll_joint 右髋滚转关节
Joint 9: right_hip_yaw_joint 右髋偏航关节
Joint 10: right_knee_joint 右膝关节
Joint 11: right_ankle_pitch_joint 右踝俯仰关节
Joint 12: right_ankle_roll_joint 右踝滚转关节
Joint 13: waist_yaw_joint 腰部偏航关节
Joint 14: waist_roll_joint 腰部滚转关节
Joint 15: waist_pitch_joint 腰部俯仰关节
Joint 16: left_shoulder_pitch_joint 左肩俯仰关节
Joint 17: left_shoulder_roll_joint 左肩滚转关节
Joint 18: left_shoulder_yaw_joint 左肩偏航关节
Joint 19: left_elbow_joint 左肘关节
Joint 20: left_wrist_roll_joint 左腕滚转关节
Joint 21: left_wrist_pitch_joint 左腕俯仰关节
Joint 22: left_wrist_yaw_joint 左腕偏航关节
Joint 23: right_shoulder_pitch_joint 右肩俯仰关节
Joint 24: right_shoulder_roll_joint 右肩滚转关节
Joint 25: right_shoulder_yaw_joint 右肩偏航关节
Joint 26: right_elbow_joint 右肘关节
Joint 27: right_wrist_roll_joint 右腕滚转关节
Joint 28: right_wrist_pitch_joint 右腕俯仰关节
Joint 29: right_wrist_yaw_joint 右腕偏航关节

身体名
Body 0: world
Body 1: pelvis 骨盆 / 髋基座
Body 2: left_hip_pitch_link 左髋俯仰连杆
Body 3: left_hip_roll_link 左髋横滚连杆
Body 4: left_hip_yaw_link 左髋偏航连杆
Body 5: left_knee_link 左膝连杆
Body 6: left_ankle_pitch_link 左踝俯仰连杆
Body 7: left_ankle_roll_link 左踝滚转连杆
Body 8: right_hip_pitch_link 右髋俯仰连杆
Body 9: right_hip_roll_link 右髋横滚连杆
Body 10: right_hip_yaw_link 右髋偏航连杆
Body 11: right_knee_link 右膝连杆
Body 12: right_ankle_pitch_link 右踝俯仰连杆
Body 13: right_ankle_roll_link 右踝滚转连杆
Body 14: waist_yaw_link 腰部偏航连杆
Body 15: waist_roll_link 腰部滚转连杆
Body 16: torso_link 躯干连杆
Body 17: left_shoulder_pitch_link 左肩俯仰连杆
Body 18: left_shoulder_roll_link 左肩滚转连杆
Body 19: left_shoulder_yaw_link 左肩偏航连杆
Body 20: left_elbow_link 左肘连杆
Body 21: left_wrist_roll_link 左腕滚转连杆
Body 22: left_wrist_pitch_link 左腕俯仰连杆
Body 23: left_wrist_yaw_link 左腕偏航连杆
Body 24: right_shoulder_pitch_link 右肩俯仰连杆
Body 25: right_shoulder_roll_link 右肩滚转连杆
Body 26: right_shoulder_yaw_link 右肩偏航连杆
Body 27: right_elbow_link 右肘连杆
Body 28: right_wrist_roll_link 右腕滚转连杆
Body 29: right_wrist_pitch_link 右腕俯仰连杆
Body 30: right_wrist_yaw_link 右腕偏航连杆

位置
Site 0: imu_in_pelvis
Site 1: left_foot
Site 2: right_foot
Site 3: imu_in_torso
'''


def content_of_sfu_npz():
    print("\n###############content_of_sfu_npz###############")
    path = "data/sfu_raw/SFU/0008/0008_Walking001_poses.npz"
    d = np.load(path, allow_pickle=True)
    print("files:", d.files)
    print("gender:", str(d["gender"]))
    print("fps:", float(d["mocap_framerate"]))
    print("trans shape:", d["trans"].shape)
    print("poses shape:", d["poses"].shape)
    print("betas shape:", d["betas"].shape)
    print("dmpls shape:", d["dmpls"].shape)

    print("\nfirst trans:", d["trans"][0])
    print("first global_orient:", d["poses"][0, :3])
    print("first body_pose sample:", d["poses"][0, 3:9])
    print("first betas:", d["betas"][:5])



def validate_SMPL_X():
    print("\n############### validate_SMPL_X ###############")
    from pathlib import Path
    for gender in ["FEMALE", "MALE", "NEUTRAL"]:
        p = Path(f"data/smplx/SMPLX_{gender}.npz")
        print(f"{p}: {p.exists()}")

def validate_converted_motion():
    print("\n############### validate_converted_motion ###############")
    
    path = "data/motions/sfu_walking001.npz"
    d = np.load(path)\

    print("converted motion keys:", list(d.keys()))

    required = [
        "fps", "joint_pos", "joint_vel",
        "root_pos", "root_quat", "root_lin_vel", "root_ang_vel",
        "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
    ]

    for key in required:
        assert key in d, f"missing key {key}"

    print("keys:", d.files)
    print("joint_pos:", d["joint_pos"].shape)
    print("body_pos_w:", d["body_pos_w"].shape)

    assert d["joint_pos"].shape[1] == 29
    assert d["body_pos_w"].shape[1] == 30
    assert d["body_pos_w"].shape[2] == 3
    assert np.isfinite(d["joint_pos"]).all()
    assert np.isfinite(d["body_pos_w"]).all()
    assert np.isfinite(d["body_quat_w"]).all()
    assert np.isfinite(d["body_lin_vel_w"]).all()
    assert np.isfinite(d["body_ang_vel_w"]).all()

    print("converted motion format OK")

def check_amp_ref_reset():
    env = G1Env(
        "my_amp/envs/unitree_g1/scene.xml",
        amp_body_names=[
            "pelvis", "torso_link",
            "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link",
            "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link",
            "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_pitch_link",
            "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_pitch_link",
        ],
        amp_anchor_name="pelvis",
    )

    loader = MotionLoader("data/motions")
    rng = np.random.default_rng(0)

    for i in range(20):
        qpos, qvel, command = loader.get_reset_state_with_command(rng)
        obs = env.reset_to_ref(qpos, qvel, command)
        assert obs is not None, "reset_to_ref returned None"
        assert np.isfinite(obs["policy"]).all()
        assert np.isfinite(obs["amp"]).all()
        assert obs["amp"].shape == (210,)

    print("reference reset + AMP observation OK")
        
def check_vec_env_amp():
    loader = MotionLoader(TRAIN_CFG["amp"]["motion_dir"])

    env = G1VecEnv(
        num_envs=4,
        amp_body_names=TRAIN_CFG["amp"]["body_names"],
        amp_anchor_name=TRAIN_CFG["amp"]["anchor_name"],
        motion_loader=loader,
        reset_from_ref_prob=1.0,
    )

    obs = env.get_observations()
    print("policy:", obs["policy"].shape)
    print("critic:", obs["critic"].shape)
    print("amp:", obs["amp"].shape)

    actions = torch.zeros(4, 29)
    obs, rewards, dones, extras = env.step(actions)

    print("rewards:", rewards.shape, "dones:", dones.shape)
    print("time_outs:", extras["time_outs"].shape)
    print("finite:", torch.isfinite(obs["policy"]).all().item())


def visualize_sfu_converted():
    print("\n ############### SFU数据转换之后的可视化 ###############")
    path = "data/motions/sfu_walking001.npz"
    # # path = "data/motions/walk_forward_loop_002__A024.npz"
    # # path = "data/motions_sfu/sfu_walking001.npz"
    # path = "data/motions_sfu/walk_backward_loop_001__A022.npz"

    motion = np.load(path)

    env = G1Env("my_amp/envs/unitree_g1/scene.xml")

    n = len(motion["joint_pos"])
    fps = float(motion["fps"][0])
    dt = 1.0 / fps
    print(f"frames: {n}, fps: {fps:.0f}, dt: {dt*1000:.2f}ms")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15

        while viewer.is_running():
            for i in range(n):
                env.data.qpos[:3] = motion["root_pos"][i]
                env.data.qpos[3:7] = motion["root_quat"][i]
                env.data.qpos[3:7] /= np.linalg.norm(env.data.qpos[3:7])
                env.data.qpos[7:] = motion["joint_pos"][i]

                env.data.qvel[:3] = motion["root_lin_vel"][i]
                env.data.qvel[3:6] = motion["root_ang_vel"][i]
                env.data.qvel[6:] = motion["joint_vel"][i]

                mujoco.mj_forward(env.model, env.data)
                viewer.sync()

                if not viewer.is_running():
                    break

                time.sleep(dt)

    print("visualization finished")


if __name__ == "__main__":
    name_of_jSMPL_X()
    name_mujoco_g1()
    content_of_sfu_npz()

    # 检查 SMPL-X 模型是否存在
    validate_SMPL_X() 

    # 检查转换后的 npz 是否满足 AMP 格式
    validate_converted_motion()

    # 检查参考帧 reset 和 AMP 观测
    check_amp_ref_reset()

    # 检查向量环境和 AMP 训练输入
    check_vec_env_amp()

    visualize_sfu_converted()
