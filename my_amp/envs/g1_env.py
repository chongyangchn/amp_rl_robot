import mujoco
import numpy as np
import torch
from my_amp.amp.amp_obs import compute_amp_features



def quat_to_rotmat(quat):
    """四元数 → 3x3 旋转矩阵"""
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*w*z,     2*x*z + 2*w*y    ],
        [2*x*y + 2*w*z,       1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x    ],
        [2*x*z - 2*w*y,       2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
    ])


class G1Env:
    def __init__(self, xml_path, amp_body_names=None, amp_anchor_name=None):
        # -------- 1. 加载模型 --------
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)


        self.body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(1, self.model.nbody)
        ]
        # 1. 处理 AMP 身体索引 (amp_body_idx)
        if amp_body_names is not None:
            # 从已有的 body_names 列表中查找对应的索引
            self.amp_body_idx = [self.body_names.index(name) for name in amp_body_names]
        else:
            default_bodies = [
                "pelvis",
                "torso_link",
                "left_hip_yaw_link",
                "left_knee_link",
                "left_ankle_pitch_link",
                "right_hip_yaw_link",
                "right_knee_link",
                "right_ankle_pitch_link",
                "left_shoulder_pitch_link",
                "left_elbow_link",
                "left_wrist_pitch_link",
                "right_shoulder_pitch_link",
                "right_elbow_link",
                "right_wrist_pitch_link",
            ]
            self.amp_body_idx = [self.body_names.index(name) for name in default_bodies if name in self.body_names]
            if not self.amp_body_idx:
                raise ValueError("G1Env: 无法找到 AMP 需要的身体部位名称，请手动传入 amp_body_names")

        # 2. 处理 AMP 锚点索引 (amp_anchor_idx)
        if amp_anchor_name is not None:
            self.amp_anchor_idx = self.body_names.index(amp_anchor_name)
        else:
            # 默认用 "pelvis" 或 "torso"
            if "pelvis" in self.body_names:
                self.amp_anchor_idx = self.body_names.index("pelvis")
            elif "torso" in self.body_names:
                self.amp_anchor_idx = self.body_names.index("torso")
            else:
                # 极端情况，用第一个身体部位（通常就是根）
                self.amp_anchor_idx = 0

        # -------- 2. 动作重复与时间步长 --------
        self.action_repeat = 2 
        self.action_scale = 0.3
        self.dt = self.model.opt.timestep * self.action_repeat


        # ---------- 3 动作与控制相关信息 + 观测量信息----------
        self.num_joints = self.model.nq - 7     # 关节数量：29
        self.action_dim = self.model.nu         # 动作维度29 (双腿：6 × 2 = 12 ；腰部：3；双臂：7 × 2 = 14；) (合计：12 + 3 + 14 = 29)
        self.obs_dim = 11 + 2 + 2 * self.num_joints + self.action_dim  + 2 + 2  # 加了 2 维相位

        # 关节限位
        self.joint_low = self.model.actuator_ctrlrange[:, 0] # 获取XML中定义的关节限位
        self.joint_high = self.model.actuator_ctrlrange[:, 1] # 获取XML中定义的关节限位
        self.joint_qpos_idx = slice(7, self.model.nq)  # 关节位置索引（跳过基座 7 维）
        self.joint_qvel_idx = slice(6, self.model.nv)  # 关节速度索引（跳过基座 6 维）

        # 左右脚和传感器
        self.left_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.torso_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso")

        # ---- body id 缓存（替代硬编码 7 和 13） ----
        self._left_foot_body = self.model.site_bodyid[self.left_foot_site]
        self._right_foot_body = self.model.site_bodyid[self.right_foot_site]

        # 默认关节位置（reset 时用到）
        self.default_joint_pos = self.data.qpos[7:].copy()

        # 上一步动作缓存
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)

        # 目标高度（XML 里 pelvis 在 0.793，加一点作为站立高度）
        self.target_height = 0.82

        # 速度指令（训练循环里可以重置它）
        self.command = np.array([0.2, 0.0], dtype=np.float32)   # [前向速度, 转向速度]

        # ── 步态参数 ──
        self.step_count = 0
        self.gait_period = 0.8          # 一个完整步态周期（秒）
        
        print(f"[Env] Loaded G1 | 控制量维度={self.action_dim} | 观测量维度={self.obs_dim} | 时间步长{self.dt:.3f}s")


    def _get_foot_contacts(self):
        """检测左右脚是否着地，返回 (2,) 数组"""
        """向量化版本：用 numpy 数组视图替代 Python for 循环"""
        contact = np.zeros(2, dtype=np.float32)
        if self.data.ncon == 0:
            return contact
        geom1 = self.data.contact.geom1[:self.data.ncon]
        geom2 = self.data.contact.geom2[:self.data.ncon]
        body1 = self.model.geom_bodyid[geom1]
        body2 = self.model.geom_bodyid[geom2]
        contact[0] = float(np.any((body1 == self._left_foot_body)
                                  | (body2 == self._left_foot_body)))
        contact[1] = float(np.any((body1 == self._right_foot_body)
                                  | (body2 == self._right_foot_body)))
        return contact

    # 1、获取观测量
    def _get_obs(self):
        """
        观测量的信息
        mujoco返回的数据是numpy类型，计算时是tensor类型，数据转换最好在rolloutbuffer里进行
        MuJoCo (CPU)  →  _get_obs()  →  RolloutBuffer  →  PPO update (GPU)
        在 buffer 里存 numpy，更新时批量转

        自由度：nq=36（7基座 + 29关节），nv=35（6基座 + 29关节速度）
        执行器：nu=29（每个关节配一个电机）

        qpos结构：
        qpos[0:3]  基座位置 (x, y, z)
        qpos[3:7]  基座四元数 (qw, qx, qy, qz)
        qpos[7:36] 29个关节位置

        qvel结构：
        qvel[0:3]  基座线速度
        qvel[3:6]  基座角速度
        qvel[6:35] 29个关节速度

        29个关节（从 qpos[7] 开始）：
        双腿（12个）：左/右 髋 pitch/roll/yaw，膝 pitch，踝 pitch/roll
        腰部（3个）：腰 yaw/roll/pitch
        左臂（7个）：肩 pitch/roll/yaw，肘 pitch，腕 roll/pitch/yaw
        右臂（7个）：同上对称
        """
        # 基座旋转矩阵
        rot = quat_to_rotmat(self.data.qpos[3:7])
        self._cached_rot = rot

        local_lin_vel = rot.T @ self.data.qvel[:3]                          # 3  基座局部线速度（前后/左右/上下）
        local_ang_vel = rot.T @ self.data.qvel[3:6]                         # 3  基座局部角速度（滚转/俯仰/偏航）
        projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0])              # 3  投影重力（躯干坐标系下的重力方向）
        cmd = self.command                                                  # 2  速度指令（前向速度 + 转向速度）
        joint_pos_delta = self.data.qpos[7:] - self.default_joint_pos       # 29 关节位置（相对默认位置的偏移）
        joint_vel = self.data.qvel[6:] * 0.05                               # 29 关节速度（缩放到合理范围）
        prev_act = self.prev_action                                         # 29 上一步的动作（平滑过渡用）
        foot_contact = self._get_foot_contacts()                            # 2  左右脚是否着地
        height = self.data.qpos[2]
        height_deviation = height - self.target_height
        torso_xmat = self.data.site_xmat[self.torso_site].reshape(3, 3)
        upright = torso_xmat[2, 2]  # 躯干 z 轴与世界 z 轴对齐程度            # 2  基座高度偏差 + 直立程度

        # ── 步态相位（sin/cos 编码，让网络知道当前位置） ──
        phase = 2.0 * np.pi * (self.step_count * self.dt / self.gait_period)
        gait_phase = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

        obs = np.concatenate([
            local_lin_vel,        # 3
            local_ang_vel,        # 3
            projected_gravity,    # 3
            cmd,                  # 2
            joint_pos_delta,      # 29
            joint_vel,            # 29
            prev_act,                  # 29
            foot_contact,         # 2
            [height_deviation, upright],  # 2
            gait_phase,  # 2
        ])

        # 合计: 3+3+3+2+29+29+29+2+2 = 102
        return obs

    def get_obs_vec(self):
        obs = self._get_obs()
        return {
            "policy" : obs, # (policy_dim,)
            "critic" : obs, # 第一步先和 policy 一样
            "amp": self.get_amp_obs(self.amp_body_idx, self.amp_anchor_idx),
        }
    
      
    def _is_done(self):
        height = self.data.qpos[2]

        if height < 0.4 or height > 1.4:
            return True
        
        rot = quat_to_rotmat(self.data.qpos[3:7])
        upright = (rot.T @ np.array([0.0, 0.0, -1.0]))[2]  # 投影重力的 z 分量
        if upright > -0.5:  # cos(45°) ≈ 0.707
            return True

        
        return False
    
    # def 

    # 环境重置
    def reset(self):
        """重置仿真状态"""
        # 1. 物理重置
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.prev_action.fill(0.0)
        self.step_count = 0                         # ← 加这行

        # ── 随机化速度指令（提升泛化性） ──
        # self.command = np.array([
        #     np.random.uniform(0.2, 0.6),  # 前向速度 0.2~0.6 m/s
        #     np.random.uniform(-0.5, 0.5), # 转向速度 -0.5~0.5 rad/s
        # ])
        self.command = np.array([0.0, 0.0])
         
        # return self._get_obs()
        return self.get_obs_vec()


    # 环境交互
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = self.default_joint_pos + action * self.action_scale # 将动作值映射到mujoco定义的范围内

        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)

        # obs = self._get_obs()
        # 关键：reward 前刷新当前姿态
        self._cached_rot = quat_to_rotmat(self.data.qpos[3:7])
        reward = self._caculate_reward(action)
        is_done = self._is_done()
        info = {}


        self.prev_action = action.copy() 
        self.step_count += 1   
        
        # return obs, reward, is_done, info
        return self.get_obs_vec(), reward, is_done, info

    def _caculate_reward(self, action):
        sigma = 0.25
        rot = self._cached_rot

        local_lin_vel = rot.T @ self.data.qvel[:3]
        local_ang_vel = rot.T @ self.data.qvel[3:6]

        lin_vel_error = float((self.command[0] - local_lin_vel[0]) ** 2)
        ang_vel_error = float((self.command[1] - local_ang_vel[2]) ** 2)

        track_lin_reward = np.exp(-lin_vel_error / sigma ** 2)
        track_ang_reward = np.exp(-ang_vel_error / sigma ** 2)

        projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0])
        upright_err = np.sum(projected_gravity[:2] ** 2)
        height_err = (self.data.qpos[2] - self.target_height) ** 2
        action_rate = np.mean((action - self.prev_action) ** 2)

        total_reward = (
            + 2.0 * track_lin_reward
            + 1.0 * track_ang_reward
            + 1.0 * np.exp(-upright_err / 0.05)
            + 1.0 * np.exp(-height_err / 0.01)
            + 0.2
            - 0.01 * action_rate
        )
        return total_reward

    def reset_to_ref(self, qpos, qvel=None, command=None):
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = qpos
        self.data.qpos[3:7] /= np.linalg.norm(self.data.qpos[3:7])

        if qvel is not None:
            self.data.qvel[:] = qvel
        else:
            self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        ref_joint_pos = qpos[7:]

        self.prev_action = np.clip(
            (ref_joint_pos - self.default_joint_pos) / self.action_scale,
            -1.0,
            1.0,
        )
        self.step_count = 0
        if command is None:
            self.command = np.array([0.0, 0.0], dtype=np.float32)
        else:
            self.command = np.array(command, dtype=np.float32)

        return self.get_obs_vec()

    
    def get_amp_obs(self, body_idx, anchor_idx):
        return compute_amp_features(
            self.data.xpos[1:],
            self.data.xquat[1:],
            self.data.cvel[1:, 3:],
            self.data.cvel[1:, :3],
            body_idx,
            anchor_idx,
        )
    
        
