# amp_rl_robot

自研 AMP + Unitree G1 项目。RL 骨架基于 rsl_rl 5.4.2（vendored 在 `rsl_rl/`），AMP 相关模块全部自己实现：环境 VecEnv、运动数据加载、判别器、AMPPPO、AmpRunner。

## 目录结构

```text
amp_rl_robot/
├── rsl_rl/                  # rsl_rl 5.4.2 源码，只读依赖
├── resources/               # G1 MuJoCo 模型资源
├── my_amp/
│   ├── envs/                # 单环境 + rsl_rl VecEnv
│   ├── motion/              # 运动数据加载、AMP 特征、动作 reset
│   ├── amp/                 # 判别器、回放缓冲、AMPPPO、AmpRunner
│   └── configs/             # 环境和训练配置
├── scripts/                 # 训练/回放脚本
├── data/motions/            # 动作数据（不进 git，用 scp 或服务器下载）
├── requirements.txt
└── README.md
```

## 本地安装（Windows 调试用）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 服务器安装（Linux + NVIDIA GPU）

```bash
conda create -n amp python=3.11 -y
conda activate amp
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 开发进度

- [ ] 1. 项目骨架 + VecEnv
- [ ] 2. 普通 PPO 基线（先不写 AMP）
- [ ] 3. 运动数据管线 + 动作 reset
- [ ] 4. AMP 观测与判别器
- [ ] 5. AMP 训练算法（AMPPPO + AmpRunner）
- [ ] 6. G1 复杂动作任务配置
- [ ] 7. 训练调试与调参
- [ ] 8. 评估、ONNX 导出、部署


# rsl_rl依赖的安装
pip install -e ./rsl_rl
pip install -e ./rsl_rl -i https://pypi.tuna.tsinghua.edu.cn/simple

# 冒烟测试代码
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# 云端代码训练

# 激活环境：
conda activate amp

# 运行代码
cd ~/cy_amp/amp_rl_robot
PYTHONPATH=. python scripts/train_amp_rsl.py


 tensorboard --logdir logs   


 云服务器训练后进行视频保存
 PYTHONPATH=. xvfb-run -a python scripts/eval_amp_headless.py