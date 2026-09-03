# mature_rsl_rl

This directory contains the mature rsl_rl implementation copied from
`Y:\RobotTransition\Project\AMP_mjlab\rsl_rl`.

It is kept in a separate folder until the legacy rsl_rl interfaces are adapted
to the current `G1VecEnv` and `TRAIN_CFG`.

Key modules:

- `algorithms/amp_ppo.py`
- `modules/discriminator.py`
- `modules/normalizer.py`
- `storage/replay_buffer.py`
- `storage/rollout_storage.py`
- `runners/amp_on_policy_runner.py`
- `utils/motion_loader.py`
