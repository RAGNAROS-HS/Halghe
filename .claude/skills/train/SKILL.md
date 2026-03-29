---
name: train
description: Start or manage an RL training run. Use when the user wants to run training, check training config, or understand how to launch the RL agent.
disable-model-invocation: true
---

## Current State
- Branch: !`git branch --show-current`
- Modified files: !`git status --short`

Start a training run for the Halghe RL agent.

1. Confirm the game server is running: `npm start` (in a separate terminal)
2. Review `rl/train.py` for any config changes needed (num envs, total timesteps, etc.)
3. Launch training: `python rl/train.py`
4. Videos are saved to `rl/videos/train_runs_batched/` every 10 episodes

If $ARGUMENTS is provided, apply it as context (e.g. "with 8 envs", "for 1M steps").
