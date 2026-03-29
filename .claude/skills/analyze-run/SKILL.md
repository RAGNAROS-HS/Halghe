---
name: analyze-run
description: Analyze a completed or in-progress RL training run. Use when the user wants to review training results, check reward curves, inspect episode videos, or debug why the agent isn't learning.
disable-model-invocation: true
---

## Current State
- Latest videos: !`ls rl/videos/train_runs_batched/ 2>/dev/null | tail -5`
- Recent git log: !`git log --oneline -5`

Analyze the RL training run for $ARGUMENTS (or the most recent run if unspecified).

1. Read `rl/train.py` to understand current hyperparameters (learning rate, gamma, clip range, etc.)
2. Read `rl/vec_env.py` to understand the observation/action/reward design
3. Check `rl/videos/train_runs_batched/` — watch the latest episode video to assess agent behavior
4. Identify issues:
   - Reward not increasing → check reward shaping in `vec_env.py`
   - Agent not moving → check action space or frame skipping config
   - Crashes/errors → check server logs and `rl-api.js`
5. Suggest specific, minimal changes to fix the identified issue
