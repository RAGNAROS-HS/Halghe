# CLAUDE.md

## Philosophy
Keep things simple and easy to fix manually. Prefer straightforward solutions over clever abstractions.

## Project Overview
Halghe is an open-source Agar.io clone with a reinforcement learning layer. The game server is Node.js; the RL training code is Python (`rl/`).

## Key Components
- `src/server/rl-api.js` — Express router: `/rl/reset_batch`, `/rl/step_batch`, `/rl/render_state`, `/rl/config`
- `src/server/rl-game.js` — isolated per-agent game instance used by the RL API
- `src/client/` — browser client (unrelated to RL)
- `rl/train.py` — **training entry point** (Actor-Critic); run this
- `rl/vec_env.py` — `BatchedHalgheEnv` Gymnasium wrapper; not run directly
- `rl/requirements.txt` — Python dependencies

## Running Locally
```bash
# 1. Start the game server
npm start

# 2. Install Python deps
pip install -r rl/requirements.txt

# 3. Run training (from the rl/ directory)
cd rl
python train.py

# Optional: monitor with TensorBoard
tensorboard --logdir logs/train
```

All hyperparameters are CLI flags — run `python train.py --help` for the full list.

## Code Style
- Favor flat, readable code over nested abstractions
- Small, focused functions
- No unnecessary error handling for cases that can't happen
- Don't add comments unless the logic isn't obvious
