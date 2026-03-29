# CLAUDE.md

## Philosophy
Keep things simple and easy to fix manually. Prefer straightforward solutions over clever abstractions.

## Project Overview
Halghe is an open-source Agar.io clone with a reinforcement learning layer. The game server is Node.js; the RL training code is Python (`rl/`).

## Key Components
- `src/server/` — Node.js game server, includes batched RL API (`/rl/reset_batch`, `/rl/step_batch`)
- `src/client/` — browser client
- `rl/vec_env.py` — Gymnasium wrapper (`BatchedHalgheEnv`) that talks to the server
- `rl/train.py` — training entry point
- `rl/requirements.txt` — Python dependencies

## Running Locally
```bash
# Start the game server
npm start

# Install Python deps
pip install -r rl/requirements.txt

# Run training
python rl/train.py
```

## Code Style
- Favor flat, readable code over nested abstractions
- Small, focused functions
- No unnecessary error handling for cases that can't happen
- Don't add comments unless the logic isn't obvious
