# Halghe: Reinforcement Learning for Agar.io

An open-source Agar.io clone with a deep RL training layer. Agents are trained using an Actor-Critic algorithm against an isolated Node.js game server.

---

## Architecture

```
┌─────────────────────────────────┐        HTTP (JSON)        ┌──────────────────────────────┐
│  Python (rl/)                   │ ◄───────────────────────► │  Node.js server              │
│                                 │                            │                              │
│  train.py        ← entry point  │  POST /rl/reset_batch      │  src/server/rl-api.js        │
│  vec_env.py      ← Gym wrapper  │  POST /rl/step_batch       │  src/server/rl-game.js       │
│                                 │  GET  /rl/render_state     │  src/server/game-logic.js    │
└─────────────────────────────────┘  GET  /rl/config          └──────────────────────────────┘
```

- **`rl/train.py`** — training entry point. Actor-Critic (policy gradient + value baseline). Run this.
- **`rl/vec_env.py`** — `BatchedHalgheEnv`: Gymnasium-compatible wrapper; manages N agents over a single HTTP connection. Not run directly.
- **`src/server/rl-api.js`** — Express router exposing the RL endpoints.
- **`src/server/rl-game.js`** — isolated per-agent game instance (no multiplayer state).

---

## How to Run

### 1. Start the game server

```bash
npm start
```

This starts the Node.js server on `http://localhost:3000` and mounts the RL API.

### 2. Install Python dependencies

```bash
pip install -r rl/requirements.txt
```

### 3. Run training

```bash
cd rl
python train.py
```

All hyperparameters are configurable via CLI:

```
--num-agents    INT    Parallel agents per episode          (default: 100)
--episodes      INT    Number of training episodes          (default: 500)
--max-steps     INT    Max env steps per episode            (default: 2000)
--frame-skip    INT    Server ticks per action              (default: 4)
--actor-lr      FLOAT  Actor learning rate                  (default: 0.001)
--critic-lr     FLOAT  Critic learning rate                 (default: 0.001)
--stddev-start  FLOAT  Initial exploration noise            (default: 0.2)
--stddev-end    FLOAT  Final exploration noise (decays to)  (default: 0.05)
--gamma         FLOAT  Discount factor                      (default: 0.99)
--server-url    STR    Game server URL                      (default: http://localhost:3000)
--log-dir       STR    TensorBoard log directory            (default: logs/train)
--video-dir     STR    Video output directory               (default: videos/train_runs_batched)
```

Example — quick smoke test:

```bash
python train.py --episodes 5 --max-steps 200 --num-agents 10
```

### 4. Monitor training

```bash
tensorboard --logdir logs/train
```

Metrics logged per episode: `reward/avg_per_agent`, `loss/actor`, `loss/critic`, `exploration/stddev`.

### 5. Watch recorded videos

Episode videos are saved every 10 episodes to `rl/videos/train_runs_batched/`.

---

## Observation & Action Spaces

| | Space | Shape | Description |
|---|---|---|---|
| **Observation** | `Box(-inf, inf)` | `(num_agents, 6)` | `[px, py, mass, food_dx, food_dy, num_enemies]` — all normalized |
| **Action** | `Box(-1, 1)` | `(num_agents, 4)` | `[dx, dy, split, fire]` |

- `dx`, `dy`: movement direction, clamped to `[-1, 1]`
- `split`: split all cells if `> 0`
- `fire`: eject mass from all cells if `> 0`

---

## Training Algorithm

**Actor-Critic** (on-policy, episode-level updates):

1. Collect a full episode of trajectories across all N agents
2. Compute discounted returns `G_t` with `γ = 0.99`
3. Normalize returns (zero-mean, unit-variance within the batch)
4. **Critic** minimizes MSE loss: `(G_t - V(s_t))²`
5. **Actor** maximizes log-probability weighted by advantage: `-E[log π(a|s) · (G_t - V(s_t))]`
   - Log-probability uses the full Gaussian formula including the normalization constant
6. Exploration noise `σ` linearly decays from `stddev-start` → `stddev-end`

---

## Reward Function

| Event | Reward |
|---|---|
| Eating food or mass | `+Δ mass` (only deliberate gain; passive decay is cancelled out) |
| Being eaten / dying | `−max(mass × 0.5, 10)` (scales with accumulated mass) |

---

## Roadmap

### Phase 1 — Foraging (current)
Single-agent environments with food and viruses. Goal: learn movement, size management, splitting.

### Phase 2 — Basic Combat
Add rule-based bots. Goal: learn evasion and hunting strategies.

### Phase 3 — Self-Play
Agents train against past versions of themselves. Goal: emergent strategies (baiting, cooperative splitting).

### Future Observation Improvements
- **Raycasting**: lidar-style rays returning distance + entity type
- **Grid maps**: local 2D grid with separate channels per entity type (CNN-friendly)
- **Entity lists + Attention**: k-nearest objects with relative velocities processed by a Transformer
