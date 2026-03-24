# Halghe: Reinforcement Learning for Agar.io

Welcome to the Reinforcement Learning (RL) initiative for **Halghe**, an open-source Agar.io clone. This document outlines our plan for training intelligent agents to play Agar.io using modern Deep Reinforcement Learning techniques.

## Current State

We have established a foundational bridge between the Halghe server and a Python-based RL training setup:

- **Batched Environment API**: The Node.js game server has been extended with a batched HTTP API (`POST /rl/reset_batch`, `POST /rl/step_batch`). This enables us to control and step multiple agents synchronously, vastly speeding up experience gathering.
- **Gymnasium Wrapper (`BatchedHalgheEnv`)**: A custom vector-like environment in Python (`rl/vec_env.py`) that interacts with the server. It supports updating `num_agents` (e.g., 100 simultaneously) and abstracts the network calls.
- **Action Space**: We use a continuous 4-dimensional action space for each agent: `[dx, dy, split, fire]`.
    - `dx`, `dy`: Directional vectors for movement (clamped between -1 and 1).
    - `split`: Triggers the cell split mechanic if `> 0`.
    - `fire`: Triggers the mass ejection mechanic if `> 0`.
- **Rendering & Video Recording**: Built-in 2D Pygame visualizer that renders the game state to an RGB array, wrapped with Gym's `RecordVideo` to automatically capture training progress.
- **Training Scaffold (`train.py`)**: A basic loop for initializing the environment and stepping through episodes.

## What We Plan To Implement

### 1. Robust Observation Space (`_build_observation`)
Currently, agents receive a blank state. Agar.io presents a challenge where the number of entities (food, viruses, players, ejected mass) varies dynamically. We plan to implement one (or a hybrid) of the following representations:
- **Raycasting (Lidar-style)**: Casting rays in 360 degrees and returning the distance and type of the first intersecting object on each ray.
- **Grid-based Spatial Maps**: Discretizing the local viewport around the agent into a 2D grid/image with different channels for entity types (food channel, enemy channel, etc.), suitable for Convolutional Neural Networks (CNNs).
- **Entity Lists + Attention**: Using an embedding model to process lists of the $k$-nearest objects and their relative velocities/masses with a Transformer/Attention-based network.

### 2. Reward Shaping
A carefully crafted reward function is crucial to prevent the agent from getting stuck in local optima (e.g., just surviving by doing nothing). 
- **Dense Rewards**: Small positive rewards for consuming food pellets (`+Δ mass`).
- **Sparse Rewards**: Large positive rewards for successfully eating another player. Large negative penalties for being eaten.
- **Penalties**: Slight penalties for prolonged inactivity or running out of bounds.

### 3. Agent Modeling & RL Algorithm
- **Deep RL Algorithm**: Given our continuous observation/action spaces and high degree of parallelization (batched environment), **Proximal Policy Optimization (PPO)** is our primary candidate. It scales exceptionally well with vector environments and provides stable training.
- **Model Architecture (TensorFlow/Keras)**:
    - **Actor-Critic**: The network will have one head outputting the policy (action probabilities/means) and another head outputting the state value (critic).
    - Depending on the chosen observation space, the feature extractor will use generic dense layers, CNNs, or self-attention blocks.

### 4. Curriculum Learning and Self-Play
Training an agent from scratch directly against highly skilled opponents is too difficult.
- **Phase 1 (Foraging)**: Start training agents in environments with only food and viruses to master movement, size management, and splitting.
- **Phase 2 (Basic Combat)**: Introduce rule-based bots to teach the agent basic evasion and hunting strategies.
- **Phase 3 (Self-Play)**: Agents train against past versions of themselves. This dynamic environment prevents overfitting to static bot behaviors and naturally escalates the complexity of strategies (e.g., baiting with mass, cooperative splitting).

## How to Run the Current Setup

1. **Start the game server with the RL endpoints API mounted.**
2. **Install Python dependencies:**
   ```bash
   pip install -r rl/requirements.txt
   ```
3. **Run the training scaffold:**
   ```bash
   python rl/train.py
   ```
4. **Check the outputs:** Rendered videos of episodes can be viewed in the `videos` directory.
