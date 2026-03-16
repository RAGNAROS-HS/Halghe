# Connecting a Neural Network to the Agar.io Clone

This document details the core logic and ideas for connecting a Reinforcement Learning (RL) Neural Network to the Agar.io Node.js server. The game relies entirely on **WebSockets**, which act as the bridge between your agent and the server.

## 1. Core Requirements

To successfully integrate the NN with the game, the following core components are required:

### A. Python Headless Client
Since the game communicates over Socket.IO, your RL agent needs to interface with the server in the same way the browser does.
* Use `python-socketio` to connect to `ws://localhost:3000`.
* Act as a game client by emitting a `type=player` handshake.
* Emulate viewport dimensions (e.g., `1920x1080` or larger) using the `gotit` event so the server knows how much data to send you.

### B. State Representation (The Observation Space)
The server broadcasts the game state via the `serverTellPlayerMove` event. Your agent needs to parse this JSON payload into a structured input tensor for the Neural Network.
* **Self State**: Your agent's coordinates ($x$, $y$), mass size, and current number of split cells.
* **Map Environment**:
    * **Food (`visibleFood`)**: An array of small particles ($x, y$ coordinates).
    * **Enemies / Players (`visiblePlayers`)**: Objects representing other players and their splits ($x, y$, mass, relations).
    * **Viruses (`visibleViruses`)**: Green spiky obstacles ($x, y$, mass) that explode players larger than them.
    * **Ejected Mass (`visibleMass`)**: Temporary food ejected by players.

### C. Action Space (The Agent's Output)
Your Neural Network will output an action, which the Python client must translate into Socket.IO emits.
The game server accepts the following actions:
1. **Move (Event `0`)**: Emits the target `$x$` and `$y$` coordinates the cell should move towards. This can be mapped to continuous actions (e.g., $dx, dy$ vectors mapped from $[-1, 1]$ into actual screen coordinates) or discrete directions (Up, Down, Left, Right).
2. **Shoot Mass (Event `1` / Key `W`)**: Ejects a piece of mass forward.
3. **Split (Event `2` / Key `Space`)**: Splits the player in half to lunge forward.

### D. Reward Function Engineering
This is critical for RL. You need an environment step loop that listens to Socket.IO and processes a reward:
* **Positive Reward**: 
    - Gaining mass (`current_mass > previous_mass`).
    - Successfully eating an enemy cell.
* **Negative Reward / Penalties**:
    - Losing mass (from decaying or shooting).
    - Hitting a virus (getting split into many tiny pieces).
    - Dying (receiving the `RIP` Socket.IO event).

## 2. Integration Ideas & Approaches

### Idea 1: OpenAI Gym/Gymnasium Wrapper
The standard approach for RL is to wrap the communication inside a Gymnasium `Env` class. 
* The `reset()` function connects the socket and emits `gotit`.
* The `step(action)` function emits the action to Socket.IO, waits for the next `serverTellPlayerMove` event (via a threading Queue or async loop), calculates the reward, and returns the next observation.

### Idea 2: Raycasting / Grid Observation Space
Raw XYZ coordinates for entities are difficult for Dense NNs to process effectively. Some proven transformations:
* **Grid Mapping (CNN approach)**: Transform the local viewport into a simplified 2D pixel grid or feature map (e.g., 64x64 grid). One channel represents food, another represents enemies, and another represents viruses. Pass this into a CNN (Convolutional Neural Network).
* **Entity Raycasting**: Cast 16-32 "rays" out from the player outwards in a circle. Report the distance and type of the first object hit by each ray (like LIDAR).

### Idea 3: Multi-Agent Training (Self-Play)
Because the Agar.io server can handle multiple connections simultaneously, you can run $N$ Python Socket.IO clients connected to the same server locally. 
This allows your agent to train via **Self-Play** (multiple instances of the neural network playing against each other).

### Idea 4: Viewport Manipulation
Because we control the client, we can exploit the API. By sending a massive `screenWidth` and `screenHeight` in the `gotit` event, the server will send the agent the position of *every object* on the map, effectively giving the agent global vision instead of local vision. (Could be useful depending on how difficult training is).

## 3. The Minimal Tech Stack
- **Game Server**: `Node.js` + `Socket.IO`
- **Agent Server Bridge**: `Python 3.9+` + `python-socketio[client]` + `asyncio`
- **RL Framework**: `Stable-Baselines3`, `Ray RLlib`, or raw `PyTorch`.
