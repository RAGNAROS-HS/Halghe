import gymnasium
from gymnasium import spaces
import numpy as np
import requests
import pygame
from typing import Optional, Tuple, Any, Dict, List

class BatchedHalgheEnv(gymnasium.Env):
    """
    A batched Gymnasium environment wrapping the Halghe RL HTTP API.
    Interacts with POST /rl/step_batch to manage multiple agents synchronously.
    
    Warning: We inherit from Env, but mimic a VectorEnv signature:
    obs shape: (num_agents, obs_dim)
    actions shape: (num_agents, action_dim)
    reward shape: (num_agents,)
    terminated shape: (num_agents,)
    truncated shape: (num_agents,)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, num_agents=100, server_url="http://localhost:3000", render_mode=None):
        super().__init__()
        self.num_agents = num_agents
        self.server_url = server_url.rstrip("/")
        self._session = requests.Session()
        self.render_mode = render_mode

        # Define action space (4-element continuous vectors)
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_agents, 4), dtype=np.float32)

        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_agents, 1), dtype=np.float32
        )

        self._raw_states = None

        # Pygame rendering state
        self.window_size = 800
        self._surface = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the batched environment to an initial state and returns the initial observation.
        """
        super().reset(seed=seed)
        resp = self._session.post(
            f"{self.server_url}/rl/reset_batch", 
            json={"num_agents": self.num_agents}
        )
        resp.raise_for_status()
        data_list = resp.json()
        
        self._raw_states = [d["state"] for d in data_list]
        obs = np.stack([self._build_observation(s) for s in self._raw_states])
        
        infos = {}
        for d in data_list:
            for k, v in d.get("info", {}).items():
                infos.setdefault(k, []).append(v)
                
        return obs, infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics for all batched agents.
        Accepts an array of actions and returns arrays of observations, rewards, etc.
        """
        # Decode batched actions
        action_payloads = [self._decode_action(actions[i]) for i in range(self.num_agents)]
        
        resp = self._session.post(
            f"{self.server_url}/rl/step_batch",
            json={"actions": action_payloads},
        )
        resp.raise_for_status()
        data_list = resp.json()

        self._raw_states = [d["state"] for d in data_list]
        obs = np.stack([self._build_observation(s) for s in self._raw_states])
        
        rewards = np.array([float(d["reward"]) for d in data_list], dtype=np.float32)
        terminated = np.array([bool(d["done"]) for d in data_list], dtype=bool)
        truncated = np.zeros(self.num_agents, dtype=bool)
        
        infos = {}
        for d in data_list:
            for k, v in d.get("info", {}).items():
                infos.setdefault(k, []).append(v)
                
        infos["raw_states"] = self._raw_states

        return obs, rewards, terminated, truncated, infos

    def _build_observation(self, raw_state: dict) -> np.ndarray:
        return np.zeros(self.single_observation_space.shape, dtype=np.float32)

    def _decode_action(self, action) -> dict:
        return {
            "dx": float(np.clip(action[0], -1, 1)),
            "dy": float(np.clip(action[1], -1, 1)),
            "split": int(action[2] > 0),
            "fire": int(action[3] > 0),
        }

    def render(self) -> Optional[np.ndarray]:
        """
        Renders the environment to an RGB array visualizing all agents on the board.
        Returns the rendering as a NumPy array if render_mode is 'rgb_array', else None.
        """
        if self.render_mode != "rgb_array":
            return None
        if not self._raw_states or self._raw_states[0] is None:
            return None

        # Render first agent's map
        raw_state = self._raw_states[0]

        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.window_size, self.window_size))

        self._surface.fill((255, 255, 255))

        map_w = raw_state.get("map", {}).get("width", 5000)
        map_h = raw_state.get("map", {}).get("height", 5000)
        scale_x = self.window_size / max(map_w, 1)
        scale_y = self.window_size / max(map_h, 1)

        def draw_circle(color, x, y, radius):
            pygame.draw.circle(
                self._surface, color, 
                (int(x * scale_x), int(y * scale_y)), 
                max(1, int(radius * max(scale_x, scale_y)))
            )

        for f in raw_state.get("food", []):
            draw_circle((0, 0, 255), f["x"], f["y"], 5)
        for mf in raw_state.get("massFood", []):
            draw_circle((0, 255, 255), mf["x"], mf["y"], 8)
        for v in raw_state.get("viruses", []):
            draw_circle((0, 255, 0), v["x"], v["y"], v.get("radius", 30))
        for e in raw_state.get("enemies", []):
            for cell in e.get("cells", []):
                draw_circle((255, 0, 0), cell["x"], cell["y"], cell.get("radius", 10))
        
        # Render ALL agents so they don't look like just one!
        for i, state in enumerate(self._raw_states):
            player = state.get("player", {})
            if player:
                # distinct stable color per agent
                color = (
                    (i * 123) % 255, 
                    (i * 231) % 255, 
                    (i * 321) % 255
                ) if i > 0 else (0, 0, 0) # first agent is black
                
                for cell in player.get("cells", []):
                    draw_circle(color, cell["x"], cell["y"], cell.get("radius", 10))

        return np.transpose(np.array(pygame.surfarray.pixels3d(self._surface)), axes=(1, 0, 2))

    def close(self):
        if self._surface is not None:
            pygame.quit()
        super().close()
