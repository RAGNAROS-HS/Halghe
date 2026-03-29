import gymnasium
from gymnasium import spaces
import numpy as np
import requests
import pygame
from typing import Optional, Tuple, Any, Dict


class BatchedHalgheEnv(gymnasium.Env):
    """
    A batched Gymnasium environment wrapping the Halghe RL HTTP API.
    Interacts with POST /rl/step_batch to manage multiple agents synchronously.

    Observations and rewards are pre-computed on the Node.js server — the HTTP
    response carries only slim {obs, reward, done, info} payloads instead of
    the full game state (food arrays, virus arrays, etc.).

    Rendering is lazy: /rl/render_state is fetched only when render() is called,
    which only happens during RecordVideo recording episodes.

    Warning: We inherit from Env but mimic a VectorEnv signature:
      obs shape:        (num_agents, obs_dim)
      actions shape:    (num_agents, action_dim)
      reward shape:     (num_agents,)
      terminated shape: (num_agents,)
      truncated shape:  (num_agents,)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, num_agents=100, server_url="http://localhost:3000", render_mode=None, frame_skip=1):
        super().__init__()
        self.num_agents = num_agents
        self.server_url = server_url.rstrip("/")
        self._session = requests.Session()
        self.render_mode = render_mode
        self.frame_skip = frame_skip

        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_agents, 4), dtype=np.float32)

        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_agents, 5), dtype=np.float32)

        # Pygame rendering state
        self.window_size = 800
        self._surface = None
        self._pygame_initialized = False

    def _request(self, method: str, path: str, **kwargs):
        """Thin wrapper around requests with a timeout and descriptive errors."""
        url = f"{self.server_url}{path}"
        try:
            resp = getattr(self._session, method)(url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            raise EnvironmentError(f"Server request failed ({method.upper()} {url}): {e}") from e

    def _ensure_pygame(self):
        """Initialize Pygame exactly once."""
        if not self._pygame_initialized:
            pygame.init()
            self._pygame_initialized = True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        resp = self._request("post", "/rl/reset_batch", json={"num_agents": self.num_agents})
        data_list = resp.json()

        obs = np.array([d["obs"] for d in data_list], dtype=np.float32)

        infos = {}
        for d in data_list:
            for k, v in d.get("info", {}).items():
                infos.setdefault(k, []).append(v)

        return obs, infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        # Send actions as a 2-D list — avoids creating 100 dict objects per step.
        # Node.js accepts both array [dx, dy, split, fire] and object {dx, dy, split, fire}.
        resp = self._request(
            "post", "/rl/step_batch",
            json={"actions": actions.tolist(), "skip": self.frame_skip},
        )
        data_list = resp.json()

        obs = np.array([d["obs"] for d in data_list], dtype=np.float32)
        rewards = np.array([d["reward"] for d in data_list], dtype=np.float32)
        terminated = np.array([d["done"] for d in data_list], dtype=bool)
        truncated = np.zeros(self.num_agents, dtype=bool)

        infos = {}
        for d in data_list:
            for k, v in d.get("info", {}).items():
                infos.setdefault(k, []).append(v)

        return obs, rewards, terminated, truncated, infos

    def render(self) -> Optional[np.ndarray]:
        """
        Fetches render state on demand from /rl/render_state and draws it.
        Only called by RecordVideo during recording episodes — zero cost otherwise.
        """
        if self.render_mode != "rgb_array":
            return None

        resp = self._request("get", "/rl/render_state")
        data = resp.json()

        render_bg = data.get("render_bg")
        all_player_cells = data.get("player_cells", [])

        if not render_bg:
            return None

        if self._surface is None:
            self._ensure_pygame()
            self._surface = pygame.Surface((self.window_size, self.window_size))

        self._surface.fill((255, 255, 255))

        map_w = render_bg.get("map", {}).get("width", 5000)
        map_h = render_bg.get("map", {}).get("height", 5000)
        scale_x = self.window_size / max(map_w, 1)
        scale_y = self.window_size / max(map_h, 1)
        scale = max(scale_x, scale_y)

        def draw_circle(color, x, y, radius):
            pygame.draw.circle(
                self._surface, color,
                (int(x * scale_x), int(y * scale_y)),
                max(1, int(radius * scale))
            )

        for f in render_bg.get("food", []):
            draw_circle((0, 0, 255), f["x"], f["y"], 5)
        for mf in render_bg.get("massFood", []):
            draw_circle((0, 255, 255), mf["x"], mf["y"], 8)
        for v in render_bg.get("viruses", []):
            draw_circle((0, 255, 0), v["x"], v["y"], v.get("radius", 30))

        for i, cells in enumerate(all_player_cells):
            color = (
                (i * 123) % 255,
                (i * 231) % 255,
                (i * 321) % 255
            ) if i > 0 else (0, 0, 0)
            for cell in cells:
                draw_circle(color, cell["x"], cell["y"], cell.get("radius", 10))

        return np.transpose(np.array(pygame.surfarray.pixels3d(self._surface)), axes=(1, 0, 2))

    def close(self):
        try:
            if self._pygame_initialized:
                pygame.quit()
                self._pygame_initialized = False
                self._surface = None
        finally:
            super().close()
