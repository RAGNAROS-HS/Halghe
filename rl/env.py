"""
HalgheEnv — Gymnasium environment skeleton for the Halghe agar.io RL API.

The server (POST /rl/reset, POST /rl/step) returns raw game state JSON.
This class handles the HTTP communication. YOU fill in _build_observation()
to transform the raw state into whatever tensor your model needs.

Usage:
    env = HalgheEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import gymnasium
from gymnasium import spaces
import numpy as np
import requests


class HalgheEnv(gymnasium.Env):
    """
    Gymnasium wrapper around the Halghe RL HTTP API.

    The server handles all game logic. This env just:
      1. Sends actions to POST /rl/step
      2. Receives raw game state JSON back
      3. Calls _build_observation() for you to transform it

    Attributes:
        server_url (str): Base URL of the game server (default: http://localhost:3000)
    """

    metadata = {"render_modes": []}

    def __init__(self, server_url="http://localhost:3000"):
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self._session = requests.Session()

        # Fetch game config from the server so you can use it for normalization
        self.game_config = self._get_config()

        # ---------------------------------------------------------------
        # ACTION SPACE
        # 4-element continuous vector:
        #   [0] dx    ∈ [-1, 1]  — horizontal move direction
        #   [1] dy    ∈ [-1, 1]  — vertical move direction
        #   [2] split ∈ [-1, 1]  — >0 triggers split  (thresholded in _decode_action)
        #   [3] fire  ∈ [-1, 1]  — >0 triggers fire    (thresholded in _decode_action)
        # ---------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # ---------------------------------------------------------------
        # OBSERVATION SPACE
        # TODO: Define this once you implement _build_observation().
        # For now it's a placeholder flat vector of size 1.
        # ---------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

        self._raw_state = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        resp = self._session.post(f"{self.server_url}/rl/reset")
        resp.raise_for_status()
        data = resp.json()
        self._raw_state = data["state"]
        obs = self._build_observation(self._raw_state)
        return obs, data.get("info", {})

    def step(self, action):
        action_dict = self._decode_action(action)
        resp = self._session.post(
            f"{self.server_url}/rl/step",
            json={"action": action_dict},
        )
        resp.raise_for_status()
        data = resp.json()

        self._raw_state = data["state"]
        obs = self._build_observation(self._raw_state)
        reward = float(data["reward"])
        terminated = bool(data["done"])
        truncated = False
        info = data.get("info", {})
        # Attach raw state to info so you can inspect it during debugging
        info["raw_state"] = self._raw_state

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # YOU FILL THESE IN
    # ------------------------------------------------------------------

    def _build_observation(self, raw_state: dict) -> np.ndarray:
        """
        Transform the raw game state JSON into your observation tensor.

        raw_state contains:
            - player:   { x, y, massTotal, cells: [{x, y, mass, radius}, ...] }
            - food:     [{ x, y }, ...]
            - viruses:  [{ x, y, mass, radius }, ...]
            - enemies:  [{ x, y, massTotal, cells: [...] }, ...]
            - massFood: [{ x, y, mass }, ...]
            - map:      { width, height }

        Returns:
            np.ndarray matching self.observation_space
        """
        # PLACEHOLDER — returns a dummy observation.
        # Replace this with your feature engineering (flat vector, grid, raycasting, etc.)
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    # ------------------------------------------------------------------
    # Action encoding / decoding
    # ------------------------------------------------------------------

    def _decode_action(self, action) -> dict:
        """
        Convert the raw action array [dx, dy, split, fire] into the dict
        the server expects.

        You can customize the thresholds for split/fire here.
        """
        return {
            "dx": float(np.clip(action[0], -1, 1)),
            "dy": float(np.clip(action[1], -1, 1)),
            "split": int(action[2] > 0),
            "fire": int(action[3] > 0),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_config(self) -> dict:
        """Fetch game configuration from the server."""
        try:
            resp = self._session.get(f"{self.server_url}/rl/config")
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError:
            print("[WARN] Could not connect to game server to fetch config. Using empty config.")
            return {}

    @property
    def raw_state(self):
        """Access the last raw state received from the server."""
        return self._raw_state
