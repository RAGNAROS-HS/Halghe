"""
test_env.py — Sanity check for the Halghe RL environment.

Tests the raw HTTP connection and game stepping WITHOUT requiring
_build_observation() to be implemented. Calls the server directly
and prints the raw state to verify the API is working.

Usage:
    1. Start the game server:   cd <project_root> && npm start
    2. Run this test:           python test_env.py

If _build_observation() is implemented, also tests the full Gymnasium env.
"""

import json
import sys
import requests


SERVER_URL = "http://localhost:3000"
NUM_TEST_STEPS = 100


def test_raw_api():
    """Test the server's RL API directly via HTTP (no Gymnasium)."""
    print("=" * 60)
    print("TEST 1: Raw HTTP API")
    print("=" * 60)

    session = requests.Session()

    # --- Reset ---
    print("\n[POST /rl/reset]")
    resp = session.post(f"{SERVER_URL}/rl/reset")
    if resp.status_code != 200:
        print(f"  FAIL: status {resp.status_code} — {resp.text}")
        return False
    data = resp.json()
    print(f"  OK: player at ({data['state']['player']['x']:.0f}, {data['state']['player']['y']:.0f}), "
          f"mass={data['state']['player']['massTotal']}, "
          f"food_count={len(data['state']['food'])}, "
          f"virus_count={len(data['state']['viruses'])}")

    # --- Steps ---
    print(f"\n[POST /rl/step] × {NUM_TEST_STEPS}")
    import random

    total_reward = 0
    for i in range(NUM_TEST_STEPS):
        action = {
            "dx": random.uniform(-1, 1),
            "dy": random.uniform(-1, 1),
            "split": int(random.random() > 0.95),   # rarely split
            "fire": int(random.random() > 0.95),     # rarely fire
        }
        resp = session.post(f"{SERVER_URL}/rl/step", json={"action": action})
        if resp.status_code != 200:
            print(f"  FAIL at step {i}: status {resp.status_code} — {resp.text}")
            return False

        data = resp.json()
        total_reward += data["reward"]

        if i % 20 == 0 or data["done"]:
            print(f"  step={data['info']['step']:>4d}  "
                  f"mass={data['info']['mass']:>5d}  "
                  f"reward={data['reward']:>+7.2f}  "
                  f"total_reward={total_reward:>+8.2f}  "
                  f"done={data['done']}")

        if data["done"]:
            print(f"  Episode ended at step {i+1}. Resetting...")
            resp = session.post(f"{SERVER_URL}/rl/reset")
            if resp.status_code != 200:
                print(f"  FAIL: reset after death — {resp.text}")
                return False
            total_reward = 0

    print(f"\n  PASS: {NUM_TEST_STEPS} steps completed successfully.")
    return True


def test_gymnasium_env():
    """Test the full Gymnasium env (only if _build_observation is implemented)."""
    print("\n" + "=" * 60)
    print("TEST 2: Gymnasium Env")
    print("=" * 60)

    try:
        from env import HalgheEnv
    except ImportError:
        print("  SKIP: Could not import HalgheEnv")
        return True

    env = HalgheEnv(server_url=SERVER_URL)

    try:
        obs, info = env.reset()
        print(f"  reset() OK: obs.shape={obs.shape}, info={info}")
    except NotImplementedError:
        print("  SKIP: _build_observation() not implemented yet (expected).")
        return True
    except Exception as e:
        print(f"  FAIL: reset() raised {type(e).__name__}: {e}")
        return False

    for i in range(20):
        action = env.action_space.sample()
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except NotImplementedError:
            print(f"  SKIP: _build_observation() not implemented (raised at step {i}).")
            return True
        except Exception as e:
            print(f"  FAIL: step() raised {type(e).__name__} at step {i}: {e}")
            return False

        if i % 5 == 0:
            print(f"  step={i:>3d}  obs.shape={obs.shape}  reward={reward:>+.2f}  done={terminated}")

        if terminated:
            obs, info = env.reset()

    print(f"\n  PASS: Gymnasium env works.")
    return True


if __name__ == "__main__":
    print(f"Testing Halghe RL API at {SERVER_URL}\n")

    try:
        # Quick connectivity check
        requests.get(f"{SERVER_URL}/rl/config", timeout=3)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {SERVER_URL}.")
        print("Make sure the game server is running: npm start")
        sys.exit(1)

    results = []
    results.append(("Raw HTTP API", test_raw_api()))
    results.append(("Gymnasium Env", test_gymnasium_env()))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    sys.exit(0 if all(r[1] for r in results) else 1)
