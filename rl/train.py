import numpy as np
import tensorflow as tf
import gymnasium as gym
from vec_env import BatchedHalgheEnv


class SimpleActor(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.mu = tf.keras.layers.Dense(action_dim, activation='tanh')  # Actions bound to [-1, 1]

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.mu(x)


def compute_discounted_returns(rewards, gamma=0.99):
    """
    Computes discounted returns for REINFORCE. O(n) time and space.
    Returns list of arrays of shape (num_agents,).
    """
    if not rewards:
        return []
    T = len(rewards)
    discounted = [None] * T  # pre-allocate, avoids O(n^2) list-prepend
    ret = np.zeros_like(rewards[-1])
    for i in range(T - 1, -1, -1):
        ret = rewards[i] + gamma * ret
        discounted[i] = ret.copy()
    return discounted


def main():
    # 1. Initialize the Batched Environment
    num_agents = 100
    base_env = BatchedHalgheEnv(num_agents=num_agents, render_mode="rgb_array", frame_skip=4)

    # Wrap to record videos (render() is only called during recording episodes)
    env = gym.wrappers.RecordVideo(
        base_env,
        video_folder="videos/train_runs_batched",
        episode_trigger=lambda ep: ep % 10 == 0  # Record every 10 episodes
    )

    action_dim = env.action_space.shape[1]

    # 2. Init Model and Optimizer (Basic REINFORCE / Policy Gradient implementation)
    model = SimpleActor(action_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    stddev = 0.2

    # JIT-compiled inference: traced once on first call, fast on all subsequent calls.
    @tf.function
    def get_actions(obs):
        mu = model(obs)
        noise = tf.random.normal(tf.shape(mu), stddev=stddev)
        return tf.clip_by_value(mu + noise, -1.0, 1.0)

    # JIT-compiled training step: avoids Python overhead for the hot path.
    @tf.function
    def train_step(states, actions, returns):
        with tf.GradientTape() as tape:
            mu = model(states)
            # Approximate log prob of normal distribution
            log_probs = -0.5 * tf.reduce_sum(tf.square((actions - mu) / stddev), axis=1)
            # Policy Gradient Loss = -E[ log_prob * Reward ]
            loss = -tf.reduce_mean(log_probs * returns)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    episodes = 500
    # 2000 steps * 4 frame_skip = 8000 server ticks per episode.
    max_steps = 2000

    for ep in range(episodes):
        obs, _ = env.reset()  # shape: (num_agents, obs_dim)
        done = False
        step_count = 0

        # Store raw numpy arrays — cheaper than storing TF tensors during the episode.
        episode_states = []   # list of np.ndarray (num_agents, obs_dim)
        episode_actions = []  # list of np.ndarray (num_agents, action_dim)
        episode_rewards = []  # list of np.ndarray (num_agents,)

        while not done and step_count < max_steps:
            action = get_actions(tf.constant(obs, dtype=tf.float32))
            action_np = action.numpy()  # (num_agents, action_dim)

            next_obs, reward, terminated, truncated, _ = env.step(action_np)

            episode_states.append(obs)
            episode_actions.append(action_np)
            episode_rewards.append(reward)  # reward shape: (num_agents,)

            obs = next_obs
            done = np.all(terminated)
            step_count += 1

        # --- End of Episode Training Update ---
        discounted_returns = compute_discounted_returns(episode_rewards)

        if episode_states:
            # Concatenate all (steps * num_agents) transitions using numpy — fast single copy.
            states_np = np.concatenate(episode_states, axis=0).astype(np.float32)
            actions_np = np.concatenate(episode_actions, axis=0).astype(np.float32)
            returns_np = np.concatenate(discounted_returns, axis=0).astype(np.float32)

            # Normalize returns to reduce variance of gradient estimates
            returns_np = (returns_np - returns_np.mean()) / (returns_np.std() + 1e-8)

            loss = train_step(
                tf.constant(states_np),
                tf.constant(actions_np),
                tf.constant(returns_np),
            )

            avg_per_agent_reward = sum(np.mean(r) for r in episode_rewards)
            print(f"Episode {ep+1} | Avg Reward/Agent: {avg_per_agent_reward:.3f} | Loss: {loss.numpy():.3f} | Steps: {step_count}")

    env.close()


if __name__ == "__main__":
    main()
