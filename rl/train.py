import argparse
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


class SimpleCritic(tf.keras.Model):
    """Value network — estimates V(s) as a baseline to reduce policy gradient variance."""
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return tf.squeeze(self.value(x), axis=-1)  # shape: (batch,)


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


def parse_args():
    p = argparse.ArgumentParser(description="Train Actor-Critic agent on Halghe")
    p.add_argument("--num-agents",    type=int,   default=100,   help="Number of parallel agents")
    p.add_argument("--episodes",      type=int,   default=500,   help="Training episodes")
    p.add_argument("--max-steps",     type=int,   default=50,  help="Max steps per episode")
    p.add_argument("--frame-skip",    type=int,   default=4,     help="Server ticks per action")
    p.add_argument("--actor-lr",      type=float, default=0.001, help="Actor learning rate")
    p.add_argument("--critic-lr",     type=float, default=0.001, help="Critic learning rate")
    p.add_argument("--stddev-start",  type=float, default=0.2,   help="Initial exploration noise")
    p.add_argument("--stddev-end",    type=float, default=0.05,  help="Final exploration noise")
    p.add_argument("--gamma",         type=float, default=0.99,  help="Discount factor")
    p.add_argument("--server-url",    type=str,   default="http://localhost:3000", help="Game server URL")
    p.add_argument("--log-dir",       type=str,   default="logs/train", help="TensorBoard log directory")
    p.add_argument("--video-dir",     type=str,   default="videos/train_runs_batched", help="Video output directory")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Initialize the Batched Environment
    num_agents = args.num_agents
    base_env = BatchedHalgheEnv(num_agents=num_agents, server_url=args.server_url,
                                render_mode="rgb_array", frame_skip=args.frame_skip)

    # Wrap to record videos (render() is only called during recording episodes)
    env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=args.video_dir,
        episode_trigger=lambda ep: ep % 10 == 0  # Record every 10 episodes
    )

    action_dim = env.action_space.shape[1]

    # 2. Init Actor-Critic Models and Optimizers
    actor = SimpleActor(action_dim)
    critic = SimpleCritic()
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=args.actor_lr)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=args.critic_lr)

    episodes    = args.episodes
    max_steps   = args.max_steps
    stddev_start = args.stddev_start
    stddev_end   = args.stddev_end

    tb_writer = tf.summary.create_file_writer(args.log_dir)

    # JIT-compiled inference: traced once on first call, fast on all subsequent calls.
    @tf.function
    def get_actions(obs, stddev):
        mu = actor(obs)
        noise = tf.random.normal(tf.shape(mu), stddev=stddev)
        return tf.clip_by_value(mu + noise, -1.0, 1.0)

    # JIT-compiled Actor-Critic training step.
    @tf.function
    def train_step(states, actions, returns, stddev):
        # --- Critic update ---
        with tf.GradientTape() as critic_tape:
            values = critic(states)
            critic_loss = tf.reduce_mean(tf.square(returns - values))
        critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

        # --- Actor update with advantage = returns - baseline ---
        with tf.GradientTape() as actor_tape:
            mu = actor(states)
            values_baseline = tf.stop_gradient(critic(states))
            advantages = returns - values_baseline

            # Full Gaussian log-probability: -0.5*(sum((a-mu)^2/sigma^2) + k*log(2*pi*sigma^2))
            action_dim = tf.cast(tf.shape(actions)[1], tf.float32)
            log_probs = -0.5 * (
                tf.reduce_sum(tf.square((actions - mu) / stddev), axis=1)
                + action_dim * tf.math.log(2.0 * np.pi * stddev ** 2)
            )
            actor_loss = -tf.reduce_mean(log_probs * advantages)

        actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
        return actor_loss, critic_loss

    for ep in range(episodes):
        obs, _ = env.reset()  # shape: (num_agents, obs_dim)
        step_count = 0

        # Linearly decay exploration noise over training
        stddev = float(stddev_start - (stddev_start - stddev_end) * ep / max(episodes - 1, 1))

        # Store raw numpy arrays — cheaper than storing TF tensors during the episode.
        episode_states = []   # list of np.ndarray (num_agents, obs_dim)
        episode_actions = []  # list of np.ndarray (num_agents, action_dim)
        episode_rewards = []  # list of np.ndarray (num_agents,)

        while step_count < max_steps:
            action = get_actions(tf.constant(obs, dtype=tf.float32), tf.constant(stddev, dtype=tf.float32))
            action_np = action.numpy()  # (num_agents, action_dim)

            next_obs, reward, terminated, truncated, _ = env.step(action_np)

            episode_states.append(obs)
            episode_actions.append(action_np)
            episode_rewards.append(reward)  # reward shape: (num_agents,)

            obs = next_obs
            # Each agent resets independently when it dies (auto-reset in step_batch).
            # End the episode only on time limit, not when any single agent dies.
            step_count += 1

        # --- End of Episode Training Update ---
        discounted_returns = compute_discounted_returns(episode_rewards, gamma=args.gamma)

        if episode_states:
            # Concatenate all (steps * num_agents) transitions using numpy — fast single copy.
            states_np = np.concatenate(episode_states, axis=0).astype(np.float32)
            actions_np = np.concatenate(episode_actions, axis=0).astype(np.float32)
            returns_np = np.concatenate(discounted_returns, axis=0).astype(np.float32)

            # Normalize returns to reduce variance of gradient estimates
            returns_np = (returns_np - returns_np.mean()) / (returns_np.std() + 1e-8)

            actor_loss, critic_loss = train_step(
                tf.constant(states_np),
                tf.constant(actions_np),
                tf.constant(returns_np),
                tf.constant(stddev, dtype=tf.float32),
            )

            avg_per_agent_reward = sum(np.mean(r) for r in episode_rewards)
            print(
                f"Episode {ep+1}/{episodes} | "
                f"Avg Reward/Agent: {avg_per_agent_reward:.3f} | "
                f"Actor Loss: {actor_loss.numpy():.3f} | "
                f"Critic Loss: {critic_loss.numpy():.3f} | "
                f"Stddev: {stddev:.3f} | "
                f"Steps: {step_count}"
            )
            with tb_writer.as_default():
                tf.summary.scalar("reward/avg_per_agent", avg_per_agent_reward, step=ep)
                tf.summary.scalar("loss/actor", actor_loss, step=ep)
                tf.summary.scalar("loss/critic", critic_loss, step=ep)
                tf.summary.scalar("exploration/stddev", stddev, step=ep)
                tb_writer.flush()

    env.close()


if __name__ == "__main__":
    main()
