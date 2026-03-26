import numpy as np
import tensorflow as tf
import gymnasium as gym
from vec_env import BatchedHalgheEnv

class SimpleActor(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.mu = tf.keras.layers.Dense(action_dim, activation='tanh') # Actions bound to [-1, 1]

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.mu(x)

def compute_discounted_returns(rewards, gamma=0.99):
    """
    Computes discounted returns for REINFORCE.
    Returns list of arrays of shape (num_agents,)
    """
    discounted = []
    # If episode_rewards is empty or wrong shape, safeguard
    if not rewards:
        return []
        
    ret = np.zeros_like(rewards[0])
    for r in reversed(rewards):
        ret = r + gamma * ret
        discounted.insert(0, ret.copy())
    return discounted

def main():
    # 1. Initialize the Batched Environment
    num_agents = 100
    base_env = BatchedHalgheEnv(num_agents=num_agents, render_mode="rgb_array", frame_skip=4)
    
    # Wrap it to record videos (captures EVERY frame out of the base env for smooth video)
    env = gym.wrappers.RecordVideo(
        base_env, 
        video_folder="videos/train_runs_batched",
        episode_trigger=lambda ep: ep % 10 == 0 # Record every 10 eps
    )
    
    # Apply FrameSkip occurs directly in the Node.js backend to bypass Python HTTP overhead!
    
    action_dim = env.action_space.shape[1]
    obs_dim = base_env.single_observation_space.shape[0]

    # 2. Init Model and Optimizer (Basic REINFORCE / Policy Gradient implementation)
    model = SimpleActor(action_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    episodes = 500
    # 2000 steps * 4 frame_skip = 8000 server ticks per episode.
    # At ~40ms per tick, this allows for ~320 seconds (over 5 minutes) of real gameplay per episode!
    max_steps = 2000 
    
    for ep in range(episodes):
        obs, _ = env.reset() # shape: (num_agents, obs_dim)
        done = False
        step_count = 0
        
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        while not done and step_count < max_steps:
            # Prepare state
            state_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
            
            # Predict mean of action distribution
            mu = model(state_tensor) # shape: (num_agents, action_dim)
            
            # Sample continuous actions around the mean with fixed stddev for exploration
            stddev = 0.2
            noise = tf.random.normal(shape=mu.shape, stddev=stddev)
            action = tf.clip_by_value(mu + noise, -1.0, 1.0)
            
            # Step the environment with chosen random actions
            next_obs, reward, terminated, truncated, _ = env.step(action.numpy())
            
            episode_states.append(state_tensor)
            episode_actions.append(action)
            episode_rewards.append(reward) # reward shape: (num_agents,)
            
            obs = next_obs
            
            # A batched env is "done" usually when all terminate, but we also rely on max_steps
            done = np.all(terminated)
            step_count += 1
            
        # --- End of Episode Training Updates ---
        discounted_returns = compute_discounted_returns(episode_rewards)
        
        if len(episode_states) > 0:
            # Flatten tensors across the batch and time dimensions for parallel loss computation
            states_flat = tf.concat(episode_states, axis=0) # (steps * num_agents, obs_dim)
            actions_flat = tf.concat(episode_actions, axis=0) # (steps * num_agents, action_dim)
            returns_flat = tf.convert_to_tensor(np.concatenate(discounted_returns, axis=0), dtype=tf.float32)
            
            # Normalize returns to decrease variance of gradient estimations
            returns_flat = (returns_flat - tf.reduce_mean(returns_flat)) / (tf.math.reduce_std(returns_flat) + 1e-8)
            
            with tf.GradientTape() as tape:
                mu = model(states_flat)
                
                # Approximate log prob of normal distribution
                log_probs = -0.5 * tf.reduce_sum(tf.square((actions_flat - mu) / stddev), axis=1)
                
                # Policy Gradient Loss = - E[ log_prob * Reward ]
                loss = -tf.reduce_mean(log_probs * returns_flat)
                
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # Calculate simple average sum of rewards for tracking
            avg_per_agent_reward = sum([np.mean(r) for r in episode_rewards])
            print(f"Episode {ep+1} finished. Avg Reward/Agent: {avg_per_agent_reward:.3f}, Loss: {loss.numpy():.3f}, Steps: {step_count}")

    env.close()

if __name__ == "__main__":
    main()
