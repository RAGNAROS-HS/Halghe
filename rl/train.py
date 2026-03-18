import numpy as np
import tensorflow as tf
import gymnasium as gym
from env import HalgheEnv

def main():
    # 1. Initialize the Environment
    base_env = HalgheEnv(render_mode="rgb_array")
    
    # Wrap it to record videos
    env = gym.wrappers.RecordVideo(
        base_env, 
        video_folder="videos",
        episode_trigger=lambda ep: ep % 50 == 0 # Record every 50th episode
    )
    
    # The action space in HalgheEnv is continuous: [dx, dy, split, fire]
    action_dim = env.action_space.shape[0]
    
    # ---------------------------------------------------------
    # TODO: Define and initialize your TensorFlow model(s) here
    # ---------------------------------------------------------
    # model = tf.keras.Sequential([...])
    # optimizer = tf.keras.optimizers.Adam()
    
    episodes = 1000
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        # ---------------------------------------------------------
        # TODO: Setup tf.GradientTape() if doing custom training steps
        # ---------------------------------------------------------
        while not done:
            
            # ---------------------------------------------------------
            # TODO: Convert `obs` to a Tensor and get an action from your model
            # ---------------------------------------------------------
            # Example placeholder: taking a random action
            action = env.action_space.sample() 
            
            # Step the environment with the chosen action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # ---------------------------------------------------------
            # TODO: Calculate loss and apply gradients
            # ---------------------------------------------------------
            # ...
            
            # Move to the next state
            obs = next_obs
            
        print(f"Episode {ep+1} finished. Total Reward: {episode_reward}")

    env.close()

if __name__ == "__main__":
    main()
