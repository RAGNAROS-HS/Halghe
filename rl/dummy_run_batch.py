import gymnasium as gym
from vec_env import BatchedHalgheEnv

def main():
    num_agents = 100
    print(f"Initializing {num_agents} environments as a single batched VectorEnv...")
    
    # We initialize one environment that handles 100 agents at once
    base_env = BatchedHalgheEnv(num_agents=num_agents, render_mode="rgb_array")
    
    # Gymnasium wrapper works out of the box!
    env = gym.wrappers.RecordVideo(
        base_env, 
        video_folder="videos/dummy_runs_batched",
        episode_trigger=lambda ep: True
    )
    
    # The new vectorized env returns batched arrays
    obs, info = env.reset()
    
    print(f"Starting random movements. Actions shape will be: {env.action_space.shape}")
    
    # We'll step for 500 frames total
    for step_num in range(500):
        # We sample once, but we get an array of shape (100, 4)!
        actions = env.action_space.sample()
        
        # ONE HTTP request per step! So much faster.
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        if (step_num + 1) % 50 == 0:
            print(f"Completed {step_num + 1}/500 steps.")

    env.close()
    
    print("Dummy run complete. Video should be saved in videos/dummy_runs_batched/")

if __name__ == "__main__":
    main()
