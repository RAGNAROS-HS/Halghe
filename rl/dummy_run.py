import gymnasium as gym
from env import HalgheEnv

def main():
    num_agents = 100
    print(f"Initializing {num_agents} environments for dummy run...")
    
    envs = []
    for i in range(num_agents):
        # Only render the first environment to save the video
        render_mode = "rgb_array" if i == 0 else None
        base_env = HalgheEnv(render_mode=render_mode)
        
        if i == 0:
            env = gym.wrappers.RecordVideo(
                base_env, 
                video_folder="videos/dummy_runs",
                episode_trigger=lambda ep: True
            )
        else:
            env = base_env
            
        envs.append(env)
    
    print("Resetting all agents...")
    for env in envs:
        env.reset()
        
    print(f"Starting random movements for {num_agents} agents...")
    
    # We'll step for 500 frames total
    for step_num in range(500):
        for i, env in enumerate(envs):
            action = env.action_space.sample()
            
            # Note: step() returns next_obs, reward, terminated, truncated, info
            _, _, terminated, truncated, _ = env.step(action)
            
            # If an agent dies/terminates, just reset it immediately so it keeps playing
            if terminated or truncated:
                env.reset()
                
        if (step_num + 1) % 50 == 0:
            print(f"Completed {step_num + 1}/500 steps.")

    print("Closing all environments...")
    for env in envs:
        env.close()
        
    print("Dummy run complete. Video should be saved in videos/dummy_runs/")

if __name__ == "__main__":
    main()
