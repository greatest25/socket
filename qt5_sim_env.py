from uav_env_core.environment import Qt5SimUAVEnv

if __name__ == "__main__":
    env = Qt5SimUAVEnv(map_width=1280, map_height=800, 
                       num_agents=3, num_enemies=3, 
                       num_static_obstacles=2, num_dynamic_obstacles=1)
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs("env_test_frames", exist_ok=True)

    for episode in range(2): # 测试2个episode
        print(f"--- Episode {episode+1} ---")
        obs = env.reset()
        env.render()
        plt.savefig(f"env_test_frames/ep{episode+1}_step_0.png")
        
        total_episode_reward = np.zeros(env.num_agents)
        for step in range(200): # 每个episode测试200步
            # 随机动作 (假设策略输出归一化的加速度，范围[-1, 1])
            actions = [env.action_space[f'agent_{i}'].sample() for i in range(env.num_agents)]
            
            obs, rewards, dones = env.step(actions)
            total_episode_reward += rewards

            if (step + 1) % 10 == 0: # 每10帧保存一次
                env.render()
                plt.savefig(f"env_test_frames/ep{episode+1}_step_{step+1}.png")
            
            print(f"Step {step+1}: Rewards = {rewards}, UAV Health = {env.uav_health}, Enemy Health = {env.enemy_health}")
            
            if all(dones):
                print(f"Episode {episode+1} finished at step {step+1} because all agents are done.")
                break
        print(f"Episode {episode+1} total rewards: {total_episode_reward}")

    env.close()
    print("Environment test completed. Check 'env_test_frames' directory for images.")