import numpy as np
from maddpg import MADDPG
from qt5_sim_env import Qt5SimUAVEnv
import time
import os
import matplotlib.pyplot as plt
import warnings
from PIL import Image
import matplotlib.animation as animation
warnings.filterwarnings('ignore')

def save_image(env_render, filename):
    # Convert the RGBA buffer to an RGB image
    image = Image.fromarray(env_render, 'RGBA')
    image = image.convert('RGB')
    image.save(filename)

def save_gif(frames, filename, fps=10):
    # Save frames as GIF
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=1000//fps,
        loop=0
    )

if __name__ == '__main__':
    # 创建3v3的Qt5模拟环境
    env = Qt5SimUAVEnv(num_agents=3, num_enemies=3, detection_radius=0.5, attack_radius=0.3)
    
    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    # 动作空间是二维的 [a_x, a_y]
    n_actions = 2
    
    # 创建MADDPG代理
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=128, fc2=128,
                           alpha=0.0001, beta=0.003, scenario='UAV_Qt5_3v3',
                           chkpt_dir='tmp/maddpg_qt5/')

    # 加载训练好的模型
    maddpg_agents.load_checkpoint()
    print('---- Models loaded, evaluating ----')
    
    # 评估参数
    n_episodes = 5
    max_steps = 300
    
    # 统计数据
    success_count = 0
    steps_to_success = []
    
    # 创建保存目录
    os.makedirs('eval_results', exist_ok=True)
    
    for episode in range(n_episodes):
        print(f'Episode {episode+1}/{n_episodes}')
        
        obs = env.reset()
        frames = []  # 用于保存GIF的帧
        
        # 记录本次episode的数据
        episode_reward = 0
        enemies_defeated = 0
        search_time = 0  # 记录搜索时间
        capture_time = 0  # 记录围捕时间
        
        for step in range(max_steps):
            # 渲染环境并保存图像
            env_render = env.render()
            frames.append(Image.fromarray(env_render).convert('RGB'))
            
            if step % 20 == 0:
                filename = f'eval_results/episode_{episode+1}_step_{step}.png'
                save_image(env_render, filename)
            
            # 选择动作
            actions = maddpg_agents.choose_action(obs, 0, evaluate=True)
            
            # 执行动作
            obs_, rewards, dones = env.step(actions)
            
            # 更新观察
            obs = obs_
            
            # 累加奖励
            episode_reward += sum(rewards)
            
            # 记录搜索和围捕时间
            if env.search_mode:
                search_time += 1
            else:
                capture_time += 1
            
            # 检查击败敌机数量
            current_enemies_defeated = sum(1 for health in env.enemy_health if health <= 0)
            if current_enemies_defeated > enemies_defeated:
                print(f"  Step {step+1}: 击败了 {current_enemies_defeated - enemies_defeated} 架敌机")
                enemies_defeated = current_enemies_defeated
            
            # 检查是否击败所有敌机
            if all(health <= 0 for health in env.enemy_health):
                success_count += 1
                steps_to_success.append(step + 1)
                print(f'  Success! All enemies defeated at step {step+1}')
                break
            
            # 检查是否结束
            if any(dones) or step == max_steps - 1:
                if not all(health <= 0 for health in env.enemy_health):
                    print(f'  Failed to defeat all enemies within {max_steps} steps')
                    print(f'  Enemies defeated: {enemies_defeated}/{env.num_enemies}')
                break
        
        # 保存本次episode的GIF
        save_gif(frames, f'eval_results/episode_{episode+1}.gif', fps=20)
        
        # 打印本次episode的统计信息
        print(f'  Total reward: {episode_reward:.2f}')
        print(f'  Search time: {search_time} steps')
        print(f'  Capture time: {capture_time} steps')
        print(f'  Enemies defeated: {enemies_defeated}/{env.num_enemies}')
        print('-----------------------------------')
    
    # 打印总体统计信息
    print('\nEvaluation Results:')
    print(f'Success rate: {success_count}/{n_episodes} = {success_count/n_episodes*100:.1f}%')
    if success_count > 0:
        avg_steps = sum(steps_to_success) / len(steps_to_success)
        print(f'Average steps to success: {avg_steps:.1f}')
    
    env.close()