import numpy as np
from maddpg import MADDPG
from sim_env import UAVEnv
from buffer import MultiAgentReplayBuffer
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

print("程序开始执行...")  # 添加调试输出

def obs_list_to_state_vector(obs):
    """
    将观察列表转换为状态向量
    
    参数:
        obs: 包含多个智能体观察的列表
        
    返回:
        state: 连接后的状态向量，用于Critic网络输入
    """
    state = np.hstack([np.ravel(o) for o in obs])
    return state

def save_image(env_render, filename):
    """
    保存环境渲染图像到文件
    
    参数:
        env_render: 环境渲染的RGBA图像缓冲区
        filename: 保存图像的文件路径
    """
    # 将RGBA缓冲区转换为RGB图像
    image = Image.fromarray(env_render, 'RGBA')  # 使用'RGBA'模式，因为缓冲区包含透明度
    image = image.convert('RGB')  # 如果不需要透明度，转换为'RGB'
    
    image.save(filename)

if __name__ == '__main__':
    print("进入主函数...")  # 添加调试输出
    
    # 初始化UAV环境
    env = UAVEnv()
    print("环境初始化完成...")  # 添加调试输出
    # print(env.info)
    
    # 获取智能体数量和观察空间维度
    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    # 动作空间是数组列表，假设每个智能体有相同的动作空间
    n_actions = 2
    
    # 初始化MADDPG算法
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=128, fc2=128,
                           alpha=0.0001, beta=0.003, scenario='UAV_Round_up',
                           chkpt_dir='tmp/maddpg/')

    # 初始化经验回放缓冲区
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=256)

    # 训练参数设置
    PRINT_INTERVAL = 100  # 打印间隔
    N_GAMES = 5000        # 训练总回合数
    MAX_STEPS = 100       # 每回合最大步数
    total_steps = 0       # 总步数计数器
    score_history = []    # 分数历史记录
    target_score_history = []  # 目标智能体分数历史
    evaluate = False      # 是否处于评估模式
    best_score = -30      # 最佳分数初始值

    # 根据模式加载检查点或开始训练
    if evaluate:
        maddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training start----')
    
    # 主训练循环
    for i in range(N_GAMES):
        # 重置环境
        obs = env.reset()
        score = 0
        score_target = 0
        dones = [False]*n_agents
        episode_step = 0
        
        # 单回合循环
        while not any(dones):
            # 评估模式下渲染环境
            if evaluate:
                # env.render()
                env_render = env.render()
                if episode_step % 10 == 0:
                    # 每10步保存一次图像
                    filename = f'images/episode_{i}_step_{episode_step}.png'
                    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 如果目录不存在则创建
                    save_image(env_render, filename)
                # time.sleep(0.01)
                
            # 选择动作
            actions = maddpg_agents.choose_action(obs, total_steps, evaluate)
            
            # 执行动作并获取新状态、奖励和终止标志
            obs_, rewards, dones = env.step(actions)

            # 将观察转换为状态向量
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            # 达到最大步数时强制终止
            if episode_step >= MAX_STEPS:
                dones = [True]*n_agents

            # 存储经验到回放缓冲区
            memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

            # 定期学习
            if total_steps % 10 == 0 and not evaluate:
                maddpg_agents.learn(memory, total_steps)

            # 更新状态和分数
            obs = obs_
            score += sum(rewards[0:2])  # 围捕无人机的总分数
            score_target += rewards[-1]  # 目标无人机的分数
            total_steps += 1
            episode_step += 1

        # 记录分数
        score_history.append(score)
        target_score_history.append(score_target)
        
        # 计算平均分数
        avg_score = np.mean(score_history[-100:])
        avg_target_score = np.mean(target_score_history[-100:])
        
        # 训练模式下保存最佳模型
        if not evaluate:
            if i % PRINT_INTERVAL == 0 and i > 0 and avg_score > best_score:
                print('New best score', avg_score, '>', best_score, 'saving models...')
                maddpg_agents.save_checkpoint()
                best_score = avg_score
                
        # 定期打印训练进度
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score), '; average target score {:.1f}'.format(avg_target_score))
    
    # 保存训练数据
    file_name = 'score_history.csv'
    if not os.path.exists(file_name):
        pd.DataFrame([score_history]).to_csv(file_name, header=False, index=False)
    else:
        with open(file_name, 'a') as f:
            pd.DataFrame([score_history]).to_csv(f, header=False, index=False)