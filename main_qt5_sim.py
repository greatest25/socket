import numpy as np
import os
import torch
import time
from maddpg import MADDPG
from qt5_sim_env import Qt5SimUAVEnv
from buffer import MultiAgentReplayBuffer
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def plot_learning_curve(x, scores, figure_file):
    """绘制学习曲线"""
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    
    plt.figure(figsize=(12, 8))
    plt.plot(x, running_avg)
    plt.title('100回合滑动平均奖励')
    plt.xlabel('Episode')
    plt.ylabel('平均奖励')
    plt.savefig(figure_file)
    plt.close()

if __name__ == '__main__':
    # 创建3v3的Qt5模拟环境
    env = Qt5SimUAVEnv(map_width=1280, map_height=800, num_agents=3, num_enemies=3, num_static_obstacles=2, num_dynamic_obstacles=1)
    
    # 获取环境参数
    n_agents = env.num_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[f'agent_{i}'].shape[0])
    critic_dims = sum(actor_dims)
    
    # 动作空间是二维的 [a_x, a_y]
    n_actions = env.action_space['agent_0'].shape[0]
    
    # MADDPG参数
    batch_size = 1024
    memory_size = 1000000
    gamma = 0.95
    alpha = 0.0001
    beta = 0.001
    fc1 = 128
    fc2 = 128
    tau = 0.01
    
    # 创建保存目录
    scenario_name = 'UAV_Qt5_3v3_SensorCircle'
    save_dir = os.path.join('tmp', 'maddpg_qt5', scenario_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建经验回放缓冲区
    memory = MultiAgentReplayBuffer(memory_size, critic_dims, actor_dims, 
                                   n_actions, n_agents, batch_size)
    
    # 创建MADDPG代理
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                          fc1=fc1, fc2=fc2, 
                          alpha=alpha, beta=beta,
                          chkpt_dir=save_dir,
                          scenario=scenario_name,
                          gamma=gamma, tau=tau)
    
    # 训练参数
    total_episodes = 2000
    max_steps_per_episode = 300
    
    # 是否加载已有模型
    load_checkpoint = False
    if load_checkpoint:
        try:
            maddpg_agents.load_checkpoint()
            print('---- 加载已有模型 ----')
        except Exception as e:
            print(f'---- 加载模型失败: {e} ----')
            print('---- 从头开始训练 ----')
    
    # 记录训练过程
    score_history = []
    best_score = -np.inf
    
    # 记录胜率
    win_history = []
    
    # 记录训练开始时间
    start_time = time.time()
    global_step_counter = 0
    
    print('---- 开始训练 ----')
    
    for episode in range(total_episodes):
        obs = env.reset()
        score = 0
        
        # 记录当前episode的数据
        enemies_defeated_this_episode = 0
        search_time_steps = 0
        capture_time_steps = 0

         # 添加避障和追踪统计
        obstacle_collision_count = 0    # 障碍物碰撞次数
        obstacle_warning_count = 0      # 障碍物警告区次数
        obstacle_avoidance_success = 0  # 成功避障次数
        tracking_success_count = 0      # 成功追踪次数（保持在最佳追踪区间）
        tracking_total_count = 0        # 总追踪尝试次数
        
        for step in range(max_steps_per_episode):
            # 选择动作
            actions = maddpg_agents.choose_action(obs, global_step_counter, evaluate=False)
            
            # 执行动作
            obs_, rewards, dones_from_env = env.step(actions)
            
            # 准备 state 和 state_ (全局状态，所有 agent obs 的拼接)
            try:
                state = np.concatenate([np.array(o, dtype=np.float32) for o in obs])
                state_ = np.concatenate([np.array(o_, dtype=np.float32) for o_ in obs_])
            except ValueError as e:
                print(f"Error concatenating observations: {e}")
                print(f"Obs: {obs}")
                print(f"Obs_: {obs_}")
                continue
            
            # 记录搜索和围捕时间
            if env.search_mode:
                search_time_steps += 1
            else:
                capture_time_steps += 1

             # 统计避障情况
            for i in range(env.num_agents):
                if env.uav_health[i] <= 0:
                    continue
                    
                # 检查是否有探测到的障碍物
                if env.obstacle_detected[i]:
                    # 找到最近的障碍物表面距离
                    min_dist_to_obs = float('inf')
                    for obs_data in env.obstacle_detected[i]:
                        dist_surface = obs_data['distance_to_center'] - obs_data['radius'] - env.uav_radius_approx
                        if dist_surface < min_dist_to_obs:
                            min_dist_to_obs = dist_surface
                    
                    # 判断是碰撞、警告区还是成功避障
                    warning_zone_threshold = env.uav_radius_approx * 2.0
                    if min_dist_to_obs < 0:  # 实际碰撞
                        obstacle_collision_count += 1
                    elif min_dist_to_obs < warning_zone_threshold:  # 警告区
                        obstacle_warning_count += 1
                    else:  # 成功避障
                        obstacle_avoidance_success += 1
            
            # 统计追踪情况
            if not env.search_mode:  # 只在围捕模式下统计
                for i in range(env.num_agents):
                    if env.uav_health[i] <= 0:
                        continue
                        
                    for j in range(env.num_enemies):
                        if env.enemy_health[j] <= 0:
                            continue
                            
                        # 计算到敌人的距离
                        dist_to_enemy = np.linalg.norm(env.multi_current_pos[i] - env.enemy_pos[j])
                        
                        # 检查是否在探测范围内
                        if dist_to_enemy <= env.detection_radius:
                            # 追踪最佳区间：在攻击范围外一点，但在探测范围内
                            optimal_track_dist_min = env.attack_radius * 1.2
                            optimal_track_dist_max = env.detection_radius * 0.9
                            
                            tracking_total_count += 1
                            if optimal_track_dist_min < dist_to_enemy < optimal_track_dist_max:
                                tracking_success_count += 1
            
            # 存储经验
            memory.store_transition(obs, state, actions, rewards, obs_, state_, dones_from_env)
            
            # 学习
            if memory.ready():
                maddpg_agents.learn(memory, global_step_counter)
            
            # 更新观察
            obs = obs_
            score += sum(rewards)
            
            global_step_counter += 1
            
            # 检查击败敌机数量
            current_enemies_defeated = sum(1 for health in env.enemy_health if health <= 0)
            if current_enemies_defeated > enemies_defeated_this_episode:
                enemies_defeated_this_episode = current_enemies_defeated
            
            # 如果所有敌机都被击败或者所有我方无人机都被击毁，或者达到最大步数
            if all(dones_from_env) or step == max_steps_per_episode - 1:
                break
        
        # 记录得分
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        # 记录胜率
        if all(health <= 0 for health in env.enemy_health):
            win_history.append(1)
        else:
            win_history.append(0)
        
        current_win_rate = np.mean(win_history[-100:]) if len(win_history) >= 100 else np.mean(win_history) if win_history else 0
        
        # 计算避障成功率和追踪成功率
        total_obstacle_encounters = obstacle_collision_count + obstacle_warning_count + obstacle_avoidance_success
        #? 避障成功率 = 成功避障次数 / 总障碍物接触次数 * 100
        obstacle_avoidance_rate = obstacle_avoidance_success / total_obstacle_encounters * 100 if total_obstacle_encounters > 0 else 0
        
        tracking_success_rate = tracking_success_count / tracking_total_count * 100 if tracking_total_count > 0 else 0
        
        # 保存最佳模型
        if avg_score > best_score and len(score_history) >= 100:
            best_score = avg_score
            maddpg_agents.save_checkpoint()
            print(f'Episode {episode+1}: 保存新的最佳模型，平均得分 = {avg_score:.2f}')
        
        # 每100个episode打印一次学习曲线并保存检查点
        if (episode + 1) % 100 == 0:
            if len(score_history) > 0:
                x_axis = [i + 1 for i in range(len(score_history))]
                plot_figure_file = os.path.join(save_dir, f'learning_curve_ep{episode+1}.png')
                plot_learning_curve(x_axis, score_history, plot_figure_file)
            
            print(f'Episode {episode+1}: 已保存检查点和学习曲线。')
        
        # 打印训练信息
        if (episode + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f'Ep {episode+1}/{total_episodes}: 得分={score:.2f}, AvgScore(100)={avg_score:.2f}, '
                  f'WinRate(100)={current_win_rate*100:.1f}%, 敌机击毁={enemies_defeated_this_episode}/{env.num_enemies}, '
                  f'避障成功率={obstacle_avoidance_rate:.1f}%, 追踪成功率={tracking_success_rate:.1f}%, '
                  f'搜索步数={search_time_steps}, 围捕步数={capture_time_steps}, '
                  f'总步数={global_step_counter}, 耗时={elapsed_time:.1f}s,'
                  f'避障次数={obstacle_avoidance_success}, 追踪次数={tracking_success_count}')   
    
    # 训练结束，保存最终模型
    print(f'训练结束，保存最终模型到: {save_dir}')
    
    # 绘制最终学习曲线
    if len(score_history) > 0:
        x_axis = [i + 1 for i in range(len(score_history))]
        final_lc_path = os.path.join(save_dir, 'final_learning_curve.png')
        plot_learning_curve(x_axis, score_history, final_lc_path)
        print(f'最终学习曲线已保存到: {final_lc_path}')

    env.close()
    print('---- 训练完成 ----')
    print(f'总训练时间: {time.time() - start_time:.1f}s')
    final_avg_score = np.mean(score_history[-100:]) if len(score_history) >=100 else np.mean(score_history) if score_history else 0
    final_win_rate = np.mean(win_history[-100:]) if len(win_history) >=100 else np.mean(win_history) if win_history else 0
    print(f'最终100回合平均得分: {final_avg_score:.2f}')
    print(f'最终100回合胜率: {final_win_rate*100:.1f}%') 