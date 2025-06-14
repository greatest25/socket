from uav_env_core.environment import Qt5SimUAVEnv
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt

def visualize_environment_with_detection(env, filename):
    """可视化环境和UAV的探测范围"""
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置图形范围
    ax.set_xlim(0, env.map_width)
    ax.set_ylim(0, env.map_height)
    
    # 绘制UAV和敌机
    for i in range(env.num_agents):
        if env.uav_health[i] > 0:  # 只显示存活UAV
            uav_pos = env.multi_current_pos[i]
            # 绘制UAV
            ax.plot(uav_pos[0], uav_pos[1], 'bo', markersize=10)
            # 绘制探测范围
            detection_circle = plt.Circle(
                (uav_pos[0], uav_pos[1]), 
                env.detection_radius,
                color='blue' if not env.tracking_mode[i] else 'green',  # 追踪模式下显示为绿色
                fill=False, 
                alpha=0.3,
                linestyle='--'
            )
            ax.add_artist(detection_circle)
            # 如果在追踪模式，显示攻击范围
            if env.tracking_mode[i]:
                attack_circle = plt.Circle(
                    (uav_pos[0], uav_pos[1]),
                    env.attack_radius,
                    color='red',
                    fill=False,
                    alpha=0.2,
                    linestyle=':'
                )
                ax.add_artist(attack_circle)
    
    # 绘制敌机
    for i in range(env.num_enemies):
        if env.enemy_health[i] > 0:  # 只显示存活敌机
            enemy_pos = env.enemy_pos[i]
            ax.plot(enemy_pos[0], enemy_pos[1], 'ro', markersize=10)
    
    # 绘制障碍物
    for obstacle in env.obstacles:
        obstacle_circle = plt.Circle(
            (obstacle['position'][0], obstacle['position'][1]),
            obstacle['radius'],
            color='gray',
            alpha=0.5
        )
        ax.add_artist(obstacle_circle)
    
    # 保存图形
    plt.title('环境状态可视化')
    plt.savefig(filename)
    plt.close()

def get_detection_info(env):
    """获取每个UAV的探测信息"""
    detection_info = []
    for i in range(env.num_agents):
        if env.uav_health[i] <= 0:  # 如果UAV已损毁
            detection_info.append({'detected_enemies': []})
            continue
            
        detected_enemies = []
        for j in range(env.num_enemies):
            if env.enemy_health[j] > 0:  # 只考虑存活的敌机
                dist = np.linalg.norm(env.multi_current_pos[i] - env.enemy_pos[j])
                if dist <= env.detection_radius:  # 在探测范围内
                    detected_enemies.append(env.enemy_pos[j].tolist())
        
        detection_info.append({'detected_enemies': detected_enemies})
    
    return detection_info

def get_uav_position(env, uav_idx):
    """获取指定UAV的位置"""
    return env.multi_current_pos[uav_idx]

def test_environment():
    # 创建评估结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"evaluation_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # 创建环境实例 (使用默认参数)
    env = Qt5SimUAVEnv(
        map_width=1280,
        map_height=800,
        num_agents=3,
        num_enemies=3,
        num_static_obstacles=2,
        num_dynamic_obstacles=1
    )
    
    # 添加辅助方法到环境实例
    env.get_detection_info = lambda: get_detection_info(env)
    env.get_uav_position = lambda i: get_uav_position(env, i)

    # 评估指标
    total_episodes = 5  # 评估回合数
    metrics = {
        'episode_rewards': [],
        'episode_steps': [],
        'enemies_destroyed': [],
        'agents_lost': [],
        'detection_times': [],  # 首次探测到敌人的时间
        'capture_times': [],    # 从探测到击毁的时间
        'success_rate': 0.0     # 成功率（所有敌机被摧毁）
    }

    for episode in range(total_episodes):
        print(f"\n=== 评估回合 {episode + 1}/{total_episodes} ===")
        obs = env.reset()
        
        # 保存初始状态可视化
        visualize_environment_with_detection(env, f"{results_dir}/ep{episode+1}_step_0.png")

        episode_reward = np.zeros(env.num_agents)
        step_count = 0
        first_detection_step = None
        enemies_alive_count = env.num_enemies
        agents_alive_count = env.num_agents

        for step in range(300):  # 每回合最多300步
            # 获取每个UAV的观察和检测信息
            detected_info = env.get_detection_info()
            
            # 基于检测信息的简单策略（模拟训练好的策略）
            actions = []
            for i in range(env.num_agents):
                if env.uav_health[i] <= 0:  # 如果UAV已损毁
                    actions.append(np.zeros(2))
                    continue

                if len(detected_info[i]['detected_enemies']) > 0:
                    # 记录首次探测时间
                    if first_detection_step is None:
                        first_detection_step = step
                    # 如果检测到敌人，移动向最近的敌人
                    target = detected_info[i]['detected_enemies'][0]
                    direction = np.array(target) - np.array(env.get_uav_position(i))
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        action = direction / norm
                    else:
                        action = np.zeros(2)
                else:
                    # 未检测到敌人时的搜索策略
                    action = env.action_space[f'agent_{i}'].sample()
                actions.append(action)

            # 执行动作
            obs, rewards, dones = env.step(actions)
            episode_reward += rewards
            step_count += 1

            # 每10步保存一次可视化结果
            if (step + 1) % 10 == 0:
                visualize_environment_with_detection(env, f"{results_dir}/ep{episode+1}_step_{step+1}.png")

            # 统计存活数量
            current_enemies_alive = sum(1 for health in env.enemy_health if health > 0)
            current_agents_alive = sum(1 for health in env.uav_health if health > 0)

            # 打印状态信息
            print(f"\nStep {step + 1}:")
            print(f"UAV Health: {env.uav_health}")
            print(f"Enemy Health: {env.enemy_health}")
            print(f"Detected enemies: {[info['detected_enemies'] for info in detected_info]}")
            print(f"Rewards: {rewards}")

            # 检查是否所有敌机都被摧毁或所有我方UAV都被摧毁
            if current_enemies_alive == 0 or current_agents_alive == 0 or all(dones):
                enemies_destroyed = enemies_alive_count - current_enemies_alive
                agents_lost = agents_alive_count - current_agents_alive
                
                # 更新指标
                metrics['episode_rewards'].append(episode_reward)
                metrics['episode_steps'].append(step_count)
                metrics['enemies_destroyed'].append(enemies_destroyed)
                metrics['agents_lost'].append(agents_lost)
                if first_detection_step is not None:
                    metrics['detection_times'].append(first_detection_step)
                    if current_enemies_alive == 0:
                        metrics['capture_times'].append(step - first_detection_step)
                
                print(f"\n回合 {episode + 1} 结束于步骤 {step + 1}")
                print(f"摧毁敌机数: {enemies_destroyed}")
                print(f"损失我方UAV数: {agents_lost}")
                print(f"总奖励: {episode_reward}")
                break

        if step == 299:  # 如果达到最大步数
            enemies_destroyed = enemies_alive_count - current_enemies_alive
            agents_lost = agents_alive_count - current_agents_alive
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_steps'].append(step_count)
            metrics['enemies_destroyed'].append(enemies_destroyed)
            metrics['agents_lost'].append(agents_lost)

    # 计算并打印评估结果
    metrics['success_rate'] = sum(1 for x in metrics['enemies_destroyed'] if x == env.num_enemies) / total_episodes

    print("\n=== 评估结果汇总 ===")
    print(f"总回合数: {total_episodes}")
    print(f"平均回合步数: {np.mean(metrics['episode_steps']):.2f}")
    print(f"平均摧毁敌机数: {np.mean(metrics['enemies_destroyed']):.2f}")
    print(f"平均损失我方UAV数: {np.mean(metrics['agents_lost']):.2f}")
    print(f"平均回合奖励: {np.mean([sum(r) for r in metrics['episode_rewards']]):.2f}")
    print(f"任务成功率: {metrics['success_rate']*100:.2f}%")
    if metrics['detection_times']:
        print(f"平均首次探测时间: {np.mean(metrics['detection_times']):.2f} 步")
    if metrics['capture_times']:
        print(f"平均围捕时间: {np.mean(metrics['capture_times']):.2f} 步")

    # 保存评估结果
    with open(f"{results_dir}/evaluation_results.txt", 'w') as f:
        f.write("=== 评估结果汇总 ===\n")
        f.write(f"总回合数: {total_episodes}\n")
        f.write(f"平均回合步数: {np.mean(metrics['episode_steps']):.2f}\n")
        f.write(f"平均摧毁敌机数: {np.mean(metrics['enemies_destroyed']):.2f}\n")
        f.write(f"平均损失我方UAV数: {np.mean(metrics['agents_lost']):.2f}\n")
        f.write(f"平均回合奖励: {np.mean([sum(r) for r in metrics['episode_rewards']]):.2f}\n")
        f.write(f"任务成功率: {metrics['success_rate']*100:.2f}%\n")
        if metrics['detection_times']:
            f.write(f"平均首次探测时间: {np.mean(metrics['detection_times']):.2f} 步\n")
        if metrics['capture_times']:
            f.write(f"平均围捕时间: {np.mean(metrics['capture_times']):.2f} 步\n")

    env.close()
    print(f"\n评估完成。结果保存在 {results_dir} 目录下。")

if __name__ == "__main__":
    test_environment()