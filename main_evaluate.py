from maddpg import MADDPG
from sim_env import UAVEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
warnings.filterwarnings('ignore')

def moving_average(data, window_size=5):
    """
    计算数据的移动平均值
    
    参数:
        data: 输入数据数组
        window_size: 移动窗口大小
        
    返回:
        平滑后的数据数组
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_velocity_magnitude(time_steps, velocities_magnitude):
    """
    绘制无人机速度大小随时间的变化图
    
    参数:
        time_steps: 时间步数组
        velocities_magnitude: 各无人机速度大小数组
    """
    plt.figure(figsize=(15, 4))  
    for i in range(len(velocities_magnitude)):
        if i!=3:
            plt.plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_magnitude[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("Magnitude")
    plt.title("UAV Velocity Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_velocity_x(time_steps, velocities_x):
    """
    绘制无人机X方向速度随时间的变化图
    
    参数:
        time_steps: 时间步数组
        velocities_x: 各无人机X方向速度数组
    """
    plt.figure(figsize=(15, 4))  
    for i in range(len(velocities_x)):
        if i!=3:
            plt.plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_x[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_x$")
    plt.title("UAV $Vel_x$")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_velocity_y(time_steps, velocities_y):
    """
    绘制无人机Y方向速度随时间的变化图
    
    参数:
        time_steps: 时间步数组
        velocities_y: 各无人机Y方向速度数组
    """
    plt.figure(figsize=(15, 4))  
    for i in range(len(velocities_y)):
        if i!=3:
            plt.plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_y[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_y$")
    plt.title("UAV $Vel_y$")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_velocities(velocities_magnitude, velocities_x, velocities_y):
    """
    在一个图表中绘制无人机速度大小、X和Y方向速度随时间的变化
    
    参数:
        velocities_magnitude: 各无人机速度大小数组
        velocities_x: 各无人机X方向速度数组
        velocities_y: 各无人机Y方向速度数组
    """
    time_steps = range(len(velocities_magnitude[0]))
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # 绘制速度大小子图
    for i in range(len(velocities_magnitude)):
        if i != 3:
            axs[0].plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            axs[0].plot(time_steps, velocities_magnitude[i], label=f'Target')
    axs[0].set_title('Speed Magnitude vs Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Speed Magnitude')
    axs[0].legend()

    # 绘制X方向速度子图
    for i in range(len(velocities_x)):
        if i != 3:
            axs[1].plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            axs[1].plot(time_steps, velocities_x[i], label=f'Target')
    axs[1].set_title('Velocity X Component vs Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Velocity X Component')
    axs[1].legend()

    # 绘制Y方向速度子图
    for i in range(len(velocities_y)):
        if i != 3:
            axs[2].plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            axs[2].plot(time_steps, velocities_y[i], label=f'Target')
    axs[2].set_title('Velocity Y Component vs Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Velocity Y Component')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 初始化UAV环境
    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 2
    
    # 初始化速度记录数组
    actor_dims = []
    velocities_magnitude = [[] for _ in range(env.num_agents)]  # 记录速度大小
    velocities_x = [[] for _ in range(env.num_agents)]  # 记录X方向速度
    velocities_y = [[] for _ in range(env.num_agents)]  # 记录Y方向速度

    # 获取每个智能体的观察空间维度
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    
    # 初始化MADDPG算法
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=128, fc2=128, alpha=0.0001, beta=0.003, scenario='UAV_Round_up',
                           chkpt_dir='tmp/maddpg/')
    
    # 加载预训练模型
    maddpg_agents.load_checkpoint()
    print('---- Evaluating ----')

    # 重置环境
    obs = env.reset()

    def update(frame):
        """
        动画更新函数，每一帧调用一次
        
        参数:
            frame: 当前帧索引
            
        返回:
            空列表，matplotlib动画需要
        """
        global obs, velocities_magnitude, velocities_x, velocities_y

        # 记录当前速度数据
        for i in range(env.num_agents):
            vel = env.multi_current_vel[i]
            v_x, v_y = vel
            speed = np.linalg.norm(vel)

            velocities_magnitude[i].append(speed)
            velocities_x[i].append(v_x)
            velocities_y[i].append(v_y)

        # 选择动作并执行
        actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
        obs_, _, dones = env.step(actions)
        
        # 渲染当前帧
        env.render_anime(frame)
        
        # 更新观察
        obs = obs_
        
        # 如果任何智能体完成任务，停止动画并打印结果
        if any(dones):
            ani.event_source.stop()
            print("Round-up finished in", frame, "steps.")
            # 以下代码被注释掉，用于绘制平滑后的速度图表
            # smoothed_velocities_magnitude = [[] for _ in range(env.num_agents)]
            # smoothed_velocities_x = [[] for _ in range(env.num_agents)]  
            # smoothed_velocities_y = [[] for _ in range(env.num_agents)] 
            # for i in range(env.num_agents):
            #     _velocity_magnitude = moving_average(velocities_magnitude[i],window_size=5)
            #     _velocity_x = moving_average(velocities_x[i],window_size=5)
            #     _velocity_y = moving_average(velocities_y[i],window_size=5)
            #     smoothed_velocities_magnitude[i]=_velocity_magnitude
            #     smoothed_velocities_x[i]=_velocity_x
            #     smoothed_velocities_y[i]=_velocity_y
            # # plot_velocities(smoothed_velocities_magnitude,smoothed_velocities_x,smoothed_velocities_y)
            # time_steps = range(len(smoothed_velocities_magnitude[0]))
            # plot_velocity_magnitude(time_steps,smoothed_velocities_magnitude)
            # plot_velocity_x(time_steps, smoothed_velocities_x)
            # plot_velocity_y(time_steps, smoothed_velocities_y)
        return []

    # 初始化总步数
    total_steps = 0

    # 创建动画
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, update, frames=10000, interval=20)
    plt.show()







    ######参数，移植落地