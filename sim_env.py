import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy

class UAVEnv:
    """
    多无人机围捕环境模拟器
    
    实现了一个多智能体环境，其中包含多个围捕无人机和一个目标无人机，
    以及随机生成的障碍物。环境支持MADDPG算法的训练和评估。
    """
    def __init__(self,length=2,num_obstacle=3,num_agents=4):
        """
        初始化UAV环境
        
        参数:
            length: 环境边界长度
            num_obstacle: 障碍物数量
            num_agents: 智能体数量（包括目标无人机）
        """
        self.length = length # length of boundary
        self.num_obstacle = num_obstacle # number of obstacles
        self.num_agents = num_agents
        self.time_step = 0.5 # update time step
        self.v_max = 0.1  # 围捕无人机最大速度
        self.v_max_e = 0.12  # 目标无人机最大速度
        self.a_max = 0.04  # 围捕无人机最大加速度
        self.a_max_e = 0.05  # 目标无人机最大加速度
        self.L_sensor = 0.2  # 激光传感器最大探测距离
        self.num_lasers = 16 # num of laserbeams
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]
        self.agents = ['agent_0','agent_1','agent_2','target']
        self.info = np.random.get_state() # get seed
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]  # 初始化障碍物
        self.history_positions = [[] for _ in range(num_agents)]  # 记录无人机历史位置用于轨迹绘制

        # 定义动作空间，每个智能体的动作为二维加速度[a_x, a_y]
        self.action_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
            } # action represents [a_x,a_y]
        
        # 定义观察空间，围捕无人机和目标无人机的观察维度不同
        self.observation_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(23,))
        }
        

    def reset(self):
        """
        重置环境状态
        
        随机初始化所有无人机的位置和速度，更新激光传感器数据
        
        返回:
            multi_obs: 所有智能体的初始观察列表
        """
        SEED = random.randint(1,1000)
        random.seed(SEED)
        self.multi_current_pos = []  # 存储所有无人机当前位置
        self.multi_current_vel = []  # 存储所有无人机当前速度
        self.history_positions = [[] for _ in range(self.num_agents)]  # 重置历史位置记录
        
        # 初始化无人机位置和速度
        for i in range(self.num_agents):
            if i != self.num_agents - 1: # if not target
                # 围捕无人机在环境左下角随机初始化
                self.multi_current_pos.append(np.random.uniform(low=0.1,high=0.4,size=(2,)))
            else: # for target
                # 目标无人机在固定位置初始化
                # self.multi_current_pos.append(np.array([1.0,0.25]))
                self.multi_current_pos.append(np.array([0.5,1.75]))
            self.multi_current_vel.append(np.zeros(2)) # initial velocity = [0,0]

        # 更新激光传感器数据和碰撞检测
        self.update_lasers_isCollied_wrapper()
        ## multi_obs is list of agent_obs, state is multi_obs after flattenned
        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self,actions):
        """
        执行环境步进
        
        根据给定的动作更新所有无人机的状态，检测碰撞，计算奖励
        
        参数:
            actions: 所有智能体的动作列表，每个动作为[a_x, a_y]
            
        返回:
            multi_next_obs: 所有智能体的下一个观察列表
            rewards: 所有智能体的奖励列表
            dones: 所有智能体的终止标志列表
        """
        last_d2target = []  # 记录上一步各围捕无人机到目标的距离
        
        # 更新所有无人机的位置和速度
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:
                # 记录围捕无人机到目标的距离
                pos_taget = self.multi_current_pos[-1]
                last_d2target.append(np.linalg.norm(pos-pos_taget))
            
            # 更新速度（应用加速度）
            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            
            # 限制速度大小不超过最大值
            vel_magnitude = np.linalg.norm(self.multi_current_vel)
            if i != self.num_agents - 1:
                # 围捕无人机速度限制
                if vel_magnitude >= self.v_max:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            else:
                # 目标无人机速度限制
                if vel_magnitude >= self.v_max_e:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_e

            # 更新位置
            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        # 更新障碍物位置
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            # 检查边界碰撞并调整速度
            for dim in [0, 1]:
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1  # 碰到边界反弹
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1  # 碰到边界反弹

        # 更新激光传感器数据和检测碰撞
        Collided = self.update_lasers_isCollied_wrapper()
        # 计算奖励和终止标志
        rewards, dones= self.cal_rewards_dones(Collided,last_d2target)   
        # 获取下一个观察
        multi_next_obs = self.get_multi_obs()
        # sequence above can't be disrupted

        return multi_next_obs, rewards, dones

    def test_multi_obs(self):
        """
        测试获取简化版的多智能体观察
        
        仅包含位置和速度信息，用于调试
        
        返回:
            total_obs: 所有智能体的简化观察列表
        """
        total_obs = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ]
            total_obs.append(S_uavi)
        return total_obs
    
    def get_multi_obs(self):
        """
        获取所有智能体的完整观察
        
        围捕无人机的观察包含：
        1. 自身状态（位置和速度）
        2. 队友状态（位置）
        3. 激光传感器数据
        4. 目标信息（距离和角度）
        
        目标无人机的观察包含：
        1. 自身状态（位置和速度）
        2. 激光传感器数据
        3. 与围捕无人机的距离
        
        返回:
            total_obs: 所有智能体的观察列表
        """
        total_obs = []
        single_obs = []
        S_evade_d = [] # dim 3 only for target
        
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            
            # 1. 自身状态（归一化）
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ] # dim 4
            
            S_team = [] # dim 4 for 3 agents 1 target
            S_target = [] # dim 2
            
            # 2. 获取队友和目标信息
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1: 
                    # 队友位置（归一化）
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0]/self.length,pos_other[1]/self.length])
                elif j == self.num_agents - 1:
                    # 目标位置（距离和角度）
                    pos_target = self.multi_current_pos[j]
                    d = np.linalg.norm(pos - pos_target)
                    theta = np.arctan2(pos_target[1]-pos[1], pos_target[0]-pos[0])
                    S_target.extend([d/np.linalg.norm(2*self.length), theta])
                    if i != self.num_agents - 1:
                        # 记录围捕无人机到目标的距离（用于目标无人机的观察）
                        S_evade_d.append(d/np.linalg.norm(2*self.length))

            # 3. 激光传感器数据
            S_obser = self.multi_current_lasers[i] # dim 16

            # 4. 组合观察
            if i != self.num_agents - 1:
                # 围捕无人机的观察
                single_obs = [S_uavi,S_team,S_obser,S_target]
            else:
                # 目标无人机的观察
                single_obs = [S_uavi,S_obser,S_evade_d]
                
            # 5. 展平观察向量
            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)
            
        return total_obs

    def cal_rewards_dones(self,IsCollied,last_d):
        """
        计算奖励和终止标志
        
        奖励函数包含多个组成部分：
        1. 接近目标奖励（r_near）
        2. 安全奖励（r_safe）
        3. 多阶段围捕奖励（r_track, r_encircle, r_capture）
        4. 完成任务奖励
        
        参数:
            IsCollied: 碰撞检测结果列表
            last_d: 上一步各围捕无人机到目标的距离列表
            
        返回:
            rewards: 所有智能体的奖励列表
            dones: 所有智能体的终止标志列表
        """
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)
        
        # 奖励函数权重
        mu1 = 0.7 # r_near
        mu2 = 0.4 # r_safe
        mu3 = 0.01 # r_multi_stage
        mu4 = 5 # r_finish
        
        # 围捕相关参数
        d_capture = 0.3  # 捕获距离阈值
        d_limit = 0.75   # 包围距离阈值
        
        ## 1 reward for single rounding-up-UAVs:
        # 计算围捕无人机接近目标的奖励
        for i in range(3):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            pos_target = self.multi_current_pos[-1]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec) # distance to target

            # 计算速度向量与目标方向的夹角余弦值
            cos_v_d = np.dot(vel,dire_vec)/(v_i*d + 1e-3)
            # 奖励与速度大小和方向相关
            r_near = abs(2*v_i/self.v_max)*cos_v_d
            # r_near = min(abs(v_i/self.v_max)*1.0/(d + 1e-5),10)/5
            rewards[i] += mu1 * r_near # TODO: if not get nearer then receive negative reward
        
        ## 2 collision reward for all UAVs:
        # 计算所有无人机的安全奖励（避障）
        for i in range(self.num_agents):
            if IsCollied[i]:
                # 碰撞惩罚
                r_safe = -10
            else:
                # 根据最近障碍物距离计算安全奖励
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1)/self.L_sensor
            rewards[i] += mu2 * r_safe

        ## 3 multi-stage's reward for rounding-up-UAVs
        # 获取所有无人机位置
        p0 = self.multi_current_pos[0]
        p1 = self.multi_current_pos[1]
        p2 = self.multi_current_pos[2]
        pe = self.multi_current_pos[-1]
        
        # 计算三角形面积，用于判断围捕状态
        S1 = cal_triangle_S(p0,p1,pe)  # 三角形0-1-目标的面积
        S2 = cal_triangle_S(p1,p2,pe)  # 三角形1-2-目标的面积
        S3 = cal_triangle_S(p2,p0,pe)  # 三角形2-0-目标的面积
        S4 = cal_triangle_S(p0,p1,p2)  # 三角形0-1-2的面积（围捕无人机形成的三角形）
        
        # 计算各围捕无人机到目标的距离
        d1 = np.linalg.norm(p0-pe)
        d2 = np.linalg.norm(p1-pe)
        d3 = np.linalg.norm(p2-pe)
        Sum_S = S1 + S2 + S3  # 三个三角形面积之和
        Sum_d = d1 + d2 + d3  # 三个距离之和
        Sum_last_d = sum(last_d)  # 上一步的距离之和
        
        # 3.1 reward for target UAV:
        # 目标无人机的奖励：距离增加则获得正奖励
        rewards[-1] += np.clip(10 * (Sum_d - Sum_last_d),-2,2)
        
        # 3.2 stage-1 track
        # 阶段1：跟踪阶段，目标在围捕无人机形成的三角形外且距离较远
        if Sum_S > S4 and Sum_d >= d_limit and all(d >= d_capture for d in [d1, d2, d3]):
            r_track = - Sum_d/max([d1,d2,d3])
            rewards[0:2] += mu3*r_track
            
        # 3.3 stage-2 encircle
        # 阶段2：包围阶段，目标在三角形外但距离较近
        elif Sum_S > S4 and (Sum_d < d_limit or any(d >= d_capture for d in [d1, d2, d3])):
            r_encircle = -1/3*np.log(Sum_S - S4 + 1)
            rewards[0:2] += mu3*r_encircle
            
        # 3.4 stage-3 capture
        # 阶段3：捕获阶段，目标在三角形内但还未完全捕获
        elif Sum_S == S4 and any(d > d_capture for d in [d1,d2,d3]):
            r_capture = np.exp((Sum_last_d - Sum_d)/(3*self.v_max))
            rewards[0:2] += mu3*r_capture
        
        ## 4 finish rewards
        # 完成任务奖励：目标在三角形内且所有围捕无人机都足够接近目标
        if Sum_S == S4 and all(d <= d_capture for d in [d1,d2,d3]):
            rewards[0:2] += mu4*10
            dones = [True] * self.num_agents

        return rewards,dones

    def update_lasers_isCollied_wrapper(self):
        """
        更新所有无人机的激光传感器数据并检测碰撞
        
        对每个无人机，计算其与所有障碍物的激光传感器交点
        
        返回:
            dones: 碰撞检测结果列表，True表示发生碰撞
        """
        self.multi_current_lasers = []
        dones = []
        
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            
            # 检查与每个障碍物的交点
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos,obs_pos,r,self.L_sensor,self.num_lasers,self.length)
                # 取每个方向上的最小激光长度
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
                
            # 如果与任何障碍物碰撞，则停止无人机
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)
                
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
            
        return dones

    def render(self):
        """
        渲染当前环境状态
        
        绘制所有无人机、障碍物和轨迹
        
        返回:
            image: 渲染的RGBA图像数组
        """
        plt.clf()
        
        # 加载无人机图标
        uav_icon = mpimg.imread('UAV.png')
        # icon_height, icon_width, _ = uav_icon.shape

        # 绘制围捕无人机
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            
            # 绘制轨迹
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)
            
            # 计算速度向量的角度
            angle = np.arctan2(vel[1], vel[0])

            # 应用旋转变换并绘制无人机图标
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1  # 调整图标大小
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

            # # 可视化激光射线（需要时可以取消注释）
            # lasers = self.multi_current_lasers[i]
            # angles = np.linspace(0, 2 * np.pi, len(lasers), endpoint=False)
            
            # for angle, laser_length in zip(angles, lasers):
            #     laser_end = np.array(pos) + np.array([laser_length * np.cos(angle), laser_length * np.sin(angle)])
            #     plt.plot([pos[0], laser_end[0]], [pos[1], laser_end[1]], 'b-', alpha=0.2)

        # 绘制目标无人机
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        self.history_positions[-1].append(copy.deepcopy(self.multi_current_pos[-1]))
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        # 绘制障碍物
        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
            
        # 设置图表范围
        plt.xlim(-0.1, self.length+0.1)
        plt.ylim(-0.1, self.length+0.1)
        plt.draw()
        plt.legend()
        
        # 将当前图形保存到缓冲区
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()
        
        # 将缓冲区转换为NumPy数组
        image = np.asarray(buf)
        return image

    def render_anime(self, frame_num):
        """
        渲染用于动画的环境状态
        
        与render方法类似，但使用彩色渐变轨迹，适用于创建动画
        
        参数:
            frame_num: 当前帧号
        """
        plt.clf()
        
        # 加载无人机图标
        uav_icon = mpimg.imread('UAV.png')

        # 绘制围捕无人机
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])
            self.history_positions[i].append(pos)
            
            # 使用渐变色绘制轨迹
            trajectory = np.array(self.history_positions[i])
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))  # 使用 viridis colormap
                plt.plot(trajectory[j:j+2, 0], trajectory[j:j+2, 1], color=color, alpha=0.7)

            # 应用旋转变换并绘制无人机图标
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

        # 绘制目标无人机
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        pos_e = copy.deepcopy(self.multi_current_pos[-1])
        self.history_positions[-1].append(pos_e)
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        # 绘制障碍物
        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        # 设置图表范围
        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()

    def close(self):
        """
        关闭环境
        
        关闭matplotlib图形窗口
        """
        plt.close()

class obstacle():
    """
    障碍物类
    
    表示环境中的圆形障碍物，具有位置、速度和半径属性
    """
    def __init__(self, length=2):
        """
        初始化障碍物
        
        随机生成障碍物的位置、速度和半径
        
        参数:
            length: 环境边界长度
        """
        # 随机生成障碍物位置（避开边缘区域）
        self.position = np.random.uniform(low=0.45, high=length-0.55, size=(2,))
        
        # 随机生成障碍物速度方向和大小
        angle = np.random.uniform(0, 2 * np.pi)
        # speed = 0.03 
        speed = 0.00 # to make obstacle fixed
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        
        # 随机生成障碍物半径
        self.radius = np.random.uniform(0.1, 0.15)