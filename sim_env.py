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
    def __init__(self,length=2,num_obstacle=3,num_agents=4):
        self.length = length # length of boundary
        self.num_obstacle = num_obstacle # number of obstacles
        self.num_agents = num_agents
        self.time_step = 0.5 # update time step
        self.v_max = 0.1
        self.v_max_e = 0.12
        self.a_max = 0.04
        self.a_max_e = 0.05
        self.L_sensor = 0.2
        self.num_lasers = 16 # num of laserbeams
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]
        self.agents = ['agent_0','agent_1','agent_2','target']
        self.info = np.random.get_state() # get seed
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]
        self.history_positions = [[] for _ in range(num_agents)]

        self.action_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
            } # action represents [a_x,a_y]
        self.observation_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(23,))
        }
        

    def reset(self):
        SEED = random.randint(1,1000)
        random.seed(SEED)
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            if i != self.num_agents - 1: # if not target
                self.multi_current_pos.append(np.random.uniform(low=0.1,high=0.4,size=(2,)))
            else: # for target
                # self.multi_current_pos.append(np.array([1.0,0.25]))
                self.multi_current_pos.append(np.array([0.5,1.75]))
            self.multi_current_vel.append(np.zeros(2)) # initial velocity = [0,0]

        # update lasers
        self.update_lasers_isCollied_wrapper()
        ## multi_obs is list of agent_obs, state is multi_obs after flattenned
        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self,actions):
        last_d2target = []
        # print(actions)
        # time.sleep(0.1)
        for i in range(self.num_agents):

            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:
                pos_taget = self.multi_current_pos[-1]
                last_d2target.append(np.linalg.norm(pos-pos_taget))
            
            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            vel_magnitude = np.linalg.norm(self.multi_current_vel)
            if i != self.num_agents - 1:
                if vel_magnitude >= self.v_max:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            else:
                if vel_magnitude >= self.v_max_e:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_e

            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        # Update obstacle positions
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            # Check for boundary collisions and adjust velocities
            for dim in [0, 1]:
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1

        Collided = self.update_lasers_isCollied_wrapper()
        rewards, dones= self.cal_rewards_dones(Collided,last_d2target)   
        multi_next_obs = self.get_multi_obs()
        # sequence above can't be disrupted

        return multi_next_obs, rewards, dones

    def test_multi_obs(self):
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
        total_obs = []
        single_obs = []
        S_evade_d = [] # dim 3 only for target
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ] # dim 4
            S_team = [] # dim 4 for 3 agents 1 target
            S_target = [] # dim 2
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1: 
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0]/self.length,pos_other[1]/self.length])
                elif j == self.num_agents - 1:
                    pos_target = self.multi_current_pos[j]
                    d = np.linalg.norm(pos - pos_target)
                    theta = np.arctan2(pos_target[1]-pos[1], pos_target[0]-pos[0])
                    S_target.extend([d/np.linalg.norm(2*self.length), theta])
                    if i != self.num_agents - 1:
                        S_evade_d.append(d/np.linalg.norm(2*self.length))

            S_obser = self.multi_current_lasers[i] # dim 16

            if i != self.num_agents - 1:
                single_obs = [S_uavi,S_team,S_obser,S_target]
            else:
                single_obs = [S_uavi,S_obser,S_evade_d]
            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)
            
        return total_obs

    def cal_rewards_dones(self,IsCollied,last_d):
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)
        mu1 = 0.7 # r_near
        mu2 = 0.4 # r_safe
        mu3 = 0.01 # r_multi_stage
        mu4 = 5 # r_finish
        d_capture = 0.3
        d_limit = 0.75
        
        # 1 reward for single rounding-up-UAVs:
        for i in range(3):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            pos_target = self.multi_current_pos[-1]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec) # distance to target

            cos_v_d = np.dot(vel,dire_vec)/(v_i*d + 1e-3)
            r_near = abs(2*v_i/self.v_max)*cos_v_d
            rewards[i] += mu1 * r_near # TODO: if not get nearer then receive negative reward
        
        # 2 collision reward for all UAVs:
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1)/self.L_sensor
            rewards[i] += mu2 * r_safe

        # 3 multi-stage's reward for rounding-up-UAVs
        p0 = self.multi_current_pos[0]
        p1 = self.multi_current_pos[1]
        p2 = self.multi_current_pos[2]
        pe = self.multi_current_pos[-1]
        S1 = cal_triangle_S(p0,p1,pe)
        S2 = cal_triangle_S(p1,p2,pe)
        S3 = cal_triangle_S(p2,p0,pe)
        S4 = cal_triangle_S(p0,p1,p2)
        d1 = np.linalg.norm(p0-pe)
        d2 = np.linalg.norm(p1-pe)
        d3 = np.linalg.norm(p2-pe)
        Sum_S = S1 + S2 + S3
        Sum_d = d1 + d2 + d3
        Sum_last_d = sum(last_d)
        # 3.1 reward for target UAV:
        rewards[-1] += np.clip(10 * (Sum_d - Sum_last_d),-2,2)
        # print(rewards[-1])
        # 3.2 stage-1 track
        if Sum_S > S4 and Sum_d >= d_limit and all(d >= d_capture for d in [d1, d2, d3]):
            r_track = - Sum_d/max([d1,d2,d3])
            rewards[0:2] += mu3*r_track
        # 3.3 stage-2 encircle
        elif Sum_S > S4 and (Sum_d < d_limit or any(d >= d_capture for d in [d1, d2, d3])):
            r_encircle = -1/3*np.log(Sum_S - S4 + 1)
            rewards[0:2] += mu3*r_encircle
        # 3.4 stage-3 capture
        elif Sum_S == S4 and any(d > d_capture for d in [d1,d2,d3]):
            r_capture = np.exp((Sum_last_d - Sum_d)/(3*self.v_max))
            rewards[0:2] += mu3*r_capture
        
        # 4 finish rewards
        if Sum_S == S4 and all(d <= d_capture for d in [d1,d2,d3]):
            rewards[0:2] += mu4*10
            dones = [True] * self.num_agents

        return rewards,dones

    def update_lasers_isCollied_wrapper(self):
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos,obs_pos,r,self.L_sensor,self.num_lasers,self.length)
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
        return dones

    def render(self):

        plt.clf()
        
        # load UAV icon
        uav_icon = mpimg.imread('UAV.png')
        # icon_height, icon_width, _ = uav_icon.shape

        # plot round-up-UAVs
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            # plot trajectory
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)
            # Calculate the angle of the velocity vector
            angle = np.arctan2(vel[1], vel[0])

            # plt.scatter(pos[0], pos[1], c='b', label='hunter')
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            # plt.imshow(uav_icon, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            # plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            icon_size = 0.1  # Adjust this size to your icon's aspect ratio
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

            # # Visualize laser rays for each UAV(can be closed when unneeded)
            # lasers = self.multi_current_lasers[i]
            # angles = np.linspace(0, 2 * np.pi, len(lasers), endpoint=False)
            
            # for angle, laser_length in zip(angles, lasers):
            #     laser_end = np.array(pos) + np.array([laser_length * np.cos(angle), laser_length * np.sin(angle)])
            #     plt.plot([pos[0], laser_end[0]], [pos[1], laser_end[1]], 'b-', alpha=0.2)

        # plot target
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        self.history_positions[-1].append(copy.deepcopy(self.multi_current_pos[-1]))
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length+0.1)
        plt.ylim(-0.1, self.length+0.1)
        plt.draw()
        plt.legend()
        # plt.pause(0.01)
        # Save the current figure to a buffer
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()
        
        # Convert buffer to a NumPy array
        image = np.asarray(buf)
        return image

    def render_anime(self, frame_num):
        plt.clf()
        
        uav_icon = mpimg.imread('UAV.png')

        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])
            self.history_positions[i].append(pos)
            
            trajectory = np.array(self.history_positions[i])
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))  # 使用 viridis colormap
                plt.plot(trajectory[j:j+2, 0], trajectory[j:j+2, 1], color=color, alpha=0.7)
            # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=1)

            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        pos_e = copy.deepcopy(self.multi_current_pos[-1])
        self.history_positions[-1].append(pos_e)
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()

    def close(self):
        plt.close()

class obstacle():
    def __init__(self, length=2):
        self.position = np.random.uniform(low=0.45, high=length-0.55, size=(2,))
        angle = np.random.uniform(0, 2 * np.pi)
        # speed = 0.03 
        speed = 0.00 # to make obstacle fixed
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        self.radius = np.random.uniform(0.1, 0.15)

class Qt5SimUAVEnv(UAVEnv):
    def __init__(self, length=2, num_obstacle=3, num_agents=4, num_enemies=3, detection_radius=0.4, attack_radius=0.2):
        super().__init__(length, num_obstacle, num_agents)
        
        # 探测和攻击参数
        self.detection_radius = detection_radius  # 探测半径
        self.attack_radius = attack_radius  # 攻击半径
        
        # 敌机参数
        self.num_enemies = num_enemies
        self.enemy_positions = []
        self.enemy_velocities = []
        self.enemy_health = [100] * num_enemies
        self.enemy_detected = [False] * num_enemies
        
        # 我方无人机参数
        self.uav_health = [100] * (self.num_agents - 1)  # 最后一个是目标，不计算健康值
        
        # 搜索状态
        self.search_mode = True  # 初始为搜索模式
        
        # 信息共享缓存
        self.shared_detections = {
            'enemies': [],
            'obstacles': []
        }
        
    def reset(self):
        # 调用父类的reset方法
        obs = super().reset()
        
        # 初始化敌机位置
        self.enemy_positions = []
        self.enemy_velocities = []
        for _ in range(self.num_enemies):
            # 随机位置，但确保与我方无人机有一定距离
            while True:
                pos = np.random.uniform(low=0.5, high=self.length-0.5, size=(2,))
                # 检查与我方无人机的距离
                min_dist = min([np.linalg.norm(pos - p) for p in self.multi_current_pos[:-1]])
                if min_dist > 0.8:  # 确保初始距离足够远
                    break
            
            self.enemy_positions.append(pos)
            # 随机初始速度
            angle = np.random.uniform(0, 2 * np.pi)
            speed = 0.05  # 敌机速度
            self.enemy_velocities.append(np.array([speed * np.cos(angle), speed * np.sin(angle)]))
        
        # 重置健康值
        self.enemy_health = [100] * self.num_enemies
        self.uav_health = [100] * (self.num_agents - 1)
        
        # 重置探测状态
        self.enemy_detected = [False] * self.num_enemies
        
        # 重置搜索状态
        self.search_mode = True
        
        # 重置共享信息
        self.shared_detections = {
            'enemies': [],
            'obstacles': []
        }
        
        # 生成限制信息的观察
        limited_obs = self._get_limited_obs()
        
        return limited_obs
        
    def step(self, actions):
        # 调用父类的step方法更新我方无人机
        _, rewards, dones = super().step(actions)
        
        # 更新敌机位置
        for i in range(self.num_enemies):
            # 更新位置
            self.enemy_positions[i] += self.enemy_velocities[i] * self.time_step
            
            # 边界检查
            for dim in [0, 1]:
                if self.enemy_positions[i][dim] < 0:
                    self.enemy_positions[i][dim] = 0
                    self.enemy_velocities[i][dim] *= -1
                elif self.enemy_positions[i][dim] > self.length:
                    self.enemy_positions[i][dim] = self.length
                    self.enemy_velocities[i][dim] *= -1
            
            # 随机改变方向
            if np.random.random() < 0.02:  # 2%概率改变方向
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.linalg.norm(self.enemy_velocities[i])
                self.enemy_velocities[i] = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        
        # 检测敌机并更新探测状态
        self._update_detection()
        
        # 处理攻击
        self._process_attacks()
        
        # 检查敌机是否全部被消灭
        if all(health <= 0 for health in self.enemy_health):
            dones = [True] * self.num_agents
            # 给予额外奖励
            for i in range(self.num_agents - 1):
                rewards[i] += 20.0
        
        # 检查我方无人机是否全部被消灭
        if all(health <= 0 for health in self.uav_health):
            dones = [True] * self.num_agents
            # 给予惩罚
            for i in range(self.num_agents - 1):
                rewards[i] -= 20.0
        
        # 生成限制信息的观察
        limited_obs = self._get_limited_obs()
        
        return limited_obs, rewards, dones
    
    def _update_detection(self):
        # 清除上一步的探测信息
        self.enemy_detected = [False] * self.num_enemies
        self.shared_detections['enemies'] = []
        self.shared_detections['obstacles'] = []
        
        # 检查每个我方无人机的探测
        for i in range(self.num_agents - 1):  # 不包括目标
            uav_pos = self.multi_current_pos[i]
            
            # 探测敌机
            for j in range(self.num_enemies):
                if self.enemy_health[j] <= 0:
                    continue  # 跳过已消灭的敌机
                    
                enemy_pos = self.enemy_positions[j]
                distance = np.linalg.norm(uav_pos - enemy_pos)
                
                if distance <= self.detection_radius:
                    self.enemy_detected[j] = True
                    # 添加到共享信息
                    self.shared_detections['enemies'].append({
                        'id': j,
                        'position': enemy_pos,
                        'velocity': self.enemy_velocities[j]
                    })
            
            # 探测障碍物
            for j, obs in enumerate(self.obstacles):
                obs_pos = obs.position
                distance = np.linalg.norm(uav_pos - obs_pos)
                
                if distance <= self.detection_radius + obs.radius:
                    # 添加到共享信息
                    self.shared_detections['obstacles'].append({
                        'id': j,
                        'position': obs_pos,
                        'radius': obs.radius
                    })
        
        # 如果有敌机被探测到，切换到围捕模式
        if any(self.enemy_detected):
            self.search_mode = False
    
    def _process_attacks(self):
        # 处理我方无人机攻击敌机
        for i in range(self.num_agents - 1):
            if self.uav_health[i] <= 0:
                continue  # 跳过已损毁的我方无人机
                
            uav_pos = self.multi_current_pos[i]
            
            # 检查是否有敌机在攻击范围内
            for j in range(self.num_enemies):
                if self.enemy_health[j] <= 0:
                    continue  # 跳过已消灭的敌机
                    
                enemy_pos = self.enemy_positions[j]
                distance = np.linalg.norm(uav_pos - enemy_pos)
                
                if distance <= self.attack_radius:
                    # 对敌机造成伤害
                    damage = np.random.uniform(5, 10)
                    self.enemy_health[j] -= damage
                    # 限制最小值为0
                    self.enemy_health[j] = max(0, self.enemy_health[j])
        
        # 处理敌机攻击我方无人机
        for i in range(self.num_enemies):
            if self.enemy_health[i] <= 0:
                continue  # 跳过已消灭的敌机
                
            enemy_pos = self.enemy_positions[i]
            
            # 检查是否有我方无人机在攻击范围内
            for j in range(self.num_agents - 1):
                if self.uav_health[j] <= 0:
                    continue  # 跳过已损毁的我方无人机
                    
                uav_pos = self.multi_current_pos[j]
                distance = np.linalg.norm(enemy_pos - uav_pos)
                
                if distance <= self.attack_radius:
                    # 对我方无人机造成伤害
                    damage = np.random.uniform(3, 8)
                    self.uav_health[j] -= damage
                    # 限制最小值为0
                    self.uav_health[j] = max(0, self.uav_health[j])
    
    def _get_limited_obs(self):
        """生成限制信息的观察向量"""
        limited_obs = []
        
        for i in range(self.num_agents):
            if i == self.num_agents - 1:
                # 目标使用原始观察
                obs = self._get_original_obs(i)
                limited_obs.append(obs)
                continue
                
            # 我方无人机
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            
            # 1. 自身信息
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max,
                self.uav_health[i]/100.0  # 添加健康值
            ]
            
            # 2. 队友信息 (始终可见)
            S_team = []
            for j in range(self.num_agents - 1):
                if j != i:
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([
                        pos_other[0]/self.length,
                        pos_other[1]/self.length,
                        self.uav_health[j]/100.0
                    ])
            
            # 3. 敌机信息 (仅包含已探测到的)
            S_enemies = []
            detected_count = 0
            
            for j in range(self.num_enemies):
                if self.enemy_detected[j] and self.enemy_health[j] > 0:
                    enemy_pos = self.enemy_positions[j]
                    enemy_vel = self.enemy_velocities[j]
                    
                    # 相对位置、速度和健康值
                    S_enemies.extend([
                        (enemy_pos[0] - pos[0])/self.length,
                        (enemy_pos[1] - pos[1])/self.length,
                        enemy_vel[0]/self.v_max_e,
                        enemy_vel[1]/self.v_max_e,
                        self.enemy_health[j]/100.0
                    ])
                    detected_count += 1
                    
                    # 最多包含3个敌机信息
                    if detected_count >= 3:
                        break
            
            # 填充未探测到的敌机信息
            while detected_count < 3:
                S_enemies.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                detected_count += 1
            
            # 4. 搜索目标 (如果在搜索模式)
            S_search = []
            if self.search_mode:
                # 生成一个搜索目标点
                search_target = self._get_search_target(i)
                S_search.extend([
                    (search_target[0] - pos[0])/self.length,
                    (search_target[1] - pos[1])/self.length
                ])
            else:
                # 如果不在搜索模式，使用0填充
                S_search.extend([0.0, 0.0])
            
            # 5. 障碍物信息 (使用激光传感器数据)
            S_obser = self.multi_current_lasers[i]
            
            # 组合所有信息
            single_obs = list(itertools.chain(S_uavi, S_team, S_enemies, S_search, S_obser))
            limited_obs.append(single_obs)
        
        return limited_obs
    
    def _get_original_obs(self, agent_idx):
        """获取原始观察向量 (用于目标)"""
        pos = self.multi_current_pos[agent_idx]
        vel = self.multi_current_vel[agent_idx]
        
        S_uavi = [
            pos[0]/self.length,
            pos[1]/self.length,
            vel[0]/self.v_max,
            vel[1]/self.v_max
        ]
        
        S_team = []
        S_target = []
        S_evade_d = []
        
        for j in range(self.num_agents):
            if j != agent_idx and j != self.num_agents - 1:
                pos_other = self.multi_current_pos[j]
                S_team.extend([pos_other[0]/self.length, pos_other[1]/self.length])
            elif j == self.num_agents - 1 and agent_idx != self.num_agents - 1:
                pos_target = self.multi_current_pos[j]
                d = np.linalg.norm(pos - pos_target)
                theta = np.arctan2(pos_target[1]-pos[1], pos_target[0]-pos[0])
                S_target.extend([d/np.linalg.norm(2*self.length), theta])
                S_evade_d.append(d/np.linalg.norm(2*self.length))
        
        S_obser = self.multi_current_lasers[agent_idx]
        
        if agent_idx != self.num_agents - 1:
            single_obs = list(itertools.chain(S_uavi, S_team, S_obser, S_target))
        else:
            single_obs = list(itertools.chain(S_uavi, S_obser, S_evade_d))
        
        return single_obs
    
    def _get_search_target(self, agent_idx):
        """为指定的无人机生成搜索目标"""
        # 简单实现：将地图分为几个区域，每个无人机负责一个
        regions = [
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.75, 0.75]
        ]
        
        # 为每个无人机分配一个区域
        region_idx = agent_idx % len(regions)
        base_pos = np.array(regions[region_idx]) * self.length
        
        # 在基础位置附近随机生成目标
        offset = np.random.uniform(-0.2, 0.2, size=(2,))
        search_target = base_pos + offset
        
        # 确保在地图范围内
        search_target = np.clip(search_target, 0.1, self.length - 0.1)
        
        return search_target
    
    def render(self):
        """重写渲染方法以显示敌机和探测范围"""
        img = super().render()
        
        # 在渲染中添加敌机和探测范围
        plt.clf()
        
        # 绘制我方无人机和目标
        for i in range(self.num_agents - 1):
            pos = self.multi_current_pos[i]
            plt.scatter(pos[0], pos[1], c='b', marker='o', s=50)
            
            # 绘制探测范围
            detection_circle = plt.Circle((pos[0], pos[1]), self.detection_radius, 
                                         color='b', fill=False, alpha=0.3)
            plt.gca().add_patch(detection_circle)
            
            # 绘制攻击范围
            attack_circle = plt.Circle((pos[0], pos[1]), self.attack_radius, 
                                      color='r', fill=False, alpha=0.2)
            plt.gca().add_patch(attack_circle)
            
            # 显示健康值
            plt.text(pos[0], pos[1] + 0.05, f"HP: {int(self.uav_health[i])}", 
                    ha='center', va='bottom', color='blue')
        
        # 绘制目标
        pos_target = self.multi_current_pos[-1]
        plt.scatter(pos_target[0], pos_target[1], c='g', marker='*', s=100)
        
        # 绘制敌机
        for i in range(self.num_enemies):
            if self.enemy_health[i] <= 0:
                continue  # 跳过已消灭的敌机
                
            pos = self.enemy_positions[i]
            plt.scatter(pos[0], pos[1], c='r', marker='x', s=50)
            
            # 如果被探测到，用不同颜色标记
            if self.enemy_detected[i]:
                detection_circle = plt.Circle((pos[0], pos[1]), 0.05, 
                                             color='orange', fill=False)
                plt.gca().add_patch(detection_circle)
            
            # 显示健康值
            plt.text(pos[0], pos[1] - 0.05, f"HP: {int(self.enemy_health[i])}", 
                    ha='center', va='top', color='red')
        
        # 绘制障碍物
        for obs in self.obstacles:
            circle = plt.Circle(obs.position, obs.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        
        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()
        
        # 保存图像到缓冲区
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        
        return img