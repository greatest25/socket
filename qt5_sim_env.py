import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
# from math_tool import *  # 移除对math_tool的依赖
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy
import math  # 添加math模块导入

# 添加cal_triangle_S函数的实现
def cal_triangle_S(p1, p2, p3):
    S = abs(0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])))
    if math.isclose(S, 0.0, abs_tol=1e-9):
        return 0.0
    else:
        return S

class Qt5SimUAVEnv:
    def __init__(self, map_width=1280, map_height=800, num_obstacle=5, num_agents=3, num_enemies=3):
        self.map_width = map_width
        self.map_height = map_height
        self.num_obstacle = num_obstacle
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        
        self.time_step = 0.1  # 时间步长 (秒)
        
        # 我方无人机参数
        self.uav_max_vel_comp = 50.0  # 各方向最大速度分量 (像素/秒)
        # 假设动作直接是加速度 (像素/秒^2)，具体大小由策略网络和Agent类输出决定
        # self.uav_max_accel_mag = 50.0 # (像素/秒^2) - 这应在Agent/Network层面处理

        # 敌方无人机参数
        self.enemy_speed_mag = 50.0  # 敌方无人机速度大小 (像素/秒)
        # self.enemy_flee_accel_mag = 50.0 # 敌机改变方向时的加速度，实际通过直接设置速度方向实现

        # 传感器和攻击参数 (像素)
        self.detection_radius = self.map_width / 6.0 # 探测半径 (探测敌机和障碍物)
        self.attack_radius = self.detection_radius / 2.0 # 攻击半径
        
        # 近似半径，用于碰撞惩罚等 (像素)
        self.uav_radius_approx = 10.0 # 假设无人机大小的近似半径
        self.enemy_radius_approx = 10.0 # 假设敌机大小的近似半径

        # 初始化障碍物检测数据
        self.obstacle_detected = [[] for _ in range(self.num_agents)]
        
        # 初始化障碍物
        self.obstacles = [self._create_obstacle() for _ in range(self.num_obstacle)]
        
        # 初始化位置和速度
        self.multi_current_pos = np.zeros((self.num_agents, 2)) # 我方无人机位置 (像素)
        self.multi_current_vel = np.zeros((self.num_agents, 2)) # 我方无人机速度 (像素/秒)
        self.enemy_pos = np.zeros((self.num_enemies, 2))       # 敌方无人机位置 (像素)
        self.enemy_vel = np.zeros((self.num_enemies, 2))       # 敌方无人机速度 (像素/秒)
        
        # 状态参数
        self.enemy_health = [100.0] * self.num_enemies
        self.enemy_detected = [False] * self.num_enemies
        self.uav_health = [100.0] * self.num_agents
        self.search_mode = True
        
        # 历史轨迹
        self.history_positions = [[] for _ in range(self.num_agents + self.num_enemies)]
        
        # 动作空间和观察空间
        self.action_space = {}
        self.observation_space = {}
        
        # 新的观测空间维度: 自身(5) + 队友(2*3=6) + 敌机(3*3=9) + 搜索(2) + 障碍物(2*3=6) = 28
        # 注意：此维度变化会影响训练脚本中 actor_dims 和 critic_dims 的计算
        obs_dim = 28 
        for i in range(self.num_agents):
            agent_id = f'agent_{i}'
            # 动作是加速度的两个分量 [ax, ay]
            self.action_space[agent_id] = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) # 假设策略输出归一化加速度
            self.observation_space[agent_id] = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    
    def reset(self):
        SEED = random.randint(1, 1000)
        random.seed(SEED)
        np.random.seed(SEED)
        
        # 定义出生区域边界 (左上角为蓝方，右下角为红方)
        spawn_margin_x = self.map_width / 10.0
        spawn_margin_y = self.map_height / 10.0

        # 重置我方无人机 (蓝方 - 左上角)
        for i in range(self.num_agents):
            self.multi_current_pos[i] = np.array([
                random.uniform(self.uav_radius_approx, spawn_margin_x - self.uav_radius_approx),
                random.uniform(self.uav_radius_approx, spawn_margin_y - self.uav_radius_approx)
            ])
            self.multi_current_vel[i] = np.zeros(2)
        
        # 重置敌方无人机 (红方 - 右下角)
        for i in range(self.num_enemies):
            self.enemy_pos[i] = np.array([
                random.uniform(self.map_width - spawn_margin_x + self.enemy_radius_approx, self.map_width - self.enemy_radius_approx),
                random.uniform(self.map_height - spawn_margin_y + self.enemy_radius_approx, self.map_height - self.enemy_radius_approx)
            ])
            angle = np.random.uniform(0, 2 * np.pi)
            self.enemy_vel[i] = np.array([self.enemy_speed_mag * np.cos(angle), self.enemy_speed_mag * np.sin(angle)])
        
        self.enemy_health = [100.0] * self.num_enemies
        self.uav_health = [100.0] * self.num_agents
        self.enemy_detected = [False] * self.num_enemies
        self.obstacle_detected = [[] for _ in range(self.num_agents)]
        self.search_mode = True
        self.history_positions = [[] for _ in range(self.num_agents + self.num_enemies)]
        
        self._update_detection_and_obstacles()
        return self._get_obs()
    
    def step(self, actions): # actions是包含各agent动作(归一化加速度)的列表
        # 记录上一步每个我方UAV到每个存活敌机的距离
        last_distances_to_enemies = []
        for i in range(self.num_agents):
            agent_distances = []
            if self.uav_health[i] > 0:
                for j in range(self.num_enemies):
                    if self.enemy_health[j] > 0:
                        agent_distances.append(np.linalg.norm(self.multi_current_pos[i] - self.enemy_pos[j]))
                    else:
                        agent_distances.append(float('inf'))
            else: # 无人机已损毁
                 agent_distances = [float('inf')] * self.num_enemies
            last_distances_to_enemies.append(agent_distances)

        # 更新我方无人机
        for i in range(self.num_agents):
            if self.uav_health[i] <= 0:
                self.multi_current_vel[i] = np.zeros(2) # 停止移动
                continue
            
            # 假设actions[i]是 [-1, 1]范围的归一化加速度, 需要乘以一个缩放因子得到实际加速度
            # 为简化，此处假设 actions[i] 直接就是 px/s^2单位的加速度值，具体缩放应在Agent/Network中完成
            # 例如: actual_accel = actions[i] * MAX_ACCELERATION_ACTUAL_UNITS
            # 我们用一个固定的最大加速度值来模拟这个缩放
            uav_input_accel_scale = 50.0 # 假设最大输入加速度为50 px/s^2 (与最大速度同量级)
            
            current_accel = np.array(actions[i]) * uav_input_accel_scale

            # 更新速度 (vx = vx + ax*dt)
            self.multi_current_vel[i] += current_accel * self.time_step
            
            # 限制速度分量
            self.multi_current_vel[i][0] = np.clip(self.multi_current_vel[i][0], -self.uav_max_vel_comp, self.uav_max_vel_comp)
            self.multi_current_vel[i][1] = np.clip(self.multi_current_vel[i][1], -self.uav_max_vel_comp, self.uav_max_vel_comp)
            
            # 更新位置 (x = x + vx*dt)
            self.multi_current_pos[i] += self.multi_current_vel[i] * self.time_step
            
            # 边界限制 (将无人机限制在地图内，碰到边界速度归零)
            if self.multi_current_pos[i][0] - self.uav_radius_approx < 0:
                self.multi_current_pos[i][0] = self.uav_radius_approx
                self.multi_current_vel[i][0] = 0
            elif self.multi_current_pos[i][0] + self.uav_radius_approx > self.map_width:
                self.multi_current_pos[i][0] = self.map_width - self.uav_radius_approx
                self.multi_current_vel[i][0] = 0
            
            if self.multi_current_pos[i][1] - self.uav_radius_approx < 0:
                self.multi_current_pos[i][1] = self.uav_radius_approx
                self.multi_current_vel[i][1] = 0
            elif self.multi_current_pos[i][1] + self.uav_radius_approx > self.map_height:
                self.multi_current_pos[i][1] = self.map_height - self.uav_radius_approx
                self.multi_current_vel[i][1] = 0

        # 更新敌方无人机
        for j in range(self.num_enemies):
            if self.enemy_health[j] <= 0:
                self.enemy_vel[j] = np.zeros(2) # 停止移动
                continue

            # 敌机AI：如果被探测到且有存活的蓝方，则逃离最近的蓝方UAV
            if self.enemy_detected[j]:
                alive_uav_pos = [self.multi_current_pos[i] for i in range(self.num_agents) if self.uav_health[i] > 0]
                if alive_uav_pos:
                    distances_to_uavs = [np.linalg.norm(self.enemy_pos[j] - p) for p in alive_uav_pos]
                    nearest_uav_pos = alive_uav_pos[np.argmin(distances_to_uavs)]
                    escape_dir = self.enemy_pos[j] - nearest_uav_pos
                    if np.linalg.norm(escape_dir) > 1e-6: # Avoid division by zero
                        escape_dir = escape_dir / np.linalg.norm(escape_dir)
                    else: # 完全重合或极近，随机一个方向
                        angle = np.random.uniform(0, 2 * np.pi)
                        escape_dir = np.array([np.cos(angle), np.sin(angle)])
                    self.enemy_vel[j] = escape_dir * self.enemy_speed_mag
                # else: 若没有存活的蓝方UAV，敌机继续按原方向运动
            # else: 若未被探测到，敌机按当前速度方向（可能随机改变）以固定速率飞行
            # （此处简化：未探测时速度在reset或上次转向时已设为固定速率和随机方向）
            # 简单随机转向逻辑 (当未被探测时)
            elif np.random.random() < 0.02: # 2% 概率随机转向
                angle = np.random.uniform(0, 2 * np.pi)
                self.enemy_vel[j] = np.array([self.enemy_speed_mag * np.cos(angle), self.enemy_speed_mag * np.sin(angle)])

            self.enemy_pos[j] += self.enemy_vel[j] * self.time_step
            
            # 敌机边界反射
            if self.enemy_pos[j][0] - self.enemy_radius_approx < 0:
                self.enemy_pos[j][0] = self.enemy_radius_approx
                self.enemy_vel[j][0] *= -1
            elif self.enemy_pos[j][0] + self.enemy_radius_approx > self.map_width:
                self.enemy_pos[j][0] = self.map_width - self.enemy_radius_approx
                self.enemy_vel[j][0] *= -1
            
            if self.enemy_pos[j][1] - self.enemy_radius_approx < 0:
                self.enemy_pos[j][1] = self.enemy_radius_approx
                self.enemy_vel[j][1] *= -1
            elif self.enemy_pos[j][1] + self.enemy_radius_approx > self.map_height:
                self.enemy_pos[j][1] = self.map_height - self.enemy_radius_approx
                self.enemy_vel[j][1] *= -1
        
        # 更新移动障碍物 (当前为静止)
        for obs_item in self.obstacles:
            # obs_item['position'] += obs_item['velocity'] * self.time_step # 若障碍物移动
            # 边界检查 (若障碍物移动)
            pass 
            
        self._update_detection_and_obstacles()
        self._process_attacks()
        rewards, dones = self._calculate_rewards(last_distances_to_enemies)
        obs = self._get_obs()
        
        return obs, rewards, dones
    
    def _update_detection_and_obstacles(self):
        # 重置探测状态
        self.enemy_detected = [False] * self.num_enemies
        self.obstacle_detected = [[] for _ in range(self.num_agents)]
        
        # 更新敌机探测状态 和 障碍物探测状态
        for i in range(self.num_agents):
            if self.uav_health[i] <= 0:
                continue
            
            agent_pos = self.multi_current_pos[i]
            # 探测敌机
            for j in range(self.num_enemies):
                if self.enemy_health[j] > 0:
                    distance_to_enemy = np.linalg.norm(agent_pos - self.enemy_pos[j])
                    if distance_to_enemy <= self.detection_radius:
                        self.enemy_detected[j] = True
            
            # 探测障碍物
            for obs_item in self.obstacles:
                obs_pos = obs_item['position']
                obs_rad = obs_item['radius']
                distance_to_obstacle_center = np.linalg.norm(agent_pos - obs_pos)
                # 如果UAV的探测范围边缘触及或穿过障碍物
                if distance_to_obstacle_center <= self.detection_radius + obs_rad:
                    self.obstacle_detected[i].append({
                        'absolute_pos': obs_pos.copy(),         # 障碍物中心绝对位置
                        'radius': obs_rad,                      # 障碍物绝对半径
                        'distance_to_center': distance_to_obstacle_center # UAV到障碍物中心的绝对距离
                    })
        
        if any(self.enemy_detected):
            self.search_mode = False
        else:
            # 如果所有敌机都未被探测到 (可能因为超出范围或已被击毁)
            # 并且还有存活的敌机，则继续搜索
            if any(h > 0 for h in self.enemy_health):
                 self.search_mode = True
            else: # 所有敌机都被击毁
                 self.search_mode = False


    def _process_attacks(self):
        # 我方攻击敌机
        for i in range(self.num_agents):
            if self.uav_health[i] <= 0: continue
            for j in range(self.num_enemies):
                if self.enemy_health[j] <= 0: continue
                distance = np.linalg.norm(self.multi_current_pos[i] - self.enemy_pos[j])
                if distance <= self.attack_radius:
                    damage = random.uniform(15, 25) # 增加伤害，更快击杀
                    self.enemy_health[j] = max(0, self.enemy_health[j] - damage)
        
        # 敌机攻击我方 (假设敌机也有相同的攻击逻辑和半径)
        for j in range(self.num_enemies):
            if self.enemy_health[j] <= 0: continue
            for i in range(self.num_agents):
                if self.uav_health[i] <= 0: continue
                distance = np.linalg.norm(self.enemy_pos[j] - self.multi_current_pos[i])
                if distance <= self.attack_radius: # 假设敌机攻击半径与我方相同
                    damage = random.uniform(5, 10)
                    self.uav_health[i] = max(0, self.uav_health[i] - damage)

    def _calculate_rewards(self, last_distances_to_enemies):
        rewards = np.zeros(self.num_agents)
        dones = [False] * self.num_agents
        
        # 存活的UAV数量
        num_alive_uavs = sum(1 for h in self.uav_health if h > 0)

        for i in range(self.num_agents):
            if self.uav_health[i] <= 0:
                rewards[i] = -10.0 # 死亡惩罚
                dones[i] = True
                continue

            # 1. 接近目标奖励 / 搜索奖励
            if self.search_mode:
                search_target_pos = self._get_search_target(i) # 动态或固定搜索点
                dist_to_search_target = np.linalg.norm(self.multi_current_pos[i] - search_target_pos)
                # 奖励探索未知区域或特定搜索点，越近奖励越高，可以用 (1 - dist_norm)
                rewards[i] += 0.5 * (1.0 - dist_to_search_target / max(self.map_width, self.map_height))
            else: # 围捕模式
                # 对每个存活的敌机计算奖励
                for enemy_idx in range(self.num_enemies):
                    if self.enemy_health[enemy_idx] > 0:
                        current_dist_to_enemy = np.linalg.norm(self.multi_current_pos[i] - self.enemy_pos[enemy_idx])
                        last_dist_to_enemy = last_distances_to_enemies[i][enemy_idx]
                        
                        # 接近敌人的奖励
                        dist_change_reward = (last_dist_to_enemy - current_dist_to_enemy) / self.detection_radius # 归一化
                        rewards[i] += 25.0 * dist_change_reward # 增加奖励系数
                        
                        # 在攻击范围内的奖励
                        if current_dist_to_enemy <= self.attack_radius:
                            rewards[i] += 1.5
                        # 在探测范围内的奖励
                        elif current_dist_to_enemy <= self.detection_radius:
                             rewards[i] += 1.0 # 增加探测奖励
            
            # 2. 团队合作奖励 (暂不实现复杂的三角包围，改为共享探测信息和协同攻击)
            #   - 如果一个UAV探测到敌人，其他UAV朝向该敌人或协同攻击时可以给予奖励
            #   - 简单的协同：如果多个UAV同时攻击一个目标，或保持在目标附近
            if not self.search_mode: # 围捕模式下
                num_attacking_same_target = 0
                closest_enemy_idx = -1
                min_dist_to_an_enemy = float('inf')

                # 找到当前UAV最近的存活敌机
                for enemy_idx in range(self.num_enemies):
                    if self.enemy_health[enemy_idx] > 0:
                        d = np.linalg.norm(self.multi_current_pos[i] - self.enemy_pos[enemy_idx])
                        if d < min_dist_to_an_enemy:
                            min_dist_to_an_enemy = d
                            closest_enemy_idx = enemy_idx
                
                if closest_enemy_idx != -1 and min_dist_to_an_enemy <= self.attack_radius: # 如果在攻击一个目标
                    # 检查其他活着的队友是否也在攻击或接近同一个目标
                    for teammate_idx in range(self.num_agents):
                        if teammate_idx != i and self.uav_health[teammate_idx] > 0:
                            dist_teammate_to_target = np.linalg.norm(self.multi_current_pos[teammate_idx] - self.enemy_pos[closest_enemy_idx])
                            if dist_teammate_to_target <= self.attack_radius * 1.2: # 队友也在附近攻击
                                num_attacking_same_target +=1
                    rewards[i] += num_attacking_same_target * 0.5 # 协同攻击奖励

            # 3. 消灭敌机奖励 (由所有对该敌机造成伤害或在附近的无人机分享)
            # (在_process_attacks中处理伤害，这里根据血量判断是否被消灭)
            # 这个奖励在全局判断后分配会更公平，这里暂时简化为个体判断
            for enemy_idx in range(self.num_enemies):
                 if self.enemy_health[enemy_idx] <= 0 and last_distances_to_enemies[i][enemy_idx] != float('inf'): # 如果这回合刚击杀
                    # 检查是否是该UAV的贡献 (例如，在附近)
                    if np.linalg.norm(self.multi_current_pos[i] - self.enemy_pos[enemy_idx]) < self.attack_radius * 1.5:
                        rewards[i] += 20.0 # 击杀奖励

            # 4. 碰撞惩罚 (与障碍物)
            min_dist_to_obs_surface = float('inf')
            for obs_data in self.obstacle_detected[i]: # obstacle_detected 包含 'distance_to_center' 和 'radius'
                dist_surface = obs_data['distance_to_center'] - obs_data['radius'] - self.uav_radius_approx
                if dist_surface < min_dist_to_obs_surface:
                    min_dist_to_obs_surface = dist_surface
            
            if min_dist_to_obs_surface < 5.0: # 距离障碍物表面小于5像素
                rewards[i] -= 5.0
            
            # 5. 时间惩罚 / 生存奖励
            rewards[i] -= 0.05 # 每步的小惩罚，鼓励尽快完成任务

        # 全局结束条件
        if all(h <= 0 for h in self.enemy_health): # 所有敌机被击毁
            for i in range(self.num_agents):
                if self.uav_health[i] > 0:
                    rewards[i] += 50.0 # 任务完成额外奖励
                dones[i] = True
        
        if num_alive_uavs == 0: # 所有我方无人机被击毁
             for i in range(self.num_agents):
                # rewards[i] -= 20.0 # 已经在个体死亡时惩罚
                dones[i] = True
        
        return rewards, dones

    def _get_obs(self):
        all_obs = []
        for i in range(self.num_agents):
            if self.uav_health[i] <= 0:
                all_obs.append(np.zeros(self.observation_space[f'agent_{i}'].shape[0], dtype=np.float32))
                continue
            
            agent_pos = self.multi_current_pos[i]
            agent_vel = self.multi_current_vel[i]

            # 1. 自身信息 (5维: pos_x, pos_y, vel_x, vel_y, health)
            obs_self = [
                agent_pos[0] / self.map_width,
                agent_pos[1] / self.map_height,
                agent_vel[0] / self.uav_max_vel_comp,
                agent_vel[1] / self.uav_max_vel_comp,
                self.uav_health[i] / 100.0
            ]
            
            # 2. 队友信息 (num_agents-1 * 3维 = 2*3=6维: rel_pos_x, rel_pos_y, health)
            obs_team = []
            for teammate_idx in range(self.num_agents):
                if teammate_idx == i: continue
                if self.uav_health[teammate_idx] > 0:
                    teammate_pos = self.multi_current_pos[teammate_idx]
                    obs_team.extend([
                        (teammate_pos[0] - agent_pos[0]) / self.map_width, # 相对位置
                        (teammate_pos[1] - agent_pos[1]) / self.map_height,
                        self.uav_health[teammate_idx] / 100.0
                    ])
                else: # 队友已损毁
                    obs_team.extend([0.0, 0.0, 0.0]) 
            
            # 3. 敌机信息 (num_enemies * 3维 = 3*3=9维: rel_pos_x, rel_pos_y, health)
            # 只提供探测到的敌机信息，未探测到的用0填充
            obs_enemy = []
            num_detected_enemies_for_obs = 0
            # 按与当前agent的距离排序，优先报告近的
            sorted_enemy_indices = sorted(
                range(self.num_enemies),
                key=lambda j: np.linalg.norm(agent_pos - self.enemy_pos[j]) if self.enemy_health[j] > 0 else float('inf')
            )

            for enemy_idx in sorted_enemy_indices:
                if self.enemy_health[enemy_idx] > 0 and self.enemy_detected[enemy_idx]:
                    if num_detected_enemies_for_obs < self.num_enemies: # 最多提供num_enemies个
                        enemy_p = self.enemy_pos[enemy_idx]
                        obs_enemy.extend([
                            (enemy_p[0] - agent_pos[0]) / self.map_width, # 相对位置
                            (enemy_p[1] - agent_pos[1]) / self.map_height,
                            self.enemy_health[enemy_idx] / 100.0
                            # 速度不提供
                        ])
                        num_detected_enemies_for_obs +=1
            # 填充未探测到或超出数量的敌机信息
            obs_enemy.extend([0.0] * (self.num_enemies * 3 - len(obs_enemy)))

            # 4. 搜索目标信息 (2维: rel_pos_x, rel_pos_y to search_target)
            obs_search = [0.0, 0.0]
            if self.search_mode:
                search_target_pos = self._get_search_target(i)
                obs_search = [
                    (search_target_pos[0] - agent_pos[0]) / self.map_width,
                    (search_target_pos[1] - agent_pos[1]) / self.map_height
                ]

            # 5. 障碍物信息 (2个最近障碍物 * 3维 = 6维: rel_pos_x, rel_pos_y, radius)
            obs_obstacle = []
            # 按距离障碍物表面排序 (distance_to_center - radius)
            # self.obstacle_detected[i] 中是字典列表，包含 'distance_to_center', 'radius', 'absolute_pos'
            sorted_obstacles = sorted(
                self.obstacle_detected[i],
                key=lambda obs_data: obs_data['distance_to_center'] - obs_data['radius']
            )
            
            num_obstacles_for_obs = 0
            for obs_data in sorted_obstacles:
                if num_obstacles_for_obs < 2: # 最多提供2个最近的障碍物信息
                    rel_pos_obs = obs_data['absolute_pos'] - agent_pos
                    obs_obstacle.extend([
                        rel_pos_obs[0] / self.map_width,
                        rel_pos_obs[1] / self.map_height,
                        obs_data['radius'] / max(self.map_width, self.map_height) # 半径归一化
                    ])
                    num_obstacles_for_obs += 1
            # 填充不足的障碍物信息
            obs_obstacle.extend([0.0] * (2 * 3 - len(obs_obstacle)))
            
            single_obs_list = obs_self + obs_team + obs_enemy + obs_search + obs_obstacle
            all_obs.append(np.array(single_obs_list, dtype=np.float32))
            
        return all_obs

    def _get_search_target(self, agent_idx):
        # 简单固定区域搜索或动态生成，这里用固定区域演示
        # 将地图分为几个区域，每个无人机负责一个 (可根据agent_idx分配)
        num_search_regions_x = 2
        num_search_regions_y = 2
        region_w = self.map_width / num_search_regions_x
        region_h = self.map_height / num_search_regions_y
        
        # Assign regions in a grid pattern
        col = agent_idx % num_search_regions_x
        row = (agent_idx // num_search_regions_x) % num_search_regions_y
        
        target_x = (col + 0.5) * region_w
        target_y = (row + 0.5) * region_h
        
        # 添加随机扰动使目标不完全固定
        target_x += random.uniform(-region_w/4, region_w/4)
        target_y += random.uniform(-region_h/4, region_h/4)

        return np.array([
            np.clip(target_x, 0, self.map_width),
            np.clip(target_y, 0, self.map_height)
        ])

    def _create_obstacle(self):
        # 避免在出生点附近生成障碍物
        spawn_protection_x = self.map_width / 5.0
        spawn_protection_y = self.map_height / 5.0
        
        while True:
            pos_x = random.uniform(0, self.map_width)
            pos_y = random.uniform(0, self.map_height)
            # 检查是否在蓝方出生区
            is_in_blue_spawn = (pos_x < spawn_protection_x and pos_y < spawn_protection_y)
            # 检查是否在红方出生区
            is_in_red_spawn = (pos_x > self.map_width - spawn_protection_x and pos_y > self.map_height - spawn_protection_y)
            
            if not (is_in_blue_spawn or is_in_red_spawn):
                break
        
        radius = random.uniform(self.map_width / 70, self.map_width / 50) # e.g., for 1280, approx 18 to 25 pixels
        return {
            'position': np.array([pos_x, pos_y]),
            'velocity': np.zeros(2), # 静止障碍物
            'radius': radius
        }

    def render(self, uav_icon_path='UAV.png', enemy_icon_path='Enemy_UAV.png'):
        plt.clf()
        ax = plt.gca() # Get current axes
        
        uav_icon, enemy_icon = None, None
        icon_pixel_size = 30 # 图标显示大小 (像素) - 需要根据 plt 坐标转换调整

        try: uav_icon = mpimg.imread(uav_icon_path)
        except: pass
        try: enemy_icon = mpimg.imread(enemy_icon_path)
        except: pass
        
        # 我方无人机 (蓝方)
        for i in range(self.num_agents):
            if self.uav_health[i] <= 0: continue
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos.copy())
            trajectory = np.array(self.history_positions[i])
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.2, linewidth=0.5)
            
            angle_rad = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel) > 1e-3 else 0

            if uav_icon is not None:
                # imshow extent is (left, right, bottom, top) in data coordinates
                # For rotation, translate to origin, rotate, translate to position
                t = transforms.Affine2D().rotate_around(0,0,angle_rad).translate(pos[0], pos[1])
                ax.imshow(uav_icon, transform=t + ax.transData, 
                           extent=[-icon_pixel_size/2, icon_pixel_size/2, -icon_pixel_size/2, icon_pixel_size/2], 
                           interpolation='bicubic')
            else:
                plt.scatter(pos[0], pos[1], c='blue', marker='o', s=50) # s is area in points^2
            
            # 探测和攻击范围圆圈
            det_circle = plt.Circle((pos[0], pos[1]), self.detection_radius, color='blue', fill=False, alpha=0.2, linewidth=0.5)
            att_circle = plt.Circle((pos[0], pos[1]), self.attack_radius, color='red', fill=False, alpha=0.1, linewidth=0.5)
            ax.add_patch(det_circle)
            ax.add_patch(att_circle)
            plt.text(pos[0], pos[1] + 15, f"B{i}:{int(self.uav_health[i])}", color='blue', fontsize=7)

        # 敌方无人机 (红方)
        for j in range(self.num_enemies):
            if self.enemy_health[j] <= 0: continue
            pos = self.enemy_pos[j]
            vel = self.enemy_vel[j] #虽然不给观测，但渲染时可以用
            self.history_positions[self.num_agents + j].append(pos.copy())
            trajectory = np.array(self.history_positions[self.num_agents + j])
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.2, linewidth=0.5)

            angle_rad = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel) > 1e-3 else 0
            
            if enemy_icon is not None:
                t = transforms.Affine2D().rotate_around(0,0,angle_rad).translate(pos[0], pos[1])
                ax.imshow(enemy_icon, transform=t + ax.transData,
                           extent=[-icon_pixel_size/2, icon_pixel_size/2, -icon_pixel_size/2, icon_pixel_size/2],
                           interpolation='bicubic')
            else:
                plt.scatter(pos[0], pos[1], c='red', marker='x', s=50)
            
            if self.enemy_detected[j]: # 被探测到时，给个橙色外圈
                detected_indicator = plt.Circle((pos[0], pos[1]), self.enemy_radius_approx + 5, color='orange', fill=False, linewidth=1)
                ax.add_patch(detected_indicator)
            plt.text(pos[0], pos[1] - 15, f"R{j}:{int(self.enemy_health[j])}", color='red', fontsize=7)
            
        # 障碍物
        for obs_item in self.obstacles:
            circle = plt.Circle(obs_item['position'], obs_item['radius'], color='gray', alpha=0.6)
            ax.add_patch(circle)
            
        plt.xlim(0, self.map_width)
        plt.ylim(0, self.map_height)
        plt.gca().set_aspect('equal', adjustable='box') #保持横纵比一致
        plt.draw()
        
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        return img

    def close(self):
        plt.close()

# 添加main函数，用于独立测试环境
if __name__ == "__main__":
    env = Qt5SimUAVEnv(map_width=1280, map_height=800, num_agents=3, num_enemies=3, num_obstacle=5)
    
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