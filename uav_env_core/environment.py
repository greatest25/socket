import numpy as np
import itertools
from gymnasium import spaces
import random
import copy
import os

from .utils import cal_triangle_S
from .obstacles import ObstacleHandler
from .rendering import Renderer
from .uav_interactions import UAVInteractionHandler
from .rewards import RewardCalculator
from .observations import ObservationGenerator

class Qt5SimUAVEnv:
    def __init__(self, map_width=1280, map_height=800, num_static_obstacles=2, num_dynamic_obstacles=1, num_agents=3, num_enemies=3):
        self.map_width = map_width
        self.map_height = map_height
        self.num_static_obstacles = num_static_obstacles
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.num_total_obstacles = self.num_static_obstacles + self.num_dynamic_obstacles
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        
        self.time_step = 0.1  # 时间步长 (秒)
        
        # 我方无人机参数
        self.uav_max_vel_comp = 50.0  # 各方向最大速度分量 (像素/秒)

        # 敌方无人机参数
        self.enemy_speed_mag = 50.0  # 敌方无人机速度大小 (像素/秒)

        # 传感器和攻击参数 (像素)
        self.detection_radius = 300.0 # 探测半径 (探测敌机和障碍物)
        self.attack_radius = self.detection_radius / 2.0 # 攻击半径
        
        # 近似半径，用于碰撞惩罚等 (像素)
        self.uav_radius_approx = 10.0 # 假设无人机大小的近似半径
        self.enemy_radius_approx = 10.0 # 假设敌机大小的近似半径
        self.obstacle_max_speed = 20.0 # 动态障碍物最大速度

        # 初始化障碍物检测数据 (会被UAVInteractionHandler更新)
        self.obstacle_detected = [[] for _ in range(self.num_agents)]
        
        # 初始化障碍物
        self.obstacles = []
        self.obstacle_handler = ObstacleHandler(self.map_width, self.map_height, self.obstacle_max_speed)
        for _ in range(self.num_static_obstacles):
            self.obstacles.append(self.obstacle_handler._create_obstacle(is_dynamic=False))
        for _ in range(self.num_dynamic_obstacles):
            self.obstacles.append(self.obstacle_handler._create_obstacle(is_dynamic=True))
        
        #todo 初始化位置和速度
        self.multi_current_pos = np.zeros((self.num_agents, 2)) # 我方无人机位置 (像素)
        self.multi_current_vel = np.zeros((self.num_agents, 2)) # 我方无人机速度 (像素/秒)
        self.enemy_pos = np.zeros((self.num_enemies, 2))       # 敌方无人机位置 (像素)
        self.enemy_vel = np.zeros((self.num_enemies, 2))       # 敌方无人机速度 (像素/秒)
        
        # 状态参数
        self.enemy_health = [100.0] * self.num_enemies
        self.enemy_detected = [False] * self.num_enemies # 会被UAVInteractionHandler更新
        self.uav_health = [100.0] * self.num_agents
        self.search_mode = True # 会被UAVInteractionHandler更新
        self.tracking_mode = [False] * self.num_agents # 会被UAVInteractionHandler更新
        
        # 历史轨迹
        self.history_positions = [[] for _ in range(self.num_agents + self.num_enemies)]

        # 初始化各个子模块
        self.uav_interaction_handler = UAVInteractionHandler(
            self.num_agents, self.num_enemies, self.uav_health, self.enemy_health, 
            self.multi_current_pos, self.enemy_pos, self.detection_radius, 
            self.attack_radius, self.uav_radius_approx, self.obstacles, 
            self.obstacle_detected, self.search_mode, self.tracking_mode
        )

        self.reward_calculator = RewardCalculator(
            self.num_agents, self.num_enemies, self.uav_health, self.enemy_health,
            self.multi_current_pos, self.enemy_pos, self.detection_radius, self.attack_radius,
            self.uav_radius_approx, self.obstacles, self.obstacle_detected, self.search_mode,
            self.tracking_mode, self.map_width, self.map_height, self._get_search_target # 传入_get_search_target引用
        )

        self.observation_generator = ObservationGenerator(
            self.map_width, self.map_height, self.num_agents, self.num_enemies, self.num_total_obstacles,
            self.uav_max_vel_comp, self.uav_health, self.multi_current_pos, self.multi_current_vel,
            self.enemy_health, self.enemy_pos, self.enemy_detected, self.obstacle_detected, self.search_mode, self.tracking_mode
        )

        self.renderer = Renderer(
            self.map_width, self.map_height, self.num_agents, self.num_enemies,
            self.detection_radius, self.attack_radius, self.uav_radius_approx, self.enemy_radius_approx
        )
        
        # 动作空间和观察空间
        self.action_space = {}
        self.observation_space = {}
        
        #! 新的观测空间维度: 自身(5) + 队友(num_agents-1 * 3) + 敌机(num_enemies * 3) + 搜索(2) + 障碍物(num_total_obstacles * 3)
        #· 例如 3v3: 5 + (2*3=6) + (3*3=9) + 2 + (3*3=9) = 31
        # 注意：此维度变化会影响训练脚本中 actor_dims 和 critic_dims 的计算
        obs_dim = self.observation_generator.obs_dim # 直接从ObservationGenerator获取

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
                random.uniform(self.map_width - spawn_margin_x + self.uav_radius_approx,self.map_width - self.uav_radius_approx),
                random.uniform(self.map_height - spawn_margin_y + self.uav_radius_approx,self.map_height - self.uav_radius_approx)
            ])
            self.multi_current_vel[i] = np.zeros(2)
        
        # 重置敌方无人机 (红方 - 右下角)
        for i in range(self.num_enemies):
            self.enemy_pos[i] = np.array([
                random.uniform(self.uav_radius_approx, spawn_margin_x - self.uav_radius_approx),
                random.uniform(self.uav_radius_approx, spawn_margin_y - self.uav_radius_approx)
            ])
            angle = np.random.uniform(0, 2 * np.pi)
            self.enemy_vel[i] = np.array([
                self.enemy_speed_mag * np.cos(angle),self.enemy_speed_mag * np.sin(angle)
            ])
        
        self.enemy_health = [100.0] * self.num_enemies
        self.uav_health = [100.0] * self.num_agents
        self.enemy_detected = [False] * self.num_enemies
        self.obstacle_detected = [[] for _ in range(self.num_agents)]
        self.search_mode = True
        self.tracking_mode = [False] * self.num_agents # 重置追踪模式
        self.history_positions = [[] for _ in range(self.num_agents + self.num_enemies)]
        
        # 重置障碍物 (如果需要)
        self.obstacles = []
        for _ in range(self.num_static_obstacles):
            self.obstacles.append(self.obstacle_handler._create_obstacle(is_dynamic=False))
        for _ in range(self.num_dynamic_obstacles):
            self.obstacles.append(self.obstacle_handler._create_obstacle(is_dynamic=True))

        # 在重置时更新交互处理器的内部引用
        self.uav_interaction_handler.uav_health = self.uav_health
        self.uav_interaction_handler.enemy_health = self.enemy_health
        self.uav_interaction_handler.multi_current_pos = self.multi_current_pos
        self.uav_interaction_handler.enemy_pos = self.enemy_pos
        self.uav_interaction_handler.obstacles = self.obstacles
        self.uav_interaction_handler.obstacle_detected = self.obstacle_detected
        self.uav_interaction_handler.search_mode = self.search_mode
        self.uav_interaction_handler.tracking_mode = self.tracking_mode

        self.reward_calculator.uav_health = self.uav_health
        self.reward_calculator.enemy_health = self.enemy_health
        self.reward_calculator.multi_current_pos = self.multi_current_pos
        self.reward_calculator.enemy_pos = self.enemy_pos
        self.reward_calculator.obstacles = self.obstacles
        self.reward_calculator.obstacle_detected = self.obstacle_detected
        self.reward_calculator.search_mode = self.search_mode
        self.reward_calculator.tracking_mode = self.tracking_mode

        self.observation_generator.uav_health = self.uav_health
        self.observation_generator.multi_current_pos = self.multi_current_pos
        self.observation_generator.multi_current_vel = self.multi_current_vel
        self.observation_generator.enemy_health = self.enemy_health
        self.observation_generator.enemy_pos = self.enemy_pos
        self.observation_generator.enemy_detected = self.enemy_detected
        self.observation_generator.obstacle_detected = self.obstacle_detected
        self.observation_generator.search_mode = self.search_mode
        self.observation_generator.tracking_mode = self.tracking_mode
        
        self.uav_interaction_handler.update_detections_and_obstacles()
        return self.observation_generator.get_obs()
    
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
            uav_input_accel_scale = 50.0 # 假设最大输入加速度为50 px/s^2 (与最大速度同量级)
            
            current_accel = np.array(actions[i]) * uav_input_accel_scale

            # 更新速度 (vx = vx + ax*dt)
            self.multi_current_vel[i] += current_accel * self.time_step
            
            # 限制速度分量
            self.multi_current_vel[i][0] = np.clip(self.multi_current_vel[i][0], -self.uav_max_vel_comp, self.uav_max_vel_comp)
            self.multi_current_vel[i][1] = np.clip(self.multi_current_vel[i][1], -self.uav_max_vel_comp, self.uav_max_vel_comp)
            
            # 更新位置 (x = x + vx*dt)
            self.multi_current_pos[i] += self.multi_current_vel[i] * self.time_step
            
            # 边界限制 (将无人机限制在地图内，碰到边界速度反弹)
            if self.multi_current_pos[i][0] - self.uav_radius_approx < 0:
                self.multi_current_pos[i][0] = self.uav_radius_approx
                self.multi_current_vel[i][0] *= -1 
            elif self.multi_current_pos[i][0] + self.uav_radius_approx > self.map_width:
                self.multi_current_pos[i][0] = self.map_width - self.uav_radius_approx
                self.multi_current_vel[i][0] *= -1
            
            if self.multi_current_pos[i][1] - self.uav_radius_approx < 0:
                self.multi_current_pos[i][1] = self.uav_radius_approx
                self.multi_current_vel[i][1] *= -1
            elif self.multi_current_pos[i][1] + self.uav_radius_approx > self.map_height:
                self.multi_current_pos[i][1] = self.map_height - self.uav_radius_approx
                self.multi_current_vel[i][1] *= -1

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
        
        # 更新障碍物 (包括动态障碍物)
        for obs_item in self.obstacles:
            if obs_item['is_dynamic']:
                obs_item['position'] += obs_item['velocity'] * self.time_step
                # 动态障碍物边界反射
                if obs_item['position'][0] - obs_item['radius'] < 0:
                    obs_item['position'][0] = obs_item['radius']
                    obs_item['velocity'][0] *= -1
                elif obs_item['position'][0] + obs_item['radius'] > self.map_width:
                    obs_item['position'][0] = self.map_width - obs_item['radius']
                    obs_item['velocity'][0] *= -1
                
                if obs_item['position'][1] - obs_item['radius'] < 0:
                    obs_item['position'][1] = obs_item['radius']
                    obs_item['velocity'][1] *= -1
                elif obs_item['position'][1] + obs_item['radius'] > self.map_height:
                    obs_item['position'][1] = self.map_height - obs_item['radius']
                    obs_item['velocity'][1] *= -1
        
        self.uav_interaction_handler.update_detections_and_obstacles()
        self.uav_interaction_handler.process_attacks() # 恢复攻击处理

        rewards, dones = self.reward_calculator.calculate_rewards(last_distances_to_enemies)
        obs = self.observation_generator.get_obs()
        
        return obs, rewards, dones
    
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

    def render(self, uav_icon_path='UAV.png', enemy_icon_path='Enemy_UAV.png'):
        return self.renderer.render(self.uav_health, self.multi_current_pos, self.multi_current_vel, 
                                   self.enemy_pos, self.enemy_vel, self.enemy_health, 
                                   self.enemy_detected, self.obstacles, self.history_positions,
                                   uav_icon_path, enemy_icon_path)

    def close(self):
        self.renderer.close()

# 由于文件已拆分，原qt5_sim_env.py中的main函数将不再适用
# 独立测试环境应通过新的main_qt5_sim.py或evaluate_qt5_sim.py来运行
# if __name__ == "__main__":
#    ...  