import numpy as np
from gymnasium import spaces # 导入spaces
import random

class ObservationGenerator:
    def __init__(self, map_width, map_height, num_agents, num_enemies, num_total_obstacles,
                 uav_max_vel_comp, uav_health, multi_current_pos, multi_current_vel,
                 enemy_health, enemy_pos, enemy_detected, obstacle_detected, search_mode, tracking_mode):
        self.map_width = map_width
        self.map_height = map_height
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        self.num_total_obstacles = num_total_obstacles
        self.uav_max_vel_comp = uav_max_vel_comp
        self.uav_health = uav_health
        self.multi_current_pos = multi_current_pos
        self.multi_current_vel = multi_current_vel
        self.enemy_health = enemy_health
        self.enemy_pos = enemy_pos
        self.enemy_detected = enemy_detected # 外部引用
        self.obstacle_detected = obstacle_detected # 外部引用
        self.search_mode = search_mode
        self.tracking_mode = tracking_mode # 各个智能体的追踪模式

        # 计算obs_dim，与环境类中的计算保持一致
        self.obs_dim = 5 + (self.num_agents - 1) * 3 + self.num_enemies * 3 + 2 + self.num_total_obstacles * 3

    def get_obs(self):
        all_obs = []
        for i in range(self.num_agents):
            if self.uav_health[i] <= 0:
                all_obs.append(np.zeros(self.obs_dim, dtype=np.float32))
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

            # 5. 障碍物信息 (num_total_obstacles个最近障碍物 * 3维 = 9维 for 3 obstacles: rel_pos_x, rel_pos_y, radius)
            obs_obstacle = []
            # 按距离障碍物表面排序 (distance_to_center - radius)
            # self.obstacle_detected[i] 中是字典列表，包含 'distance_to_center', 'radius', 'absolute_pos'
            sorted_obstacles = sorted(
                self.obstacle_detected[i],
                key=lambda obs_data: obs_data['distance_to_center'] - obs_data['radius']
            )
            
            num_obstacles_for_obs = 0
            # 最多提供 self.num_total_obstacles 个障碍物信息
            max_obs_obstacles = self.num_total_obstacles 
            for obs_data in sorted_obstacles:
                if num_obstacles_for_obs < max_obs_obstacles:
                    rel_pos_obs = obs_data['absolute_pos'] - agent_pos
                    obs_obstacle.extend([
                        rel_pos_obs[0] / self.map_width,
                        rel_pos_obs[1] / self.map_height,
                        obs_data['radius'] / max(self.map_width, self.map_height) # 半径归一化
                    ])
                    num_obstacles_for_obs += 1
            # 填充不足的障碍物信息
            obs_obstacle.extend([0.0] * (max_obs_obstacles * 3 - len(obs_obstacle)))
            
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