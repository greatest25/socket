import numpy as np
import random

class RewardCalculator:
    def __init__(self, num_agents, num_enemies, uav_health, enemy_health,
                 multi_current_pos, enemy_pos, detection_radius, attack_radius,
                 uav_radius_approx, obstacles, obstacle_detected, search_mode, tracking_mode,
                 map_width, map_height, get_search_target_func):
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        self.uav_health = uav_health
        self.enemy_health = enemy_health
        self.multi_current_pos = multi_current_pos
        self.enemy_pos = enemy_pos
        self.detection_radius = detection_radius
        self.attack_radius = attack_radius
        self.uav_radius_approx = uav_radius_approx
        self.obstacles = obstacles
        self.obstacle_detected = obstacle_detected # 这是外部的引用
        self.search_mode = search_mode
        self.tracking_mode = tracking_mode
        self.map_width = map_width
        self.map_height = map_height
        self._get_search_target = get_search_target_func # 传入方法引用

    def calculate_rewards(self, last_distances_to_enemies):
        rewards = np.zeros(self.num_agents)
        dones = [False] * self.num_agents
        
        # 存活的UAV数量
        num_alive_uavs = sum(1 for h in self.uav_health if h > 0)

        for i in range(self.num_agents):
            if self.uav_health[i] <= 0:
                rewards[i] = -10.0 # 死亡惩罚
                dones[i] = True
                continue

            # 1. 模式奖励 (搜索 / 追踪 / 围捕)
            if self.search_mode: # 全局搜索模式
                search_target_pos = self._get_search_target(i)
                dist_to_search_target = np.linalg.norm(self.multi_current_pos[i] - search_target_pos)
                rewards[i] += 0.5 * (1.0 - dist_to_search_target / max(self.map_width, self.map_height)) # 奖励探索未知区域

            elif self.tracking_mode[i]: # 当前无人机处于追踪模式
                nearest_enemy_idx = -1
                min_dist = float('inf')
                # 找到该智能体能探测到的最近的敌机
                for j in range(self.num_enemies):
                    if self.enemy_health[j] > 0:
                        current_dist_to_enemy = np.linalg.norm(self.multi_current_pos[i] - self.enemy_pos[j])
                        # Ensure the agent actually sees the enemy to track it
                        if current_dist_to_enemy <= self.detection_radius and current_dist_to_enemy < min_dist:
                            min_dist = current_dist_to_enemy
                            nearest_enemy_idx = j
                
                if nearest_enemy_idx != -1: # 找到了一个最近的被追踪敌机
                    optimal_track_dist_min = self.attack_radius * 1.2 # 保持在敌人攻击范围外一点
                    optimal_track_dist_max = self.detection_radius * 0.8 # 保持在探测范围的较近部分
                    tracking_reward_component = 0.0

                    if optimal_track_dist_min < min_dist < optimal_track_dist_max:
                        tracking_reward_component += 5.0 # 在最佳追踪区间的强奖励
                        # 如果从远处进入此区间，给予额外奖励
                        if last_distances_to_enemies[i][nearest_enemy_idx] != float('inf') and \
                           last_distances_to_enemies[i][nearest_enemy_idx] > min_dist and \
                           last_distances_to_enemies[i][nearest_enemy_idx] > optimal_track_dist_max:
                            tracking_reward_component += 2.0 * (last_distances_to_enemies[i][nearest_enemy_idx] - min_dist) / self.detection_radius
                    elif min_dist < optimal_track_dist_min: # 单独时靠太近
                        tracking_reward_component -= 3.0 # 较强的惩罚
                    elif min_dist > optimal_track_dist_max: # 离太远但仍在探测范围内
                        # 如果从较近位置漂移开，给予惩罚
                        if last_distances_to_enemies[i][nearest_enemy_idx] != float('inf') and \
                           last_distances_to_enemies[i][nearest_enemy_idx] < min_dist and \
                           last_distances_to_enemies[i][nearest_enemy_idx] < optimal_track_dist_max * 1.1 : 
                             tracking_reward_component -= 1.5
                    
                    rewards[i] += tracking_reward_component
            
            else: # 围捕模式 (非搜索模式，且当前无人机不处于追踪模式，则进入围捕/交战模式)
                # 对每个存活的敌机计算个体交战奖励
                for enemy_idx in range(self.num_enemies):
                    if self.enemy_health[enemy_idx] > 0:
                        current_dist_to_enemy = np.linalg.norm(self.multi_current_pos[i] - self.enemy_pos[enemy_idx])
                        last_dist_to_enemy = last_distances_to_enemies[i][enemy_idx]
                        
                        agent_can_see_enemy = (current_dist_to_enemy <= self.detection_radius)

                        if agent_can_see_enemy: # 只考虑该智能体能看到的敌人
                            # 确定智能体i是否在与敌机enemy_idx的对抗中"孤军奋战"
                            num_supporting_teammates = 0
                            support_vicinity_radius_sq = (self.detection_radius * 1.5)**2 

                            for k in range(self.num_agents):
                                if k == i or self.uav_health[k] <= 0: 
                                    continue
                                dist_sq_teammate_to_enemy = np.sum(np.square(self.multi_current_pos[k] - self.enemy_pos[enemy_idx]))
                                if dist_sq_teammate_to_enemy < support_vicinity_radius_sq:
                                    num_supporting_teammates += 1
                            
                            # 标准交战奖励 (有队友支援时)
                            if num_supporting_teammates > 0: 
                                if last_dist_to_enemy != float('inf'): 
                                    dist_change_reward = (last_dist_to_enemy - current_dist_to_enemy) / self.detection_radius 
                                    rewards[i] += 25.0 * dist_change_reward 
                                
                                if current_dist_to_enemy <= self.attack_radius:
                                    rewards[i] += 1.5
                                elif current_dist_to_enemy > self.attack_radius: 
                                    rewards[i] += 2.0  # 有支援时，在探测范围内的奖励
            
            # 2. 消灭敌机奖励
            for enemy_idx in range(self.num_enemies):
                 if self.enemy_health[enemy_idx] <= 0 and last_distances_to_enemies[i][enemy_idx] != float('inf'): 
                    if np.linalg.norm(self.multi_current_pos[i] - self.enemy_pos[enemy_idx]) < self.attack_radius * 1.5:
                        rewards[i] += 20.0 # 击杀奖励

            # 3. 碰撞惩罚 (与障碍物)
            min_dist_to_obs_surface = float('inf')
            collided_this_step = False 

            for obs_data in self.obstacle_detected[i]: 
                dist_surface = obs_data['distance_to_center'] - obs_data['radius'] - self.uav_radius_approx
                if dist_surface < min_dist_to_obs_surface:
                    min_dist_to_obs_surface = dist_surface
            
            # 优先处理实际碰撞
            if min_dist_to_obs_surface < 0: 
                rewards[i] -= 20.0 # 更大的惩罚
                collided_this_step = True
            # 增加成功避障奖励
            elif min_dist_to_obs_surface > self.uav_radius_approx * 2.0 and min_dist_to_obs_surface < self.detection_radius: 
                rewards[i] += 1.0  # 添加积极避障奖励
            # 如果没有实际碰撞，再检查是否在警告区
            elif not collided_this_step:
                warning_zone_threshold = self.uav_radius_approx * 2.0 
                if min_dist_to_obs_surface < warning_zone_threshold : 
                    penalty_factor = 1.0 - (min_dist_to_obs_surface / warning_zone_threshold)
                    rewards[i] -= 2.0 * penalty_factor 
            
            # 4. 时间惩罚 / 生存奖励
            rewards[i] -= 0.05 

        # 全局结束条件
        if all(h <= 0 for h in self.enemy_health): 
            for i in range(self.num_agents):
                if self.uav_health[i] > 0:
                    rewards[i] += 50.0 
                dones[i] = True
        
        if num_alive_uavs == 0: 
             for i in range(self.num_agents):\
                dones[i] = True
        
        return rewards, dones