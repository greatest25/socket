import numpy as np
import random

class UAVInteractionHandler:
    def __init__(self, num_agents, num_enemies, uav_health, enemy_health, multi_current_pos, enemy_pos,
                 detection_radius, attack_radius, uav_radius_approx, obstacles, obstacle_detected, search_mode, tracking_mode):
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
        self.obstacle_detected = obstacle_detected
        self.search_mode = search_mode # 全局搜索模式
        self.tracking_mode = tracking_mode # 各个智能体的追踪模式

    def update_detections_and_obstacles(self):
        # 重置探测状态
        self.enemy_detected = [False] * self.num_enemies
        self.obstacle_detected_internal = [[] for _ in range(self.num_agents)] # 使用内部变量，方便更新外部引用
        
        # 更新敌机探测状态 和 障碍物探测状态
        for i in range(self.num_agents):
            if self.uav_health[i] <= 0:
                continue
            
            agent_pos = self.multi_current_pos[i]
            # 探测敌机
            agent_detects_any_enemy = False  # 当前智能体是否探测到任何敌人
            
            for j in range(self.num_enemies):
                if self.enemy_health[j] > 0:
                    distance_to_enemy = np.linalg.norm(agent_pos - self.enemy_pos[j])
                    if distance_to_enemy <= self.detection_radius:
                        self.enemy_detected[j] = True
                        agent_detects_any_enemy = True

            # 更新该智能体的追踪模式
            if agent_detects_any_enemy:
                self.tracking_mode[i] = True  # 探测到敌人，进入追踪模式
            elif self.tracking_mode[i]:
                # 如果已经在追踪，设置一个"记忆"，让智能体继续追踪一段时间
                # 这可以模拟无人机对目标的短期记忆
                if random.random() > 0.05:  # 95%的概率保持追踪状态
                    self.tracking_mode[i] = True
                else:
                    self.tracking_mode[i] = False
            else:
                self.tracking_mode[i] = False
            
            # 探测障碍物
            for obs_item in self.obstacles:
                obs_pos = obs_item['position']
                obs_rad = obs_item['radius']
                distance_to_obstacle_center = np.linalg.norm(agent_pos - obs_pos)
                # 如果UAV的探测范围边缘触及或穿过障碍物
                if distance_to_obstacle_center <= self.detection_radius + obs_rad:
                    self.obstacle_detected_internal[i].append({
                        'absolute_pos': obs_pos.copy(),         # 障碍物中心绝对位置
                        'radius': obs_rad,                      # 障碍物绝对半径
                        'distance_to_center': distance_to_obstacle_center # UAV到障碍物中心的绝对距离
                    })
        
        # 更新全局搜索模式
        if any(self.enemy_detected):
            self.search_mode = False  # 至少有一个敌人被探测到，全局不再是搜索模式
        else:
            # 如果所有敌机都未被探测到 (可能因为超出范围或已被击毁)
            # 并且还有存活的敌机，则继续搜索
            if any(h > 0 for h in self.enemy_health):
                 self.search_mode = True
            else: # 所有敌机都被击毁
                 self.search_mode = False

        # 更新外部引用，以便环境类可以访问最新的探测数据
        self.obstacle_detected[:] = self.obstacle_detected_internal[:] 

    def process_attacks(self):
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