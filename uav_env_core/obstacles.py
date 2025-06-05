import numpy as np
import random

class ObstacleHandler:
    def __init__(self, map_width, map_height, obstacle_max_speed):
        self.map_width = map_width
        self.map_height = map_height
        self.obstacle_max_speed = obstacle_max_speed

    def _create_obstacle(self, is_dynamic=False):
        # 避免在出生点附近生成障碍物
        spawn_protection_x = self.map_width / 5.0
        spawn_protection_y = self.map_height / 5.0
        
        while True:
            pos_x = random.uniform(0, self.map_width)
            pos_y = random.uniform(0, self.map_height)
            # 检查是否在蓝方出生区 (我方无人机现在出生在右下角)
            is_in_our_spawn = (pos_x > self.map_width - spawn_protection_x and pos_y > self.map_height - spawn_protection_y)
            # 检查是否在红方出生区 (敌机出生在左上角)
            is_in_enemy_spawn = (pos_x < spawn_protection_x and pos_y < spawn_protection_y)
            
            if not (is_in_our_spawn or is_in_enemy_spawn):
                break
        
        radius = 80.0 # 固定半径为80像素
        
        velocity = np.zeros(2)
        if is_dynamic:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = random.uniform(self.obstacle_max_speed * 0.5, self.obstacle_max_speed)
            velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])

        return {
            'position': np.array([pos_x, pos_y]),
            'velocity': velocity, 
            'radius': radius,
            'is_dynamic': is_dynamic
        }