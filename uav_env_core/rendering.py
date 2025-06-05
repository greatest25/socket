import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.image as mpimg
import matplotlib.backends.backend_agg as agg
import numpy as np
from PIL import Image

class Renderer:
    def __init__(self, map_width, map_height, num_agents, num_enemies, 
                 detection_radius, attack_radius, uav_radius_approx, enemy_radius_approx):
        self.map_width = map_width
        self.map_height = map_height
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        self.detection_radius = detection_radius
        self.attack_radius = attack_radius
        self.uav_radius_approx = uav_radius_approx
        self.enemy_radius_approx = enemy_radius_approx
        # 这些属性会在render方法中动态传入，所以这里不需要作为类属性
        # self.uav_health
        # self.multi_current_pos
        # self.multi_current_vel
        # self.enemy_pos
        # self.enemy_vel
        # self.enemy_health
        # self.enemy_detected
        # self.obstacles
        # self.history_positions

    def render(self, uav_health, multi_current_pos, multi_current_vel, 
               enemy_pos, enemy_vel, enemy_health, enemy_detected, 
               obstacles, history_positions, uav_icon_path='UAV.png', enemy_icon_path='Enemy_UAV.png'):
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
            if uav_health[i] <= 0: continue
            pos = multi_current_pos[i]
            vel = multi_current_vel[i]
            history_positions[i].append(pos.copy())
            trajectory = np.array(history_positions[i])
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
            plt.text(pos[0], pos[1] + 15, f"B{i}:{int(uav_health[i])}", color='blue', fontsize=7)

        # 敌方无人机 (红方)
        for j in range(self.num_enemies):
            if enemy_health[j] <= 0: continue
            pos = enemy_pos[j]
            vel = enemy_vel[j] #虽然不给观测，但渲染时可以用
            history_positions[self.num_agents + j].append(pos.copy())
            trajectory = np.array(history_positions[self.num_agents + j])
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.2, linewidth=0.5)

            angle_rad = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel) > 1e-3 else 0
            
            if enemy_icon is not None:
                t = transforms.Affine2D().rotate_around(0,0,angle_rad).translate(pos[0], pos[1])
                ax.imshow(enemy_icon, transform=t + ax.transData,
                           extent=[-icon_pixel_size/2, icon_pixel_size/2, -icon_pixel_size/2, icon_pixel_size/2],
                           interpolation='bicubic')
            else:
                plt.scatter(pos[0], pos[1], c='red', marker='x', s=50)
            
            if enemy_detected[j]: # 被探测到时，给个橙色外圈
                detected_indicator = plt.Circle((pos[0], pos[1]), self.enemy_radius_approx + 5, color='orange', fill=False, linewidth=1)
                ax.add_patch(detected_indicator)
            plt.text(pos[0], pos[1] - 15, f"R{j}:{int(enemy_health[j])}", color='red', fontsize=7)
            
        # 障碍物
        for obs_item in obstacles:
            color = 'dimgray' if obs_item['is_dynamic'] else 'gray' # Dynamic obstacles are darker
            circle = plt.Circle(obs_item['position'], obs_item['radius'], color=color, alpha=0.6)
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