import numpy as np
import math

# 1 simulate lidar
def update_lasers(pos, obs_pos, r, L, num_lasers, bound):
    """
    更新激光传感器数据并检测碰撞
    
    参数:
        pos: 无人机当前位置坐标 [x, y]
        obs_pos: 障碍物位置坐标 [x, y]
        r: 障碍物半径
        L: 激光传感器最大探测距离
        num_lasers: 激光束数量
        bound: 环境边界长度
        
    返回:
        laser_lengths: 各方向激光长度列表
        isInObs: 是否与障碍物或边界发生碰撞
    """
    # 计算无人机到障碍物的距离
    distance_to_obs = np.linalg.norm(np.array(pos) - np.array(obs_pos))
    # 判断是否在障碍物内部或超出边界
    isInObs = distance_to_obs < r \
                or pos[0] < 0 \
                or pos[0] > bound \
                or pos[1] < 0 \
                or pos[1] > bound
    
    # 如果碰撞，返回全零激光数据
    if isInObs:
        return [0.0] * num_lasers, isInObs
    
    # 生成均匀分布的激光角度
    angles = np.linspace(0, 2 * np.pi, num_lasers, endpoint=False)
    laser_lengths = [L] * num_lasers
    
    # 检查每个激光与障碍物的交点
    for i, angle in enumerate(angles):
        intersection_dist = check_obs_intersection(pos, angle, obs_pos, r, L)
        if laser_lengths[i] > intersection_dist:
            laser_lengths[i] = intersection_dist
    
    # 检查每个激光与边界墙的交点
    for i, angle in enumerate(angles):
        wall_dist = check_wall_intersection(pos, angle, bound, L)
        if laser_lengths[i] > wall_dist:
            laser_lengths[i] = wall_dist
    
    return laser_lengths, isInObs

def check_obs_intersection(start_pos, angle, obs_pos, r, max_distance):
    """
    计算激光与圆形障碍物的交点距离
    
    参数:
        start_pos: 激光起始点坐标 [x, y]
        angle: 激光方向角度（弧度）
        obs_pos: 障碍物中心坐标 [x, y]
        r: 障碍物半径
        max_distance: 激光最大探测距离
        
    返回:
        交点距离，如无交点则返回最大探测距离
    """
    ox = obs_pos[0]
    oy = obs_pos[1]

    # 计算激光终点
    end_x = start_pos[0] + max_distance * np.cos(angle)
    end_y = start_pos[1] + max_distance * np.sin(angle)

    # 计算射线方向向量
    dx = end_x - start_pos[0]
    dy = end_y - start_pos[1]
    # 计算起点到障碍物中心的向量
    fx = start_pos[0] - ox
    fy = start_pos[1] - oy

    # 求解二次方程 at^2 + bt + c = 0
    # 该方程表示射线参数方程与圆的交点
    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = (fx**2 + fy**2) - r**2

    # 计算判别式
    discriminant = b**2 - 4 * a * c

    # 如果判别式非负，则有交点
    if discriminant >= 0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        # 检查交点是否在激光范围内
        if 0 <= t1 <= 1:
            return t1 * max_distance
        if 0 <= t2 <= 1:
            return t2 * max_distance

    # 无交点，返回最大距离
    return max_distance

def check_wall_intersection(start_pos, angle, bound, L):
    """
    计算激光与边界墙的交点距离
    
    参数:
        start_pos: 激光起始点坐标 [x, y]
        angle: 激光方向角度（弧度）
        bound: 环境边界长度
        L: 激光最大探测距离
        
    返回:
        交点距离，如无交点则返回最大探测距离
    """
    # 计算激光方向的正弦和余弦值
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    L_ = L
    
    # 检查与上边界(y = bound)的交点
    if sin_theta > 0:  
        L_ = min(L_, abs((bound - start_pos[1]) / sin_theta))
    
    # 检查与下边界(y = 0)的交点
    if sin_theta < 0:  
        L_ = min(L_, abs(start_pos[1] / -sin_theta))

    # 检查与右边界(x = bound)的交点
    if cos_theta > 0: 
        L_ = min(L_, abs((bound - start_pos[0]) / cos_theta))
    
    # 检查与左边界(x = 0)的交点
    if cos_theta < 0: 
        L_ = min(L_, abs(start_pos[0] / -cos_theta))

    return L_

def cal_triangle_S(p1, p2, p3):
    """
    计算三角形面积
    
    使用叉积计算由三个点组成的三角形的面积
    
    参数:
        p1, p2, p3: 三角形三个顶点的坐标 [x, y]
        
    返回:
        三角形面积，如果三点共线则返回0
    """
    # 使用叉积公式计算三角形面积
    S = abs(0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])))
    # 处理数值精度问题，如果面积接近0则返回0
    if math.isclose(S, 0.0, abs_tol=1e-9):
        return 0.0
    else:
        return S