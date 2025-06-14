import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import shutil
import tempfile
from datetime import datetime
from maddpg import MADDPG
from qt5_sim_env import Qt5SimUAVEnv

class ModelTester:
    """
    模型测试类，用于测试、评估和保存最佳模型
    """
    def __init__(self, scenario_name='UAV_Round_up', 
                 chkpt_dir='tmp/maddpg/', 
                 results_dir='evaluation_results',
                 animations_dir='animations'):
        """
        初始化模型测试器
        
        参数:
            scenario_name: 场景名称，用于保存和加载模型
            chkpt_dir: 检查点保存目录
            results_dir: 评估结果保存目录
            animations_dir: 动画保存目录
        """
        self.scenario_name = scenario_name
        self.chkpt_dir = chkpt_dir
        
        # 创建结果和动画目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"{results_dir}_{timestamp}"
        self.animations_dir = f"{animations_dir}_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.animations_dir, exist_ok=True)
        
        # 初始化评估指标
        self.metrics = {
            'episode_rewards': [],
            'episode_steps': [],
            'enemies_destroyed': [],
            'agents_lost': [],
            'detection_times': [],  # 首次探测到敌人的时间
            'capture_times': [],    # 从探测到击毁的时间
            'success_rate': 0.0     # 成功率（所有敌机被摧毁）
        }
    
    def save_image(self, env_render, filename):
        """保存环境渲染图像"""
        image = Image.fromarray(env_render, 'RGBA')
        image = image.convert('RGB')
        image.save(filename)
    
    def save_animation(self, env, maddpg_agents, save_path, animation_steps=500, fps=10):
        """生成并保存动画"""
        print(f"正在生成动画，步数: {animation_steps}, FPS: {fps}...")
        
        # 创建临时目录存储帧
        temp_dir = os.path.join(os.path.dirname(save_path), 'temp_frames_for_gif')
        os.makedirs(temp_dir, exist_ok=True)
        
        frames = []
        
        # 重置环境
        obs = env.reset()
        
        try:
            # 生成动画帧
            for step in range(animation_steps):
                # 渲染当前帧并保存
                env_render = env.render() # 直接调用render获取图像
                if env_render is None:
                    print("Warning: env.render() returned None. Skipping frame.")
                    continue
                
                # 将图像数组保存到临时文件
                frame_path = os.path.join(temp_dir, f"frame_{step:04d}.png")
                self.save_image(env_render, frame_path) # 使用我们已有的save_image
                frames.append(frame_path)
                
                # 选择动作（评估模式）
                actions = maddpg_agents.choose_action(obs, 0, evaluate=True)
                
                # 执行动作
                obs, _, _ = env.step(actions)       
                        
                # 如果所有敌机或所有UAV都被摧毁，提前结束
                if all(h <= 0 for h in env.enemy_health) or all(h <= 0 for h in env.uav_health):
                    print(f"动画提前结束，步骤 {step}/{animation_steps}")
                    break
            
            # 使用imageio创建GIF
            if frames:
                print("正在从帧生成GIF...")
                with imageio.get_writer(save_path, mode='I', fps=fps) as writer:
                    for frame_path in frames:
                        image = imageio.imread(frame_path)
                        writer.append_data(image)
                print(f"动画已保存到: {save_path}")
            else:
                print("没有生成任何帧，无法创建动画。")

        finally:
            # 清理临时文件
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def evaluate_model(self, env, maddpg_agents, n_episodes=5, max_steps=500):
        """
        评估模型性能
        
        参数:
            env: 环境实例
            maddpg_agents: MADDPG代理实例
            n_episodes: 评估回合数
            max_steps: 每回合最大步数
            
        返回:
            metrics: 评估指标字典
        """
        print(f"开始评估模型，共{n_episodes}回合，每回合最大{max_steps}步...")
        
        # 重置评估指标
        self.metrics = {
            'episode_rewards': [],
            'episode_steps': [],
            'enemies_destroyed': [],
            'agents_lost': [],
            'detection_times': [],
            'capture_times': [],
            'success_rate': 0.0
        }
        
        successful_episodes = 0
        
        for ep in range(n_episodes):
            print(f"评估回合 {ep+1}/{n_episodes}")
            obs = env.reset()
            episode_reward = []
            
            # 追踪首次探测和围捕时间
            first_detection = None
            enemies_destroyed_time = {}
            
            for step in range(max_steps):
                # 选择动作（评估模式）
                actions = maddpg_agents.choose_action(obs, 0, evaluate=True)
                
                # 执行动作
                obs_, rewards, dones = env.step(actions)
                
                # 记录奖励
                episode_reward.append(sum(rewards))
                
                # 检查是否首次探测到敌人
                if first_detection is None and not env.search_mode:
                    first_detection = step
                
                # 检查是否有敌机被摧毁
                for i in range(env.num_enemies):
                    if i not in enemies_destroyed_time and env.enemy_health[i] <= 0:
                        enemies_destroyed_time[i] = step
                
                # 更新观察
                obs = obs_
                
                # 如果所有敌机或所有UAV都被摧毁，提前结束
                if all(h <= 0 for h in env.enemy_health) or all(h <= 0 for h in env.uav_health):
                    break
            
            # 记录本回合指标
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['episode_steps'].append(step + 1)
            
            # 计算摧毁的敌机数量
            enemies_destroyed = sum(1 for h in env.enemy_health if h <= 0)
            self.metrics['enemies_destroyed'].append(enemies_destroyed)
            
            # 计算损失的UAV数量
            agents_lost = sum(1 for h in env.uav_health if h <= 0)
            self.metrics['agents_lost'].append(agents_lost)
            
            # 记录探测时间
            if first_detection is not None:
                self.metrics['detection_times'].append(first_detection)
            
            # 记录从探测到击毁的时间
            if enemies_destroyed > 0 and first_detection is not None:
                avg_capture_time = sum(enemies_destroyed_time.values()) / len(enemies_destroyed_time) - first_detection
                self.metrics['capture_times'].append(avg_capture_time)
            
            # 判断任务是否成功（所有敌机被摧毁）
            if all(h <= 0 for h in env.enemy_health):
                successful_episodes += 1
        
        # 计算成功率
        self.metrics['success_rate'] = successful_episodes / n_episodes
        
        # 保存评估结果
        self._save_evaluation_results()
        
        return self.metrics
    
    def _save_evaluation_results(self):
        """保存评估结果到文件"""
        with open(f"{self.results_dir}/evaluation_results.txt", 'w') as f:
            f.write("=== 评估结果汇总 ===\n")
            f.write(f"总回合数: {len(self.metrics['episode_steps'])}\n")
            f.write(f"平均回合步数: {np.mean(self.metrics['episode_steps']):.2f}\n")
            f.write(f"平均摧毁敌机数: {np.mean(self.metrics['enemies_destroyed']):.2f}\n")
            f.write(f"平均损失我方UAV数: {np.mean(self.metrics['agents_lost']):.2f}\n")
            
            # 计算平均回合奖励
            avg_reward = np.mean([sum(r) for r in self.metrics['episode_rewards']])
            f.write(f"平均回合奖励: {avg_reward:.2f}\n")
            
            f.write(f"任务成功率: {self.metrics['success_rate']*100:.2f}%\n")
            
            if self.metrics['detection_times']:
                f.write(f"平均首次探测时间: {np.mean(self.metrics['detection_times']):.2f} 步\n")
            
            if self.metrics['capture_times']:
                f.write(f"平均围捕时间: {np.mean(self.metrics['capture_times']):.2f} 步\n")
        
        print(f"评估结果已保存到 {self.results_dir}/evaluation_results.txt")
    
    def test_best_model(self, env, maddpg_agents, episode, avg_score):
        """
        测试最佳模型并保存动画
        
        参数:
            env: 环境实例
            maddpg_agents: MADDPG代理实例
            episode: 当前回合数
            avg_score: 平均分数
        """
        print(f"正在测试最佳模型 (回合 {episode}, 分数 {avg_score:.2f})...")
        
        # 评估模型
        self.evaluate_model(env, maddpg_agents)
        
        # 生成并保存动画
        animation_path = os.path.join(self.animations_dir, f'best_model_episode_{episode}_score_{avg_score:.1f}.gif')
        self.save_animation(env, maddpg_agents, animation_path)
        
        # 保存模型推理文件
        self._save_inference_models(maddpg_agents)
        
        return animation_path
    
    def _save_inference_models(self, maddpg_agents):
        """保存模型推理文件"""
        print("正在保存推理模型...")
        for i, agent in enumerate(maddpg_agents.agents):
            try:
                agent.actor.save_for_inference()
                print(f"智能体 {i} 推理模型保存成功")
            except Exception as e:
                print(f"保存智能体 {i} 推理模型时出错: {e}")