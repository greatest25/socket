import torch as T
from networks import ActorNetwork, CriticNetwork
import numpy as np

class Agent:
    """
    MADDPG算法中的单个智能体类
    
    实现了智能体的初始化、动作选择、网络参数更新以及模型的保存和加载功能
    """
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.0001, beta=0.0002, fc1=128,
                    fc2=128, gamma=0.99, tau=0.01):
        """
        初始化智能体
        
        参数:
            actor_dims: Actor网络的输入维度
            critic_dims: Critic网络的输入维度
            n_actions: 动作空间的维度
            n_agents: 环境中智能体的总数
            agent_idx: 当前智能体的索引
            chkpt_dir: 模型检查点保存目录
            alpha: Actor网络的学习率
            beta: Critic网络的学习率
            fc1: 第一个全连接层的神经元数量
            fc2: 第二个全连接层的神经元数量
            gamma: 折扣因子
            tau: 目标网络软更新系数
        """
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)
        

    def choose_action(self, observation, time_step, evaluate=False):
        """
        根据当前观察选择动作
        
        参数:
            observation: 环境观察状态
            time_step: 当前时间步，用于计算探索噪声
            evaluate: 是否处于评估模式，评估模式下不添加探索噪声
            
        返回:
            action_np: 选择的动作，numpy数组格式
        """
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)

        # exploration - 添加探索噪声
        max_noise = 0.75
        min_noise = 0.01
        decay_rate = 0.999995

        noise_scale = max(min_noise, max_noise * (decay_rate ** time_step))
        noise = 2 * T.rand(self.n_actions).to(self.actor.device) - 1 # [-1,1)范围内的随机噪声
        if not evaluate:
            noise = noise_scale * noise
        else:
            noise = 0 * noise
        
        action = actions + noise
        action_np = action.detach().cpu().numpy()[0]
        # 限制动作幅度，确保不超过最大值0.04
        magnitude = np.linalg.norm(action_np)
        if magnitude > 0.04:
            action_np = action_np / magnitude * 0.04
        return action_np

    def update_network_parameters(self, tau=None):
        """
        更新目标网络参数
        
        使用软更新方式更新目标Actor和目标Critic网络的参数
        
        参数:
            tau: 软更新系数，如果为None则使用默认值
        """
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        """
        保存所有网络模型到检查点文件
        """
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        """
        从检查点文件加载所有网络模型
        """
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

