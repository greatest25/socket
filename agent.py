import torch as T
from networks import ActorNetwork, CriticNetwork
import numpy as np

"""
#!MADDPG算法中的Agent类
输入：
    actor_dims：actor网络的输入维度
    critic_dims：critic网络的输入维度
    n_actions：动作空间的维度
    n_agents：代理数量
    agent_idx：当前代理的索引
    chkpt_dir：检查点保存目录
    alpha：actor网络的学习率
    beta：critic网络的学习率
    fc1：第一层全连接层的神经元数量
    fc2：第二层全连接层的神经元数量
    gamma：折扣因子
    tau：软更新参数
输出：
    action：当前代理的动作
    learn：学习函数
    save_models：保存模型函数
    load_models：加载模型函数
    update_network_parameters：更新网络参数函数
"""
class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.0001, beta=0.0002, fc1=128,
                    fc2=128, gamma=0.99, tau=0.01):
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
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
    
        # exploration
        max_noise = 0.75
        min_noise = 0.1  # 增加最小噪声值，从0.01改为0.1
        decay_rate = 0.999995
    
        noise_scale = max(min_noise, max_noise * (decay_rate ** time_step))
        noise = 2 * T.rand(self.n_actions).to(self.actor.device) - 1 # [-1,1)
        #·训练模式下添加噪声
        if not evaluate:
            noise = noise_scale * noise
        else:
            noise = 0 * noise
        
        action = actions + noise
        #·限制动作范围
        action_np = action.detach().cpu().numpy()[0]
        magnitude = np.linalg.norm(action_np)
        if magnitude > 0.04:
            action_np = action_np / magnitude * 0.04
        return action_np

    #·更新target网络参数
    def update_network_parameters(self, tau=None):
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
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.actor.save_for_inference()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()