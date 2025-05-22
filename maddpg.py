import os
import torch as T
import torch.nn.functional as F
from agent import Agent
# from torch.utils.tensorboard import SummaryWriter

class MADDPG:
    """
    多智能体深度确定性策略梯度(MADDPG)算法实现
    
    管理多个DDPG智能体，实现集中式训练和分布式执行的多智能体强化学习算法
    """
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.02, fc1=128, 
                 fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        """
        初始化MADDPG算法
        
        参数:
            actor_dims: 每个智能体的Actor网络输入维度列表
            critic_dims: Critic网络的输入维度
            n_agents: 智能体数量
            n_actions: 每个智能体的动作空间维度
            scenario: 场景名称，用于构建检查点目录
            alpha: Actor网络的学习率
            beta: Critic网络的学习率
            fc1: 第一个全连接层的神经元数量
            fc2: 第二个全连接层的神经元数量
            gamma: 折扣因子
            tau: 目标网络软更新系数
            chkpt_dir: 模型检查点保存目录
        """
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        # self.writer = SummaryWriter(log_dir=os.path.join(chkpt_dir, 'logs'))

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        """
        保存所有智能体的模型检查点
        """
        print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    def load_checkpoint(self):
        """
        加载所有智能体的模型检查点
        """
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, time_step, evaluate):
        """
        为所有智能体选择动作
        
        参数:
            raw_obs: 每个智能体的观察列表
            time_step: 当前时间步，用于计算探索噪声
            evaluate: 是否处于评估模式
            
        返回:
            actions: 所有智能体的动作列表
        """
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx], time_step, evaluate)
            actions.append(action)
        return actions

    def learn(self, memory, total_steps):
        """
        MADDPG算法的学习过程
        
        实现集中式训练，每个智能体的Critic网络可以访问所有智能体的信息
        
        参数:
            memory: 经验回放缓冲区
            total_steps: 当前总训练步数，用于记录日志
        """
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        # 将numpy数组转换为PyTorch张量并移动到适当的设备
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        old_agents_actions = []
    
        # 获取所有智能体的目标动作和当前动作
        for agent_idx, agent in enumerate(self.agents):

            new_states = T.tensor(actor_new_states[agent_idx], 
                                dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])

        # 将所有智能体的动作连接起来
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        # 更新每个智能体的网络
        for agent_idx, agent in enumerate(self.agents):
            # 计算目标Q值
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                target = rewards[:,agent_idx] + (1-dones[:,0].int())*agent.gamma*critic_value_

            # 计算当前Q值和Critic损失
            critic_value = agent.critic.forward(states, old_actions).flatten()
            
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            # 计算Actor损失并更新
            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            oa = old_actions.clone()
            # 替换当前智能体的动作为Actor网络输出
            oa[:,agent_idx*self.n_actions:agent_idx*self.n_actions+self.n_actions] = agent.actor.forward(mu_states)            
            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

            # TensorBoard日志记录（已注释）
            # self.writer.add_scalar(f'Agent_{agent_idx}/Actor_Loss', actor_loss.item(), total_steps)
            # self.writer.add_scalar(f'Agent_{agent_idx}/Critic_Loss', critic_loss.item(), total_steps)

            # for name, param in agent.actor.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Actor_Gradients/{name}', param.grad, total_steps)
            # for name, param in agent.critic.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Critic_Gradients/{name}', param.grad, total_steps)
            
        # 软更新所有智能体的目标网络
        for agent in self.agents:    
            agent.update_network_parameters()
