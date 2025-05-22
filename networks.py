import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    """
    MADDPG算法中的评论家网络
    
    用于评估状态-动作对的价值，接收所有智能体的状态和动作作为输入
    """
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        """
        初始化评论家网络
        
        参数:
            beta: 学习率
            input_dims: 状态空间维度
            fc1_dims: 第一个全连接层的神经元数量
            fc2_dims: 第二个全连接层的神经元数量
            n_agents: 智能体数量
            n_actions: 每个智能体的动作空间维度
            name: 网络名称，用于保存检查点
            chkpt_dir: 检查点保存目录
        """
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        # 网络结构：状态和动作连接后输入，经过两个全连接层，输出Q值
        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        # 优化器和学习率调度器
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.33)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        """
        前向传播计算Q值
        
        参数:
            state: 环境状态
            action: 所有智能体的动作
            
        返回:
            q: 状态-动作对的Q值
        """
        # 将状态和动作连接后输入网络
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        """
        保存网络参数到检查点文件
        """
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)    
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        """
        从检查点文件加载网络参数
        """
        self.load_state_dict(T.load(self.chkpt_file, map_location=self.device))


class ActorNetwork(nn.Module):
    """
    MADDPG算法中的演员网络
    
    用于根据当前状态选择最优动作
    """
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        """
        初始化演员网络
        
        参数:
            alpha: 学习率
            input_dims: 状态空间维度
            fc1_dims: 第一个全连接层的神经元数量
            fc2_dims: 第二个全连接层的神经元数量
            n_actions: 动作空间维度
            name: 网络名称，用于保存检查点
            chkpt_dir: 检查点保存目录
        """
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        # 网络结构：状态输入，经过两个全连接层，输出动作
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        # 优化器和学习率调度器
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        """
        前向传播计算动作
        
        参数:
            state: 环境状态
            
        返回:
            pi: 选择的动作，范围在[-1,1]之间
        """
        # 使用Leaky ReLU激活函数
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        # 使用Softsign激活函数将输出限制在[-1,1]范围内
        pi = nn.Softsign()(self.pi(x)) # [-1,1]

        return pi

    def save_checkpoint(self):
        """
        保存网络参数到检查点文件
        """
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        """
        从检查点文件加载网络参数
        """
        self.load_state_dict(T.load(self.chkpt_file, map_location=self.device))

