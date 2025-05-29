import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
参数
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
"""

"""
Critic Network
learn:
    1. 从经验回放中随机采样一批数据
    2. 计算目标Q值
    3. 计算当前Q值
    4. 计算损失函数
    5. 反向传播
    6. 更新网络参数
    7. 软更新目标网络参数
    8. 学习率衰减
    9. 保存模型
    10. 加载模型
    11. 保存模型用于推理
    12. 加载模型用于推理
"""
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.33)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)    
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file, map_location=T.device('cpu')))

    def save_for_inference(self):
        params = {}
        for name, param in self.named_parameters():
            params[name] = param.detach().cpu().numpy().tolist()
        
        import json
        # 修改保存路径为当前目录
        import os
        filename = os.path.basename(self.chkpt_file) + '_inference.json'
        with open(filename, 'w') as f:
            json.dump(params, f)

"""
Actor Network
learn:
    1. 从经验回放中随机采样一批数据
    2. 计算当前动作
    3. 计算损失函数
    4. 反向传播
    5. 更新网络参数
    6. 软更新目标网络参数
    7. 学习率衰减
    8. 保存模型
    9. 加载模型
    10. 保存模型用于推理
    11. 加载模型用于推理
"""
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        pi = nn.Softsign()(self.pi(x)) # [-1,1]

        return pi

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file, map_location=self.device))

    def save_for_inference(self):
        params = {}
        for name, param in self.named_parameters():
            params[name] = param.detach().cpu().numpy().tolist()
        
        import json
        # 修改保存路径为当前目录
        import os
        filename = os.path.basename(self.chkpt_file) + '_inference.json'
        with open(filename, 'w') as f:
            json.dump(params, f)
