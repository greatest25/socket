import numpy as np

class MultiAgentReplayBuffer:
    """
    多智能体环境的经验回放缓冲区
    
    用于存储和采样多智能体环境中的交互经验，支持MADDPG算法的训练
    """
    def __init__(self, max_size, critic_dims, actor_dims, 
            n_actions, n_agents, batch_size):
        """
        初始化经验回放缓冲区
        
        参数:
            max_size: 缓冲区最大容量
            critic_dims: 评论家网络的输入维度
            actor_dims: 每个智能体的演员网络输入维度列表
            n_actions: 每个智能体的动作空间维度
            n_agents: 智能体数量
            batch_size: 批量采样大小
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        # 为全局状态、奖励和终止标志创建内存
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        """
        初始化每个智能体的Actor网络相关内存
        
        为每个智能体创建观察、下一个观察和动作的存储空间
        """
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))


    def store_transition(self, raw_obs, state, action, reward,raw_obs_, state_, done):
        """
        存储一个交互经验到缓冲区
        
        参数:
            raw_obs: 每个智能体的原始观察列表
            state: 全局状态
            action: 每个智能体的动作列表
            reward: 每个智能体的奖励列表
            raw_obs_: 每个智能体的下一个原始观察列表
            state_: 下一个全局状态
            done: 每个智能体的终止标志列表
        """
        # 注释说明：如果内存容量已满并且计数器大于0，则重新初始化actor内存
        # 但实际上这段代码被注释掉了，因为这会导致critic和actor内存不同步的问题
        
        #if self.mem_cntr % self.mem_size == 0 and self.mem_cntr > 0:
        #    self.init_actor_memory()
        
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        """
        从缓冲区随机采样一批经验
        
        返回:
            采样的经验批次，包括每个智能体的状态、动作、奖励、下一个状态和终止标志
        """
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, \
               actor_new_states, states_, terminal

    def ready(self):
        """
        检查缓冲区是否已准备好进行采样
        
        返回:
            布尔值，表示缓冲区中的样本数是否达到批量大小
        """
        if self.mem_cntr >= self.batch_size:
            return True
