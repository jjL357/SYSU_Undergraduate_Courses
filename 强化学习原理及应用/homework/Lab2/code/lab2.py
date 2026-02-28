import numpy as np
import random
import matplotlib.pyplot as plt

# CliffWalk 环境类
class CliffWalkEnv:
    def __init__(self):
        # 环境的宽度和高度
        self.width = 12
        self.height = 4
        # 设置起点和目标点
        self.start = (3, 0)
        self.goal = (3, 11)
        # 悬崖的位置（从(0,1)到(0,10)的区域）
        self.cliff = [(3, x) for x in range(1, 11)]
        # 当前状态（起始位置）
        self.state = self.start
        # 可执行的动作：上、下、左、右
        self.actions = ['up', 'down', 'left', 'right']
    
    # 重置环境状态，返回起始位置
    def reset(self):
        self.state = self.start
        return self.state
    
    # 执行一个动作并返回下一个状态和奖励
    def step(self, action):
        x, y = self.state
        # 根据动作更新位置
        if action == 'up':
            next_state = (max(0, x-1), y)
        elif action == 'down':
            next_state = (min(self.height-1, x+1), y)
        elif action == 'left':
            next_state = (x, max(0, y-1))
        elif action == 'right':
            next_state = (x, min(self.width-1, y+1))
        
        # 如果到达了悬崖，重置到起始位置并返回负奖励
        if next_state in self.cliff:
            next_state = self.start
            reward = -100  # 悬崖的惩罚值
        # 如果到达了目标点，奖励为0
        elif next_state == self.goal:
            reward = 0
        # 否则每步的默认奖励是-1
        else:
            reward = -1
        
        self.state = next_state
        return next_state, reward

# SARSA 算法
class SARSA:
    def __init__(self, env, alpha=0.5, gamma=0.99, epsilon=0.05):
        # 初始化 SARSA 参数
        self.env = env
        self.alpha = alpha   # 学习率
        self.gamma = gamma   # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.q_table = np.zeros((env.height, env.width, len(env.actions)))  # 初始化 Q 表
        self.actions = env.actions
    
    # 选择动作，采用 epsilon-greedy 策略
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # 探索随机动作
        else:
            x, y = state
            action_values = self.q_table[x, y, :]  # 获取当前状态的所有动作值
            return self.actions[np.argmax(action_values)]  # 选择最大 Q 值的动作
    
    # 更新 Q 表，SARSA 更新公式
    def update(self, state, action, reward, next_state, next_action):
        x, y = state
        next_x, next_y = next_state
        action_idx = self.actions.index(action)  # 当前动作的索引
        next_action_idx = self.actions.index(next_action)  # 下一步动作的索引
        
        q_predict = self.q_table[x, y, action_idx]
        q_target = reward + self.gamma * self.q_table[next_x, next_y, next_action_idx]
        
        # 更新 Q 表
        self.q_table[x, y, action_idx] += self.alpha * (q_target - q_predict)

# Q-learning 算法
class QLearning:
    def __init__(self, env, alpha=0.5, gamma=0.99, epsilon=0.05):
        # 初始化 Q-learning 参数
        self.env = env
        self.alpha = alpha   # 学习率
        self.gamma = gamma   # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.q_table = np.zeros((env.height, env.width, len(env.actions)))  # 初始化 Q 表
        self.actions = env.actions
    
    # 选择动作，采用 epsilon-greedy 策略
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # 探索随机动作
        else:
            x, y = state
            action_values = self.q_table[x, y, :]  # 获取当前状态的所有动作值
            return self.actions[np.argmax(action_values)]  # 选择最大 Q 值的动作
    
    # 更新 Q 表，Q-learning 更新公式
    def update(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        action_idx = self.actions.index(action)  # 当前动作的索引
        
        q_predict = self.q_table[x, y, action_idx]
        # Q-learning 采用最大 Q 值
        q_target = reward + self.gamma * np.max(self.q_table[next_x, next_y, :])
        
        # 更新 Q 表
        self.q_table[x, y, action_idx] += self.alpha * (q_target - q_predict)

# 运行CliffWalk
def run_agent(env, agent, episodes=1000):
    path = []  # 用于记录每回合的路径
    final_path = []  # 最终路径
    rewards = []  # 记录每一回合的累计奖励
    for episode in range(episodes):
        state = env.reset()  # 重置环境，获取初始状态
        action = agent.choose_action(state)  # 根据当前状态选择动作
        episode_reward = 0  # 每回合的累计奖励
        episode_path = []  # 存储每一回合的路径
        while state != env.goal:  # 直到到达目标
            episode_path.append(state)  # 将当前状态加入路径
            next_state, reward = env.step(action)  # 执行动作，获得下一个状态和奖励
            episode_reward += reward  # 累加奖励
            
            # 根据不同算法更新参数
            if isinstance(agent, SARSA):
                next_action = agent.choose_action(next_state)  # 根据下一个状态选择动作
                agent.update(state, action, reward, next_state, next_action)  # 更新 Q 表
                action = next_action  # 更新当前动作
            elif isinstance(agent, QLearning):
                agent.update(state, action, reward, next_state)  # 更新 Q 表
                action = agent.choose_action(next_state)  # 选择下一个动作
            
            state = next_state  # 更新当前状态
        
        episode_path.append(env.goal)  # 将目标点加入路径
        path.append(episode_path)  # 保存当前回合的路径
        rewards.append(episode_reward)  # 记录当前回合的奖励
    
    # 获取最后一回合的路径，即从起点到目标的路径
    if path:
        final_path = path[-1]
    return final_path, rewards

# 绘制迷宫和路径
def plot_maze(final_path, cliff, start, goal, width=12, height=4):
    # 创建一个迷宫图
    maze = np.ones((height, width))  # 初始化迷宫，每个格子设置为 1（表示墙）
    
    # 将悬崖的位置设置为 -1
    for x, y in cliff:
        maze[x, y] = -1  # 设置悬崖
    
    # 将起点设置为 0
    maze[start[0], start[1]] = 2  # 起点为 2
    
    # 将目标点设置为 3
    maze[goal[0], goal[1]] = 3  # 目标点为 3
    
    # 将路径设置为 0.5 或者其他数字
    for x, y in final_path:
        if maze[x, y] != 2 and maze[x, y] != 3 and maze[x, y] != -1:
            maze[x, y] = 0.5  # 设置路径为 0.5，避免与起点和目标点冲突
    
    # 绘制迷宫图
    fig, ax = plt.subplots(figsize=(12, 4))  # 创建图形
    cax = ax.imshow(maze, cmap='coolwarm', interpolation='nearest')  # 显示迷宫
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)  # 设置横坐标网格
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)  # 设置纵坐标网格
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)  # 设置网格线
    
    # 标注起点、目标点
    ax.scatter(start[1], start[0], color='blue', s=100, label='Start', zorder=10)
    ax.scatter(goal[1], goal[0], color='green', s=100, label='Goal', zorder=10)
    
    # 绘制悬崖
    cliff_x, cliff_y = zip(*cliff)
    ax.scatter(cliff_y, cliff_x, color='red', s=100, label='Cliff', zorder=5)
    
    # 绘制最终路径
    path_x, path_y = zip(*final_path)
    ax.scatter(path_y, path_x, color='orange', s=30, label='Path', zorder=3)
    
    # 设置图标题和标签
    ax.set_title('Cliff Walk Environment - Maze Representation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # 添加图例
    ax.legend(loc='best')
    
    # 显示迷宫图
    plt.show()


# 创建环境
env = CliffWalkEnv()

# 创建SARSA和Q-learning
sarsa_agent = SARSA(env)
qlearning_agent = QLearning(env)

# 运行SARSA
print("Running SARSA...")
sarsa_final_path, sarsa_rewards = run_agent(env, sarsa_agent, episodes=1000)

# 可视化SARSA的最终路径
plot_maze(sarsa_final_path, env.cliff, env.start, env.goal)

# 运行Q-learning
print("Running Q-learning...")
qlearning_final_path, qlearning_rewards = run_agent(env, qlearning_agent, episodes=1000)

# 可视化Q-learning的最终路径
plot_maze(qlearning_final_path, env.cliff, env.start, env.goal)

# 绘制奖励对比图
plt.figure(figsize=(12, 6))
episodes = range(100, 1000)  # 以回合数作为x轴
plt.plot(episodes, sarsa_rewards[100:], label="SARSA", color="blue")
plt.plot(episodes, qlearning_rewards[100:], label="Q-learning", color="green")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Episode Reward Comparison: SARSA vs Q-learning")
plt.legend()
plt.show()