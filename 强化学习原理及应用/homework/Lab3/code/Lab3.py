import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 随机生成迷宫 (1为障碍，0为通道)
def generate_maze(dim=10, p=0.2):
    maze = np.random.choice([0, 1], size=(dim, dim), p=[1-p, p])  # 以 p 的概率填充障碍
    maze[0, 0] = 0  # 确保起点可通行
    maze[dim-1, dim-1] = 0  # 确保终点可通行
    return maze

# 迷宫环境类
class MazeEnv:
    def __init__(self, maze):
        self.maze = maze
        self.start = (0, 0)
        self.end = (len(maze)-1, len(maze)-1)
        self.state = self.start
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        row, col = self.state
        new_row, new_col = row + self.actions[action][0], col + self.actions[action][1]

        # 确保新位置在迷宫内部并且是通道
        if 0 <= new_row < len(self.maze) and 0 <= new_col < len(self.maze) and self.maze[new_row][new_col] == 0:
            self.state = (new_row, new_col)
        else:
            return self.state, -1, False  # 无效动作，返回当前状态，奖励-1

        # 检查是否到达终点
        if self.state == self.end:
            return self.state, 100, True  # 终点，奖励100

        return self.state, -1, False  # 普通步伐，奖励-1

# DQN智能体
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.batch_size = 64
        self.gamma = 0.9  # 折扣因子

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_dim))  # 探索
        state_tensor = torch.FloatTensor(state)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()  # 利用

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)
        rewards_tensor = torch.FloatTensor(rewards)
        actions_tensor = torch.LongTensor(actions)
        dones_tensor = torch.BoolTensor(dones)

        # Q值更新
        q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states_tensor).max(1)[0]
        target_q_values = rewards_tensor + (self.gamma * next_q_values * ~dones_tensor)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# 训练DQN并记录总奖励变化
def train_dqn(episodes=1000):
    reward_history = []  # 用于记录每一轮的总奖励
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            step += 1
            if step > 2500: # 防止走不出的情况 陷入死循环
                print(f"Episode {e}/{episodes} Fail to the destination")
                break
        reward_history.append(total_reward)  # 记录每一轮的奖励
        if e % 100 == 0:
            print(f"Episode {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
    
    return reward_history  # 返回奖励变化的历史记录

# 获取并展示最佳路径的策略
def get_best_policy():
    policy = np.zeros_like(maze, dtype=int)
    state = env.reset()
    path = []  # 保存最佳路径
    while state != env.end:
        row, col = state
        action = agent.act(state)
        policy[row, col] = action
        path.append(state)
        next_state, _, done = env.step(action)
        state = next_state
    path.append(env.end)  # 添加终点
    return policy, path

# 可视化迷宫并显示最佳路径上的箭头
def show_best_path(maze, path, policy):
    fig, ax = plt.subplots()
    ax.imshow(maze, cmap='binary')

    actions_to_char = {
        0: '\u2191',  # ↑
        1: '\u2193',  # ↓
        2: '\u2190',  # ←
        3: '\u2192'   # → 
    }

    # 只绘制最佳路径上的箭头
    for (row, col) in path:
        if maze[row, col] == 0:  # 只有通道上的位置才绘制箭头
            action = policy[row, col]
            ax.text(col, row, actions_to_char[action], ha='center', va='center', color='blue', fontsize=10)

    plt.show()

# 可视化迷宫布局
def show_maze_layout(maze):
    fig, ax = plt.subplots()
    ax.imshow(maze, cmap='binary')  # 二值化的迷宫，障碍物为黑色，通道为白色

    # 标注起点和终点
    ax.text(0, 0, 'S', ha='center', va='center', color='red', fontsize=12)  # 起点
    ax.text(maze.shape[0]-1, maze.shape[1]-1, 'E', ha='center', va='center', color='green', fontsize=12)  # 终点

    plt.title("Maze Layout")
    plt.show()

# 环境与DQN设置
maze = generate_maze(dim=10, p=0.25)
show_maze_layout(maze)

env = MazeEnv(maze)
state_dim = 2  # 状态是坐标(x, y)
action_dim = 4  # 四个方向
agent = DQNAgent(state_dim, action_dim)

# 训练DQN并展示最佳路径
reward_history = train_dqn(episodes=1000)

# 获取最佳策略并计算最佳路径
best_policy, best_path = get_best_policy()

# 可视化最佳路径
show_best_path(maze, best_path, best_policy)

# 绘制总奖励变化图
plt.plot(reward_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward vs Episode')
plt.show()
