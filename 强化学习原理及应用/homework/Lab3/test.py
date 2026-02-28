import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

# 游戏环境：井字棋
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9  # 3x3的棋盘
        self.winner = None
        self.game_over = False
    
    def print_board(self):
        """绘制棋盘和棋子"""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['', '', ''])
        ax.set_yticklabels(['', '', ''])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect('equal')
        
        # 绘制网格线
        ax.plot([0, 0], [0, 2], color='black', lw=3)
        ax.plot([1, 1], [0, 2], color='black', lw=3)
        ax.plot([2, 2], [0, 2], color='black', lw=3)
        ax.plot([0, 2], [0, 0], color='black', lw=3)
        ax.plot([0, 2], [1, 1], color='black', lw=3)
        ax.plot([0, 2], [2, 2], color='black', lw=3)
        
        # 绘制棋子
        for i, val in enumerate(self.board):
            row = i // 3  # 计算行
            col = i % 3   # 计算列
            if val == 'X':
                ax.text(col, row, 'X', fontsize=24, ha='center', va='center', color='red')
            elif val == 'O':
                ax.text(col, row, 'O', fontsize=24, ha='center', va='center', color='blue')

        plt.show()

    def get_state(self):
        """返回当前棋盘的状态（数值型数组形式）"""
        return [1 if x == 'X' else -1 if x == 'O' else 0 for x in self.board]

    def available_actions(self):
        """返回空位的索引"""
        return [i for i, x in enumerate(self.board) if x == ' ']

    def make_move(self, action, player):
        """玩家做出选择,返回是否胜利"""
        self.board[action] = player
        self.winner = self.check_winner()
        return self.winner

    def check_winner(self):
        """检查是否有获胜者"""
        winning_combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # 横向
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # 纵向
            (0, 4, 8), (2, 4, 6)  # 对角线
        ]
        for (x, y, z) in winning_combinations:
            if self.board[x] == self.board[y] == self.board[z] and self.board[x] != ' ':
                return self.board[x]
        if ' ' not in self.board:  # 如果棋盘满了,判定为平局
            return 'Draw'
        return None

    def reset(self):
        """重置棋盘"""
        self.board = [' '] * 9
        self.winner = None
        self.game_over = False

# DQN Agent with Experience Replay and Target Network
class DQNAgent:
    def __init__(self, learning_rate=0.5, discount_factor=0.99, exploration_rate=0.05, player="X", batch_size=32, memory_size=10000):
        self.q_table = {}  # Q值表,键是状态,值是状态-动作对
        self.learning_rate = learning_rate  # 学习率
        self.discount_factor = discount_factor  # 折扣因子
        self.exploration_rate = exploration_rate  # 探索率
        self.player = player # 棋手"X"或"O"
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = []  # 经验池
        self.model = nn.Sequential(
            nn.Linear(9, 128),  # 输入是一个长度为9的向量
            nn.ReLU(),
            nn.Linear(128, 9)  # 输出9个Q值，对应9个可用位置
        )
        self.target_model = copy.deepcopy(self.model)  # 目标网络
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def get_q_value(self, state, action):
        """返回指定状态和动作的Q值"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 增加batch维度
        q_values = self.model(state_tensor)  # 前向传播得到所有Q值
        return q_values[0][action].item()  # 获取指定动作的Q值

    def update_q_value(self, state, action, reward, next_state, done):
        """更新Q值网络"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        q_values = self.model(state_tensor)
        next_q_values = self.target_model(next_state_tensor)

        target = reward
        
        target += self.discount_factor * torch.max(next_q_values).item()  # 如果游戏没结束，使用最大Q值进行更新
        
        next_q_values[0][action] = target

        # 计算损失
        loss = self.loss_fn(q_values, next_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action(self, state):
        """epsilon-greedy 策略选择动作"""
        available_actions = state.available_actions()  # 获取合法的动作（即空位的索引）
        
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(available_actions)  # 探索：随机选择一个合法动作
        else:
            state_tensor = torch.tensor(state.get_state(), dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state_tensor)  # 获取所有动作的Q值
            
            # 获取Q值最大的动作，并且确保它是合法的
            q_values = q_values.squeeze(0).detach().numpy()  # 转换为numpy数组
            available_q_values = {action: q_values[action] for action in available_actions}  # 只保留合法动作的Q值
            
            # 返回具有最大Q值的合法动作
            best_action = max(available_q_values, key=available_q_values.get)
            return best_action

    def remember(self, state, action, reward, next_state, done):
        """将经历添加到经验池"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)  # 如果经验池已满，则移除最旧的经验
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """从经验池中采样进行训练"""
        if len(self.memory) < self.batch_size:
            return

        # 随机采样一批经验
        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in batch:
            self.update_q_value(state, action, reward, next_state, done)

    def update_target_network(self):
        """更新目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())


def train_agent(episodes=10000):
    agent_X = DQNAgent(player="X")
    env = TicTacToe()
    
    for episode in tqdm(range(episodes)):
        X_first  = np.random.randint(0, 2)  # 生成 0 或 1 的随机整数
        env.reset()
        state = env.get_state()  
        done = False
        X_action = None
        X_next_state = None 
        X_state = copy.deepcopy(state)
        while not done:
            if X_first == 0:
                X_action = agent_X.choose_action(env)  # 选择一个动作
                X_state = copy.deepcopy(state)
                winner = env.make_move(X_action, agent_X.player)  # 执行该动作
                X_next_state = env.get_state()
            else:
                action = random.choice(env.available_actions())  # 选择一个动作
                winner = env.make_move(action,"O")  # 执行该动作
            
            if winner == 'X':
                reward = 100
                done = True
            elif winner == 'O':
                reward = -100
                done = True
            elif winner == 'Draw':
                reward = -5 # 平局
                done = True
            else:
                reward = -1 # 未分出胜负,继续游戏
            
            next_state = env.get_state()  # 下一状态是棋盘
            
            if X_action is not None:    
                agent_X.remember(X_state, X_action, reward, X_next_state, done)
                agent_X.replay()  # 执行经验回放
                agent_X.update_target_network()  # 定期更新目标网络
                
            X_first ^= 1
            state = next_state  # 更新状态
        
    return agent_X

def play_game(agent_X,human = False):
    """让智能体与自己对战,显示棋盘"""
    env = TicTacToe()
    env.reset()
    state = env
    done = False
    whose_turn  = np.random.randint(0, 2)  # 生成 0 或 1 的随机整数
    while not done:
        # 打印当前棋盘
        env.print_board()
        print()
        
        if whose_turn == 0:
            action = agent_X.choose_action(state)  # 选择一个动作
            winner = env.make_move(action, agent_X.player)  # 执行该动作
                
        elif human:
            input0 = input("输入你下的位置(例如: 0 0)  :")
            input0 = input0.split()
            input1 = int(input0[0])
            input2 = int(input0[1])
            action = input1 * 3 + input2
            assert action in state.available_actions() , "输入的位置不合法"
            winner = env.make_move(action,"O")  # 执行该动作
            
        else :
            action = random.choice(state.available_actions())  # 选择一个动作
            winner = env.make_move(action,"O")  # 执行该动作
        whose_turn ^= 1
        
        if winner == 'X':
            done = True
        elif winner == 'O':
            done = True
        elif winner == 'Draw':
            done = True
        
        next_state = env  # 下一状态是棋盘
        state = next_state  # 更新状态
        
    # 打印最终棋盘
    env.print_board()
    if winner == 'Draw':
        print("It's a draw!")
    else:
        print(f"{winner} wins!")

# 训练智能体并播放游戏
agent_X = train_agent(episodes=10000)

agent_X.exploration_rate = 0.0
for i in range(3):
    play_game(agent_X, human=False)
play_game(agent_X, human=True)
