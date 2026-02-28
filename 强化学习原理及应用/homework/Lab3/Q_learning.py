import random
import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy

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


    def get_state(self):
        """返回当前棋盘的状态（元组形式）"""
        return tuple(self.board)  # 转换为元组,方便作为键



class QLearningAgent:
    def __init__(self, learning_rate=0.5, discount_factor=0.99, exploration_rate=0.05,player = "X"):
        self.q_table = {}  # Q值表,键是状态,值是状态-动作对
        self.learning_rate = learning_rate  # 学习率
        self.discount_factor = discount_factor  # 折扣因子
        self.exploration_rate = exploration_rate  # 探索率
        self.player = player # 棋手"X"或"O"

    def get_q_value(self, state, action):
        """返回指定状态和动作的Q值,若不存在,则返回0"""
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state, done):        
        """更新Q值表"""

        best_next_action = self.get_best_action(next_state)  # 只有游戏未结束时才选择最优动作
        next_q_value = self.get_q_value(next_state.get_state(), best_next_action)
        
        # 更新公式
        self.q_table[(state.get_state(), action)] = (1 - self.learning_rate) * self.get_q_value(state.get_state(), action) + self.learning_rate * (reward + self.discount_factor * next_q_value)

    def get_best_action(self, state):
        """返回当前状态下最优的动作"""
        available_actions = state.available_actions()
        if not available_actions:  # 如果没有可用动作（游戏结束）,返回一个默认动作
            return None
        q_values = [self.get_q_value(state.get_state(), action) for action in available_actions]
        max_q_value = max(q_values)
        best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q_value]
        return random.choice(best_actions)  # 如果有多个最佳动作,随机选择一个
    
    def choose_action(self, state):
        """epsilon-greedy 策略选择动作"""
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(state.available_actions())  # 探索：随机选择一个动作
        else:
            return self.get_best_action(state)  # 利用：选择Q值最大的动作


def train_agent(episodes=10000):
    agent_X = QLearningAgent(player="X")
    env = TicTacToe()
    
    for episode in tqdm(range(episodes)):
        whose_turn  = np.random.randint(0, 2)  # 生成 0 或 1 的随机整数
        env.reset()
        state = env  # 当前状态是棋盘
        done = False
        X_action = None
        X_next_state = None 
        X_state = copy.deepcopy(env)
        while not done:
            if whose_turn == 0:
                X_action = agent_X.choose_action(state)  # 选择一个动作
                X_state = copy.deepcopy(env)
                winner = env.make_move(X_action, agent_X.player)  # 执行该动作
                X_next_state = env
            else:
                action = random.choice(state.available_actions())  # 选择一个动作
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
            
            next_state = env  # 下一状态是棋盘
            
                
            if X_action is not None:    
                agent_X.update_q_value(X_state, X_action, reward, X_next_state, done)
                
            whose_turn ^= 1
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
agent_X = train_agent(episodes=100000)

            
agent_X.exploration_rate = 0.0
for i in range(3):
    play_game(agent_X,human = False)
play_game(agent_X,human = True)
