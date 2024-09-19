# priority buffer
# first 500 in 36 Episode 

import gym
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import random
from collections import deque
import matplotlib.pyplot as plt


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)  # 添加一个全连接层

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 计算输出
        return x


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.pos = 0

    def len(self):
        return len(self.buffer)

    def push(self, *transition):
        max_prio = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = np.array(self.priorities)
        else:
            prios = np.array(self.priorities)[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs) # different possibility
        transitions = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        obs, actions, rewards, next_obs, dones = zip(*transitions)
        return np.array(obs), np.array(actions), np.array(rewards), np.array(next_obs), np.array(dones), indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def clean(self):
        self.buffer.clear()
        self.priorities.clear()


class DQN:
    def __init__(self, env, input_size, hidden_size, output_size):
        self.env = env
        self.eval_net = QNet(input_size, hidden_size, output_size)
        self.target_net = QNet(input_size, hidden_size, output_size)
        self.optim = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.eps = args.eps
        self.gamma = args.gamma
        self.buffer = PrioritizedReplayBuffer(args.capacity)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0

    def choose_action(self, obs):
        if np.random.uniform(0, 1) < self.eps:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.eval_net(obs)
                action = q_values.argmax().item()
        return action

    def store_transition(self, *transition):
        self.buffer.push(*transition)

    def learn(self):
        if self.buffer.len() < args.batch_size:
            return

        if self.learn_step % args.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        obs, actions, rewards, next_obs, dones, indices, weights = self.buffer.sample(args.batch_size)
        obs = torch.FloatTensor(obs)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_obs = torch.FloatTensor(next_obs)
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights)

        q_eval = self.eval_net(obs).gather(1, actions).squeeze(1)
        q_next = self.target_net(next_obs).max(1)[0].detach()
        q_target = rewards + self.gamma * (1 - dones) * q_next

        loss = (weights * self.loss_fn(q_eval, q_target)).mean() # add weight
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        priorities = (q_eval - q_target).abs().detach().numpy() + 1e-5
        self.buffer.update_priorities(indices, priorities) # update


def main():
    env = gym.make(args.env)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = DQN(env, o_dim, args.hidden, a_dim)

    episode_rewards = []
    recent_avg_rewards = []

    for i_episode in range(args.n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step_cnt = 0

        while not done and step_cnt < 500:
            step_cnt += 1
            env.render()
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done)
            episode_reward += reward
            obs = next_obs

            if agent.buffer.len() >= args.batch_size:
                agent.learn()
            
        episode_rewards.append(episode_reward)

        if len(episode_rewards) >= 100:
            recent_avg_rewards.append(np.mean(episode_rewards[-100:]))
        else:
            recent_avg_rewards.append(np.mean(episode_rewards))

        print(f"Episode: {i_episode}, Reward: {episode_reward}")

    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(recent_avg_rewards, label='Recent 100 Episodes Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="environment name")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--hidden", default=64, type=int, help="dimension of hidden layer")
    parser.add_argument("--n_episodes", default=500, type=int, help="number of episodes")
    parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
    parser.add_argument("--capacity", default=10000, type=int, help="capacity of replay buffer")
    parser.add_argument("--eps", default=0.1, type=float, help="epsilon of ε-greedy")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("--update_target", default=100, type=int, help="frequency to update target network")
    args = parser.parse_args()
    main()
