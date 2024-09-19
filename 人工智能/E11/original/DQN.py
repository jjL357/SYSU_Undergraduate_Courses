# coding=gb2312
# requirements
# - Python 3.7
# - PyTorch 1.7
# - Gym
# - (Optional) wandb

import gym
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import random
import matplotlib.pyplot as plt
class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        #构建两层的神经网络
        # TODO write another linear layer here with 
        # inputsize "hidden_size" and outputsize "output_size"
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #前向传播
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        # TODO: calculate output with layer fc2
        x = F.relu(self.fc2(x))
        return x


class ReplayBuffer:
    def __init__(self, capacity):#初始化经验池
        self.buffer = []
        self.capacity = capacity

    def len(self):#返回经验池中经验数量
        return len(self.buffer)

    def push(self, *transition):#加入经验池
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, n):#从经验池中随机获得经验
        index = np.random.choice(len(self.buffer), n)
        batch = [self.buffer[i] for i in index]
        return zip(*batch)

    def clean(self):#清空经验池
        self.buffer.clear()


class DQN:
    def __init__(self, env, input_size, hidden_size, output_size):#初始化
        self.env = env
        self.eval_net = QNet(input_size, hidden_size, output_size)
        self.target_net = QNet(input_size, hidden_size, output_size)
        self.optim = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.eps = args.eps
        self.buffer = ReplayBuffer(args.capacity)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0
    
    def choose_action(self, obs):#选择动作
        # epsilon-greedy
        #action=0
        
        if np.random.uniform() <= self.eps:
            #随机选择
            # TODO: choose an action in [0, self.env.action_space.n) randomly
            action = np.random.randint(0,self.env.action_space.n) 
            #self.env.action_space.n=2
        else:
            #选取神经网络输出值大的动作
            #pass
            # TODO: get an action with "eval_net" according to observation "obs"
            obs_tmp=torch.FloatTensor(obs)
            with torch.no_grad():#no_grad不保存为反向传播储备的值
               vals=self.eval_net(obs_tmp)#输出值
            action=vals.argmax().item()#选取神经网络输出值大的动作
        return action

    def store_transition(self, *transition):#存经验
        self.buffer.push(*transition)
        
    def learn(self):
        if self.eps > args.eps_min:
            self.eps *= args.eps_decay

        if self.learn_step % args.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1 
        
        obs, actions, rewards, next_obs, dones = self.buffer.sample(args.batch_size)#获取经验学习
        actions = torch.LongTensor(actions)  # LongTensor to use gather latter
        dones = torch.FloatTensor(dones)
        rewards = torch.FloatTensor(rewards)
        

        # TODO: calculate q_eval with eval_net and q_next with target_net
        
        # TODO: q_target = r + gamma * (1-dones) * q_next
        # TODO: calculate loss between "q_eval" and "q_target" with loss_fn
        # TODO: optimize the network with self.optim
       

        # TODO: calculate q_eval with eval_net and q_next with target_net
        q_eval = self.eval_net(np.array(obs)).gather(1,actions.unsqueeze(1)).squeeze(1)
        q_next = torch.max(self.target_net(np.array(next_obs)), dim = 1)[0]
        
        q_target = rewards + args.gamma * (1-dones) * q_next#目标值

        loss = self.loss_fn(q_eval, q_target)#计算损失
        self.optim.zero_grad()#清空上一轮梯度
        loss.backward()#反向传播
        self.optim.step()#优化反向传播


def main():
    #env = gym.make(args.env, render_mode='human')
    r=[]
    env = gym.make(args.env)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = DQN(env, o_dim, args.hidden, a_dim) 
    for i_episode in range(args.n_episodes):
        obs = env.reset()[0]
        episode_reward = 0
        done = False
        step_cnt=0
        while not done and step_cnt<500:
            step_cnt+=1
            #env.render()
            action = agent.choose_action(obs)
            next_obs, reward, done, info, _ = env.step(action) 
            agent.store_transition(obs, action, reward, next_obs, done)
            episode_reward += reward
            obs = next_obs

            if agent.buffer.len() >= args.capacity:
                agent.learn()
        print(f"Episode: {i_episode}, Reward: {episode_reward}")
        r.append(episode_reward)
    plt.plot([i for i in range(1,len(r)+1)],r,color="r")
    plt.title('reward')
    plt.show()   
    average_reward=[]
    sum=0
    for i in  range(len(r)):
        if i%10==9:
            average_reward.append(sum/10)
            sum=0
        sum+=r[i]
    plt.plot([i for i in range(len(r)//10)],average_reward,color="r")
    plt.title('average reward')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        default="CartPole-v1",  type=str)
    parser.add_argument("--lr",             default=1e-3,       type=float)
    parser.add_argument("--hidden",         default=64,         type=int)
    parser.add_argument("--n_episodes",     default=500,       type=int)
    parser.add_argument("--gamma",          default=0.99,       type=float)
    parser.add_argument("--log_freq",       default=100,        type=int)
    parser.add_argument("--capacity",       default=5000,      type=int)
    parser.add_argument("--eps",            default=1,        type=float)
    parser.add_argument("--eps_min",        default=0.05,       type=float)
    parser.add_argument("--batch_size",     default=128,        type=int)
    parser.add_argument("--eps_decay",      default=0.999,      type=float)
    parser.add_argument("--update_target",  default=100,        type=int)
    args = parser.parse_args()
    main()