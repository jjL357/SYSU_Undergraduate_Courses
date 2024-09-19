# DQN Lab #

## Get Started ##
依赖库版本  
> torch==2.3.0  
> numpy==1.26.4  
> pygame==2.5.3  
> gym==0.23.0  
> matplotlib==3.9.0  

## code ##
- original部分请参考E11.pdf
- extension部分：

    code | description | result
    -----| ------ | ------
    DQN1.py | 基础版本 | 能达标
    DQN2.py | 网络参考成品代码 | 还没跑过  
    其它 | 5种优化代码 | 具体情况参考下一节

- 结果解释：
我没充分测试各种优化和参数，仅从已知的一些结果说一些粗浅简介  
> 1. Priority Buffer和DDQN能看出比较明显的优化  
> 2. Priority Buffer + decay Epsilon + Noise的不知道为什么效果不太好  
> 3. 这个model的单次表现比较看运气，譬如运气好的时候，我用 Priority Buffer 实现在 36Episode 就跑出了 500 分，Priority Buffer 和 Priority Buffer + decay Epsilon + Noise Net 实现，都在 70Episode左右实现了连续十次的 500分

## 可能的优化 ###

- 这部分对 DQN 的优化主要参考`NTU`李宏毅老师的讲解， 李老师提到一篇论文`Rainbow:Combining Improvements in Deep Reinforcement Learning`，实现多种`DQN`优化， 我对其中的5种进行了尝试，以下详细介绍：

1. Double DQN:DQN网络倾向于高估自己的得分，因为网络具有误差，网络会倾向于选择自己高估的action；Double DQN用原有的target net和eval net对action进行双重选择，减少误差带来的负面影响，代码在DDQN.py中实现  

2. Dueling DQN：修改网络结构，原来网络直接输出 Q，修改成 Q = V + A，A 可能可以一步实现 Q 多步的修改，实现加速，代码在 DDQN.py 中

3. 实现Priority Buffer: Buffer 中有些资料更具有参考意义，譬如我的实现中赋予 Loss 较大的资料以更大的优先级，从而更大概率从这些资料中进一步学习，代码在 PriorityBuf.py中实现  

4. Decay Epsilon： 在训练开始的时候， Eps 较大， 充分尝试不同的 policy； 随着训练逐
渐递减， 收敛于一个较好的 policy， 在`Pri_Eps_NoiDQN.py`中实现  

5. Noise Net: 在每个Episode开始的时候，向网络中添加一个噪声，其原理与采用Epsilon类似，在`Pri_Eps_NoiDQN.py`中实现

- 以下部分来自TA讲解
DQN模型稳健的关键在于reward的设计，原来只要没死就`reward=1`不符合实际情况，设置需要更符合实际。
如TA的代码设置`new_reward = (0.01 - abs(theta))`，其中theta为杆的偏移角，200轮即可平均超过475分，并且相当稳健  
腾讯一篇[RL玩王者荣耀的论文](https://ojs.aaai.org/index.php/AAAI/article/view/6144)就展示了reward设计的重要性  
实现时需要注意：较优的状态的reward要是正的，否则模型将趋向于迅速终止游戏  