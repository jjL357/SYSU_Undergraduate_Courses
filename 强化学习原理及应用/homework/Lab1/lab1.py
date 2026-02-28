import numpy as np

# 迷宫: 1为障碍 / 0为通道
maze = np.array([
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
    [ 0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.],
    [ 1.,  0.,  0.,  1.,  1.,  0.,  0.,  1.],
    [ 1.,  1.,  0.,  0.,  1.,  0.,  1.,  1.],
    [ 1.,  0.,  1.,  0.,  1.,  0.,  0.,  1.],
    [ 1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]
])

dimensions = maze.shape[0] # 迷宫维度
terminal_state = (6,8) # 终止状态

actions = [(-1,0),(1,0),(0,-1),(0,1)] # 动作集合: 上下左右

# 状态集合
states = []
for i in range(dimensions): # 搜索合法的状态 
        for j in range(dimensions):
            if(maze[i][j]==0):
                states.append((i,j))
                
# 奖励
reward = np.ones((dimensions,dimensions))
reward = -reward 

# 动作转移概率
P_a =  np.zeros((dimensions,dimensions,4))          
for s in states:
    row = s[0] 
    col = s[1]
    valid_actions = []
    for i in range(len(actions)):
        new_row = row + actions[i][0] 
        new_col = col + actions[i][1]
        if new_row < 0 or new_col < 0 or new_row >= dimensions or new_col >= dimensions:
            continue
        else:
            if maze[new_row][new_col] == 0 :
                valid_actions.append(i)
    for ns in valid_actions:
        P_a[row][col][ns] = 1 

# 对最接近出口的状态做特殊处理,使其可以走出迷宫 
P_a[6][7][2] = 1
P_a[6][7][3] = 1


def policy_iteration(discount_factor = 0.9 , iterations = 10000):
    states_num = len(states)
    values = np.zeros(states_num)
    
    # 初始化每个状态的策略(找到上下左右中第一个可行的动作最为初始的策略) 
    policy = np.zeros(states_num)
    policy = policy.astype(int)
    for i in range(states_num):
        row = states[i][0]
        col = states[i][1]
        for j in range(4):
            if P_a[row][col][j] > 0:
                policy[i] = j
                break
    
    for i in range(iterations):
        # policy-evaluation 过程
        e = 0.0001 # 收敛条件
        q_values = np.zeros(states_num) # 当前策略的价值
        while True:
            old_q_values = q_values.copy()
            for s_idx in range(states_num):
                # 下一步状态
                row = states[s_idx][0]
                col = states[s_idx][1]
                new_row = row + actions[int(policy[s_idx])][0] 
                new_col = col + actions[int(policy[s_idx])][1]
                if new_row == terminal_state[0] and new_col == terminal_state[1]: # 到达出口
                    q_values[s_idx] = P_a[row][col][int(policy[s_idx])] * (reward[row][col] + discount_factor * 50000)
                else : # 未到达出口
                    new_value_index = states.index((new_row,new_col))
                    q_values[s_idx] = P_a[row][col][int(policy[s_idx])] * (reward[row][col] + discount_factor * old_q_values[new_value_index])
            
            if np.sum(np.fabs(q_values - old_q_values)) < e: # 达到收敛条件, 跳出policy-evaluation过程
                break
            
        # policy-improvement
        old_values = q_values.copy()
        old_policy = policy.copy()
        # 计算并判断是否达到最佳策略
        for s_idx in range(states_num):
            row = states[s_idx][0]
            col = states[s_idx][1]
            values_for_action = []
            for a_idx in range(len(actions)):
                if P_a[row][col][a_idx] == 0: 
                    continue
                # 下一步状态
                new_row = row + actions[a_idx][0] 
                new_col = col + actions[a_idx][1]
                if new_row == terminal_state[0] and new_col == terminal_state[1]:# 到达出口
                    values_for_action.append((P_a[row][col][a_idx] * (reward[row][col] + discount_factor * 50000),a_idx))
                else : #未达到出口
                    new_value_index = states.index((new_row,new_col))
                    values_for_action.append((P_a[row][col][a_idx] * (reward[row][col] + discount_factor * old_values[new_value_index]),a_idx))
            max_value = max(values_for_action, key=lambda x: x[0])
            values[s_idx] = max_value[0] 
            policy[s_idx] = max_value[1] 
        # 判断是否达到最佳策略    
        if np.all(policy == old_policy):
            break
    return values,policy

def value_iteration(discount_factor = 0.9 , iterations = 10000):
    states_num = len(states)
    values = np.zeros(states_num)
    policy = np.zeros(states_num)
    e = 0.001 # 收敛条件
    
    for i in range(iterations):
        old_values = values.copy()
        for s_idx in range(states_num): # 对每个状态计算
            row = states[s_idx][0]
            col = states[s_idx][1]
            values_for_action = []
            for a_idx in range(len(actions)): # 遍历每种动作
                if P_a[row][col][a_idx] == 0:
                    continue
                # 下一步状态
                new_row = row + actions[a_idx][0] 
                new_col = col + actions[a_idx][1]
                if new_row == terminal_state[0] and new_col == terminal_state[1]: # 到达出口
                    values_for_action.append((P_a[row][col][a_idx] * (reward[row][col] + discount_factor * 50000),a_idx))
                else :
                    new_value_index = states.index((new_row,new_col)) # 未达到出口
                    values_for_action.append((P_a[row][col][a_idx] * (reward[row][col] + discount_factor * old_values[new_value_index]),a_idx))
            max_value = max(values_for_action, key=lambda x: x[0])
            values[s_idx] = max_value[0] 
            policy[s_idx] = max_value[1] # 这里跟踪记录对应的策略, 简化后续提取策略的步骤
        # 判断是否收敛
        if np.sum(np.fabs(values - old_values)) < e :
            break
    return values , policy

# 可视化values
def show_values(values):
    print("The values:")
    for i in range(dimensions):
        for j in range(dimensions):
            if (i,j) in states:
                index = states.index((i,j))
                print(f"{values[index]:4.1f}", end=' ')
            else :
                print(f"{'NAN':<4}", end=' ')
        print()        

# 可视化policy     
def show_policy(policy,actions_to_char):
    print("The policy:")
    for i in range(dimensions):
        for j in range(dimensions):
            if (i,j) in states:
                index = states.index((i,j))
                print(actions_to_char[policy[index]] ,end=' ')
            else :
                print("*" ,end = ' ')
        print()
        
if __name__ == '__main__':
    
    ### 选择使用policy_iteration 或者 values iteration ###
    values , policy = policy_iteration()
    # values , policy = value_iteration()
    
    actions_to_char = {# 将action的索引对应为上下左右的箭头符号,便于可视化
        0 : '\u2191',
        1 : '\u2193',
        2 : '\u2190',
        3 : '\u2192'
    }
    
    show_values(values) # 可视化values
    show_policy(policy,actions_to_char) # 可视化policy
    
