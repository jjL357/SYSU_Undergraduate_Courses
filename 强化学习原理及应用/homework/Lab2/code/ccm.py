import numpy as np
import matplotlib.pyplot as plt

episodes = 500  #迭代次数
epsilon = 0.05  #选择随机方向的概率
alpha = 0.5     #学习率
gamma = 0.99    #衰减值
rows = 4        
cols = 12
direction = 4   

# q-learning 中找到最大的Q_s_a
def find_max(Q_table,pos):
    k=int(12 * pos[0] + pos[1])
    max_value=np.max(Q_table[k])
    return max_value

# 选择动作
def epsilon_greedy(location, q_t):
    random  = np.random.random() #[0,1.0)
    if random < epsilon :
        move = np.random.randint(direction)
    else:
        move = np.argmax(q_t[location]) 
    return move

# 移动
def move(pos,mov):
    x=pos[0]
    y=pos[1]
    if mov==0 and x>0:  #上
        x-=1
    elif mov==1 and x<3:    #下
        x+=1
    elif mov==2 and y>0:    #左
        y-=1
    elif mov==3 and y<11:   #右
        y+=1
    return (x,y)

# 计算reward并调整位置
def reward(new_loc):
    ef=False
    rw=0
    if new_loc==47:
        ef=True
    else:
        ef=False
    if( 37<=new_loc and new_loc <= 46): #从悬崖掉落
        rw = -100
        new_loc=36 
    else:
        rw=-1
    return new_loc,rw, ef

# Q-learning
def q_learning():
    Q_table=np.zeros((rows*cols,4)) #Q表
    clif_map=np.zeros((rows,cols))  #地图，用于每次存储路径
    rw_list=[]  #记录Reward的变化
    for step in range(episodes):
        #初始化S
        sum=0
        pos=(3,0)
        end_flag=False
        clif_map=np.zeros((rows,cols))
        clif_map[3][0]=1
        while(not end_flag):
            loc=int(12 * pos[0] + pos[1])
            mov=epsilon_greedy(loc,Q_table)
            pos=move(pos,mov)
            clif_map[pos[0]][pos[1]]=1
            new_loc=int(12 * pos[0] + pos[1])
            new_loc,rwd, end_flag= reward(new_loc)
            sum+=rwd
            max_q_s_a=find_max(Q_table,pos)
            # 根据公式更新Q表
            Q_table[loc][mov]=Q_table[loc][mov]+alpha*(rwd+(gamma*max_q_s_a)-Q_table[loc][mov])
            pos=(int(new_loc/12),int(new_loc%12))
        rw_list.append(sum)
    return clif_map,Q_table,rw_list

# sarsa
def sarsa():
    Q_table=np.zeros((rows*cols,4))
    clif_map=np.zeros((rows,cols))
    rw_list=[]
    for step in range(episodes):
        sum=0
        pos=(3,0)
        end_flag=False
        clif_map=np.zeros((rows,cols))
        clif_map[3][0]=1
        loc=int(12 * pos[0] + pos[1])
        mov=epsilon_greedy(loc,Q_table)
        while(not end_flag):
            loc=int(12 * pos[0] + pos[1])
            pos=move(pos,mov)
            clif_map[pos[0]][pos[1]]=1
            new_loc=int(12 * pos[0] + pos[1])
            new_loc,rwd, end_flag= reward(new_loc)
            sum+=rwd
            #选择下一个状态的新动作
            next_mov=epsilon_greedy(new_loc,Q_table)
            
            q_s_a=Q_table[new_loc][next_mov]
            # 根据公式更新Q表
            Q_table[loc][mov]=Q_table[loc][mov]+alpha*(rwd+(gamma*q_s_a)-Q_table[loc][mov])
            
            #更新位置和动作
            mov=next_mov
            pos=(int(new_loc/12),int(new_loc%12))

        rw_list.append(sum)

    return clif_map,Q_table,rw_list

# 可视化结果
def show_result(path_q,path_s,l_q,l_s):
    for i in range(1,11):
        path_q[3][i]=-0.1
        path_s[3][i]=-0.1
    
    for i in range(4):
        for j in range(12):
            if path_q[i][j]==1:
                path_q[i][j]=0.05
            if path_s[i][j]==1:
                path_s[i][j]=0.05
    
    plt.subplot(2, 1, 1)
    plt.title("Q-learning path")
    plt.axis('off')
    plt.imshow(path_q)
    plt.subplot(2, 1, 2)
    plt.title("sarsa path")
    plt.axis('off')
    plt.imshow(path_s)
    plt.show()

    show_q=[]
    show_s=[]
    for i in np.arange(0,500,0.1):
        j=int(i)
        if(j!=0):
            show_q.append(0.5*l_q[j-1]+0.5*l_q[j])
            show_s.append(0.5*l_s[j-1]+0.5*l_s[j])
        else:
            show_q.append(l_q[j])
            show_s.append(l_s[j])

    plt.plot(np.arange(0,500,0.1),show_q)
    plt.plot(np.arange(0,500,0.1),show_s)

    plt.show()

if __name__ == "__main__":
    print("cliff_map:")
    
    print("0 0 0 0 0 0 0 0 0 0 0 0\n","0 0 0 0 0 0 0 0 0 0 0 0\n",
    "0 0 0 0 0 0 0 0 0 0 0 0\n","s * * * * * * * * * * e\n",sep='')
    path_q,q_q,l_q=q_learning()
    path_s,q_s,l_s=sarsa()
    print("Q_learnig table:\n",'up\t\tdown\t\tleft\t\tright\n',q_q)
    print("sarsa table:\n",'up\t\tdown\t\tleft\t\tright\n',q_s)

    show_result(path_q,path_s,l_q,l_s)
    