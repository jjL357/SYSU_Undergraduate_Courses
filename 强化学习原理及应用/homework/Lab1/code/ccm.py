import numpy as np
import matplotlib.pyplot as plt
dim=8
num_states=[]

terminal_state=(6,8)

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

action=[(-1,0),(0,1),(1,0),(0,-1)] #上右下左
direction=["↑","→","↓","←"]
P_a=np.zeros((dim*dim,4))

def init():
    for i in range(dim):
        for j in range(dim):
            if(maze[i][j]==0):
                num_states.append((i,j))


    for i in range(dim*dim):
        x=int(i/dim)
        y=i%dim
        if(maze[x][y]==1):
            continue
        sum=0
        for k in range(len(action)):
            if((x+action[k][0])<0 or (x+action[k][0])>=8 or (y+action[k][1])<0 or (y+action[k][1])>=8):
                continue
            if (maze[x+action[k][0]][y+action[k][1]]==0):
                P_a[i][k]=1.
                sum+=1.
        for k in range(len(action)):
            if (P_a[i][k]==1):
                P_a[i][k]/=sum
    P_a[55][1]=0.5
    P_a[55][3]=0.5


def value_iteration(gamma = 0.9, num_of_iterations = 10000):

    N_STATES=len(num_states)
    N_ACTIONS=len(action)
    error=0.0001

    values = np.zeros(N_STATES)
    rewards=[-1]*N_STATES  

    for i in range(num_of_iterations):
        values_tmp = values.copy()

        for idx in range(N_STATES):
            v_a = []
            s=num_states[idx][0]*dim+num_states[idx][1]
            for a in range(N_ACTIONS):
                if(P_a[s][a]!=0):
                    
                    next_x=num_states[idx][0]+action[a][0]
                    next_y=num_states[idx][1]+action[a][1]
                    if(next_x==terminal_state[0] and next_y==terminal_state[1]):
                        v_a.append(P_a[s][a]*(rewards[idx]+gamma*20000))
                    else:
                        s1=num_states.index((next_x,next_y))
                        v_a.append(P_a[s][a]*(rewards[idx]+gamma*values_tmp[s1]))
            values[idx]=max(v_a)    #值迭代
        if max([abs(values[s] - values_tmp[s]) for s in range(N_STATES)]) < error:
            break

#policy update
    policy = np.zeros([N_STATES])
    for idx in range(N_STATES):
        the_max=-99999.
        cx=num_states[idx][0]
        cy=num_states[idx][1]
        s=num_states[idx][0]*dim+num_states[idx][1]
        flag=-1
        for a in range(N_ACTIONS):
            if(P_a[s][a]!=0):
               nx=cx+action[a][0]
               ny=cy+action[a][1]
               if(nx==terminal_state[0] and ny==terminal_state[1]):
                the_max=P_a[s][a]*(rewards[idx]+gamma*20000)
                flag=a
                continue
               s1=num_states.index((nx,ny))
               if(P_a[s][a]*(rewards[idx]+gamma*values[s1])>the_max):
                the_max=P_a[s][a]*(rewards[idx]+gamma*values[s1])
                flag=a
               
        policy[idx]=flag

    return values,policy

def show(values,type):
    k=0
    for i in range(dim):
        for j in range(dim):
            if(type=="values"):
                if(maze[i][j]==1):
                    print("%-5s"%" NAN",end=' ')
                else:
                    print("%-5.2f"%values[k],end=' ')
                    k+=1
            elif(type=="policy"):
                if(maze[i][j]==1):
                    print("%s"%"*",end=' ')
                else:
                    print(direction[int(values[k])],end=' ')
                    k+=1
                
        print()

def view(v,type):
    temp=np.zeros((dim,dim))
    k=0
    for i in range(dim):
        for j in range(dim):
            if(maze[i][j]==0):
                temp[i][j]=v[k]
                k+=1
            else:
                temp[i][j]=-9999
    plt.imshow(temp,norm="asinh")
    k=0
    for i in range(dim):
        for j in range(dim):
            if(maze[i][j]==0 and type=="value matrix"):
                plt.text(j-0.5,0.1+i,format(temp[i][j], '.2f'),fontsize=7)
            if(maze[i][j]==0 and type=="policy matrix"):
                plt.text(j-0.5,0.1+i,direction[int(v[k])],fontsize=14)
                k+=1
    plt.title(type)
    plt.xticks(())
    plt.yticks(())
    plt.show()


if __name__ == "__main__":
    init()
    v,p=value_iteration()
    show(v,"values")
    show(p,"policy")
    view(v,"value matrix")
    view(p,"policy matrix")