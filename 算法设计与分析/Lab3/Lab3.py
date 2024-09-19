import copy
import time
def is_valid(m,n,row,col):#判断该位置是否合法
    if row<0 or col<0 or row>=m or col>=n:
        return False
    return True

def where_not_supervise(supervise):#寻找第一个没有被监督的房间的位置
    for i in range(0,len(supervise)):
        for j in range(0,len(supervise[0])):
            if supervise[i][j]==0:
                return i,j
    return -1,-1

def place(board,supervise,row,col):#在row,col位置放置一个哨兵
    board[row][col]=1
    # 同时需要记录在这一步放置哨兵新增的被监督房间的位置，方便后面的回溯操作
    m,n = len(board),len(board[0])
    dirs = [(0,0),(1,0),(-1,0),(0,1),(0,-1)]
    new_supervised = []#记录在这一步放置哨兵新增的被监督房间的位置
    for dir in dirs:
        new_row,new_col = row+dir[0],col+dir[1]
        if is_valid(m,n,new_row,new_col) and supervise[new_row][new_col]==0:
            supervise[new_row][new_col]=1
            new_supervised.append((new_row,new_col))
    return new_supervised

def dfs(board,supervise,best_cost,best_plan,cost,supervise_count):
    m,n = len(board),len(board[0])
    # 分支限界
    if cost+(m*n-supervise_count)//5>best_cost:
        return best_cost,best_plan
    #所有房间都已被监督
    if supervise_count == m*n: 
        if cost < best_cost: # 与之前的最优解比较
            best_cost = cost
            best_plan = copy.deepcopy(board)
        return best_cost,best_plan

    # 寻找第一个未被监督的房间位置
    row,col = where_not_supervise(supervise)
    dirs = [(1,0),(0,0),(0,1)]
    # 在第一个未被监督的房间的原位置、右位置、下位置放置哨兵搜索
    for dir in dirs:
        new_row,new_col = row+dir[0],col+dir[1]
        if is_valid(m,n,new_row,new_col)==False:
            continue
        be_supervise = place(board,supervise,new_row,new_col)
        best_cost,best_plan = dfs(board,supervise,best_cost,best_plan,cost+1,supervise_count+len(be_supervise))
        # 回溯操作，将此步放置哨兵的效果回退
        board[new_row][new_col] = 0
        for ele in be_supervise:
            supervise[ele[0]][ele[1]] = 0        
    return best_cost,best_plan

def read_input(input):#读文件获取矩阵阵列的大小
    MN = []
    with open(input, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = (line.strip()).split()
            MN.append((int(tmp[0]),int(tmp[1])))     
    return MN

def write_output(best_cost,best_plan,output_path):#将结果输出到文件中
    with open(output_path, 'a') as f:
        f.write(str(best_cost)+'\n')
        for line in best_plan:
            f.write(str(line)+'\n')
            
def solve(MN,output_path):
    for mn in MN:
        time_start = time.time()
        m,n = mn[0],mn[1]
        best_cost = (m*n)//3+1 # 搜索过程中记录的最优放置哨兵数
        best_plan = None # 搜索过程中记录的最优放置方法
        board = [[0]*n for i in range(m)] # 哨兵放置矩阵
        supervised = [[0]*n for i in range(m)] # 记录房间是否被监督的状态
        best_cost,best_plan = dfs(board,supervised,best_cost,best_plan,0,0)
        time_end = time.time()
        time_c= time_end - time_start   
        print('time cost', time_c, 's')
        write_output(best_cost,best_plan,output_path)
        
        
if __name__ == "__main__":
    input_path = "D:\d_code\\algorithm\Lab3\input.txt"
    output_path = "D:\d_code\\algorithm\Lab3\output.txt"
    MN = read_input(input_path)
    solve(MN,output_path)
    

