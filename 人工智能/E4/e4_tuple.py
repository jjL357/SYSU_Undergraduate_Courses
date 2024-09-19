# coding=gb2312
import copy
import math
from queue import PriorityQueue
import time
import os
import string
sum = 0
class Puzzle:
    def __new__(cls,list): 
        return super().__new__(cls)
    def __init__(self,list):
        self.state = tuple(list)#数码表,用元组来记录
        self.path = []#路径
        self.g = 0#g
        self.h = 0#h
        self.zero = self.state.index(0)#记录元组中0的下标
        self.hn()#计算h(n)

    def __lt__(self, other):#优先队列中的小于比较
        return self.g+self.h < other.g+other.h or (self.g+self.h == other.g+other.h and self.h < other.h)
    def hn(self):#求h(n)
        self.h = 0
        for i in range(16):
                if( self.state[i] != 0):
                    self.h += abs((self.state[i]-1)%4-i%4)+abs(int(((self.state[i])-1)/4)-i//4)
    

def A_star(p):#环检测A*
    global sum
    pri = PriorityQueue()
    pri.put(p)
    visited = set()#set去重，环检测
    while not pri.empty():
        puz = pri.get()
        #tmp.print_mat()
        if puz.state in visited :#环检测
            continue
        sum+=1
        if  puz.h == 0:#寻找到答案，返回
            print("Finding",sum,"nodes")
            return puz
        
        visited.add(puz.state)#
        #生成后继
        zero = puz.state.index(0)
        x = zero%4
        y = zero//4
        move = []
        if x > 0:move.append(-1)#左移
        if y > 0:move.append(-4)#上移
        if x < 3:move.append(1)#右移
        if y < 3:move.append(4)#下移
        for i in range(len(move)):
            l = list(puz.state)
            l[zero] = l[zero+move[i]]
            l[zero+move[i]] = 0
            puz_child = Puzzle(l)
            puz_child.g = puz.g+1
            puz_child.path = copy.deepcopy(puz.path)
            puz_child.path.append(l[zero])
            if puz_child.state in visited:continue
            pri.put(puz_child)


def IDA_start(path,puz):#列表环检测的IDA_star
    max_limit = 9999
    i = 0
    while i < max_limit:
       visited=set()
       ans= dfs(puz,visited,i,path)
       if ans == 0:
            return True
       elif ans > i:
           i = ans
       else:
           i+=1
       

def dfs(p,visited,limit,path):
    res = 9999
    visited.add(p.state)
    Pri=PriorityQueue()
    if(p.h + p.g > limit):
        return p.h + p.g#返回代价
    if p.h == 0:
        return 0
    #生成后继
    zero = p.zero
    x = zero%4
    y = zero//4
    move = []
    if x > 0:move.append(-1)#左移
    if y > 0:move.append(-4)#上移
    if x < 3:move.append(1)#右移
    if y < 3:move.append(4)#下移
    for i in range(len(move)):
        l = list(p.state)
        l[zero] = l[zero+move[i]]
        l[zero+move[i]] = 0
        puz_child = Puzzle(l)
        puz_child.g = p.g+1
        if puz_child.state in visited:
            continue
        Pri.put(puz_child)

    while not Pri.empty():
        puz = Pri.get()
        path.append(puz.state[zero])
        ans = dfs(puz,visited,limit,path)
        if ans > 0 and ans < res:#计算所有相邻状态的估价函数值，选取估价函数最小的状态进行递归
            res = ans
        if ans == 0:
            return 0 #找到答案,返回
        else:
            visited.remove(puz.state)
            path.pop()
        if res > ans:
            res = ans
    return res

def main():
    f = open(r"E4\input\E4_input5.txt",'r')

    matrix = []
    for i in range (4):
        tmp = ((f.readline()).strip()).split()
        for j in range(len(tmp)):
                matrix.append(int(tmp[j]))
    a = Puzzle(matrix)#初始状态

    
    time_start = time.time() 
    ans = A_star(a)
    print(ans.path)
    print(ans.g)
    time_end = time.time()    
    time_c = time_end - time_start 
    print('A*:','time cost', time_c, 's')

    f1 = open(r"E4\result\result5.txt",'a')

    f1.write(' '.join('%s' %id for id in ans.path))
    f1.write('\n')
    s= "Finding "+ str(sum)+" nodes"
    f1.write(s)
    f1.write('\n')
    f1.write(str(ans.g))
    f1.write('\n')
    s = 'A*: '+'time cost'+str(time_c)+ 's'
    f1.write(s)
    f1.write('\n')
    
    time_start = time.time() 
    path = []
    IDA_start(path,a)
    print(path)
    print(str(len(path)))
    time_end = time.time()    
    time_c = time_end - time_start 
    print('IDA*:','time cost ', time_c, 's')


    f1.write(' '.join('%s' %id for id in path))
    f1.write('\n')
    f1.write(str(len(path)))
    f1.write('\n')
    s ='IDA*: '+'time cost '+str(time_c)+ 's'
    f1.write(s)
    f1.write('\n')
    f1.write('\n')


if __name__ == '__main__':
    main()

