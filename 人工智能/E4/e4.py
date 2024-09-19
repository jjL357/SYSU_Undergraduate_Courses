# coding=gb2312
import copy
import math
from queue import PriorityQueue
import time
import os
import string
class Puzzle:
    def __new__(cls):
        return super().__new__(cls)
    def __init__(self):
        self.matrix=[[0]*4 for i in range(4)]#数码表
        self.path=[]#路径
        self.g=0#g
        self.h=0#h
    def __lt__(self, other):#优先队列中的小于比较
        return self.g+self.h < other.g+other.h
    def initialize_input(self,f):#输入初始化
        for i in range (4):
            tmp=((f.readline()).strip()).split()
            for j in range(len(tmp)):
                self.matrix[i][j]=int(tmp[j])
        self.hn()
    def initialize_assign(self,list):#传入列表初始化
        self.matrix=copy.deepcopy(list)
        self.hn()
    
    def hn(self):#求hn
        self.h=0
        for i in range(4):
            for j in range(4):
                if(not self.matrix[i][j]==0):
                    self.h += abs((self.matrix[i][j]-1)%4-j)+abs(int(((self.matrix[i][j])-1)/4)-i)
        
    def print_mat(self):#打印数码表
        for i in range (4):
            for j in range(4):
                print(self.matrix[i][j],end=' ')
            print()
        print(self.g+self.h)
        print("----------------------------------")

def where_zero(list):#寻找0(空)的位置
    for i in range(4):
        for j in range(4):
            if list[i][j]==0:
                return [i,j] 
    return False


def A_star(visited,pri):#列表环检测的A_start
    
    while not pri.empty():
        puz=pri.get()
        #tmp.print_mat()
        if puz.matrix in visited :
            continue
        if  puz.h==0:
            
            return puz
        
        visited.append(puz.matrix)
        tmp=where_zero(puz.matrix)
        x0=tmp[0]
        y0=tmp[1]
        x=[-1,0,0,1]
        y=[0,1,-1,0]
    
        for i in range(4):
            puz_child=Puzzle()
            x_new=x0+x[i]
            y_new=y0+y[i]
            if x_new<0 or x_new>3 or y_new<0 or y_new>3 :
                    continue
            
            for j in range(4):
                for k in range(4):
                    puz_child.matrix[j][k]=puz.matrix[j][k]
            puz_child.matrix[x0][y0]=puz_child.matrix[x_new][y_new]
            puz_child.matrix[x_new][y_new]=0
            puz_child.hn()

            puz_child.g=puz.g+1
            puz_child.path=copy.deepcopy(puz.path)
            puz_child.path.append(puz_child.matrix[x0][y0])
            pri.put(puz_child)

def A_star2(visited,pri):#字典环检测A*
    
    while not pri.empty():
        
        puz=pri.get()
        #tmp.print_mat()
        if trans(puz.matrix) in visited :
            continue
        if  puz.h==0:
            return puz
        
        visited[trans(puz.matrix)]=1
        tmp=where_zero(puz.matrix)
        x0=tmp[0]
        y0=tmp[1]
        x=[-1,0,0,1]
        y=[0,1,-1,0]
    
        for i in range(4):
            puz_child=Puzzle()
            x_new=x0+x[i]
            y_new=y0+y[i]
            if x_new<0 or x_new>3 or y_new<0 or y_new>3 :
                    continue
            
            for j in range(4):
                for k in range(4):
                    puz_child.matrix[j][k]=puz.matrix[j][k]
            puz_child.matrix[x0][y0]=puz_child.matrix[x_new][y_new]
            puz_child.matrix[x_new][y_new]=0
            puz_child.hn()

            puz_child.g=puz.g+1
            puz_child.path=copy.deepcopy(puz.path)
            puz_child.path.append(puz_child.matrix[x0][y0])
            pri.put(puz_child)


def IDa_find(path,p):#列表环检测的IDA_star
    
    max_limit=200
    for i in range(max_limit):
       visited=[]
       ans= dfs_loop(p,visited,i,path)
       if ans==True:
            return True

def dfs_loop(p,visited,limit,path):
    if(p.h+p.g>limit):
        return False
    if p.h==0:
        return True
    tmp=where_zero(p.matrix)
    x0=tmp[0]
    y0=tmp[1]
    x=[-1,0,0,1]
    y=[0,1,-1,0]
    P=PriorityQueue()
    visited.append(p.matrix)
    for i in range(4):
        x_new=x0+x[i]
        y_new=y0+y[i]
        if x_new<0 or x_new>3 or y_new<0 or y_new>3 :
                continue
        tmp1=copy.deepcopy(p.matrix)
        temp=tmp1[x0][y0]
        tmp1[x0][y0]=tmp1[x_new][y_new]
            
        tmp1[x_new][y_new]=temp
        p_child=Puzzle()
        p_child.initialize_assign(tmp1)
        p_child.g=p.g+1
        if p_child.matrix in visited :
            continue
        P.put(p_child)
    while not P.empty():
        p_child=P.get()
        path.append(p_child.matrix[x0][y0])
        ans=dfs_loop(p_child,visited,limit,path)
        if ans==False:
            path.pop()
        else :
            return True
    return False


def IDa_find_hash(path,p):#字典环检测的IDA
    
    max_limit=150
    for i in range(max_limit):
       visited={}
       ans= dfs_loop_hash(p,visited,i,path)
       if ans==True:
            return True




def dfs_loop_hash(p,visited,limit,path):
    visited[trans(p.matrix)]=1
    if(p.h+p.g>limit):
        return False
    if p.h==0:
        return True
    tmp=where_zero(p.matrix)
    x0=tmp[0]
    y0=tmp[1]
    x=[-1,0,0,1]
    y=[0,1,-1,0]
    
    for i in range(4):
        x_new=x0+x[i]
        y_new=y0+y[i]
        if x_new<0 or x_new>3 or y_new<0 or y_new>3 :
                continue
        tmp1=copy.deepcopy(p.matrix)
        temp=tmp1[x0][y0]
        tmp1[x0][y0]=tmp1[x_new][y_new]
            
        tmp1[x_new][y_new]=temp
        p_child=Puzzle()
        p_child.initialize_assign(tmp1)
        p_child.g=p.g+1
        if trans(p_child.matrix) in visited :
            continue
        path.append(tmp1[x0][y0])
        ans=dfs_loop_hash(p_child,visited,limit,path)
        if ans==False:
            del visited[trans(p_child.matrix)]
            path.pop()
        else :
            return True
    return False


def IDa_find_hash_pri(path,p):#加了优先队列
    i=-1
    max_limit=200
    while i <max_limit:
       i+=1
       visited={}
       ans= dfs_loop_hash_pri(p,visited,i,path)
       if ans==0:
            return True
       elif ans>0 and ans>i:
           i=ans
    

def dfs_loop_hash_pri(p,visited,limit,path):
    #print(p.matrix)
    flag=0
    visited[trans(p.matrix)]=1
    if(p.h+p.g>limit):
        return p.h+p.g
    if p.h==0:
        return 0
    tmp=where_zero(p.matrix)
    x0=tmp[0]
    y0=tmp[1]
    x=[-1,0,0,1]
    y=[0,1,-1,0]
    pri=PriorityQueue()
    res=10000
    
    for i in range(4):
        x_new=x0+x[i]
        y_new=y0+y[i]
        if x_new<0 or x_new>3 or y_new<0 or y_new>3 :
                continue
        tmp1=copy.deepcopy(p.matrix)
        tmp1[x0][y0]=tmp1[x_new][y_new]
        tmp1[x_new][y_new]=0
        
        p_child=Puzzle()
        p_child.initialize_assign(tmp1)
        p_child.g=p.g+1
        if trans(p_child.matrix) in visited :
            continue
        pri.put(p_child)
    while not pri.empty():
        puz=pri.get()
        path.append(puz.matrix[x0][y0])
        
        ans=dfs_loop_hash_pri(puz,visited,limit,path)
        if ans==0:
            return 0
        else:
            del visited[trans(puz.matrix)]
            path.pop()
        if res>ans:
            res=ans
    return res



def trans(list):
    s=''
    for i in range(len(list)):
        for j in range(len(list[i])):
            s+=str(list[i][j])
    return s
def tur(list):
    tur=(i for i in list)
    return tur
    pass
def a_find4(p):#元组
    visited=set()
    while not p.empty():
        
        puz=p.get()
        #tmp.print_mat()
        if tur(puz.matrix) in visited :
            continue
        if  puz.h==0:
            
            return puz
        
        visited.add(tur(puz.matrix))
        tmp=where_zero(puz.matrix)
        x0=tmp[0]
        y0=tmp[1]
        x=[-1,0,0,1]
        y=[0,1,-1,0]
    
        for i in range(4):
            puz_child=Puzzle()
            x_new=x0+x[i]
            y_new=y0+y[i]
            if x_new<0 or x_new>3 or y_new<0 or y_new>3 :
                    continue
            
            for j in range(4):
                for k in range(4):
                    puz_child.matrix[j][k]=puz.matrix[j][k]
            puz_child.matrix[x0][y0]=puz_child.matrix[x_new][y_new]
            puz_child.matrix[x_new][y_new]=0
            puz_child.hn()

            puz_child.g=puz.g+1
            puz_child.path=copy.deepcopy(puz.path)
            puz_child.path.append(puz_child.matrix[x0][y0])
            p.put(puz_child)
p=PriorityQueue()
a=Puzzle()
visited={}
f = open(r"E4\E4_input1.txt",'r')
a.initialize_input(f)
time_start = time.time() 
p.put(a)
ans=A_star2(visited,p)
#print(ans)
#ans.print_mat()
print(ans.path)
print(ans.g)
time_end = time.time()    
time_c= time_end - time_start 
print('A*:','time cost', time_c, 's')

f1=open(r"E4\result\result1.txt",'a')

f1.write(' '.join('%s' %id for id in ans.path))
f1.write('\n')
f1.write(str(ans.g))
f1.write('\n')
s='A*: '+'time cost '+str(time_c)+ 's'
f1.write(s)
f1.write('\n')



time_start = time.time() 
path=[]
IDa_find_hash_pri(path,a)
print(path)
print(str(len(path)))
time_end = time.time()    
time_c= time_end - time_start 
print('IDA*:','time cost ', time_c, 's')


f1.write(' '.join('%s' %id for id in path))
f1.write('\n')
f1.write(str(len(path)))
f1.write('\n')
s='IDA*: '+'time cost '+str(time_c)+ 's'
f1.write(s)
f1.write('\n')
f1.write('\n')


