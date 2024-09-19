# coding=gb2312
import matplotlib.pyplot as plt
import numpy as np
import random
import string
colours=["b","g","r","c","m","y", "k"]#绘图颜色
N=0#数据的数目

mistake=0.001#达到收敛的误差
def read_file(f):#读取文本信息
    data=[]#坐标
    x=[]#横坐标
    y=[]#纵坐标
    for line in f:
        if line=="X1,X2\n":
            continue
        line=line.strip().split(",")
        data.append([float(line[0]),float(line[1])])
        x.append(float(line[0]))
        y.append(float(line[1]))
    global N
    N=len(x)
    return data,x,y

def O_distance(A,B):#计算欧氏距离
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))
def O_distance_np(A,B):#np优化后的计算欧氏距离
    x1 = np.array(A)
    x2 = np.array(B)
    return np.sqrt(np.sum((x1 - x2)**2))

def choose_center_random(x,y,k):#初始随机选择样本点作为中点
    #k中点的数目
    has_choose=set()#set去重，防止随机选取
    has_chosen=[]#已选择作为中点的样本点(用下标表示)
    has_choose.add(-1)
    for i in range(k):
        choose_index=-1
        while(choose_index in has_choose):##set去重
            choose_index=random.randint(0,N-1)#包含上下限：[a, b]
        has_choose.add(choose_index)
        has_chosen.append(choose_index)
    center=[]
    for i in has_chosen:
        center.append([x[i],y[i]])#记录中点
    return center

def choose_center_distance(x,y,k):#初始随机选择样本点作为中点
    #k中点的数目
    has_choose=set()#set去重，防止随机选取
    has_chosen=[]#已选择作为中点的样本点(用下标表示)
    has_choose.add(-1)
    distance=[0 for i in range(len(x))]
    while(len(has_chosen)<k):
        choose_index=-1
        for i in range(len(has_chosen)):
            distance_sum=0
            for j in range(len(x)):
                distance[j]=O_distance_np([x[has_chosen[i]],y[has_chosen[i]]],[x[j],y[j]])
                distance[j]*=distance[j]
                distance_sum+=distance[j]
            for j in range(len(x)):
                distance[j]/=distance_sum
            while(choose_index in has_choose):##set去重
                pro=random.uniform(0,1)#包含上下限：[a, b]
                pro_tmp=0
                for j in range(len(x)):
                    if pro>=pro_tmp and pro<=pro_tmp+distance[j]:
                        pro_tmp+=distance[j]
                        choose_index=j
                        break
        has_choose.add(choose_index)
        has_chosen.append(choose_index)
    center=[]
    for i in has_chosen:
        center.append([x[i],y[i]])#记录中点
    return center

def classify(category,center,data):#分类
    category=[[] for i in range(len(center))]#category[i]记录第i类有哪些点
    for i in range(N):
        distance=[]#记录该点到每个中点的距离
        for j in range(len(center)):
            distance.append(O_distance_np(data[i],center[j]))#计算欧氏距离
        min_index=np.argsort(distance)[0]#将距离最近的中点的类作为我们对该点的归类
        category[min_index].append(i)
    return category

def calculate_center(category,center,data):#计算每个类的各个点坐标的平均值作为新中点，更新中点
    n=0#统计需要更新的中点数目
    for i in range(len(center)):
        x_sum=0
        y_sum=0
        if len(category[i])==0:
            continue
        for j in range(len(category[i])):
            x_sum+=data[category[i][j]][0]
            y_sum+=data[category[i][j]][1]
        x_average=x_sum/(len(category[i]))#该类横坐标平均值
        y_average=y_sum/(len(category[i]))#该类纵坐标平均值
        if abs(x_average-center[i][0]) <mistake and abs(y_average-center[i][1])<mistake:#若新旧中点误差小于规定的误差，则收敛
            n+=1
        else:#未收敛，则更新
            center[i]=[x_average,y_average] 
    return n==len(center)#若全部收敛，则返回True

def draw(category,center,data):#绘图
    final_x=[[] for i in range(len(center))]#每个类点的横坐标
    final_y=[[] for i in range(len(center))]#每个类点的纵坐标
    for i in range(len(center)):
        for j in range(len(category[i])):
            final_x[i].append(data[category[i][j]][0])
            final_y[i].append(data[category[i][j]][1])
    for i in range(len(center)):
        color=colours[i]#该类点的颜色
        plt.scatter([x for x in final_x[i]],[y for y in final_y[i]],c=color,s=len([x for x in final_x[i]]))
        plt.scatter(center[i][0],center[i][1],marker='x',c="red")
    plt.show()

def calculate_SSE(category,center,data):#计算分类后的SSE
    k_SSE=0
    for i in range(len(category)):
        for j in range(len(category[i])):
            k_SSE+=O_distance_np(center[i],data[category[i][j]])
    return k_SSE

def k_means(data,x,y,k):#
    category=[[] for i in range(k)]#类
    center=choose_center_random(x,y,k)#初始随机选择样本点作为中点
    #center=choose_center_distance(x,y,k)#根据对已有中心距离选择中心
    stop=False
    while(stop==False):#中点收敛则停止
        category=classify(category,center,data)#分类
        stop=calculate_center(category,center,data)#更新中点，中点收敛则stop=True
    k_SSE=calculate_SSE(category,center,data)#计算SSE
    print(k_SSE)#输出SSE
    draw(category,center,data)#绘图
    return k_SSE

def main():
    f = open(r"ai\E9\kmeans_data.csv",'r')
    data,x,y=read_file(f)
    SSE_list=[]
    for k in range(1,8):
        k_SSE=k_means(data,x,y,k)
        SSE_list.append(k_SSE)
    # 绘图
    plt.plot([i for i in range(1,8)],SSE_list,)
    plt.title("SSE")
    plt.show()


    #plt.plot([k for k in range(1,len(SSE_list)+1)],SSE_list)


if __name__ == '__main__':
    main()