# coding=gb2312
import matplotlib.pyplot as plt
import numpy as np
import random
import string
colours=["b","g","r","c","m","y", "k"]#��ͼ��ɫ
N=0#���ݵ���Ŀ

mistake=0.001#�ﵽ���������
def read_file(f):#��ȡ�ı���Ϣ
    data=[]#����
    x=[]#������
    y=[]#������
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

def O_distance(A,B):#����ŷ�Ͼ���
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))
def O_distance_np(A,B):#np�Ż���ļ���ŷ�Ͼ���
    x1 = np.array(A)
    x2 = np.array(B)
    return np.sqrt(np.sum((x1 - x2)**2))

def choose_center_random(x,y,k):#��ʼ���ѡ����������Ϊ�е�
    #k�е����Ŀ
    has_choose=set()#setȥ�أ���ֹ���ѡȡ
    has_chosen=[]#��ѡ����Ϊ�е��������(���±��ʾ)
    has_choose.add(-1)
    for i in range(k):
        choose_index=-1
        while(choose_index in has_choose):##setȥ��
            choose_index=random.randint(0,N-1)#���������ޣ�[a, b]
        has_choose.add(choose_index)
        has_chosen.append(choose_index)
    center=[]
    for i in has_chosen:
        center.append([x[i],y[i]])#��¼�е�
    return center

def choose_center_distance(x,y,k):#��ʼ���ѡ����������Ϊ�е�
    #k�е����Ŀ
    has_choose=set()#setȥ�أ���ֹ���ѡȡ
    has_chosen=[]#��ѡ����Ϊ�е��������(���±��ʾ)
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
            while(choose_index in has_choose):##setȥ��
                pro=random.uniform(0,1)#���������ޣ�[a, b]
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
        center.append([x[i],y[i]])#��¼�е�
    return center

def classify(category,center,data):#����
    category=[[] for i in range(len(center))]#category[i]��¼��i������Щ��
    for i in range(N):
        distance=[]#��¼�õ㵽ÿ���е�ľ���
        for j in range(len(center)):
            distance.append(O_distance_np(data[i],center[j]))#����ŷ�Ͼ���
        min_index=np.argsort(distance)[0]#������������е������Ϊ���ǶԸõ�Ĺ���
        category[min_index].append(i)
    return category

def calculate_center(category,center,data):#����ÿ����ĸ����������ƽ��ֵ��Ϊ���е㣬�����е�
    n=0#ͳ����Ҫ���µ��е���Ŀ
    for i in range(len(center)):
        x_sum=0
        y_sum=0
        if len(category[i])==0:
            continue
        for j in range(len(category[i])):
            x_sum+=data[category[i][j]][0]
            y_sum+=data[category[i][j]][1]
        x_average=x_sum/(len(category[i]))#���������ƽ��ֵ
        y_average=y_sum/(len(category[i]))#����������ƽ��ֵ
        if abs(x_average-center[i][0]) <mistake and abs(y_average-center[i][1])<mistake:#���¾��е����С�ڹ涨����������
            n+=1
        else:#δ�����������
            center[i]=[x_average,y_average] 
    return n==len(center)#��ȫ���������򷵻�True

def draw(category,center,data):#��ͼ
    final_x=[[] for i in range(len(center))]#ÿ�����ĺ�����
    final_y=[[] for i in range(len(center))]#ÿ������������
    for i in range(len(center)):
        for j in range(len(category[i])):
            final_x[i].append(data[category[i][j]][0])
            final_y[i].append(data[category[i][j]][1])
    for i in range(len(center)):
        color=colours[i]#��������ɫ
        plt.scatter([x for x in final_x[i]],[y for y in final_y[i]],c=color,s=len([x for x in final_x[i]]))
        plt.scatter(center[i][0],center[i][1],marker='x',c="red")
    plt.show()

def calculate_SSE(category,center,data):#���������SSE
    k_SSE=0
    for i in range(len(category)):
        for j in range(len(category[i])):
            k_SSE+=O_distance_np(center[i],data[category[i][j]])
    return k_SSE

def k_means(data,x,y,k):#
    category=[[] for i in range(k)]#��
    center=choose_center_random(x,y,k)#��ʼ���ѡ����������Ϊ�е�
    #center=choose_center_distance(x,y,k)#���ݶ��������ľ���ѡ������
    stop=False
    while(stop==False):#�е�������ֹͣ
        category=classify(category,center,data)#����
        stop=calculate_center(category,center,data)#�����е㣬�е�������stop=True
    k_SSE=calculate_SSE(category,center,data)#����SSE
    print(k_SSE)#���SSE
    draw(category,center,data)#��ͼ
    return k_SSE

def main():
    f = open(r"ai\E9\kmeans_data.csv",'r')
    data,x,y=read_file(f)
    SSE_list=[]
    for k in range(1,8):
        k_SSE=k_means(data,x,y,k)
        SSE_list.append(k_SSE)
    # ��ͼ
    plt.plot([i for i in range(1,8)],SSE_list,)
    plt.title("SSE")
    plt.show()


    #plt.plot([k for k in range(1,len(SSE_list)+1)],SSE_list)


if __name__ == '__main__':
    main()