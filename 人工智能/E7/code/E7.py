# coding=gb2312
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import time
import numpy as np
from queue import PriorityQueue
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
def O_distance(A,B):#����ŷ�Ͼ���
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))
def O_distance_np(A,B):#np�Ż���ļ���ŷ�Ͼ���
    x1 = np.array(A)
    x2 = np.array(B)
    return np.sqrt(np.sum((x1 - x2)**2))
def Man_distance(A,B):#���������پ���
    return (sum([abs(a - b) for (a,b) in zip(A,B)]))
def Man_distance_np(A,B):#np�Ż���ļ��������پ���
    x1 = np.array(A)
    x2 = np.array(B)
    return (np.sum(abs(x1 - x2)))
def Lp_distance(A,B,n):#����Lp����(���Ͼ���)
    return pow((sum([pow(abs(a-b),n) for (a,b) in zip(A,B)])),1/n)
def Lp_distance_np(A,B):#np�Ż���ļ���Lp����(���Ͼ���)
    x1 = np.array(A)
    x2 = np.array(B)
    return pow((np.sum(abs(x1 - x2)**(len(A)))),1/len(A))
def knn1(train_set,test_set,k,train_data,test_data):#ʹ���������ƶȽ���knn����
    correct_num=0#ͳ�Ʒ�����ȷ�ĸ���
    time_start=time.time()#��¼��ʼʱ��
    m=train_set.shape[0]#ѵ����Ԫ�ظ���
    n=test_set.shape[0]#���Լ�Ԫ�ظ���
    simi=cosine_similarity(test_set,train_set)#�����������ƶ�
    for i in range(n):
        indice=np.argsort(simi[i])[-k:]#��С�������򣬻�ȡ���k�����±�(�������ƶ�Խ��Խ��)
        t=[0 for l in range(6)]#��¼ѡȡ����k��Ԫ�ص��и����ռ�˶���Ԫ��
        distance=[0 for l in range(6)]#��¼ѡȡ����k��Ԫ�ص��и�������С����(�������ƶ�Խ��Խ��)
        for j in range(k):
            t[train_data[indice[j]][1]-1]+=1
            distance[train_data[indice[j]][1]-1]+=simi[i][indice[j]]
        max_id=0
        max=0
        max_weight=-10000000000
        for p in range(len(t)):
            if max<t[p]:
                max=t[p]
                max_id=p
                max_weight=distance[p]
            elif max==t[p] and max_weight<distance[p]:
                max=t[p]
                max_id=p
                max_weight=distance[p]
        if max_id==test_data[i][1]-1:#��Ԥ����ȷ
            correct_num+=1
    time_end=time.time()#����ʱ��
    spend_time=time_end-time_start#����ʱ��
    print("���������ʽ:�������ƶ�")
    print("ѡȡkֵ:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/n*100,"%")
    print("--------------------------")
def knn2(train_set,test_set,k,train_data,test_data):#ʹ��δʹ��np�Ż���ŷ�Ͼ������knn����
    correct_num=0
    time_start=time.time()
    train_matrix=train_set.toarray()
    test_matrix=test_set.toarray()
    for i in range(len(test_matrix)):
        distance=[]
        for j in range(len(train_matrix)):
            distance.append(O_distance(test_matrix[i],train_matrix[j]))
        indice=np.argsort(distance)[0:k]
        t=[0 for l in range(6)]
        distance_tmp=[0 for l in range(6)]
        for j in range(k):
            t[train_data[indice[j]][1]-1]+=1
            distance_tmp[train_data[indice[j]][1]-1]+=distance[indice[j]]
        max_id=0
        max=0
        max_weight=100000000000000000
        for p in range(len(t)):
            if max<t[p]:
                max=t[p]
                max_id=p
                max_weight=distance_tmp[p]
            elif max==t[p] and max_weight>distance_tmp[p]:
                max=t[p]
                max_id=p
                max_weight=distance_tmp[p]
        if max_id==test_data[i][1]-1:
            correct_num+=1
    time_end=time.time()
    spend_time=time_end-time_start
    print("���������ʽ:ŷ�Ͼ���(δ�Ż�)")
    print("ѡȡkֵ:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/len(test_data)*100,"%")
    print("--------------------------")
def knn3(train_set,test_set,k,train_data,test_data):#ʹ��δʹ��np�Ż��������پ������knn����
    correct_num=0
    time_start=time.time()
    train_matrix=train_set.toarray()
    test_matrix=test_set.toarray()
    for i in range(len(test_matrix)):
        distance=[]
        for j in range(len(train_matrix)):
            distance.append(Man_distance(test_matrix[i],train_matrix[j]))
        indice=np.argsort(distance)[0:k]
        t=[0 for l in range(6)]
        distance_tmp=[0 for l in range(6)]
        for j in range(k):
            t[train_data[indice[j]][1]-1]+=1
            distance_tmp[train_data[indice[j]][1]-1]+=distance[indice[j]]
        max_id=0
        max=0
        max_weight=100000000000000000
        for p in range(len(t)):
            if max<t[p]:
                max=t[p]
                max_id=p
                max_weight=distance_tmp[p]
            elif max==t[p] and max_weight>distance_tmp[p]:
                max=t[p]
                max_id=p
                max_weight=distance_tmp[p]
        if max_id==test_data[i][1]-1:
            correct_num+=1
    time_end=time.time()
    spend_time=time_end-time_start
    print("���������ʽ:�����پ���(δ�Ż�)")
    print("ѡȡkֵ:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/len(test_data)*100,"%")
    print("--------------------------")
def knn4(train_set,test_set,k,train_data,test_data):#ʹ��δʹ��np�Ż���Lp����(���Ͼ���)����knn����
    correct_num=0
    time_start=time.time()
    train_matrix=train_set.toarray()
    test_matrix=test_set.toarray()
    for i in range(len(test_matrix)):
        distance=[]
        for j in range(len(train_matrix)):
            distance.append(Lp_distance(test_matrix[i],train_matrix[j],len(test_matrix[i])))
        indice=np.argsort(distance)[0:k]
        t=[0 for l in range(6)]
        distance_tmp=[0 for l in range(6)]
        for j in range(k):
            t[train_data[indice[j]][1]-1]+=1
            distance_tmp[train_data[indice[j]][1]-1]+=distance[indice[j]]
        max_id=0
        max=0
        max_weight=100000000000000000
        for p in range(len(t)):
            if max<t[p]:
                max=t[p]
                max_id=p
                max_weight=distance_tmp[p]
            elif max==t[p] and max_weight>distance_tmp[p]:
                max=t[p]
                max_id=p
                max_weight=distance_tmp[p]
        if max_id==test_data[i][1]-1:
            correct_num+=1
    time_end=time.time()
    spend_time=time_end-time_start
    print("���������ʽ:Lp����(δ�Ż�)")
    print("ѡȡkֵ:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/len(test_data)*100,"%")
    print("--------------------------")
def knn5(train_set,test_set,k,train_data,test_data):#ʹ��np�Ż������ŷ�Ͼ������knn����
    correct_num=0
    time_start=time.time()
    train_matrix=np.array(train_set.toarray())
    test_matrix=np.array(test_set.toarray())
    for i in range(len(test_matrix)):
        distance=[]
        for j in range(len(train_matrix)):
            distance.append(O_distance_np(test_matrix[i],train_matrix[j]))
        indice=np.argsort(distance)[0:k]
        t=[0 for l in range(6)]
        distance_tmp=[0 for l in range(6)]
        for j in range(k):
            t[train_data[indice[j]][1]-1]+=1
            distance_tmp[train_data[indice[j]][1]-1]+=distance[indice[j]]
        max_id=0
        max=0
        max_weight=100000000000000000
        for p in range(len(t)):
            if max<t[p]:
                max=t[p]
                max_id=p
                max_weight=distance_tmp[p]
            elif max==t[p] and max_weight>distance_tmp[p]:
                max=t[p]
                max_id=p
                max_weight=distance_tmp[p]
        if max_id==test_data[i][1]-1:
            correct_num+=1  
    time_end=time.time()
    spend_time=time_end-time_start
    print("���������ʽ:ŷ�Ͼ���(�Ż���)")
    print("ѡȡkֵ:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/len(test_data)*100,"%")
    print("--------------------------")
def knn6(train_set,test_set,k,train_data,test_data):#ʹ��np�Ż�����������پ������knn����
    correct_num=0
    time_start=time.time()
    train_matrix=np.array(train_set.toarray())
    test_matrix=np.array(test_set.toarray())
    for i in range(len(test_matrix)):
        distance=[]
        for j in range(len(train_matrix)):
            distance.append(Man_distance_np(test_matrix[i],train_matrix[j]))
        indice=np.argsort(distance)[0:k]
        t=[0 for l in range(6)]
        distance_tmp=[0 for l in range(6)]
        for j in range(k):
            t[train_data[indice[j]][1]-1]+=1
            distance_tmp[train_data[indice[j]][1]-1]+=distance[indice[j]]
        max_id=0
        max=0
        max_weight=100000000000000000
        for p in range(len(t)):
            if max<t[p]:
                max=t[p]
                max_id=p
                max_weight=distance_tmp[p]
            elif max==t[p] and max_weight>distance_tmp[p]:
                max=t[p]
                max_id=p
                max_weight=distance_tmp[p]
        if max_id==test_data[i][1]-1:
            correct_num+=1  
    time_end=time.time()
    spend_time=time_end-time_start
    print("���������ʽ:�����پ���(�Ż���)")
    print("ѡȡkֵ:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/len(test_data)*100,"%")
    print("--------------------------")
def knn7(train_set,test_set,k,train_data,test_data):#ʹ��np�Ż������Lp����(���Ͼ���)����knn����
    correct_num=0
    time_start=time.time()
    train_matrix=np.array(train_set.toarray())
    test_matrix=np.array(test_set.toarray())
    for i in range(len(test_matrix)):
        distance=[]
        for j in range(len(train_matrix)):
            distance.append(Lp_distance_np(test_matrix[i],train_matrix[j]))
        indice=np.argsort(distance)[0:k]
        t=[0 for l in range(6)]
        distance_tmp=[0 for l in range(6)]
        for j in range(k):
            t[train_data[indice[j]][1]-1]+=1
            distance_tmp[train_data[indice[j]][1]-1]+=distance[indice[j]]
        max_id=0
        max=0
        max_weight=100000000000000000
        for p in range(len(t)):
            if max<t[p]:
                max=t[p]
                max_id=p
                max_weight=distance_tmp[p]
            elif max==t[p] and max_weight>distance_tmp[p]:
                max=t[p]
                max_id=p
                max_weight=distance_tmp[p]
        if max_id==test_data[i][1]-1:
            correct_num+=1 
    time_end=time.time()
    spend_time=time_end-time_start
    print("���������ʽ:Lp����(�Ż���)")
    print("ѡȡkֵ:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/len(test_data)*100,"%")
    print("--------------------------")
def read_file(f):
    data=[]#��Ŷ�Ӧ����Ϣ����
    sentence=[]#����ı���Ϣ
    for line in f:
        tmp=line.strip().split(" ")
        if tmp[0]=="documentId":#�����������ı�
            continue
        tag=int(tmp[0])#�ı����
        category=int(tmp[1])#����
        emotion=tmp[2]##�����ǩ
        sentence_tmp=''
        for i in range(3,len(tmp)):#��ȡÿһ��ĵ���
            sentence_tmp+=tmp[i]
            if i!=len(tmp)-1:sentence_tmp+=" "
        sentence.append(sentence_tmp)
        data.append([tag,category,emotion,sentence_tmp])
    return data,sentence

f = open(r"ai\E7\Classification\test.txt",'r')
train_data,train_sentence=read_file(f)#��ѵ����
f = open(r"ai\E7\Classification\train.txt",'r')
test_data,test_sentence=read_file(f)#�����Լ�
t=TfidfVectorizer()#TfidfVectorizer��ȡ�ı�����
train=t.fit_transform(train_sentence)#��ȡѵ������������ʱ����һ��sparseϡ�����
test=t.transform(test_sentence)#��ȡ���Լ���������ʱ����һ��sparseϡ�����
train_matrix=train.toarray()#ת�����б�
test_matrix=test.toarray()#ת�����б�
k=15
knn1(train,test,k,train_data,test_data)#ʹ���������ƶȽ���knn����
#knn2(train,test,k,train_data,test_data)#ʹ��δʹ��np�Ż���ŷ�Ͼ������knn����
#knn3(train,test,k,train_data,test_data)#ʹ��δʹ��np�Ż��������پ������knn����
#knn4(train,test,k,train_data,test_data)#ʹ��δʹ��np�Ż���Lp����(���Ͼ���)����knn����
knn5(train,test,k,train_data,test_data)#ʹ��np�Ż������ŷ�Ͼ������knn����
knn6(train,test,k,train_data,test_data)#ʹ��np�Ż�����������پ������knn����
knn7(train,test,k,train_data,test_data)#ʹ��np�Ż������Lp����(���Ͼ���)����knn����
