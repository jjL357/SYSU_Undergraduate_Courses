# coding=gb2312
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import time
import numpy as np
from queue import PriorityQueue
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
def O_distance(A,B):#计算欧氏距离
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))
def O_distance_np(A,B):#np优化后的计算欧氏距离
    x1 = np.array(A)
    x2 = np.array(B)
    return np.sqrt(np.sum((x1 - x2)**2))
def Man_distance(A,B):#计算曼哈顿距离
    return (sum([abs(a - b) for (a,b) in zip(A,B)]))
def Man_distance_np(A,B):#np优化后的计算曼哈顿距离
    x1 = np.array(A)
    x2 = np.array(B)
    return (np.sum(abs(x1 - x2)))
def Lp_distance(A,B,n):#计算Lp距离(闵氏距离)
    return pow((sum([pow(abs(a-b),n) for (a,b) in zip(A,B)])),1/n)
def Lp_distance_np(A,B):#np优化后的计算Lp距离(闵氏距离)
    x1 = np.array(A)
    x2 = np.array(B)
    return pow((np.sum(abs(x1 - x2)**(len(A)))),1/len(A))
def knn1(train_set,test_set,k,train_data,test_data):#使用余弦相似度进行knn分类
    correct_num=0#统计分类正确的个数
    time_start=time.time()#记录开始时间
    m=train_set.shape[0]#训练集元素个数
    n=test_set.shape[0]#测试集元素个数
    simi=cosine_similarity(test_set,train_set)#计算余弦相似度
    for i in range(n):
        indice=np.argsort(simi[i])[-k:]#从小到大排序，获取最后k个的下标(余弦相似度越大越好)
        t=[0 for l in range(6)]#记录选取出的k个元素的中各类别占了多少元素
        distance=[0 for l in range(6)]#记录选取出的k个元素的中各类别的最小距离(余弦相似度越大越好)
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
        if max_id==test_data[i][1]-1:#若预测正确
            correct_num+=1
    time_end=time.time()#结束时间
    spend_time=time_end-time_start#花费时间
    print("距离度量方式:余弦相似度")
    print("选取k值:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/n*100,"%")
    print("--------------------------")
def knn2(train_set,test_set,k,train_data,test_data):#使用未使用np优化的欧氏距离进行knn分类
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
    print("距离度量方式:欧氏距离(未优化)")
    print("选取k值:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/len(test_data)*100,"%")
    print("--------------------------")
def knn3(train_set,test_set,k,train_data,test_data):#使用未使用np优化的曼哈顿距离进行knn分类
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
    print("距离度量方式:曼哈顿距离(未优化)")
    print("选取k值:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/len(test_data)*100,"%")
    print("--------------------------")
def knn4(train_set,test_set,k,train_data,test_data):#使用未使用np优化的Lp距离(闵氏距离)进行knn分类
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
    print("距离度量方式:Lp距离(未优化)")
    print("选取k值:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/len(test_data)*100,"%")
    print("--------------------------")
def knn5(train_set,test_set,k,train_data,test_data):#使用np优化过后的欧氏距离进行knn分类
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
    print("距离度量方式:欧氏距离(优化后)")
    print("选取k值:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/len(test_data)*100,"%")
    print("--------------------------")
def knn6(train_set,test_set,k,train_data,test_data):#使用np优化过后的曼哈顿距离进行knn分类
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
    print("距离度量方式:曼哈顿距离(优化后)")
    print("选取k值:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/len(test_data)*100,"%")
    print("--------------------------")
def knn7(train_set,test_set,k,train_data,test_data):#使用np优化过后的Lp距离(闵氏距离)进行knn分类
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
    print("距离度量方式:Lp距离(优化后)")
    print("选取k值:",k)
    print("spend time:",spend_time,"s")
    print("accuracy:",correct_num/len(test_data)*100,"%")
    print("--------------------------")
def read_file(f):
    data=[]#存放对应的信息特征
    sentence=[]#存放文本信息
    for line in f:
        tmp=line.strip().split(" ")
        if tmp[0]=="documentId":#跳过非数据文本
            continue
        tag=int(tmp[0])#文本序号
        category=int(tmp[1])#情绪
        emotion=tmp[2]##分类标签
        sentence_tmp=''
        for i in range(3,len(tmp)):#提取每一句的单词
            sentence_tmp+=tmp[i]
            if i!=len(tmp)-1:sentence_tmp+=" "
        sentence.append(sentence_tmp)
        data.append([tag,category,emotion,sentence_tmp])
    return data,sentence

f = open(r"ai\E7\Classification\test.txt",'r')
train_data,train_sentence=read_file(f)#读训练集
f = open(r"ai\E7\Classification\train.txt",'r')
test_data,test_sentence=read_file(f)#读测试集
t=TfidfVectorizer()#TfidfVectorizer提取文本特征
train=t.fit_transform(train_sentence)#读取训练集特征，此时返回一个sparse稀疏矩阵
test=t.transform(test_sentence)#读取测试集特征，此时返回一个sparse稀疏矩阵
train_matrix=train.toarray()#转换成列表
test_matrix=test.toarray()#转换成列表
k=15
knn1(train,test,k,train_data,test_data)#使用余弦相似度进行knn分类
#knn2(train,test,k,train_data,test_data)#使用未使用np优化的欧氏距离进行knn分类
#knn3(train,test,k,train_data,test_data)#使用未使用np优化的曼哈顿距离进行knn分类
#knn4(train,test,k,train_data,test_data)#使用未使用np优化的Lp距离(闵氏距离)进行knn分类
knn5(train,test,k,train_data,test_data)#使用np优化过后的欧氏距离进行knn分类
knn6(train,test,k,train_data,test_data)#使用np优化过后的曼哈顿距离进行knn分类
knn7(train,test,k,train_data,test_data)#使用np优化过后的Lp距离(闵氏距离)进行knn分类
