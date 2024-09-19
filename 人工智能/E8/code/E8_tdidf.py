# coding=gb2312
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import numpy as np

lam=0.001
emotion={1:'anger', 2:'disgust',3: 'fear',4: 'joy', 5:'sad',6: 'surprise'}

def read_file(f):#��ȡ�ı���Ϣ
    totalnum=0#��ͬ���ʵ���Ŀ
    d={}#ÿ����в�ͬ���ʵ���Ŀ
    d_total={}#�������㲻ͬ���ʵ���Ŀ
    category_num=[0 for i in range(6)]#��ͬ��е��ı���Ŀ
    total=[0 for i in range(6)]#ÿ����в��ظ����ʵ���Ŀ
    
    data=[]#��¼�ı���Ϣ
    sentence=[]#��¼�ı�
    for line in f:
        tmp=line.strip().split(" ")
        if tmp[0]=="documentId":#�����޹���Ϣ
            continue
        tag=int(tmp[0])#���
        category=int(tmp[1])#��б��
        emotion=tmp[2]#���
        sentence_tmp=''
        category_num[category-1]+=1
        for i in range(3,len(tmp)):
            if tmp[i] not in d_total:
                d_total[tmp[i]]=1
                totalnum+=1
            if (tmp[i],category-1) in d:
                d[(tmp[i],category-1)]+=1
                total[category-1]+=1
            else :
                d[(tmp[i],category-1)]=1
            sentence_tmp+=tmp[i]
            if i!=len(tmp)-1:sentence_tmp+=" "
        sentence.append(sentence_tmp)
        data.append([tag,category,emotion,sentence_tmp])
    return data,sentence,d,total,totalnum,category_num
#��ȡ�ļ�
f = open(r"C:\Users\������\Desktop\code\ai\E8\Classification\train.txt",'r')
train_data,train_sentence,d_train,total_train,total1,train_category_num=read_file(f)
f = open(r"C:\Users\������\Desktop\code\ai\E8\Classification\test.txt",'r')
test_data,test_sentence,d_test,total_test,total2,test_category_num=read_file(f)
#TfidfVectorizer��ȡ�ı���Ϣ���õ�train_matrix����
t=TfidfVectorizer()
train=t.fit_transform(train_sentence)
train_matrix=train.toarray()
#�����Լ��ı������б���
for i in range(len(test_sentence)):
    tmp=test_sentence[i].split(' ')
    test_sentence[i]=tmp
#����һЩ׼���õ�������
prob_train=[0 for i in range(6)]#ѵ����ÿ����еĵ��ʵ�tdidf��
prob_word=[[0]*len(train_matrix[0]) for i in range(6)]
pro_total_word=[0 for i in range(len(train_matrix[0]))] #ѵ����ÿ�����ʵ�tiidf��
for i in range(len(train_matrix)):#��������
    for j in range(len(train_matrix[i])):
        prob_train[train_data[i][1]-1]+=train_matrix[i][j]
        prob_word[train_data[i][1]-1][j]+=train_matrix[i][j]
        pro_total_word[j]+=train_matrix[i][j]

sum=0
for i in range(6):
    sum+=prob_train[i]#�����ĵ���tdidf�ܺ�

correct_total=0#�ܹ�Ԥ��Ե���Ŀ
correct_pred=[0 for i in range(6)]#ÿ�����Ԥ��Ե���Ŀ
test_category=[0 for i in range(6)]#���Լ�ÿ����е���Ŀ

for i in range(len(test_data)):
    pre=-1#Ԥ�����б��
    max=0#�������
    test_category[test_data[i][1]-1]+=1
    vocabulary=t.vocabulary_#TfidfVectorizer()���ֵ�
    for j in range(6):
        prob=1
        for k in range(len(test_sentence[i])):
            if test_sentence[i][k] in vocabulary:#��Ԥ�������ı����ʼ���
                if prob_word[j][vocabulary[test_sentence[i][k]]]!=0:
                    #prob_word[j][vocabulary[test_sentence[i][k]]]�õ�����Ԥ������е�tdidf��
                    #prob_train[j]Ԥ����е�tdidf��
                    #total1ѵ�������ʼ��ϴ�С
                    #train_category_num[j]Ԥ����еĵ��ʼ��ϴ�С
                    prob_tmp=(prob_word[j][vocabulary[test_sentence[i][k]]]+lam)/(prob_train[j]+train_category_num[j]*lam)
                else :#����Ԥ�������ı����ʼ��У���ѵ�������ʼ�����
                    prob_tmp=(prob_word[j][vocabulary[test_sentence[i][k]]]+lam)/(prob_train[j]+total1*lam)
            else :#����ѵ�������ʼ����У�����ʱ����
                prob_tmp=1
            prob*=prob_tmp#prob_tmpÿ�����ʵ���������
        prob*=train_category_num[j]/len(train_data)#�����������
        if max<prob:
            max=prob
            pre=j
    if pre==test_data[i][1]-1:
        correct_total+=1
        correct_pred[pre]+=1
#������
for i in range(6):
    print(emotion[i+1],":",correct_pred[i]/test_category[i])
print("Total accuracy:",correct_total/len(test_data))
