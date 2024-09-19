# coding=gb2312
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
emotion={1:'anger', 2:'disgust',3: 'fear',4: 'joy', 5:'sad',6: 'surprise'}
lam=0.001
def read_file(f):#��ȡ�ı���Ϣ
    d={}#ÿ����в�ͬ���ʵ���Ŀ
    total_dict={}#�ı����ʼ���
    category_prob=[0 for i in range(6)]#
    total=0#�ı�����
    prob_train=[0 for i in range(6)]#ÿ���������ֵĸ���(�������)
    data=[]#�ı���Ϣ
    sentence=[]#�ı�
    for line in f:
        tmp=line.strip().split(" ")
        if tmp[0]=="documentId":#�����޹���Ϣ
            continue
        tag=int(tmp[0])
        category=int(tmp[1])
        emotion=tmp[2]
        category_prob[category-1]+=1
        sentence_tmp=''
        for i in range(3,len(tmp)):
            total_dict[tmp[i]]=1
            if (tmp[i],category-1) in d:
                d[(tmp[i],category-1)]+=1
            else :
                d[(tmp[i],category-1)]=1
            total+=1
            prob_train[category-1]+=1
            sentence_tmp+=tmp[i]
            if i!=len(tmp)-1:sentence_tmp+=" "
        sentence.append(sentence_tmp)
        data.append([tag,category,emotion,sentence_tmp])
    for k in d.keys():
        d[k]=(d[k]+lam)/(prob_train[k[1]]+6*lam)
        
    for i in range(6):
        prob_train[i]/=total

    return data,sentence,prob_train,d,total,category_prob,total_dict
#��ȡ�ļ���Ϣ
f = open(r"C:\Users\������\Desktop\code\ai\E8\Classification\train.txt",'r')
train_data,train_sentence,prob_train,d_train,total_train,train_catagory_sum,train_dict=read_file(f)
f = open(r"C:\Users\������\Desktop\code\ai\E8\Classification\test.txt",'r')
test_data,test_sentence,prob_test,d_test,total_test,test_catagory_sum,test_dict=read_file(f)

for k in train_catagory_sum:
    k/=len(train_data)
right=0
for i in range(len(test_sentence)):
    tmp=test_sentence[i].split(' ')
    test_sentence[i]=tmp
correct_pred=[0 for i in range(6)]
test_category=[0 for i in range(6)]
for i in range(len(test_data)):
    test_category[test_data[i][1]-1]+=1
    max=0
    pre_id=0
    for j in range(6):
        prob_sum=1
        for k in range(len(test_sentence[i])):
            if (test_sentence[i][k],j) in d_train:#��Ԥ�������ı����ʼ���
                prob_tmp=d_train[(test_sentence[i][k],j)]
            elif test_sentence[i][k] in train_dict:#����Ԥ�������ı����ʼ��У���ѵ�������ʼ�����
                prob_tmp=lam/(prob_train[j]*total_train+6*lam)
            else :#����ѵ�������ʼ����У�����ʱ����
                prob_tmp=1
            prob_sum*=prob_tmp#prob_tmpÿ�����ʵ���������
        prob_sum*=train_catagory_sum[j]#�����������
        if prob_sum>max:
            max=prob_sum
            pre_id=j
    if pre_id==test_data[i][1]-1:
        right+=1
        correct_pred[pre_id]+=1
#����ÿ����е�׼ȷ��
for i in range(6):
    correct_pred[i]/=test_category[i]
#������
for i in range(6):
    print(emotion[i+1],":",correct_pred[i])
print("Total accuracy",right/len(test_data))


