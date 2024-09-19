# coding=gb2312
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
emotion={1:'anger', 2:'disgust',3: 'fear',4: 'joy', 5:'sad',6: 'surprise'}
lam=0.001
def read_file(f):#读取文本信息
    d={}#每种情感不同单词的数目
    total_dict={}#文本单词集合
    category_prob=[0 for i in range(6)]#
    total=0#文本总数
    prob_train=[0 for i in range(6)]#每种情绪出现的概率(先验概率)
    data=[]#文本信息
    sentence=[]#文本
    for line in f:
        tmp=line.strip().split(" ")
        if tmp[0]=="documentId":#跳过无关信息
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
#读取文件信息
f = open(r"C:\Users\刘俊杰\Desktop\code\ai\E8\Classification\train.txt",'r')
train_data,train_sentence,prob_train,d_train,total_train,train_catagory_sum,train_dict=read_file(f)
f = open(r"C:\Users\刘俊杰\Desktop\code\ai\E8\Classification\test.txt",'r')
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
            if (test_sentence[i][k],j) in d_train:#在预测的情感文本单词集中
                prob_tmp=d_train[(test_sentence[i][k],j)]
            elif test_sentence[i][k] in train_dict:#不在预测的情感文本单词集中，在训练集单词集合中
                prob_tmp=lam/(prob_train[j]*total_train+6*lam)
            else :#不在训练集单词集合中，计算时忽略
                prob_tmp=1
            prob_sum*=prob_tmp#prob_tmp每个单词的条件概率
        prob_sum*=train_catagory_sum[j]#乘以先验概率
        if prob_sum>max:
            max=prob_sum
            pre_id=j
    if pre_id==test_data[i][1]-1:
        right+=1
        correct_pred[pre_id]+=1
#计算每种情感的准确率
for i in range(6):
    correct_pred[i]/=test_category[i]
#输出结果
for i in range(6):
    print(emotion[i+1],":",correct_pred[i])
print("Total accuracy",right/len(test_data))


