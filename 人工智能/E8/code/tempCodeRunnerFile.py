# coding=gb2312
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import numpy as np

lam=0.001
emotion={1:'anger', 2:'disgust',3: 'fear',4: 'joy', 5:'sad',6: 'surprise'}

def read_file(f):#读取文本信息
    totalnum=0#不同单词的数目
    d={}#每种情感不同单词的数目
    d_total={}#辅助计算不同单词的数目
    category_num=[0 for i in range(6)]#不同情感的文本数目
    total=[0 for i in range(6)]#每种情感不重复单词的数目
    
    data=[]#记录文本信息
    sentence=[]#记录文本
    for line in f:
        tmp=line.strip().split(" ")
        if tmp[0]=="documentId":#跳过无关信息
            continue
        tag=int(tmp[0])#标号
        category=int(tmp[1])#情感标号
        emotion=tmp[2]#情感
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
#读取文件
f = open(r"C:\Users\刘俊杰\Desktop\code\ai\E8\Classification\train.txt",'r')
train_data,train_sentence,d_train,total_train,total1,train_category_num=read_file(f)
f = open(r"C:\Users\刘俊杰\Desktop\code\ai\E8\Classification\test.txt",'r')
test_data,test_sentence,d_test,total_test,total2,test_category_num=read_file(f)
#TfidfVectorizer提取文本信息，得到train_matrix计算
t=TfidfVectorizer()
train=t.fit_transform(train_sentence)
train_matrix=train.toarray()
#将测试集文本放入列表中
for i in range(len(test_sentence)):
    tmp=test_sentence[i].split(' ')
    test_sentence[i]=tmp
#计算一些准备用到的数据
prob_train=[0 for i in range(6)]#训练集每个情感的单词的tdidf和
prob_word=[[0]*len(train_matrix[0]) for i in range(6)]
pro_total_word=[0 for i in range(len(train_matrix[0]))] #训练集每个单词的tiidf和
for i in range(len(train_matrix)):#计算数据
    for j in range(len(train_matrix[i])):
        prob_train[train_data[i][1]-1]+=train_matrix[i][j]
        prob_word[train_data[i][1]-1][j]+=train_matrix[i][j]
        pro_total_word[j]+=train_matrix[i][j]

sum=0
for i in range(6):
    sum+=prob_train[i]#整个文档的tdidf总和

correct_total=0#总共预测对的数目
correct_pred=[0 for i in range(6)]#每种情感预测对的数目
test_category=[0 for i in range(6)]#测试集每种情感的数目

for i in range(len(test_data)):
    pre=-1#预测的情感标号
    max=0#后验概率
    test_category[test_data[i][1]-1]+=1
    vocabulary=t.vocabulary_#TfidfVectorizer()的字典
    for j in range(6):
        prob=1
        for k in range(len(test_sentence[i])):
            if test_sentence[i][k] in vocabulary:#在预测的情感文本单词集中
                if prob_word[j][vocabulary[test_sentence[i][k]]]!=0:
                    #prob_word[j][vocabulary[test_sentence[i][k]]]该单词在预测情感中的tdidf和
                    #prob_train[j]预测情感的tdidf和
                    #total1训练集单词集合大小
                    #train_category_num[j]预测情感的单词集合大小
                    prob_tmp=(prob_word[j][vocabulary[test_sentence[i][k]]]+lam)/(prob_train[j]+train_category_num[j]*lam)
                else :#不在预测的情感文本单词集中，在训练集单词集合中
                    prob_tmp=(prob_word[j][vocabulary[test_sentence[i][k]]]+lam)/(prob_train[j]+total1*lam)
            else :#不在训练集单词集合中，计算时忽略
                prob_tmp=1
            prob*=prob_tmp#prob_tmp每个单词的条件概率
        prob*=train_category_num[j]/len(train_data)#乘以先验概率
        if max<prob:
            max=prob
            pre=j
    if pre==test_data[i][1]-1:
        correct_total+=1
        correct_pred[pre]+=1
#输出结果
for i in range(6):
    print(emotion[i+1],":",correct_pred[i]/test_category[i])
print("Total accuracy:",correct_total/len(test_data))
