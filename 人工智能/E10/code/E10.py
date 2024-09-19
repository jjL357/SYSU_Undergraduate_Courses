# coding=gb2312
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
learning_rate=0.8
iter=100
class Layer():#������Ԫ����
    def __init__(self,input_size):
        self.bias=0#ƫ����
        self.input=[]#����
        self.output=0#���(���data1������ÿ����Ԫֻ��һ�����)
        self.weight=np.random.rand(input_size)#Ȩ��

    def layer_function(self,input):#����Ԫ�ļ����
        x=np.dot(self.weight,np.array(input))+self.bias#�������Ȩ�ؼ�ƫ��
        self.input=[i for i in input]#��¼����Ԫ������
        return sigmoid(x)
    
def sigmoid(x):#sigmoid����  
    return 1 / (1 + np.exp(-x))  

def forward(input,layers):#ǰ�򴫲�(�������data1,���ǰһ���������Ԫ������Ǻ�һ��ÿһ����Ԫ������)
    layers_output=[]#����ÿһ����Ԫ�����
    for i in range(len(layers)):
        layers_output.append([])
        for j in range(len(layers[i])):
            if i==0:#��һ�������������
                layers[i][j].output=layers[i][j].layer_function(input)
                layers_output[-1].append(layers[i][j].output)
            else:#�������������ǰһ��������Ԫ�����
                layers[i][j].output=layers[i][j].layer_function(layers_output[i-1])
                layers_output[-1].append(layers[i][j].output)
    output=layers[-1][-1].output
    return output#�����������

def backward(input,layers,correct_output):#���򴫲�
    d=[[] for i in range(len(layers))]#��¼ÿһ�����ʧ����
    loss=0#��¼��ʧֵ
    old_weight=[[] for i in range(len(layers))]#��¼δ���µ�ÿһ��ÿһ����Ԫ��Ȩ��
    for i in range(len(layers)):
        index=len(layers)-i-1#�Ӻ���ǰ����
        for j in range(len(layers[index])):
            old_weight[index].append([w for w in layers[index][j].weight])
            output=layers[index][j].output#���
            if i==0:#���һ�㼴�����
                loss+=(correct_output-output)**2#������ʧ
                d_out=output*(1-output)*(correct_output-output)#������ʧ�����ĵ���
                d[index].append(d_out)
            else:#���ز�
                #������ʧ�����ĵ���
                d_out=output*(1-output)
                tmp_sum=0
                for k in range(len(layers[index+1])):
                    tmp_sum+=old_weight[index+1][k][j]*d[index+1][k]
                d_out*=tmp_sum
                d[index].append(d_out)
            #��ÿȨ�ؽ��и���
            for k in range(len(layers[index][j].weight)):
                layers[index][j].weight[k]+=d_out*learning_rate*layers[index][j].input[k]
            layers[index][j].bias+=d_out*learning_rate
    return loss

def train(times,input,correct_output,layers):#ѵ������
    loss=[0 for i in range(times)]#ÿ��ѵ������ʧ
    while(times>0):#timesѵ������
        for i in range(len(input)):
            forward(input[i],layers)#ǰ�򴫲�
            loss[iter-times]+=backward(input[i],layers,correct_output[i])#���򴫲�
        times-=1
    return loss        
                
def one(x):#��һ��
    o=[]#��¼�������������ֵ����Сֵ�����滭ͼ��Ҫ����һ��
    for j in range(len(x[0])):
        min_x=100000000
        max_x=-10000000
        for i in range(len(x)):
            if max_x<x[i][j]:
                max_x=x[i][j]
            if min_x>x[i][j]:
                min_x=x[i][j]
        for i in range(len(x)):
            x[i][j]=2*(x[i][j]-min_x)/(max_x-min_x)-1#��һ��
        o.append([max_x,min_x])
    return x,o
            
def find_y(x1,theta,o):#���һ��������߽߱��yֵ
    return [((-theta[0]-theta[1]*((x_1-o[0][1])*2/(o[0][0]-o[0][1])-1))/theta[2]+1)/2*(o[1][0]-o[1][1])+o[1][1] for x_1 in x1]

def read_file(f,k):#��ȡ�ı���Ϣ
    input=[]#����
    output=[]#���
    accept=[]#¼ȡ����������
    refuse=[]#�ܾ�����������
    for line in f:
        input.append([])
        line=line.strip().split(",")
        for i in range(1,k):
            input[-1].append(float(line[0])**i)
            input[-1].append(float(line[1])**i)
        output.append(int(line[2]))
    for i in range(len(input)):
        if output[i]:
            accept.append([input[i][0],input[i][1]])
        else:
            refuse.append([input[i][0],input[i][1]])
    return input,output,accept,refuse

def calculate_accuracy(layers,input,output):#����Ԥ��׼ȷ��
    c=0
    for i in range(len(input)):
        pred=forward(input[i],layers)
        if pred>=0.5:
            pred=1
        else :
            pred=0
        if pred==output[i]:
            c+=1
    print("accuracy",":",c/len(input))

def draw_boundary(L,accept,refuse,o):#�������߽߱�
    w=[L.bias]#w���������ƫ�ú�Ȩ��
    for i in L.weight:
        w.append(i)
    x = np.array([0, 100])
    y=find_y(x,w,o)#���һ��������߽߱��yֵ
    plt.plot(x,y,color='r',label="decision boundary")
    for i in range(len(accept)):
        plt.scatter(accept[i][0],accept[i][1],c="g")
    for i in range(len(refuse)):
        plt.scatter(refuse[i][0],refuse[i][1],c="b")
    plt.show()

def draw_loss(loss):#������ʧ����
    plt.plot([i for i in range(1,iter+1)],loss,color="r",label="loss")
    plt.show()

def nn(k):#������Ԥ��
    f = open(r"ai\E10\data1.txt",'r')
    input,output,accept,refuse=read_file(f,k)#��ȡ��������
    input,o=one(input)#���������ݽ��й�һ
    L=Layer(len(input[0]))#���data1ֻ��Ҫһ������������㣬��������㣬�����ֱ������
    layers=[[L]]#ÿһ�����Ԫ
    loss=train(iter,input,output,layers)#ѵ��
    calculate_accuracy(layers,input,output)#����Ԥ��׼ȷ��
    draw_boundary(L,accept,refuse,o)#�������߽߱�
    draw_loss(loss)#������ʧ����

def main():
    nn(2)

if __name__ == '__main__':
    main()

