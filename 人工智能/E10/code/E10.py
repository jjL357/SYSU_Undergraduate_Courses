# coding=gb2312
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
learning_rate=0.8
iter=100
class Layer():#定义神经元的类
    def __init__(self,input_size):
        self.bias=0#偏置项
        self.input=[]#输入
        self.output=0#输出(针对data1，设置每个神经元只有一个输出)
        self.weight=np.random.rand(input_size)#权重

    def layer_function(self,input):#该神经元的激活函数
        x=np.dot(self.weight,np.array(input))+self.bias#输入乘以权重加偏置
        self.input=[i for i in input]#记录该神经元的输入
        return sigmoid(x)
    
def sigmoid(x):#sigmoid函数  
    return 1 / (1 + np.exp(-x))  

def forward(input,layers):#前向传播(这里针对data1,设计前一层的所有神经元的输出是后一层每一个神经元的输入)
    layers_output=[]#记入每一层神经元的输出
    for i in range(len(layers)):
        layers_output.append([])
        for j in range(len(layers[i])):
            if i==0:#第一层的输入是数据
                layers[i][j].output=layers[i][j].layer_function(input)
                layers_output[-1].append(layers[i][j].output)
            else:#其他层的输入是前一层所有神经元的输出
                layers[i][j].output=layers[i][j].layer_function(layers_output[i-1])
                layers_output[-1].append(layers[i][j].output)
    output=layers[-1][-1].output
    return output#返回最后的输出

def backward(input,layers,correct_output):#反向传播
    d=[[] for i in range(len(layers))]#记录每一层的损失导数
    loss=0#记录损失值
    old_weight=[[] for i in range(len(layers))]#记录未更新的每一层每一个神经元的权重
    for i in range(len(layers)):
        index=len(layers)-i-1#从后向前传播
        for j in range(len(layers[index])):
            old_weight[index].append([w for w in layers[index][j].weight])
            output=layers[index][j].output#输出
            if i==0:#最后一层即输出层
                loss+=(correct_output-output)**2#计算损失
                d_out=output*(1-output)*(correct_output-output)#计算损失函数的导数
                d[index].append(d_out)
            else:#隐藏层
                #计算损失函数的导数
                d_out=output*(1-output)
                tmp_sum=0
                for k in range(len(layers[index+1])):
                    tmp_sum+=old_weight[index+1][k][j]*d[index+1][k]
                d_out*=tmp_sum
                d[index].append(d_out)
            #对每权重进行更新
            for k in range(len(layers[index][j].weight)):
                layers[index][j].weight[k]+=d_out*learning_rate*layers[index][j].input[k]
            layers[index][j].bias+=d_out*learning_rate
    return loss

def train(times,input,correct_output,layers):#训练函数
    loss=[0 for i in range(times)]#每次训练的损失
    while(times>0):#times训练次数
        for i in range(len(input)):
            forward(input[i],layers)#前向传播
            loss[iter-times]+=backward(input[i],layers,correct_output[i])#反向传播
        times-=1
    return loss        
                
def one(x):#归一化
    o=[]#记录数据样本的最大值和最小值，后面画图需要返归一化
    for j in range(len(x[0])):
        min_x=100000000
        max_x=-10000000
        for i in range(len(x)):
            if max_x<x[i][j]:
                max_x=x[i][j]
            if min_x>x[i][j]:
                min_x=x[i][j]
        for i in range(len(x)):
            x[i][j]=2*(x[i][j]-min_x)/(max_x-min_x)-1#归一化
        o.append([max_x,min_x])
    return x,o
            
def find_y(x1,theta,o):#逆归一化计算决策边界的y值
    return [((-theta[0]-theta[1]*((x_1-o[0][1])*2/(o[0][0]-o[0][1])-1))/theta[2]+1)/2*(o[1][0]-o[1][1])+o[1][1] for x_1 in x1]

def read_file(f,k):#读取文本信息
    input=[]#输入
    output=[]#输出
    accept=[]#录取的样本数据
    refuse=[]#拒绝的样本数据
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

def calculate_accuracy(layers,input,output):#计算预测准确率
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

def draw_boundary(L,accept,refuse,o):#画出决策边界
    w=[L.bias]#w存放输出层的偏置和权重
    for i in L.weight:
        w.append(i)
    x = np.array([0, 100])
    y=find_y(x,w,o)#逆归一化计算决策边界的y值
    plt.plot(x,y,color='r',label="decision boundary")
    for i in range(len(accept)):
        plt.scatter(accept[i][0],accept[i][1],c="g")
    for i in range(len(refuse)):
        plt.scatter(refuse[i][0],refuse[i][1],c="b")
    plt.show()

def draw_loss(loss):#画出损失曲线
    plt.plot([i for i in range(1,iter+1)],loss,color="r",label="loss")
    plt.show()

def nn(k):#神经网络预测
    f = open(r"ai\E10\data1.txt",'r')
    input,output,accept,refuse=read_file(f,k)#读取样本数据
    input,o=one(input)#对样本数据进行归一
    L=Layer(len(input[0]))#针对data1只需要一层输出层和输入层，这是输出层，输入层直接输入
    layers=[[L]]#每一层的神经元
    loss=train(iter,input,output,layers)#训练
    calculate_accuracy(layers,input,output)#计算预测准确率
    draw_boundary(L,accept,refuse,o)#画出决策边界
    draw_loss(loss)#画出损失曲线

def main():
    nn(2)

if __name__ == '__main__':
    main()

