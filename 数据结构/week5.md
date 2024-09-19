# Lab Week05

## 实验内容
Application of Circular Queues
The producer–consumer problem is a common paradigm for cooperating processes. A producer process produces items that is consumed by a consumer process. A FIFO buffer shared by these two processes is designed to be filled by the producer and emptied by the consumer. The shared buffer is implemented as a circular array with a limit n = QUEUE_CAPACITY.
 Let i, j be two different random number within [0, n-1], and the positions from i to j in clockwise (i.e., i % n, (i+1) % n, …, (i+k) % n = j) are assumed to be occupied. Design a C program to implement
(1) Giving a pair of i, j, the Producer process that pushes items into 
the buffer
(2) Giving another pair of i, j, the Consumer process that pops items 
out of the buffer
## 实验重点以及设计思路
1. 为了表示某些位置为空，以队列中某个位置的元素为0，代表对列中某个位置为空。
2. 初始化时将队列i到j位置用随机数（除0）填满。
3. 注意边界情况，防止队列未满而未手动停止而停止等情况。如：push时当front=rear时为满、pop时front=rear且此位置元素为0为空。
   
## 代码实现

```C
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<string.h>
#define size 10
typedef struct queue{
	int *base;
	int front;
	int rear;
}Squeue;
void push(Squeue *q,int e){
    q->base[q->rear]=e;
    q->rear=(q->rear+1)%size;
}
int pop(Squeue *q){
    int e=q->base[q->front];
    q->base[q->front]=0;
    q->front=(q->front+1)%size;
    if(e)printf("consume:%d\n",e);
    else printf("consume:nothing\n");
    return e;
}
void show(Squeue *q){
    for(int i=0;i<size;i++)printf("%d ", q->base[i]);
    printf("\n");
}//打印循环队列
void Initiqueue(Squeue *q,int i,int j){
    q->base=(int*)malloc(sizeof(int)*size);
    for(int i=0;i<size;i++)q->base[i]=0;//以队列中某个位置的元素为0，代表对列中某个位置为空
    q->front = i;
    q->rear=i;
    while(1){
        int e=1+abs(rand()%(size-1));
        push(q,e);//初始化时将队列i到j位置用随机数（除了0）填满
        if(q->rear==(j+1)%size)break;
        
    }
}
int GetQueueLength(Squeue *q){
    return (q->rear-q->front+size)%size;
}
void consume( Squeue *q,int i,int j){
	q->front=i;
	while(1){
	if(q->front!=j){	
        pop(q);
        show(q);
}
   else	if(q->front==j){
   if(q->base[q->front]!=0){
   	pop(q);
    show(q);
    
   }
   return ;
   } 
    }
	
}
void Inputsqueue(Squeue *q){
  int temp;
    while(1){
    	if(q->rear==q->front){
            printf("The circular queue is full!\n");
            break;}
        scanf("%d",&temp);//输入除了0以外放在的数字在队列的rear位置上，出入-1时停止
        if(temp==-1)break;
        else push(q,temp);
    }
}
int main(){
	srand((unsigned)time(NULL));
	int i1=rand()%(size-1);
	int j1=rand()%(size-1); 
    printf("The random numbers for initializing:%d %d\n",i1,j1);
    Squeue q;
    Initiqueue(&q,i1,j1);
    printf("The initial circular queue:\n");
    show(&q);
    printf("Please input the numbers into the circular queue except 0 and you can input -1 to stop:\n");
    Inputsqueue(&q);
    printf("After Producing,the circular queue:\n");
    show(&q);
    int i2=rand()%(size-1);
	int j2=rand()%(size-1);
	printf("THe random numbers for consuming:%d %d\n",i2,j2);
	consume(&q,i2,j2);
    printf("After consuming, the circular queue:\n");
    show(&q); 
	return 0;
} 
```
## 测试样例
**因为程序中的随机数是随机产生的，程序开始前无法正确个数的输入样例，以下为程序执行的几次样例** 
1. 
The random numbers for initializing:6 1
The initial circular queue:
6 8 0 0 0 0 7 7 2 2
Please input the numbers into the circular queue except 0 and you can input -1 to stop:
5//此行为手动输入
4//此行为手动输入
3//此行为手动输入
2//此行为手动输入
The circular queue is full!
After Producing,the circular queue:
6 8 5 4 3 2 7 7 2 2
THe random numbers for consuming:4 6
consume:3
6 8 5 4 0 2 7 7 2 2
consume:2
6 8 5 4 0 0 7 7 2 2
consume:7
6 8 5 4 0 0 0 7 2 2 
After consuming, the circular queue:
6 8 5 4 0 0 0 7 2 2
2. 
The random numbers for initializing:5 1
The initial circular queue:
2 2 0 0 0 7 9 4 2 5
Please input the numbers into the circular queue except 0 and you can input -1 to stop:
5//此行为手动输入
6//此行为手动输入
-1//此行为手动输入
After Producing,the circular queue:
2 2 5 6 0 7 9 4 2 5
THe random numbers for consuming:8 0
consume:2
2 2 5 6 0 7 9 4 0 5
consume:5
2 2 5 6 0 7 9 4 0 0
consume:2
0 2 5 6 0 7 9 4 0 0
After consuming, the circular queue:
0 2 5 6 0 7 9 4 0 0

## 结果分析
1. 设循环队列的大小为n，因为吃个程序操作都是线性时间，所以这个程序的空间复杂度为O(n),时间复杂度为O(n)。
2. 同时要注意循环队列的边界问题。
 
 
 
 
 
 
 