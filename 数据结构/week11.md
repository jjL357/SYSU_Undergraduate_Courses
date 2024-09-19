# Lab Week 11
## 实验内容：Implementation of Topological Sorting
Refer to the example of the digraph and data structures in Lecture 21, Slide 42-65.
(1) Represent a digraph with an Adjacency List
(2) Determine the data structures for the process of improved topological sorting
(3) Design a C program implementing the improved topological sorting on the digraph

## 实验重点
1. 用队列来存放入度为0的点，每次将队列的头取出，对其连接的顶点进行入读减1的操作，如果入读为0将其放入数组中存放。
2. 用邻接表保存点和点与点之间的关系，先遍历一次，将入度为0的点放入队列中，后续再利用队列对点进行操作。
3. 当顶点放进答案数组中，将以该顶点入度设为-1表示该节点已经被访问过并放入答案中。
4. 利用结构体构建邻接表时将其连出的顶点保存，方便后续拓扑排序操作。
5. 拓扑排序时将入度为0的顶点加入答案中，并对其连出的顶点入度减1，判断其入度是否，若是则将其加入答案，重复此操作。
## 设计思路与代码分析
1. 设计队列
```c
typedef struct queue{
alist *nums[100];
int head;
int tail;
}queue;
void init_queue(queue*q){
    q->head=0;
    q->tail=0;
}
void push(queue*q, alist*val){
    q->nums[q->tail++]=val;
}
void pop(queue*q){
    q->head++;
}
int empty(queue*q){
    if(q->head==q->tail)return 1;
    return 0;
}
alist* front(queue*q){
    
    return q->nums[q->head];
}
```
2. 设计邻接表,邻接表中的out存放以其为起点连的顶点的地址，方便后续拓扑排序操作
```c
typedef struct AdjacencyList{
int val;
int insize;
int outsize;
struct AdjacencyList**out;
}alist;
void init_alist(alist*l,int val){
    l->insize=0;
    l->outsize=0;
    l->val=val;
    l->out=(alist**)malloc(sizeof(alist*)*size);
}
```
3. 构建图即构建点之间的关系
```c
int find(int x,alist*a,int n){
    for(int i=0;i<n;i++){
        if(a[i].val==x)return i;
    }
}
void construct_graph(int x,alist*a,alist*nums,int n){
    a->out[a->outsize++]=&nums[find(x,nums,n)];
    nums[find(x,nums,n)].insize++;
}
```
4. 对入读为0的顶点进行操作并放入答案数组中，同时利用递归对以其为起点连接的顶点进行拓扑排序。
```c
void topo(alist*a,int*nums,int *s){                 
nums[(*s)++]=a->val;
a->insize-=1;
for(int i=0;i<a->outsize;i++){
    (*(a->out[i])).insize-=1;
    if((*(a->out[i])).insize==0)topo(&(*a->out[i]),nums,s);
}
}
```
## 代码实现
```c
#include<stdio.h>
#include<malloc.h>
#define size 100
typedef struct AdjacencyList{
int val;
int insize;
int outsize;
struct AdjacencyList**out;
}alist;
typedef struct queue{
alist *nums[100];
int head;
int tail;
}queue;
void init_queue(queue*q){
    q->head=0;
    q->tail=0;
}
void push(queue*q, alist*val){
    q->nums[q->tail++]=val;
}
void pop(queue*q){
    q->head++;
}
int empty(queue*q){
    if(q->head==q->tail)return 1;
    return 0;
}
alist* front(queue*q){
    
    return q->nums[q->head];
}
void init_alist(alist*l,int val){
    l->insize=0;
    l->outsize=0;
    l->val=val;
    l->out=(alist**)malloc(sizeof(alist*)*size);
}
int find(int x,alist*a,int n){
    for(int i=0;i<n;i++){
        if(a[i].val==x)return i;
    }
}
void construct_graph(int x,alist*a,alist*nums,int n){
    a->out[a->outsize++]=&nums[find(x,nums,n)];
    nums[find(x,nums,n)].insize++;
}
void topo(alist*a,int*nums,int *s){                 
nums[(*s)++]=a->val;
a->insize-=1;
for(int i=0;i<a->outsize;i++){
    (*(a->out[i])).insize-=1;
    if((*(a->out[i])).insize==0)topo(&(*a->out[i]),nums,s);
}
}
int main(){
    int ans[size];
    alist v[size];
    int vsize=0;
    queue q;
    printf("Please input the number of vertices:\n");
    int n;
    scanf("%d",&n);
    printf("Please input the vertices:\n");
    for(int i=0;i<n;i++){
    int x;
    scanf("%d",&x);    
    init_alist(&v[i],x);    
    }
    printf("Please input the relationship of each vertices and input -1 to quit:\n");
    for(int i=0;i<n;i++){
        printf("The %d to:\n",v[i].val);
        while(1){
        int x;
        scanf("%d",&x);
        if(x==-1)break ;
        construct_graph(x,&v[i],v,n);
        }
    }
     for(int i=0;i<n;i++){
      if(v[i].insize==0){
        v[i].insize-=1;
        push(&q,&v[i]);
      }
    }
    int s=0;
    while(!empty(&q)){
        alist *temp=front(&q);
        pop(&q);
        if(temp->insize==-1)topo(temp,ans,&s);
    }
    for(int i=0;i<n;i++)printf("%d ",ans[i]);
    for(int i=0;i<n;i++)free(v[i].out);
    return 0;
}
```
## 测试样例
1.**样例一** 
Please input the number of vertices:
6
Please input the vertices:
1 2 3 4 5 6
Please input the relationship of each vertices and input -1 to quit:
The 1 to:
2
3 4 -1
The 2 to:
-1
The 3 to:
2 5
-1
The 4 to:
5
-1
The 5 to:
-1
The 6 to:
4 5
-1
1 3 2 6 4 5 
2.**样例二** 
Please input the number of vertices:
5
Please input the vertices:        
1 2 3 4 5
Please input the relationship of each vertices and input -1 to quit:
The 1 to:
2
4
-1
The 2 to:
3
4
-1
The 3 to:
5
-1
The 4 to:
5
3
-1
The 5 to:
-1
1 2 4 3 5 
## 结果分析
设顶点的个数为n,边的个数为m。
构建图即创建邻接表的过程的时间复杂度为O(m*n)，因为构建联系两个顶点的每个边时要对顶点进行遍历找到对应的两个端点。
而拓扑排序需要对每个顶点和每条边进行操作，所以拓扑排序的的时间复杂度为O(n+m)。