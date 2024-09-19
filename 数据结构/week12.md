# Lab Week12 Implementation of Critical Path Algorithm
Refer to the example of the digraph and data structures in Lecture 21, Slide 72-93.
(1) Represent a digraph with an Adjacency List
(2) Determine the data structures for the process of Critical Path algorithm
(3) Design a C program implementing the Critical Path algorithm on the digraph

## 实验重点
1. 用队列来存放入度为0的点，每次将队列的头取出，对其连接的顶点进行入读减1的操作，如果入读为0将其放入数组中存放。
2. 用邻接表保存边的信息、任务的信息，如任务名、完成任务时间和关键时间和父亲任务等等，先遍历一次，将入度为0的点放入队列中。
4. 每次利用队列对点进行操作，将队列的头拿出进行拓扑排序。
5. 利用拓扑排序计算每个顶点的critical time,寻找关键路径。

## 设计思路与代码分析
1. 构建邻接表和图，EdgeNode保存边的信息、记录终点的坐标、起点的坐标保存在包含边的VertexNode中。Task结构体保存任务的信息，如任务名、完成任务时间和关键时间和父亲任务，VertexNode保存EdgeNode、Task和顶点的入度和坐标，graph保存各个顶点的邻接表。
```c
typedef struct EdgeNode{
    int index; 
    struct EdgeNode *next; 
}EdgeNode;
typedef struct Task{
char task_name;
double  task_time;
double critical_time;
struct Task*parental_task;
}Task;
typedef struct Vertex 
{    
    int in_degree;
    int index;
    Task t;
    EdgeNode e;
}VertexNode,AdjList[size];
typedef struct
{
    AdjList adjList;
    int numVertexes;
    int numEdges;
}graph;
```
2. 构建队列和初始化队列、push、pop、判断队列是否为空和返回队列头等操作。
```c
typedef struct queue{
VertexNode *tasks[100];
int head;
int tail;
}queue;
void init_queue(queue*q){
    q->head=0;
    q->tail=0;
}
void push(queue*q, VertexNode*t){
    q->tasks[q->tail++]=t;
}
void pop(queue*q){
    q->head++;
}
int empty(queue*q){
    if(q->head==q->tail)return 1;
    return 0;
}
VertexNode* front(queue*q){
    return q->tasks[q->head];
}
```
3. 邻接表的初始化,将顶点入度初始化为0，程序开始时没有邻接的边，父亲任务为NULL等等。
```c
void initial_task(Task*t,char s,double time){
t->task_name=s;
t->task_time=time;
t->critical_time=0;
t->parental_task=NULL;
}
void initial_vnode(int i,char s,double time,VertexNode*v){
initial_task(&v->t,s,time);
v->in_degree=0;
v->e.next=NULL;
v->e.index=i;
v->index=i;
}
```
4. 连接顶点和构建边
```c
void construct_task_graph(VertexNode*start,VertexNode*end){
end->in_degree+=1;
EdgeNode*head=&start->e;
EdgeNode*temp=start->e.next;
while(temp){
    head=head->next;
    temp=temp->next;
}
temp=(EdgeNode*)malloc(sizeof(EdgeNode));
temp->index=end->e.index;
temp->next=NULL;
head->next=temp;
}
```
5. 打印输出任务
```c
void show_task(graph*g,int n){
    for(int i=0;i<n;i++){
        printf("%c: task_time:%f critical_time:%f ",g->adjList[i].t.task_name,g->adjList[i].t.task_time,g->adjList[i].t.critical_time);
        printf("parental_task:");
        if(g->adjList[i].t.parental_task==NULL)printf("NULL");
        else printf("%c",g->adjList[i].t.parental_task->task_name);
        printf("\n"); 
    }
}
```
6. 利用拓扑排序计算每个顶点的critical time,方便寻找关键路径
```c
void topo_critical_path(queue*q,graph*g,int index){
EdgeNode*temp=g->adjList[index].e.next;
while(temp){
    g->adjList[temp->index].in_degree--;
    if(g->adjList[temp->index].t.critical_time==0){
        g->adjList[temp->index].t.critical_time=g->adjList[temp->index].t.task_time+g->adjList[index].t.critical_time;
        g->adjList[temp->index].t.parental_task=&g->adjList[index].t;
    }
    else if(g->adjList[temp->index].t.critical_time<g->adjList[temp->index].t.task_time+g->adjList[index].t.critical_time){
        g->adjList[temp->index].t.critical_time=g->adjList[temp->index].t.task_time+g->adjList[index].t.critical_time;
        g->adjList[temp->index].t.parental_task=&g->adjList[index].t;
    }
    if(g->adjList[temp->index].in_degree==0){
        push(q,&g->adjList[temp->index]);
    }
    temp=temp->next;
}
}
```
7. 寻找关键路径并打印关键路径
```c
show_task(&g,n);
char c[size];
int cn=0;
double max=0;
int maxid=-1;
for(int i=0;i<n;i++){
if(g.adjList[i].t.critical_time>max)
{
    max=g.adjList[i].t.critical_time;
    maxid=i;
}
}
c[cn++]=g.adjList[maxid].t.task_name;
Task*temp=g.adjList[maxid].t.parental_task;
while(temp){
    c[cn++]=temp->task_name;
    temp=temp->parental_task;
}
printf("The critical time of this JNW is %f\n",max);
printf("The critical path is:");
for(int i=cn-1;i>=0;i--){
    printf("%c",c[i]);
    if(i!=0)printf("->");
    }

```

## 代码实现
```c
#include<stdio.h>
#include<malloc.h>
#include<string.h>
#define size 100
typedef struct EdgeNode{
    int index; 
    struct EdgeNode *next; 
}EdgeNode;
typedef struct Task{
char task_name;
double  task_time;
double critical_time;
struct Task*parental_task;
}Task;
typedef struct Vertex 
{    
    int in_degree;
    int index;
    Task t;
    EdgeNode e;
}VertexNode,AdjList[size];
typedef struct
{
    AdjList adjList;
    int numVertexes;
    int numEdges;
}graph;
typedef struct queue{
VertexNode *tasks[100];
int head;
int tail;
}queue;
void initial_task(Task*t,char s,double time){
t->task_name=s;
t->task_time=time;
t->critical_time=0;
t->parental_task=NULL;
}
void initial_vnode(int i,char s,double time,VertexNode*v){
initial_task(&v->t,s,time);
v->in_degree=0;
v->e.next=NULL;
v->e.index=i;
v->index=i;
}
void construct_task_graph(VertexNode*start,VertexNode*end){
end->in_degree+=1;
EdgeNode*head=&start->e;
EdgeNode*temp=start->e.next;
while(temp){
    head=head->next;
    temp=temp->next;
}
temp=(EdgeNode*)malloc(sizeof(EdgeNode));
temp->index=end->e.index;
temp->next=NULL;
head->next=temp;
}
void init_queue(queue*q){
    q->head=0;
    q->tail=0;
}
void push(queue*q, VertexNode*t){
    q->tasks[q->tail++]=t;
}
void pop(queue*q){
    q->head++;
}
int empty(queue*q){
    if(q->head==q->tail)return 1;
    return 0;
}
VertexNode* front(queue*q){
    return q->tasks[q->head];
}
void show_task(graph*g,int n){
    for(int i=0;i<n;i++){
        printf("%c: task_time:%f critical_time:%f ",g->adjList[i].t.task_name,g->adjList[i].t.task_time,g->adjList[i].t.critical_time);
        printf("parental_task:");
        if(g->adjList[i].t.parental_task==NULL)printf("NULL");
        else printf("%c",g->adjList[i].t.parental_task->task_name);
        printf("\n"); 
    }
}
void topo_critical_path(queue*q,graph*g,int index){
EdgeNode*temp=g->adjList[index].e.next;
while(temp){
    g->adjList[temp->index].in_degree--;
    if(g->adjList[temp->index].t.critical_time==0){
        g->adjList[temp->index].t.critical_time=g->adjList[temp->index].t.task_time+g->adjList[index].t.critical_time;
        g->adjList[temp->index].t.parental_task=&g->adjList[index].t;
    }
    else if(g->adjList[temp->index].t.critical_time<g->adjList[temp->index].t.task_time+g->adjList[index].t.critical_time){
        g->adjList[temp->index].t.critical_time=g->adjList[temp->index].t.task_time+g->adjList[index].t.critical_time;
        g->adjList[temp->index].t.parental_task=&g->adjList[index].t;
    }
    if(g->adjList[temp->index].in_degree==0){
        push(q,&g->adjList[temp->index]);
    }
    temp=temp->next;
}
}
int main(){
graph g;
queue q;
init_queue(&q);
int n,m;
printf("Input the numbers of vertexes and edges\n");
scanf("%d%d",&n,&m);
for(int i=0;i<n;i++){
    printf("Input the task and time:");
    char c;
    double it;
    char t,f;
    scanf("%c%c%c%lf",&f,&c,&t,&it);
    initial_vnode(i,c,it,&g.adjList[i]);
}
show_task(&g,n);
printf("Input the edges(index of vertexes):\n");
for(int i=1;i<=m;i++){
    int start,end;
    scanf("%d",&start);
    scanf("%d",&end);
    construct_task_graph(&g.adjList[start],&g.adjList[end]);
}
for(int i=0;i<n;i++){
    printf("%c: ",g.adjList[i].t.task_name);
    EdgeNode*p;
    p=g.adjList[i].e.next;
    while(p){
      printf("%c",g.adjList[p->index].t.task_name);
      p=p->next;
      if(p)printf("->");
}
printf("\n");
}
for(int i=0;i<n;i++){
    if(g.adjList[i].in_degree==0){
        g.adjList[i].t.critical_time=g.adjList[i].t.task_time;
        push(&q,&g.adjList[i]);
}
}
while(!empty(&q)){
int index=(front(&q))->index;
topo_critical_path(&q,&g,index);
pop(&q);
}
show_task(&g,n);
char c[size];
int cn=0;
double max=0;
int maxid=-1;
for(int i=0;i<n;i++){
if(g.adjList[i].t.critical_time>max)
{
    max=g.adjList[i].t.critical_time;
    maxid=i;
}
}
c[cn++]=g.adjList[maxid].t.task_name;
Task*temp=g.adjList[maxid].t.parental_task;
while(temp){
    c[cn++]=temp->task_name;
    temp=temp->parental_task;
}
printf("The critical time of this JNW is %f\n",max);
printf("The critical path is:");
for(int i=cn-1;i>=0;i--){
    printf("%c",c[i]);
    if(i!=0)printf("->");
    }
return 0;
}
```
## 测试样例
1.**样例一** 
Input the numbers of vertexes and edges
6 8
Input the task and time:a 5.2
Input the task and time:b 6.1
Input the task and time:c 4.7
Input the task and time:d 8.1
Input the task and time:e 9.5
Input the task and time:f 17.1
a: task_time:5.200000 critical_time:0.000000 parental_task:NULL
b: task_time:6.100000 critical_time:0.000000 parental_task:NULL
c: task_time:4.700000 critical_time:0.000000 parental_task:NULL
d: task_time:8.100000 critical_time:0.000000 parental_task:NULL
e: task_time:9.500000 critical_time:0.000000 parental_task:NULL
f: task_time:17.100000 critical_time:0.000000 parental_task:NULL
Input the edges(index of vertexes):
0 1
0 3
1 2
1 3
1 4
2 3
4 2
5 4
a: b->d
b: c->d->e
c: d
d:
e: c
f: e
a: task_time:5.200000 critical_time:5.200000 parental_task:NULL
b: task_time:6.100000 critical_time:11.300000 parental_task:a
c: task_time:4.700000 critical_time:31.300000 parental_task:e
d: task_time:8.100000 critical_time:39.400000 parental_task:c
e: task_time:9.500000 critical_time:26.600000 parental_task:f
f: task_time:17.100000 critical_time:17.100000 parental_task:NULL
The critical time of this JNW is 39.400000
The critical path is:f->e->c->d
2.**样例二** 
Input the numbers of vertexes and edges
5 6
Input the task and time:a 0.5
Input the task and time:b 0.4
Input the task and time:c 0.1
Input the task and time:d 0.3
Input the task and time:e 0.7
a: task_time:0.500000 critical_time:0.000000 parental_task:NULL
b: task_time:0.400000 critical_time:0.000000 parental_task:NULL
c: task_time:0.100000 critical_time:0.000000 parental_task:NULL
d: task_time:0.300000 critical_time:0.000000 parental_task:NULL
e: task_time:0.700000 critical_time:0.000000 parental_task:NULL
Input the edges(index of vertexes):
0 1
0 3
1 2
2 4
3 4
3 2
a: b->d
b: c
c: e
d: e->c
e:
a: task_time:0.500000 critical_time:0.500000 parental_task:NULL
b: task_time:0.400000 critical_time:0.900000 parental_task:a
c: task_time:0.100000 critical_time:1.000000 parental_task:b
d: task_time:0.300000 critical_time:0.800000 parental_task:a
e: task_time:0.700000 critical_time:1.700000 parental_task:c
The critical time of this JNW is 1.700000
The critical path is:a->b->c->e
## 结果分析
假设图中的顶点个数为n，边的数目为m。
为每个顶点构建邻接表的时间复杂度为O(n+m)。
在拥有邻接表的基础上用拓扑排序计算每个任务的critical time的时间复杂度为O(n+m)。
在完成拓扑排序、计算得到每个任务的critical time，寻找关键路径的时间复杂度为O(n+m)。

