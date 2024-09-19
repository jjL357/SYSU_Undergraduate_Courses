# Lab Week13
## 实验内容
Implementation of PRIM Algorithm Using min-Heap
Refer to the example of the digraph and data structures in Lecture 
23, Slide 9-34.
(1) Represent a digraph with an Adjacency List
(2) Determine the data structures for the process of PRIM algorithm using min-Heap
(3) Design a C program implementing the PRIM algorithm on the digraph using min-Heap

## 实验重点
1. 使用最小堆帮助我们寻找最短边，每次从堆顶出最短边，若其的端点存在未被访问的节点则这条边是我们寻找的边，否则pop，继续寻找。
2. 每次循环取出符合要求的边，将其加入答案中，然后访问未被访问结点，并将其标志为已访问，并依次扫描每一条包含该结点的边，将每一条边push到最小堆中.
3. VertexNode加入visited标志，visited=1时为已经访问并加入到树中，为0则未被访问，这样防止我们对同一个节点多次访问。
4. 做好堆调整维护操作，每次加入新的边都要调整最小堆。

## 设计思路与代码分析
1. 构建邻接表和图，EdgeNode保存边的信息、记录终点的坐标、起点的坐标保存在包含边的VertexNode中。Task结构体保存任务的信息，如任务名、完成任务时间和关键时间和父亲任务，VertexNode保存EdgeNode、Task和顶点的入度和坐标，graph保存各个顶点的邻接。VertexNode中visited=1时为已经访问并加入到树中，为0则未被访问，这样防止我们对同一个节点多次访问。
```c
typedef struct Edge{
    int start;
    int end;
    int length; 
    struct Edge*next; 
}Edge;
typedef struct Vertex 
{    
    int visited;
    int number;
    int index;
    int distance;
    struct Vertex*parent;
    Edge e;
}VertexNode,AdjList[size];
typedef struct
{   
    int nums_vertex;
    int nums_edges;
    AdjList adjList;
}Graph;
```
2.邻接表的初始化,将parent和next指针赋值为NULL，visited赋值0表示未访问.在构建图：输入边，利用下拨安排找到边的两个顶点在邻接表的位置，新建边结点edge连接在顶点邻接表的next指针。  
```c   
void initial_vertex(VertexNode*v,int name,int index){
    v->distance=infinite;
    v->index=index;
    v->number=name;
    v->parent=NULL;
    v->visited=0;
    v->e.start=index;
    v->e.length=0;
    v->e.next=NULL;
    v->e.end=index;
}
void initial_edges(Edge*e,int start,int end,int length){
Edge*pre=e;
Edge*cur=e->next;
while(cur){
    pre=pre->next;
    cur=cur->next;
}
cur=(Edge*)malloc(sizeof(Edge));
cur->start=start;
cur->end=end;
cur->length=length;
cur->next=NULL;
pre->next=cur;
}
void initial_graph(Graph *g,int n,int m){
    g->nums_edges=m;
    g->nums_vertex=n;
    printf("Please input the number of each vertex:\n");
    for(int i=1;i<=n;i++){
    int name;
    scanf("%d",&name);
    initial_vertex(&g->adjList[i],name,i);
    }
    printf("Please input edges(the indexes of vertexes and the length of edges):\n");
    for(int i=1;i<=m;i++){
    int start,end,length;
    scanf("%d%d%d",&start,&end,&length);
    initial_edges(&g->adjList[start].e,start,end,length);
    initial_edges(&g->adjList[end].e,end,start,length);
    }
}
```
3. 堆的构建、初始化、push、pop和调整维护等操作。（堆中每个元素为边的地址，比较的根据是边的权重）
```c
typedef struct
{   
  int n;
  Edge *heap_edge[size];  
}Min_heap;
void initial_heap(Min_heap*min_heap){
    min_heap->n=0;
}
void print_heap(Min_heap*heap){
printf("The heap: \n");
for(int i=0;i<heap->n;i++)printf("%d->%d:%d",heap->heap_edge[i]->start,heap->heap_edge[i]->end,heap->heap_edge[i]->length);
printf("\n");
}
void adjust_min_heap(Min_heap* min_heap,int start,int end){
for(int l=start*2+1;l<=end;l=l*2+1){
    int min_id=l;
    if(l+1<=end && min_heap->heap_edge[l]->length  >min_heap->heap_edge[l+1]->length){
        min_id=l+1;
    }
    if(min_heap->heap_edge[min_id]->length<min_heap->heap_edge[start]->length){
        Edge*temp=min_heap->heap_edge[min_id];
        min_heap->heap_edge[min_id]=min_heap->heap_edge[start];
        min_heap->heap_edge[start]=temp;
        start=min_id;
    }
    else break;
    l=min_id;
}  
}
void struct_min_heap(Min_heap*min_heap,Edge*e,int heap_size){
min_heap->heap_edge[heap_size]=e;
heap_size++;
int start=heap_size-1;
for(int i=(start-1)/2;i>=0;i=(start-1)/2){
    if(min_heap->heap_edge[start]->length >=min_heap->heap_edge[i]->length)break;
    else {
        Edge*temp=min_heap->heap_edge[i];
        min_heap->heap_edge[i]=min_heap->heap_edge[start];
        min_heap->heap_edge[start]=temp;
        start=i;
    }
    if(i==0)break;
}
}
void push(Min_heap*min_heap,Edge*e){
struct_min_heap(min_heap,e,min_heap->n);
min_heap->n++;
}
void pop(Min_heap*min_heap){
Edge*temp=min_heap->heap_edge[0];
min_heap->heap_edge[0]=min_heap->heap_edge[min_heap->n-1];
adjust_min_heap(min_heap,0,min_heap->n-1);   
min_heap->n--; 
}
```
4. Prim算法：将第一个点加入树中，使用一个最小堆优化，将已经访问过的节点的边加入堆中排序，每次循环搜索每次从堆顶出最短边，若其的端点存在未被访问的节点则这条边是我们寻找的边，否则pop，继续寻找。每次循环取出符合要求的边，将其加入答案中，然后访问未被访问结点，并将其标志为已访问，并依次扫描每一条包含该结点的边，将每一条边push到最小堆中.并更新其父亲节点。  
```c
void prim(Graph*g,VertexNode**v,Edge**e){
    Min_heap min_heap;
    initial_heap(&min_heap);
    int Vn=0,En=0;
    g->adjList[1].visited=1;
    Edge*temp=g->adjList[1].e.next;
    while(temp){
        push(&min_heap,temp);
        temp=temp->next;
    }
    v[Vn++]=&g->adjList[1];
    while(Vn<g->nums_vertex){
    int target=min_heap.heap_edge[0]->end;
    if(g->adjList[target].visited==0){
    v[Vn++]=&g->adjList[target];
    g->adjList[target].visited=1;
    g->adjList[target].parent=&g->adjList[min_heap.heap_edge[0]->start];
    e[En++]=min_heap.heap_edge[0];
    pop(&min_heap);
    Edge*temp2=g->adjList[target].e.next;
    while(temp2){
        push(&min_heap,temp2);
        temp2=temp2->next;
    }
    }
    else{
    pop(&min_heap);
    }
    }
}
```
5. 打印最小生成树的边和顶点，并输出最小生成树的权值。
```c
void print_spanning_tree(VertexNode**v,Edge**e,int n,int*sum){
printf("The spanning tree:");
for(int i=0;i<n-1;i++){
    (*sum)+=e[i]->length;
    printf("%d->%d ",e[i]->start,e[i]->end);
}
printf("\n");
printf("The minimum weight is %d\n",(*sum));
}

```
## 代码实现
```c
#include<stdio.h>
#include<malloc.h>
#include<string.h>
#include<limits.h>
#define size 100
#define infinite INT_MAX
typedef struct Edge{
    int start;
    int end;
    int length; 
    struct Edge*next; 
}Edge;
typedef struct Vertex 
{    
    int visited;
    int number;
    int index;
    int distance;
    struct Vertex*parent;
    Edge e;
}VertexNode,AdjList[size];
typedef struct
{   
    int nums_vertex;
    int nums_edges;
    AdjList adjList;
}Graph;
typedef struct
{   
  int n;
  Edge *heap_edge[size];  
}Min_heap;
void initial_vertex(VertexNode*v,int name,int index){
    v->distance=infinite;
    v->index=index;
    v->number=name;
    v->parent=NULL;
    v->visited=0;
    v->e.start=index;
    v->e.length=0;
    v->e.next=NULL;
    v->e.end=index;
}
void initial_edges(Edge*e,int start,int end,int length){
Edge*pre=e;
Edge*cur=e->next;
while(cur){
    pre=pre->next;
    cur=cur->next;
}
cur=(Edge*)malloc(sizeof(Edge));
cur->start=start;
cur->end=end;
cur->length=length;
cur->next=NULL;
pre->next=cur;
}
void initial_graph(Graph *g,int n,int m){
    g->nums_edges=m;
    g->nums_vertex=n;
    printf("Please input the number of each vertex:\n");
    for(int i=1;i<=n;i++){
    int name;
    scanf("%d",&name);
    initial_vertex(&g->adjList[i],name,i);
    }
    printf("Please input edges(the indexes of vertexes and the length of edges):\n");
    for(int i=1;i<=m;i++){
    int start,end,length;
    scanf("%d%d%d",&start,&end,&length);
    initial_edges(&g->adjList[start].e,start,end,length);
    initial_edges(&g->adjList[end].e,end,start,length);
    }
}
void print_graph(Graph*g){
    for(int i=1;i<=g->nums_vertex;i++){
        printf("%d:",g->adjList[i].number);
        Edge*temp=g->adjList[i].e.next;
        while(temp){
            printf("%d",temp->end);
            if(temp->next)printf("->");
            temp=temp->next;
        }
        printf("\n");
    }
}
void initial_heap(Min_heap*min_heap){
    min_heap->n=0;
}
void print_heap(Min_heap*heap){
printf("The heap: \n");
for(int i=0;i<heap->n;i++)printf("%d->%d:%d",heap->heap_edge[i]->start,heap->heap_edge[i]->end,heap->heap_edge[i]->length);
printf("\n");
}
void adjust_min_heap(Min_heap* min_heap,int start,int end){
for(int l=start*2+1;l<=end;l=l*2+1){
    int min_id=l;
    if(l+1<=end && min_heap->heap_edge[l]->length  >min_heap->heap_edge[l+1]->length){
        min_id=l+1;
    }
    if(min_heap->heap_edge[min_id]->length<min_heap->heap_edge[start]->length){
        Edge*temp=min_heap->heap_edge[min_id];
        min_heap->heap_edge[min_id]=min_heap->heap_edge[start];
        min_heap->heap_edge[start]=temp;
        start=min_id;
    }
    else break;
    l=min_id;
}  
}
void struct_min_heap(Min_heap*min_heap,Edge*e,int heap_size){
min_heap->heap_edge[heap_size]=e;
heap_size++;
int start=heap_size-1;
for(int i=(start-1)/2;i>=0;i=(start-1)/2){
    if(min_heap->heap_edge[start]->length >=min_heap->heap_edge[i]->length)break;
    else {
        Edge*temp=min_heap->heap_edge[i];
        min_heap->heap_edge[i]=min_heap->heap_edge[start];
        min_heap->heap_edge[start]=temp;
        start=i;
    }
    if(i==0)break;
}
}
void push(Min_heap*min_heap,Edge*e){
struct_min_heap(min_heap,e,min_heap->n);
min_heap->n++;
}
void pop(Min_heap*min_heap){
Edge*temp=min_heap->heap_edge[0];
min_heap->heap_edge[0]=min_heap->heap_edge[min_heap->n-1];
adjust_min_heap(min_heap,0,min_heap->n-1);   
min_heap->n--; 
}
void prim(Graph*g,VertexNode**v,Edge**e){
    Min_heap min_heap;
    initial_heap(&min_heap);
    int Vn=0,En=0;
    g->adjList[1].visited=1;
    Edge*temp=g->adjList[1].e.next;
    while(temp){
        push(&min_heap,temp);
        temp=temp->next;
    }
    v[Vn++]=&g->adjList[1];
    while(Vn<g->nums_vertex){
    int target=min_heap.heap_edge[0]->end;
    if(g->adjList[target].visited==0){
    v[Vn++]=&g->adjList[target];
    g->adjList[target].visited=1;
    g->adjList[target].parent=&g->adjList[min_heap.heap_edge[0]->start];
    e[En++]=min_heap.heap_edge[0];
    pop(&min_heap);
    Edge*temp2=g->adjList[target].e.next;
    while(temp2){
        push(&min_heap,temp2);
        temp2=temp2->next;
    }
    }
    else{
    pop(&min_heap);
    }
    }
}
void print_spanning_tree(VertexNode**v,Edge**e,int n,int*sum){
printf("The spanning tree:");
for(int i=0;i<n-1;i++){
    (*sum)+=e[i]->length;
    printf("%d->%d ",e[i]->start,e[i]->end);
}
printf("\n");
printf("The minimum weight is %d\n",(*sum));
}
int main(){
Graph g;
int n,m;
int sum=0;
printf("Please input the numbers of Vertexes and edges\n");
scanf("%d%d",&n,&m);
initial_graph(&g,n,m);
VertexNode**V=(VertexNode**)malloc(sizeof(VertexNode*)*n);
Edge**E=(Edge**)malloc(sizeof(Edge*)*m);
prim(&g,V,E);
print_spanning_tree(V,E,n,&sum);
return 0;
}
```
## 测试样例
1.**样例一** 
Please input the numbers of Vertexes and edges
8 15
Please input the number of each vertex:
1 2 3 4 5 6 7 8
Please input edges(the indexes of vertexes and the length of edges):
1 2 28 
1 3 2
1 5 1
2 4 9
2 5 8
2 6 10
3 6 24
3 8 27
4 5 5
4 7 8
4 8 7
5 6 26
6 7 8 
6 8 1
7 8 7
The spanning tree:1->5 1->3 5->4 4->8 8->6 8->7 5->2 
The minimum weight is 31
2.**样例二** 
Please input the numbers of Vertexes and edges
9 20
Please input the number of each vertex:
1 2 3 4 5 6 7 8 9
Please input edges(the indexes of vertexes and the length of edges):
1 2 4
1 4 1
1 5 8
2 3 2
2 4 2
2 5 6
2 6 1
3 5 2
3 6 5
4 5 11
4 7 9
4 8 8
5 6 3
5 7 4
5 8 1
5 9 5
6 8 7
6 9 8
7 8 6
8 9 3
The spanning tree:1->4 4->2 2->6 2->3 3->5 5->8 8->9 5->7
The minimum weight is 16
## 结果分析  
假设图的顶点数为 n ,边数为 m  
1. 在最开始构建图的时候，将边和点联系起来的时间复杂度为O(m+n) 。  
2. Prim算法中利用了最小堆的优化，每次循环寻找最坏的情况是将堆中的所有边搜索，由于采用了最小堆优化，所以每次搜索边的时间复杂度为 O(logm)，每次堆的维护和调整也需要 O(logm)。
3. Prim算法中，要对n-1个顶点的相关的边进行搜索，要循环n-1次。
综上所述，整个程序的时间复杂度为 O(nlogm)。  
