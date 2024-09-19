# Lab Week14 Implementation of Dijkstra Algorithm Using min-Heap
## 一、实验内容
1. Represent a digraph with an Adjacency List
2. Determine the data structures for the process of Dijkstra algorithm using min-Heap
3. Design a C program implementing the Dijkstra algorithm on the digraph using min-Heap
## 实验重点
1. 注意初始化节点时将next指针赋值为NULL，visited赋值 0 表示未访问，将图的第一个顶点设置为初始顶点，distance初始化为0
2. 利用迪杰斯特拉算法，寻找最短路径：利用最小堆，将所有节点放入最小堆中，比较每个结点的dsitance,源点的distance=0、visited=0,直接与源点相邻的节点的distance为直接连接边的权重， while 循环每次将 distance最小且未被访问过的的结点取出m并将其标志为已访问，依次扫描每一条包含该结点的边，如果边的另一个顶点的distance > 边的权重 + 该顶点到源点的距离distance，更新另一个顶点的distance = weight + 该顶点到源点的距离，并更新其父亲节点。
3. 利用最小堆找出distance最小且未被访问的节点。
4. 每次改变节点的distance时，最小堆中节点信息有可能改变，所以每次循环都要调整最小堆。
## 设计思路与代码分析 
1. 创建边、节点邻接表、图的结构体
```c
typedef struct Edge{
    int start;
    int end;
    int length; 
    struct Edge*next; 
}Edge;
```
边包含的起点和终点的坐标，以及边的权值
```c
typedef struct Vertex 
{    
    int visited;
    int number;
    int index;
    int distance;
    struct Vertex*parent;
    Edge e;
}VertexNode,AdjList[size];
```
结点包含了节点的名称，节点的下标，是否被访问的标记visited，与源点的距离distance，并且包含了关于以这个节点为起点的边
```c
typedef struct
{   
    int nums_vertex;
    int nums_edges;
    AdjList adjList;
}Graph;
```
图包含了邻接表和边和顶点的数量
2. 构建最小堆和最小堆的push、pop和调整最小堆的相关操作
```c
typedef struct
{   
  int n;
  VertexNode *heap_Vertex[size];  
}Min_heap;
void initial_heap(Min_heap*min_heap){
    min_heap->n=0;
}
void adjust_min_heap(Min_heap* min_heap,int start,int end){
for(int l=start*2+1;l<=end;l=l*2+1){
    int min_id=l;
    if(l+1<=end && min_heap->heap_Vertex[l]->distance  >min_heap->heap_Vertex[l+1]->distance){
        min_id=l+1;
    }
    if(min_heap->heap_Vertex[min_id]->distance < min_heap->heap_Vertex[start]->distance){
        VertexNode*temp=min_heap->heap_Vertex[min_id];
        min_heap->heap_Vertex[min_id]=min_heap->heap_Vertex[start];
        min_heap->heap_Vertex[start]=temp;
        start=min_id;
    }
    else break;
    l=min_id;
}  
}

VertexNode*front(Min_heap*min_heap){
    if(!empty(min_heap))return min_heap->heap_Vertex[0];
}
void struct_min_heap(Min_heap*min_heap,VertexNode*v,int heap_size){
min_heap->heap_Vertex[heap_size]=v;
heap_size++;
int start=heap_size-1;
for(int i=(start-1)/2;i>=0;i=(start-1)/2){
    if(min_heap->heap_Vertex[start]->distance >=min_heap->heap_Vertex[i]->distance)break;
    else {
        VertexNode*temp=min_heap->heap_Vertex[i];
        min_heap->heap_Vertex[i]=min_heap->heap_Vertex[start];
        min_heap->heap_Vertex[start]=temp;
        start=i;
    }
    if(i==0)break;
}
}
void push(Min_heap*min_heap,VertexNode*v){
struct_min_heap(min_heap,v,min_heap->n);
min_heap->n++;
}
void pop(Min_heap*min_heap){
VertexNode*temp=min_heap->heap_Vertex[0];
min_heap->heap_Vertex[0]=min_heap->heap_Vertex[min_heap->n-1];
adjust_min_heap(min_heap,0,min_heap->n-1);   
min_heap->n--; 
}
int empty(Min_heap*min_heap){
    return min_heap->n==0;
}
```
3. 构建图(输入每个结点的名称 name，初始化邻接表。j节点的初始化：将next指针赋值为NULL，visited赋值 0 表示未访问，distance初始化infinite. 再将边edge连接在顶点邻接表的next指针。)
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
}```
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
    printf("Please input the number of each vertex\n");
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
    }
}
```
4. 打印输出图和邻接表
```c
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
```
5. 迪杰斯特拉算法寻找最短路径(利用最小堆，将所有节点放入最小堆中，比较每个结点的dsitance,源点的distance=0、visited=0,直接与源点相邻的节点的distance为直接连接边的权重， while 循环每次将 distance最小且未被访问过的的结点取出m并将其标志为已访问，依次扫描每一条包含该结点的边，如果边的另一个顶点的distance > 边的权重 + 该顶点到源点的距离distance，更新另一个顶点的distance = weight + 该顶点到源点的距离，并更新其父亲节点,每次都要重新调整最小堆。  )
```c
void Dijkstra(Graph*g,int s){
    Min_heap m;
    initial_heap(&m);
    int n=g->nums_vertex,i;
    g->adjList[s].distance=0;
    for(i=1;i<=n;i++)push(&m,&g->adjList[i]);
    for(i=1;i<n;i++){
    VertexNode*temp=front(&m);
    while(temp->visited==1){
        pop(&m);
        temp=front(&m);
    }
    temp->visited=1;
    Edge*cur=temp->e.next;
    while(cur){
        if(cur->length+temp->distance < g->adjList[cur->end].distance){
            g->adjList[cur->end].distance= cur->length+temp->distance;
            g->adjList[cur->end].parent=temp;
        }
        cur=cur->next;
    }
    for(int j=m.n-1;j>=0;j--){
    adjust_min_heap(&m,j,m.n-1);
    }
    }
}
```
6. 打印输出源点到各点的最短距离和最短路径
```c
void print_distance(Graph*g,int s){
for(int i=1;i<=g->nums_vertex;i++){
    printf("The distance from %d to %d is : %d\n",g->adjList[s].number,g->adjList[i].number,g->adjList[i].distance);
    printf("The road from %d to %d :  ",g->adjList[s].number,g->adjList[i].number);
    VertexNode*temp=&g->adjList[i];
    int m=0;
    int r[size];
    while(temp){
        r[m++]=temp->number;
        temp=temp->parent;
    }
    for(int j=m-1;j>=0;j--){
    printf("%d",r[j]);
    if(j!=0)printf("->");
    }
    printf("\n");
}
}
```
## 测试样例
1.样例一
Please input the numbers of Vertexes and edges
7 10
Please input the number of each vertex
1 2 3 4 5 6 7
Please input edges(the indexes of vertexes and the length of edges):
1 3 2
1 2 5
2 4 1
3 4 6
3 6 8
2 5 6
4 6 2
4 5 1
6 7 3
5 7 7
1:3->2
2:4->5
3:4->6
4:6->5
5:7
6:7
7:
Please input the index of source vertex:
1
The distance from 1 to 1 is : 0
The road from 1 to 1 :  1
The distance from 1 to 2 is : 5
The road from 1 to 2 :  1->2
The distance from 1 to 3 is : 2
The road from 1 to 3 :  1->3
The distance from 1 to 4 is : 6
The road from 1 to 4 :  1->2->4
The distance from 1 to 5 is : 7
The road from 1 to 5 :  1->2->4->5
The distance from 1 to 6 is : 8
The road from 1 to 6 :  1->2->4->6
The distance from 1 to 7 is : 11
The road from 1 to 7 :  1->2->4->6->7
2. 样例二
   Please input the numbers of Vertexes and edges
8 15
Please input the number of each vertex
1 2 3 4 5 6 7 8
Please input edges(the indexes of vertexes and the length of edges):
1 2 28
1 5 1
1 3 2
2 6 10
2 4 9
3 6 24
3 8 27
4 5 5
4 7 8
4 8 7
5 6 26
5 2 8
6 7 7
6 8 1
7 8 8
1:2->5->3
2:6->4
3:6->8
4:5->7->8
5:6->2
6:7->8
7:8
8:
Please input the index of source vertex:
1
The distance from 1 to 1 is : 0
The road from 1 to 1 :  1
The distance from 1 to 2 is : 9
The road from 1 to 2 :  1->5->2
The distance from 1 to 3 is : 2
The road from 1 to 3 :  1->3
The distance from 1 to 4 is : 18
The road from 1 to 4 :  1->5->2->4
The distance from 1 to 5 is : 1
The road from 1 to 5 :  1->5
The distance from 1 to 6 is : 19
The road from 1 to 6 :  1->5->2->6
The distance from 1 to 7 is : 26
The road from 1 to 7 :  1->5->2->4->7
The distance from 1 to 8 is : 20
The road from 1 to 8 :  1->5->2->6->8

## 结果分析 
假设图的顶点数为 n ,边数为 m  
1. 在最开始构建图的时候，将边和点联系起来的时间复杂度为O(m+n) 。  
2. Dijkstra算法中利用了最小堆的寻找，每次循环寻找最坏的情况是将堆中的所有点搜索，所以每次搜索边的时间复杂度为 O(logn)，每次遍历边计算distance时间复杂度为 O(m),每次改变节点的distance时，最小堆中节点信息有可能改变，所以每次循环都要调整最小堆,时间复杂度为 O(nlogn)。
3. Dijkstra算法中，要对n-1个顶点进行搜索，要循环n-1次。
综上所述，整个程序的时间复杂度为 O(n^nlogn)。 

## 总结与疑问  
1. 总结：比较Prim算法和Dijkstra算法
目的：Prim算法解决的是连通无向有权图中最小生成树的问题，而 Dijkstra 算法解决的是源点到目标点的最短路径问题。
过程：Prim 算法是寻找已经访问的点和未访问的点的权重最小的边，而prim算法是寻找未访问且distance最小的点。
1. 疑问：迪杰特斯拉每次从堆里拿出最小的distance的节点后，再算出新的节点的distance，这个时候堆里面的distance变了，不能只是堆的push pop这样做O（logn）的操作，要对整个堆重新排序，那就要O(nlogn)，那还不如直接每次遍历数组找最小的，只要O(n) ，是否不使用堆直接遍历会更快？




