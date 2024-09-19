# Lab Week 17 Implementation of M-way Merge Sort by min-Heap
## 实验内容
问题描述：
设有 M = 8 个有序队列，每个队列的关键字从小到大排列，需要将这些队列元素归并成一个新的从小到大的有序队列。
算法描述：
构造一个有 M 个结点的完全小顶堆 H，每个堆结点固定对应一个队列，该堆结点的值是其对应队列的队首元素的值（可以通过指
针描述）。利用这个小顶堆实现 M-路归并排序。
排序的每一趟是：
1. 堆顶结点 top(H) 的值即为当前未归并部分的最小值。设堆顶结点对应于队列 K，输出 front(K);
2. Dequeue(K)，成功时队首元素被更新并仍然由堆顶结点描述；失败时队列 K 为空，此时 pop(H) 删除堆顶结点，将堆尾结
点作为新的堆顶；
3. 针对堆顶实施 Percolating 维护，注意到结点移动时仍然保持与原队列的对应关系不变。
##  实验重点与代码分析 
1. 构建有序队列结构体。
```c
typedef struct queue{
int nums[Size];
int size;
int id;
int start;
int end;
}Sorted_Queue;
```
2. 构建最小堆结构体，存放queue存放有序队列的地址，size表示最小堆中有序队列的个数。
```c
typedef struct heap{
Sorted_Queue*queue[Size];
int size;
}MinHeap;
```
3. 实现有序队列的相关操作，判断是否为空和满，push、pop和获取首元素。
```c
int q_empty(Sorted_Queue*q){
    if(q->start==q->end)return 1;
    return 0;
}
int q_top(Sorted_Queue*q){
    if(q_empty(q)){
        printf("The queue is empty!\n");
        return -1;
    }
    return q->nums[q->start];
}
void q_pop(Sorted_Queue*q){
    if(q_empty(q)){
        printf("The queue is empty!\n");
        return ; 
}
    q->start++;
    q->size--;
}
void q_push(Sorted_Queue*q,int val){
    if(q->end==q->start&&q->end!=0){
        printf("The queue is full!\n");
        return ; 
}
    q->nums[q->end]=val;   
    q->end++;
    q->size++;
}
```
4. 利用最大堆排序，将有序队列中的元素从小到大排列。
```c
void adjust_sortedqueue(Sorted_Queue* sq,int start,int end){
for(int l=start*2+1;l<=end;l=l*2+1){
    int min_id=l;
    
    if(l+1<=end&&sq->nums[l]<sq->nums[l+1]){
        
        min_id=l+1;
    }
    
    if(sq->nums[min_id]>sq->nums[start]){
        int temp=sq->nums[start];
        sq->nums[start]=sq->nums[min_id];
        sq->nums[min_id]=temp;
        start=min_id;
    }
    else break;
    l=min_id;
}  
}
void struct_sortqueue(Sorted_Queue* sq){
   for(int i=(sq->end-1)/2;i>=0;i--){
    adjust_sortedqueue(sq,i,sq->end-1);
   } 
}
void sort_sortedqueue(Sorted_Queue*sq){
    struct_sortqueue(sq);
    for(int i=sq->end-1;i>=0;i--){
        int temp=sq->nums[0];
        sq->nums[0]=sq->nums[i];
        sq->nums[i]=temp;
    adjust_sortedqueue(sq,0,i-1);
    }
}
```
5. 初始化有序队列，将size、start、end赋值为0，id赋值为其该有序队列在队列数组中的下标,输入队列中的元素，最后对其排序。
```c
void initi_sortedqueue(Sorted_Queue*sq,int i){
    sq->size=0;
    sq->start=0;
    sq->end=0;
    sq->id=i;
    printf("Please input the number of elements in this queue:\n");
    int n;
    scanf("%d",&n);
    printf("Please input the elements in this queue:\n");
    for(int i=0;i<n;i++){
        int val;
        scanf("%d",&val);
        q_push(sq,val);
    }
    sort_sortedqueue(sq);
}
```
6. 将队列的地址push进最小堆中。
```c
void push_minheap(MinHeap*m,Sorted_Queue*sq){
  if(sq->end==0)return ;
  m->queue[m->size++]=sq;
 
}
```
7. 最小堆的调整操作。
```c
void adjust_minheap(MinHeap*m,int start){
    int end=m->size-1;  
for(int l=start*2+1;l<=end;l=l*2+1){
    int min_id=l;
    if(l+1<=end&&q_top(m->queue[l]) > q_top(m->queue[l+1])){
        
        min_id=l+1;
    }
    
    if(q_top(m->queue[min_id])<q_top(m->queue[start])){
        
        Sorted_Queue* temp=m->queue[start];
        m->queue[start]=m->queue[min_id];
        m->queue[min_id]=temp;
        start=min_id;
        
    }
    else break;
    l=min_id;
}
}
```
8. 利用最小堆的调整操作，构建起最小堆。
```c
void struct_minheap(MinHeap*m){
   for(int i=(m->size-1)/2;i>=0;i--){
    adjust_minheap(m,i);
   } 
}
```
9. 返回堆顶结点的值即为当前的最小值，同时删除该元素，若该队列size=0,则从最小堆中删除，每次pop删除堆顶结点，将堆尾结点作为新的堆顶；每次最小堆的调整都将堆顶与堆末尾地址交换，在进行对堆的调整和维护。
```c
int m_top(MinHeap*m){
    if(m->size<=0){
        printf("The heap is empty!\n");
        return -1;
    }
    int x= q_top(m->queue[0]);
    q_pop(m->queue[0]);
    if(m->queue[0]->size==0){
        Sorted_Queue* temp=m->queue[0];
        m->queue[0]=m->queue[m->size-1];
        m->queue[m->size-1]=temp;
        m->size--;
    }
    adjust_minheap(m,0);
    return x;
}
```
10. 每次得到堆顶队列的最小值，进行归并排序。
```c
    while(minheap.size>0){
        ans.nums[ans.end++]=m_top(&minheap);
        ans.size++;
    }
```


## 测试样例
1. 样例一
The 1th queue:
Please input the number of elements in this queue:
4 
Please input the elements in this queue:
3 1 2 3
The 2th queue:
Please input the number of elements in this queue:
4
Please input the elements in this queue:
3 4 5 6
The 3th queue:
Please input the number of elements in this queue:
4
Please input the elements in this queue:
3 7 8 9
The 4th queue:
Please input the number of elements in this queue:
4
Please input the elements in this queue:
3 10 11 12
The 5th queue:
Please input the number of elements in this queue:
4
Please input the elements in this queue:
3 13 14 15
The 6th queue:
Please input the number of elements in this queue:
4
Please input the elements in this queue:
3 16 17 18 
The 7th queue:
Please input the number of elements in this queue:
3 19 20 21
Please input the elements in this queue:
The 8th queue:
Please input the number of elements in this queue:
4
Please input the elements in this queue:
3 22 23 24
After merging,the queue is :
1 2 3 3 3 3 3 3 3 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
2. 样例二
The 1th queue:
Please input the number of elements in this queue:
3
Please input the elements in this queue:
22 23 24
The 2th queue:
Please input the number of elements in this queue:
2   
Please input the elements in this queue:
4
5
The 3th queue:
Please input the number of elements in this queue:
4
Please input the elements in this queue:
6 7 8 9
The 4th queue:
Please input the number of elements in this queue:
3
Please input the elements in this queue:
11
12
10
The 5th queue:
Please input the number of elements in this queue:
3
Please input the elements in this queue:
14
15
13
The 6th queue:
Please input the number of elements in this queue:
4
Please input the elements in this queue:
16
17 18 19
The 7th queue:
Please input the number of elements in this queue:
2  
Please input the elements in this queue:
20 21
The 8th queue:
Please input the number of elements in this queue:
3
Please input the elements in this queue:
1
2
300
After merging,the queue is :
1 2 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 300
## 完整代码
```c
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<string.h>
#define Size 100
#define M 8
typedef struct queue{
int nums[Size];
int size;
int id;
int start;
int end;
}Sorted_Queue;
typedef struct heap{
Sorted_Queue*queue[Size];
int size;
}MinHeap;
int q_empty(Sorted_Queue*q){
    if(q->start==q->end)return 1;
    return 0;
}
int q_top(Sorted_Queue*q){
    if(q_empty(q)){
        printf("The queue is empty!\n");
        return -1;
    }
    return q->nums[q->start];
}
void q_pop(Sorted_Queue*q){
    if(q_empty(q)){
        printf("The queue is empty!\n");
        return ; 
}
    q->start++;
    q->size--;
}
void q_push(Sorted_Queue*q,int val){
    if(q->end==q->start&&q->end!=0){
        printf("The queue is full!\n");
        return ; 
}
    q->nums[q->end]=val;   
    q->end++;
    q->size++;
}

void adjust_sortedqueue(Sorted_Queue* sq,int start,int end){
    
for(int l=start*2+1;l<=end;l=l*2+1){
    int min_id=l;
    
    if(l+1<=end&&sq->nums[l]<sq->nums[l+1]){
        
        min_id=l+1;
    }
    
    if(sq->nums[min_id]>sq->nums[start]){
        int temp=sq->nums[start];
        sq->nums[start]=sq->nums[min_id];
        sq->nums[min_id]=temp;
        start=min_id;
    }
    else break;
    l=min_id;
}  
}
void struct_sortqueue(Sorted_Queue* sq){
   for(int i=(sq->end-1)/2;i>=0;i--){
    adjust_sortedqueue(sq,i,sq->end-1);
   } 
}
void sort_sortedqueue(Sorted_Queue*sq){
    struct_sortqueue(sq);
    for(int i=sq->end-1;i>=0;i--){
        int temp=sq->nums[0];
        sq->nums[0]=sq->nums[i];
        sq->nums[i]=temp;
    adjust_sortedqueue(sq,0,i-1);
    }
}
void initi_sortedqueue(Sorted_Queue*sq,int i){
    sq->size=0;
    sq->start=0;
    sq->end=0;
    sq->id=i;
    printf("Please input the number of elements in this queue:\n");
    int n;
    scanf("%d",&n);
    printf("Please input the elements in this queue:\n");
    for(int i=0;i<n;i++){
        int val;
        scanf("%d",&val);
        q_push(sq,val);
    }
    sort_sortedqueue(sq);
}

void print_sortedqueue(Sorted_Queue*sq){
    for(int i=sq->start;i<sq->end;i++)printf("%d ",sq->nums[i]);
    printf("\n");
}

void push_minheap(MinHeap*m,Sorted_Queue*sq){
  if(sq->end==0)return ;
  m->queue[m->size++]=sq;
 
}

void adjust_minheap(MinHeap*m,int start){
    int end=m->size-1;
    
for(int l=start*2+1;l<=end;l=l*2+1){
    int min_id=l;
    if(l+1<=end&&q_top(m->queue[l]) > q_top(m->queue[l+1])){
        
        min_id=l+1;
    }
    
    if(q_top(m->queue[min_id])<q_top(m->queue[start])){
        
        Sorted_Queue* temp=m->queue[start];
        m->queue[start]=m->queue[min_id];
        m->queue[min_id]=temp;
        start=min_id;
        
    }
    else break;
    l=min_id;
}  
}
void struct_minheap(MinHeap*m){
   for(int i=(m->size-1)/2;i>=0;i--){
    adjust_minheap(m,i);
   } 
}
int m_top(MinHeap*m){
    if(m->size<=0){
        printf("The heap is empty!\n");
        return -1;
    }
    int x= q_top(m->queue[0]);
    q_pop(m->queue[0]);
    if(m->queue[0]->size==0){
        Sorted_Queue* temp=m->queue[0];
        m->queue[0]=m->queue[m->size-1];
        m->queue[m->size-1]=temp;
        m->size--;
    }
    
    adjust_minheap(m,0);
    return x;
}
int main(){
    Sorted_Queue sortedqueue[M];
    MinHeap minheap;
    minheap.size=0;
    for(int i=0;i<M;i++){
    printf("The %dth queue:\n",i+1);
    initi_sortedqueue(&sortedqueue[i],i);
    push_minheap(&minheap,&sortedqueue[i]);
    }
    
    struct_minheap(&minheap);
    Sorted_Queue  ans;
    ans.start=0;
    ans.end=0;
    while(minheap.size>0){
        ans.nums[ans.end++]=m_top(&minheap);
        ans.size++;
    }
    printf("After merging,the queue is :\n");
    for(int i=0;i<ans.end;i++)printf("%d ",ans.nums[i]);
	return 0;
} 

```
## 结果分析 
设第i个队列中元素的个数为ni,即从第一个到第m个的队列的元素个数为n1、n2……nm,设其中最大的为max,总个数为n.
1. m个队列排序的时间复杂度为O(mlogmax).
2. 最小堆要pop出n个元素，每次pop后都要进行维护调整，所以时间复杂度为O(nlogm).
3. 综上所述，整个程序的时间复杂度为O(nlogm+mlogmax)。 

