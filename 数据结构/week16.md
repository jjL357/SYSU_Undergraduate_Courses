# Lab Week 16 
## 实验内容
(1) Determine the array structure for a complete max-heap  
(2) Input 20 random numbers, building a complete max-heap to store the numbers by making use of the process of heapification  
(3) Implement an in-place sorting, making use of POP operation for the built complete max-heap  
##  实验重点与代码分析 
//插入数据时，下标为1开始插入
1. 创建堆的结构体,包括存放数据的数组和目前数组元素的个数大小size
```c  
typedef struct Max_heap{
int nums[Size];
int size;
}Max_heap;
```
2. 初始化数组
```c
void initial_maxheap(Max_heap*m){
    m->size=0;
}
```
3. 交换函数
```c
void swap(int*nums,int i,int j){
    int temp=nums[i];
    nums[i]=nums[j];
    nums[j]=temp;
}
```
4. 调整堆：将选定顶点的左儿子与右儿子比较大小，取数值较为大的点与父亲节点比较，若小于则退出，否则交换两点，之后再将选中交换后的儿子节点继续重复此操作直到到达堆顶或者是退出。
```c  
void MaxheapAdjust(Max_heap*m,int start,int end){
    for(int i=start*2;i<=end;i=i*2) {
    if(i+1<=end&&m->nums[i]<m->nums[i+1]){
        i=i+1;
    }
    if(m->nums[i]>m->nums[start]){
    swap(m->nums,start,i);
    start=i;
    }
    else break;
    }
}
```
5. 堆的排序：先通过堆的调整构建起大顶堆，之后再将堆顶的元素和堆末尾的元素交换，再从堆顶向下调整堆，第二次，再将堆顶的元素和堆末尾倒数第二个的元素交换重复调整操作直到需要交换元素的下标相等，完成后排序结束。
```c
void struct_Maxheap(Max_heap*m){
    for(int i=(m->size-1)/2;i>=0;i--){
        MaxheapAdjust(m,i,m->size-1);
    }
    for(int i=m->size-1;i>=0;i--){
        swap(m->nums,0,i);
        MaxheapAdjust(m,0,i-1);
    }
}
```
6. 堆中元素的打印输出
```c
void printMaxheap(Max_heap*m){
    for(int i=0;i<m->size;i++){
        printf("%d ",m->nums[i]);
    }
    printf("\n");
}
```

## 测试样例
1. 样例一
Before sorting:
4674 11334 27102 21783 4762 23292 17353 13686 18018 25399 22936 7368 31083 25947 32432 13106 3162 29764 7183 6408
After sorting:
3162 4674 4762 6408 7183 7368 11334 13106 13686 17353 18018 21783 22936 23292 25399 25947 27102 29764 31083 32432 
2. 样例二
Before sorting:
5739 9139 18118 2104 24977 25454 16204 9666 26845 19977 7487 21727 6629 13330 25349 29228 29540 7075 14173 7738
After sorting:
2104 5739 6629 7075 7487 7738 9139 9666 13330 14173 16204 18118 19977 21727 24977 25349 25454 26845 29228 29540
## 完整代码
```c
#include<stdio.h>
#include<malloc.h>
#include<stdlib.h>
#include<time.h>
#define Size 100
//插入数据时，下标为1开始插入
typedef struct Max_heap{
int nums[Size];
int size;
}Max_heap;
void initial_maxheap(Max_heap*m){
    m->size=0;
}
void swap(int*nums,int i,int j){
    int temp=nums[i];
    nums[i]=nums[j];
    nums[j]=temp;
}
void MaxheapAdjust(Max_heap*m,int start,int end){
    for(int i=start*2;i<=end;i=i*2) {
    if(i+1<=end&&m->nums[i]<m->nums[i+1]){
        i=i+1;
    }
    if(m->nums[i]>m->nums[start]){
    swap(m->nums,start,i);
    start=i;
    }
    else break;
    }
}
void struct_Maxheap(Max_heap*m){
    for(int i=(m->size-1)/2;i>=0;i--){
        MaxheapAdjust(m,i,m->size-1);
    }
    for(int i=m->size-1;i>=0;i--){
        swap(m->nums,0,i);
        MaxheapAdjust(m,0,i-1);
    }
}
void printMaxheap(Max_heap*m){
    for(int i=0;i<m->size;i++){
        printf("%d ",m->nums[i]);
    }
    printf("\n");
}
int main(){
srand((unsigned)time(NULL));
Max_heap m;
for(int i=0;i<20;i++){
    int x=rand();
    m.nums[i]=x;
}
m.size=20;
printf("Before sorting:\n");
printMaxheap(&m);
struct_Maxheap(&m);
printf("After sorting:\n");
printMaxheap(&m);
return 0;
}
```
## 结果分析 
（1）每一次调整堆的操作。最坏情况下要从上至下遍历棵树，假设该二叉树的节点个数为 n，则每一次调整的时间复杂度为O(lnn)。
（2）在堆排序的过程中，先构建好大顶堆，要进行n次的交换，每次交换后都要进行调整操作，每次调整堆的时间复杂度为 O(lnn) ，所以此排序过程的时间复杂度为 O(nlnn)。  
（3）综上所述，整个程序的时间复杂度为O(nlnn)。

