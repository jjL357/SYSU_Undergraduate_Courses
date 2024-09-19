# 期中小测题目二

## 实验内容
题目二：可变分区存储管理模拟器
用一个足够大的字节数组 S 模拟管理的空间对象。到达任务 A 的信息包含任务 ID 和申请空间字节数 R，离开任务 D 的信息包含任务 ID。
建立一个单链表 U 管理已分配空间，表结点信息包括对应的任务 ID、分配首地址 Td（对应的 S 的下标）和占用字节数 R。链表 U 按照任务 ID 的递增排序，初始化为空链表。
建立一个双循环链表 F 管理空白分区，表结点信息包含对应的空白分区首地址 Fd（对应的 S 的下标）和包含的字节数 P。链表 F 按照 Fd 的递增排序，初始化为单一包含整个 S 的结点，而且当前指针 h 指向 S 的第一个下标（即0）。
分配：从 F 的当前指针 h 开始为到达任务 A 向前搜索一个 P ≥ R 的结点 M 并随时更新 h，循环回到出发点时报错（无法分配），否则按照任务 ID、Fd（Td = Fd）和 R 构造一个 U 的结点 N 并插入 U，同时维护 M 的信息（Fd = Fd + R, P = P – R。当 P = 0 时删除 M）。
回收：在 U 中搜索离开任务 D 的 ID 对应的结点 N，失败时报错，否则按照 Td（Fd = Td）和 R（P = R）构造一个 F 的结点 M，将 M 插入 F，必要时合并邻接的空白区；从 U 中删除 N。
1. 用 C 语言编写实现上述需求的程序，并设计适当的测试用例对程序进行验证。
2. 上述算法的分配部分的空白区搜索策略称为 Next-fit。你能否列举一些其他的搜索策略。

## 实验重点
1. 构建一个单链表管理已分配空间，链表节点按任务id值递增排序，每个节点要保存任务id、分配首地址和占用字节数，方便后续回收空间。
2. 建立一个双循环链表 F 管理空白分区，记录首地址和包含的字节数 ，若空白区域相邻要进行合并，不相邻用不同节点表示。
3. 创建任务时先从空白分区分配空间，若空间不足够，则分配失败，不创建任务；否则截取空白分区，再将任务信息添加到任务管理链表。
4. 任务离开时，要进行空间回收，从任务管理链表获取信息，再将对应的空间回收到空白空间，若相邻则要合并。
## 设计思路
1. 定义结构体:Used_list管理已分配空间,Empty_list管理空白分区
```c
typedef struct Used_list{//管理已分配空间
int start;
int size;
int id;
struct Used_list*next;
}Used_list;
typedef struct Empty_list{ //管理空白分区
int start;
int size;
struct Empty_list* pre;
struct Empty_list* next;
}Empty_list;
```
2. 结构体的初始化,两个链表的头节点不放数据，为操作方便。EmptyList的头节点的next存放整个空白空间。
```c
Used_list* init_Usedlist(){
   Used_list*u=(Used_list*)malloc(sizeof(Used_list));//为操作方便构造头节点，不放任何数据
   u->next=NULL;
   return u;
}

Empty_list * init_Emptylist(){
Empty_list*e=(Empty_list*)malloc(sizeof(Empty_list));//为操作方便构造头节点，不放任何数据
Empty_list*e_next=(Empty_list*)malloc(sizeof(Empty_list));//初始化为单一包含整个 S 的结点
e_next->size=Size;
e_next->start=0;
e->next=e_next;//构造循环链表
e->pre=e_next;
e_next->next=e;
e_next->pre=e;
return e;
}
```
3. 创建任务并分配空间,空间不足无法分配时flag=0
```c
Empty_list*malloc_space_emptylist(Empty_list*e,int space,int *flag,int *start){//为任务分配空间，无法分配时flag=0
Empty_list*cur=e->next;
while(cur!=e){
 if(cur->size > space){
        *start=cur->start;
        cur->size-=space;
        cur->start+=space;
        *flag=1;
        return e;
    }
else if(cur->size==space){
    *flag=1;
    *start=cur->start;
    cur->pre->next=cur->next;
    cur->next->pre=cur->pre;
    free(cur);
    return e;
}
else cur=cur->next;
}
return e;
} 

Used_list* malloc_space_usedlist(Used_list*u,int space,int id,int start){//创建任务
Used_list*temp=(Used_list*)malloc(sizeof(Used_list));
temp->id=id;
temp->size=space;
temp->start=start;
Used_list*pre=u;
Used_list*cur=u->next;
while(cur!=NULL){

if(cur->id>temp->id){
    pre->next=temp;
    temp->next=cur;
    return u;
}

cur=cur->next;
pre=pre->next;
}
pre->next=temp;
temp->next=NULL;
return u;
}
```
4. 打印输出
```c
void print_usedlist(Used_list*u){//按id顺序打印输出任务id
    Used_list*cur=u->next;
    while(cur){
        printf("%d ",cur->id);
        if(cur->next)printf("-> ");
        cur=cur->next;
    }
    printf("\n");
}

void print_emptylist(Empty_list*e){//打印输出空白空间
    Empty_list*cur=e->next;
    printf("[");
    while(cur!=e){
        printf("(%d,%d)",cur->start,cur->start+cur->size-1);
        cur=cur->next;
        if(cur!=e)printf(",");
    }
    printf("]");
    printf("\n");
}
```
5. 任务离开，回收空间，从任务管理链表获取信息，再将对应的空间回收到空白空间
```c
int delete_usedlist(Used_list*u,int id,int*flag,int*delete_space){//离开任务，从任务链表中删除任务，若删除失败，flag=0
Used_list*pre=u;
Used_list*cur=u->next;
int delete_start;//记录删除任务起始位置
while(cur!=NULL){
if(cur->id==id){
    *flag=1;
    delete_start=cur->start;
    pre->next=cur->next;
    *delete_space=cur->size;
    free(cur);
    return delete_start;
}
cur=cur->next;
pre=pre->next;
}
return delete_start;
}

void recycle_emptylist(Empty_list*e,int space,int start){//回收空间
Empty_list*temp=(Empty_list*)malloc(sizeof(Empty_list));
temp->size=space;
temp->start=start;
Empty_list*cur=e->next;
if(cur==e){
    cur->next=temp;
    cur->pre=temp;
    temp->pre=cur;
    temp->next=cur;
    return;
}
while(cur!=e){
if((cur->start) > (temp->start)){
    
    Empty_list*temp2=cur->pre;
    cur->pre->next=temp;
    temp->pre=cur->pre;
    temp->next=cur;
    cur->pre=temp;
    return;
}
else cur=cur->next;
}
Empty_list*temp3=e->pre;
temp->pre=temp3;
temp3->next=temp;
temp->next=e;
e->pre=temp;
}
```

6. 合并空白空间
```c
void merge_emptylist(Empty_list*e){//合并空白空间
    Empty_list*cur=e->next;
    while(cur!=e){
    if(cur->next==e)return;
    if(cur->next!=e&&(cur->start+cur->size) >=cur->next->start){
        if((cur->next->start + cur->next->size - cur->start )> cur->size)cur->size =cur->next->start + cur->next->size - cur->start;
        Empty_list*temp=cur->next;
        cur->next=cur->next->next;
        cur->next->pre=cur;
        free(temp);
    }
    else cur=cur->next;
    }
}
```
7. free分配内存
```c
void free_emptylist(Empty_list*e){
    Empty_list*cur=e->next;
    while(cur!=e){
        Empty_list*temp=cur;
        cur=cur->next;
        free(temp);
    }
    free(e);
}

void free_usedlist(Used_list*u){
    Used_list*cur=u;
    while(cur){
        Used_list*temp=cur;
        cur=cur->next;
        free(temp);
    }
}
```

## 代码实现

```C
#include<stdio.h>
#include<malloc.h>
#define Size 1001
typedef struct Used_list{//管理已分配空间
int start;
int size;
int id;
struct Used_list*next;
}Used_list;
typedef struct Empty_list{ //管理空白分区
int start;
int size;
struct Empty_list*pre;
struct Empty_list*next;
}Empty_list;

Used_list* init_Usedlist(){
   Used_list*u=(Used_list*)malloc(sizeof(Used_list));//为操作方便构造头节点，不放任何数据
   u->next=NULL;
   return u;
}

Empty_list * init_Emptylist(){
Empty_list*e=(Empty_list*)malloc(sizeof(Empty_list));//为操作方便构造头节点，不放任何数据
Empty_list*e_next=(Empty_list*)malloc(sizeof(Empty_list));//初始化为单一包含整个 S 的结点
e_next->size=Size;
e_next->start=0;
e->next=e_next;//构造循环链表
e->pre=e_next;
e_next->next=e;
e_next->pre=e;
return e;
}

Empty_list*malloc_space_emptylist(Empty_list*e,int space,int *flag,int *start){//为任务分配空间，无法分配时flag=0
Empty_list*cur=e->next;
while(cur!=e){
 if(cur->size > space){
        *start=cur->start;
        cur->size-=space;
        cur->start+=space;
        *flag=1;
        return e;
    }
else if(cur->size==space){
    *flag=1;
    *start=cur->start;
    cur->pre->next=cur->next;
    cur->next->pre=cur->pre;
    free(cur);
    return e;
}
else cur=cur->next;
}
return e;
} 

Used_list* malloc_space_usedlist(Used_list*u,int space,int id,int start){//创建任务
Used_list*temp=(Used_list*)malloc(sizeof(Used_list));
temp->id=id;
temp->size=space;
temp->start=start;
Used_list*pre=u;
Used_list*cur=u->next;
while(cur!=NULL){

if(cur->id>temp->id){
    pre->next=temp;
    temp->next=cur;
    return u;
}

cur=cur->next;
pre=pre->next;
}
pre->next=temp;
temp->next=NULL;
return u;
}

void print_usedlist(Used_list*u){//按id顺序打印输出任务id
    Used_list*cur=u->next;
    while(cur){
        printf("%d ",cur->id);
        if(cur->next)printf("-> ");
        cur=cur->next;
    }
    printf("\n");
}

void print_emptylist(Empty_list*e){//打印输出空白空间
    Empty_list*cur=e->next;
    printf("[");
    while(cur!=e){
        printf("(%d,%d)",cur->start,cur->start+cur->size-1);
        cur=cur->next;
        if(cur!=e)printf(",");
    }
    printf("]");
    printf("\n");
}
int delete_usedlist(Used_list*u,int id,int*flag,int*delete_space){//离开任务，从任务链表中删除任务，若删除失败，flag=0
Used_list*pre=u;
Used_list*cur=u->next;
int delete_start;//记录删除任务起始位置
while(cur!=NULL){
if(cur->id==id){
    *flag=1;
    delete_start=cur->start;
    pre->next=cur->next;
    *delete_space=cur->size;
    free(cur);
    return delete_start;
}
cur=cur->next;
pre=pre->next;
}
return delete_start;
}

void recycle_emptylist(Empty_list*e,int space,int start){//回收空间
Empty_list*temp=(Empty_list*)malloc(sizeof(Empty_list));
temp->size=space;
temp->start=start;
Empty_list*cur=e->next;
if(cur==e){
    cur->next=temp;
    cur->pre=temp;
    temp->pre=cur;
    temp->next=cur;
    return;
}
while(cur!=e){
if((cur->start) > (temp->start)){
    
    Empty_list*temp2=cur->pre;
    cur->pre->next=temp;
    temp->pre=cur->pre;
    temp->next=cur;
    cur->pre=temp;
    return;
}
else cur=cur->next;
}
Empty_list*temp3=e->pre;
temp->pre=temp3;
temp3->next=temp;
temp->next=e;
e->pre=temp;
}
void merge_emptylist(Empty_list*e){//合并空白空间
    Empty_list*cur=e->next;
    while(cur!=e){
    if(cur->next==e)return;
    if(cur->next!=e&&(cur->start+cur->size) >=cur->next->start){
        if((cur->next->start + cur->next->size - cur->start )> cur->size)cur->size =cur->next->start + cur->next->size - cur->start;
        Empty_list*temp=cur->next;
        cur->next=cur->next->next;
        cur->next->pre=cur;
        free(temp);
    }
    else cur=cur->next;
    }
}

void free_emptylist(Empty_list*e){
    Empty_list*cur=e->next;
    while(cur!=e){
        Empty_list*temp=cur;
        cur=cur->next;
        free(temp);
    }
    free(e);
}

void free_usedlist(Used_list*u){
    Used_list*cur=u;
    while(cur){
        Used_list*temp=cur;
        cur=cur->next;
        free(temp);
    }
}
int main(){
    Used_list*U;
    Empty_list*E;
  
    U=init_Usedlist();
    E=init_Emptylist();
    printf("初始化:\n");
    printf("已分配任务id及其顺序:");
    print_usedlist(U);
    printf("空白分区:");
    print_emptylist(E);
    while(1){
        printf("请输入操作:\n");
        printf("(-1:退出程序   1:创建任务   0:回收任务)\n");
        int x;
        scanf("%d",&x);
        if(x==-1)break;
        if(x==1){
            int id;
            int flag=0;
            int space_size;
           printf("请输入创建任务id和创建任务的空间:\n");
            scanf("%d%d",&id,&space_size);
            int create_start;
            malloc_space_emptylist(E,space_size,&flag,&create_start);
            if(flag)malloc_space_usedlist(U,space_size,id,create_start);
            else{ 
                printf("分配空间失败\n");
                printf("分配失败后:\n");
                printf("已分配任务id及其顺序:");
                print_usedlist(U);
                printf("空白分区:");
                print_emptylist(E);
                continue;
            }
                printf("任务创建成功\n");
                printf("已分配任务id:");
                print_usedlist(U);
                printf("空白分区:");
                print_emptylist(E);
            }
        else{
            int flag=0;
            int space_size;
            printf("请输入回收的任务id:\n");
            int delete_id;
            scanf("%d",&delete_id);
            int delete_space;
            int delete_start=delete_usedlist(U,delete_id,&flag,&delete_space);
            if(flag)recycle_emptylist(E,delete_space,delete_start);
            else{
                printf("回收失败,回收任务id不存在\n");
                printf("回收失败后:\n");
                printf("已分配任务id及其顺序:");
                print_usedlist(U);
                printf("空白分区:");
                print_emptylist(E);
                continue;
            }   
                
                merge_emptylist(E); 
                printf("回收成功,回收任务id为%d\n",delete_id);
                printf("回收成功后:\n");
                printf("已分配任务id及其顺序:");
                print_usedlist(U);
                printf("空白分区:");
                print_emptylist(E);
        }
    }
    free_emptylist(E);
    free_usedlist(U);
    printf("结束程序\n");
    return 0;
}
```

## 测试样例
1. 
**输入**：
1
1 1
1
2 65
1
9 765
1 
5 86
0
4
0
5
0
1
0
2
0
9
-1

**输出**：
初始化:
已分配任务id及其顺序:
空白分区:[(0,1000)]
请输入操作:
(-1:退出程序   1:创建任务   0:回收任务)
请输入创建任务id和创建任务的空间:
任务创建成功
已分配任务id:1 
空白分区:[(1,1000)]
请输入操作:
(-1:退出程序   1:创建任务   0:回收任务)
请输入创建任务id和创建任务的空间:
任务创建成功
已分配任务id:1 -> 2 
空白分区:[(66,1000)]
请输入操作:
(-1:退出程序   1:创建任务   0:回收任务)
请输入创建任务id和创建任务的空间:
任务创建成功
已分配任务id:1 -> 2 -> 9 
空白分区:[(831,1000)]
请输入操作:
(-1:退出程序   1:创建任务   0:回收任务)
请输入创建任务id和创建任务的空间:
任务创建成功
已分配任务id:1 -> 2 -> 5 -> 9 
空白分区:[(917,1000)]
请输入操作:
(-1:退出程序   1:创建任务   0:回收任务)
请输入回收的任务id:
回收失败,回收任务id不存在
回收失败后:
已分配任务id及其顺序:1 -> 2 -> 5 -> 9 
空白分区:[(917,1000)]
请输入操作:
(-1:退出程序   1:创建任务   0:回收任务)
请输入回收的任务id:
回收成功,回收任务id为5
回收成功后:
已分配任务id及其顺序:1 -> 2 -> 9 
空白分区:[(831,1000)]
请输入操作:
(-1:退出程序   1:创建任务   0:回收任务)
请输入回收的任务id:
回收成功,回收任务id为1
回收成功后:
已分配任务id及其顺序:2 -> 9 
空白分区:[(0,0),(831,1000)]
请输入操作:
(-1:退出程序   1:创建任务   0:回收任务)
请输入回收的任务id:
回收成功,回收任务id为2
回收成功后:
已分配任务id及其顺序:9 
空白分区:[(0,65),(831,1000)]
请输入操作:
(-1:退出程序   1:创建任务   0:回收任务)
请输入回收的任务id:
回收成功,回收任务id为9
回收成功后:
已分配任务id及其顺序:
空白分区:[(0,1000)]
请输入操作:
(-1:退出程序   1:创建任务   0:回收任务)
结束程序
1. 

## 结果分析

 
 
 