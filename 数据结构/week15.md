# Lab Week 15. 
## 实验内容：Double Hashing
Consider a hash table with M = 26 = 64 bins. Given a 12-bit unsigned number, take the lowest 6 bits as the primary hash function to get bin number, and the highest 6 bits as the secondary 
hash function to get jump size. 
Refer to the example in Lecture 28, Slide 93-107.
(1) Determine the array structures for a hash table
(2) Hash a list of 40 random 12-bit positive numbers
(3) Calculate the average number of probes
##  实验重点与代码分析 
1. 创建存放元素的位置Hash_bin的结构体，包括它在哈希表中的下标index,存放的元素val，和是否为空的标志empty,empty为1时此位置为空，为0时不为空，是否为空的标志empty方便我们后续插入判断此位置是否为空，决定是否要插入进去或者寻找其他空位。
```c
typedef struct Hash_bin{
    int empty;
    int val;
    int index;
}Hash_bin;
```
2. 创建哈希表的结构体，包括一个存放元素的数组,size记录数组的大小，cur_size记录哈希表中元素的数目，probe_time记录探测次数。
```c
typedef struct Hash_table{
    int size;
    int cur_size;
    int probe_time;
    Hash_bin hash_table[Size];
}Hash_table;
```
3. 初始化哈希表，将size设为数组的大小，将cur_size、probe_time赋值为0，数组中所有位置的empty标志设为1，表示空位置。
```c
void initial_hashtable(Hash_table*hashtable){
hashtable->size=Size;
hashtable->probe_time=0;
hashtable->cur_size=0;
for(int i=0;i<hashtable->size;i++){
    hashtable->hash_table[i].empty=1;
    hashtable->hash_table[i].index=i;
}
}
```
4. 取元素的低六位作为哈希值，可用位运算操作
```c
int hash(int x,Hash_table*hashtable){
    int m=hashtable->size;
    return x&(m-1);
}
```
5. 取元素的高六位作为条约的跨度，可用位运算操作
```c
int hash_jump(int x,Hash_table*hashtable){
    int j=(x>>6);
    if(j%2==1)return j;
    else return j|1;

}
```
6. 哈希表的插入函数，若哈希表满，则返回、不进行此操作；否则取元素的哈希值，若哈希表的元素的哈希值位置为空，则直接插入，且probe_time++，hashtable->cur_size加一，若第一次插入的位置已满，则利用第二次哈希得到探测的幅度，不断在初始的哈希值位置向后探测，直到寻找到了空位置，每次探测或是插入hashtable->cur_size加一
```c
void insert(int val,Hash_table*hashtable){
if(hashtable->cur_size==hashtable->size){
    printf("The hash table is full!\n");
    return;
}
int m=hashtable->size;
int h=hash(val,hashtable);
while(!hashtable->hash_table[h].empty){
hashtable->probe_time++;
h=(h+hash_jump(val,hashtable))%m;
}
hashtable->hash_table[h].empty=0;
hashtable->hash_table[h].val=val;
hashtable->probe_time++;
hashtable->cur_size++;
}
```
7. 打印输出哈希表中的内容
```c
void print_hashtable(Hash_table*hashtable){
    printf("The hashtable:\n");
    for(int i=0;i<hashtable->size;i++){
        if(hashtable->hash_table[i].empty==0)printf("index:%d   element:%d\n",i,hashtable->hash_table[i].val);
        else printf("index:%d   element:NULL\n");
    }
}
```
## 测试样例
1. 样例一
Insert 1990
Insert 1737
Insert 1897
Insert 2681
Insert 2219
Insert 1381
Insert 251
Insert 472
Insert 2452
Insert 987
Insert 3907
Insert 581
Insert 3518
Insert 327
Insert 2685
Insert 606
Insert 866
Insert 3281
Insert 1986
Insert 2853
Insert 1671
Insert 3375
Insert 639
Insert 2983
Insert 1562
Insert 364
Insert 2597
Insert 4060
Insert 3904
Insert 3816
Insert 2082
Insert 3791
Insert 747
Insert 161
Insert 529
Insert 1022
Insert 2476
Insert 2992
Insert 3117
Insert 2052
The hashtable: 
index:0   element:3904
index:1   element:NULL
index:2   element:1986
index:3   element:3907
index:4   element:2052
index:5   element:581
index:6   element:1990
index:7   element:327
index:8   element:NULL
index:9   element:1737
index:10   element:NULL
index:11   element:NULL
index:12   element:NULL
index:13   element:1022
index:14   element:2597
index:15   element:3791
index:16   element:NULL
index:17   element:3281
index:18   element:2853
index:19   element:2476
index:20   element:2452
index:21   element:NULL
index:22   element:NULL
index:23   element:NULL
index:24   element:472
index:25   element:NULL
index:26   element:1562
index:27   element:987
index:28   element:4060
index:29   element:NULL
index:30   element:606
index:31   element:NULL
index:32   element:NULL
index:33   element:161
index:34   element:866
index:35   element:529
index:36   element:2082
index:37   element:1381
index:38   element:NULL
index:39   element:2983
index:40   element:3816
index:41   element:1897
index:42   element:NULL
index:43   element:2219
index:44   element:364
index:45   element:3117
index:46   element:NULL
index:47   element:3375
index:48   element:2992
index:49   element:NULL
index:50   element:NULL
index:51   element:1671
index:52   element:NULL
index:53   element:NULL
index:54   element:747
index:55   element:NULL
index:56   element:NULL
index:57   element:2681
index:58   element:NULL
index:59   element:251
index:60   element:NULL
index:61   element:2685
index:62   element:3518
index:63   element:639
The probe time is 53
The average is 1.325000
2. 样例二
Insert 2160
Insert 3599
Insert 1042
Insert 598
Insert 2000
Insert 771
Insert 2355
Insert 3901
Insert 895
Insert 775
Insert 312
Insert 2871
Insert 3764
Insert 576
Insert 1605
Insert 1066
Insert 2762
Insert 2928
Insert 2975
Insert 1432
Insert 1357
Insert 1472
Insert 3170
Insert 3198
Insert 3157
Insert 1981
Insert 626
Insert 636
Insert 3346
Insert 2976
Insert 2483
Insert 1077
Insert 169
Insert 2258
Insert 893
Insert 1785
Insert 2878
Insert 1711
Insert 3658
Insert 418
The hashtable: 
index:0   element:576
index:1   element:NULL
index:2   element:NULL
index:3   element:771
index:4   element:NULL
index:5   element:1605
index:6   element:NULL
index:7   element:775
index:8   element:NULL
index:9   element:NULL
index:10   element:2762
index:11   element:NULL
index:12   element:418
index:13   element:1357
index:14   element:NULL
index:15   element:3599
index:16   element:2000
index:17   element:NULL
index:18   element:1042
index:19   element:NULL
index:20   element:NULL
index:21   element:3157
index:22   element:598
index:23   element:1472
index:24   element:1432
index:25   element:NULL
index:26   element:2483
index:27   element:NULL
index:28   element:1981
index:29   element:2928
index:30   element:NULL
index:31   element:2975
index:32   element:2976
index:33   element:NULL
index:34   element:3170
index:35   element:NULL
index:36   element:893
index:37   element:NULL
index:38   element:NULL
index:39   element:NULL
index:40   element:NULL
index:41   element:169
index:42   element:1066
index:43   element:2878
index:44   element:NULL
index:45   element:NULL
index:46   element:3658
index:47   element:1711
index:48   element:2160
index:49   element:3346
index:50   element:626
index:51   element:2355
index:52   element:3764
index:53   element:1077
index:54   element:NULL
index:55   element:2871
index:56   element:312
index:57   element:1785
index:58   element:NULL
index:59   element:2258
index:60   element:636
index:61   element:3901
index:62   element:3198
index:63   element:895
The probe time is 64
The average is 1.600000
## 完整代码
```c
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#define Size 64
typedef struct Hash_bin{
    int empty;
    int val;
    int index;
}Hash_bin;
typedef struct Hash_table{
    int size;
    int cur_size;
    int probe_time;
    Hash_bin hash_table[Size];
}Hash_table;

void initial_hashtable(Hash_table*hashtable){
hashtable->size=Size;
hashtable->probe_time=0;
hashtable->cur_size=0;
for(int i=0;i<hashtable->size;i++){
    hashtable->hash_table[i].empty=1;
    hashtable->hash_table[i].index=i;
}
}
int hash(int x,Hash_table*hashtable){
    int m=hashtable->size;
    return x&(m-1);
}
int hash_jump(int x,Hash_table*hashtable){
    int j=(x>>6);
    if(j%2==1)return j;
    else return j|1;

}
void insert(int val,Hash_table*hashtable){
if(hashtable->cur_size==hashtable->size){
    printf("The hash table is full!\n");
    return;
}
int m=hashtable->size;
int h=hash(val,hashtable);
while(!hashtable->hash_table[h].empty){
hashtable->probe_time++;
h=(h+hash_jump(val,hashtable))%m;
}
hashtable->hash_table[h].empty=0;
hashtable->hash_table[h].val=val;
hashtable->probe_time++;
hashtable->cur_size++;
}
void print_hashtable(Hash_table*hashtable){
    printf("The hashtable:\n");
    for(int i=0;i<hashtable->size;i++){
        if(hashtable->hash_table[i].empty==0)printf("index:%d   element:%d\n",i,hashtable->hash_table[i].val);
        else printf("index:%d   element:NULL\n");
    }
}
int main(){
	srand((unsigned)time(NULL));
    Hash_table hashtable;
    initial_hashtable(&hashtable);
	int n=pow(2,12);
    for(int i=0;i<40;i++){
        int x=rand()%n;
        insert(x,&hashtable);
        printf("Insert %d\n",x);
    }
    print_hashtable(&hashtable);
    printf("The probe time is %d\n",hashtable.probe_time);
    printf("The average is %f\n",(double)(hashtable.probe_time)/hashtable.cur_size);
	return 0;
} 
```
## 结果分析 
设插入的元素个数为n,hashtable 的容量为 n。
1. 每次哈希表插入元素的最优情况是第一次经过哈希函数得到的位置为空，直接插入，实现插入的时间复杂度为O(1)，故最理想的时候就复杂度为O(n)；但随着插入的元素个数n越来越大，每次实现O(1)插入的可能性越来越小，越来越可能发生哈希冲突，要经过二次哈希来避免冲突。
2. 最坏情况,假设 hashtable 的容量为 n ，当 hashtable 只剩下一个空位置且刚好第 n 次探测才找到，此时的时间复杂度为 O(n)。 
3. 大多数情况下每次哈希表插入元素的时间复杂度为O(1)，只有最后几次插入也可能会达到O(n)，当达到O(n)的插入时间复杂度时，我们需要考虑是否要将哈希表扩容。
4. 综上所述，考虑到插入的元素个数n越来越大，越来越可能发生哈希冲突，插入操作的时间复杂度不会是1的可能性越来越小，但平均情况下接近 O(1)，如果随着插入的元素个数n越来越大导致探测时间接近n，我们需要考虑是否要将哈希表扩容。

