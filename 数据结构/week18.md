# Lab Week 18
## 实验内容
Refer to the example in Lecture 30, Slide 71-166  
(1) Determine the array structure for a Quicksort algorithm  
(2) Input 20 random numbers and implement an in-place Quicksort
## 实验重点
快速排序 QuickSort 将采用分治策略来把一个数组按照跟基准元素的比较分为两个子数组，然后调用递归排序子数组。 
1. 挑选基准值  
挑出数组的首元素、尾元素和中间元素，选出它们中间的数，并放到数组的最后，这个中间数也就是“基准” pivot。 
2. 将基准值左右两边分为大于的一边和小于等于的一边
从数组的两边向中间遍历，实现所有比基准值小或相等的元素放在左边的数组，把所有比基准值大的元素放在右边的数组，直到左右两个遍历的指针相遇，结束遍历。在将基准值插入到对应的位置。此时基准值右边的数都大于基准值，左边的数都小于基准值。
3. 递归调用排序子序列  
递归地将小于基准值元素的子数组和大于基准值元素的子数组排序，直到切分的数组只有一个元素时，结束递归或有两个元素，决定是否交换后结束。  
## 代码分析 
1. 元素间的交换函数
```c
void swap(int*nums,int x,int y){
    int temp=nums[x];
    nums[x]=nums[y];
    nums[y]=temp;
}
void swap2(int *x,int *y){
    int temp=*x;
    *x=*y;
    *y=temp;
}
```
2. 寻找三个元素之间的中间值，方便后续寻找pivot。
```c
int find_midvalue(int x,int y,int z){
    if(x > y) swap2(&x, &y);
	if(y > z) swap2(&y, &z);
	if(x > y) swap2(&x, &y);
	return y;
}
```
3. 快排：挑选基准值pivot；通过双指针和交换实现基准；值右边的数都大于基准值，左边的数都等于小于基准值；递归调用排序子序列。  
```c
void quicksort(int *nums,int start,int end){
if(end-start<=0)return ;
else if(end-start==1){
    if(nums[start]>nums[end])swap(nums,start,end);
    return;
}
int mid=start+(end-start)/2;
int pivot=find_midvalue(nums[start],nums[mid],nums[end]);
int pivot_id=mid;
if(nums[start]==pivot)pivot_id=start;
else if(nums[end]==pivot)pivot_id=end;
swap(nums,end,pivot_id);
int l=start,r=end-1;
while(l<r){
while(nums[r]>=pivot&&r>l)r--;
while(nums[l]<pivot&&r>l)l++;
if(r>l){
    swap(nums,l,r);
}
}
if(nums[r]<=pivot)r++;
swap(nums,r,end);
quicksort(nums,start,r-1);
quicksort(nums,r+1,end);
}
```
## 测试样例
1. 样例一
Before sorting:
6781 13603 14851 13912 17449 26421 22983 27491 17736 27491 32226 5986 31132 8624 27740 30219 17561 23572 31328 26288
After sorting:
5986 6781 8624 13603 13912 14851 17449 17561 17736 22983 23572 26288 26421 27491 27491 27740 30219 31132 31328 32226
1. 样例二
Before sorting:
9315 31294 16584 9284 26167 26139 13413 16917 18442 7950 15755 7398 7699 3517 12890 16929 20643 1930 10174 29656
After sorting:
1930 3517 7398 7699 7950 9284 9315 10174 12890 13413 15755 16584 16917 16929 18442 20643 26139 26167 29656 31294
## 完整代码
```c
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<string.h>
void swap(int*nums,int x,int y){
    int temp=nums[x];
    nums[x]=nums[y];
    nums[y]=temp;
}
void swap2(int *x,int *y){
    int temp=*x;
    *x=*y;
    *y=temp;
}
int find_midvalue(int x,int y,int z){
    if(x > y) swap2(&x, &y);
	if(y > z) swap2(&y, &z);
	if(x > y) swap2(&x, &y);
	return y;
}
void quicksort(int *nums,int start,int end){
if(end-start<=0)return ;
else if(end-start==1){
    if(nums[start]>nums[end])swap(nums,start,end);
    return;
}
int mid=start+(end-start)/2;
int pivot=find_midvalue(nums[start],nums[mid],nums[end]);
int pivot_id=mid;
if(nums[start]==pivot)pivot_id=start;
else if(nums[end]==pivot)pivot_id=end;
swap(nums,end,pivot_id);
int l=start,r=end-1;
while(l<r){
while(nums[r]>=pivot&&r>l)r--;
while(nums[l]<pivot&&r>l)l++;
if(r>l){
    swap(nums,l,r);
}
}
if(nums[r]<=pivot)r++;
swap(nums,r,end);
quicksort(nums,start,r-1);
quicksort(nums,r+1,end);
}
int main(){
    int nums[20];
srand((unsigned)time(NULL));
for(int i=0;i<20;i++){
    int x=rand();
    nums[i]=x;
}
printf("Before sorting:\n");
for(int i=0;i<20;i++)printf("%d ",nums[i]);
printf("\n");
quicksort(nums,0,20-1);
printf("After sorting:\n");
for(int i=0;i<20;i++)printf("%d ",nums[i]);
printf("\n");
return 0;
}
```
## 结果分析 
尽管取了最左最右最中间的中间值，也不能保证是最优情况。
1. 最优情况下时间复杂度：快速排序最优的情况就是每一次取到的元素都刚好平分整个数组快速排序最优的情况下时间复杂度为：O( nlogn )
2. 最差情况下时间复杂度：最差的情况就是每一次取到的元素就是数组中最小/最大的，这种情况其实就是冒泡排序了(每一次都排好一个元素的顺序)快速排序最差的情况下时间复杂度为：O( n^2 )
3. 平均时间复杂度：快速排序的平均时间复杂度也是：O(nlogn)
4. 空间复杂度：首先就地快速排序使用的空间是O(1)的，也就是个常数级；而真正消耗空间的就是递归调用了，因为每次递归就要保持一些数据最优的情况下空间复杂度为：O(logn)；每一次都平分数组的情况最差的情况下空间复杂度为：O( n ) ；退化为冒泡排序的情况。


