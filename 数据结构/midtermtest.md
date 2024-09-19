# 期中小测题目一

## 实验内容
题目一：四元 Huffman 编码
给定一个非负叶子权序列，将构造二元 Huffman 算法的迭代步作如下修改：k 个权值由小到大排序，以最小的四个权值从小到大作为四个儿子构造其父亲结点，新构造的结点的权值是此四个儿子的权值之和。在权序列中删去此四个儿子对应的权值，增加新构造的父亲结点的权值。因此，每一次迭代使得权序列的元素数目减少3。
1. 叶子的初始数目需要符合什么样的约束，才能保证算法得到一棵四元正则根树？
2. 在算法得到的四元正则有根有序树（必要时适当增加权值为 0 的叶子以满足上述结论）上，将每个内部结点的外向边从左到右编码为 00, 01, 10, 11，得到一棵四元编码树。
3. 确定适当的数据结构，用 C 语言编写实现上述需求的程序，并设计适当的测试用例对程序进行验证。

## 实验重点以及设计思路
1. 叶子的初始数目n需要符合n%3=1，才能保证算法得到一棵四元正则根树,所以构造前适当增加权值为 0 的叶子。当n%3=0时补充一个0，当n%3==2时补充两个0.
2. 利用排序或者遍历要找到最小的四个节点构造哈夫曼树。
3. 要记录叶子间的使用情况，防止重复利用。
4. 可以利用回溯算法对哈夫曼树的叶子节点进行编码。
5. 注意使用结构体时，不能直接复制相等，易出现段错误。   
## 代码实现

```C
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<limits.h>
#include<malloc.h>
typedef struct Tree{
char*Hcode;//节点的哈夫曼编码 
int val;
int isleaf;//是否是被用为叶子节点，1是，0不是 
int number;
struct Tree **members;
}Tree;

int min(Tree*t,int tree_k){//寻找最小且未被遍历的节点 
	int index=-1,min=INT_MAX;
	for(int i=0;i<tree_k;i++){
	if(t[i].isleaf<1&&t[i].val<min){
		min=t[i].val;
		index=i;	
	}
	}
	if(index!=-1)t[index].isleaf=1;
	return index;	
}

void init(Tree*t,int*nums,int k){//初始化节点 

	for(int i=0;i<k;i++){
		t[i].val=nums[i];
        t[i].number=0;
		t[i].isleaf=0;
		t[i].members=(Tree**)malloc(sizeof(Tree*)*4);
		
	}
}

void create(Tree*t,int tree_k){//创造新节点和哈夫曼树 
	int sum=0;
	t[tree_k].val=0;
	t[tree_k].number=0;
    t[tree_k].members=(Tree**)malloc(sizeof(Tree*)*4);
	while(t[tree_k].number<4){
    int temp=min(t,tree_k);
	sum=t[temp].val;
	t[tree_k].val+=sum;
	t[tree_k].number++;
	t[tree_k].members[t[tree_k].number-1]=&t[temp];
	}
	t[tree_k].isleaf=0;
}

void createcode(Tree*t,char *s){//利用回溯构建哈夫曼编码 
	char c[100];
	for(int i=0;i<strlen(s);i++)c[i]=s[i];
    c[strlen(s)]='\0';
	if(t->isleaf>=0){
        t->Hcode=(char*)malloc(sizeof(char)*100);
		if(t->number<4){
        for(int i=0;i<strlen(s);i++){
			t->Hcode[i]=s[i];
		}
        }
        t->Hcode[strlen(s)]='\0';
		for(int i=0;i<t->number;i++){
			if(i==0){
			int l=strlen(s);
			s[l]='0';
			s[l+1]='0';
			s[l+2]='\0';
			createcode(t->members[i],s);
            for(int j=0;j<strlen(c);j++)s[j]=c[j];
           s[strlen(c)]='\0';

			}
			else if(i==1){
			int l=strlen(s);
			s[l]='0';
			s[l+1]='1';
			s[l+2]='\0';
			createcode(t->members[i],s);
            for(int j=0;j<strlen(c);j++)s[j]=c[j];
            s[strlen(c)]='\0';

			}
			else if(i==2){
			int l=strlen(s);
			s[l]='1';
			s[l+1]='0';
			s[l+2]='\0';
			createcode(t->members[i],s);
            for(int j=0;j<strlen(c);j++)s[j]=c[j];
           s[strlen(c)]='\0';

			}
			else if(i==3){
			int l=strlen(s);
			s[l]='1';
			s[l+1]='1';
			s[l+2]='\0';
			createcode(t->members[i],s);
             for(int j=0;j<strlen(c);j++)s[j]=c[j];
           s[strlen(c)]='\0';
			}
		}
		}
} 
void print(Tree*trees,int cur_k){//打印输出哈夫曼编码 
for(int i=0;i<cur_k;i++)printf("%d:%s\n",trees[i].val,trees[i].Hcode);
printf("\n");
}
int main(){
    int k;
	printf("Please input the number of weights:\n");
	scanf("%d",&k);
	int*vals=(int*)malloc(sizeof(int)*(k+1));
	printf("Please input the weights:\n");
	for(int i=0;i<k;i++)scanf("%d",&vals[i]);
    if(k%3==0){
        vals[k++]=0;
    }
    else if(k%3==2){
        vals[k++]=0;
        vals[k++]=0;
    }
	int cur_k=k;
	int tree_k=k;
	Tree*trees=(Tree*)malloc(sizeof(Tree)*(2*k+1));
	init(trees,vals,cur_k);
    while(1){
	    int full=0;
		for(int j=0;j<tree_k;j++){
			if(trees[j].isleaf<=0&&j!=tree_k-1)break;
			if(j==tree_k-1)full=1;
		}
		if(full)break;
		create(trees,tree_k);
		tree_k++;
		}
	printf("The nodes are:\n");
    for(int i=0;i<tree_k;i++)printf("%d ",trees[i].val);
    printf("\n");
	char c[100];
	c[0]='\0';
	printf("The nodes and the codes are:\n");
	createcode(&trees[tree_k-1],c);
	print(trees,cur_k);
	free(vals);
	free(trees);
    return 0;
}
```

## 代码分析
1. 结构体定义
```c
typedef struct Tree{
char*Hcode;//节点的哈夫曼编码 
int val;
int isleaf;//是否是被用为叶子节点，1是，0不是 
int number;
struct Tree **members;
}Tree;
```
2. 遍历所有节点一遍，找到权重最小且未成为其他节点子树的节点的下标
```c
int min(Tree*t,int tree_k){//寻找最小且未被遍历的节点 
	int index=-1,min=INT_MAX;
	for(int i=0;i<tree_k;i++){
	if(t[i].isleaf<1&&t[i].val<min){
		min=t[i].val;
		index=i;	
	}
	}
	if(index!=-1)t[index].isleaf=1;
	return index;	
}
```
3. 初始化各个节点
```c   
void init(Tree*t,int*nums,int k){//初始化节点 
	for(int i=0;i<k;i++){
		t[i].val=nums[i];
        t[i].number=0;
		t[i].isleaf=0;
		t[i].members=(Tree**)malloc(sizeof(Tree*)*4);
	}
}
```
4. 创造新节点和构建哈夫曼树 
```c
void create(Tree*t,int tree_k){//创造新节点和哈夫曼树 
	int sum=0;
	t[tree_k].val=0;
	t[tree_k].number=0;
    t[tree_k].members=(Tree**)malloc(sizeof(Tree*)*4);
	while(t[tree_k].number<4){
    int temp=min(t,tree_k);
	sum=t[temp].val;
	t[tree_k].val+=sum;
	t[tree_k].number++;
	t[tree_k].members[t[tree_k].number-1]=&t[temp];
	}
	t[tree_k].isleaf=0;
}
```
5. 利用回溯构建哈夫曼编
```c 
void createcode(Tree*t,char *s){//利用回溯构建哈夫曼编码 
	char c[100];
	for(int i=0;i<strlen(s);i++)c[i]=s[i];
    c[strlen(s)]='\0';
	if(t->isleaf>=0){
        t->Hcode=(char*)malloc(sizeof(char)*100);
		if(t->number<4){
        for(int i=0;i<strlen(s);i++){
			t->Hcode[i]=s[i];
		}
        }
        t->Hcode[strlen(s)]='\0';
		for(int i=0;i<t->number;i++){
			if(i==0){
			int l=strlen(s);
			s[l]='0';
			s[l+1]='0';
			s[l+2]='\0';
			createcode(t->members[i],s);
            for(int j=0;j<strlen(c);j++)s[j]=c[j];
           s[strlen(c)]='\0';

			}
			else if(i==1){
			int l=strlen(s);
			s[l]='0';
			s[l+1]='1';
			s[l+2]='\0';
			createcode(t->members[i],s);
            for(int j=0;j<strlen(c);j++)s[j]=c[j];
            s[strlen(c)]='\0';

			}
			else if(i==2){
			int l=strlen(s);
			s[l]='1';
			s[l+1]='0';
			s[l+2]='\0';
			createcode(t->members[i],s);
            for(int j=0;j<strlen(c);j++)s[j]=c[j];
           s[strlen(c)]='\0';

			}
			else if(i==3){
			int l=strlen(s);
			s[l]='1';
			s[l+1]='1';
			s[l+2]='\0';
			createcode(t->members[i],s);
             for(int j=0;j<strlen(c);j++)s[j]=c[j];
           s[strlen(c)]='\0';

			}
		}
		}
} 
```
6. 当除最大的节点以外的节点都成为其他节点的子树时停止构造
```c
    while(1){
	    int full=0;
		for(int j=0;j<tree_k;j++){
			if(trees[j].isleaf<=0&&j!=tree_k-1)break;
			if(j==tree_k-1)full=1;
		}
		if(full)break;
		create(trees,tree_k);
		tree_k++;
		}
```
7. 释放空间
```c
free(vals);
free(trees);
```

## 测试样例
1. 
输入：
8
1 2 3 4 5 6 7 8 
输出：
Please input the number of weights:
8
Please input the weights:
1 2 3 4 5 6 7 8
The nodes are:
1 2 3 4 5 6 7 8 0 0 3 15 36
The nodes and the codes are:
1:110110
2:110111
3:1100
4:1110
5:1111
6:00
7:01
8:10
0:110100
0:110101
2. 
输入：
13
1 2 3 4 5 6 7 8 9 10 11 12 13
输出：
Please input the weights:
1 2 3 4 5 6 7 8 9 10 11 12 13
The nodes are:
1 2 3 4 5 6 7 8 9 10 11 12 13 10 26 40 91
The nodes and the codes are:
1:111000
2:111001
3:111010
4:111011
5:1000
6:1001
7:1010
8:1011
9:1100
10:1101
11:1111
12:00
13:01

## 结果分析
设n为叶子节点数
1. 在构造四元哈夫曼树时主要的时间耗费在寻找最小的节点上，时间复杂度为O(n)。
2. 构造新的节点、利用回溯算法构造哈夫曼编码的时间都不超过线性复杂度。
3. 在检查是否构造全部完成时，检查判断的时间复杂度O(n)。
4. 综上所述，整个算法的时间复杂度为O(n^2)。
 
 
 
 
 
 
 