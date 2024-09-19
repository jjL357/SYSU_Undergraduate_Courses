# Lab Week07

## 实验内容
Huffman Trees and Huffman Coding
(1) Determine the data structures for a binary Huffman tree
(2) Implement the algorithm of creating a binary Huffman tree for a 
given positive integer sequence
(3) Design and implement a Huffman coding prototype

## 实验重点以及设计思路
1. 先将输入的n个权重复制到节点中，并储存到数组中，每次找出其中权重最小的两个节点构成新的节点，直到一共有2*n-1个节点。
2. 为了避免选用重复的节点构建新的节点，可以用一个数组记录每个结点的使用情况进行去重。
3. 编码时节点为左子树编码后继续加一个'0'，右子树加'1'。
4. 最后对输入的编码进行解码，'0'指针向左走，'1'指针向右走，直到找的对应的节点，如果不存在，则返回"Not found!"。
   
## 代码实现

```C
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<limits.h>
#include<malloc.h>
typedef struct Tree{
char*Hcode;
int weight;
struct Tree*left;
struct  Tree*right;
}Tree;
int findmin(Tree* w,int size,int*is){
    int min=INT_MAX;
    int minid;
    for(int i=0;i<size;i++){
        if((!is[i])&&min>w[i].weight){
            min=w[i].weight;
            minid=i;
        }
    }
    is[minid]=1;
    return minid;
}

void Huff(Tree*w,int cursize,int treesize,int *is){
if(treesize==0){
    printf("No weight!\n");
    return ;
}
if(cursize==1)return ;
while(cursize<treesize){
int id1=findmin(w,cursize,is);
int id2=findmin(w,cursize,is);
w[cursize].weight=w[id1].weight+w[id2].weight;
w[cursize].left=&w[id1];
w[cursize].right=&w[id2];
cursize++;
}
}

void CreateCode(Tree*root,char*c,int k){
	if(!root)return ;
	int l=strlen(c);
	if(k!=0){
		if(k==1){
		c[l++]='0';
		}
		else c[l++]='1';
		c[l]='\0';
		root->Hcode=(char*)malloc(sizeof(char)*(l+2));
		for(int i=0;i<l;i++){
			root->Hcode[i]=c[i];
		}
		root->Hcode[l]='\0';
	}
	CreateCode(root->left,c,1);
	CreateCode(root->right,c,2);
	l=strlen(c);
	if(k!=0)c[l-1]='\0';
}
void printHuff(Tree*root){
	printf("%d",root->weight);
	if(!root->left&&!root->right)return ;
	printf("(");
	printHuff(root->left);
	printf(",");
	printHuff(root->right);
	printf(")");
	} 
void encode(Tree*t,char*c,int n){
	int k=0;
	for(int i=0;i<n;i++){
		if(!strcmp(c,t[i].Hcode)){
			k=1;
			printf("%s: %d\n",c,t[i].weight);
			break;
		}
	}
	if(!k)printf("Not found!\n");
}
void printCode(Tree*root){
if(!root)return;
if(!root->left&&!root->right)printf("%d:%s\n",root->weight,root->Hcode);
printCode(root->left);
printCode(root->right);
}
void encode1(Tree*h,char*c){
Tree*t=h;
for(int i=0;i<strlen(c);i++){
if(c[i]=='0'){
if(t->left)t=t->left;
else {
	printf("Not found!\n");
	return;
}
}
else{
if(t->right)t=t->right;
else {
	printf("Not found!\n");
	return;
}	
}
}
if(!t->left&&!t->right)printf("%s:%d\n",c,t->weight);
else{
	printf("Not found!\n");
	return;
}
}
int main(){
    int n;
    printf("Please input the number of weights:\n");
    scanf("%d",&n);
    if(n<=0){
    	printf("No weight!\n");
	}
	else{
    int size=n;
    int treesize=2*size-1;
    int*isused=(int*)malloc(sizeof(int)*treesize);
    for(int i=0;i<treesize;i++)isused[i]=0;
    Tree*trees=(Tree*)malloc(sizeof(Tree)*treesize);
    printf("Please input weights:\n");
    for(int i=1;i<=n;i++){
    int w;
    scanf("%d",&w);
    trees[i-1].Hcode=NULL;
    trees[i-1].right=NULL;
    trees[i-1].left=NULL;
    trees[i-1].weight=w;
    }
    int cursize=n;
    Huff(trees,cursize,treesize,isused);
    printf("The Huffman tree is:\n");
    printHuff(&trees[treesize-1]);
    printf("\n");
  
    char c[100];
    c[0]='\0';
    if(treesize)CreateCode(&trees[treesize-1],c,0);
    printf("The Huffman codes are:\n");
	if(treesize)printCode(&trees[treesize-1]);
	printf("Please input the number of codes which you want to encode:\n");
	int m;
	scanf("%d",&m);
	printf("Please input the code:\n");
	for(int i=1;i<=m;i++){
		char temp[100];
		scanf("%s",&temp);
		encode1(&trees[treesize-1],temp);
	}
	free(trees);
    free(isused);
    }
    return 0;
}
```
## 测试样例
1. **输入**：
0
**输出**：
Please input the number of weights:
0
No weight! 
2. **输入**：
4
7 5 4 2
0 10 110 111
**输出**：
Please input the number of weights:
4
Please input weights:
7 5 4 2
The Huffman tree is:
18(7,11(5,6(2,4)))
The Huffman codes are:
7:0
5:10
2:110
4:111
Please input the number of codes which you want to encode:
4
Please input the code:
0 10 110 111
0:7
10:5
110:2
111:4
3. **输入**：
10
1 1 1 1 2 3 3 4 6 15
0 1000 10010 10011 10100 10101 1011 110 1110 1111 00 11111 00000
**输出** ：
Please input the number of weights:
10 
Please input weights:
1 1 1 1 2 3 3 4 6 15
The Huffman tree is:
37(15,22(9(4(2,2(1,1)),5(2(1,1),3)),13(6,7(3,4))))
The Huffman codes are:
15:0
2:1000
1:10010
1:10011
1:10100
1:10101
3:1011
6:110
3:1110
4:1111
Please input the number of codes which you want to encode:
13
Please input the code:
0 1000 10010 10011 10100 10101 1011 110 1110 1111 00 11111 00000
0:15
1000:2
10010:1
10011:1
10100:1
10101:1
1011:3
110:6
1110:3
1111:4
Not found!
Not found!
Not found!   


## 结果分析
设输入的权重个数为n,树高为h=log n。
1. 构建哈夫曼树时寻找权重最小的两个节点时的时间复杂度为O(n),要创建n-1个新的节点，所以构建哈夫曼树时的时间复杂度为O(n^2)。
2. 在为节点编码时，运用了在父亲节点的基础上进行编码，所以编码时的时间复杂度为O(n*log n)。
3. 解码寻找对应的权重的时间复杂度为O(log n)。
4. 综上所述，整个程序的时间复杂度为O(n^2)。
 
 
 
 
 
 
 