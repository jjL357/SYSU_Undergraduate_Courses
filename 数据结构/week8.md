# Lab Week08

## 实验内容
Balance Rotations of AVL Trees
An AVL tree is a binary search tree where every node in the tree 
has a balance factor of -1, 0 or +1. But any insertion into the AVL 
tree may cause violations of balance in four cases: Left-Left, Left-Right, Right-Right, and Right-Left. Single Rotations and Double 
Rotations are introduced to fix these cases.
(1) Determine the data structures for an AVL tree
(2) Implement one and more insertions into the AVL tree
(3) Implement the four possible rotations to fix violations of balance 
caused by the insertions

## 实验重点以及设计思路
1. 先构建结构体，结构体中包括节点的值、父亲节点、左节点、右节点。
2. 插入时通过BST树的特性搜索查找到对应的插入位置。
3. 插入后通过对插入节点的父亲节点向上寻找是否存在非平衡的节点，若存在则进行调整，否则结束。
4. 调整时对非平衡节点的节点向上判断是哪种非平衡情况，根据不同的情况实现左旋、右旋、左右旋、右左旋等方式进行挑战。
5. 最后用前序、中序、后序遍历输出树，并释放空间。
   
## 代码实现
**（代码中带有注释）**
```c
#include<stdio.h>
#include<malloc.h>
#include<math.h>
#include<string.h>
typedef struct AvlTree{
int val;
struct AvlTree*left;
struct AvlTree*right;
struct AvlTree*parent;
}AvlTree;
int  height(AvlTree*root){
if(!root)return 0;
return 1+fmax(height(root->left),height(root->right));
}
int isbalance(AvlTree*root){//判断节点是否平衡 
    if(!root)return 1;
    if(fabs(height(root->right)-height(root->left))<2)return 1;
    else return 0;
}
AvlTree* CreateNode(){ //为新建节点开创空间 
	AvlTree*p=(AvlTree*)malloc(sizeof(AvlTree));
	return p;
}
AvlTree*insert(AvlTree*root,int x,AvlTree*par){//插入节点 
	if(root==NULL){
        root=CreateNode();
		root->val=x;
        root->left=NULL;
        root->right=NULL;
        root->parent=par;
        return root;
    }
    AvlTree*temp; 
    if(root->val<x)root->right=insert(root->right,x,root);
    else root->left=insert(root->left,x,root);
    return root;
}
void pre(AvlTree*root){//前序遍历 
    if(!root){
        printf("NULL ");
        return ;
    }
    printf("%d ",root->val);
    pre(root->left);
    pre(root->right);
}
void ino(AvlTree*root){//中序遍历 
    if(!root){
        printf("NULL ");
        return ;
    }
    ino(root->left);
    printf("%d ",root->val);
    ino(root->right);
}
void post(AvlTree*root){//后序遍历 
    if(!root){
        printf("NULL ");
        return ;
    }
    post(root->left);
    post(root->right);
    printf("%d ",root->val);
}
void printAvlTree(AvlTree*root){//输出树 
    printf("Preorder:\n");
    pre(root);
    printf("\n");
    printf("Inorder:\n");
    ino(root);
    printf("\n");
    printf("Postorder:\n");
    post(root);
    printf("\n");
    //printbyheight(root);
    //printf("\n");
}

AvlTree* find(AvlTree*root,int x){//寻找插入的节点 
     if(!root)return NULL;
	 if(root->val==x){
        return root;
		}
    if(root->val<x)return find(root->right,x);
    else return find(root->left,x);
}

AvlTree*left_left_Adjust(AvlTree*root,AvlTree*t){//左旋 
	int k=0;
	if(t==root){
		k=1;
	}
	AvlTree*head=t->parent;
	AvlTree*temp=t->left;
	t->left=t->left->right;
	if(temp->right)temp->right->parent=t;
	temp->right=t;
	t->parent=temp;
	if(head){
		temp->parent=head;
		if(head->left==t)head->left=temp;
		else head->right=temp;
	}
    else temp->parent=NULL;
	if(k)return t->parent; 
	else return NULL; 
}
AvlTree*right_right_Adjust(AvlTree*root,AvlTree*t){//右旋 
	int k=0;
	if(t==root){
		k=1;
	}
	AvlTree*head=t->parent;
	AvlTree*temp=t->right;
	t->right=t->right->left;
	if(temp->left)temp->left->parent=t;
	temp->left=t;
	t->parent=temp;
	if(head){
		temp->parent=head;
		if(head->left==t)head->left=temp;
		else head->right=temp;
	}
	else temp->parent=NULL;
	if(k)return t->parent; 
	else return NULL; 
}
AvlTree*left_right_Adjust(AvlTree*root,AvlTree*t){//左右旋 
	int k=0;
	if(t==root){
		k=1;
	}
	AvlTree*temp=right_right_Adjust(t,t->left);
	temp=left_left_Adjust(t,t);
	if(k)return temp; 
	else return NULL; 
}
AvlTree*right_left_Adjust(AvlTree*root,AvlTree*t){//右左旋转 
	int k=0;
	if(t==root){
		k=1;
	}
	AvlTree*temp=left_left_Adjust(t,t->right); 
	temp=right_right_Adjust(t,t);
	if(k)return temp; 
	else return NULL; 
}
AvlTree* AvlAdjust(AvlTree*root,AvlTree*t){////插入后调整树 
	AvlTree*temp=t;
	while(isbalance(t)&&t){
		if(t->parent==NULL)break;
		t=t->parent;
	}
	if(t==root&&isbalance(t)){
	printf("This insertation is fine.\n");
	return NULL;
	}
	int c=0;
	if(find(t->left,temp->val)&&find(t->left->left,temp->val))c=1;
	else if(find(t->right,temp->val)&&find(t->right->right,temp->val))c=2;
	else if(find(t->left,temp->val)&&find(t->left->right,temp->val))c=3;
	else if(find(t->right,temp->val)&&find(t->right->left,temp->val))c=4;
	printf("This insertation causes imbalancement.\n",c);
	if(c==1)printf("This insertation causes left-left imbalancement.\n");
	else if(c==2)printf("This insertation causes right-right imbalancement.\n");
	else if(c==3)printf("This insertation causes left-right imbalancement.\n");
	else if(c==4)printf("This insertation causes right-left imbalancement.\n");
	if(c==1)temp=left_left_Adjust(root,t);
	else if(c==2)temp=right_right_Adjust(root,t);
	else if(c==3)temp=left_right_Adjust(root,t);
	else if(c==4)temp=right_left_Adjust(root,t);
    if(temp!=NULL)root=temp;
	return root;	
}
void destroyAvl(AvlTree*root){//释放空间 
	if(!root)return ;
	destroyAvl(root->left);
	destroyAvl(root->right);
	free(root);
}
int main(){
    printf("Please input the number of nodes:\n");
    int n;
    scanf("%d",&n);
    printf("Please input the nodes:\n");
    AvlTree*root=NULL;
    for(int i=1;i<=n;i++){
    int x;
    AvlTree*t=NULL;
    scanf("%d",&x);
    printf("Insert %d\n",x);
    root=insert(root,x,NULL);
    t=find(root,x);
    AvlTree*temp;
    temp=AvlAdjust(root,t);
    if(temp!=NULL)root=temp;
    printAvlTree(root);
    }
    destroyAvl(root); 
    return 0;
}
```
## 测试样例
1. 输入：
5
1 2 3 4 5
输出：
Please input the number of nodes:
5
Please input the nodes:
1 2 3 4 5
Insert 1
This insertation is fine.
Preorder:
1 NULL NULL
Inorder:
NULL 1 NULL
Postorder:
NULL NULL 1 
Insert 2
This insertation is fine.
Preorder:
1 NULL 2 NULL NULL
Inorder:
NULL 1 NULL 2 NULL
Postorder:
NULL NULL NULL 2 1
Insert 3
This insertation causes imbalancement.
This insertation causes right-right imbalancement.
Preorder:
2 1 NULL NULL 3 NULL NULL
Inorder:
NULL 1 NULL 2 NULL 3 NULL 
Postorder:
NULL NULL 1 NULL NULL 3 2
Insert 4
This insertation is fine.
Preorder:
2 1 NULL NULL 3 NULL 4 NULL NULL 
Inorder:
NULL 1 NULL 2 NULL 3 NULL 4 NULL
Postorder:
NULL NULL 1 NULL NULL NULL 4 3 2 
Insert 5
This insertation causes imbalancement.
This insertation causes right-right imbalancement.
Preorder:
2 1 NULL NULL 4 3 NULL NULL 5 NULL NULL 
Inorder:
NULL 1 NULL 2 NULL 3 NULL 4 NULL 5 NULL
Postorder:
NULL NULL 1 NULL NULL 3 NULL NULL 5 4 2
2. 输入：
3 
1 2 3
输出：
Please input the number of nodes:
3
Please input the nodes:
1 2 3
Insert 1
This insertation is fine.
Preorder:
1 NULL NULL
Inorder:
NULL 1 NULL
Postorder:
NULL NULL 1
Insert 2
This insertation is fine.
Preorder:
1 NULL 2 NULL NULL
Inorder:
NULL 1 NULL 2 NULL
Postorder:
NULL NULL NULL 2 1
Insert 3
This insertation causes imbalancement.
This insertation causes right-right imbalancement.
Preorder:
2 1 NULL NULL 3 NULL NULL
Inorder:
NULL 1 NULL 2 NULL 3 NULL
Postorder:
NULL NULL 1 NULL NULL 3 2

## 结果分析
**设n为节点个数**
在AVL树在结点高度上采用相对平衡的策略，使其平均性能接近于BST的最好情况的性能。因此，对于包含n个结点的AVL树，其最坏情况下的查找、插入和判断哪一种非平衡情况的操作时间复杂度均为O(log n)。
而调整旋转的时间复杂度为O(1)。所以整个程序的时间复杂度为O(log n)。
 
 
 
 
 
 