# Lab Week03 Application of Lists

## 实验内容
Application of Lists
Let **n** be a positive integer with specified initial value **n**=30. Use an array of**n** units to store polynomials of one indeterminate. Design and implement a C program to
(1) Determine the data structure of the array
(2) Use a stack **S** in the array to store unused units. Initially all the units are unused and the 0 to (**n**-1) indexes spread randomly on the stack 
(3) Use two list **LP** and **LQ** in the array to store two polynomials P and Q of one indeterminate
(4)Show the map of the array
(5)Adding **P** and **Q**, restore the results as a list LPQ in the array, and release all the garbage units to the Stack **S**
(6)Show the map of the array again

## 实验目的
通过编写程序进行多项式相加，帮助我们更好地了解静态链表的分配空间、释放、连接、插入等操作。

### 代码实现

```C
#include<stdio.h>
#define size 31
//expn代表指数(exponetial),coe代表系数(coefficient) 
typedef struct static_List{
	int expn; 
	float coef;
}List;//一个单项式 
typedef struct ploy{
	List list;
	int next;
}Ploy;//一个单项式，有next指引后续 
int List_malloc(Ploy *node){
	int index=node[0].next;
	node[0].next=node[index].next;
	return index;
}//为多项式的每一项分配空间，node[0].next作为索引返回，next为0的那一项为多项式的最后一项 
void List_initialize(Ploy *node){
for(int i=0;i<size-1;i++)node[i].next=i+1;
node[size-1].next=0;
}
void List_free(Ploy *node,int index){
	if(index>=size)return;
	node[index].next=node[0].next;
	node[index].list.coef=0;
	node[index].list.expn=0;
	node[0].next=index;
}//将系数为0的多项式的数组返回申请的空间 
void sort(Ploy *node,int begin,int nodecur){//插入单项式并对多项式按指数进行升序排列 
	if(node[nodecur].list.coef==0){
	List_free(node,nodecur);
	return;
}//插入的多项式系数为0，释放此多项式 
	int pre=begin,cur=node[begin].next;
	if(!cur){
		node[begin].next=nodecur;
		node[nodecur].next=cur;
		return;
	}//插入的单项式为多项式的第一项 
	while(cur&&node[nodecur].list.expn>node[cur].list.expn){
		pre=cur;
		cur=node[cur].next;
	}//找到指数大于等于插入单项式的单项式或者此单项式为多项式指数最大的 
	if(!cur){
		node[pre].next=nodecur;
		node[nodecur].next=cur;
		return;
	}//单项式为多项式指数最大的
	else if(node[nodecur].list.expn==node[cur].list.expn){//指数相等 
		if(node[nodecur].list.coef==-node[cur].list.coef){
			node[pre].next=node[cur].next;
			List_free(node,cur);
			List_free(node,nodecur);
		}//相加系数为0 
		else {
			node[cur].list.coef+=node[nodecur].list.coef;
			List_free(node,nodecur);
		}
	}
	else {
		node[pre].next=nodecur;
		node[nodecur].next=cur;
	}//插入多项式的中间 
}
void print(Ploy *node,int cur){//打印多项式 
	cur=node[cur].next;
	if(!cur){
		printf("%d\n",0);
		return;
	}//多项式为0 
	int count=1;
	while(cur){
        	if(node[cur].list.coef<0){
        		if(node[cur].list.coef!=-1){
        	    if(node[cur].list.expn==0)printf("%.0f",node[cur].list.coef);//指数为0 
				else if(node[cur].list.expn<0)printf("%.0fx^(%d)",node[cur].list.coef,node[cur].list.expn);
			    else printf("%.0fx^%d",node[cur].list.coef,node[cur].list.expn);
				}
        		else if(node[cur].list.expn==0)printf("%.0f",node[cur].list.coef);//指数为0 
        		else if(node[cur].list.expn<0)printf("-x^(%d)",node[cur].list.expn);
        		else printf("-x^%d",node[cur].list.expn);
			}//单项式系数为负数 
			else if(count==1){
			if(node[cur].list.coef!=1){//系数为1时，不输出系数，除非指数为0 
			if(node[cur].list.expn==0)printf("%.0f",node[cur].list.coef);//指数为0 	
			else if(node[cur].list.expn<0)printf("%.0fx^(%d)",node[cur].list.coef,node[cur].list.expn);
			else printf("%.0fx^%d",node[cur].list.coef,node[cur].list.expn);
			}
			else if(node[cur].list.expn==0)printf("x^%d",node[cur].list.expn);//指数为0 
			else if(node[cur].list.expn<0)printf("x^(%d)",node[cur].list.expn);
        	else printf("%.0f",node[cur].list.coef);
			}//多项式的第一项且系数大于等于0 
			else{
			if(node[cur].list.coef!=1){//系数为1时，不输出系数，除非指数为0 
			if(node[cur].list.expn==0)printf("+%.0f",node[cur].list.coef);//指数为0
			else if(node[cur].list.expn<0)printf("+%.0fx^(%d)",node[cur].list.coef,node[cur].list.expn);
			else printf("+%.0fx^%d",node[cur].list.coef,node[cur].list.expn);
			}
			else if(node[cur].list.expn==0)printf("+%.0f",node[cur].list.coef);//指数为0
			else if(node[cur].list.expn<0)printf("+%.0fx^(%d)",node[cur].list.coef,node[cur].list.expn);
        	else printf("+x^%d",node[cur].list.expn);
			}
		count++;
		cur=node[cur].next;
}
	printf("\n");
}
void add(Ploy *P,Ploy *Q,Ploy *LPQ,int Pbegin,int Qbegin,int LPQbegin){//多项式相加（利用sort函数进行相加） 
int p=P[Pbegin].next;
while(p){
	int lpq=List_malloc(LPQ);
	int expn=P[p].list.expn;
	int coef=P[p].list.coef;
	LPQ[lpq].list.coef=coef;
	LPQ[lpq].list.expn=expn;
	sort(LPQ,LPQbegin,lpq);
	p=P[p].next;
}
int q=Q[Qbegin].next;
while(q){
	int lpq=List_malloc(LPQ);
	int expn=Q[q].list.expn;
	int coef=Q[q].list.coef;
	LPQ[lpq].list.coef=coef;
	LPQ[lpq].list.expn=expn;
	sort(LPQ,LPQbegin,lpq);
	q=Q[q].next;
}
}
int main(){
	Ploy P[size],Q[size],LPQ[size];//为P、Q、LPQ创立独立的多项式空间 ，为相加后不影响P、Q的多项式，创立LPQ 
	List_initialize(P);
	List_initialize(Q);
	List_initialize(LPQ);
	int Pbegin=List_malloc(P);
	int Qbegin=List_malloc(Q);
	int LPQbegin=List_malloc(LPQ);
	P[Pbegin].next=0;
	Q[Qbegin].next=0;
	LPQ[LPQbegin].next=0;
	int Pnums,Qnums;//Pnums、Qnums为输入P、Q多项式的项数
	printf("PLease input the size of polynomials P:\n");
	scanf("%d",&Pnums);
	printf("Please input the coefficient and exponent of P:\n");
	for(int i=1;i<=Pnums;i++){
		int Pcur=List_malloc(P);
		scanf("%f%d",&P[Pcur].list.coef,&P[Pcur].list.expn);
		sort(P,Pbegin,Pcur);//将多项式插入并且进行排序 
	}
    printf("PLease input the size of polynomials Q:\n");
	scanf("%d",&Qnums);
	printf("Please input the coefficient and exponent of Q:\n");
	for(int i=1;i<=Qnums;i++){
		int Qcur=List_malloc(Q);
		scanf("%f%d",&Q[Qcur].list.coef,&Q[Qcur].list.expn);
		sort(Q,Qbegin,Qcur);//将多项式插入并且进行排序 
	}
	printf("P=");
	print(P,Pbegin);
	printf("Q=");
	print(Q,Qbegin);
	add(P,Q,LPQ,Pbegin,Qbegin,LPQbegin);
	printf("LQP=L+Q\n") ;
	printf("LPQ=");
	print(LPQ,LPQbegin);
	return 0;
}
```
### 代码分析
1. 结构体static_List为单项式，结构体ploy也为单项式，但有next作为下一项的索引，方便将不同的项连接起来
```c
//expn代表指数(exponetial),coe代表系数(coefficient) 
typedef struct static_List{
	int expn; 
	float coef;
}List;//一个单项式 
typedef struct ploy{
	List list;
	int next;
}Ploy;//一个单项式，有next指引后续 
```
2. //为P、Q、LPQ创立独立的多项式空间 ，为相加后不影响P、Q的多项式，创立LPQ 
```c
Ploy P[size],Q[size],LPQ[size];
```
3. 从多项式空间为多项式的每一项分配空间，node[0].next作为索引返回，next为0的那一项为多项式的最后一项
```c
int List_malloc(Ploy *node){
	int index=node[0].next;
	node[0].next=node[index].next;
	return index;
}
```
4. 将多项式的空间初始化把连接起来，最后一项的next为0（当next为0时，此单项式为多项式的最后一项
```c
void List_initialize(Ploy *node){
for(int i=0;i<size-1;i++)node[i].next=i+1;
node[size-1].next=0;
}
```
5. 将不需要的多项的的ploy返回多项式空间（例如将系数为0的单项式返回到多项式空间）
```c
void List_free(Ploy *node,int index){
	if(index>=size)return;
	node[index].next=node[0].next;
	node[index].list.coef=0;
	node[index].list.expn=0;
	node[0].next=index;
}
```
6. 将单项式插入到多项式中，并同时进行顺序排列
```c
void sort(Ploy *node,int begin,int nodecur){
	if(node[nodecur].list.coef==0){
	List_free(node,nodecur);
	return;
} 
	int pre=begin,cur=node[begin].next;
	if(!cur){
		node[begin].next=nodecur;
		node[nodecur].next=cur;
		return;
	}
	while(cur&&node[nodecur].list.expn>node[cur].list.expn){
		pre=cur;
		cur=node[cur].next;
	}
	if(!cur){
		node[pre].next=nodecur;
		node[nodecur].next=cur;
		return;
	}
	else if(node[nodecur].list.expn==node[cur].list.expn){
		if(node[nodecur].list.coef==-node[cur].list.coef){
			node[pre].next=node[cur].next;
			List_free(node,cur);
			List_free(node,nodecur);
		}
		else {
			node[cur].list.coef+=node[nodecur].list.coef;
			List_free(node,nodecur);
		}
	}
	else {
		node[pre].next=nodecur;
		node[nodecur].next=cur;
	}
}
```
7. 将P、Q利用sort函数插入到LPQ中并进行排序并相加
```c
void add(Ploy *P,Ploy *Q,Ploy *LPQ,int Pbegin,int Qbegin,int LPQbegin){//多项式相加（利用sort函数进行相加） 
int p=P[Pbegin].next;
while(p){
	int lpq=List_malloc(LPQ);
	int expn=P[p].list.expn;
	int coef=P[p].list.coef;
	LPQ[lpq].list.coef=coef;
	LPQ[lpq].list.expn=expn;
	sort(LPQ,LPQbegin,lpq);
	p=P[p].next;
}
int q=Q[Qbegin].next;
while(q){
	int lpq=List_malloc(LPQ);
	int expn=Q[q].list.expn;
	int coef=Q[q].list.coef;
	LPQ[lpq].list.coef=coef;
	LPQ[lpq].list.expn=expn;
	sort(LPQ,LPQbegin,lpq);
	q=Q[q].next;
}
}
```
8. 打印多项式，注意特殊情况的输出（如：系数为1，指数不为0时不输出系数，指数为0时只输出系数等等）
```c
void print(Ploy *node,int cur){//打印多项式 
	cur=node[cur].next;
	if(!cur){
		printf("%d\n",0);
		return;
	}//多项式为0 
	int count=1;
	while(cur){
        	if(node[cur].list.coef<0){
        		if(node[cur].list.coef!=-1){
        	    if(node[cur].list.expn==0)printf("%.0f",node[cur].list.coef);//指数为0 
				else if(node[cur].list.expn<0)printf("%.0fx^(%d)",node[cur].list.coef,node[cur].list.expn);
			    else printf("%.0fx^%d",node[cur].list.coef,node[cur].list.expn);
				}
        		else if(node[cur].list.expn==0)printf("%.0f",node[cur].list.coef);//指数为0 
        		else if(node[cur].list.expn<0)printf("-x^(%d)",node[cur].list.expn);
        		else printf("-x^%d",node[cur].list.expn);
			}//单项式系数为负数 
			else if(count==1){
			if(node[cur].list.coef!=1){//系数为1时，不输出系数，除非指数为0 
			if(node[cur].list.expn==0)printf("%.0f",node[cur].list.coef);//指数为0 	
			else if(node[cur].list.expn<0)printf("%.0fx^(%d)",node[cur].list.coef,node[cur].list.expn);
			else printf("%.0fx^%d",node[cur].list.coef,node[cur].list.expn);
			}
			else if(node[cur].list.expn==0)printf("x^%d",node[cur].list.expn);//指数为0 
			else if(node[cur].list.expn<0)printf("x^(%d)",node[cur].list.expn);
        	else printf("%.0f",node[cur].list.coef);
			}//多项式的第一项且系数大于等于0 
			else{
			if(node[cur].list.coef!=1){//系数为1时，不输出系数，除非指数为0 
			if(node[cur].list.expn==0)printf("+%.0f",node[cur].list.coef);//指数为0
			else if(node[cur].list.expn<0)printf("+%.0fx^(%d)",node[cur].list.coef,node[cur].list.expn);
			else printf("+%.0fx^%d",node[cur].list.coef,node[cur].list.expn);
			}
			else if(node[cur].list.expn==0)printf("+%.0f",node[cur].list.coef);//指数为0
			else if(node[cur].list.expn<0)printf("+%.0fx^(%d)",node[cur].list.coef,node[cur].list.expn);
        	else printf("+x^%d",node[cur].list.expn);
			}
		count++;
		cur=node[cur].next;
}
	printf("\n");
}
```
## 实验思路
为P、Q、LPQ创建多项式空间，在多项式需要空间时利用List_malloc函数为多项式得到分配空间的索引，将单项式插入到多项式时利用sort函数进行插入并排序的操作，当单项式的系数为0时利用List_free函数将此单项式从多项式中删除，并把分配的空间返回多项式空间中，再用add函数将P和Q相加得到LPQ，最后利用print函数输出。
## 实验重点
1. 为了保证在P、Q相加后得到PLQ的同时P、Q的多项式不会受到改变，所以为P、Q、LPQ都创立单独的多项式空间，保证P、Q、LPQ的操作是相互独立的，不会受到影响。
2. 为了找出当前多项式的最后一项可以将多项式的最后一项的next赋值为0，由此遍历时可以从next为0得知此单项式为多项式的最后一项。
3. 在进行多项式相加时我们可以利用之前编写的sort的函数将单项式插入并拍序，但为了不影响P、Q、LPQ各自单独的空间我们将此单项式先赋值给LPQ的空间中再进行相加。
4. 在将LPQ打印输出时注意特殊情况的输出（如：系数为1，指数不为0时不输出系数，指数为0时只输出系数等等）
5. 当输入的系数为0时可以不将此单项式插入
## 实验分析
1. 为了保证在P、Q相加后得到PLQ的同时P、Q的多项式不会受到改变，利用为P、Q、LPQ申请独立的空间，在P、Q、LPQ需要空间存储多项式时再用List_malloc分配空间，整个程序的空间主要消耗在这里，所以整个程序的复杂度为O（n），List_malloc时间复杂度为O（1）。(n为申请空间多项式的项数)
2. 为了将每一项连接起来，在ploy中增加next，表示下一项的位置，如果这一项为0，则表示此项为多项式的最后一项，List_initialize的时间复杂度为O（n)。
3. 为了将系数为0的单项式从多项式中删除，避免占用多余的空间，利用List_free删除并返回。如果要对不用的空间进行清理时，只需要从Poly[0]到Poly[Poly[0].next]到最后一项next为0的空间进行清理删除List_free的时间复杂度为O（1），如果要清理，清理的时间复杂度为O（n）。
4. 利用sort进行插入排序时，因为插入时的多项式已排好序，所以sort的时间复杂度为O(n)。
5. 用add多项式进行相加时利用了sort函数，add时间复杂度为O（n）.
6. print函数将多项式遍历一遍输出，时间复杂度为O（n）。
7. 所以这个程序的时间复杂度为O(n),空间复杂度为O(n)。

 
 
 
 
 
 
 
 