# Lab Week04 Application of Stack
## 实验报告
Design a C program to implement
1. Evaluation of an arithmetic expression in postfix notation
2. Arithmetic expression translation:infix to postfix conversion
## 实验重点
1. 运用栈实现把中缀表达式转换为后缀表达式。
2. 再通过后缀表达式计算结果。
3. 要注意运算符的优先级。
4. 要对多位数和小数进行特殊处理。
## 实验设计
1. 先运用栈和函数比较运算符的优先级，把数字直接压入栈中，将符号放入另外一个栈中进行比较，再放入栈中，进行实现把中缀表达式转换为后缀表达式。
2. 在将后缀表达式通过出栈进行运算得到结果。
3. 注意在计算时要对多位数和小数进行特殊处理。

## 代码实现
```c
#include<stdio.h>
#include<malloc.h>
#include<string.h>
#include<stdlib.h>
#include <math.h>
#include<string.h>
#define size 100
#define true 1;
#define false 0;
//可以多位数、小数、括号等进行加减乘除的运输
typedef struct Stack{ 
	char *nums;
	char *top;
	int stacksize;
}stack;
void InitiStack(stack * s){//初始化栈 
	s->nums=(char*)malloc(sizeof(char)*size);
    if(!s->nums){
    	printf("Initialize unsuccessfully!\n");
    	return;
	}
	s->stacksize=size;
	s->top=s->nums;
}
char GetTop(stack *s){//返回栈顶元素 
	char temp=*(s->top--);
	s->top++;
	return temp;
}
void push(stack *s,char e){//压入栈顶 
	s->stacksize++;
	s->top++;
	*s->top=e;
}
int empty(stack* s){//判断是否栈为空 
	if(s->top==s->nums)return true;
    return false;
}
void pop(stack *s){//删除栈顶元素 
	--s->top;
	}
void destroy(stack *s){//释放栈 
	free(s->nums);
	s->top=NULL;
	s->stacksize=0;
}
int cmp(char c){//比较运算符、括号的优先级 
	if(c==' ')return 0;
	else if(c=='+'||c=='-')return 1;
	else if(c=='*'||c=='/')return 2;
	else return -1; 
	
}
void reversePolish(char c[],stack* s1,stack* s2,stack* s3){//将中缀表达式转换为后缀表达式 
	for(int i=0;i<strlen(c);i++){
		if('0'<=c[i]&&c[i]<='9'){
			push(s2,c[i]);
		}
		else if(c[i]=='-'&&empty(s2))push(s2,'-');
		else if(c[i]=='.')push(s2,c[i]); 
		else {
			if(c[i]=='('||c[i]=='['){
			push(s1,c[i]);
			}
			else if(c[i]==')'){
				while(1){
				if(GetTop(s1)=='('){
					pop(s1);
					break;
				}
				if(empty(s1))break;
				push(s2,' ');
				push(s2,GetTop(s1));
				pop(s1);	
				}
			}
			else if(c[i]==']'){
				while(1){
				if(GetTop(s1)=='['){
					pop(s1);
					break;
				}
				if(empty(s1))break;
				push(s2,' ');
				push(s2,GetTop(s1));
				pop(s1);	
				}
			}
		    else if(GetTop(s1)=='('||GetTop(s1)=='['){
		    push(s2,' ');
			push(s1,c[i]);
			}
			else if(cmp(c[i])<=cmp(GetTop(s1))||empty(s1)){
			while(cmp(c[i])<=cmp(GetTop(s1))){
				if(empty(s1))break;
				push(s2,' ');
				push(s2,GetTop(s1));
				pop(s1);
			}
			push(s2,' ');
			push(s1,c[i]);	
			}
			else {	
			push(s2,' ');
			push(s1,c[i]);
			}
		}
	}
	while(!empty(s1)){
		push(s2,' ');
		push(s2,GetTop(s1));
		pop(s1);
	}
	while(!empty(s2)){
		push(s1,GetTop(s2));
		push(s3,GetTop(s2));	
		pop(s2);
	}
}
double sol(stack *s){//计算结果 
	double nums[100];
	int flag=1;
	int sign=1;
	memset(nums,0,sizeof(nums));
	int k=0;
	while(!empty(s)){
	if(GetTop(s)>='0'&&GetTop(s)<='9'){
    if(flag==1)nums[k]=nums[k]*10+GetTop(s)-'0';
    else nums[k]+=(GetTop(s)-'0')*pow(10,--flag);
    pop(s);
    if(GetTop(s)==' '){
	nums[k]*=sign;
	sign=1;
	k++;
	flag=1;
	}
	}
	else if(GetTop(s)=='.'){
		pop(s);
		flag=0;
	}
	else if(GetTop(s)==' '){
	pop(s);
	}
	else {
	if(GetTop(s)=='*'){
	nums[k-2]*=nums[k-1];
	nums[k-1]=0;
	k--;
	pop(s); 
	}
	else if(GetTop(s)=='+'){
	nums[k-2]+=nums[k-1];
	nums[k-1]=0;
	k--;
	pop(s);
	}
	else if(GetTop(s)=='-'&&(k==0&&nums[0]==0)){
    sign=-1;
	pop(s);
	}
	else if(GetTop(s)=='-'){
	nums[k-2]-=nums[k-1];
	nums[k-1]=0;
	k--;
	pop(s);
	}
	else if(GetTop(s)=='/'){
	if(nums[k-1]==0){
		printf("Error\n");
		return 0;
	}
	nums[k-2]/=nums[k-1];
	nums[k-1]=0;
	k--;
	pop(s);
	}
	}
}
return nums[0];
}
int main(){
	char s[100];
	printf("Please input the infix notation:\n");
	scanf("%s",&s);
	printf("The postfix notation: \n");
	stack s1;
	stack s2;
	stack s3;
	InitiStack(&s1);
	InitiStack(&s2);
	InitiStack(&s3);	
	reversePolish(s,&s1,&s2,&s3);
	while(!empty(&s1)){
		printf("%c",GetTop(&s1));
		pop(&s1);
	}
	printf("\n");
	printf("The result: "); 
	printf("%f\n",sol(&s3));
	destroy(&s1);
	destroy(&s2);
	destroy(&s3);
	return 0;
}
```
### 代码分析
1. 对栈的定义、操作的实现
```c
typedef struct Stack{ 
	char *nums;
	char *top;
	int stacksize;
}stack;
void InitiStack(stack * s){//初始化栈 
	s->nums=(char*)malloc(sizeof(char)*size);
    if(!s->nums){
    	printf("Initialize unsuccessfully!\n");
    	return;
	}
	s->stacksize=size;
	s->top=s->nums;
}
char GetTop(stack *s){//返回栈顶元素 
	char temp=*(s->top--);
	s->top++;
	return temp;
}
void push(stack *s,char e){//压入栈顶 
	s->stacksize++;
	s->top++;
	*s->top=e;
}
int empty(stack* s){//判断是否栈为空 
	if(s->top==s->nums)return true;
    return false;
}
void pop(stack *s){//删除栈顶元素 
	--s->top;
	}
void destroy(stack *s){//释放栈 
	free(s->nums);
	s->top=NULL;
	s->stacksize=0;
}
```
2. 比较运算符、括号的优先级
```c
int cmp(char c){ 
	if(c==' ')return 0;
	else if(c=='+'||c=='-')return 1;
	else if(c=='*'||c=='/')return 2;
	else return -1; 
	
}
```
3. 将中缀表达式转换为后缀表达式
```c
void reversePolish(char c[],stack* s1,stack* s2,stack* s3){ 
	for(int i=0;i<strlen(c);i++){
		if('0'<=c[i]&&c[i]<='9'){
			push(s2,c[i]);
		}
		else if(c[i]=='-'&&empty(s2))push(s2,'-');
		else if(c[i]=='.')push(s2,c[i]); 
		else {
			if(c[i]=='('||c[i]=='['){
			push(s1,c[i]);
			}
			else if(c[i]==')'){
				while(1){
				if(GetTop(s1)=='('){
					pop(s1);
					break;
				}
				if(empty(s1))break;
				push(s2,' ');
				push(s2,GetTop(s1));
				pop(s1);	
				}
			}
			else if(c[i]==']'){
				while(1){
				if(GetTop(s1)=='['){
					pop(s1);
					break;
				}
				if(empty(s1))break;
				push(s2,' ');
				push(s2,GetTop(s1));
				pop(s1);	
				}
			}
		    else if(GetTop(s1)=='('||GetTop(s1)=='['){
		    push(s2,' ');
			push(s1,c[i]);
			}
			else if(cmp(c[i])<=cmp(GetTop(s1))||empty(s1)){
			while(cmp(c[i])<=cmp(GetTop(s1))){
				if(empty(s1))break;
				push(s2,' ');
				push(s2,GetTop(s1));
				pop(s1);
			}
			push(s2,' ');
			push(s1,c[i]);	
			}
			else {	
			push(s2,' ');
			push(s1,c[i]);
			}
		}
	}
	while(!empty(s1)){
		push(s2,' ');
		push(s2,GetTop(s1));
		pop(s1);
	}
	while(!empty(s2)){
		push(s1,GetTop(s2));
		push(s3,GetTop(s2));	
		pop(s2);
	}
}
```
4. 计算结果
```c
double sol(stack *s){ 
	double nums[100];
	int flag=1;
	int sign=1;
	memset(nums,0,sizeof(nums));
	int k=0;
	while(!empty(s)){
	if(GetTop(s)>='0'&&GetTop(s)<='9'){
    if(flag==1)nums[k]=nums[k]*10+GetTop(s)-'0';
    else nums[k]+=(GetTop(s)-'0')*pow(10,--flag);
    pop(s);
    if(GetTop(s)==' '){
	nums[k]*=sign;
	sign=1;
	k++;
	flag=1;
	}
	}
	else if(GetTop(s)=='.'){
		pop(s);
		flag=0;
	}
	else if(GetTop(s)==' '){
	pop(s);
	}
	else {
	if(GetTop(s)=='*'){
	nums[k-2]*=nums[k-1];
	nums[k-1]=0;
	k--;
	pop(s); 
	}
	else if(GetTop(s)=='+'){
	nums[k-2]+=nums[k-1];
	nums[k-1]=0;
	k--;
	pop(s);
	}
	else if(GetTop(s)=='-'&&(k==0&&nums[0]==0)){
    sign=-1;
	pop(s);
	}
	else if(GetTop(s)=='-'){
	nums[k-2]-=nums[k-1];
	nums[k-1]=0;
	k--;
	pop(s);
	}
	else if(GetTop(s)=='/'){
	if(nums[k-1]==0){
		printf("Error\n");
		return 0;
	}
	nums[k-2]/=nums[k-1];
	nums[k-1]=0;
	k--;
	pop(s);
	}
	}
}
return nums[0];
}
```
## 测试样例
1. 输入：1+1
   输出：
Please input the infix notation:
1+1
The postfix notation:
1 1 +
The result: 2.000000

2. 输入：100+100*(600-500)/5
   输出：
Please input the infix notation:
100+100*(600-500)/5
The postfix notation:
100 100 600 500 - * 5 / +
The result: 2100.000000
3. 输入：33.4765+98.08*[87-9*(757-6.8)]
输出：
Please input the infix notation:
33.4765+98.08*[87-9*(757-6.8)]
The postfix notation:
33.4765 98.08 87 9 757 6.8 - * - * +
The result: -653650.107500

## 结果分析（时间分析）
设n为输入的符号和数字的总个数在，入栈、出栈是在参数时间内完成的，对栈进行遍历时是线性时间复杂度，运用栈实现把中缀表达式转换为后缀表达式、再通过后缀表达式计算结果都是线性时间复杂度，则整个程序的时间复杂度为O(n)。
