# Lab Week02 Comparison of Different  Complexity Classes

## 实验内容
Comparisons of Different Complexity Classes

Let **n** bea positive integer with specified initial value. Design and implement a program to find the least number **N** such that g(**n**)>=f(**n**) when **n**>**N**.(Programming in C,you may include<math.h>)

(1)n=1,f(n)=n and g(n)=nln(n)
(2)n=1,f(n)=pow(n,10) and g(n)=pow(2,n)
(3)n=2,f(n)=pow(2,n) and g(n)=n!

### 代码实现

```C
#include<stdio.h>
#include<math.h>
int main(){
	//(1)n=1,f(n)=n and g(n)=nln(n),df1(n)=1,dg1=1+log(n)
	//设f1为函数f(n)的值,g1为函数g(n)的值,df1为f(n)的导数，dg1为g(n)的导数 
	int n1=1,f1,g1,df1,dg1;
	while(1){
		f1=n1;
		g1=n1*log(n1);
		df1=1;
		dg1=1+log(n1);
		if(g1>=f1&&dg1>=df1)break;
		n1++;
	}
	printf("%d\n",--n1);
	//(2)n=1,f(n)=pow(n,10) and g(n)=pow(2,n),df2==10*pow(n,9),dg2=pow(2,n)*log(2);
	int n2=1,f2,g2,df2,dg2;
	while(1){
		if(pow(n2,10)<=pow(2,n2)&&pow(2,n2)*log(2)>=10*pow(n2,9)){
		break;
	}
	n2++;
	}
	printf("%d\n",--n2);
	//(3)n=2,f(n)=pow(2,n) and g(n)=n!,df3=pow(2,n)*log(2)
	//设f3为函数f(n)的值,g3为函数g(n)的值 
	int n3=2,f3,g3=1;
	while(1){
		f3=pow(2,n3);
		g3*=n3;
		if(g3>=f3)break;
		n3++;
	}
	printf("%d\n",--n3);
	return 0;
}
```
#### 输出结果：

2

58

3
## 实验报告
### 实验内容及目的
#### 实验内容

运用C语言编写程序，在给定的特定初始值的正整数和不同代数复杂性类的f(**n**)和g(**n**),找到最小的整数**N**使得两个函数f(**n**）和g(**n**)在**n**>**N**的情况下g(**n**)>=f(**n**)。

#### 实验目的

1. 运用程序找到最小的正整数**N**，使得当**n**>**N**时g(**n**)>=f(**n**)。
2. 通过程序更好地了解不同的函数的增长速度,明白不同的算法的时间复杂度之间的差距。
3. 通过观察不同函数增长速度的差距，了解不同的时间复杂度的差异，引导我们选择更加快速高效的算法。
4. 注意题目要求n>N,程序中n1、n2、n3为满足n>=N，g(n)>=f(n)时的最小正整数N，所以最后的输出需要-1。

#### 实验重点

1. 要考虑不同函数的增长速度，不能仅仅考虑一个值附近的区间，如（2）中f(**n**)=pow(n,10),g(**n**)=pow(2,n),当**n**=1时给g(1)=2>f(1)=1,不代表**n**>=1后，g(**n**)>=f(**n**),如**n**=2时f(2)=1024>g(2)=4，因为在n=58前f(**n**)的增长速度大于g(**n**)的增长速度，所以在2<=2<59前f(**n**)>g(**n**)。
2. （2)中当g(**n**)>f(**n**)时，n=59,此时g(**n**)的大小已经超过了int数据类型的范围，所以不能将g(**n**)的值赋给int数据类型的变量。
3.  （3）中的g(**n**)=**n**!的导数不容易求的，所以实现对函数的增长速度进行分析得出g(**n**)**n**!的的增长速度很快，远快于f(**n**)，所以随着**n**的增大，g(**n**)超过f(**n**)。
4.  注意题目要求**n**>**N**,程序中**n**1、**n**2、**n**3为满足**n**>=**N**，g(**n**)>=f(**n**)时的最小正整数N，所以最后的输出需要-1。

#### 结果分析
1. (1)n=1,f(n)=n and g(n)=nln(n)
当**n**>**N**=2,g(**n**)>=f(**n**),g(**n**)的增长速度略大于f(**n**)，即O(**n**)<O(**nln(n)**)。
2. (2)n=1,f(n)=pow(n,10) and g(n)=pow(2,n)
当**n**>**N**=58,g(n)>=f(n),g(n)的增长速度大于f(n)，即O(pow(**n**,10))<O(pow(2,**n**))。   
3. (3)n=2,f(n)=pow(2,n) and g(n)=n!
当**n**>**N**=3,g(n)>=f(n),g(n)的增长速度远远大于f(n)，即O(pow(2,**n**))<O(**n**!)。
4. 通过程序分析得出O(**n**)<O(**n**ln**n**)<O(pow(**n**,10))<O(pow(2,**n**))<O(**n**!)。  

 
 
 
 
 
 
 
 