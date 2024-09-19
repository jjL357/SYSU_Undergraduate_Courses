#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<malloc.h>
#include <time.h>
#include<limits.h>
typedef struct Tree{
char*Hcode;
double weight;
int index;
struct Tree*left;
struct  Tree*right;
int used;
}Tree;
int findmin(Tree* w,int size,int*is){
    double min=2;
    int minid;
    for(int i=0;i<256;i++){
        if((!w[i].used)&&min>w[i].weight){
            min=w[i].weight;
            minid=i;
        }
    }
    w[minid].used=1;
    return minid;
}
void Huff(Tree*w,int cursize,int treesize,int *is){
while(cursize<treesize){
//printf("%d %d ",cursize,treesize);
int id1=findmin(w,cursize,is);
int id2=findmin(w,cursize,is);
w[cursize].weight=w[id1].weight+w[id2].weight;
//printf("%d   %d       %f:%c %f:%c\n",id1,id2,w[id1].weight,w[id1].index,w[id2].weight,w[id2].index);
w[cursize].left=&w[id1];
w[cursize].right=&w[id2];
w[cursize].used=0;
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
	//printf("%lf ddd%cddd",root->weight,root->index);
	if(!root->left&&!root->right)return ;
	printf("(");
	printHuff(root->left);
	printf(",");
	printHuff(root->right);
	printf(")");
	} 
void printCode(Tree*trees,double*store,int sum){
printf("The Huffman codes are:\n");
    printf("   *------------> Huffman Code Table <-----------*\n");
    printf("    ____________________________________________\n      charcater |  frequency  |     code\n");
    printf("    ____________|_____________|_________________\n");
	for(int i=0;i<sum;i++){
		printf("	%c	%f	%s\n",trees[i].index,trees[i].weight,trees[i].Hcode);
		}
}


int encode(Tree*h,char*c){
Tree*t=h;
for(int i=0;i<strlen(c);i++){
if(c[i]=='0'){
if(t->left)t=t->left;
else {

	return -1;
}
}
else {
if(t->right)t=t->right;
else {
	
	return -1;
}	
}
}
if(!t->left&&!t->right){
return t->index;
}
else{
	return -1;
}
}

int getFileSize(char * strFileName)   
{  
    FILE * fp = fopen(strFileName, "r");  
    fseek(fp, 0L, SEEK_END);  
    int size = ftell(fp);  
    fclose(fp);  
    return size;  
}  
double* read_count(FILE* file,long long* n,int*isused,int *sum)
{
    int* store=(int*)malloc(sizeof(int)*128);
    memset(store,0,sizeof(int)*128);
    double* frequency=(double*)malloc(sizeof(double)*128);
    memset(frequency,0,sizeof(int)*128);
    if(file==NULL)printf("error!\n");
    char c=fgetc(file);
    while(c!=EOF)
    {
        
        store[c]+=1.0;
        (*n)++;
        
        
        c=fgetc(file);
    }
    for(int i=0;i<128;i++)
    {
        frequency[i]=(double)store[i]/(*n);
        if(store[i]!=0){
		(*sum)++;
		isused[i]=0;
		}
    }
    return frequency;
}

int main()
{   
     
    int temp[128];
    int sum=0;
	int n=128;
    int size=n;
    int treesize=2*size-1;
    int*isused=(int*)malloc(sizeof(int)*treesize);
    for(int i=0;i<treesize;i++)isused[i]=-1;
    Tree*trees=(Tree*)malloc(sizeof(Tree)*treesize);
    
	
	
	//读文件，读取文件字符 
    FILE* file;
    long long* nn=(long long *)malloc(sizeof(long long));
    *nn=0;
    file=fopen("D:\\textyj\\Text.txt","r");
    double* store=read_count(file,nn,isused,&sum);//统计字符出现次数，计算概率 
    fclose(file);
    
    
    //哈夫曼编码 
    printf("\n");
    treesize=2*sum-1;
    int j=0;
    for(int i=1;i<=sum;i++,j++){
    trees[i-1].used=0;
    trees[i-1].Hcode=NULL;
    trees[i-1].right=NULL;
    trees[i-1].left=NULL;
    while(isused[j]==-1)j++;
    trees[i-1].weight=store[j];
    trees[i-1].index=j;
    temp[j]=i-1;
	}
	for(int i=sum;i<256;i++){
	trees[i].Hcode=NULL;
    trees[i].right=NULL;
    trees[i].left=NULL;	
    trees[i].used=-1;
	}
    int cursize=sum;
	Huff(trees,cursize,treesize,isused);
    char c[100];
    c[0]='\0';
    if(treesize)CreateCode(&trees[treesize-1],c,0);
    printCode(trees,store,sum);
	
	
	//压缩 
	FILE* fp;
	FILE*p;
    fp=fopen("D:\\textyj\\Text.txt","r");
    p=fopen("D:\\textyj\\Compress.txt","wb");
    int begintime = clock(); //计时开始
    char cc=fgetc(fp);
    char ch=0;
    int len=0;
    while(cc!=EOF){
    	for(int i=0;i<strlen(trees[temp[cc]].Hcode);i++){
    		
    		if(trees[temp[cc]].Hcode[i]=='1'){
    			ch=ch*2;
    			ch=ch+1;
			}
			else {
			ch=ch*2;
		}
			len++;
			if(len==8){
				len=0;
				fwrite(&ch, sizeof(char), 1, p);
				ch=0;
			}
			
		}
		cc=fgetc(fp);
	}
	
	int xx=0;
	while(len>0&&len+xx<8){
  	    xx++;
  	    ch=ch*2;
	    if(len+xx==8)fwrite(&ch, sizeof(char), 1, p);
	   
	}
	fclose(p);
	int oldFileSize=getFileSize("D:\\textyj\\Text.txt");
	int newFileSize=getFileSize("D:\\textyj\\Compress.txt");
	int endtime = clock(); //计时结束
	printf("压缩时间: %fs\n", (endtime - begintime) / ((double)CLOCKS_PER_SEC));
	printf("该文件压缩率为：%.2lf\n",(double)newFileSize/(double)oldFileSize);
	fclose(fp);
	fclose(p);
	
	
	//解压 
	char queue[128];
	int start=0,end=0;
	FILE*fp4=fopen("D:\\textyj\\Compress.txt","rb");
	FILE*fp3=fopen("D:\\textyj\\Decompress.txt","w+");
	cc=fgetc(fp4);
	int FileSize=getFileSize("D:\\textyj\\Compress.txt");
	int tt=0;
	int begintime2=clock(); 
    while(tt<FileSize){
    	int len1=0;
    	char ctemp[8];
    	tt++;
    	int ct=cc;
    	
    	for(int j=0;j<8;j++){
    		if(ct<0)ct+=256;
    		if(tt==FileSize&&j<8-len&&len!=0){
    		ctemp[7-j]='\0';
			}
			else{
            
    		if(ct%2==0)ctemp[7-j]='0';
    		else ctemp[7-j]='1';
			}
    		ct=ct/2;
		}
		
    	for(int i=0;i<8;i++){    	   
            queue[end++]=ctemp[i];
            queue[end]='\0';
    		if(encode(&trees[treesize-1],queue)!=-1){
    			char s=encode(&trees[treesize-1],queue);
    			fwrite(&s, sizeof(char), 1, fp3);
    			queue[0]='\0';
    			end=0;
			}
			
		}
		cc=fgetc(fp4);
	}
	int endtime2=clock();
	printf("解压时间: %fs\n\n", (endtime2 - begintime2) / ((double)CLOCKS_PER_SEC));
    return 0;
}
