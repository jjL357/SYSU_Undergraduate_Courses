#include <iostream>
#include<string>
#include<stdio.h>
using namespace std;
int main(){
    string s ;
    while(cin>>s){
        int x = 10;
        int sum = 0 ;
        for(int i = 0; i< s.size() ; i++){
            if('0'<=s[i] && '9'>=s[i]){
                sum += (s[i]-'0')*x;
                x--;
            }
        }
        int y = sum % 11;
        s += '-';
        if (y==1){
            s += 'X';
        }
        else if (y==0){
             s += '0';
        }
        else{
            s += '0' + 11-y;
        }
        cout<< s <<endl;
        // 检查输入流的状态，如果到达文件结束，则退出循环
        

    }
}

