#include <iostream>
#include <stack>
#include<vector>
#include <unordered_map>
using namespace std;


int main() {
    int m,n;
    while(1){
        cin>>n>>m;
        if(m==0&&n==0)break;
        vector<vector<int>>dfa(n,vector<int>(m,-1));
        unordered_map<int,int>ff;
        char c;
        int t=0;
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                cin>>t;
                dfa[i][j]=t; 
                
            }
        }
        int x;
        while(cin>>x){
            if(x==-1)break;
            ff[x]=1;
        }
        
        while(1){
            
            string s;
            cin>>s;
            
            if(s.size()==1&&s[0]=='#')break;
            int st=0;
            for(int i=0;i<s.size();i++){
                st = dfa[st][s[i]-'a'];
                if(st==-1)break;
            }
            if(ff.count(st))cout<<"YES"<<endl;
            else cout<<"NO"<<endl;

        }

    }
    return 0;
}
