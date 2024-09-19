#include <iostream>
#include <stack>
#include<vector>
#include <unordered_map>
#include<queue>
using namespace std;

int dfs(int st,string s,vector<vector<vector<int>>>&dfa,unordered_map<int,int>&ff,int start,unordered_map<int ,int>&fff,vector<int>&xxx){
    if(fff.count(st))return 0;
    if(start==s.size()){
        if(ff.count(st)){
        // for(int a=0;a<xxx.size();a++)cout<<" "<<xxx[a];
        //  cout<<endl;
            return 1;
            }
        else {
            queue<int>q;
            unordered_map<int,int>m;
            m[st]=1;
            q.push(st);
            while(!q.empty()){
                int qq = q.front();
                q.pop();
                if(ff.count(qq))return 1;
                m[qq]=1;
                for(int i=0;i<dfa[qq][0].size();i++){
                    if(!m.count(dfa[qq][0][i]))q.push(dfa[qq][0][i]);
                }
                
            }
            return 0;
        }
    }

    unordered_map<int,int>ffff;
    int x=0;
    
    for(int i=0;i<dfa[st][0].size();i++){
        ffff = fff;
        ffff[st]=1;
        xxx.push_back(dfa[st][0][i]);
        x|=dfs(dfa[st][0][i],s,dfa,ff,start,ffff,xxx);
        xxx.pop_back();
    }
    if(x==1){
        return 1;
    }
    fff.clear();
    int t= dfa[st][s[start]-'a'+1].size();



    // if(t==0)return dfs(st,s,dfa,ff,start+1,fff);
    
    for(int i=0;i<t;i++){
        xxx.push_back(dfa[st][s[start]-'a'+1][i]);
        int xx=dfs(dfa[st][s[start]-'a'+1][i],s,dfa,ff,start+1,fff,xxx);
         xxx.pop_back();
        x |=xx;
        if(x==1)break;
    }


    return x;

}

int main() {
    int m,n;
    while(1){
        cin>>n>>m;
        if(m==0&&n==0)break;
        vector<vector<vector<int>>>dfa(n,vector<vector<int>>(m+1));
        unordered_map<int,int>ff;
        unordered_map<int,int>fff;
        char c;
        int t=0;
        for(int i=0;i<n;i++){
            
            for(int j=0;j<m;j++){
                string ss;
                cin>>ss;
                int xx=0;
                for(int k=0;k<ss.size();k++){
                    if(ss[k]==','||(ss[k]=='}'&&ss.size()>2)){
                        dfa[i][j].push_back(xx);
                        xx=0;
                    }
                    else if('0'<=ss[k]&&'9'>=ss[k]){
                        xx*=10;
                        xx+=ss[k]-'0';
                    }

                }
            }
        }
        // for(int i=0;i<m+1;i++){
        //     for(int j=0;j<dfa[0][i].size();j++){
        //         cout<<"ll:"<<dfa[0][i][j]<<" "<<endl;
        //     }
        // }
        
        
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
            vector<int>xxx;
            st = dfs(st,s,dfa,ff,0,fff,xxx);
            if(st==1)cout<<"YES"<<endl;
            else cout<<"NO"<<endl;

        }

    }
    return 0;
}
