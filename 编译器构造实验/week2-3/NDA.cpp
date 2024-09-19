#include <iostream>
#include <string>
#include <vector>
#include <stack>
#include <set>
#include <unordered_set>
#include <unordered_map>
using namespace std;
void process(vector<vector<vector<int>>>&step,string s,int ix,int iy){
    int x = 0;
    for(int i=0;i<s.size();i++){
        if('0'<=s[i]&&s[i]<='9'){
            x *= 10;
            x += s[i]-'0';
        }
        else if(s[i]==','||(s.size()>2&&s[i]=='}')){
            step[ix][iy].push_back(x);
            x=0;
        }
    }

}
void show(vector<vector<vector<int>>>&step){
    cout<<"--------------------------"<<endl;
    for(int i=0;i<step.size();i++){
        for(int j=0;j<step[i].size();j++){
            cout<<"{";
            for(int k=0;k<step[i][j].size();k++){
                cout<<" "<<step[i][j][k]<<" ";
            }
            cout<<"}"<<" ";
        }
        cout<<endl;
    }
     cout<<"--------------------------"<<endl;
}

void show_closure(vector<set<int>>&closure){
    cout<<"--------------------------"<<endl;
    for(int i=0;i<closure.size();i++){
        for(auto &state:closure[i]){
            cout<<" "<<state<<" ";
        }
        cout<<endl;
        }
        cout<<endl;
    
     cout<<"--------------------------"<<endl;
}
void ND(vector<vector<vector<int>>>&step,vector<set<int>>&closure){
    for(int i=0;i<step.size();i++){
       stack<int>s;
      
       for(int j=0;j<step[i][0].size();j++){
            s.push(step[i][0][j]);
            closure[i].insert(step[i][0][j]);
       }
       while(!s.empty()){
            int x = s.top();
            s.pop();
            
            for(int k=0;k<step[x][0].size();k++){
                if(closure[i].count(step[x][0][k]))continue;
                else{
                    closure[i].insert(step[x][0][k]);
                    s.push(step[x][0][k]);
                }
            }
       }
    }
}
int dfs(string s,vector<vector<vector<int>>>&step,vector<set<int>>&closure,unordered_map<int,int>&fs,int cur,int state){
    if(cur==s.size()){
        if(fs.count(state))return 1;
        for( auto&it:closure[state]){
            if(fs.count(it))return 1;
        }
        return 0;
    }
    int ans=0;
    vector<int>temp;
    temp.push_back(state);
    for(auto&it:closure[state]){
        temp.push_back(it);
    }
    for(int i=0;i<temp.size();i++){
        int x=temp[i];
        //if(step[x][s[cur]-'a'+1].size()<=0)continue;
        for(int j=0;j<step[x][s[cur]-'a'+1].size();j++){
            ans|=dfs(s,step,closure,fs,cur+1,step[x][s[cur]-'a'+1][j]);
            if(ans==1)return 1;
        }
    }
    return ans;

}
int valid(string s,vector<vector<vector<int>>>&step,vector<set<int>>&closure,unordered_map<int,int>&fs){
    return dfs(s,step,closure,fs,0,0);
}
// set<int> stage(string s,vector<vector<vector<int>>>&step,vector<set<int>>&closure,unordered_map<int,int>&fs){

// }
int valid2(string s,vector<vector<vector<int>>>&step,vector<set<int>>&closure,unordered_map<int,int>&fs){
    set<int>temp;
    temp = closure[0];
    temp.insert(0);
    for(int i=0;i<s.size();i++){
        set<int> t2;
        for(auto&it:temp){
            for(int j=0;j<step[it][s[i]-'a'+1].size();j++){
                t2.insert(step[it][s[i]-'a'+1][j]);
            }
        }
        set<int> t3;
        for(auto&it:t2){
            t3.insert(it);
            for(auto &tx:closure[it]){
                t3.insert(tx);
            }
        }
        temp = t3;
    }
    for(auto&it:temp){
        if(fs.count(it))return 1;
    }
    return 0;
}
int main(){
    while(1){
        int m,n;
        cin>>n>>m;
        if(n==0&&m==0)break;
        unordered_map<int,int>fs;
        vector<set<int>>closure(n);
        //cout<<closure.size();
        
        vector<vector<vector<int>>>step(n,vector<vector<int>>(m));
        //cout<<m<<n;
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                string s;
                cin>>s;
                //cout<<s<<endl;
                process(step,s,i,j);
                //cout<<i<<j<<endl;
            }
        }
        //show(step);
        //cout<<"ND";
        ND(step,closure);
        //cout<<"ND";
        //show_closure(closure);
        int x;
        while(cin>>x){
            if(x==-1)break;
            else fs[x] = 1;
        }
        string s;
        while(cin>>s){

            if(s=="#")break;
            if(valid2(s,step,closure,fs))cout<<"YES"<<endl;
            else cout<<"NO"<<endl;
        }
    }

}
/*
4 3
{} {0,1} {0}
{} {} {2}
{} {} {3}
{} {} {}
3 -1
aaabb
abbab
abbaaabb
abbb
#
5 3
{1,3} {} {}
{} {2} {}
{} {2} {}
{} {} {4}
{} {} {4}
2 4 -1
abab
aaa
b
bbaaaaa
#
0 0
*/

/*
5 3
{1,3} {} {}
{2} {2} {}
{4} {2} {}
{3} {} {4}
{2} {} {4}
*/