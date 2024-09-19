#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>
using namespace std;
void print_ele(unordered_map<string,int>&mp){
    cout<<"---------------------"<<endl;
    for(auto&it:mp){
        cout<<it.first<<":   "<<it.second<<endl;
    }
    cout<<"---------------------"<<endl;
    cout<<endl;
}
unordered_map<string,int>process2(const string&substance){
    unordered_map<string,int>mp;
    for(int i = 0 ;i < substance.size() ;i++){
        if(substance[i] == '('){
            int pre = i;
            int t = 0;
            i++;
            while(substance[i]!=')'||t!=0){
                if(substance[i]=='(')t++;
                if(substance[i]==')')t--;
                i++;
            }
            int co = 0;
            string tmp = substance.substr(pre+1 , i-pre-1);
            //cout<<tmp<<"||||"<<endl;
            unordered_map<string,int>mp_tmp = process2(tmp);

            i++;
            while(i < substance.size() && substance[i] <='9'&&substance[i]>='0' ){
                co*=10;
                co += substance[i] -'0';
                i++;
            }
            i--;
            if(co==0){
                co=1;
            }
            for(auto&it:mp_tmp){
                string e = it.first;
                int c = it.second;
                if(mp.count(e)){
                    mp[e] += c*co;
                }
                else {
                    mp[e] = c*co;
                }
            }
        }
        else {
            int pre = i;
            while(i + 1 < substance.size() && 'a'<=substance[i+1]&&substance[i+1]<='z'){
                i++;
            }
            //cout<<pre<<i<<endl;
            string ele = substance.substr(pre,i + 1 -pre);
            int co = 0;
            i++;
            while(i < substance.size() && substance[i] <='9'&&substance[i]>='0'){
                co*=10;
                co += substance[i] -'0';
                i++;
            }
            i--;
            if(co==0){
                co=1;
            }
            if(mp.count(ele)){
                    mp[ele] += co;
            }
            else {
                    mp[ele] = co;
            }
       
        }
        
    }
    return mp; 
}
void process(const string&substance,unordered_map<string,int>&mp){
    //cout<<"Process1: "<<substance<<"| ";
    string substance2;
    int co = 1;
    if('1'<=substance[0]&&substance[0]<='9'){
        int i = 0;
        for(i = 0; i < substance.size() ;i++){
            if('0'<=substance[i]&&substance[i]<='9'){
                continue;
            }
            else break;
        }
        string num = substance.substr(0,i);
        substance2 = substance.substr(i,substance.size() - i);
        co = 0;
        for(int i = 0; i < num.size() ;i++){
            co *= 10;
            co += num[i] - '0';
        }
    }
    else{
        substance2 = substance;
    }
    //cout<<co<<" "<<substance2<<endl;
    unordered_map<string,int>mp_tmp = process2(substance2);
    for(auto&it:mp_tmp){
        string e = it.first;
        int c = it.second;
        if(mp.count(e)){
            mp[e] += c*co;
        }
        else {
            mp[e] = c*co;
        }
    }

}
int main(){
    int n;
    cin>>n;
    while(n>0){
        string equation;
        unordered_map<string,int>Left_element,Right_element;
        cin>>equation;
        int pre = 0 ,i = 0;
        for(i = 0; i < equation.size() ;i++){
            if(equation[i] == '='){
                process(equation.substr(pre,i - pre),Left_element);
                pre = i + 1;
                break;
            }
            if(equation[i] == '+'){
                process(equation.substr(pre,i - pre),Left_element);
                pre = i + 1;
            }
        }
        for(; i < equation.size() ;i++){
            if(i == equation.size() - 1){
                process(equation.substr(pre,i+1- pre),Right_element);
                pre = i + 1;
                break;
            }
            if(equation[i] == '+'){
                process(equation.substr(pre,i - pre),Right_element);
                pre = i + 1;
            }
        }
        int flag = 1;
        //print_ele(Left_element);
        //print_ele(Right_element);
        for(auto&it:Left_element){
            string element = it.first;
            int co = it.second;
            if(Right_element.count(element)){
                if(co == Right_element[element]){
                    Right_element.erase(element);
                }
                else{
                    flag = 0;
                    break;
                }
            }
            else {
                flag = 0;
                break;
            }
        }
        if(!Right_element.empty())flag = 0;
        if(flag)cout<<"Y"<<endl;
        else cout<<"N"<<endl;
        n--;
    }
    return 0;
}