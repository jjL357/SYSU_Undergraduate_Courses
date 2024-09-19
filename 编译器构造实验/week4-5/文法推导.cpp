/*
E  -> TE'
E' -> +TE'|e
T  -> FT'
T' -> *FT'|e
F  -> (E)|id
*/
#include<iostream>
#include<vector>
using namespace std;
string process_nonterminal(string dedection, char terminal){
    if(dedection == "E")return "TE'";
    else if(dedection == "T")return "FT'";
    else if(dedection == "E'"){
        if(terminal == '+')return"+TE'"; 
        else return "";
    } 
    else if(dedection == "T'"){
        if(terminal== '*')return "*FT'";
        else return ""; 
    }
    else if(dedection == "F"){
        if(terminal == '(')return "(E)"; 
        else if(terminal <= '9' && terminal >= '0'){
            string temp = "";
            temp += terminal;
            return temp;
        }
    }
    return "Syntax Error";
}
int find_nonterminal(string deduction){
    for(int i =0 ; i< deduction.size() ;i++){
        if(deduction[i] == 'E' || deduction[i] == 'F' || deduction[i] == 'T'){
            return i;
        }
    }
    return -1;
}
int get_nonterminal_length(string deduction, int pos){
    if(pos == -1)return 0;
    else if(pos + 1 < deduction.size() && deduction[pos+1] == '\'')return 2;
    return 1;
}
int main(){
    string s;
    while(cin >> s && s != "#"){
        vector<string> deductions;
        deductions.push_back("E");
        int s_cur = 0;
        int d_cur = 0;
        int flag = 1;
        while(1){
            string deduction = deductions.back();
            int pos = find_nonterminal(deduction);
            int nonterminal_len = get_nonterminal_length(deduction,pos);
            string deduction_tmp = process_nonterminal(deduction.substr(pos,nonterminal_len),s[s_cur]);
            if(deduction_tmp == "Syntax Error")flag = 0;
            string right =  deduction.substr(0,pos);
            string left = deduction.substr(pos+nonterminal_len,deduction.size() - pos - nonterminal_len);
            string new_deduction = right + deduction_tmp + left;
            int next_pos = find_nonterminal(new_deduction);

            
            while(d_cur < next_pos && s_cur<s.size()){
                if(new_deduction[d_cur]!=s[s_cur]){
                    flag = false;
                    break;
                }
                d_cur++;
                s_cur++;
            }
            //cout <<pos<<" "<<nonterminal_len<<" "<<s_cur<<" "<<deduction_tmp <<" "<<new_deduction<<endl;
            deductions.push_back(new_deduction);
            if(s_cur == s.size()&&d_cur == new_deduction.size()){
                 for(auto &it: deductions){
                     cout<<it<<endl;
                 }
                cout<<endl;
                break;
            }
            if(d_cur == new_deduction.size()&&s_cur != s.size())flag=0;

            if(!flag){
                cout<<"Syntax Error"<<endl<<endl;
                break;
            }
            
            

        }
    }
    return 0;
}