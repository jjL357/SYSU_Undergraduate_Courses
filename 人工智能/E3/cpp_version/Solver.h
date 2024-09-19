#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <iostream>
#include <algorithm>
using namespace std;

#ifndef SOLVER_H
#define SOLVER_H


/* helper function */
bool isCapital(char c){
    return c>='A'&&c<='Z';
}

/*本题适用,这几个函数用之前必须确认是term*/
bool isConst(string s){
    return s.length()>1;
}

bool isFunc(string s){
    return s[s.length()-1] == ')';
}

bool isVar(string s){
    return s.size() == 1;
}

/* 本题函数内只有变量 */
bool isfwithV(string var,string func){
    for(int i=0;i<func.size();i++){
        if(func[i]==var[0]) return true;
    }
    return false;
}

/* 原子数据结构 */
class Atom{
public:
    string Predicate; // 谓词和取反
    vector<string> Term; // 项的列表
    Atom(string s){
    int p = 0, q = 0, end = s.size() - 1;
        if(s[p]=='~'){
            Predicate += s[0];
            p++;
        }
        while(s[p]!='('){
            Predicate += s[p];
            p++;
        }
        p++;

        if(s[end]==','){
            end -= 2;
        }else{
            end -= 1;
        }
        //  现在只剩 term 和 "," 以 ","分隔即可
        q = p + 1;
        while(q<=end){
            if(s[q]==','){
                string str_term;
                for(int k = p;k<q;k++){
                    str_term += s[k];
                }
                Term.push_back(str_term);
                p = q + 1;
                q += 2;
            }else{
                q++;
            }           
        }
        string str_term;//最后一个term
        for(int k = p;k<q;k++){
            str_term += s[k];
        }
        Term.push_back(str_term);
    }
    string getPredName(){
        if(Predicate[0]=='~'){
            string ret;
            for(int i=1;i<Predicate.size();i++) ret += Predicate[i];
            return ret;
        }else{
            return Predicate;
        }
    }

    /*是否取反*/
    bool isNeg(){
        return Predicate[0] == '~';
    }
};

// 直接连带前num_input条都先输出了
vector<vector<Atom> > readFromFile(const std::string& filePath){
    vector<string> strings;
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filePath << endl;       
    }
    int n;
    file >> n;
    file.ignore();

    string line;
    for (int i = 0; i < n; ++i) {
        getline(file, line);  // 读取每行字符串
        strings.push_back(line);
        cout<<line<<endl;//直接输出了
    }
    file.close();

    /* 将子句string处理为vector<Atom> */
    vector<vector<Atom> > ClauseSet;
    for(int i=0;i<n;i++){
        vector<Atom> Clause;
        string line = strings[i];
        string origin;//去空格和头尾括号
        int p = 0,q = 0,end = line.size() - 1;
        if(line[p]=='('){
            p++;
            end--;
        }
        while(p<=end){
            if(line[p]==' '){
                p++;
                continue;
            }
            origin += line[p];
            p++;
        }

        p = 0,q = 1;//双指针,分别表示子句头尾
        int len_pred = 1;//用于处理两位谓词如：非p
        if(origin[p]=='~'){
            len_pred = 2;
        }
        end = origin.size()-1;
        while(q<=end){//遇到大写字母或¬就停下,将可能含都好的串传给Atom类
            if(isCapital(origin[q])||origin[q]=='~'){
                if(len_pred==1){
                    string str_atom;
                    for(int k=p;k<q;k++){
                        str_atom += origin[k];
                    }
                    //cout<<"str_atom is:"<<str_atom<<endl; 
                    Atom atom(str_atom);
                    Clause.push_back(atom);

                    p = q;
                    if(origin[p]=='~'){
                        len_pred = 2;
                    }
                }else if(len_pred==2){
                    len_pred--;
                }
            }
            q++;
        }
        string str_atom;//最后结尾处一条
        for(int k=p;k<q;k++){
            str_atom += origin[k];
        }
        //cout<<"str_atom is:"<<str_atom<<endl; 
        Atom atom(str_atom);
        Clause.push_back(atom);

        ClauseSet.push_back(Clause);
    }   
    return ClauseSet;
}

// 输出函数
void printSolution(vector<string> answer){
    for(int i=0;i<answer.size();i++){
        cout<<answer[i]<<endl;
    }
}

int test_counter = 0;

void test_print_clause(vector<Atom> Clause){
    for(int i=0;i<Clause.size();i++){
        cout<<Clause[i].Predicate<<'(';
        for(int j=0;j<Clause[i].Term.size();j++){
            cout<<Clause[i].Term[j]<<' ';
        }
        cout<<") ";
    }
    cout<<endl;
}

//记录归结信息
struct Prt{
    int idxC1;// 子句编号
    int idxP1;// 第几个谓词的归结
    int idxC2;
    int idxP2;
    vector<string> old_name;
    vector<string> new_name;
};

class Solver{
private:
    int ori_size;
    vector<vector<Atom> > ClauseSet;//子句集
    vector<Prt> parent;//关系集
    vector<string> solve_ret;//答案
    vector<int> ans_idx;//相关子句在关系集和子句集的下标
    map<int,int> map;//相关子句的新下标
    int num_input;// 输入数量
    /* 获取新子句,若合成空子句则返回true ; 为了回溯方便还要记录一下*/
    bool Resolution(vector<Atom> C1,vector<Atom> C2,int pos1,int pos2){
        for(int i=0;i<C1.size();i++){
            for(int j=0;j<C2.size();j++){
                Atom a1 = C1[i];//抽两个原子
                Atom a2 = C2[j];
                if((a1.getPredName()==a2.getPredName())&&(a1.isNeg()!=a2.isNeg())){//谓词满足要求
                    vector<string> old_name;
                    vector<string> new_name;
                    //if(pos1==4&&pos2==8) cout<<"here"<<endl;
                    if(mgu(a1,a2,old_name,new_name)){//找到替换的了或者本来就相等
                        vector<Atom> ret = merge(C1,C2,old_name,new_name,i,j);

                        // if(test_counter++<100){
                        //     test_print_clause(ret);
                        // }
                        // if(pos1==4&&pos2==8){
                        //     test_print_clause(ret);
                        // }

                        Prt prt;
                        prt.idxC1=pos1;prt.idxC2=pos2;prt.idxP1=i;prt.idxP2=j;
                        prt.old_name=old_name;prt.new_name=new_name;
                        parent.push_back(prt);//parent和ClauseSet的同下标是对应的

                        //cout<<"merge for a time"<<endl;
                        // if(ret[0].Predicate=="~Green"&&ret[0].Term[0]=="bb"){
                        //     cout<<"~G:"<<ClauseSet.size()<<endl;
                        // }
                        // if(ret[0].Predicate=="Green"&&ret[0].Term[0]=="bb"){
                        //     cout<<"G:"<<ClauseSet.size()<<endl;
                        // }
                        // if(pos1==7&&pos2==9){
                        //     cout<<"hi"<<endl;
                        // }

                        if(ret.size()==0){
                            //cout<<"find an answer"<<endl;
                            return true;//一对子句只有一处可以合成吧，找到就直接合
                        }
                        else return false;
                    }
                }
            }
        }
        return false;
    }
    /* 能否找到合一的赋值,若找到则将赋值放在两个vec中 */
    bool mgu(Atom a1,Atom a2,vector<string>& old_name,vector<string>& new_name){
        /* 逐一比较每一个项能否替换，每换一个查看是否相等，换要整个换，vec也得换 */
        if(atom_term_eq(a1,a2)) return true;
        for(int i=0;i<a1.Term.size();i++){
            string oldt;
            string newt;
            if(mgu_helper(a1.Term[i],a2.Term[i],oldt,newt)){//找到直接两个原子内的项和替换表都开换
                for(int j=0;j<a1.Term.size();j++){
                    substitute(a1.Term[i],oldt,newt);
                    substitute(a2.Term[i],oldt,newt);
                }
                for(int j=0;j<new_name.size();j++){//因为old都是变量一般变的是new
                    substitute(new_name[j],oldt,newt);
                }
                if(oldt.size()){
                    old_name.push_back(oldt);
                    new_name.push_back(newt);
                }
            }else{
                return false;
            }
            if(atom_term_eq(a1,a2))return true;
        }
        return false;
    }
    bool atom_term_eq(Atom a1,Atom a2){
        for(int i=0;i<a1.Term.size();i++){
            if(a1.Term[i]!=a2.Term[i]) return false;
        }
        return true;
    }
    /* 两个项能否替换，若能换是哪个,递归实现 */
    bool mgu_helper(string term1,string term2,string& oldt,string& newt){
        if(term1==term2) return true;
        if(isVar(term1)){// 先看e1能否换e2，若不行再看e2能否换e1
            if(isFunc(term2)&&isfwithV(term1,term2)){
                return false;
            }
            oldt = term1;
            newt = term2;
            return true;
        }else if(isFunc(term1)){
            if(isFunc(term2)){
                string inner_t1;
                for(int i=2;i<term1.size()-1;i++){
                    inner_t1 += term1[i];
                }
                string inner_t2;
                for(int i=2;i<term2.size()-1;i++){
                    inner_t2 += term2[i];
                }
                return mgu_helper(inner_t1,inner_t2,oldt,newt);
            }
            //考虑e2换e1
            if(isVar(term2)){
                oldt = term2;
                newt = term1;
                return true;
            }
        }else if(isVar(term2)){
            oldt = term2;
            newt = term1;
            return true;
        }else{
            return false;
        }
    }
    /* 替换子句中的所有出现，消去相反，合成新子句，去重，放入子句集中,并返回*/
    vector<Atom> merge(vector<Atom> C1,vector<Atom> C2,vector<string> old_name,vector<string> new_name,int pos1,int pos2){
        vector<Atom> retC;
        for(int i=0;i<C1.size();i++){//原子
            for(int j=0;j<C1[i].Term.size();j++){//项
                for(int k=0;k<old_name.size();k++){
                    substitute(C1[i].Term[j],old_name[k],new_name[k]);
                }
            }
            if(i!=pos1) retC.push_back(C1[i]);//去反
        }
        for(int i=0;i<C2.size();i++){//原子
            for(int j=0;j<C2[i].Term.size();j++){//项
                for(int k=0;k<old_name.size();k++){
                    substitute(C2[i].Term[j],old_name[k],new_name[k]);
                }
            }
            /* 去重 */
            bool flag = true;
            for(int k=0;k<retC.size();k++){
                if((C1[k].getPredName()==C2[i].getPredName())&&(C1[k].isNeg()==C2[i].isNeg())&&atom_term_eq(C1[k],C2[i])){
                    flag = false;
                }
            }
            if(flag&&(pos2!=i)) retC.push_back(C2[i]);//去反
        }
        ClauseSet.push_back(retC);
        return retC;
    }
    /* 换一个term,因为有函数所以是递归实现 */
    void substitute(string& term,string oldt,string newt){
        if(isVar(term)){
            if(term==oldt){
                term = newt;
            }
        }else if(isFunc(term)){
            string tmp;
            string term_cp = term;
            for(int i=2;i<term.size()-1;i++){
                tmp += term[i];
            }
            term = tmp;//去掉一层函数
            substitute(term,oldt,newt);
            tmp = "";//回溯，把那层函数加回去
            tmp += term_cp[0];tmp += "(";
            tmp += term; tmp += ")";
            term = tmp;
        }else{
            return;
        }
    }
    /*从parent中获取相关答案的下标，放在ans_idx,搜索回溯*/
    void getAnswer(){
        int idx = ClauseSet.size() - 1;
        ans_helper(idx);//先筛出相关子句
        // 排序，然后重新映射
        sort(ans_idx.begin(),ans_idx.end());
        ans_idx.erase(unique(ans_idx.begin(), ans_idx.end()), ans_idx.end());
        for(int i=0;i<ans_idx.size();i++){
            map[ans_idx[i]] = num_input + i;
        }
        //合成出输出,然后塞入solve_ret中     
        for(int i=0;i<ans_idx.size();i++){
            string str = merge_str(ans_idx[i]);
            solve_ret.push_back(str);
        }   
    }
    /* 递归获取ans */
    void ans_helper(int idx){
        if(idx>=num_input){
            ans_idx.push_back(idx);
        }else{
            return;
        }
        Prt prt = parent[idx];
        ans_helper(prt.idxC1);
        ans_helper(prt.idxC2);
    }
    /* R[_a,_b]{x=_,y=_}(P(term,f(g(x))),Q(aaa)) */
    string merge_str(int idx){
        string str;
        str += "R[";
        Prt prt = parent[idx];
        if(prt.idxC1<num_input){
            str += to_string(prt.idxC1 + 1);
        }else{
            str += to_string(map[prt.idxC1]+1);
        }
        str += (char)(prt.idxP1 + 'a');
        str += ',';

        if(prt.idxC2<num_input){
            str += to_string(prt.idxC2 + 1);
        }else{
            str += to_string(map[prt.idxC2]+1);
        }
        str += (char)(prt.idxP2 + 'a'); 
        str += "]";
        if(prt.old_name.size()){
            str += "{";
            for(int i=0;i<prt.old_name.size()-1;i++){
                str += prt.old_name[i];
                str += '=';
                str += prt.new_name[i];
                str += ',';
            }
            str += prt.old_name[prt.old_name.size()-1];
            str += '=';
            str += prt.new_name[prt.old_name.size()-1];
            str += '}';
        }
        str += " = ";
        vector<Atom> Clause = ClauseSet[idx];
        bool add_suf = false;
        if(Clause.size()>1){
            add_suf = true;
            str += '(';
        }
        if(!Clause.size()){
            return str + "[]";
        }
        for(int i=0;i<(int)(Clause.size()-1);i++){// 除最后一个以外的原子
            str += Clause[i].Predicate;
            str += '(';
            for(int j=0;j<(int)(Clause[i].Term.size()-1);j++){
                str += Clause[i].Term[j];
                str += ',';
            }
            str += Clause[i].Term[Clause[i].Term.size()-1];//最后一项
            str += "),";
        }
        str += Clause[Clause.size()-1].Predicate;//最后一个原子
        str += '(';
        for(int j=0;j<Clause[Clause.size()-1].Term.size()-1;j++){
            str += Clause[Clause.size()-1].Term[j];
            str += ',';
        }
        str += Clause[Clause.size()-1].Term[Clause[Clause.size()-1].Term.size()-1];
        str += ')';

        if(add_suf){
            str += ')';
        }
        return str;
    }
public:
    Solver(vector<vector<Atom> > ClauseSet){
        this->ClauseSet = ClauseSet;
        num_input = ClauseSet.size();
    }
    ~Solver(){}
    int getNumInput(){
        return num_input;
    }
    /* 回传待输出答案 */
    vector<string> solve(){
        for(int i=0;i<ClauseSet.size();i++){
            Prt prt;
            parent.push_back(prt);//这部分不用，只是用来占初始的位置
        }
        int ptr = ClauseSet.size();
        while(1){
            for(int i=0;i<ptr;i++){
                for(int j=i+1;j<ptr;j++){//j 是不是从i+1开始
                    if(i!=j){//不能是同一句
                        if(Resolution(ClauseSet[i],ClauseSet[j],i,j)){
                            //cout<<"find an answer!"<<endl;
                            getAnswer();
                            return solve_ret;
                        }
                    }
                }
            }
            ptr = ClauseSet.size();
        }
        // for(int i=0;i<ClauseSet.size();i++){
        //     for(int j=i;j<ClauseSet.size();j++){//j 是不是从i+1开始
        //         if(i!=j){//不能是同一句
        //             if(Resolution(ClauseSet[i],ClauseSet[j],i,j)){
        //                 //cout<<"find an answer!"<<endl;
        //                 getAnswer();
        //                 return solve_ret;
        //             }
        //         }
        //     }
        // }

        return solve_ret;
    }
};


#endif