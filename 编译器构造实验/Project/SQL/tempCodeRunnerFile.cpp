#include <iostream>
#include <string>
#include <vector> 
#include <unordered_map>
#include <unordered_set>
#include <sstream>

using namespace std;

// 表名及其列
/* 
Student(sid,dept,age)
Course(cid,name)
Teacher(tid,dept,age)
Grade(sid,cid,score)
Teach(cid,tid)
*/

// TABLE根据表名存储不同表的记录
unordered_map<string, vector<vector<string>>> TABLE;

// 列与记录索引的映射
unordered_map<string, int> col_to_index = {
    {"sid", 0}, {"cid", 1}, {"tid", 2}, {"score", 3},
    {"name", 4}, {"dept", 5}, {"age", 6}
};

 std::unordered_map<std::string, std::vector<int>> col_exist = {
        {"Student", {1, 0, 0, 0, 0, 1, 1}},
        {"Course", {0, 1, 0, 0, 1, 0, 0}},
        {"Teacher", {0, 0, 1, 0, 0, 1, 1}},
        {"Grade", {1, 1, 0, 1, 0, 0, 0}},
        {"Teach", {0, 1, 1, 0, 0, 0, 0}}
    };

std::unordered_map<std::string, std::vector<int>> col_id = {
        {"Student", {0, 5, 6}},
        {"Course", {1, 4}},
        {"Teacher", {2, 5, 6}},
        {"Grade", {0, 1, 3}},
        {"Teach", {1, 2}}
    };
vector<vector<string>>conditions_table1;
vector<vector<string>>conditions_table2;
vector<pair<int,string>>conditions_table_mutual;
vector<vector<int>>Col;

// 处理输入记录，存储到TABLE对应的表中
void process_input() {
    for (int i = 0; i < 5; i++) {
        int n;
        cin >> n;
        while (n > 0) {
            n--;
            string sid, age, dept, cid, tid, score, name;
            vector<string> temp(7, "");
            switch (i) {
                case 0:
                    cin >> sid >> dept >> age;
                    temp[0] = sid; temp[5] = dept; temp[6] = age;
                    TABLE["Student"].push_back(temp);
                    break;
                case 1:
                    cin >> cid >> name;
                    temp[1] = cid; temp[4] = name;
                    TABLE["Course"].push_back(temp);
                    break;
                case 2:
                    cin >> tid >> dept >> age;
                    temp[2] = tid; temp[5] = dept; temp[6] = age;
                    TABLE["Teacher"].push_back(temp);
                    break;
                case 3:
                    cin >> sid >> cid >> score;
                    temp[0] = sid; temp[1] = cid; temp[3] = score;
                    TABLE["Grade"].push_back(temp);
                    break;
                default:
                    cin >> cid >> tid;
                    temp[1] = cid; temp[2] = tid;
                    TABLE["Teach"].push_back(temp);
                    break;
            }
        }
    }
}

// 移除字符串中的所有空格
string removeSpaces(const string& str) {
    string result;
    for (char ch : str) {
        if (!isspace(static_cast<unsigned char>(ch))) { // 使用isspace检查是否为空格
            result += ch;
        }
    }
    return result;
}

vector<string> splitClause(const string& str, const string& delimiter) {
    vector<string> result;
    bool inQuotes = false;
    size_t start = 0;
    size_t delimiterLength = delimiter.length();

    for (size_t i = 0; i < str.length(); ++i) {
        if (str[i] == '\"') {
            inQuotes = !inQuotes;  // Toggle the inQuotes flag
        }
        // Check for delimiter outside of quotes
        if (!inQuotes && i + delimiterLength <= str.length() && str.substr(i, delimiterLength) == delimiter) {
            string subStr = str.substr(start, i - start);
            if (subStr.front() != '\"' || subStr.back() != '\"') {
                subStr = removeSpaces(subStr);  // Remove spaces if not within quotes
            }
            result.push_back(subStr);
            start = i + delimiterLength;
            i += delimiterLength - 1; // Move index to the end of delimiter
        }
    }
    // Add the last part
    string lastSubStr = str.substr(start);
    if (lastSubStr.front() != '\"' || lastSubStr.back() != '\"') {
        lastSubStr = removeSpaces(lastSubStr);  // Remove spaces if not within quotes
    }
    if(lastSubStr.size()>0)result.push_back(lastSubStr);

    return result;
}

// 分割字符串的函数，使用整个字符串作为分隔符(string)，同时去除无用的空格
vector<string>  splitString(const string& str, const string& delimiters) {
    vector<string> result;
    size_t pos = 0;
    size_t delimiterPos;
    // 只要还有分隔符，就继续分割
    while ((delimiterPos = str.find(delimiters, pos)) != string::npos) {
        // 如果分隔符前面有文本，添加到结果中
        if (delimiterPos != pos) {
            result.push_back(removeSpaces(str.substr(pos, delimiterPos - pos)));
        }
        // 更新位置，跳过分隔符
        pos = delimiterPos + delimiters.length();
    } 
    // 添加剩余的文本
    if (pos != str.length()) {
        result.push_back(removeSpaces(str.substr(pos)));
    }
    return result;
}

// 输出字符串集合
void output_strings(const vector<string>& output){
    for(int i = 0;i < output.size() ;i++){
        cout<<output[i];
        if(i != output.size())cout<<" "; 
    }
}
// 对查询一张表的条件的左右值进行处理
// w : 条件字符串 str: 判别符
vector<string> process_condition_for_one_table(const vector<string>& col,string w, string str) {
    vector<string> t;
    t =  splitClause(w, str);
    string x = t[0], y = t[1];
    // 针对一张表的查询 查询内容中可去除表名
    vector<string> xt =  splitClause(x, ".");
    vector<string> yt =  splitClause(y, ".");
    string xx = xt.size() == 1 ? xt[0] : xt[1];
    string yy = yt.size() == 1 ? yt[0] : yt[1];

    if(col_to_index.count(xx)){
        xx = col[col_to_index[xx]];
    }
    // 处理引号
    else if (xx[0] == '\"' && xx.back() == '\"') {
        xx = xx.substr(1, xx.size() - 2);
    }
    if(col_to_index.count(yy)){
        yy = col[col_to_index[yy]];
    }
    // 处理引号
    else if (yy[0] == '\"' && yy.back() == '\"') {
        yy = yy.substr(1, yy.size() - 2);
    }
    return {xx, yy};
}

// 对查询2张表的条件的左右值进行处理
// t:左右值 table:表名集合 it1: table1的记录 it2: table2的记录 
vector<string> process_condition_for_two_table(const vector<string>& t,const vector<string>& table,const vector<string>& it1,const vector<string>& it2) {
    string x = t[0], y = t[1];
    string xx, yy;
    vector<string> xt =  splitClause(x, ".");
    vector<string> yt =  splitClause(y, ".");
    if (xt.size() == 1) {
        if (!col_to_index.count(xt[0])) { 
            xx = xt[0];
            if (xx[0] == '\"' && xx.back() == '\"') xx = xx.substr(1, xx.size() - 2);
        } else xx = it1[col_to_index[xt[0]]] != "" ? it1[col_to_index[xt[0]]] : it2[col_to_index[xt[0]]];
    } else {
        if (xt[0] == table[0]) xx = it1[col_to_index[xt[1]]];
        else xx = it2[col_to_index[xt[1]]];
    }
    if (yt.size() == 1) {
        if (!col_to_index.count(yt[0])) {
            yy = yt[0];
            if (yy[0] == '\"' && yy.back() == '\"') yy = yy.substr(1, yy.size() - 2);
        } else yy = it1[col_to_index[yt[0]]] != "" ? it1[col_to_index[yt[0]]] : it2[col_to_index[yt[0]]];
    } else {
        if (yt[0] == table[0]) yy = it1[col_to_index[yt[1]]];
        else yy = it2[col_to_index[yt[1]]];
    }
    return {xx,yy};
}

// 处理对1张表的查询
void process_queries_for_one_table(const vector<string>& table, const vector<string>& attribute, const vector<string>& where) {
    for (auto& it : TABLE[table[0]]) {
            int flag = 1; // 条件是否成立标志

            // 判断where之后的条件是否成立
            for (auto& w : where) {
                // 针对不同判别符处理
                if (w.find("=") != -1) {// =
                    vector<string> t = process_condition_for_one_table(it,w, "=");
                    string xx = t[0], yy = t[1];
                    if (xx != yy) {
                        flag = 0;
                        break;
                    }
                } else if (w.find(">") != -1) {// >
                    vector<string> t = process_condition_for_one_table(it,w, ">");
                    string xx = t[0], yy = t[1];
                    if (stoll(xx) <= stoll(yy)) {
                        flag = 0;
                        break;
                    }
                } else if (w.find("<") != -1) { // <
                    vector<string> t = process_condition_for_one_table(it,w, "<");
                    string xx = t[0], yy = t[1];
                    if (stoll(xx) >= stoll(yy)) {
                        flag = 0;
                        break;
                    }
                }
            }

            // 如果条件成立,输出查询结果
            if (flag) {
                if (attribute[0] == "*") { // 输出查询表的所有列
                    vector<string>output;
                    for (int i = 0; i < 7; i++) {
                        if (it[i] != "") {
                            output.push_back(it[i]);
                        }
                    }
                    output_strings(output);
                    cout << endl;
                } else { // 输出查询的列
                    vector<string>output;
                    for (int i = 0; i < attribute.size(); i++) {
                        // 由于是对一张表的查询,可以只关注.之后的列名
                        string a = attribute[i];
                        vector<string> t =  splitClause(a, ".");
                        string s;
                        if (t.size() == 1) s = t[0];
                        else s = t[1];
                        output.push_back(it[col_to_index[s]]);
                    }
                    output_strings(output);
                    cout << endl;
                }
            }
        }
}

// 处理对两张表的查询
void process_queries_for_two_table(const vector<string>& table, const vector<string>& attribute, const vector<string>& where) {
    for (auto& it1 : TABLE[table[0]]) {
        for (auto& it2 : TABLE[table[1]]) {
            int flag = 1; // 判断where之后的条件是否成立的标志
            // 判断where之后的条件是否成立
            for (auto& w : where) {
                vector<string> t;
                // 针对不同的判别符进行处理
                if (w.find("=") != -1) { // = 
                    t =  splitClause(w, "=");
                    vector<string> temp = process_condition_for_two_table(t,table,it1,it2);
                    string xx = temp[0],yy = temp[1];
                    if (xx != yy) {
                        flag = 0;
                        break;
                    }
                } else if (w.find(">") != -1) { // > 
                    t =  splitClause(w, ">");
                    vector<string> temp = process_condition_for_two_table(t,table,it1,it2);
                    string xx = temp[0],yy = temp[1];
                    if (stoll(xx) <= stoll(yy)) {
                        flag = 0;
                        break;
                    }
                } else if (w.find("<") != -1) {// < 
                    t =  splitClause(w, "<");
                    vector<string> temp = process_condition_for_two_table(t,table,it1,it2);
                    string xx = temp[0],yy = temp[1];
                    if (stoll(xx) >= stoll(yy)) {
                        flag = 0;
                        break;
                    }
                }
            }
            
            // 如果条件成立,输出查询结果
            if (flag) {
                vector<string>output;
                if (attribute[0] == "*") { // 输出查询表中的所有列
                    for (int i = 0; i < 7; i++) {
                        if (it1[i] != "") {
                           output.push_back(it1[i]);
                        }
                    }
                    for (int i = 0; i < 7; i++) {
                        if (it2[i] != "") {
                           output.push_back(it2[i]);
                        }
                    }
                    output_strings(output);
                    cout << endl;
                } else { // 输出查询表中的查询对应的列
                    vector<string>output;
                    for (int i = 0; i < attribute.size(); i++) {
                        string a = attribute[i];
                        // 判断 TABLE.COLUMN ,输出对应TABLE的COLUMN
                        vector<string> t =  splitClause(a, ".");
                        if (t.size() == 1) {
                            string x = it1[col_to_index[t[0]]] == "" ?  it2[col_to_index[t[0]]] : it1[col_to_index[t[0]]];
                            output.push_back(x);

                        } else {
                            string x = t[0] == table[0] ? it1[col_to_index[t[1]]] : it2[col_to_index[t[1]]];
                            output.push_back(x);
                        }
                    }
                    output_strings(output);
                    cout << endl;
                }
            }
        }
    }
}

int judge(const string&a,const string &b,const string &z ){
    if(z=="="){
        return a==b;
    }
    else if(z==">")return stoll(a) > stoll(b);
    else return stoll(a) < stoll(b);
}

// 处理对两张表的查询
void process_queries_for_two_table_2(const vector<string>& table, const vector<string>& attribute, const vector<string>& where,const vector<vector<string>>&table1,const vector<vector<string>>table2) {
    vector<int>index1,index2;
    
     for(int i =0 ;i<table2.size();i++){
        int flag = 1;
        for(auto&it:conditions_table2){
            string a ,b,c;
            a = it[0],b = it[1],c = it[2];
            if(!judge(table2[i][col_to_index[a]],b,c)){
                flag = 0;
                break;
            }
        }
        if(flag)index2.push_back(i);
    }

    for(int i =0 ;i<table1.size();i++){
        int flag = 1;
        for(auto&it:conditions_table1){
            string a ,b,c;
            a = it[0],b = it[1],c = it[2];
            if(!judge(table1[i][col_to_index[a]],b,c)){
                flag = 0;
                break;
            }
        }
        if(flag)index1.push_back(i);
    }
    //cout<<conditions_table1.size()<<conditions_table2.size
    for(auto&i1:index1){
        vector<string>it1 = table1[i1];
        for(auto&i2:index2){
            vector<string>it2 = table2[i2];
            int flag = 1;

            
            if(conditions_table_mutual.size()>0){
            int t1,t2;
            string c1,c2,z;
            t1 = conditions_table_mutual[0].first,c1 = conditions_table_mutual[0].second;
            t2 = conditions_table_mutual[1].first,c2 = conditions_table_mutual[1].second;
            
            z = conditions_table_mutual[2].second;
            //cout<<t1<<" "<<t2<<" "<<c1<<" "<<c2<<" "<<z<<endl;
            c1 = t1==0 ? it1[col_to_index[c1]] : it2[col_to_index[c1]];
            c2 = t2==0 ? it1[col_to_index[c2]] : it2[col_to_index[c2]];
            if(!judge(c1,c2,z))flag = 0;
            }
            if(flag){
                vector<string>output;
                if (attribute[0] == "*") { // 输出查询表中的所有列
                    for (auto&i:col_id[table[0]]) {
                           output.push_back(it1[i]);
                    }
                    for (auto&i:col_id[table[1]]) {
                           output.push_back(it2[i]);
                    }
                    output_strings(output);
                    cout << endl;
                } else { // 输出查询表中的查询对应的列
                    vector<string>output;
                    for(auto&it:Col){
                        int x = it[0],y = it[1];
                        string temp = x == 0?it1[y]:it2[y];
                        output.push_back(temp);
                    }
                    output_strings(output);
                    cout << endl;
                }
            }
        }
    }

}

// 处理查询信息
void process_queries(const vector<string>& table, const vector<string>& attribute, const vector<string>& where) {
    if (table.size() == 1) { // 查询一张表
       process_queries_for_one_table(table,attribute,where);
    } 
    else { // 查询两张表
        //process_queries_for_two_table_(table,attribute,where);
       process_queries_for_two_table_2(table,attribute,where,TABLE[table[0]],TABLE[table[1]]);
    }
}

int main() {
    // 处理输入记录
    process_input();
    int n; // 查询语句的个数
    cin >> n;
    cin.ignore(); // 处理输入n后的换行符
    // 处理查询
    while (n > 0) {
        n--;
        string s;
        // 读取每个查询
        while (getline(cin, s)) {
            // 对查询语句的不同部分进行分割
            string attribute_str;
            string table_str;
            string where_str;
            int a = s.find("SELECT");
            int t = s.find("FROM");
            int w = s.find("WHERE") == -1 ? s.size() + 1 : s.find("WHERE");
            attribute_str = s.substr(a + 7, t - a - 8);
            table_str = s.substr(t + 5, w - t - 5);
            where_str = (w == s.size() + 1 ? "" : s.substr(w + 6));

            vector<string> table =  splitClause(table_str, ","); // 查询的表集合
            vector<string> attribute =  splitClause(attribute_str, ","); // 查询的列集合
            vector<string> where = splitClause(where_str, "AND"); // 查询的条件集合

            conditions_table1.clear();
            conditions_table2.clear();
            conditions_table_mutual.clear();
            if(table.size()==2){
            for(auto&w:where){
                vector<string>t;
                string x,y,z;
                if (w.find("=") != -1) { // = 
                    t =  splitClause(w, "=");
                    z = "=";
                } else if (w.find(">") != -1) { // > 
                    t =  splitClause(w, ">");
                    z = ">";
                } else if (w.find("<") != -1) {// < 
                    t =  splitClause(w, "<");
                    z = "<";
                }
                x = t[0],y = t[1];
                if((y[0] == '\"'&&y.back()=='\"' )){
                    y = y.substr(1,y.size()-2);
                    vector<string>temp = splitClause(x,".");
                    if(temp.size()==1){
                        if(col_exist[table[0]][col_to_index[temp[0]]]){
                            conditions_table1.push_back({x,y,z});
                        }
                        else {
                            conditions_table2.push_back({x,y,z});
                        }
                    }
                    else {
                        if(table[0]==temp[0]){
                            conditions_table1.push_back({temp[1],y,z});
                        }
                        else {
                            conditions_table2.push_back({temp[1],y,z});
                        }
                    }
                }
                else if(!col_to_index.count(y)&&y.find(".")==-1){
                    vector<string>temp = splitClause(x,".");
                    if(temp.size()==1){
                        if(col_exist[table[0]][col_to_index[temp[0]]]){
                            conditions_table1.push_back({x,y,z});
                        }
                        else {
                            conditions_table2.push_back({x,y,z});
                        }
                    }
                    else {
                        if(table[0]==temp[0]){
                            conditions_table1.push_back({temp[1],y,z});
                        }
                        else {
                            conditions_table2.push_back({temp[1],y,z});
                        }
                    }
                }
                else{
                    int t1,t2;
                    string c1,c2;
                    vector<string>temp = splitClause(x,".");
                    
                    if(temp.size()==1){
                        if(col_exist[table[0]][col_to_index[temp[0]]]){
                            t1 = 0;
                            c1 = x;
                        }
                        else {
                            t1 = 1;
                            c1 = x;
                        }
                    }
                    else {
                        if(table[0]==temp[0]){
                            t1 = 0;
                            c1 = temp[1];
                        }
                        else {
                            t1 = 1;
                            c1 = temp[1];
                        }
                    }

                    temp = splitClause(y,".");
                    
                    if(temp.size()==1){
                        if(col_exist[table[0]][col_to_index[temp[0]]]){
                            t2 = 0;
                            c2 = y;
                        }
                        else {
                            t2 = 1;
                            c2 = y;
                        }
                    }
                    else {
                        if(table[0]==temp[0]){
                            t2 = 0;
                            c2 = temp[1];
                        }
                        else {
                            t2 = 1;
                            c2 = temp[1];
                        }
                    }
                conditions_table_mutual.push_back(make_pair(t1,c1));
                conditions_table_mutual.push_back(make_pair(t2,c2));
                conditions_table_mutual.push_back(make_pair(0,z));
                } 
            }
            Col.clear();
            if(attribute[0]!="*"){
                for (int i = 0; i < attribute.size(); i++) {
                        string a = attribute[i];
                        // 判断 TABLE.COLUMN ,输出对应TABLE的COLUMN
                        vector<string> t =  splitClause(a, ".");
                        int x,y;
                        if (t.size() == 1) {
                            y = col_to_index[t[0]];
                            x = col_exist[table[0]][y]?0:1;
                        } else {
                            y = col_to_index[t[1]];
                            x = t[0] == table[0]? 0:1; 
                        }
                        Col.push_back({x,y});
                    }
            }

            }
            //cout<<conditions_table1.size();
            process_queries(table, attribute, where);
            break;
        }
    }
    return 0;
}
