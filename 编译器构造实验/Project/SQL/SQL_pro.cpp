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

// 每张表的某个列是否存在
 std::unordered_map<std::string, std::vector<int>> col_exist = {
        {"Student", {1, 0, 0, 0, 0, 1, 1}},
        {"Course", {0, 1, 0, 0, 1, 0, 0}},
        {"Teacher", {0, 0, 1, 0, 0, 1, 1}},
        {"Grade", {1, 1, 0, 1, 0, 0, 0}},
        {"Teach", {0, 1, 1, 0, 0, 0, 0}}
    };

// 每张表存在列的索引
std::unordered_map<std::string, std::vector<int>> col_id = {
        {"Student", {0, 5, 6}},
        {"Course", {1, 4}},
        {"Teacher", {2, 5, 6}},
        {"Grade", {0, 1, 3}},
        {"Teach", {1, 2}}
};

// 只针对第1张表的条件限制
vector<vector<string>>conditions_table1;
// 只针对第2张表的条件限制
vector<vector<string>>conditions_table2;
// 针对2张表比较的的条件限制
vector<pair<int,string>>conditions_table_mutual;
// 查询语句中查询的列信息 Col[i]代表  查询的第i个列是第Col[i][0] + 1表 的第Col[i][1]+1列
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

// 按给定的分隔符划分(但不会划分引号里的分隔符)
vector<string> splitClause(const string& str, const string& delimiter) {
    vector<string> result;
    bool inQuotes = false;
    size_t start = 0;
    size_t delimiterLength = delimiter.length();

    for (size_t i = 0; i < str.length(); ++i) {
        if (str[i] == '\"') {
            inQuotes = !inQuotes; 
        }
       
        if (!inQuotes && i + delimiterLength <= str.length() && str.substr(i, delimiterLength) == delimiter) {
            string subStr = str.substr(start, i - start);
            if (subStr.front() != '\"' || subStr.back() != '\"') {
                subStr = removeSpaces(subStr); 
            }
            result.push_back(subStr);
            start = i + delimiterLength;
            i += delimiterLength - 1;
        }
    }
    
    string lastSubStr = str.substr(start);
    if (lastSubStr.front() != '\"' || lastSubStr.back() != '\"') {
        lastSubStr = removeSpaces(lastSubStr); 
    }
    if(lastSubStr.size()>0)//避免空串
    result.push_back(lastSubStr);

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
                    for (auto&i:col_id[table[0]]) {
                           output.push_back(it[i]);
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

// 判断函数
int judge(const string&a,const string &b,const string &z ){
    if(z=="="){
        return a==b;
    }
    else if(z==">")return stoll(a) > stoll(b);
    else return stoll(a) < stoll(b);
}

// 利用数据库查询优化和数据局部性对2张表查询的处理
void process_queries_for_two_table(const vector<string>& table, const vector<string>& attribute, const vector<string>& where,const vector<vector<string>>&table1,const vector<vector<string>>table2) {
    string c1,z;
    //c1两张表中比较的列  z判别符
    if(conditions_table_mutual.size()>0){
    c1 = conditions_table_mutual[0].second;
    z = conditions_table_mutual[2].second;}
    vector<pair<int,string>>t1; // 筛选后的表1记录的索引和对应的列
    vector<pair<int,string>>t2; // 筛选后的表2记录的索引和对应的列
    // 对表1进行筛选
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
        if(flag){
            t1.push_back(make_pair(i,table1[i][col_to_index[c1]]));
        }
    }
    // 对表2进行筛选
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
         if(flag){
            t2.push_back(make_pair(i,table2[i][col_to_index[c1]]));
        }
    }

    // 利用数据局部性优化(直接用表记录的两重循环,大量数据换出换入缓存导致miss带来的延迟太大,故用较小数据结构来实现)
    vector<pair<int,int>>tt;  // 条件匹配的两个记录在两张表中的索引
    for(auto&i1:t1){
        int i = i1.first;
        string x = i1.second; 
        for(auto&i2:t2){
             int j = i2.first;
             string y = i2.second; 
             if(!conditions_table_mutual.size()||judge(x,y,z))tt.push_back(make_pair(i,j));
        }
    }    

    // 输出查询结果
    for(auto&x:tt){
        int i = x.first,j = x.second;
        if (attribute[0] == "*") { // 输出查询表中的所有列
         vector<string>output;
        for (auto&i1:col_id[table[0]]) {
                output.push_back(table1[i][i1]);
        }
        for (auto&i2:col_id[table[1]]) {
                output.push_back(table2[j][i2]);
        }
        output_strings(output);
        cout << endl;
    } else { // 输出查询表中的查询对应的列
        vector<string>output;
        for(auto&it:Col){
            int x = it[0],y = it[1];
            string temp = x == 0?table1[i][y]:table2[j][y];
            output.push_back(temp);
        }
        output_strings(output);
        cout << endl;
    }
    }
    
}

// 处理查询信息
void process_queries(const vector<string>& table, const vector<string>& attribute, const vector<string>& where) {
    if (table.size() == 1) { // 查询一张表
       process_queries_for_one_table(table,attribute,where);
    } 
    else { // 查询两张表
        process_queries_for_two_table(table,attribute,where,TABLE[table[0]],TABLE[table[1]]);
    }
}

// 对针对两张表的查询的条件与查询的列预处理
void preprocess_queries_for_two_table(const vector<string>& table, const vector<string>& attribute, const vector<string>& where){
    // 对针对两张表的查询的条件预处理
    for(auto&w:where){
        // 分割出条件中的左值x 右值y 和 分割符z 
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

        // 对右值为字符串的情况处理
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
        // 对右值是数字的情况处理
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
        // 对左右值都是表的列处理
        else{
            int t1,t2;
            string c1,c2;
            // 处理左值
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

            // 处理右值
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

    // 对查询的列预处理
    if(attribute[0]!="*"){
        for (int i = 0; i < attribute.size(); i++) {
                string a = attribute[i];
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

            int a = s.find("SELECT");// attribute_str是SELECT到FROM之间的子字符串,即查询的列；
            int t = s.find("FROM");// table_str是FROM到WHERE之间的子字符串,即查询的表;
            int w = s.find("WHERE") == -1 ? s.size() + 1 : s.find("WHERE");//  where_str是WHERE之后的子字符串(若无WHERE,则该where\_str为"")，即查询的条件
            attribute_str = s.substr(a + 7, t - a - 8);
            table_str = s.substr(t + 5, w - t - 5);
            where_str = (w == s.size() + 1 ? "" : s.substr(w + 6));

            // 分割成集合
            vector<string> table =  splitClause(table_str, ","); // 查询的表集合
            vector<string> attribute =  splitClause(attribute_str, ","); // 查询的列集合
            vector<string> where = splitClause(where_str, "AND"); // 查询的条件集合

            // 将上一轮查询的数据清除
            conditions_table1.clear();
            conditions_table2.clear();
            conditions_table_mutual.clear();
            Col.clear();

            // 对针对两张表的查询的条件与查询的列预处理
            if(table.size()==2){
                preprocess_queries_for_two_table(table,attribute,where);
            }

            // 处理查询并输出查询结果
            process_queries(table, attribute, where);
            break;
        }
    }
    return 0;
}
