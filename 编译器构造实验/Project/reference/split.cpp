#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <array>
#include <unordered_map>
using namespace std;

enum data_type {Int, Str, Undefined};
struct data_each {
    int num;
    string s;
    data_type type;
    data_each() : type(Undefined) {}
    data_each(int n) : num(n), type(Int) {}
    data_each(const string& str) : s(str), type(Str) {}
};
struct column_cmp_constant {
    data_each constant;
    vector<vector<int>> column;
    int cmp;
};
struct col_and_col {
    vector<vector<int>> col1;
    vector<vector<int>> col2;
};
struct table {
    string name;
    int col, row;
    int id;
    vector<string> col_name;
    vector<data_type> col_type;
    vector<vector<data_each>> dataRow;
    
    table(const string& table_name, int table_id, int col_num) 
        : name(table_name), col(col_num), row(0),id(table_id) {}
};
map<string,int> nameToId;
table Table[5] = {
    table("Student", 0, 3),
    table("Course", 1, 2),
    table("Teacher", 2, 3),
    table("Grade", 3, 3),
    table("Teach", 4, 2)
};

void InitTable() {
    Table[0].col_name = {"sid", "dept", "age"};
    Table[0].col_type = {Str, Str, Int};
    nameToId[Table[0].name] = Table[0].id;
    Table[1].col_name = {"cid", "name"};
    Table[1].col_type = {Str, Str};
    nameToId[Table[1].name] = Table[1].id;
    Table[2].col_name = {"tid", "dept", "age"};
    Table[2].col_type = {Str, Str, Int};
    nameToId[Table[2].name] = Table[2].id;
    Table[3].col_name = {"sid", "cid", "score"};
    Table[3].col_type = {Str, Str, Int};
    nameToId[Table[3].name] = Table[3].id;
    Table[4].col_name = {"cid", "tid"};
    Table[4].col_type = {Str, Str};
    nameToId[Table[4].name] = Table[4].id;
}

void read_data() {
    for (int k = 0; k < 5; k++) {
        cin >> Table[k].row;
        for (int i = 0; i < Table[k].row; i++) {
            vector<data_each> temp;
            for (int j = 0; j < Table[k].col; j++) {
                if (Table[k].col_type[j] == Int) {
                    int num;
                    cin >> num;
                    data_each t(num);
                    temp.push_back(t);
                } else {
                    string str;
                    cin >> str;
                    data_each s(str);
                    temp.push_back(s);
                }
            }
            Table[k].dataRow.push_back(temp);
        }
    }
}


bool is_digit(char c) {
    return isdigit(static_cast<unsigned char>(c));
}

bool is_char(char c) {
    return isalpha(static_cast<unsigned char>(c));
}



vector<string> word_split(char* str) {
    vector<string> sql;
    int i = 0;
    string tmp;
    bool flag = true; 
    while (true) {
        char current_char = str[i];
        if (current_char == '\0' || current_char == ' ') {
            if (!tmp.empty()) {
                sql.push_back(tmp);
                tmp.clear();
            }
        } else if (current_char == ',' || current_char == '.' || current_char == '<' || 
                   current_char == '>' || current_char == '=' || current_char == '*') {
            if (!tmp.empty()) {
                sql.push_back(tmp);
                tmp.clear();
            }
            tmp.push_back(current_char);
            sql.push_back(tmp);
            tmp.clear();
        } else if (current_char == '\"') {
            if (!tmp.empty()) {
                sql.push_back(tmp);
                tmp.clear();
            }
            tmp.push_back(current_char);
            if (flag) 
                tmp.push_back('0');
            else 
                tmp.push_back('1');
            flag = !flag;
            sql.push_back(tmp);
            tmp.clear();
        } else if (is_digit(current_char) || is_char(current_char)) {
            tmp.push_back(current_char);
        } 
        if (current_char == '\0') 
            break;
        else 
            i++;
    }
    return sql;
}

array<int, 3> getLoc(const vector<string>& sql) {
    array<int, 3> location = {0, 0, 0};
    for (int i = 0; i < static_cast<int>(sql.size()); i++) {
        if (sql[i] == "SELECT") location[0] = i;
        if (sql[i] == "FROM")   location[1] = i;
        if (sql[i] == "WHERE")  location[2] = i;
    }
    if (location[2] == 0) location[2] =  static_cast<int>(sql.size());
    return location;
}

array<int, 2> get_table(const vector<string>& sql, const array<int, 3>& loc) {
    array<int, 2> tableId = {-1, -1}; 

    if (loc[2] - loc[1] - 1 >= 2) {
        tableId[0] = nameToId[sql[loc[1] + 1]];
        tableId[1] = nameToId[sql[loc[1] + 3]];
    } else {
        tableId[0] = nameToId[sql[loc[1] + 1]];
        tableId[1] = -1;
    }
    return tableId;
}
vector<vector<int>> get_col(const vector<string>& sql, const array<int, 3>& loc, array<int, 2>& tableId){
    vector<vector<int>> matrix;
    vector<int> table;
    vector<int> col;
    if(tableId[1]==-1){
        for(int i = loc[0] + 1; i < loc[1]; i++){
            if(nameToId.count(sql[i]) == 1 && sql[i]!=","){
                table.push_back(tableId[0]);
                int col_loc;
                for(int j = 0; j < static_cast<int>(Table[tableId[0]].col_name.size());j++){
                    if(sql[i+2] == Table[tableId[0]].col_name[j]) {
                        col_loc=j;
                        break;
                    }
                }
                col.push_back(col_loc);
                i=i+2;
            }else if(sql[i] == "*"){
                for(int j = 0; j <  static_cast<int>(Table[tableId[0]].col_name.size());j++){
                    table.push_back(tableId[0]);
                    col.push_back(j);
                }
            }else if(sql[i]!=","){
                table.push_back(tableId[0]);
                int col_loc;
                for(int j = 0; j <  static_cast<int>(Table[tableId[0]].col_name.size());j++){
                    if(sql[i] == Table[tableId[0]].col_name[j]) {
                        col_loc=j;
                        break;
                    }
                }
                col.push_back(col_loc);
            }
        }
    }else{
        for(int i = loc[0] + 1; i < loc[1]; i++){
            if(nameToId.count(sql[i])==1 && sql[i]!=","){
                table.push_back(nameToId[sql[i]]);
                int col_loc;
                for(int j = 0; j <  static_cast<int>(Table[nameToId[sql[i]]].col_name.size());j++){
                    if(sql[i+2] == Table[nameToId[sql[i]]].col_name[j]) {
                        col_loc=j;
                        break;
                    }
                }
                col.push_back(col_loc);
                i=i+2;
            }else if(sql[i]=="*"){
                for(int j = 0; j <  static_cast<int>(Table[tableId[0]].col_name.size());j++){
                    table.push_back(tableId[0]);
                    col.push_back(j);
                }
                for(int j = 0; j <  static_cast<int>(Table[tableId[1]].col_name.size());j++){
                    table.push_back(tableId[1]);
                    col.push_back(j);
                }
            }else if(sql[i]!=","){
                int flag=0;
                for(int j = 0; j <  static_cast<int>(Table[tableId[0]].col_name.size());j++){
                    if(sql[i] == Table[tableId[0]].col_name[j]){
                        flag=1;
                        table.push_back(tableId[0]);
                        col.push_back(j);
                        break;
                    }
                }
                if(flag==0){
                    for(int j = 0; j <  static_cast<int>(Table[tableId[1]].col_name.size());j++){
                        if(sql[i] == Table[tableId[1]].col_name[j]){
                            table.push_back(tableId[1]);
                            col.push_back(j);
                            break;
                        }
                    }
                }
            }
        }
    }
    matrix.push_back(table);
    matrix.push_back(col);
    return matrix;
}

void get_after_where(const vector<string>& sql, const array<int, 3>& loc, array<int, 2>& tableId,
        vector<column_cmp_constant>& w1,vector<col_and_col> &w2){
    if(loc[2]== static_cast<int>(sql.size())){
        return;
    }   
    int left = loc[2];
    while(left< static_cast<int>(sql.size())){
        int middle = left + 1;
        while(middle <  static_cast<int>(sql.size()) && sql[middle]!="<" && sql[middle]!="=" &&sql[middle]!=">") middle++;
        int right = middle + 1;
        while(right <  static_cast<int>(sql.size()) && (sql[right]!="AND"||sql[right-1]=="\"0")) right++;
        if(sql[left+2] == "." && sql[right-2] == "."){
            col_and_col temp1;
            array<int,3> col_left = {left,middle,-1};
            temp1.col1=get_col(sql,col_left,tableId);
            array<int,3> col_right = {middle,right,-1};
            temp1.col2=get_col(sql,col_right,tableId);
            w2.push_back(temp1);  
        }else{
            column_cmp_constant temp2;
            array<int,3> col_left = {left,middle,-1};
            temp2.column = get_col(sql,col_left,tableId);
            if(sql[right-1] == "\"1"){
                temp2.constant.s=sql[right-2];
                temp2.constant.type=Str;
            }else{
                int num = 0;
                for (char ch : sql[right-1]) {
                    num = num * 10 + (ch - '0');
                }
                temp2.constant.num=num;
                temp2.constant.type=Int;
            }
            if(sql[middle]=="=") temp2.cmp = 0;
            else if(sql[middle]==">") temp2.cmp = 1;
            else if(sql[middle]=="<") temp2.cmp = -1;
            w1.push_back(temp2);
        }
        left=right;

    }  
}

vector<vector<int>> query(const vector<column_cmp_constant>& w1, vector<col_and_col>& w2, array<int, 2>& tableId) {
    vector<vector<int>> result;

    // 预处理 w1 条件，按表分类
    unordered_map<int, vector<column_cmp_constant>> tableConditions;
    for (const auto& cond : w1) {
        tableConditions[cond.column[0][0]].push_back(cond);
    }

    // 检查第一个表的条件
    vector<int> table1Matches;
    for (int i = 0; i < Table[tableId[0]].row; i++) {
        bool match = true;
        for (const auto& cond : tableConditions[tableId[0]]) {
            const auto& cell = Table[tableId[0]].dataRow[i][cond.column[1][0]];
            if (cond.constant.type == Int) {
                if ((cond.cmp == 0 && cell.num != cond.constant.num) ||
                    (cond.cmp == -1 && cell.num >= cond.constant.num) ||
                    (cond.cmp == 1 && cell.num <= cond.constant.num)) {
                    match = false;
                    break;
                }
            } else {
                if (cell.s != cond.constant.s) {
                    match = false;
                    break;
                }
            }
        }
        if (match) {
            table1Matches.push_back(i);
        }
    }
    
    if (table1Matches.empty()) return result;

    if (tableId[1] != -1) {
        // 检查第二个表的条件
        vector<int> table2Matches;
        for (int i = 0; i < Table[tableId[1]].row; i++) {
            bool match = true;
            for (const auto& cond : tableConditions[tableId[1]]) {
                const auto& cell = Table[tableId[1]].dataRow[i][cond.column[1][0]];
                if (cond.constant.type == Int) {
                    if ((cond.cmp == 0 && cell.num != cond.constant.num) ||
                        (cond.cmp == -1 && cell.num >= cond.constant.num) ||
                        (cond.cmp == 1 && cell.num <= cond.constant.num)) {
                        match = false;
                        break;
                    }
                } else {
                    if (cell.s != cond.constant.s) {
                        match = false;
                        break;
                    }
                }
            }
            if (match) {
                table2Matches.push_back(i);
            }
        }
        
        if (table2Matches.empty()) return result;

        // 处理表之间的连接条件
        vector<int> a1, a2;
        for (int i : table1Matches) {
            for (int j : table2Matches) {
                bool match = true;
                for (const auto& colCondition : w2) {
                    const auto& firstTableCols = (tableId[0] == colCondition.col1[0][0]) ? colCondition.col1 : colCondition.col2;
                    const auto& secondTableCols = (tableId[1] == colCondition.col2[0][0]) ? colCondition.col2 : colCondition.col1;
                    const auto& firstCell = Table[tableId[0]].dataRow[i][firstTableCols[1][0]];
                    const auto& secondCell = Table[tableId[1]].dataRow[j][secondTableCols[1][0]];

                    if (firstCell.type == Int) {
                        if (firstCell.num != secondCell.num) {
                            match = false;
                            break;
                        }
                    } else {
                        if (firstCell.s != secondCell.s) {
                            match = false;
                            break;
                        }
                    }
                }
                if (match) {
                    a1.push_back(i);
                    a2.push_back(j);
                }
            }
        }
        result.push_back(a1);
        result.push_back(a2);
    } else {
        result.push_back(table1Matches);
    }
    return result;
}


void showResult(const vector<vector<int>>& matrix,const vector<vector<int>>& result,array<int, 2>& tableId){
    if( static_cast<int>(result.size())==1){
        for(int i=0;i< static_cast<int>(result[0].size());i++){
            for(int j =0 ;j< static_cast<int>(matrix[0].size());j++){
                if(Table[matrix[0][j]].dataRow[result[0][i]][matrix[1][j]].type==Int)
                    cout<<Table[matrix[0][j]].dataRow[result[0][i]][matrix[1][j]].num<<" ";
                else 
                    cout<<Table[matrix[0][j]].dataRow[result[0][i]][matrix[1][j]].s<<" ";
            }
            cout<<endl;
        }
    }else{
        for(int i=0;i< static_cast<int>(result[0].size());i++){
            for(int j =0 ;j< static_cast<int>(matrix[0].size());j++){ 
                if(tableId[0]==matrix[0][j]){
                    if(Table[matrix[0][j]].dataRow[result[0][i]][matrix[1][j]].type==Int)
                        cout<<Table[matrix[0][j]].dataRow[result[0][i]][matrix[1][j]].num<<" ";
                    else 
                        cout<<Table[matrix[0][j]].dataRow[result[0][i]][matrix[1][j]].s<<" ";
                }else{
                    if(Table[matrix[0][j]].dataRow[result[1][i]][matrix[1][j]].type==Int)
                        cout<<Table[matrix[0][j]].dataRow[result[1][i]][matrix[1][j]].num<<" ";
                    else 
                        cout<<Table[matrix[0][j]].dataRow[result[1][i]][matrix[1][j]].s<<" ";
                }
            }
            cout<<endl;
        }
    }
}
int main() {
    InitTable();
    read_data();
    int n;
    cin>>n;
    cin.ignore();
    for ( int i =0 ; i < n ; i++){
        char q[200];
        fgets(q,sizeof(q),stdin);
        vector<string> sql = word_split(q);

        array<int, 3> loc=getLoc(sql);

        //cout<<loc[0]<<" "<<loc[1]<<" "<<loc[2]<<endl;
        array<int, 2> tableId = get_table(sql,loc);

        //cout<<tableId[0]<<" "<<tableId[1]<<endl;
        vector<vector<int>> matrix = get_col(sql,loc,tableId);

        /*
        for(int j=0;j<matrix.size();j++){
            for(int k = 0;k<matrix[j].size();k++){
                cout<<matrix[j][k]<<" ";
            }
            cout<<endl;
        }*/
        vector<column_cmp_constant> w1;
        vector<col_and_col> w2;
        get_after_where(sql,loc,tableId,w1,w2);
        /*
        if(!w1.empty()){
            for(int j = 0;j<w1.size();j++){
                cout<<w1[j].column[0][0]<<" "<<w1[j].column[1][0]<<endl;
                cout<<w1[j].cmp<<endl;
                if(w1[j].constant.type==Str) cout<<w1[j].constant.s<<endl;
                else cout<<w1[j].constant.num<<endl;
            }
        }
        if(!w2.empty()){
            for(int j = 0;j<w2.size();j++){
                cout<<w2[j].col1[0][0]<<" "<<w2[j].col1[1][0]<<endl;
                cout<<w2[j].col2[0][0]<<" "<<w2[j].col2[1][0]<<endl;
            }
        }*/
        vector<vector<int>>result = query(w1, w2, tableId);
        if(!result.empty())
            showResult(matrix,result,tableId);
    }
    system("pause");
    return 0;
}
