#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <regex>

using namespace std;

// 数据库表的定义
struct Record {
    unordered_map<string, string> fields;
};

struct Table {
    string name;
    vector<string> columns;
    vector<Record> records;

    void initColumns() {
        for (auto& col : columns) {
            col = name + "." + col;
        }
    }

    void initRecords() {
        for (auto& record : records) {
            unordered_map<string, string> newFields;
            for (const auto& field : record.fields) {
                newFields[name + "." + field.first] = field.second;
            }
            record.fields = newFields;
        }
    }

    void initialize() {
        initColumns();
        initRecords();
    }
};

// 简单的数据库，包含多张表
class SimpleDatabase {
public:
    void addTable(const Table& table) {
        tables[table.name] = table;
    }

    Table& getTable(const string& name) {
        if (tables.find(name) == tables.end()) {
            throw runtime_error("Table not found: " + name);
        }
        return tables[name];
    }

    unordered_map<string, Table>& getTables() {
        return tables;
    }

private:
    unordered_map<string, Table> tables;
};

// SQL解析器和执行器
class SQLExecutor {
public:
    SQLExecutor(SimpleDatabase& db) : database(db) {}

    vector<Record> executeQuery(const string& query) {
        smatch matches;
        if (!regex_match(query, matches, selectRegex)) {
            throw runtime_error("Malformed query: invalid SELECT statement.");
        }

        string columnsPart = matches[1];
        string tablesPart = matches[2];
        string condition = matches[3];

        

        vector<string> columns;
        bool selectAll = false;
        if (columnsPart == "*") {
            selectAll = true;
        } else {
            columns = splitColumns(columnsPart);
            columns = resolveColumnNames(columns, tablesPart);
        }
        vector<string> tableNames = splitTables(tablesPart);
        vector<Table> Tables;
        Tables.push_back(database.getTable(tableNames[0]));
        if(tableNames.size() == 2)Tables.push_back(database.getTable(tableNames[1]));
        vector<string> mergedColumns = mergeColumns(Tables);
        vector<vector<string>> conditions = parseconditions(condition, mergedColumns, tableNames);

        // for(auto& cond : conditions)
        // {
        //     cout<<cond[0]<<' '<<cond[1]<<' '<<cond[2]<<endl;
        // }

        if(tableNames.size() == 2)Tables.push_back(database.getTable(tableNames[1]));
        //cout<<tableNames.size()<<endl;
        if (tableNames.size() == 1) {
            executeSingleTableQuery(tableNames[0], columns, selectAll, conditions);
            //printResults(results, selectAll ? database.getTable(tableNames[0]).columns : columns);
        } else if (tableNames.size() == 2) {
            //cout<<1<<endl;
            auto judge = filterTables(tableNames, conditions);
            executeJoinQuery(Tables, columns, selectAll, conditions, judge, mergedColumns);
            //printResults(results, selectAll ? mergedColumns : columns);
        } else {
            throw runtime_error("Malformed query: invalid number of tables.");
        }

        return {}; // 返回空结果集，只是为了符合函数的返回类型
    }

private:
    SimpleDatabase& database;
    regex selectRegex = regex(R"(^SELECT\s+(.+?)\s+FROM\s+(.+?)(?:\s+WHERE\s+(.+))?$)");

    void executeSingleTableQuery(const string& tableName, const vector<string>& columns, bool selectAll, vector<vector<string>>& condition) {
        Table& table = database.getTable(tableName);
        vector<Record> result;
        Record Actualresult;

        for (const auto& record : table.records) {
            if (evaluateConditions(record, condition, tableName)) {
                Actualresult = selectAll ? record : projectRecord(record, columns);
                vector<string> queryColumns = selectAll ? table.columns : columns;
                bool first = true;
                for (const auto& col : queryColumns) {
                    if (!first) cout << " ";
                    cout << Actualresult.fields.at(col);
                    first = false;
                }
                cout << endl;
            }

        }
    }

    vector<vector<bool>> filterTables(const vector<string>& tableNames, vector<vector<string>>& condition) {
        vector<vector<bool>> result(2);
        Table& table1 = database.getTable(tableNames[0]);
        Table& table2 = database.getTable(tableNames[1]);
        result[0].resize(table1.records.size(), false);
        result[1].resize(table2.records.size(), false);
        
        for(int i = 0; i < table1.records.size(); i++)
        {
            if(evaluateConditions_filter(table1.records[i], condition, table1.name))
            {
                result[0][i] = true;
            }
            else
            {
                result[0][i] = false;
            }
        }
        for(int i = 0; i < table2.records.size(); i++)
        {
            if(evaluateConditions_filter(table2.records[i], condition, table2.name))
            {
                result[1][i] = true;
            }
            else
            {
                result[1][i] = false;
            }
        }
        
        return result;
    }

    void executeJoinQuery(const vector<Table>& tables, const vector<string>& columns, bool selectAll, vector<vector<string>>& condition, vector<vector<bool>> judge, vector<string>& mergedColumns) {
        const Table& table1 = tables[0];
        const Table& table2 = tables[1];

        string tableName = table1.name + ' ' + table2.name;

        vector<Record> result;
        Record res;

        int index = 0;
        bool flag = true;
        while(index < condition.size())
        {
            if(isCol(condition[index][0]) && isCol(condition[index][2]))
            {
                break;
            }
            index++;
        }
        if(index == condition.size())
        {
            flag = false;
        }
        if(flag)
        {
            string left = condition[index][0];
            string right = condition[index][2];
            if(table1.records[0].fields.find(left) == table1.records[0].fields.end())swap(left, right);
        
            // Using hash table to accelerate the join process
            unordered_map<string, vector<Record>> hashTable;

            for(int i=0;i<table2.records.size();i++)
            {
                if(!judge[1][i])continue;
                Record rec = table2.records[i];
                string key = rec.fields.at(right);
                hashTable[key].push_back(rec);
            }
            for (int i=0;i<table1.records.size();i++) {
                if(!judge[0][i])continue;
                Record rec1 = table1.records[i];
                string key = rec1.fields.at(left); // Change "common_field" to the actual common field name
                if (hashTable.find(key) != hashTable.end()) {
                    for (const auto& rec2 : hashTable[key]) {
                        Record mergedRecord = mergeRecords(rec1, rec2);
                        if (evaluateConditions(mergedRecord, condition, tableName)) {
                            Record res = selectAll ? mergedRecord : projectRecord(mergedRecord, columns);
                            vector<string> queryColumns = selectAll ? mergedColumns : columns;
                            bool first = true;
                            for (const auto& col : queryColumns) {
                                if (!first) cout << " ";
                                cout << res.fields.at(col);
                                first = false;
                            }
                            cout << endl;
                        }
                    }
                }
            }
        }

        else 
        {
            for (int i = 0; i < table1.records.size(); ++i) 
            {
                if (!judge[0][i]) continue;
                Record rec1 = table1.records[i];
                for (int j = 0; j < table2.records.size(); ++j) 
                {
                    if (!judge[1][j]) continue;
                    Record rec2 = table2.records[j];
                    Record mergedRecord = mergeRecords(rec1, rec2);
                    if (evaluateConditions(mergedRecord, condition, tableName)) 
                    {
                        res = selectAll ? mergedRecord : projectRecord(mergedRecord, columns);
                        vector<string> queryColumns = selectAll ? mergedColumns : columns;
                        bool first = true;
                        for (const auto& col : queryColumns) {
                            if (!first) cout << " ";
                            cout << res.fields.at(col);
                            first = false;
                        }
                        cout << std::endl;
                    }
                }
            }
        }
    }

    bool evaluateCondition(const Record& record, vector<string>& condition, const string& tableName) {
        if (condition.empty()) return true;

        string col = condition[0];
        string cmp = condition[1];
        string value = condition[2];
        auto valueSplitPos = value.find('.') != string::npos;
        auto colSplitPos = col.find('.') != string::npos;
        if(valueSplitPos && record.fields.find(value) == record.fields.end())return true;
        if(colSplitPos && record.fields.find(col) == record.fields.end())return true;
        string left = colSplitPos ? record.fields.at(col) : col;
        string right = valueSplitPos ? record.fields.at(value) : value;

        if(right[0] == '"')right = right.substr(1,right.size()-2);

        if (cmp == "=") {
            return left == right;
        } else if (cmp == ">") {
            return stoi(left) > stoi(right);
        } else if (cmp == "<") {
            return stoi(left) < stoi(right);
        } 
        return true;
    }

    vector<vector<string>> parseconditions(const string& conditions, vector<string>& columns, vector<string>& tableNames)
    {
        vector<vector<string>> result;
        if(conditions.empty())return result;
        string cleanedConditions = conditions;
        regex andRegex(R"(\s+AND\s+)");
        sregex_token_iterator iter(cleanedConditions.begin(), cleanedConditions.end(), andRegex, -1);
        sregex_token_iterator end;
        while (iter != end) {
            vector<string> temp;
            string condition = *iter++;
            condition = trim(condition);
            regex conditionRegex(R"((\w+\.\w+|\w+)\s*(=|<|>)\s*("[^"]*"|\d+|\w+\.\w+|\w+))");
            smatch matches;
            if (!regex_search(condition, matches, conditionRegex)) {
                throw runtime_error("Malformed condition: invalid WHERE clause.");
            }
            string col = matches[1];
            string cmp = matches[2];
            string value = matches[3];
            //cout<<col<<' '<<cmp<<' '<<value<<"   ";
            for(auto& column : columns)
            {
                //cout<<column<<' ';
                if(col == column || column.find(col) != string::npos)
                {
                    col = column;
                    break;
                }
            }
            for(auto& val : columns)
            {
                if(value == val || val.find(value) != string::npos)
                {
                    value = val;
                    break;
                }
            }
            temp.push_back(col);
            temp.push_back(cmp);
            temp.push_back(value);
            result.push_back(temp);
        }
        return result;
    }
    bool isCol(const string& col)
    {
        if(col.size() == 0)return false;
        if(col[0] == '"')return false;
        if(col[0] <= '9' && col[0] >= '0')return false;
        return true;
    }
    bool evaluateConditions_filter(const Record& record, vector<vector<string>>& conditions, const string& tableName) {
        if (conditions.empty()) return true;

        for(auto& condition : conditions)
        {
            if(isCol(condition[2]) && isCol(condition[0]))return true;
            if (!evaluateCondition(record, condition, tableName)) return false;
        }

        return true;
    }
    bool evaluateConditions(const Record& record, vector<vector<string>>& conditions, const string& tableName) {
        if (conditions.empty()) return true;

        for(auto& condition : conditions)
        {
            if (!evaluateCondition(record, condition, tableName)) return false;
        }

        return true;
    }

    Record projectRecord(const Record& record, const vector<string>& columns) {
        Record result;
        for (const auto& col : columns) {
            try {
                result.fields[col] = record.fields.at(col);
            } catch (const out_of_range&) {
                throw runtime_error("Column not found in record: " + col);
            }
        }
        return result;
    }

    Record mergeRecords(const Record& record1, const Record& record2) {
        Record result = record1;
        result.fields.insert(record2.fields.begin(), record2.fields.end());
        return result;
    }

    vector<string> mergeColumns(const vector<Table>& tables) {
        vector<string> mergedColumns;
        for (const auto& table : tables) {
            mergedColumns.insert(mergedColumns.end(), table.columns.begin(), table.columns.end());
        }
        return mergedColumns;
    }

    vector<string> splitColumns(const string& columnsPart) {
        vector<string> columns;
        size_t start = 0;
        size_t end = columnsPart.find(',');

        while (end != string::npos) {
            columns.push_back(trim(columnsPart.substr(start, end - start)));
            start = end + 1;
            end = columnsPart.find(',', start);
        }

        columns.push_back(trim(columnsPart.substr(start)));

        return columns;
    }

    vector<string> splitTables(const string& tablesPart) {
        vector<string> tableNames;
        size_t start = 0;
        size_t end = tablesPart.find(',');

        while (end != string::npos) {
            tableNames.push_back(trim(tablesPart.substr(start, end - start)));
            start = end + 1;
            end = tablesPart.find(',', start);
        }

        tableNames.push_back(trim(tablesPart.substr(start)));

        return tableNames;
    }

    string trim(const string& str) {
        string result;
        result.reserve(str.size());
        for (char ch : str) {
            if (!isspace(ch)) {
                result.push_back(ch);
            }
        }
        return result;
    }

    vector<string> resolveColumnNames(const vector<string>& columns, const string& tablesPart) {
        vector<string> resolvedColumns;
        vector<string> tableNames = splitTables(tablesPart);

        for (const auto& col : columns) {
            string resolvedCol = col;
            if (col.find('.') == string::npos) {
                resolvedCol = resolveColumnName(col, tableNames);
            }
            resolvedColumns.push_back(resolvedCol);
        }

        return resolvedColumns;
    }

    string resolveColumnName(const string& column, const vector<string>& tableNames) {
        string resolvedColumn;
        bool found = false;
        for (const auto& tableName : tableNames) {
            auto& table = database.getTable(tableName);
            for (const auto& col : table.columns) {
                if (col.find(column) != string::npos) {
                    if (found) {
                        throw runtime_error("Ambiguous column name: " + column);
                    }
                    resolvedColumn = col;
                    found = true;
                }
            }
        }

        if (!found) {
            throw runtime_error("Column not found: " + column);
        }

        return resolvedColumn;
    }
};

int main() {
    SimpleDatabase db;

    vector<string> tableNames = {"Student", "Course", "Teacher", "Grade", "Teach"};
    unordered_map<string, vector<string>> tableColumns = {
        {"Student", {"sid", "dept", "age"}},
        {"Course", {"cid", "name"}},
        {"Teacher", {"tid", "dept", "age"}},
        {"Grade", {"sid", "cid", "score"}},
        {"Teach", {"cid", "tid"}}
    };

    for (const auto& tableName : tableNames) {
        int n;
        cin >> n;
        Table table;
        table.name = tableName;
        table.columns = tableColumns[tableName];
        for (int i = 0; i < n; ++i) {
            Record record;
            for (const auto& col : table.columns) {
                string value;
                cin >> value;
                record.fields[col] = value;
            }
            table.records.push_back(record);
        }
        table.initialize();
        db.addTable(table);
    }

    int m;
    cin >> m;
    cin.ignore();

    SQLExecutor executor(db);
    for (int i = 0; i < m; ++i) {
        string query;
        getline(cin, query);
        try {
            executor.executeQuery(query);
        } catch (const exception& ex) {
            cerr << "Error: " << ex.what() << endl;
        }
    }

    return 0;
}