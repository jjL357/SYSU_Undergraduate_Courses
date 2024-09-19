#include <iostream>
#include <string>
#include <vector>
#include <stack>
#include <queue>
#include <set>
#include <map>
#include <unordered_set>
#include <climits>

using namespace std;
// 计算状态集合的闭包
set<int> epsilonClosure(vector<vector<set<int>>>& trans, set<int> nums) 
{
    stack<int> s;
    set<int> states;
    for(auto& it : nums) // 将nums中的元素全部入栈，同时加入states集合
    {
        s.push(it);
        states.insert(it);
    }
    while (!s.empty())   // 通过栈的方式，计算闭包
    {
        int state = s.top();    // 取出栈顶元素
        s.pop();
        for (int next : trans[state][0])    // 通过0号转换，计算下一个状态
        {
            if(next == -1)continue; // 如果是-1，表示stuck，不进行处理
            if(states.count(next))continue; // 如果已经在states集合中，不进行处理
            else    // 否则，将next加入states集合，并入栈
            {
                states.insert(next);
                s.push(next);
            }
        }
    }
    return states;
}

// 计算states状态集合经过symbol转换后能到达的所有状态
set<int> nextState(set<int>& states, int symbol, vector<vector<set<int>>>& trans) 
{
    set<int> nextStates;
    for (int state : states) 
    {
        for (int next : trans[state][symbol])   // 通过symbol转换，计算下一个状态
        {
            if(next == -1)continue;
            else
            {
                nextStates.insert(next);
            }
        }
    }
    return nextStates;
}

// 通过计算得到的DFA表格，判断输入字符串是否可接受
bool isAccepted(string& str, vector<vector<set<int>>>& trans, set<int>& accepted)
{
    set<int> curstate = epsilonClosure(trans, {0});
    for(int i=0;i<str.size();i++)   // 逐个字符处理
    {
        set<int> temp = nextState(curstate, str[i] - 'a' + 1,trans); // 计算下一个状态
        curstate = epsilonClosure(trans,temp);  // 计算下一个状态的闭包
    }
    for(auto& it : curstate)    // 判断是否有可接受状态
    {
        if(accepted.count(it))return true;
    }
    return false;
}

// 打印map
void print_map(vector<vector<set<int>>>& res)
{
    for(auto& it : res)
    {
        for(auto& it1 : it)
        {
            cout<<"{";
            for(auto& it2 : it1)
            {
                cout<<it2<<" ";
            }
            cout<<"}";
        }
        cout<<endl;
    }
}

// 读取输入
vector<vector<set<int>>> read_line(int N, int M)
{
    vector<vector<set<int>>> trans(N, vector<set<int>>(M));
    for (int i = 0; i < N; ++i) 
    {
        // 输入都是字符串形式
        for (int j = 0; j < M; ++j) 
        {
            string str;
            cin >> str;
            string judge = str.substr(1, str.size()-2);
            if(judge.size() == 0)
            {
                trans[i][j].insert(-1); // 如果为空，用-1标识stuck
                continue;
            }
            string temp = "";
            for (int k = 0; k < str.size(); ++k) 
            {
                if (str[k] == ',' || str[k] == '{' || str[k] == '}') 
                {
                    if(temp.size())
                    {   
                        trans[i][j].insert(stoi(temp));
                        temp = "";
                    }
                }
                else if(str[k] == ' ')continue;
                else 
                {
                    temp += str[k];
                }
            }
        }
    }
    return trans;
}

int main() 
{
    int N, M;
    while (cin >> N >> M && (N && M)) 
    {
        vector<vector<set<int>>> trans = read_line(N, M);
        set<int> acceptStates;
        int state;
        while (cin >> state && state != -1) 
        {
            acceptStates.insert(state);
        }
        string str;
        set<int> store = epsilonClosure(trans,{0});
        while (cin >> str && str != "#") 
        {
            // 在这里处理每个待识别的字符串
            if(isAccepted(str, trans, acceptStates))
            {
                cout<<"YES"<<endl;
            }
            else
            {
                cout<<"NO"<<endl;
            }
        }
    }
    return 0;
}
