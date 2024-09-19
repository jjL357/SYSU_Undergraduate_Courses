#include <iostream>
#include <stack>
#include <unordered_map>
using namespace std;

bool isValid(string s) {
    int n = s.size();

    unordered_map<char, char> pairs = {
        {')', '('},
        {']', '['},
        {'}', '{'}
    };
    stack<char> stk;
    for (char ch : s) {
        if (pairs.count(ch)) {
            if (stk.empty() || stk.top() != pairs[ch]) {
                return false; // 如果栈为空或者栈顶括号不匹配当前括号，返回false
            }
            stk.pop(); // 匹配成功，弹出栈顶括号
        }
        else if((ch=='(') ||(ch=='{')||(ch=='[')){
            stk.push(ch); // 如果是左括号，入栈
        }
    }
    return stk.empty(); // 最后如果栈为空，说明所有括号都匹配
}

int main() {
    string s;
    int x;
    cin>>x;
    while(x>0){
    cin >> s;
    if (isValid(s)) {
        cout << "Yes" << endl;
    } else {
        cout << "No" << endl;
    }
    x--;
    }
    return 0;
}
