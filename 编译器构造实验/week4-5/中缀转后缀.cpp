
/*
用栈实现中缀表达式转化为后缀表达式
算法思想
如果检测到数字，则直接加入到后缀表达式中

如果检测到运算符时：

若为‘（’，入栈
若为‘）’，则依次将栈中的运算符加入后缀表达式，直到出现‘（’，并从栈中删除‘（’
若为‘+’，‘-’，‘*’，‘/’
栈空，入栈
栈顶元素为‘（’,入栈
高于栈顶元素优先级，入栈
否则，依次弹出栈顶运算符，直到一个优先级比它低的运算符或‘（’为止
遍历完成，若栈非空，依次弹出栈中所有元素
*/
#include <iostream>
#include <stack>
#include <unordered_map>

using namespace std;

int main() {
    string s;
    unordered_map<char, int> mp;
    mp['*'] = 2;
    mp['/'] = 2;
    mp['+'] = 1;
    mp['-'] = 1;
    mp['('] = 0;

    while (cin >> s && s != "#") {
        stack<char> puc;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] >= '0' && s[i] <= '9') {
                cout << s[i];
            } else if (s[i] == '(') {
                puc.push(s[i]);
            } else if (s[i] == ')') {
                while (!puc.empty() && puc.top() != '(') {
                    cout << puc.top();
                    puc.pop();
                }
                puc.pop(); // Pop the '('
            } else { // Operator
                while (!puc.empty() && mp[puc.top()] >= mp[s[i]]) {
                    cout << puc.top();
                    puc.pop();
                }
                puc.push(s[i]);
            }
        }
        while (!puc.empty()) {
            cout << puc.top();
            puc.pop();
        }
        cout << endl;
    }
    return 0;
}
