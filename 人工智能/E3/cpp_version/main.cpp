#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "Solver.h"

using namespace std;

void test_read(vector<vector<Atom> > ClauseSet){
    for(int k=0;k<ClauseSet.size();k++){
        vector<Atom> Clause = ClauseSet[k];
        for(int i=0;i<Clause.size();i++){
            Atom at = Clause[i];
            cout<<"P is:"<<at.Predicate<<endl;
            cout<<"T is:";
            for(int j=0;j<at.Term.size();j++){
                cout<<at.Term[j]<<' ';
            }
            cout<<endl;
        }
    }
}

int main() {
    string filePath = "append1.txt";
    vector<vector<Atom> > ClauseSet = readFromFile(filePath);
    //test_read(ClauseSet);//先测试一下
    Solver solver(ClauseSet);
    vector<string> answer = solver.solve();
    printSolution(answer);
    return 0;
}
