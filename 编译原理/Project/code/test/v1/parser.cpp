#include "brain.h"

#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <iterator>
#include <map>
#include <random>
#include <iostream>
#include <set>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <ctime>

using namespace std;

#define LEX "LEX"
#define DET "DET"
#define SUBJ "SUBJ"
#define OBJ "OBJ"
#define VERB "VERB"
#define PREP "PREP"
#define PREP_P "PREP_P"
#define ADJ "ADJ"
#define ADVERB "ADVERB"
#define DISINHIBIT "DISINHIBIT"
#define INHIBIT  "INHIBIT"
#define ACTIVATE_ONLY  "ACTIVATE_ONLY"
#define CLEAR_DET  "CLEAR_DET"

int LEX_SIZE = 20;

// 所有的区域名称
std::vector<std::string> AREAS = {LEX, DET, SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P};
// 显式区域名称列表
std::vector<std::string> EXPLICIT_AREAS = {LEX};
// 循环区域名称列表
std::vector<std::string> RECURRENT_AREAS = {SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P};

// 函数模板，用于将 set 转换为 vector
template <typename T>
std::vector<T> set_to_vector(const std::set<T>& s) {
    return std::vector<T>(s.begin(), s.end());
}

// 函数模板，用于将 unordered_map<string, set<T>> 转换为 unordered_map<string, vector<T>>
template <typename T>
std::unordered_map<std::string, std::vector<T>> set_map_to_vector_map(const std::unordered_map<std::string, std::set<T>>& umap_set) {
    std::unordered_map<std::string, std::vector<T>> umap_vec;
    for (const auto& pair : umap_set) {
        umap_vec[pair.first] = set_to_vector(pair.second);
    }
    return umap_vec;
}

// Rule 类
class Rule {
public:
    Rule(int index, std::string rule_name, std::string action, std::string area,
         std::string area1, std::string area2);
    int index;
    std::string rule_name;
    std::string action;
    std::string area;
    std::string area1;
    std::string area2;
};

// Rule 类构造函数
Rule::Rule(int index, std::string rule_name, std::string action, std::string area,
           std::string area1, std::string area2)
    : index(index), rule_name(rule_name), action(action),
      area(area), area1(area1), area2(area2) {}

// Rules 类
class Rules {
public:
    Rules(int index);
    Rules(){};
    Rules(int index, std::vector<Rule> PRE_RULES, std::vector<Rule> POST_RULES)
        : index(index), PRE_RULES(PRE_RULES), POST_RULES(POST_RULES) {};
    void add_PreRule(int index, std::string rule_name, std::string action, std::string area,
                     std::string area1 = "", std::string area2 = "");
    void add_PostRule(int index, std::string rule_name, std::string action, std::string area,
                      std::string area1 = "", std::string area2 = "");
    int index;
    std::vector<Rule> PRE_RULES;
    std::vector<Rule> POST_RULES;
};

// Rules 类构造函数
Rules::Rules(int index) : index(index) {}

// 添加前置规则的方法
void Rules::add_PreRule(int index, std::string rule_name, std::string action, std::string area,
                        std::string area1, std::string area2) {
    Rule pre(index, rule_name, action, area, area1, area2);
    PRE_RULES.push_back(pre);
}

// 添加后置规则的方法
void Rules::add_PostRule(int index, std::string rule_name, std::string action, std::string area,
                         std::string area1, std::string area2) {
    Rule post(index, rule_name, action, area, area1, area2);
    POST_RULES.push_back(post);
}

// 创建针对不同词性的通用规则的函数
Rules generic_noun(int index) {
    Rules noun_rules(index);

    noun_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", LEX, SUBJ);
    noun_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", LEX, OBJ);
    noun_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", LEX, PREP_P);
    noun_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", DET, SUBJ);
    noun_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", DET, OBJ);
    noun_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", DET, PREP_P);
    noun_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", ADJ, SUBJ);
    noun_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", ADJ, OBJ);
    noun_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", ADJ, PREP_P);
    noun_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", VERB, OBJ);
    noun_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", PREP_P, PREP);
    noun_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", PREP_P, SUBJ);
    noun_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", PREP_P, OBJ);

    noun_rules.add_PostRule(0, "AreaRule", INHIBIT, DET, "", "");
    noun_rules.add_PostRule(0, "AreaRule", INHIBIT, ADJ, "", "");
    noun_rules.add_PostRule(0, "AreaRule", INHIBIT, PREP_P, "", "");
    noun_rules.add_PostRule(0, "AreaRule", INHIBIT, PREP, "", "");
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", LEX, SUBJ);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", LEX, OBJ);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", LEX, PREP_P);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", ADJ, SUBJ);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", ADJ, OBJ);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", ADJ, PREP_P);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", DET, SUBJ);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", DET, OBJ);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", DET, PREP_P);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", VERB, OBJ);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", PREP_P, PREP);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", PREP_P, VERB);
    noun_rules.add_PostRule(1, "FiberRule", DISINHIBIT, "", LEX, SUBJ);
    noun_rules.add_PostRule(1, "FiberRule", DISINHIBIT, "", LEX, OBJ);
    noun_rules.add_PostRule(1, "FiberRule", DISINHIBIT, "", DET, SUBJ);
    noun_rules.add_PostRule(1, "FiberRule", DISINHIBIT, "", DET, OBJ);
    noun_rules.add_PostRule(1, "FiberRule", DISINHIBIT, "", ADJ, SUBJ);
    noun_rules.add_PostRule(1, "FiberRule", DISINHIBIT, "", ADJ, OBJ);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", PREP_P, SUBJ);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", PREP_P, OBJ);
    noun_rules.add_PostRule(0, "FiberRule", INHIBIT, "", VERB, ADJ);

    return noun_rules;
}

Rules generic_trans_verb(int index){
    Rules verb_rules(index);
    
    verb_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", LEX, VERB);
    verb_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", VERB, SUBJ);
    verb_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", VERB, ADVERB);
    verb_rules.add_PreRule(1, "AreaRule", "", DISINHIBIT, ADVERB);


    verb_rules.add_PostRule(0, "FiberRule", INHIBIT, "", LEX, VERB);
    verb_rules.add_PostRule(0, "AreaRule", "", DISINHIBIT, OBJ);
    verb_rules.add_PostRule(0, "AreaRule", "", INHIBIT, SUBJ);
    verb_rules.add_PostRule(0, "AreaRule", "", INHIBIT, ADVERB);
    verb_rules.add_PostRule(0, "FiberRule", DISINHIBIT, "", PREP_P, VERB);

    return verb_rules;
}

Rules generic_intrans_verb(int index){
    Rules verb_rules(index);
    
    verb_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", LEX, VERB);
    verb_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", VERB, SUBJ);
    verb_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", VERB, ADVERB);
    verb_rules.add_PreRule(1, "AreaRule", "", DISINHIBIT, ADVERB);


    verb_rules.add_PostRule(0, "FiberRule", INHIBIT, "", LEX, VERB);
    verb_rules.add_PostRule(0, "AreaRule", "", INHIBIT, SUBJ);
    verb_rules.add_PostRule(0, "AreaRule", "", INHIBIT, ADVERB);
    verb_rules.add_PostRule(0, "FiberRule", DISINHIBIT, "", PREP_P, VERB);

    return verb_rules;
}

Rules generic_copula(int index){
    Rules copula_rules(index);
    
    copula_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", LEX, VERB);
    copula_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", VERB, SUBJ);

    copula_rules.add_PostRule(0, "FiberRule", INHIBIT, "", LEX, VERB);
    copula_rules.add_PostRule(0, "AreaRule", "", INHIBIT, OBJ);
    copula_rules.add_PostRule(0, "AreaRule", "", INHIBIT, SUBJ);
    copula_rules.add_PostRule(0, "FiberRule", DISINHIBIT, "", ADJ, VERB);

    return copula_rules;
}

Rules generic_adverb(int index){
    Rules adverb_rules(index);
    
    adverb_rules.add_PreRule(0, "AreaRule", DISINHIBIT, ADVERB);
    adverb_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", LEX,ADVERB);

    adverb_rules.add_PostRule(0, "FiberRule", INHIBIT, "", LEX, ADVERB);
    adverb_rules.add_PostRule(1, "AreaRule", "", INHIBIT, ADVERB);


    return adverb_rules;
}

Rules generic_determinant(int index) {
    Rules determinant_rules(index);

    determinant_rules.add_PreRule(0, "AreaRule", DISINHIBIT, DET);
    determinant_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", LEX, DET);

    determinant_rules.add_PostRule(0, "FiberRule", INHIBIT, "", LEX, DET);
    determinant_rules.add_PostRule(0, "FiberRule", INHIBIT, "", VERB, ADJ);

    return determinant_rules;
}

Rules generic_adjective(int index) {
    Rules adjective_rules(index);

    adjective_rules.add_PreRule(0, "AreaRule", DISINHIBIT, ADJ);
    adjective_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", LEX, ADJ);

    adjective_rules.add_PostRule(0, "FiberRule", INHIBIT, "", LEX, ADJ);
    adjective_rules.add_PostRule(0, "FiberRule", INHIBIT, "", VERB, ADJ);

    return adjective_rules;
}

Rules generic_preposition(int index) {
    Rules preposition_rules(index);

    preposition_rules.add_PreRule(0, "AreaRule", DISINHIBIT, PREP);
    preposition_rules.add_PreRule(0, "FiberRule", DISINHIBIT, "", LEX, PREP);

    preposition_rules.add_PostRule(0, "FiberRule", INHIBIT, "", LEX, PREP);
    preposition_rules.add_PostRule(0, "AreaRule", "", DISINHIBIT, PREP_P);
    preposition_rules.add_PostRule(1, "FiberRule", INHIBIT, "", LEX, SUBJ);
    preposition_rules.add_PostRule(1, "FiberRule", INHIBIT, "", LEX, OBJ);
    preposition_rules.add_PostRule(1, "FiberRule", INHIBIT, "", DET, SUBJ);
    preposition_rules.add_PostRule(1, "FiberRule", INHIBIT, "", DET, OBJ);
    preposition_rules.add_PostRule(1, "FiberRule", INHIBIT, "", ADJ, SUBJ);
    preposition_rules.add_PostRule(1, "FiberRule", INHIBIT, "", ADJ, OBJ);

    return preposition_rules;
}

// 不同词的词类通用规则
std::unordered_map<std::string, Rules> LEXEME_DICT = {
    {"the", generic_determinant(0)},
    {"a", generic_determinant(1)},
    {"dogs", generic_noun(2)},
    {"cats", generic_noun(3)},
    {"mice", generic_noun(4)},
    {"people", generic_noun(5)},
    {"chase", generic_trans_verb(6)},
    {"love", generic_trans_verb(7)},
    {"bite", generic_trans_verb(8)},
    {"of", generic_preposition(9)},
    {"big", generic_adjective(10)},
    {"bad", generic_adjective(11)},
    {"run", generic_intrans_verb(12)},
    {"fly", generic_intrans_verb(13)},
    {"quickly", generic_adverb(14)},
    {"in", generic_preposition(15)},
    {"are", generic_copula(16)},
    {"man", generic_noun(17)},
    {"woman", generic_noun(18)},
    {"saw", generic_trans_verb(19)}
};

//英语readout的规则集合
std::unordered_map<std::string, std::vector<std::string>> ENGLISH_READOUT_RULES = {
    {VERB, {LEX, SUBJ, OBJ, PREP_P, ADVERB, ADJ}},
    {SUBJ, {LEX, DET, ADJ, PREP_P}},
    {OBJ, {LEX, DET, ADJ, PREP_P}},
    {PREP_P, {LEX, PREP, ADJ, DET}},
    {PREP, {LEX}},
    {ADJ, {LEX}},
    {DET, {LEX}},
    {ADVERB, {LEX}},
    {LEX, {}}
};


class ParserBrain:public Brain {
public:    
    std::unordered_map<std::string, Rules> lexeme_dict;
    std::vector<std::string> all_areas;
    std::vector<std::string> recurrent_areas;
    std::vector<std::string> initial_areas;
    std::unordered_map<std::string, vector<std::string>> readout_rules;

    std::unordered_map<std::string, std::unordered_map<std::string, std::set<int>>> fiber_states;
    std::unordered_map<std::string, std::set<int>> area_states;
    std::unordered_map<std::string, std::set<std::string>> activated_fibers;

    ParserBrain(double p, 
                std::unordered_map<std::string, Rules> lexeme_dict = {}, 
                std::vector<std::string> all_areas = {}, 
                std::vector<std::string> recurrent_areas = {}, 
                std::vector<std::string> initial_areas = {}, 
                std::unordered_map<std::string, vector<std::string>> readout_rules = {})
        : Brain(p), 
          lexeme_dict(lexeme_dict), 
          all_areas(all_areas), 
          recurrent_areas(recurrent_areas), 
          initial_areas(initial_areas), 
          readout_rules(readout_rules) {
        initialize_states();
    }
    // 初始化状态
    void initialize_states() {
        for (const auto& from_area : all_areas) {
            //fiber_states[from_area] = unordered_map<string, set<int>>();
            for (const auto& to_area : all_areas) {
                fiber_states[from_area][to_area].insert(0);
            }
        }
        for (const auto& area : all_areas) {
            area_states[area].insert(0);
        }
        for (const auto& area : initial_areas) {
            area_states[area].erase(0);
        }
    }
    // 应用Fiber规则
    void applyFiberRule(const Rule& rule) {
        if (rule.action == INHIBIT) {
            fiber_states[rule.area1][rule.area2].insert(rule.index);
            fiber_states[rule.area2][rule.area1].insert(rule.index);
        } else if (rule.action == DISINHIBIT) {
            
            fiber_states[rule.area1][rule.area2].erase(rule.index);
            fiber_states[rule.area2][rule.area1].erase(rule.index);
        }
    }
    // 应用Area规则
    void applyAreaRule(const Rule& rule) {
        if (rule.action == INHIBIT) {
            area_states[rule.area].insert(rule.index);
        } else if (rule.action == DISINHIBIT) {
            area_states[rule.area].erase(rule.index);
        }
    }
    // 应用规则
    bool applyRule(const Rule& rule) {
        if (rule.rule_name=="FiberRule") {
            applyFiberRule(rule);
            return true;
        }
        if (rule.rule_name=="AreaRule") {
            applyAreaRule(rule);
            return true;
        }
        return false;
    }

    void parse_project() {
        auto project_map = getProjectMap();
        remember_fibers(project_map);
        project({}, project_map);
    }

    void remember_fibers(const unordered_map<std::string, std::vector<std::string>>& project_map) {
        for (const auto& [from_area, to_areas] : project_map) {
            activated_fibers[from_area].insert(to_areas.begin(),to_areas.end());
        }
    }

    bool recurrent(const string& area) const {
        return find(recurrent_areas.begin(),recurrent_areas.end(),area) != recurrent_areas.end();
    }
    // 获取投影
    virtual unordered_map<std::string, std::vector<std::string>> getProjectMap() {
        unordered_map<std::string, std::vector<std::string>> proj_map;
        for (const auto& area1 : all_areas) {
            if (area_states[area1].empty()) {
                for (const auto& area2 : all_areas) {
                    if (area1 == "LEX" && area2 == "LEX") continue;
                    if (area_states[area2].empty()) {
                        if (fiber_states[area1][area2].empty()) {
                            if (!area_by_name[area1].winners.empty()) {
                                proj_map[area1].push_back(area2);
                            }
                            if (!area_by_name[area2].winners.empty()) {
                                proj_map[area2].push_back(area2);
                            }
                        }
                    }
                }
            }
        }
        return proj_map;
    }
    // 激活单词
    void activateWord(const string& area_name, const string& word) {
        Area& area = area_by_name[area_name];
        int k = area.k;
        int assembly_start = lexeme_dict[word].index * k;
        std::vector<int> result_vector(k);
        std::iota(result_vector.begin(), result_vector.end(), assembly_start);
        area.winners = result_vector;
        area.fix_assembly();
    }

    void activateIndex(const string& area_name, int index) {
        Area& area = area_by_name[area_name];
        int k = area.k;
        int assembly_start = index * k;
        std::vector<int> result_vector(k);
        std::iota(result_vector.begin(), result_vector.end(), assembly_start);
        area.winners = result_vector;
        area.fix_assembly();
    }

    string interpretAssemblyAsString(const string& area_name) {
        return getWord(area_name, 0.7);
    }

    virtual string getWord(const string& area_name, double min_overlap = 0.7) {
        const Area& area = area_by_name.at(area_name);
        if (area.winners.empty()) {
            throw runtime_error("Cannot get word because no assembly in " + area_name);
        }
        set<int> winners(area.winners.begin(), area.winners.end());
        int area_k = area.k;
        float threshold = min_overlap * area_k;
        for (const auto& [word, lexeme] : lexeme_dict) {
            int word_index = lexeme.index;
            set<int> word_assembly;
            for (int i = word_index * area_k; i < (word_index + 1) * area_k; ++i) {
                word_assembly.insert(i);
            }
            vector<int> intersection;
            set_intersection(winners.begin(), winners.end(), word_assembly.begin(), word_assembly.end(), back_inserter(intersection));
            if (intersection.size() >= threshold) {
                return word;
            }
        }
        return "<NON-WORD>";
    }
    // // 获取纤维
    unordered_map<string, set<string>> getActivatedFibers() {
        unordered_map<string, set<string>> pruned_activated_fibers;
        for (const auto& [from_area, to_areas] : activated_fibers) {
            for (const auto& to_area : to_areas) {
                for(int i=0;i<readout_rules[from_area].size();i++){
                    if(readout_rules[from_area][i]==to_area) {
                        pruned_activated_fibers[from_area].insert(to_area);
                        break;
                        }
                }
                // 下面三行被上面三行替代
                // if (readout_rules[from_area].count(to_area)) {
                //     pruned_activated_fibers[from_area].push_back(to_area);
                // }
            }
        }
        return pruned_activated_fibers;
    }   
};

class EnglishParserBrain : public ParserBrain {
public:
    EnglishParserBrain(double p, int non_LEX_n = 2000, int non_LEX_k = 100, int LEX_k = 20, 
                       double default_beta = 0.2, double LEX_beta = 1.0, double recurrent_beta = 0.05, 
                       double interarea_beta = 0.5, bool verbose = false)
        : ParserBrain(p, LEXEME_DICT, AREAS, RECURRENT_AREAS, {"LEX", "SUBJ", "VERB"}, ENGLISH_READOUT_RULES), 
          verbose(verbose) {
        
        int LEX_n = LEX_SIZE * LEX_k;
        add_explicit_area("LEX", LEX_n, LEX_k, default_beta);

        int DET_k = LEX_k;
        add_area("SUBJ", non_LEX_n, non_LEX_k, default_beta);
        add_area("OBJ", non_LEX_n, non_LEX_k, default_beta);
        add_area("VERB", non_LEX_n, non_LEX_k, default_beta);
        add_area("ADJ", non_LEX_n, non_LEX_k, default_beta);
        add_area("PREP", non_LEX_n, non_LEX_k, default_beta);
        add_area("PREP_P", non_LEX_n, non_LEX_k, default_beta);
        add_area("DET", non_LEX_n, non_LEX_k, default_beta);
        //add_area("DET", non_LEX_n, DET_k, default_beta);
        add_area("ADVERB", non_LEX_n, non_LEX_k, default_beta);

        // 初始化自定义可塑性
        std::unordered_map<std::string, std::vector<std::pair<std::string, float>>> custom_plasticities;
        for (const auto& area : RECURRENT_AREAS) {
            custom_plasticities["LEX"].emplace_back(area, LEX_beta);
            custom_plasticities[area].emplace_back("LEX", LEX_beta);
            custom_plasticities[area].emplace_back(area, recurrent_beta);
            for (const auto& other_area : RECURRENT_AREAS) {
                if (other_area != area) {
                    custom_plasticities[area].emplace_back(other_area, interarea_beta);
                }
            }
        }
        update_plasticities(custom_plasticities);
    }

    unordered_map<std::string, std::vector<std::string>> getProjectMap()override
    {
        unordered_map<std::string, std::vector<std::string>> proj_map = ParserBrain::getProjectMap();
        if (proj_map.find("LEX") != proj_map.end() && proj_map["LEX"].size() > 2) {
            throw std::runtime_error("Got that LEX projecting into many areas: " + std::to_string(proj_map["LEX"].size()));
        }
        return proj_map;
    }

    std::string getWord(const std::string& area_name, double min_overlap = 0.7)override{
        std::string word = ParserBrain::getWord(area_name, min_overlap);
        if (!word.empty()) {
            return word;
        }
        // if (word.empty() && area_name == "DET") {
        //     std::set<int> winners = area_states[area_name];
        //     int area_k = 100; // 假设 area_k 为 100
        //     double threshold = min_overlap * area_k;
        //     int nodet_index = DET_SIZE - 1;
        //     int nodet_assembly_start = nodet_index * area_k;
        //     std::set<int> nodet_assembly;
        //     for (int i = nodet_assembly_start; i < nodet_assembly_start + area_k; ++i) {
        //         nodet_assembly.insert(i);
        //     }
        //     std::set<int> intersection;
        //     std::set_intersection(winners.begin(), winners.end(), nodet_assembly.begin(), nodet_assembly.end(),
        //                           std::inserter(intersection, intersection.begin()));
        //     if (intersection.size() > threshold) {
        //         return "<null-det>";
        //     }
        // }
        return "<NON-WORD>";
    }

private:
    bool verbose;

};

// class ParserDebugger {
// public:
//     ParserDebugger(ParserBrain* brain, const std::vector<std::string>& all_areas, const std::vector<std::string>& explicit_areas)
//         : b(brain), all_areas(all_areas), explicit_areas(explicit_areas) {}

//     void run() {
//         std::string command;
//         std::cout << "DEBUGGER: ENTER to continue, 'P' for PEAK \n";
//         std::getline(std::cin, command);
//         while (!command.empty()) {
//             if (command == "P") {
//                 peak();
//                 return;
//             } else {
//                 std::cout << "DEBUGGER: Command not recognized...\n";
//                 std::cout << "DEBUGGER: ENTER to continue, 'P' for PEAK \n";
//                 std::getline(std::cin, command);
//             }
//         }
//     }

//     void peak() {
//         std::unordered_map<std::string, int> remove_map;
//         // Temporarily set beta to 0
//         b->disable_plasticity = true;
//         b->save_winners = true;

//         for (const auto& area : all_areas) {
//             b->area_by_name[area].unfix_assembly();
//         }

//         while (true) {
//             std::string test_proj_map_string;
//             std::cout << "DEBUGGER: enter projection map, eg. {\"VERB\": [\"LEX\"]}, or ENTER to quit\n";
//             std::getline(std::cin, test_proj_map_string);
//             if (test_proj_map_string.empty()) {
//                 break;
//             }

//             Json::Value test_proj_map_json;
//             Json::CharReaderBuilder reader;
//             std::string errs;
//             std::stringstream s(test_proj_map_string);
//             if (!Json::parseFromStream(reader, s, &test_proj_map_json, &errs)) {
//                 std::cerr << "Failed to parse projection map: " << errs << std::endl;
//                 continue;
//             }

//             std::unordered_map<std::string, std::vector<std::string>> test_proj_map;
//             for (const auto& key : test_proj_map_json.getMemberNames()) {
//                 for (const auto& val : test_proj_map_json[key]) {
//                     test_proj_map[key].push_back(val.asString());
//                 }
//             }

//             std::set<std::string> to_area_set;
//             for (const auto& [from_area, to_area_list] : test_proj_map) {
//                 for (const auto& to_area : to_area_list) {
//                     to_area_set.insert(to_area);
//                     if (b->area_by_name[to_area].saved_winners.empty()) {
//                         b->area_by_name[to_area].saved_winners.push_back(b->area_by_name[to_area].winners);
//                     }
//                 }
//             }

//             for (const auto& to_area : to_area_set) {
//                 remove_map[to_area]++;
//             }

//             b->project({}, test_proj_map);

//             for (const auto& area : explicit_areas) {
//                 if (to_area_set.find(area) != to_area_set.end()) {
//                     std::string area_word = b->interpretAssemblyAsString(area);
//                     std::cout << "DEBUGGER: in explicit area " << area << ", got: " << area_word << std::endl;
//                 }
//             }

//             std::string print_assemblies;
//             std::cout << "DEBUGGER: print assemblies in areas? Eg. 'LEX,VERB' or ENTER to cont\n";
//             std::getline(std::cin, print_assemblies);
//             if (print_assemblies.empty()) {
//                 continue;
//             }

//             std::istringstream iss(print_assemblies);
//             std::string print_area;
//             while (std::getline(iss, print_area, ',')) {
//                 std::cout << "DEBUGGER: Printing assembly in area " << print_area << std::endl;
//                 std::cout << "{ ";
//                 for (const auto& winner : b->area_by_name[print_area].winners) {
//                     std::cout << winner << " ";
//                 }
//                 std::cout << "}\n";
//                 if (std::find(explicit_areas.begin(), explicit_areas.end(), print_area) != explicit_areas.end()) {
//                     std::string word = b->interpretAssemblyAsString(print_area);
//                     std::cout << "DEBUGGER: in explicit area got assembly = " << word << std::endl;
//                 }
//             }
//         }

//         // Restore assemblies (winners) and w values to before test projections
//         for (const auto& [area, num_test_projects] : remove_map) {
//             b->area_by_name[area].winners = b->area_by_name[area].saved_winners[0];
//             b->area_by_name[area].saved_w = std::vector<int>(b->area_by_name[area].saved_w.begin(), b->area_by_name[area].saved_w.begin() + b->area_by_name[area].saved_w.size() - num_test_projects - 1);
//             b->area_by_name[area].saved_winners.clear();
//         }
//         b->disable_plasticity = false;
//         b->save_winners = false;
//     }

// private:
//     ParserBrain* b;
//     std::vector<std::string> all_areas;
//     std::vector<std::string> explicit_areas;
// };
/*
void parse(std::string sentence="cats chase mice", std::string language="English", double p=0.1, int LEX_k=20, 
	    int project_rounds=20, bool verbose=true, bool debug=false, readout_method=ReadoutMethod.FIBER_READOUT){

}
*/

enum class ReadoutMethod {
    FIXED_MAP_READOUT = 1,
    FIBER_READOUT = 2,
    NATURAL_READOUT = 3
};

// 读取输出函数 read_out
void read_out(const string& area, const unordered_map<string, vector<string>>& mapping, EnglishParserBrain& b, vector<vector<string>>& dependencies) {
    // 获取要读取的目标区域列表
    auto& to_areas = mapping.at(area);

    // 读取当前区域到目标区域的映射
    unordered_map<string, vector<string>> a1;
    for (auto& to : to_areas) {
        a1[area].push_back(to);
    }

    // 项目映射到目标区域
    b.project({}, a1);
    auto this_word = b.getWord("LEX");

    // 遍历每一个目标区域
    for (const auto& to_area : to_areas) {
        if (to_area == "LEX") continue;

        // 准备项目映射
        unordered_map<string, vector<string>> a3;
        unordered_map<string, vector<string>> a2{{to_area, {"LEX"}}};

        // 项目映射到目标区域
        b.project(a3, a2);
        auto other_word = b.getWord("LEX");
        dependencies.push_back({this_word, other_word, to_area});
    }

    // 递归读取输出，处理非 "LEX" 的目标区域
    for (const auto& to_area : to_areas) {
        if (to_area != "LEX") {
            read_out(to_area, mapping, b, dependencies);
        }
    }
}

// 解析辅助函数 parseHelper
int parseHelper(EnglishParserBrain& b, const string& sentence, float p, int LEX_k, int project_rounds, bool verbose, bool debug,
    unordered_map<string, Rules>& lexeme_dict,
    const vector<string>& all_areas, const vector<string>& explicit_areas,
    ReadoutMethod readout_method, const unordered_map<string, vector<string>>& readout_rules) {

    // 解析句子获取单词列表
    string w = "";
    vector<string> words;
    for (int i = 0; i < sentence.length(); i++) {
        if (sentence[i] == ' ') {
            words.push_back(w);
            w = "";
        } else {
            w += sentence[i];
        }
    }
    words.push_back(w);

    // 对每一个单词进行处理
    for (const string& word : words) {
        Rules& lexeme = lexeme_dict.at(word);
        b.activateWord("LEX", word);
        if (verbose) {
            cout << "Activated word: " << word << endl;
            // 输出激活的单词
            for (auto& it : b.area_by_name["LEX"].winners) {
                cout << it << " ";
            }
            cout << endl;
        }

        // 应用预规则
        for (Rule& rule : lexeme.PRE_RULES) {
            b.applyRule(rule);
        }

        // 获取项目映射
        auto proj_map = b.getProjectMap();
        for (const auto& [area, value] : proj_map) {
            // 如果 "LEX" -> area 不在映射中，则修复组装
            if (find(proj_map["LEX"].begin(), proj_map["LEX"].end(), area) == proj_map["LEX"].end()) {
                b.area_by_name[area].fix_assembly();
                if (verbose) {
                    cout << "FIXED assembly bc not LEX->this area in: " << area << endl;
                }
            } else if (area != "LEX") {
                // 否则，解除组装并清空赢家
                b.area_by_name[area].unfix_assembly();
                b.area_by_name[area].winners.clear();
                if (verbose) {
                    cout << "ERASED assembly because LEX->this area in " << area << endl;
                }
            }
        }

        // 进行多轮项目映射
        for (int i = 0; i < project_rounds; ++i) {
            b.parse_project();
        }

        // 应用后规则
        for (Rule& rule : lexeme.POST_RULES) {
            b.applyRule(rule);
        }
    }

    // 禁用可塑性
    b.disable_plasticity = true;
    for (const auto& area : all_areas) {
        b.area_by_name[area].unfix_assembly();
    }

    // 读取依赖关系
    vector<vector<string>> dependencies;

    if (readout_method == ReadoutMethod::FIBER_READOUT) {
        unordered_map<string, set<string>> activated_fibers = b.getActivatedFibers();
        if (verbose) {
            cout << "Got activated fibers for readout:" << endl;
            // 输出激活的纤维
            for (auto& it : activated_fibers) {
                cout << it.first << ": ";
                for (auto& x : it.second) {
                    cout << x << " ";
                }
                cout << endl;
            }
        }

        // 读取输出
        read_out("VERB", set_map_to_vector_map(activated_fibers), b, dependencies);
        cout << "-------------------" << endl;
        cout << "Got dependencies: " << endl;
        // 输出依赖关系
        for (auto& it : dependencies) {
            for (auto& x : it) cout << x << " ";
            cout << endl;
        }
        cout << "-------------------" << endl;
    }
    return words.size();
}

// 解析函数 parse
int parse(const std::string& sentence = "big people bite the big dogs quickly", const std::string& language = "English",
           double p = 0.1, int LEX_k = 20, int project_rounds = 20, bool verbose = true, 
           bool debug = false, ReadoutMethod readout_method = ReadoutMethod::FIBER_READOUT) {
    
    std::unordered_map<std::string, Rules> lexeme_dict;
    std::vector<std::string> all_areas;
    std::vector<std::string> explicit_areas;
    std::unordered_map<std::string, std::vector<std::string>> readout_rules;

    EnglishParserBrain brain = EnglishParserBrain(p,2000,100, LEX_k,0.2,1.0,0.05,0.5, verbose);
    
    lexeme_dict = LEXEME_DICT;
    all_areas = AREAS;
    explicit_areas = EXPLICIT_AREAS;
    readout_rules = ENGLISH_READOUT_RULES;
    
    if (language != "English"){
        cout<<"Not English?"<<endl;
        return -1;
    }
    return parseHelper(brain, sentence, p, LEX_k, project_rounds, verbose, debug, 
                lexeme_dict, all_areas, explicit_areas, readout_method, readout_rules);
}

int main(){
    std::clock_t start = std::clock();
    int words_size = parse();
    std::clock_t end = std::clock();

    // 计算运行时间
    double duration = 1000.0 * (end - start) / CLOCKS_PER_SEC;

    // 输出运行时间
    std::cout << "cost time :" << duration << " ms" << std::endl;
    std::cout << "time / len(words) : " <<duration/words_size<< "ms/word";
    return 0;
}