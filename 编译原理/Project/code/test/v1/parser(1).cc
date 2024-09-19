#include "brain.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <variant>
#include <functional>

#include <fstream>

using namespace std;
class TreeNode {
public:
    std::string word;
    std::string role;
    std::vector<TreeNode*> children;
    TreeNode* parent;

    TreeNode(const std::string& word, const std::string& role)
        : word(word), role(role), parent(nullptr) {}

    void add_child(TreeNode* child) {
        children.push_back(child);
        child->parent = this;
    }

    std::string to_string() const {
        return word + " (" + role + ")";
    }
};

TreeNode* get_node(const std::string& word, const std::string& role, std::unordered_map<std::string, TreeNode*>& nodes) {
    std::string key = word + role;
    if (nodes.find(key) == nodes.end()) {
        nodes[key] = new TreeNode(word, role);
    }
    return nodes[key];
}

TreeNode* buildDependencyTree(const std::vector<std::vector<std::string>>& dependency_list, const std::string& root_word, const std::string& root_role) {
    std::unordered_map<std::string, TreeNode*> nodes;

    for (const auto& dep : dependency_list) {
        const std::string& parent_word = dep[0];
        const std::string& child_word = dep[1];
        const std::string& parent_role = dep[2];
        const std::string& child_role = dep[3];

        TreeNode* parent = get_node(parent_word, parent_role, nodes);
        TreeNode* child = get_node(child_word, child_role, nodes);
        parent->add_child(child);
    }

    return get_node(root_word, root_role, nodes);
}

void printTree(TreeNode* node, const std::string& prefix = "") {
    if (!node) return;

    std::cout << prefix;

    if (!node->children.empty()) {
        std::cout << "+-- " << node->to_string() << std::endl;
    } else {
        std::cout << "|-- " << node->to_string() << std::endl;
    }

    for (TreeNode* child : node->children) {
        printTree(child, prefix + "    ");
    }
}

// BrainAreas
enum BrainAreas {
    LEX, DET, SUBJ, OBJ, VERB, PREP, PREP_P, ADJ, ADVERB, NOM, ACC, DAT
};

// Actions
enum Actions {
    DISINHIBIT, INHIBIT, ACTIVATE_ONLY, CLEAR_DET
};

// Fixed area stats for explicit areas
const int LEX_SIZE = 20;

// Definitions
vector<string> AREAS = { "LEX", "DET", "SUBJ", "OBJ", "VERB", "ADJ", "ADVERB", "PREP", "PREP_P" };
vector<string> EXPLICIT_AREAS = { "LEX" };
vector<string> RECURRENT_AREAS = { "SUBJ", "OBJ", "VERB", "ADJ", "ADVERB", "PREP", "PREP_P" };

// Struct definitions
struct AreaRule {
    Actions action;
    BrainAreas area;
    int index;
};

struct FiberRule {
    Actions action;
    BrainAreas area1;
    BrainAreas area2;
    int index;
};
unordered_map<BrainAreas, vector<BrainAreas>> ENGLISH_READOUT_RULES = {
    {VERB, {LEX, SUBJ, OBJ, PREP_P, ADVERB, ADJ}},
    {SUBJ, {LEX, DET, ADJ, PREP_P}},
    {OBJ, {LEX, DET, ADJ, PREP_P}},
    {PREP_P, {LEX, PREP, ADJ, DET}},
    {ADJ, {LEX}},
    {PREP, {LEX}},
    {DET, {LEX}},
    {ADVERB, {LEX}},
    {LEX, {}}
};
// Function definitions
unordered_map<string, BrainAreas> brain_area_map = {
    {"LEX", LEX}, {"DET", DET}, {"SUBJ", SUBJ}, {"OBJ", OBJ}, {"VERB", VERB},
    {"PREP", PREP}, {"PREP_P", PREP_P}, {"ADJ", ADJ}, {"ADVERB", ADVERB},
    {"NOM", NOM}, {"ACC", ACC}, {"DAT", DAT}
};
unordered_map<BrainAreas, string> area_brain_map = {
    {LEX, "LEX"}, {DET, "DET"}, {SUBJ, "SUBJ"}, {OBJ, "OBJ"}, {VERB, "VERB"},
    {PREP, "PREP"}, {PREP_P, "PREP_P"}, {ADJ, "ADJ"}, {ADVERB, "ADVERB"},
    {NOM, "NOM"}, {ACC, "ACC"}, {DAT, "DAT"}
};

using Rule = std::variant<AreaRule, FiberRule>;

struct Lexeme {
    int index;
    vector<Rule> pre_rules;
    vector<Rule> post_rules;
};

Lexeme generic_noun(int index) {
    return {
        index,
        {
            FiberRule{DISINHIBIT, LEX, SUBJ, 0}, FiberRule{DISINHIBIT, LEX, OBJ, 0}, FiberRule{DISINHIBIT, LEX, PREP_P, 0},
            FiberRule{DISINHIBIT, DET, SUBJ, 0}, FiberRule{DISINHIBIT, DET, OBJ, 0}, FiberRule{DISINHIBIT, DET, PREP_P, 0},
            FiberRule{DISINHIBIT, ADJ, SUBJ, 0}, FiberRule{DISINHIBIT, ADJ, OBJ, 0}, FiberRule{DISINHIBIT, ADJ, PREP_P, 0},
            FiberRule{DISINHIBIT, VERB, OBJ, 0}, FiberRule{DISINHIBIT, PREP_P, PREP, 0}, FiberRule{DISINHIBIT, PREP_P, SUBJ, 0},
            FiberRule{DISINHIBIT, PREP_P, OBJ, 0}
        },
        {
            AreaRule{INHIBIT, DET, 0}, AreaRule{INHIBIT, ADJ, 0}, AreaRule{INHIBIT, PREP_P, 0}, AreaRule{INHIBIT, PREP, 0},
            FiberRule{INHIBIT, LEX, SUBJ, 0}, FiberRule{INHIBIT, LEX, OBJ, 0}, FiberRule{INHIBIT, LEX, PREP_P, 0},
            FiberRule{INHIBIT, ADJ, SUBJ, 0}, FiberRule{INHIBIT, ADJ, OBJ, 0}, FiberRule{INHIBIT, ADJ, PREP_P, 0},
            FiberRule{INHIBIT, DET, SUBJ, 0}, FiberRule{INHIBIT, DET, OBJ, 0}, FiberRule{INHIBIT, DET, PREP_P, 0},
            FiberRule{INHIBIT, VERB, OBJ, 0}, FiberRule{INHIBIT, PREP_P, PREP, 0}, FiberRule{INHIBIT, PREP_P, VERB, 0},
            FiberRule{DISINHIBIT, LEX, SUBJ, 1}, FiberRule{DISINHIBIT, LEX, OBJ, 1}, FiberRule{DISINHIBIT, DET, SUBJ, 1},
            FiberRule{DISINHIBIT, DET, OBJ, 1}, FiberRule{DISINHIBIT, ADJ, SUBJ, 1}, FiberRule{DISINHIBIT, ADJ, OBJ, 1},
            FiberRule{INHIBIT, PREP_P, SUBJ, 0}, FiberRule{INHIBIT, PREP_P, OBJ, 0}, FiberRule{INHIBIT, VERB, ADJ, 0}
        }
    };
}

Lexeme generic_trans_verb(int index) {
    return {
        index,
        {
            FiberRule{DISINHIBIT, LEX, VERB, 0}, FiberRule{DISINHIBIT, VERB, SUBJ, 0}, FiberRule{DISINHIBIT, VERB, ADVERB, 0},
            AreaRule{DISINHIBIT, ADVERB, 1}
        },
        {
            FiberRule{INHIBIT, LEX, VERB, 0}, AreaRule{DISINHIBIT, OBJ, 0}, AreaRule{INHIBIT, SUBJ, 0}, AreaRule{INHIBIT, ADVERB, 0},
            FiberRule{DISINHIBIT, PREP_P, VERB, 0}
        }
    };
}

Lexeme generic_intrans_verb(int index) {
    return {
        index,
        {
            FiberRule{DISINHIBIT, LEX, VERB, 0}, FiberRule{DISINHIBIT, VERB, SUBJ, 0}, FiberRule{DISINHIBIT, VERB, ADVERB, 0},
            AreaRule{DISINHIBIT, ADVERB, 1}
        },
        {
            FiberRule{INHIBIT, LEX, VERB, 0}, AreaRule{INHIBIT, SUBJ, 0}, AreaRule{INHIBIT, ADVERB, 0}, FiberRule{DISINHIBIT, PREP_P, VERB, 0}
        }
    };
}

Lexeme generic_copula(int index) {
    return {
        index,
        {
            FiberRule{DISINHIBIT, LEX, VERB, 0}, FiberRule{DISINHIBIT, VERB, SUBJ, 0}
        },
        {
            FiberRule{INHIBIT, LEX, VERB, 0}, AreaRule{DISINHIBIT, OBJ, 0}, AreaRule{INHIBIT, SUBJ, 0}, FiberRule{DISINHIBIT, ADJ, VERB, 0}
        }
    };
}

Lexeme generic_adverb(int index) {
    return {
        index,
        {
            AreaRule{DISINHIBIT, ADVERB, 0}, FiberRule{DISINHIBIT, LEX, ADVERB, 0}
        },
        {
            FiberRule{INHIBIT, LEX, ADVERB, 0}, AreaRule{INHIBIT, ADVERB, 1}
        }
    };
}

Lexeme generic_determinant(int index) {
    return {
        index,
        {
            AreaRule{DISINHIBIT, DET, 0}, FiberRule{DISINHIBIT, LEX, DET, 0}
        },
        {
            FiberRule{INHIBIT, LEX, DET, 0}, FiberRule{INHIBIT, VERB, ADJ, 0}
        }
    };
}

Lexeme generic_adjective(int index) {
    return {
        index,
        {
            AreaRule{DISINHIBIT, ADJ, 0}, FiberRule{DISINHIBIT, LEX, ADJ, 0}
        },
        {
            FiberRule{INHIBIT, LEX, ADJ, 0}, FiberRule{INHIBIT, VERB, ADJ, 0}
        }
    };
}

Lexeme generic_preposition(int index) {
    return {
        index,
        {
            AreaRule{DISINHIBIT, PREP, 0}, FiberRule{DISINHIBIT, LEX, PREP, 0}
        },
        {
            FiberRule{INHIBIT, LEX, PREP, 0}, AreaRule{DISINHIBIT, PREP_P, 0}, FiberRule{INHIBIT, LEX, SUBJ, 1},
            FiberRule{INHIBIT, LEX, OBJ, 1}, FiberRule{INHIBIT, DET, SUBJ, 1}, FiberRule{INHIBIT, DET, OBJ, 1},
            FiberRule{INHIBIT, ADJ, SUBJ, 1}, FiberRule{INHIBIT, ADJ, OBJ, 1}
        }
    };
}

// LEXEME_DICT
unordered_map<string, Lexeme> LEXEME_DICT = {
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

// ENGLISH_READOUT_RULES


class ParserBrain : public Brain {
public:
    unordered_map<string, Lexeme> lexeme_dict;
    vector<string> all_areas;
    vector<string> recurrent_areas;
    vector<string> initial_areas;
    unordered_map<string, unordered_map<string, set<int>>> fiber_states;
    unordered_map<string, set<int>> area_states;
    unordered_map<string, set<string>> activated_fibers;
    unordered_map<BrainAreas, vector<BrainAreas>> readout_rules;

    ParserBrain(float p, unordered_map<string, Lexeme> lexeme_dict, vector<string> all_areas, vector<string> recurrent_areas, vector<string> initial_areas, unordered_map<BrainAreas, vector<BrainAreas>> readout_rules)
        : Brain(p), lexeme_dict(lexeme_dict), all_areas(all_areas), recurrent_areas(recurrent_areas), initial_areas(initial_areas), readout_rules(readout_rules) {
        initialize_states();
    }

    void initialize_states() {
        for (const auto& from_area : all_areas) {
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
    void applyFiberRule(const FiberRule& rule) {
        if (rule.action == INHIBIT) {
            fiber_states[area_brain_map[rule.area1]][area_brain_map[rule.area2]].insert(rule.index);
            fiber_states[area_brain_map[rule.area2]][area_brain_map[rule.area1]].insert(rule.index);
        }
        else if (rule.action == DISINHIBIT) {
            fiber_states[area_brain_map[rule.area1]][area_brain_map[rule.area2]].erase(rule.index);
            fiber_states[area_brain_map[rule.area2]][area_brain_map[rule.area1]].erase(rule.index);
        }
    }

    void applyAreaRule(const AreaRule& rule) {
        if (rule.action == INHIBIT) {
            area_states[area_brain_map[rule.area]].insert(rule.index);
        }
        else if (rule.action == DISINHIBIT) {
            area_states[area_brain_map[rule.area]].erase(rule.index);
        }
    }

    void applyRule(const Rule& rule) {
        if (holds_alternative<FiberRule>(rule)) {
            applyFiberRule(get<FiberRule>(rule));
        }
        else if (holds_alternative<AreaRule>(rule)) {
            applyAreaRule(get<AreaRule>(rule));
        }
    }

    void parse_project() {

        auto project_map = getProjectMap();
        //cout<<"HELLO1"<<endl;
        remember_fibers(project_map);

        project({}, project_map);
        //cout<<"HELLO2"<<endl;
    }

    // For fiber-activation readout, remember all fibers that were ever fired.
    void remember_fibers(const unordered_map<string, vector<string>>& project_map) {
        for (const auto& from_area : project_map) {
            for (const auto& to_area : from_area.second) {
                activated_fibers[from_area.first].insert(to_area);
            }
        }
    }

    bool recurrent(const string& area) {
        return find(recurrent_areas.begin(), recurrent_areas.end(), area) != recurrent_areas.end();
    }

    virtual unordered_map<string, vector<string>> getProjectMap() {
        unordered_map<string, vector<string>> proj_map;
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


    void activateWord(const string& area_name, const string& word) {
        Area& area = area_by_name[area_name];
        int k = area.k;
        //cout<<"area.k: "<<area.k<<endl;
        int assembly_start = lexeme_dict[word].index * k;
        //cout<<assembly_start
        area.winners = vector<int>(k);
        for (int i = 0; i < k; i++) {
            area.winners[i] = (assembly_start + i);
        }
        //cout<<area.winners.size()<<endl;
        //area.winners = vector<int>(assembly_start, assembly_start + k);
        area.fix_assembly();
    }

    void activateIndex(const string& area_name, int index) {
        Area& area = area_by_name[area_name];
        int k = area.k;
        int assembly_start = index * k;
        area.winners = vector<int>(assembly_start, assembly_start + k);
        area.fix_assembly();
    }

    string interpretAssemblyAsString(const string& area_name) {
        return getWord(area_name, 0.7);
    }

    virtual string getWord(const string& area_name, float min_overlap = 0.7) {
        if (area_by_name[area_name].winners.empty()) {
            throw runtime_error("Cannot get word because no assembly in " + area_name);
        }
        set<int> winners(area_by_name[area_name].winners.begin(), area_by_name[area_name].winners.end());
        // cout<<"here"<<endl;
        // for(auto win :winners)cout<<win<<" ";
        // cout<<endl;
        // cout<<"here"<<endl;
        int area_k = area_by_name[area_name].k;
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

    unordered_map<string, set<string>> getActivatedFibers() {
        unordered_map<string, set<string>> pruned_activated_fibers;
        for (const auto& [from_area, to_areas] : activated_fibers) {
            for (const auto& to_area : to_areas) {
                if (find(readout_rules[brain_area_map[from_area]].begin(), readout_rules[brain_area_map[from_area]].end(), brain_area_map[to_area]) != readout_rules[brain_area_map[from_area]].end()) {
                    pruned_activated_fibers[from_area].insert(to_area);
                }
            }
        }
        return pruned_activated_fibers;
    }
};

class EnglishParserBrain : public ParserBrain {
public:
    EnglishParserBrain(float p, int non_LEX_n = 2000, int non_LEX_k = 100, int LEX_k = 20,
        float default_beta = 0.2, float LEX_beta = 1.0, float recurrent_beta = 0.05, float interarea_beta = 0.5, bool verbose = false)
        : ParserBrain(p, LEXEME_DICT, AREAS, RECURRENT_AREAS, { "LEX", "SUBJ", "VERB" }, ENGLISH_READOUT_RULES) {
        if (verbose) cout << "Initializing EnglishParserBrain..." << endl;

        int LEX_n = LEX_SIZE * LEX_k;
        add_explicit_area("LEX", LEX_n, LEX_k, default_beta);

        add_area("SUBJ", non_LEX_n, non_LEX_k, default_beta);
        add_area("OBJ", non_LEX_n, non_LEX_k, default_beta);
        add_area("VERB", non_LEX_n, non_LEX_k, default_beta);
        add_area("ADJ", non_LEX_n, non_LEX_k, default_beta);
        add_area("PREP", non_LEX_n, non_LEX_k, default_beta);
        add_area("PREP_P", non_LEX_n, non_LEX_k, default_beta);
        add_area("DET", non_LEX_n, non_LEX_k, default_beta);
        add_area("ADVERB", non_LEX_n, non_LEX_k, default_beta);

        unordered_map<string, vector<pair<string, float>>> custom_plasticities;
        for (const auto& area : RECURRENT_AREAS) {
            custom_plasticities["LEX"].emplace_back(area, LEX_beta);
            custom_plasticities[area].emplace_back("LEX", LEX_beta);
            custom_plasticities[area].emplace_back(area, recurrent_beta);
            for (const auto& other_area : RECURRENT_AREAS) {
                if (other_area == area) continue;
                custom_plasticities[area].emplace_back(other_area, interarea_beta);
            }
        }
        update_plasticities(custom_plasticities, {});
    }

    unordered_map<string, vector<string>> getProjectMap() override {
        auto proj_map = ParserBrain::getProjectMap();
        if (proj_map.find("LEX") != proj_map.end() && proj_map["LEX"].size() > 2) {
            throw runtime_error("LEX projecting into many areas: " + to_string(proj_map["LEX"].size()));
        }
        return proj_map;
    }

    string getWord(const string& area_name, float min_overlap = 0.7) override {
        string word = ParserBrain::getWord(area_name, min_overlap);
        if (!word.empty()) {
            return word;
        }
        return "<NON-WORD>";
    }
};

class ParserDebugger {
public:
    Brain& b;
    vector<string> all_areas;
    vector<string> explicit_areas;

    ParserDebugger(Brain& brain, const vector<string>& all_areas, const vector<string>& explicit_areas)
        : b(brain), all_areas(all_areas), explicit_areas(explicit_areas) {}

    void run() {
        string command;
        cout << "DEBUGGER: ENTER to continue, 'P' for PEAK\n";
        getline(cin, command);
        while (!command.empty()) {
            if (command == "P") {
                peak();
                return;
            }
            else {
                cout << "DEBUGGER: Command not recognized...\n";
                cout << "DEBUGGER: ENTER to continue, 'P' for PEAK\n";
                getline(cin, command);
            }
        }
    }

    void peak() {
        unordered_map<string, int> remove_map;
        b.disable_plasticity = true;
        b.save_winners = true;

        for (const auto& area : all_areas) {
            b.area_by_name[area].unfix_assembly();
        }
        while (true) {
            string test_proj_map_string;
            cout << "DEBUGGER: enter projection map, eg. {\"VERB\": [\"LEX\"]}, or ENTER to quit\n";
            getline(cin, test_proj_map_string);
            if (test_proj_map_string.empty()) {
                break;
            }
            istringstream ss(test_proj_map_string);
            unordered_map<string, vector<string>> test_proj_map;
            string area1, area2;
            char ch;
            while (ss >> ch && ch != '{') {}
            while (ss >> ch && ch != '}') {
                ss >> ch >> area1 >> ch >> ch;
                vector<string> areas;
                while (ss >> ch && ch != ']') {
                    if (ch != '"' && ch != ',') {
                        ss.unget();
                        ss >> area2;
                        areas.push_back(area2);
                    }
                }
                test_proj_map[area1] = areas;
                ss >> ch;
            }

            set<string> to_area_set;
            for (const auto& [_, to_area_list] : test_proj_map) {
                for (const auto& to_area : to_area_list) {
                    to_area_set.insert(to_area);
                    if (b.area_by_name[to_area].saved_winners.empty()) {
                        b.area_by_name[to_area].saved_winners.push_back(b.area_by_name[to_area].winners);
                    }
                }
            }

            for (const auto& to_area : to_area_set) {
                remove_map[to_area]++;
            }

            b.project({}, test_proj_map);
            for (const auto& area : explicit_areas) {
                // if (to_area_set.count(area)) {
                //     string area_word = b.interpretAssemblyAsString(area);
                //     cout << "DEBUGGER: in explicit area " << area << ", got: " << area_word << endl;
                // }
            }

            cout << "DEBUGGER: print assemblies in areas? Eg. 'LEX,VERB' or ENTER to cont\n";
            string print_assemblies;
            getline(cin, print_assemblies);
            if (print_assemblies.empty()) {
                continue;
            }
            istringstream ss_print(print_assemblies);
            string print_area;
            while (getline(ss_print, print_area, ',')) {
                cout << "DEBUGGER: Printing assembly in area " << print_area << endl;
                for (const auto& winner : b.area_by_name[print_area].winners) {
                    cout << winner << " ";
                }
                cout << endl;
                // if (find(explicit_areas.begin(), explicit_areas.end(), print_area) != explicit_areas.end()) {
                //     string word = b.interpretAssemblyAsString(print_area);
                //     cout << "DEBUGGER: in explicit area got assembly = " << word << endl;
                // }
            }
        }

        for (const auto& [area, num_test_projects] : remove_map) {
            b.area_by_name[area].winners = b.area_by_name[area].saved_winners[0];
            b.area_by_name[area].w = b.area_by_name[area].saved_w[-num_test_projects - 1];
            b.area_by_name[area].saved_w = vector<int>(b.area_by_name[area].saved_w.begin(), b.area_by_name[area].saved_w.end() - num_test_projects);
        }
        b.disable_plasticity = false;
        b.save_winners = false;
        for (const auto& area : all_areas) {
            b.area_by_name[area].saved_winners.clear();
        }
    }
};


enum ReadoutMethod {
    FIXED_MAP_READOUT,
    FIBER_READOUT,
    NATURAL_READOUT
};

void parseHelper(ParserBrain& b, const string& sentence, float p, int LEX_k, int project_rounds, bool verbose, bool debug,
    const unordered_map<string, Lexeme>& lexeme_dict, const vector<string>& all_areas, const vector<string>& explicit_areas,
    ReadoutMethod readout_method, const unordered_map<BrainAreas, vector<BrainAreas>>& readout_rules) {
    
    // 开始计

    istringstream ss(sentence);
    string word;
    vector<string> words;
    while (ss >> word) {
        words.push_back(word);
    }

    for (const auto& word : words) {
        const auto& lexeme = lexeme_dict.at(word);
        b.activateWord("LEX", word);
        if (verbose) {
            cout << "Activated word: " << word << endl;
            for (const auto& winner : b.area_by_name["LEX"].winners) {
                cout << winner << " ";
            }
            cout << endl;
        }

        for (const auto& rule : lexeme.pre_rules) {
            b.applyRule(rule);
        }

        auto proj_map = b.getProjectMap();
        for (const auto& area : proj_map) {
            if (find(proj_map["LEX"].begin(), proj_map["LEX"].end(), area.first) == proj_map["LEX"].end()) {
                b.area_by_name[area.first].fix_assembly();
                if (verbose) {
                    cout << "FIXED assembly bc not LEX->this area in: " << area.first << endl;
                }
            }
            else if (area.first != "LEX") {
                b.area_by_name[area.first].unfix_assembly();
                b.area_by_name[area.first].winners.clear();
                if (verbose) {
                    cout << "ERASED assembly because LEX->this area in " << area.first << endl;
                }
            }
        }

        for (int i = 0; i < project_rounds; ++i) {
            b.parse_project();
        }

        for (const auto& rule : lexeme.post_rules) {
            b.applyRule(rule);
        }
    }

    // Readout
    b.disable_plasticity = true;
    for (const auto& area : all_areas) {
        b.area_by_name[area].unfix_assembly();
    }

    vector<vector<string>> dependencies;
    function<void(const string&, const unordered_map<BrainAreas, vector<BrainAreas>>&)> read_out;
    read_out = [&](const string& area, const unordered_map<BrainAreas, vector<BrainAreas>>& mapping) {
        const auto& to_areas = mapping.at(brain_area_map[area]);
        vector<string> temp;
        for (auto& to_area : to_areas) {
            temp.push_back(area_brain_map[to_area]);
        }
        unordered_map<string, vector<string>> temparea;
        temparea[area] = temp;
        
        b.project({}, temparea);

        string this_word = b.getWord("LEX");
        
        for (const auto& to_area : to_areas) {
            if (to_area == LEX) continue;
            unordered_map<string, vector<string>> temp;
            vector<string> temp1;
            string temp2 = area_brain_map.at(to_area);
            temp1.push_back("LEX");
            temp[temp2] = temp1;
            b.project({}, temp);
            string other_word = b.getWord("LEX");
            dependencies.push_back({ this_word, other_word, area,area_brain_map[to_area] });
        }

        for (const auto& to_area : to_areas) {
            if (to_area != LEX) {
                read_out(area_brain_map[to_area], mapping);
            }
        }
    };

    if (readout_method == FIBER_READOUT) {
        auto activated_fibers = b.getActivatedFibers();
        if (verbose) {
            cout << "Got activated fibers for readout:" << endl;
            for (const auto& [from_area, to_areas] : activated_fibers) {
                cout << from_area << ": ";
                for (const auto& to_area : to_areas) {
                    cout << to_area << " ";
                }
                cout << endl;
            }
        }

        unordered_map<BrainAreas, vector<BrainAreas>> mp;
        for (auto& af : activated_fibers) {
            string first_item = af.first;
            vector<string> second_item(af.second.begin(), af.second.end());
            BrainAreas key = brain_area_map[first_item];
            vector<BrainAreas> value;
            for (auto& s : second_item) {
                value.push_back(brain_area_map[s]);
            }
            mp[key] = value;
        }

        read_out("VERB", mp);
        vector<string> temp;
        ofstream f;
        f.open("/home/www/桌面/Work_Space/编译原理大作业/assemblies_final/testfile/cpp_dependencies.txt",ios::out|ios::app);
        f << "Dependencies: " << endl;
        for (const auto& dep : dependencies) {
            string this_word = dep[0] + ' ' + dep[1] + ' ' + dep[2] + ' ' + dep[3];
            temp.push_back(this_word);
        }
        sort(temp.begin(), temp.end());
        // 根据空格分割字符串,并输出到文件中
        for(auto t : temp)
        {
            stringstream ss(t);
            string word;
            vector<string> words;
            while (ss >> word) {
                words.push_back(word);
            }
            // 输出到文件中

            f << words[0] << " " << words[1] << " " << words[2] << " " << words[3] << endl;
        }
        f<<"\n-----------------------------------------------\n"<<endl;
        f.close();
        
        // TreeNode* parse_tree = buildDependencyTree(dependencies, dependencies[0][0],"VERB");
        // printTree(parse_tree);
    }
}


void parse(const string& sentence = "big people bite the big dogs", const string& language = "English", float p = 0.1, int LEX_k = 20,
    int project_rounds = 20, bool verbose = false, bool debug = false, ReadoutMethod readout_method = FIBER_READOUT) {
    if (language == "English") {
        EnglishParserBrain b(p, 2000, 100, LEX_k, 0.2, 1.0, 0.05, 0.5, verbose);
        parseHelper(b, sentence, p, LEX_k, project_rounds, verbose, debug, LEXEME_DICT, AREAS, EXPLICIT_AREAS, readout_method, ENGLISH_READOUT_RULES);
    }
}