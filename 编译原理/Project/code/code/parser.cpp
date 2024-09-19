#include "brain.h"

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>

#include "brain_util.h"


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


std::vector<std::string>AREAS = {LEX, DET, SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P} ;
std::vector<std::string>EXPLICIT_AREAS = {LEX};
std::vector<std::string>RECURRENT_AREAS = {SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P};


class Rule {
    public:
        Rule(int index,std::string rule_name,std::string action,std::string area,
        std::string area1,std::string area2);
    protected:
        int index;
        std::string rule_name;
        std::string action;
        std::string area;
        std::string area1;
        std::string area2;

};

Rule::Rule(int index,std::string rule_name,std::string action,std::string area,
        std::string area1,std::string area2):index(index),rule_name(rule_name),action(action),
        area(area),area1(area1),area2(area2){}

class Rules{
    public:
        Rules(int index);
        void add_PreRule(int index,std::string rule_name,std::string action,std::string area,
        std::string area1,std::string area2);
        void add_PostRule(int index,std::string rule_name,std::string action,std::string area,
        std::string area1,std::string area2);

    protected:
        int index;
        std::vector<Rule>PRE_RULES;
        std::vector<Rule>POST_RULES;

};

Rules::Rules(int index):index(index){}

void Rules::add_PreRule(int index,std::string rule_name,std::string action,std::string area,
                        std::string area1 = "",std::string area2 = ""){
        Rule pre(index,rule_name,action,area,area1,area2);
        PRE_RULES.push_back(pre);
}

void Rules::add_PostRule(int index,std::string rule_name,std::string action,std::string area,
                        std::string area1 = "",std::string area2 = ""){
        Rule post(index,rule_name,action,area,area1,area2);
        POST_RULES.push_back(post);
}

Rules generic_noun(int index){
    Rules noun_rules(index);

    noun_rules.add_PreRule(0,"FiberRule",DISINHIBIT,"",LEX,SUBJ); 
    noun_rules.add_PreRule(0,"FiberRule",DISINHIBIT,"",LEX,OBJ); 
    noun_rules.add_PreRule(0,"FiberRule",DISINHIBIT,"",LEX,PREP_P); 
    noun_rules.add_PreRule(0,"FiberRule",DISINHIBIT,"",DET,SUBJ); 
    noun_rules.add_PreRule(0,"FiberRule",DISINHIBIT,"",DET,OBJ); 
    noun_rules.add_PreRule(0,"FiberRule",DISINHIBIT,"",DET,PREP_P); 
    noun_rules.add_PreRule(0,"FiberRule",DISINHIBIT,"",ADJ,SUBJ); 
    noun_rules.add_PreRule(0,"FiberRule",DISINHIBIT,"",ADJ,OBJ); 
    noun_rules.add_PreRule(0,"FiberRule",DISINHIBIT,"",ADJ,PREP_P); 
    noun_rules.add_PreRule(0,"FiberRule",DISINHIBIT,"",VERB,OBJ); 
    noun_rules.add_PreRule(0,"FiberRule",DISINHIBIT,"",PREP_P,PREP); 
    noun_rules.add_PreRule(0,"FiberRule",DISINHIBIT,"",PREP_P,SUBJ); 
    noun_rules.add_PreRule(0,"FiberRule",DISINHIBIT,"",PREP_P,OBJ); 

    noun_rules.add_PostRule(0,"AreaRule",INHIBIT,DET,"","");
    noun_rules.add_PostRule(0,"AreaRule",INHIBIT,ADJ,"","");
    noun_rules.add_PostRule(0,"AreaRule",INHIBIT,PREP_P,"","");
    noun_rules.add_PostRule(0,"AreaRule",INHIBIT,PREP,"","");
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
    noun_rules.add_PostRule(0, "FiberRule", DISINHIBIT, "", LEX, SUBJ);
    noun_rules.add_PostRule(0, "FiberRule", DISINHIBIT, "", LEX, OBJ);
    noun_rules.add_PostRule(0, "FiberRule", DISINHIBIT, "", DET, SUBJ);
    noun_rules.add_PostRule(0, "FiberRule", DISINHIBIT, "", DET, OBJ);
    noun_rules.add_PostRule(0, "FiberRule", DISINHIBIT, "", ADJ, SUBJ);
    noun_rules.add_PostRule(0, "FiberRule", DISINHIBIT, "", ADJ, OBJ);
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

















class ParserBrain : public Brain {
public:
    ParserBrain(double p, 
                std::map<std::string, Rules> lexeme_dict = {}, 
                std::vector<std::string> all_areas = {}, 
                std::vector<std::string> recurrent_areas = {}, 
                std::vector<std::string> initial_areas = {}, 
                std::map<std::string, std::vector<std::string>> readout_rules = {})
        : Brain(p), 
          lexeme_dict(lexeme_dict), 
          all_areas(all_areas), 
          recurrent_areas(recurrent_areas), 
          initial_areas(initial_areas), 
          readout_rules(readout_rules) {
        initialize_states();
    }

    void initialize_states() {
        for (const auto& from_area : all_areas) {
            fiber_states[from_area] = std::map<std::string, std::set<int>>();
            for (const auto& to_area : all_areas) {
                fiber_states[from_area][to_area].insert(0);
            }
        }
    }

private:
    std::map<std::string, Rules> lexeme_dict;
    std::vector<std::string> all_areas;
    std::vector<std::string> recurrent_areas;
    std::vector<std::string> initial_areas;
    std::map<std::string, std::vector<std::string>> readout_rules;

    std::map<std::string, std::map<std::string, std::set<int>>> fiber_states;
    std::map<std::string, std::set<int>> area_states;
    std::map<std::string, std::set<int>> activated_fibers;
};



/*
void parse(std::string sentence="cats chase mice", std::string language="English", double p=0.1, int LEX_k=20, 
	    int project_rounds=20, bool verbose=true, bool debug=false, readout_method=ReadoutMethod.FIBER_READOUT){

}
*/



int main(){
    //parse();
    return 0;
}