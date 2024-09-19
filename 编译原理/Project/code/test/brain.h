#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <queue>

#include <numeric> 
#include <stdexcept>

#include <iterator>
#include <array>

#include <set>

#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>



using namespace std;

/**/
struct BinomialParams {
    int n; // 试验次数
    double p; // 成功概率
};

class CONNECTOME{
    public:
        int row;
        int col;
        vector<vector<float>> con;
        void rowpadding(int new_row,float value){
            for(int i = 0; i<new_row;i++){
                vector<float>tmp(col,value);
                con.push_back(tmp);
            }
            row += new_row;
        }
        void colpadding(int new_col,float value){
            for(int i = 0; i<row;i++){
               for(int j =0 ;j <new_col;j++){
                con[i].push_back(value);
               }
            }
            col +=new_col;
        }
};

class Area {
public:
     Area() : name(""), n(0), k(0), beta(0.05), w(0), _new_w(0), num_first_winners(-1), fixed_assembly(false), explicitArea(false) {} // 默认构造函数
    Area(const std::string& name, int n, int k, double beta = 0.05, int w = 0, bool explicitArea = false)
        : name(name), n(n), k(k), beta(beta), w(w), _new_w(0), num_first_winners(-1), fixed_assembly(false), explicitArea(explicitArea){}

    void _update_winners() {
        winners = _new_winners;
        if (!explicitArea) {
            w = _new_w;
        }
    }

    void update_beta_by_stimulus(const std::string& name, double new_beta) {
        beta_by_stimulus[name] = new_beta;
    }

    void update_area_beta(const std::string& name, double new_beta) {
        beta_by_area[name] = new_beta;
    }

    void fix_assembly() {
        if (winners.empty()) {
            throw std::runtime_error("Area " + name + " does not have assembly; cannot fix.");
        }
        fixed_assembly = true;
    }

    void unfix_assembly() {
        fixed_assembly = false;
    }

    int getNumEverFired() const {
        if (explicitArea) {
            return num_ever_fired;
        } else {
            return w;
        }
    }



    std::string name;
    int n;
    int k;
    double beta;
    std::unordered_map<std::string, double> beta_by_stimulus;
    std::unordered_map<std::string, double> beta_by_area;
    int w;
    int _new_w; 
    std::vector<int> saved_w;
    std::vector<int> winners;
    std::vector<int> _new_winners;  
    std::vector<std::vector<int>> saved_winners;
    int num_first_winners;
    bool fixed_assembly;
    bool explicitArea;
    int num_ever_fired  ;  // This should be initialized as needed in explicit simulation mode
    std::vector<bool>ever_fired;
};


class Brain {
public:
    std::unordered_map<std::string, Area> area_by_name;
    std::unordered_map<std::string, int> stimulus_size_by_name;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<float>>> connectomes_by_stimulus;  // Mapping from stimulus-name to activation-vector for areas
    std::unordered_map<std::string, std::unordered_map<std::string, CONNECTOME>> connectomes;  // Mapping from source area-name to target area-name to connections
    double p;
    bool save_size;
    bool save_winners;
    bool disable_plasticity;
     std::mt19937 _rng;
    std::uniform_real_distribution<float> _uniform_dist; 
    bool _use_normal_ppf;

    Brain(double p, bool save_size = true, bool save_winners = false, int seed = 0)
        : p(p), save_size(save_size), save_winners(save_winners), disable_plasticity(false), _rng(seed),_use_normal_ppf(false) ,_uniform_dist(0.0f, 1.0f){}

    void add_stimulus(const std::string& stimulus_name, int size) {
        stimulus_size_by_name[stimulus_name] = size;
        std::unordered_map<std::string, std::vector<float>> this_stimulus_connectomes;
        for (auto& area_pair : area_by_name) {
            const std::string& area_name = area_pair.first;
            Area& area = area_pair.second;

            if (area.explicitArea) {
                std::vector<float> connectome(size);
                connectome = binomial_distribution(size, p,area.n);
                this_stimulus_connectomes[area_name] = std::move(connectome);
            } else {
                this_stimulus_connectomes[area_name] = std::vector<float>();  // empty
            }
            area.beta_by_stimulus[stimulus_name] = area.beta;
        }
        connectomes_by_stimulus[stimulus_name] = std::move(this_stimulus_connectomes);
    }

    void add_area(const std::string& area_name, int n, int k, float beta) {
        area_by_name[area_name] = Area(area_name, n, k, beta);
        for (auto& kv : connectomes_by_stimulus) {
            kv.second[area_name].resize(0);
            area_by_name[area_name].beta_by_stimulus[kv.first] = beta;
        }
        std::unordered_map<std::string, CONNECTOME>new_connectomes;
        for (auto& kv : area_by_name) {
            std::string other_area_name = kv.first;
            int other_area_size = area_by_name[other_area_name].explicitArea ? area_by_name[other_area_name].n : 0;
            new_connectomes[other_area_name].col = other_area_size;
             new_connectomes[other_area_name].row= 0;
            if (other_area_name != area_name) {
                connectomes[other_area_name][area_name].con.resize(other_area_size, std::vector<float>(0));
                 connectomes[other_area_name][area_name].row = other_area_size;
                 connectomes[other_area_name][area_name].col = 0;

            }
            area_by_name[other_area_name].beta_by_area[area_name] = area_by_name[other_area_name].beta;
            area_by_name[area_name].beta_by_area[other_area_name] = beta;
        }
        connectomes[area_name] = new_connectomes;

    }

    void add_explicit_area(const std::string& area_name,
                              int n, int k, float beta,
                              float custom_inner_p = -1,
                              float custom_out_p = -1,
                              float custom_in_p = -1) {

    area_by_name[area_name] = Area(area_name, n, k, beta, n, true);
    area_by_name[area_name].ever_fired = std::vector<bool>(n, false);
    area_by_name[area_name].num_ever_fired = 0;

    for (auto& kv : connectomes_by_stimulus) {
        const std::string& stim_name = kv.first;
        auto& stim_connectomes = kv.second;

        stim_connectomes[area_name] = binomial_distribution(stimulus_size_by_name[stim_name],
             p, n);
        area_by_name[area_name].beta_by_stimulus[stim_name] = beta;
    }

    float inner_p = (custom_inner_p != -1) ? custom_inner_p : p;
    float in_p = (custom_in_p != -1) ? custom_in_p : p;
    float out_p = (custom_out_p != -1) ? custom_out_p : p;

    std::unordered_map<std::string, CONNECTOME> new_connectomes;
    for (auto& kv : area_by_name) {
        const std::string& other_area_name = kv.first;
        Area& other_area = kv.second;

        if (other_area_name == area_name) {
            new_connectomes[other_area_name].con = binomial_distribution2(1, inner_p, n, n);
            new_connectomes[other_area_name].row = n;
            new_connectomes[other_area_name].col = n;
        } else {
            if (other_area.explicitArea) {
                int other_n = area_by_name[other_area_name].n;
                new_connectomes[other_area_name].con = binomial_distribution2(1, out_p, n, other_n);
                 new_connectomes[other_area_name].row = n;
                 new_connectomes[other_area_name].row = other_n;
                connectomes[other_area_name][area_name].con = binomial_distribution2(1, in_p, other_n, n);

                 connectomes[other_area_name][area_name].row = other_n;
                  connectomes[other_area_name][area_name].col = n;
            } else {
                /*
                std::vector<std::vector<float>>temp(n);
                new_connectomes[other_area_name].con = temp;  // 
                new_connectomes[other_area_name].row = n;
                new_connectomes[other_area_name].col = 0;
                connectomes[other_area_name][area_name].con = std::vector<std::vector<float>>();  // empty
                connectomes[other_area_name][area_name].row = 0;
                connectomes[other_area_name][area_name].col = 0;
                
                */
               new_connectomes[other_area_name].con.clear();
               connectomes[other_area_name][area_name].con.clear();
            }
        }
        area_by_name[other_area_name].beta_by_area[area_name] = area_by_name[other_area_name].beta;
        area_by_name[area_name].beta_by_area[other_area_name] = beta;
    }
    connectomes[area_name] = new_connectomes;
}

  void update_plasticity(const std::string& from_area, const std::string& to_area, double new_beta) {
    area_by_name[to_area].beta_by_area[from_area] = new_beta;
    }

void update_plasticities(const std::unordered_map<std::string, std::vector<std::pair<std::string, double>>>& area_update_map = {},
                                const std::unordered_map<std::string, std::vector<std::pair<std::string, double>>>& stim_update_map = {}) {
    // Update plasticities from area to area
    for (const auto& kv : area_update_map) {
        const std::string& to_area = kv.first;
        for (const auto& update_rule : kv.second) {
            const std::string& from_area = update_rule.first;
            double new_beta = update_rule.second;
            update_plasticity(from_area, to_area, new_beta);
        }
    }

    // Update plasticities from stim to area
    for (const auto& kv : stim_update_map) {
        const std::string& area = kv.first;
        Area& the_area = area_by_name[area];
        for (const auto& update_rule : kv.second) {
            const std::string& stim = update_rule.first;
            double new_beta = update_rule.second;
            the_area.beta_by_stimulus[stim] = new_beta;
        }
    }
}

void activate(const std::string& area_name, int index) {
    Area& area = area_by_name[area_name];
    int k = area.k;
    int assembly_start = k * index;
    area.winners.clear();
    for (int i = assembly_start; i < assembly_start + k; ++i) {
        area.winners.push_back(i);
    }
    area.fix_assembly();
    }

void project(std::unordered_map<std::string, std::vector<std::string>>& areas_by_stim,
                    std::unordered_map<std::string, std::vector<std::string>>& dst_areas_by_src_area,
                    int verbose = 0) {
    std::unordered_map<std::string, std::vector<std::string>> stim_in;
    std::unordered_map<std::string, std::vector<std::string>> area_in;
   
    // Validate stim_area, area_area well defined
    for (auto& pair : areas_by_stim) {
        std::string stim = pair.first;
        if (stimulus_size_by_name.find(stim) == stimulus_size_by_name.end()) {
            throw std::invalid_argument("Not in brain.stimulus_size_by_name: " + stim);
        }
        for (std::string& area_name : pair.second) {
            if (area_by_name.find(area_name) == area_by_name.end()) {
                throw std::invalid_argument("Not in brain.area_by_name: " + area_name);
            }
            stim_in[area_name].push_back(stim);
        }
    }
    
    for (auto& pair : dst_areas_by_src_area) {
        std::string from_area_name = pair.first;
        if (area_by_name.find(from_area_name) == area_by_name.end()) {
            throw std::invalid_argument(from_area_name + " not in brain.area_by_name");
        }
        for (std::string& to_area_name : pair.second) {
            if (area_by_name.find(to_area_name) == area_by_name.end()) {
                throw std::invalid_argument("Not in brain.area_by_name: " + to_area_name);
            }
            area_in[to_area_name].push_back(from_area_name);
        }
    }

    std::unordered_set<std::string> to_update_area_names;
    for (auto& pair : stim_in) {
        to_update_area_names.insert(pair.first);
    }
    for (auto& pair : area_in) {
        to_update_area_names.insert(pair.first);
    }
    
    for (auto& area_name : to_update_area_names) {
        Area& area = area_by_name[area_name];
        int num_first_winners = project_into(area, stim_in[area_name], area_in[area_name],verbose);
        
        area.num_first_winners = num_first_winners;
        if (save_winners) {
            area.saved_winners.push_back(area._new_winners);
        }
    }

    for (auto& area_name : to_update_area_names) {
        Area& area = area_by_name[area_name];
        area._update_winners();
        if (save_size) {
            area.saved_w.push_back(area.w);
        }
    }

}



















int project_into(Area& target_area, const std::vector<std::string>& from_stimuli,
                        const std::vector<std::string>& from_areas, int verbose = 0) {
    
    auto& area_by_name = this->area_by_name;
    auto& _rng = this->_rng;
    std::string target_area_name = target_area.name;

    if (verbose >= 1) {
        std::cout << "Projecting ";
        for (const auto& stim : from_stimuli) {
            std::cout << stim << ", ";
        }
        std::cout << " and ";
        for (const auto& area : from_areas) {
            std::cout << area << ", ";
        }
        std::cout << "into " << target_area.name << std::endl;
    }

    // If projecting from area with no assembly, throw an error.
    for (const auto& from_area_name : from_areas) {
        auto& from_area = area_by_name[from_area_name];
        if (from_area.winners.empty() || from_area.w == 0) {
            throw std::runtime_error("Projecting from area with no assembly: " + from_area_name);
        }
    }
    


    // For experiments with a "fixed" assembly in some area.
    int num_first_winners_processed = 0;
    std::vector<float> first_winner_inputs;
    std::vector<std::vector<float>>inputs_by_first_winner_index (num_first_winners_processed);


    
    if (target_area.fixed_assembly) {
        
        target_area._new_winners = target_area.winners;
        target_area._new_w = target_area.w;
        first_winner_inputs.clear();
        num_first_winners_processed = 0;
    }
    else{
        
    target_area_name = target_area.name;
    std::vector<float> prev_winner_inputs(target_area.w, 0.0f);
    // for (const auto& stim : from_stimuli) {
    //     auto& stim_inputs = connectomes_by_stimulus[stim][target_area_name];
    //     for (size_t i = 0; i < target_area.w; ++i) {
    //         prev_winner_inputs[i] += stim_inputs[i];
    //     }
    // }
  
    for (const auto& from_area_name : from_areas) {
        auto& connectome = connectomes[from_area_name][target_area_name];
        auto& from_area = area_by_name[from_area_name];
        for (int w : from_area.winners) {
            for (size_t i = 0; i < target_area.w; ++i) {
                prev_winner_inputs[i] += connectome.con[w][i];
            }
        }
    }
    
    if (verbose >= 2) {
        std::cout << "prev_winner_inputs: ";
        for (float input : prev_winner_inputs) {
            std::cout << input << " ";
        }
        std::cout << std::endl;
    }
    // 000000000000000000
    // Simulate potential new winners
    
    std::vector<float> all_potential_winner_inputs;
    int total_k = 0;
    // Case: Area is not explicit
    std::vector<int> input_size_by_from_area_index;
    int num_inputs = 0;
      
    if (!target_area.explicitArea) {
        
       input_size_by_from_area_index.clear();
       num_inputs = 0;

        // for (const auto& stim : from_stimuli) {
        //     int local_k = stimulus_size_by_name[stim];
        //     input_size_by_from_area_index.push_back(local_k);
        //     num_inputs++;
        // }
        
        for (const auto& from_area_name : from_areas) {
            int effective_k = area_by_name[from_area_name].winners.size();
            input_size_by_from_area_index.push_back(effective_k);
            num_inputs++;
        }

        total_k = std::accumulate(input_size_by_from_area_index.begin(),
                                      input_size_by_from_area_index.end(), 0);
        if (verbose >= 2) {
            std::cout << "total_k=" << total_k << " and input_size_by_from_area_index=";
            for (int size : input_size_by_from_area_index) {
                std::cout << size << " ";
            }
            std::cout << std::endl;
        }

        float effective_n = target_area.n - target_area.w;
        if (effective_n <= target_area.k) {
            throw std::runtime_error("Remaining size of area too small to sample k new winners.");
        }

        double quantile = (effective_n - target_area.k) / static_cast<double>(effective_n);
       
        float alpha = binom_ppf(quantile, total_k, this->p);
         
        if (verbose >= 2) {
            std::cout << "Alpha = " << alpha << std::endl;
        }
        
        std::vector<int> potential_new_winner_inputs = binom_rvs(total_k, this->p, target_area.k);

        if (verbose >= 2) {
            std::cout << "potential_new_winner_inputs: ";
            for (int input : potential_new_winner_inputs) {
                std::cout << input << " ";
            }
            std::cout << std::endl;
        }
        all_potential_winner_inputs = prev_winner_inputs;
        all_potential_winner_inputs.insert(all_potential_winner_inputs.end(),
                                           potential_new_winner_inputs.begin(),
                                           potential_new_winner_inputs.end());
        
    }
    // 111111111111111
    // Find the indices of new winners
    else{
        all_potential_winner_inputs = prev_winner_inputs;
    }
    
    std::vector<int> new_winner_indices = getNLargestIndices(all_potential_winner_inputs, target_area.k);
    // If area is explicit, update ever_fired

    if (target_area.explicitArea) {
        for (int winner : new_winner_indices) {
            if (!target_area.ever_fired[winner]) {
                target_area.ever_fired[winner] = true;
                target_area.num_ever_fired++;
            }
        }
    }
    
    // Process first winner inputs if area is not explicit
    num_first_winners_processed = 0;
    
    if (!target_area.explicitArea) {
        first_winner_inputs.clear();
        //first_winner_inputs.reserve(target_area.k);
        for (int i = 0; i < target_area.k; ++i) {
            if (new_winner_indices[i] >= target_area.w) {
                first_winner_inputs.push_back(all_potential_winner_inputs[new_winner_indices[i]]);
                new_winner_indices[i] = target_area.w + num_first_winners_processed;
                num_first_winners_processed++;
            }
        }
    }

    target_area._new_winners = new_winner_indices;
    target_area._new_w = target_area.w + num_first_winners_processed;

    if (verbose >= 2) {
        std::cout << "new_winnerspo: ";
        for (int winner : target_area._new_winners) {
            std::cout << winner << " ";
        }
        std::cout << std::endl;
    }


    inputs_by_first_winner_index.resize(num_first_winners_processed);
    for(int i =0 ;i<num_first_winners_processed;i++){
      std::vector<int> input_indices = uniqueRandomChoices(total_k,int(first_winner_inputs[i]));
      std::vector<float> num_connections_by_input_index(num_inputs,0);
      float total_so_far = 0;
      for(int j =0 ;j<num_inputs;j++){//here
        num_connections_by_input_index[j] = std::accumulate(input_indices.begin(), input_indices.end(), 0,
        [&total_so_far, &input_size_by_from_area_index, &j](int sum, int w) {
            return sum + (total_so_far <= w && w <input_size_by_from_area_index[j]+ total_so_far  );
        });
        total_so_far += input_size_by_from_area_index[j];
      } 
      inputs_by_first_winner_index[i] = num_connections_by_input_index;

         if (verbose >= 2) {
      std::cout << "For first_winner #" << i << " with input "
                << first_winner_inputs[i] << " split as so: "
                << "[";
      for (size_t j = 0; j < num_connections_by_input_index.size(); ++j) {
          std::cout << num_connections_by_input_index[j];
          if (j < num_connections_by_input_index.size() - 1) {
              std::cout << ", ";
          }
      }
      std::cout << "]" << std::endl;
    }
    }
    }
   
    int num_inputs_processed = 0;

    
//     for (const string& stim : from_stimuli) {
//     auto& connectomes = connectomes_by_stimulus[stim];
//     vector<float>& target_connectome = connectomes[target_area_name];

//     if (num_first_winners_processed > 0) {
//         target_connectome.resize(target_area._new_w);
//     }
//     else{
//         target_connectome = connectomes[target_area_name];
//     }

//     vector<float> first_winner_synapses(target_connectome.begin() + target_area.w, target_connectome.end());
//     for (int i = 0; i < num_first_winners_processed; ++i) {
//         first_winner_synapses[i] = inputs_by_first_winner_index[i][num_inputs_processed]; // 这里假定 inputs_by_first_winner_index 已定义
//     }

//     double stim_to_area_beta = target_area.beta_by_stimulus[stim];
//     if (disable_plasticity) {
//         stim_to_area_beta = 0.0;
//     }

//     for (int i : target_area._new_winners) {
//         target_connectome[i] *= (1 + stim_to_area_beta);
//     }

//     if (verbose >= 2) {
//         cout << stim << " now looks like: " << endl;
//         for (float val : connectomes[target_area_name]) {
//             cout << val << " ";
//         }
//         cout << endl;
//     }

//     num_inputs_processed += 1;
// }

if (!target_area.explicitArea && num_first_winners_processed > 0) {
    for (auto& kv : connectomes_by_stimulus) {
        const string& stim_name = kv.first;
        if (find(from_stimuli.begin(), from_stimuli.end(), stim_name) != from_stimuli.end()) {
            continue;
        }

        vector<float>& the_connectome = kv.second[target_area_name];
        the_connectome.resize(target_area._new_w);

        for (int i = target_area.w; i < target_area._new_w; ++i) {
            the_connectome[i] = binomial_distribution(stimulus_size_by_name[stim_name],p,1)[0];
        }
    }
}
    
    for(auto&from_area_name : from_areas){
       
      int from_area_w = area_by_name[from_area_name].w;
      std::vector<int> &from_area_winners = area_by_name[from_area_name].winners;
      std::set<int>from_area_winners_set;
      for(auto&it:from_area_winners){
        from_area_winners_set.insert(it);
      }
    
      std::unordered_map<std::string,CONNECTOME>&from_area_connectomes = connectomes[from_area_name];
      //from_area_connectomes[target_area_name] = padArray(from_area_connectomes[target_area_name],num_first_winners_processed);
      
      std::vector<std::vector<float>>&the_connectome = from_area_connectomes[target_area_name].con;
      from_area_connectomes[target_area_name].colpadding(num_first_winners_processed,0);//= padArray( from_area_connectomes[target_area_name],num_first_winners_processed);
       
      for(int i =0 ;i<num_first_winners_processed;i++){
        
        float total_in = inputs_by_first_winner_index[i][num_inputs_processed];
        std::vector<int>sample_indices = random_choice(from_area_winners,int(total_in));
         
        for(auto&j:sample_indices){
          the_connectome[j][target_area.w+i] = 1.0;
        }
       
        for(int j=0;j<from_area_w;j++){
          if(!from_area_winners_set.count(j)){
           
            std::vector<float>tmp=  binomial_distribution(1,p,1);
             //if(target_area.w+i<the_connectome[j].size())//here
            the_connectome[j][target_area.w +i] = tmp[0];


          }
        }
       
      }
    
      float area_to_area_beta = disable_plasticity? 0 : target_area.beta_by_area[from_area_name];
      for(auto&i:target_area._new_winners){
        for(auto&j:from_area_winners){
          //  if(j<the_connectome.size()&&i<the_connectome[j].size())//here
          the_connectome[j][i]*=1.0+area_to_area_beta;
        }
      }
         
      if(verbose>=2){
        std::cout<<"Connectome of " << from_area_name <<" to "<<
        target_area_name <<" is now: " ;
        for(auto&it:the_connectome){
          for(auto&x:it){
            std::cout<<x<<" ";
          }
          std::cout<<std::endl;
        }
        std::cout<<std::endl;
      }
      num_inputs_processed += 1;
    }
    
    
    for(auto&kv:area_by_name){
        
      std::string other_area_name = kv.first;
      Area& other_area = kv.second;
      
      int flag = 0;
      for(auto&it:from_areas){
        if(it==other_area_name){
          flag =1;
          break;
        }
      }
      
      if(!flag){
        
        connectomes[other_area_name][target_area_name] .colpadding(num_inputs_processed,0);
      

        for(int i = 0 ;i < connectomes[other_area_name][target_area_name].row;i++){
            for(int j =target_area.w;j<connectomes[other_area_name][target_area_name].col;j++){
                connectomes[other_area_name][target_area_name].con[i][j] = binomial_distribution(1,p,1)[0];
            }
        }
      }
   
        
      connectomes[target_area_name][other_area_name].rowpadding(num_inputs_processed,0);
        
       for(int i = target_area.w ;i < connectomes[other_area_name][target_area_name].row;i++){
            for(int j =0;j<connectomes[other_area_name][target_area_name].col;j++){
                connectomes[other_area_name][target_area_name].con[i][j] = binomial_distribution(1,p,1)[0];
            }
        }
        
        
       
    if(verbose>=2){
      std::cout<<"Connectome of "<< target_area_name <<" to "<<other_area_name<<" is now "<<std::endl;
      for(auto&it:connectomes[target_area_name][other_area_name].con){
        for(auto&x:it){
          std::cout<<x<<" ";
        }
        std::cout<<std::endl;
      }
    }
    
    }
    cout<<endl;
  return num_first_winners_processed;
}
//std::binomial_distribution<int>(1,p)(rng)

private:

// 函数：生成二项分布的随机数，并填充指定的二维数组区域（浮点类型）
std::vector<std::vector<float>> fill_binomial(int start_col, double p, int num_rows, int num_cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(p);
    std::vector<std::vector<float>> matrix;
    for (int i = 0; i < num_rows; ++i) {
        for (int j = start_col; j < start_col + num_cols; ++j) {
            matrix[i][j] = static_cast<float>(d(gen));
        }
    }
    return matrix;
}

std::vector<int> random_choice(const std::vector<int>& input, int num_choices) {
    std::vector<int> indices(input.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., input.size() - 1
    
    // Random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    
    std::shuffle(indices.begin(), indices.end(), rng); // Shuffle the indices
    
    std::vector<int> result;
    for (int i = 0; i < num_choices; ++i) {
        result.push_back(input[indices[i]]);
    }
    
    return result;
}
std::vector<std::vector<float>> pad_vector(std::vector<std::vector<float>>& input, int pad_rows, int pad_cols) {
     std::vector<std::vector<float>> padded;
    int original_rows = input.size();
    if(!original_rows)return padded;
    int original_cols = input[0].size();
    int new_rows = original_rows + pad_rows;
    int new_cols = original_cols + pad_cols;

    // 创建一个新的二维向量，并初始化为0
    padded.resize(new_rows); 

    // 复制原始向量的值到新的二维向量中
    for (int i = 0; i < original_rows; ++i) {
        padded[i].resize(input[i].size(), 0.0f);
        for (int j = 0; j < input[i].size(); ++j) {
            padded[i][j] = input[i][j];
        }
    }

    // 将新二维向量赋值回输入向量
    return padded;
}

// Function to pad the array
std::vector<std::vector<float>> padArray(const std::vector<std::vector<float>>& array, int num_first_winners_processed) {
    int originalRows = array.size();
    std::vector<std::vector<float>> paddedArray(originalRows);
    if(!originalRows) return paddedArray;
    int originalCols = array[0].size();
    //cout<< originalRows;
    // Create a new array with additional columns
    
    // Copy the original array into the new array
    for (int i = 0; i < originalRows; ++i) {
       //<<array[i].size();
       paddedArray[i].resize(array[i].size() + num_first_winners_processed, 0);
        for (int j = 0; j < originalCols; ++j) {
            paddedArray[i][j] = array[i][j];
        }
    }
    
    return paddedArray;
}

float BinimQuantile(int k ,float p ,float percent){
    double pi = std::pow(1.0-p,k);
    double mul = (1.0*p)/(1.0-p);
    double total_p = pi;
    int i=0;
    while(total_p < percent){
        pi *= ((k - i)* mul )/(i+1);
        total_p+=pi;
        ++i;
    }
    return i;
}

// 计算最大公约数（Euclidean algorithm）
unsigned long long gcd(unsigned long long a, unsigned long long b) {
    return b ? gcd(b, a % b) : a;
}

// 计算组合数 C(n, k)
double combin(int n, int k) {
    double result = 1.0;
    for (int i = 0; i < k; ++i) {
        result *= (n - i) / (i + 1);
    }
    return result;
}

// 计算二项分布的累积分布函数（CDF）
double binomial_cdf(int k, int total_k, double p) {
    double cdf = 0.0;
    for (int i = 0; i <= k; ++i) {
        cdf += combin(total_k, i) * std::pow(p, i) * std::pow(1.0 - p, total_k - i);
    }
    return cdf;
}

// 二项分布的PPF（Percent Point Function）
int binom_ppf(double quantile, int total_k, double p) {
    if (quantile < 0.0 || quantile > 1.0) {
        throw std::invalid_argument("Quantile must be between 0 and 1.");
    }
    if (quantile == 1.0) {
        return total_k;
    }

    // 使用二分搜索查找PPF
    int lower = 0;
    int upper = total_k;
    while (lower < upper) {
        int middle = (lower + upper) / 2;
        double cdf = binomial_cdf(middle, total_k, p);
        if (cdf < quantile) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }

    // 由于二项分布的离散性，我们取最接近但不大于quantile的CDF值对应的k
    return lower;
}


// 模拟二项分布抽样直到达到目标成功次数或最大试验次数
std::vector<int> binom_rvs(int total_k, double p, int target_k) {
    std::vector<int> trials;
    std::mt19937 generator(std::random_device{}()); // 随机数生成器
    std::binomial_distribution<int> distribution(total_k, p); // 二项分布

    int successes = 0;
    for (int trial = 0; successes < target_k; ++trial) {
        int result = distribution(generator); // 生成单次试验结果
        trials.push_back(result);
        successes += result;
    }

    return trials;
}


    std::vector<float> binomial_distribution(int time, float prob,int size) {
        std::vector<float> result(size,0);
        for (int i = 0; i < size; ++i) {
            for(int j = 0 ;j < time ;j++){
            result[i] += _rng() < prob ? 1.0f : 0.0f;
        }
        }
        
        return result;
    
    }
     std::vector<std::vector<float>> binomial_distribution2(int time, float prob,int size,int size2) {
        std::vector<std::vector<float>> result(size,std::vector<float>(size2,0));
        for (int i = 0; i < size; ++i) {
          for(int k =0 ;k<size2;k++){
            for(int j = 0 ;j < time ;j++){
            result[i][k] += _rng() < prob ? 1.0f : 0.0f;
            }
        }
        }
        return result;
}




// 生成无放回的随机选择
std::vector<int> uniqueRandomChoices(int total_k, int num_choices) {
    std::vector<int> choices;
    choices.reserve(num_choices);
    std::mt19937 generator(std::random_device{}()); // 创建随机数生成器
    std::uniform_int_distribution<int> distribution(0, total_k - 1); // 定义分布范围

    // 确保请求的选择次数不超过总范围
    if (num_choices > total_k) {
        throw std::invalid_argument("Number of choices exceeds total range.");
    }

    // 使用集合来确保选择的随机性（无放回）
    std::unordered_set<int> choice_set;
    while (choice_set.size() < num_choices) {
        int choice = distribution(generator); // 生成随机数
        choice_set.insert(choice); // 插入集合（自动去重）
    }



    // 将集合转换为向量并返回
    std::vector<int> result(choice_set.begin(), choice_set.end());
    return result;
}

// 定义一个辅助函数，获取最大的 k 个元素的索引
std::vector<int> getNLargestIndices(const std::vector<float>& inputs, int k) {
    // 创建一个优先队列，用于存储对 (值, 索引) 对
    std::priority_queue<std::pair<float, int>> pq;

    // 遍历输入向量，将值和索引插入优先队列
    for (int i = 0; i < inputs.size(); ++i) {
        pq.push(std::make_pair(inputs[i], i));
    }

    // 获取最大的 k 个元素的索引
    std::vector<int> largest_indices;
    for (int i = 0; i < k; ++i) {
        largest_indices.push_back(pq.top().second);
        pq.pop();
    }

    return largest_indices;
}


};   
#endif // NEMO_BRAIN_H_
