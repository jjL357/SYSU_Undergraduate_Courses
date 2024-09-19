#ifndef NEMO_BRAIN_H_
#define NEMO_BRAIN_H_
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>


#include <numeric> 
#include <stdexcept>

#include <iterator>
#include <array>




struct BinomialParams {
    int n; // 试验次数
    double p; // 成功概率
};


class Area {
public:

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
    std::map<std::string, std::map<std::string, std::vector<float>>> connectomes_by_stimulus;  // Mapping from stimulus-name to activation-vector for areas
    std::map<std::string, std::map<std::string, std::vector<std::vector<float>>>> connectomes;  // Mapping from source area-name to target area-name to connections
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
        std::map<std::string, std::vector<float>> this_stimulus_connectomes;
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

        std::map<std::string, std::vector<std::vector<float>>> new_connectomes;
        for (auto& kv : area_by_name) {
            std::string other_area_name = kv.first;
            int other_area_size = area_by_name[other_area_name].explicitArea ? area_by_name[other_area_name].n : 0;
            new_connectomes[other_area_name].resize(0);
            if (other_area_name != area_name) {
                connectomes[other_area_name][area_name].resize(other_area_size, std::vector<float>(0));
            }
            area_by_name[other_area_name].beta_by_area[area_name] = area_by_name[other_area_name].beta;
            area_by_name[area_name].beta_by_area[other_area_name] = beta;
        }
        connectomes[area_name] = new_connectomes;
    }
    void add_explicit_area(const std::string& area_name,
                              int n, int k, float beta,
                              float custom_inner_p = 0,
                              float custom_out_p = 0,
                              float custom_in_p = 0) {

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

    float inner_p = (custom_inner_p != 0) ? custom_inner_p : p;
    float in_p = (custom_in_p != 0) ? custom_in_p : p;
    float out_p = (custom_out_p != 0) ? custom_out_p : p;

    std::map<std::string, std::vector<std::vector<float>>> new_connectomes;
    for (auto& kv : area_by_name) {
        const std::string& other_area_name = kv.first;
        Area& other_area = kv.second;

        if (other_area_name == area_name) {
            new_connectomes[other_area_name] = binomial_distribution2(1, inner_p, n, n);
        } else {
            if (other_area.explicitArea) {
                int other_n = area_by_name[other_area_name].n;
                new_connectomes[other_area_name] = binomial_distribution2(1, out_p, n, other_n);
                connectomes[other_area_name][area_name] = binomial_distribution2(1, in_p, other_n, n);
            } else {
                new_connectomes[other_area_name] = std::vector<std::vector<float>>();  // empty
                connectomes[other_area_name][area_name] = std::vector<std::vector<float>>();  // empty
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

void update_plasticities(const std::map<std::string, std::vector<std::pair<std::string, double>>>& area_update_map = {},
                                const std::map<std::string, std::vector<std::pair<std::string, double>>>& stim_update_map = {}) {
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

void Brain::project(std::unordered_map<std::string, std::vector<std::string>>& areas_by_stim,
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
        int num_first_winners = project_into(area, stim_in[area_name], area_in[area_name], verbose);
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
    auto& stimulus_size_by_name = this->stimulus_size_by_name;
    auto& connectomes_by_stimulus = this->connectomes_by_stimulus;
    auto& connectomes = this->connectomes;
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
    std::vector<std::vector<float>>inputs_by_first_winner_index (num_first_winners_processed);
    if (target_area.fixed_assembly) {
        target_area._new_winners = target_area.winners;
        target_area._new_w = target_area.w;
    }
    else{
    // Compute inputs from stimuli
    std::vector<float> prev_winner_inputs(target_area.w, 0.0f);
    for (const auto& stim : from_stimuli) {
        auto& stim_inputs = connectomes_by_stimulus[stim][target_area_name];
        for (size_t i = 0; i < target_area.w; ++i) {
            prev_winner_inputs[i] += stim_inputs[i];
        }
    }

    // Compute inputs from areas
    for (const auto& from_area_name : from_areas) {
        auto& connectome = connectomes[from_area_name][target_area_name];
        auto& from_area = area_by_name[from_area_name];
        for (int w : from_area.winners) {
            for (size_t i = 0; i < target_area.w; ++i) {
                prev_winner_inputs[i] += connectome[w][i];
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

    // Simulate potential new winners
    std::vector<float> all_potential_winner_inputs = prev_winner_inputs;
    int num_inputs = 0;
    int total_k = 0;
    // Case: Area is not explicit
    std::vector<int> input_size_by_from_area_index;
    if (!target_area.explicitArea) {
       
        int num_inputs = 0;

        for (const auto& stim : from_stimuli) {
            int local_k = stimulus_size_by_name[stim];
            input_size_by_from_area_index.push_back(local_k);
            num_inputs++;
        }

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

        int effective_n = target_area.n - target_area.w;
        if (effective_n <= target_area.k) {
            throw std::runtime_error("Remaining size of area too small to sample k new winners.");
        }

        double quantile = (effective_n - target_area.k) / static_cast<double>(effective_n);

        // Use normal approximation
        int alpha = binom_ppf(quantile, total_k, p);
        if (verbose >= 2) {
            std::cout << "Alpha = " << alpha << std::endl;
        }

        std::vector<int> potential_new_winner_inputs = binom_rvs(total_k, p, target_area.k);
        if (verbose >= 2) {
            std::cout << "potential_new_winner_inputs: ";
            for (int input : potential_new_winner_inputs) {
                std::cout << input << " ";
            }
            std::cout << std::endl;
        }

        all_potential_winner_inputs.insert(all_potential_winner_inputs.end(),
                                           potential_new_winner_inputs.begin(),
                                           potential_new_winner_inputs.end());
    }

    // Find the indices of new winners

    std::vector<int> new_winner_indices(all_potential_winner_inputs.begin(),all_potential_winner_inputs.begin()+target_area.k);
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
    
    std::vector<float> first_winner_inputs;
    if (!target_area.explicitArea) {
        first_winner_inputs.reserve(target_area.k);
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
        std::cout << "new_winners: ";
        for (int winner : target_area._new_winners) {
            std::cout << winner << " ";
        }
        std::cout << std::endl;
    }


    inputs_by_first_winner_index.resize(num_first_winners_processed);
    for(int i =0 ;i<num_first_winners_processed;i++){
      std::vector<int> input_indices = uniqueRandomChoices(total_k,first_winner_inputs[i]);
      std::vector<float> num_connections_by_input_index(num_inputs,0);
      float total_so_far = 0;
      for(int j =0 ;j<num_inputs;j++){
        num_connections_by_input_index[j] = std::accumulate(input_indices.begin(), input_indices.end(), 0,
        [&total_so_far, &input_size_by_from_area_index, &j](int sum, int w) {
            return sum + (total_so_far <= w && w < total_so_far + input_size_by_from_area_index[j]);
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
    /*
    connectomes = self.connectomes_by_stimulus[stim]
    */

    for(auto&from_area_name : from_areas){
      int from_area_w = area_by_name[from_area_name].w;
      std::vector<int> from_area_winners = area_by_name[from_area_name].winners;
      std::set<int>from_area_winners_set;
      for(auto&it:from_area_winners){
        from_area_winners_set.insert(it);
      }
      std::map<std::string,std::vector<std::vector<float>>>from_area_connectomes = connectomes[from_area_name];
      from_area_connectomes[target_area_name] = padArray(from_area_connectomes[target_area_name],num_first_winners_processed);
      std::vector<std::vector<float>>the_connectome = from_area_connectomes[target_area_name];
      for(int i =0 ;i<num_first_winners_processed;i++){
        float total_in = inputs_by_first_winner_index[i][num_inputs_processed];
        std::vector<int>sample_indices = random_choice(from_area_winners,int(total_in));
        for(auto&j:sample_indices){
          the_connectome[j][target_area.w+i] = 1.0;
        }
        for(int j=0;j<from_area_w;j++){
          if(!from_area_winners_set.count(j)){
            the_connectome[j][target_area.w +i] = binomial_distribution(1,p,1)[0];
          }
        }
      }
      float area_to_area_beta = disable_plasticity? 0 : target_area.beta_by_area[from_area_name];
      for(auto&i:target_area._new_winners){
        for(auto&j:from_area_winners){
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
      std::map<std::string,std::vector<std::vector<float>>>other_area_connectomes = connectomes[other_area_name];
      int flag = 0;
      for(auto&it:from_areas){
        if(it==other_area_name){
          flag =1;
          break;
        }
      }
      if(!flag){
        std::vector<std::vector<float>>the_other_area_connectome = padArray(other_area_connectomes[target_area_name],num_first_winners_processed);
        other_area_connectomes[target_area_name] =  padArray(other_area_connectomes[target_area_name],num_first_winners_processed);
        std::vector<std::vector<float>>tmp = binomial_distribution2(1,p,the_other_area_connectome.size(),target_area._new_w - target_area.w);
        
        for(int l1 = 0;l1<the_other_area_connectome.size();l1++){
            for(int l2 = target_area.w;l2<the_other_area_connectome[l1].size();l2++){
                the_other_area_connectome[l1][l2] = tmp[l1][l2 - target_area.w];
            }
        }
      }
      std::map<std::string,std::vector<std::vector<float>>> target_area_connectomes = connectomes[target_area_name];
      std::vector<std::vector<float>> the_target_area_connectome = pad_vector( target_area_connectomes[other_area_name],num_first_winners_processed,0);
      target_area_connectomes[other_area_name] = pad_vector( target_area_connectomes[other_area_name],num_first_winners_processed,0);
      std::vector<std::vector<float>>tmp = binomial_distribution2(1,p,target_area._new_w - target_area.w,the_target_area_connectome[0].size());
       
        for(int l1 = target_area.w;l1<the_target_area_connectome.size();l1++){
            for(int l2 = 0;l2<the_target_area_connectome[l1].size();l2++){
                the_target_area_connectome[l1][l2] = tmp[l1 - target_area.w][l2];
            }
        }
    if(verbose>=2){
      std::cout<<"Connectome of "<< target_area_name <<" to "<<other_area_name<<" is now "<<std::endl;
      for(auto&it:connectomes[target_area_name][other_area_name]){
        for(auto&x:it){
          std::cout<<x<<" ";
        }
        std::cout<<std::endl;
      }
    }
    }

  
  return num_first_winners_processed;
}



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
    int original_rows = input.size();
    int original_cols = input[0].size();
    int new_rows = original_rows + pad_rows;
    int new_cols = original_cols + pad_cols;

    // 创建一个新的二维向量，并初始化为0
    std::vector<std::vector<float>> padded(new_rows, std::vector<float>(new_cols, 0.0f));

    // 复制原始向量的值到新的二维向量中
    for (int i = 0; i < original_rows; ++i) {
        for (int j = 0; j < original_cols; ++j) {
            padded[i][j] = input[i][j];
        }
    }

    // 将新二维向量赋值回输入向量
    return padded;
}

// Function to pad the array
std::vector<std::vector<float>> padArray(const std::vector<std::vector<float>>& array, int num_first_winners_processed) {
    int originalRows = array.size();
    int originalCols = array[0].size();

    // Create a new array with additional columns
    std::vector<std::vector<float>> paddedArray(originalRows, std::vector<float>(originalCols + num_first_winners_processed, 0));

    // Copy the original array into the new array
    for (int i = 0; i < originalRows; ++i) {
        for (int j = 0; j < originalCols; ++j) {
            paddedArray[i][j] = array[i][j];
        }
    }

    return paddedArray;
}


// 计算最大公约数（Euclidean algorithm）
unsigned long long gcd(unsigned long long a, unsigned long long b) {
    return b ? gcd(b, a % b) : a;
}

// 计算组合数（C(n, k) = n! / (k! * (n-k)!)）
unsigned long long combin(int n, int k) {
    if (k > n) return 0;
    if (k > n - k) k = n - k; // 使用较小的k值来减少乘法次数
    unsigned long long result = 1;
    for (int i = 0; i < k; ++i) {
        result *= (n - i);
        if (result > (std::numeric_limits<unsigned long long>::max)() / (i + 1)) {
            // 避免整数溢出
            result /= (i + 1);
            result *= (i + 1); // 等价于result /= (i + 1) * (i + 1)
        }
    }
    return result / gcd(n - k, k); // 约去公共因子以减少结果的大小
}


double binomial_cdf(int k, const BinomialParams& params) {
    // 计算二项分布的累积分布函数（CDF）
    double cdf = 0.0;
    for (int i = 0; i <= k; ++i) {
        cdf += combin(params.n, i) * std::pow(params.p, i) * std::pow(1 - params.p, params.n - i);
    }
    return cdf;
}

double binom_ppf(double p, int k,double p2) {
    BinomialParams params;
    params.n = k;
    params.p = p2;
    // 计算二项分布的百分点函数（PPF）
    int k = std::round(params.n * params.p); // 初始猜测值
    double cdf = binomial_cdf(k, params);
    
    // 使用牛顿-拉弗森迭代法求解
    const double tolerance = 1e-6;
    while (std::abs(cdf - p) > tolerance) {
        double f_derivative = 0.0;
        for (int i = 0; i <= k; ++i) {
            f_derivative += combin(params.n, i) * i * std::pow(params.p, i-1) * std::pow(1 - params.p, params.n - i);
        }
        k -= (cdf - p) / f_derivative;
        cdf = binomial_cdf(k, params);
    }
    
    return k;
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
        return result;
    }
    }
     std::vector<std::vector<float>> binomial_distribution2(int time, float prob,int size,int size2) {
        std::vector<std::vector<float>> result(size,std::vector<float>(size2,0));
        for (int i = 0; i < size; ++i) {
          for(int k =0 ;k<size2;k++){
            for(int j = 0 ;j < time ;j++){
            result[i][k] += _rng() < prob ? 1.0f : 0.0f;
            }
        }
        return result;
}

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

};   
#endif // NEMO_BRAIN_H_
