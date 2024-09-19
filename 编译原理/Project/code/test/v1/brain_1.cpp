#include "brain.h"
#include <boost/math/distributions/binomial.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <random>
#include <iostream>
#include <algorithm>
using namespace std;

float BinomQuantile(int k, float p, float percent) {
    double pi = std::pow(1.0 - p, k);
    double mul = (1.0 * p) / (1.0 - p);
    double total_p = pi;
    int i = 0;
    while (total_p < percent) {
        pi *= ((k - i) * mul) / (i + 1);
        total_p += pi;
        ++i;
    }
    return i;
}

std::vector<int> sample_indices_func(const std::vector<int>& from_area_winners, int total_in, std::mt19937& rng) {
    std::vector<int> sample;
    std::sample(from_area_winners.begin(), from_area_winners.end(), std::back_inserter(sample), total_in, rng);
    return sample;
}


std::vector<int> choose_random_indices(int total_k, int num_choices, std::mt19937& rng) {
    std::vector<int> indices(total_k);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(num_choices);
    return indices;
}

bool compare_indices(const std::pair<int, float>& a, const std::pair<int, float>& b) {
    return a.second > b.second;
}

std::vector<int> find_largest_indices(const std::vector<float>& values, int k) {
    int n = values.size();
    std::vector<std::pair<int, float>> indexed_values(n);

    for (int i = 0; i < n; ++i) {
        indexed_values[i] = std::make_pair(i, values[i]);
    }

    std::partial_sort(indexed_values.begin(), indexed_values.begin() + k, indexed_values.end(),
        [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second > b.second;
        });

    std::vector<int> largest_indices(k);
    for (int i = 0; i < k; ++i) {
        largest_indices[i] = indexed_values[i].first;
    }

    return largest_indices;
}
template<typename Trng>
float TruncatedNorm(float a, Trng& rng) {
    if (a <= 0.0f) {
        std::normal_distribution<float> norm(0.0f, 1.0f);
        for (;;) {
            const float x = norm(rng);
            if (x >= a) return x;
        }
    }
    else {
        const float alpha = (a + std::sqrt(a * a + 4)) * 0.5f;
        std::exponential_distribution<float> d(alpha);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (;;) {
            const float z = a + d(rng);
            const float dz = z - alpha;
            const float rho = std::exp(-0.5f * dz * dz);
            if (u(rng) < rho) return z;
        }
    }
}


// Helper functions
float binom_ppf(float quantile, int total_k, float p) {
    std::binomial_distribution<int> distribution(total_k, p);
    std::vector<int> binom_samples;
    std::mt19937 gen(42);
    for (int i = 0; i < 10000; ++i) {
        binom_samples.push_back(distribution(gen));
    }
    std::sort(binom_samples.begin(), binom_samples.end());
    return binom_samples[static_cast<int>(quantile * binom_samples.size())];
}

Area::Area(const std::string& name, int n, int k, float beta, int w, bool explicit_area)
    : name(name), n(n), k(k), beta(beta), w(w), num_first_winners(-1), fixed_assembly(false), explicit_area(explicit_area), num_ever_fired(0) {}

void Area::update_winners() {
    winners = _new_winners;
    if (!explicit_area) {
        w = _new_w;
    }
}

void Area::update_area_beta(const std::string& name, float new_beta) {
    beta_by_area[name] = new_beta;
}

void Area::fix_assembly() {
    if (winners.empty()) {
        throw std::runtime_error("Area does not have assembly; cannot fix.");
    }
    fixed_assembly = true;
}

void Area::unfix_assembly() {
    fixed_assembly = false;
}

int Area::get_num_ever_fired() const {
    if (explicit_area) {
        return num_ever_fired;
    }
    else {
        return w;
    }
}

Brain::Brain(float p, bool save_size, bool save_winners, int seed)
    : p(p), save_size(save_size), save_winners(save_winners), disable_plasticity(false), rng(seed) {}

void Brain::add_area(const std::string& area_name, int n, int k, float beta) {
    area_by_name[area_name] = Area(area_name, n, k, beta);

    std::unordered_map<std::string, Fiber> new_connectomes;
    for (auto& pair : area_by_name) {
        const std::string& other_area_name = pair.first;
        Area& other_area = pair.second;
        int other_area_size = other_area.explicit_area ? other_area.n : 0;
        new_connectomes[other_area_name].colcount = other_area_size;
        new_connectomes[other_area_name].rowcount = 0;

        if (other_area_name != area_name) {
            connectomes[other_area_name][area_name].sym = vector<vector<float>>(other_area_size, vector<float>());
            connectomes[other_area_name][area_name].rowcount = other_area_size;
            connectomes[other_area_name][area_name].colcount = 0;
        }
        other_area.beta_by_area[area_name] = other_area.beta;
        area_by_name[area_name].beta_by_area[other_area_name] = beta;
    }
    connectomes[area_name] = new_connectomes;
}

void Brain::add_explicit_area(const std::string& area_name, int n, int k, float beta,
    float custom_inner_p, float custom_out_p, float custom_in_p) {
    float inner_p = (custom_inner_p == -1) ? p : custom_inner_p;
    float in_p = (custom_in_p == -1) ? p : custom_in_p;
    float out_p = (custom_out_p == -1) ? p : custom_out_p;

    area_by_name[area_name] = Area(area_name, n, k, beta, n, true);
    Area& area = area_by_name[area_name];
    area.ever_fired = std::vector<bool>(n, false);

    std::unordered_map<std::string, Fiber> new_connectomes;
    for (auto& pair : area_by_name) {
        const std::string& other_area_name = pair.first;
        Area& other_area = pair.second;
        if (other_area_name == area_name) {
            new_connectomes[other_area_name].sym = std::vector<std::vector<float>>(n, std::vector<float>(n));
            new_connectomes[other_area_name].colcount = n;
            new_connectomes[other_area_name].rowcount = n;
            std::binomial_distribution<int> distribution(1, inner_p);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    new_connectomes[other_area_name].sym[i][j] = distribution(rng);
                }
            }
        }
        else {
            if (other_area.explicit_area) {
                int other_n = other_area.n;
                new_connectomes[other_area_name].sym = std::vector<std::vector<float>>(n, std::vector<float>(other_n));
                new_connectomes[other_area_name].rowcount = n;
                new_connectomes[other_area_name].colcount = other_n;
                std::binomial_distribution<int> distribution(1, out_p);
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < other_n; ++j) {
                        new_connectomes[other_area_name].sym[i][j] = distribution(rng);
                    }
                }
                connectomes[other_area_name][area_name].sym = std::vector<std::vector<float>>(other_n, std::vector<float>(n));
                distribution = std::binomial_distribution<int>(1, in_p);
                for (int i = 0; i < other_n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        connectomes[other_area_name][area_name].sym[i][j] = distribution(rng);
                    }
                }
            }
            else {
                new_connectomes[other_area_name].sym.clear();
                connectomes[other_area_name][area_name].sym.clear();
            }
        }
        other_area.beta_by_area[area_name] = other_area.beta;
        area.beta_by_area[other_area_name] = beta;
    }
    connectomes[area_name] = new_connectomes;
}

void Brain::update_plasticity(const std::string& from_area, const std::string& to_area, float new_beta) {
    area_by_name[to_area].beta_by_area[from_area] = new_beta;
}

void Brain::update_plasticities(const std::unordered_map<std::string, std::vector<std::pair<std::string, float>>>& area_update_map,
    const std::unordered_map<std::string, std::vector<std::pair<std::string, float>>>& stim_update_map) {
    for (const auto& pair : area_update_map) {
        const std::string& to_area = pair.first;
        for (const auto& update_rule : pair.second) {
            const std::string& from_area = update_rule.first;
            float new_beta = update_rule.second;
            update_plasticity(from_area, to_area, new_beta);
        }
    }
}

void Brain::activate(const std::string& area_name, int index) {
    Area& area = area_by_name[area_name];
    int k = area.k;
    int assembly_start = k * index;
    area.winners.resize(k);
    for (int i = 0; i < k; ++i) {
        area.winners[i] = assembly_start + i;
    }
    area.fix_assembly();
}

void Brain::project(const std::unordered_map<std::string, std::vector<std::string>>& areas_by_stim,
    const std::unordered_map<std::string, std::vector<std::string>>& dst_areas_by_src_area, int verbose)
{
    std::unordered_map<std::string, std::vector<std::string>> area_in;


    for (const auto& pair : dst_areas_by_src_area) {
        const std::string& from_area_name = pair.first;
        if (area_by_name.find(from_area_name) == area_by_name.end()) {
            throw std::out_of_range(from_area_name + " not in brain.area_by_name");
        }
        for (const std::string& to_area_name : pair.second) {
            if (area_by_name.find(to_area_name) == area_by_name.end()) {
                throw std::out_of_range("Not in brain.area_by_name: " + to_area_name);
            }
            area_in[to_area_name].push_back(from_area_name);
        }
    }

    std::set<std::string> to_update_area_names;
    for (const auto& pair : area_in) {
        to_update_area_names.insert(pair.first);
    }
    for (const std::string& area_name : to_update_area_names) {
        Area& area = area_by_name[area_name];
        int num_first_winners = project_into(area, area_in[area_name], verbose);
        area.num_first_winners = num_first_winners;
        if (save_winners) {
            area.saved_winners.push_back(area._new_winners);
        }
    }

    for (const std::string& area_name : to_update_area_names) {
        Area& area = area_by_name[area_name];
        area.update_winners();
        if (save_size) {
            area.saved_w.push_back(area.w);
        }
    }
}

int Brain::project_into(Area& target_area, const vector<string>& from_areas, int verbose) {
    int new_winners_count; // num_first_winners_processed
    int processed_input_count; // num_inputs_processed
    vector<vector<int>> connections_by_new_winner; // inputs_by_first_winner_index

    string target_area_name = target_area.name;
    for (const auto& source_area_name : from_areas) { // from_area_name -> source_area_name
        if (area_by_name[source_area_name].winners.empty() || area_by_name[source_area_name].w == 0) {
            cout << "Projecting from area with no assembly: " << source_area_name << endl;
            return -1;
        }
    }

    if (target_area.fixed_assembly) {
        target_area._new_winners = target_area.winners;
        target_area._new_w = target_area.w;
        new_winners_count = 0;
    }
    else {
        vector<int> winners_per_source_area; // input_size_by_from_area_index
        vector<int> cumulative_winners_per_source_area; // cumulative_input_size_by_from_area_index
        int total_source_areas; // num_inputs
        int total_winners; // total_k
        vector<float> previous_winners_input(target_area.w, 0.0f); // prev_winner_inputs
        vector<float> all_potential_winner_inputs;


        for (int source_area_idx = 0; source_area_idx < from_areas.size(); ++source_area_idx) {
            const auto& source_area_name = from_areas[source_area_idx];
            auto& connectome = connectomes[source_area_name][target_area_name].sym;
            auto& winners = area_by_name[source_area_name].winners;

            for (const auto& winner : winners) {
                for (size_t i = 0; i < previous_winners_input.size(); ++i) {

                    previous_winners_input[i] += connectome[winner][i];
                }
            }
        }

        if (!target_area.explicit_area) {
            float effective_neurons, quantile_value, alpha_value, mean, stddev, lower_bound, upper_bound;
            vector<float> potential_new_winner_inputs(target_area.k);
            total_source_areas = 0;
            total_winners = 0;
            cumulative_winners_per_source_area.push_back(total_winners);
            for (const auto& source_area_name : from_areas) {
                int active_winners = area_by_name[source_area_name].winners.size(); // effective_k
                winners_per_source_area.push_back(active_winners);
                total_source_areas += 1;
                total_winners += active_winners;
                cumulative_winners_per_source_area.push_back(total_winners);
            }

            effective_neurons = target_area.n - target_area.w;
            if (effective_neurons <= target_area.k) {
                cout << "Remaining size of area " << target_area_name << " too small to sample k new winners." << endl;
                return -1;
            }

            quantile_value = (effective_neurons - target_area.k) / effective_neurons;
            alpha_value = BinomQuantile(total_winners, p, quantile_value);
            mean = total_winners * p;
            stddev = sqrt(total_winners * p * (1.0 - p));
            lower_bound = (alpha_value - mean) / stddev;

            for (auto& input : potential_new_winner_inputs)
                input = min<float>(total_winners, round(TruncatedNorm(lower_bound, rng) * stddev + mean));

            all_potential_winner_inputs.resize(previous_winners_input.size() + potential_new_winner_inputs.size());
            copy(previous_winners_input.begin(), previous_winners_input.end(), all_potential_winner_inputs.begin());
            copy(potential_new_winner_inputs.begin(), potential_new_winner_inputs.end(), all_potential_winner_inputs.begin() + previous_winners_input.size());
        }
        else {
            all_potential_winner_inputs.assign(previous_winners_input.begin(), previous_winners_input.end());
        }

        std::vector<int> new_winner_indices = find_largest_indices(all_potential_winner_inputs, target_area.k);
        vector<float> initial_winner_inputs; // first_winner_inputs
        new_winners_count = 0;
        if (!target_area.explicit_area) {
            int new_winner_indices_len = new_winner_indices.size();
            for (int i = 0; i < new_winner_indices_len; i++) {
                if (new_winner_indices[i] >= target_area.w) {
                    initial_winner_inputs.push_back(all_potential_winner_inputs[new_winner_indices[i]]);
                    new_winner_indices[i] = target_area.w + new_winners_count;
                    ++new_winners_count;
                }
            }
        }

        target_area._new_winners = new_winner_indices;
        target_area._new_w = target_area.w + new_winners_count;
        connections_by_new_winner = vector<vector<int>>(new_winners_count, vector<int>());

        for (auto i = 0; i < new_winners_count; i++) {
            vector<int> input_indices = choose_random_indices(total_winners, initial_winner_inputs[i], rng);
            vector<int> connections_per_source_area(total_source_areas, 0); // num_connections_by_input_index

            for (auto j = 0; j < total_source_areas; j++) {
                for (const auto& winner : input_indices) {
                    if (cumulative_winners_per_source_area[j + 1] > winner && winner >= cumulative_winners_per_source_area[j])
                        connections_per_source_area[j] += 1;
                }
            }
            connections_by_new_winner[i] = connections_per_source_area;
        }
    }

    processed_input_count = 0;
    for (const auto& source_area_name : from_areas) {
        int& source_area_w = area_by_name[source_area_name].w;
        vector<int>& source_area_winners = area_by_name[source_area_name].winners;
        Fiber& source_to_target_connectome = connectomes[source_area_name][target_area_name];

        source_to_target_connectome.Colpadding(new_winners_count, 0);

        for (int i = 0; i < new_winners_count; ++i) {
            int total_connections = connections_by_new_winner[i][processed_input_count]; // total_in
            vector<int> selected_indices = sample_indices_func(source_area_winners, total_connections, rng); // sample_indices
            for(auto& idx : selected_indices)
                source_to_target_connectome.sym[idx][target_area.w + i] = 1.0;

            for (int j = 0; j < source_area_w; ++j)
                if (find(source_area_winners.begin(), source_area_winners.end(), j) == source_area_winners.end())
                    source_to_target_connectome.sym[j][target_area.w + i] = std::binomial_distribution<int>(1, p)(rng);
        }

        float area_beta = disable_plasticity ? 0 : target_area.beta_by_area[source_area_name]; // area_to_area_beta

        for (const auto& i : target_area._new_winners)
            for (const auto& j : source_area_winners)
                source_to_target_connectome.sym[j][i] *= 1.0 + area_beta;
        processed_input_count++;
    }

    for (auto& pair : area_by_name) {
        const std::string& other_area_name = pair.first;
        if (find(from_areas.begin(), from_areas.end(), other_area_name) == from_areas.end()) {
            Fiber& other_to_target_connectome = connectomes[other_area_name][target_area_name];

            other_to_target_connectome.Colpadding(new_winners_count, 0);
            for (int i = 0; i < other_to_target_connectome.rowcount; ++i) {
                for (int j = target_area.w; j < target_area._new_w; ++j) {
                    other_to_target_connectome.sym[i][j] = std::binomial_distribution<int>(1, p)(rng);
                }
            }
        }
        Fiber& target_to_other_connectome = connectomes[target_area_name][other_area_name];
        target_to_other_connectome.Rowpadding(new_winners_count, 0);
        for (int i = target_area.w; i < target_area._new_w; ++i) {
            for (int j = 0; j < target_to_other_connectome.colcount; ++j) {
                target_to_other_connectome.sym[i][j] = std::binomial_distribution<int>(1, p)(rng);
            }
        }
    }
    return new_winners_count;
}




int Brain::project_into(Area& target_area, const vector<string>& from_areas){
    int num_first_winners_processed;           // 首次获胜者数量
    int num_inputs_processed;                  // 表示当前处理到的输入来源
    vector<vector<uint32_t>> inputs_by_first_winner_index;      // 表示每个首次获胜者的输入信息
    string target_area_name = target_area.name;
    for(const auto& from_area_name : from_areas){
        if(area_by_name[from_area_name].winners.empty() || area_by_name[from_area_name].w == 0){
            cout<<"Projecting from area with no assembly: "<<from_area_name<<endl;
            return -1;
        }
    }

    
    if(target_area.fixed_assembly){
        target_area.new_winners = target_area.winners; 
        target_area.new_w = target_area.w;
        num_first_winners_processed = 0;
    }
    else{
        vector<uint32_t>  input_size_by_from_area_index;    // 每个脑区的输入神经元数
        uint32_t num_inputs;                                // 记录总的输入数量
        uint32_t total_k;                                        // 记录总输入神经元数量   
        vector<float> prev_winner_inputs(target_area.w, 0.0f);      // 上一轮胜者神经元在本轮的输入
        vector<float> all_potential_winner_inputs;          // 所有潜在胜者的神经元输入；非显示时为上一轮胜者的输入和潜在胜者的输入的拼接
        // 计算上一轮所有脑区在目的脑区上的获胜神经元输入
        for(const auto & from_area_name : from_areas){
            vector<vector<float>> connectome = connectomes[from_area_name][target_area_name].weights;
            for(const auto & w : area_by_name[from_area_name].winners){
                for(auto i = 0; i < prev_winner_inputs.size(); i++){
                    prev_winner_inputs[i] += connectome[w][i];
                }
            }
        }
        
        // 如果某个区域（area）不是显式指定的，那么将模拟该区域可能的新获胜者（potential new winners），数量为k
        if(!target_area.explicit_area){
            float effective_n, quantile, alpha, mu, std, a, b;
            vector<float> potential_new_winner_inputs(target_area.k);

            num_inputs = 0;                                //记录总的输入数量
            for(const auto& from_area_name : from_areas){
                uint32_t effective_k = area_by_name[from_area_name].winners.size();
                input_size_by_from_area_index.push_back(effective_k);
                num_inputs += 1;
            }

            total_k = 0;
            for(const auto& num : input_size_by_from_area_index)
                total_k += num;
            
            effective_n = target_area.n - target_area.w;
            if(effective_n <= target_area.k){
                cout<<"Remaining size of area "<<target_area_name<<" too small to sample k new winners."<<endl;
                return -1;
            }

            quantile = (effective_n - target_area.k) / effective_n;
            alpha = BinomQuantile(total_k, p, quantile);
            mu = total_k * p;                                  // 均值mu是所有输入中预期成功的数量，因为总共total_k个神经元参与，每个有p的概率根当前区域的w个连接上。建立的连接平均未total_k * p
            std = sqrt(total_k * p * (1.0 - p));          // 同上，二项分布的标准差
            a = (alpha - mu) / std;
            // 从截断的正态分布中随机生成target_area.k个样本，值均大于alpha，代表了潜在新获胜者的输入值
            for (auto& input: potential_new_winner_inputs)
                input = min<float>(total_k, round(TruncatedNorm(a, rng) * std + mu));       // 由于原版本的截断函数无上限，这里扔掉超上限部分

            // 拼接
            all_potential_winner_inputs.resize(prev_winner_inputs.size() + potential_new_winner_inputs.size());
            copy(prev_winner_inputs.begin(), prev_winner_inputs.end(), all_potential_winner_inputs.begin()); 
            copy(potential_new_winner_inputs.begin(), potential_new_winner_inputs.end(), all_potential_winner_inputs.begin() + prev_winner_inputs.size()); 

        }  
        else {
            all_potential_winner_inputs.assign(prev_winner_inputs.begin(), prev_winner_inputs.end());
        }

        unordered_set<uint32_t>  new_winner_indices = getTopK(all_potential_winner_inputs , target_area.k);     // 获取前k的最大索引
        vector<float> first_winner_inputs;          // 保存了第i个首次获胜者的当前输入值
        num_first_winners_processed = 0;
        // 不显示指定区域，会可能产生首次获胜者
        if(!target_area.explicit_area){
            // 如果获胜者索引大于或等于目标区域的曾经获胜者数量，那么它是在当前区域首次获胜者。
            // 原因： 曾经winner的索引必然位于前w个，因为更新是连续的
            unordered_set<uint32_t> temp;
            for(float index : new_winner_indices){
                if(index >= target_area.w){
                    first_winner_inputs.push_back(all_potential_winner_inputs[index]);
                    temp.insert(target_area.w + num_first_winners_processed);
                    num_first_winners_processed += 1;
                }
                else{
                    temp.insert(index);
                }
            }
            new_winner_indices = temp;
        } 

        target_area.new_winners = new_winner_indices;                       // 新获胜者的索引
        target_area.new_w = target_area.w + num_first_winners_processed;    // 呼应前面“曾经获胜者数量连续更新
        inputs_by_first_winner_index = vector<vector<uint32_t>>(num_first_winners_processed, vector<uint32_t>());       // 二维，[i][j]表示第i个首次获胜者的输入中，由第j个来源提供的神经元数量

        for(auto i = 0; i < num_first_winners_processed; i++){
            // 从total_k个神经元中，随机选择first_winner_inputs[i]个作为 输入的神经元，他们共同提供了first_winner_inputs[i]的总输入
            // input_indices是他们的索引
            vector<uint32_t> input_indices = choice(total_k, first_winner_inputs[i], rng);
            vector<uint32_t> num_connections_by_input_index(num_inputs, 0);         // 存储每个输入来源提供了多少神经元(连接)给当前神经元
            uint32_t total_so_far = 0;
            for(auto j = 0 ; j < num_inputs; j++){
                // 如果w处于total_so_far到total_so_far+当前来源的神经元数量，那么w就是由当前来源提供的，sum统计这样的w的数量，作为当前来源提供的神经元数量
                // 因为w是索引在0到total_k之间，而每个来源会提供total_k中连续的多个神经元
                for(auto w : input_indices){
                    if(total_so_far + input_size_by_from_area_index[j] > w && w >= total_so_far)
                        num_connections_by_input_index[j] += 1;
                }
                total_so_far += input_size_by_from_area_index[j];
            }
            inputs_by_first_winner_index[i] = num_connections_by_input_index;
        }      
    }

    
    // 为每个输入区域到目标区域的connectomes：
    // 添加 num_first_winners_processed 列，
    // 对于 num_first_winners_processed 中的每个首胜神经元，为选中的神经元填充 (1+beta)
    // 对于每个重复获胜者 i ，每个 输入区域的获胜者j 中，connectome[j][i] 乘以 (1+beta)
    num_inputs_processed = 0;
    for(const auto& from_area_name : from_areas){
        float from_area_w = area_by_name[from_area_name].w;
        unordered_set<uint32_t> from_area_winners = area_by_name[from_area_name].winners;
        Fiber& the_connectome = connectomes[from_area_name][target_area_name];

        // the_connectome 扩展 num_first_winners_processed 列
        the_connectome.paddingCol(num_first_winners_processed);

        // 对每个首胜神经元，初始化有关的the_connectome初始值
        for(uint32_t i = 0; i < num_first_winners_processed; ++i){
            // 当前输入源 对 首胜神经元提供的神经元连接数
            uint32_t total_in = inputs_by_first_winner_index[i][num_inputs_processed];  
            // 从当前区域的winners采样total_in个作为提供的神经元，索引形式
            vector<uint32_t> sample_indices = choice(from_area_winners, total_in, rng); 
            // 每个索引对目的区域的新神经元建立连接
            for(auto& j : sample_indices)                                           
                the_connectome.weights[j][target_area.w + i] = 1.0;
            // 当前区域中曾经胜者中的非winner神经元，随机连接
            for(uint32_t j = 0; j < from_area_w; ++j)
                if(from_area_winners.find(j) == from_area_winners.end())
                    the_connectome.weights[j][target_area.w + i] = genBinomialNum(rng, 1, p);     
        }
        // 若readout禁止可塑性，则beta为0，其意为不更新        
        float area_to_area_beta = disable_plasticity? 0 : target_area.beta_by_area[from_area_name];
        // 更新两个区域winner之间的连接权值，注意用new_winners
        for(const auto& i : target_area.new_winners) 
            for(const auto& j : from_area_winners)
                the_connectome.weights[j][i] *= 1.0 + area_to_area_beta;

        num_inputs_processed++;
    }

    // 扩展未参与本次投影的其他区域到目标区域的connectomes
    // 也拓展目标区域到上述区域的connectomes
    for(const auto& other_area_name : views::keys(area_by_name)){
        if(find(from_areas.begin(), from_areas.end(), other_area_name) == from_areas.end()) {
            Fiber& the_other_area_connectome = connectomes[other_area_name][target_area_name];

            // the_other_area_connectome 扩展 num_first_winners_processed 列
            the_other_area_connectome.paddingCol(num_first_winners_processed);
            // 对扩展的的神经元和首胜神经元之间的突触权值用二项分布随机初始化
            for (uint32_t i = 0; i < the_other_area_connectome.rows; ++i) {
                for(uint32_t j = target_area.w; j < target_area.new_w; ++j) {
                    the_other_area_connectome.weights[i][j] = genBinomialNum(rng, 1, p);
                }
            }
        }

        // connectomes[target_area_name][other_area_name] 添加 num_first_winners_processed 行, 二项分布概率p进行初始化
        Fiber& connectome = connectomes[target_area_name][other_area_name];
        connectome.paddingRow(num_first_winners_processed);

        for(uint32_t i = target_area.w; i < target_area.new_w; ++i){
            for(uint32_t j = 0; j < connectome.cols; ++j) {
                connectome.weights[i][j] = genBinomialNum(rng, 1, p);
            }
        }
    }

    return num_first_winners_processed;
}