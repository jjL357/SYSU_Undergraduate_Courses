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
#include <queue>
#include <numeric> 
#include <stdexcept>
#include <iterator>
#include <iostream>
#include <unordered_map>
#include <unordered_set>



using namespace std;

// 函数：生成二项分布的随机数，并填充指定的二维数组区域（浮点类型）
std::vector<std::vector<float>> fill_binomial(int start_col, double p, int num_rows, int num_cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(p);
    std::vector<std::vector<float>> matrix(num_rows, std::vector<float>(start_col + num_cols, 0.0f)); // 初始化矩阵
    for (int i = 0; i < num_rows; ++i) {
        for (int j = start_col; j < start_col + num_cols; ++j) {
            matrix[i][j] = static_cast<float>(d(gen)); // 生成二项分布的随机数并填充矩阵
        }
    }
    return matrix;
}

// 截断正态分布采样
template<typename Trng>
float TruncatedNorm(float a, Trng& _rng) {
    if (a <= 0.0f) {
        std::normal_distribution<float> norm(0.0f, 1.0f);
        for (;;) {
            const float x = norm(_rng);
            if (x >= a) return x;
        }
    } else {
        const float alpha = (a + std::sqrt(a * a + 4)) * 0.5f;
        std::exponential_distribution<float> d(alpha);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (;;) {
            const float z = a + d(_rng);
            const float dz = z - alpha;
            const float rho = std::exp(-0.5f * dz * dz);
            if (u(_rng) < rho) return z;
        }
    }
}

// 函数：对二维向量进行填充
std::vector<std::vector<float>> pad_vector(std::vector<std::vector<float>>& input, int pad_rows, int pad_cols) {
    std::vector<std::vector<float>> padded;
    int original_rows = input.size();
    if (original_rows == 0) return padded;
    int original_cols = input[0].size();
    int new_rows = original_rows + pad_rows;
    int new_cols = original_cols + pad_cols;

    // 创建一个新的二维向量，并初始化为0
    padded.resize(new_rows, std::vector<float>(new_cols, 0.0f));

    // 复制原始向量的值到新的二维向量中
    for (int i = 0; i < original_rows; ++i) {
        for (int j = 0; j < original_cols; ++j) {
            padded[i][j] = input[i][j];
        }
    }
    return padded;
}

// 函数：对数组进行填充
std::vector<std::vector<float>> padArray(const std::vector<std::vector<float>>& array, int num_first_winners_processed) {
    int originalRows = array.size();
    std::vector<std::vector<float>> paddedArray(originalRows);
    if (originalRows == 0) return paddedArray;
    int originalCols = array[0].size();
    for (int i = 0; i < originalRows; ++i) {
        paddedArray[i].resize(array[i].size() + num_first_winners_processed, 0);
        for (int j = 0; j < originalCols; ++j) {
            paddedArray[i][j] = array[i][j];
        }
    }
    return paddedArray;
}

// 函数：计算二项分布的分位数
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

// 函数：计算二项分布的分位点函数
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

// 生成无放回的随机选择
std::vector<int> uniqueRandomChoices(int total_k, int num_choices, std::mt19937& _rng) {
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
    int n = inputs.size();
    std::vector<std::pair<int, float>> indexed_values(n);
    for (int i = 0; i < n; ++i) {
        indexed_values[i] = std::make_pair(i, inputs[i]);
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



struct BinomialParams {
    int n; // 试验次数
    double p; // 成功概率
};


class Connectome { // 主要是为了方便获取列数(例如vector nums中为空能以获得列数nums[0].size())
    public:
        int row; // 行数
        int col; // 列数
        vector<vector<float>> con; // 存储连接矩阵的二维向量

        Connectome() {};

        // 函数：对连接矩阵增加新行，并填充指定的值
        void rowpadding(int new_row, float value) {
            for (int i = 0; i < new_row; i++) {
                vector<float> tmp(col, value); // 创建新行并填充指定值
                con.push_back(tmp); // 将新行添加到矩阵中
            }
            row += new_row; // 更新行数
        }

        // 函数：对连接矩阵增加新列，并填充指定的值
        void colpadding(int new_col, float value) {
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < new_col; j++) {
                    con[i].push_back(value); // 在每行末尾添加指定数量的新列，并填充指定值
                }
            }
            col += new_col; // 更新列数
        }
};

class Area {
public:
    // 默认构造函数：初始化所有成员变量
    Area() : name(""), n(0), k(0), beta(0.05), w(0), _new_w(0), num_first_winners(-1), fixed_assembly(false), explicitArea(false), num_ever_fired(0) {}

    // 参数化构造函数：初始化成员变量，并根据参数进行赋值
    Area(const std::string& name, int n, int k, double beta = 0.05, int w = 0, bool explicitArea = false)
        : name(name), n(n), k(k), beta(beta), w(w), _new_w(0), num_first_winners(-1), fixed_assembly(false), explicitArea(explicitArea), num_ever_fired(0) {}

    // 更新赢家（winners）的函数
    void update_winners() {
        winners = _new_winners; // 更新赢家列表
        if (!explicitArea) { // 如果不是显式区域
            w = _new_w; 
        }
    }

    // 根据刺激更新 beta 值的函数
    void update_beta_by_stimulus(const std::string& name, double new_beta) {
        beta_by_stimulus[name] = new_beta; // 更新刺激的 beta 值
    }

    // 更新区域 beta 值的函数
    void update_area_beta(const std::string& name, double new_beta) {
        beta_by_area[name] = new_beta; // 更新区域的 beta 值
    }

    // 固定集合状态的函数
    void fix_assembly() {
        if (winners.empty()) { // 如果赢家列表为空
            throw std::runtime_error("Area " + name + " does not have assembly; cannot fix."); // 抛出异常
        }
        fixed_assembly = true; // 固定集合状态
    }

    // 取消固定集合状态的函数
    void unfix_assembly() {
        fixed_assembly = false; // 取消固定集合状态
    }

    // 获取曾经激活过的神经元数量的函数
    int getNumEverFired() const {
        if (explicitArea) { 
            return num_ever_fired; 
        } else {
            return w; 
        }
    }

    // 成员变量
    std::string name; // 区域名称
    int n; // 总神经元数
    int k; // 激活神经元数
    double beta; 
    std::unordered_map<std::string, double> beta_by_stimulus; // 刺激对应的 beta 值映射
    std::unordered_map<std::string, double> beta_by_area; // 区域对应的 beta 值映射
    int w; 
    int _new_w; 
    std::vector<int> saved_w; 
    std::vector<int> winners; // 当前赢家列表
    std::vector<int> _new_winners; // 更新后的赢家列表
    std::vector<std::vector<int>> saved_winners; // 保存的赢家列表
    int num_first_winners; 
    bool fixed_assembly; // 集合状态是否固定
    bool explicitArea; // 是否是显式区域
    int num_ever_fired; 
    std::vector<bool> ever_fired; 
};



class Brain {
public:
    // 存储不同区域的名称和对应的 Area 对象
    std::unordered_map<std::string, Area> area_by_name;

    // 存储每个刺激名称及其大小
    std::unordered_map<std::string, int> stimulus_size_by_name;

    // 存储每个刺激对应的激活向量，按区域名称映射
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<float>>> connectomes_by_stimulus;

    // 存储区域之间的连接，从源区域名称到目标区域名称到连接对象的映射
    std::unordered_map<std::string, std::unordered_map<std::string, Connectome>> connectomes;

    // 连接概率
    double p;
    bool save_size;
    bool save_winners;
    bool disable_plasticity;
    std::mt19937 _rng;
    std::uniform_real_distribution<float> _uniform_dist;
    bool _use_normal_ppf;

  
    Brain(double p, bool save_size = true, bool save_winners = false, int seed = 0)
        : p(p), save_size(save_size), save_winners(save_winners), disable_plasticity(false), _rng(seed), _use_normal_ppf(false) {}

    void add_area(const std::string& area_name, int n, int k, float beta) {
    // 向 area_by_name 添加新的区域
    area_by_name[area_name] = Area(area_name, n, k, beta);
    // 创建一个新的 connectomes 映射用于存储连接信息
    std::unordered_map<std::string, Connectome> new_connectomes;
    
    // 遍历所有已存在的区域
    for (auto& kv : area_by_name) {
        const std::string& other_area_name = kv.first;
        // 获取当前遍历区域的大小（如果是显式区域）
        int other_area_size = area_by_name[other_area_name].explicitArea ? area_by_name[other_area_name].n : 0;
        // 初始化 new_connectomes 中当前遍历区域的连接矩阵大小
        new_connectomes[other_area_name].col = other_area_size;
        new_connectomes[other_area_name].row = 0;
        // 如果当前遍历区域不是新添加的区域
        if (other_area_name != area_name) {
            // 初始化 connectomes 映射中从当前遍历区域到新添加区域的连接矩阵大小
            connectomes[other_area_name][area_name].con.resize(other_area_size, std::vector<float>(0));
            connectomes[other_area_name][area_name].row = other_area_size;
            connectomes[other_area_name][area_name].col = 0;
        }
        // 设置当前遍历区域到新添加区域的 beta 值
        area_by_name[other_area_name].beta_by_area[area_name] = area_by_name[other_area_name].beta;
        area_by_name[area_name].beta_by_area[other_area_name] = beta;
    }
    // 将新的 connectomes 映射添加到 connectomes 中
    connectomes[area_name] = new_connectomes;
    }


    void add_explicit_area(const std::string& area_name,
                       int n, int k, float beta,
                       float custom_inner_p = -1,
                       float custom_out_p = -1,
                       float custom_in_p = -1) {
    // 创建一个显式区域，并将其添加到 area_by_name 映射中
    area_by_name[area_name] = Area(area_name, n, k, beta, n, true);
    // 初始化显式区域的 ever_fired 和 num_ever_fired 属性
    area_by_name[area_name].ever_fired = std::vector<bool>(n, false);
    area_by_name[area_name].num_ever_fired = 0;

    // 根据参数或默认值设置 inner_p、in_p 和 out_p
    float inner_p = (custom_inner_p != -1) ? custom_inner_p : p;
    float in_p = (custom_in_p != -1) ? custom_in_p : p;
    float out_p = (custom_out_p != -1) ? custom_out_p : p;
    // 创建一个新的 connectomes 映射，用于存储连接信息
    std::unordered_map<std::string, Connectome> new_connectomes;
    // 遍历所有已存在的区域
    for (auto& kv : area_by_name) {
        const std::string& other_area_name = kv.first;
        Area& other_area = kv.second;
        // 如果是当前新添加的区域
        if (other_area_name == area_name) {
            // 初始化新添加区域到自身的连接矩阵
            new_connectomes[other_area_name].con = std::vector<std::vector<float>>(n, std::vector<float>(n));
            new_connectomes[other_area_name].col = n;
            new_connectomes[other_area_name].row = n;
            // 使用二项分布填充连接矩阵
            std::binomial_distribution<int> distribution(1, inner_p);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    new_connectomes[other_area_name].con[i][j] = distribution(_rng);
                }
            }
        }
        else {
            // 如果当前遍历的区域也是显式区域
            if (other_area.explicitArea) {
                int other_n = other_area.n;
                // 初始化新添加区域到当前遍历区域的连接矩阵
                new_connectomes[other_area_name].con = std::vector<std::vector<float>>(n, std::vector<float>(other_n));
                new_connectomes[other_area_name].row = n;
                new_connectomes[other_area_name].col = other_n;
                // 使用二项分布填充连接矩阵
                std::binomial_distribution<int> distribution(1, out_p);
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < other_n; ++j) {
                        new_connectomes[other_area_name].con[i][j] = distribution(_rng);
                    }
                }
                // 初始化当前遍历区域到新添加区域的连接矩阵
                connectomes[other_area_name][area_name].con = std::vector<std::vector<float>>(other_n, std::vector<float>(n));
                distribution = std::binomial_distribution<int>(1, in_p);
                for (int i = 0; i < other_n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        connectomes[other_area_name][area_name].con[i][j] = distribution(_rng);
                    }
                }
            }
            else {
                // 如果当前遍历的区域不是显式区域，清空其连接信息
                new_connectomes[other_area_name].con.clear();
                connectomes[other_area_name][area_name].con.clear();
            }
        }
        // 设置其他区域到新添加区域的 beta 值
        other_area.beta_by_area[area_name] = other_area.beta;
        // 设置新添加区域到其他区域的 beta 值
        area_by_name[area_name].beta_by_area[other_area_name] = beta;
    }
    // 将新的 connectomes 映射添加到 connectomes 中
    connectomes[area_name] = new_connectomes;
    }


    void update_plasticity(const std::string& from_area, const std::string& to_area, double new_beta) {
        area_by_name[to_area].beta_by_area[from_area] = new_beta;
    }

    void update_plasticities(const std::unordered_map<std::string, std::vector<std::pair<std::string, float>>>& area_update_map = {}) {
        // Update plasticities from area to area
        for (const auto& kv : area_update_map) {
            const std::string& to_area = kv.first;
            for (const auto& update_rule : kv.second) {
                const std::string& from_area = update_rule.first;
                double new_beta = update_rule.second;
                update_plasticity(from_area, to_area, new_beta);
            }
            }
    }

   void activate(const std::string& area_name, int index) {
    // 激活指定区域的一个特定索引处的神经元集合（一个集合）
    Area& area = area_by_name[area_name]; 
    int k = area.k; 
    int assembly_start = k * index; 
    area.winners.clear(); 
    for (int i = assembly_start; i < assembly_start + k; ++i) {
        area.winners.push_back(i); // 将集合中的神经元添加到获胜神经元列表中
    }
    area.fix_assembly(); 
    }

    void project(const std::unordered_map<std::string, std::vector<std::string>>& areas_by_stim,
        const std::unordered_map<std::string, std::vector<std::string>>& dst_areas_by_src_area)
    {   
        // area_in: 映射，从源区域到投影到它的目标区域的列表
        std::unordered_map<std::string, std::vector<std::string>> area_in;

        for (auto& pair : dst_areas_by_src_area) {
            std::string from_area_name = pair.first;
            vector<string> tmp =  pair.second;
            if (area_by_name.find(from_area_name) == area_by_name.end()) {
                throw std::invalid_argument(from_area_name + " not in brain.area_by_name");
            }
            for (std::string& to_area_name : tmp) {
                if (area_by_name.find(to_area_name) == area_by_name.end()) {
                    throw std::invalid_argument("Not in brain.area_by_name: " + to_area_name);
                }
            area_in[to_area_name].push_back(from_area_name);
            }
        }
        // 需要更新的目标区域名称集合
        std::set<std::string> to_update_area_names;
        for (const auto& pair : area_in) {
            to_update_area_names.insert(pair.first);
        }
         // 第一次遍历：将源区域投影到目标区域，并更新第一组获胜者
        for (auto& area_name : to_update_area_names) {
            Area& area = area_by_name[area_name];
            int num_first_winners = project_into(area, area_in[area_name]);
            area.num_first_winners = num_first_winners;
            if (save_winners) {
                area.saved_winners.push_back(area._new_winners);
            }
        }
         // 第二次遍历：更新获胜者并保存区域大小
        for (auto& area_name : to_update_area_names) {
            Area& area = area_by_name[area_name];
            area.update_winners();
            if (save_size) {
                area.saved_w.push_back(area.w);
            }
        }
    }

    int project_into(Area& target_area, const vector<string>& from_areas) {
        
        int num_inputs_processed;
        int processed_input_count; 

        // If projecting from area with no assembly, throw an error.
        for (const auto& from_area_name : from_areas) {
            auto& from_area = area_by_name[from_area_name];
            if (from_area.winners.empty() || from_area.w == 0) {
                throw std::runtime_error("Projecting from area with no assembly: " + from_area_name);
            }
        }

        vector<vector<int>> inputs_by_first_winner_index; 
        string target_area_name = target_area.name;

        if (target_area.fixed_assembly) {
            target_area._new_winners = target_area.winners;
            target_area._new_w = target_area.w;
            num_inputs_processed = 0;
        }
        else {
            
            vector<int> cumulative_winners_per_source_area; 
            int total_source_areas; 
            int total_winners; 
            vector<float> previous_winners_input(target_area.w, 0.0f);
            vector<float> all_potential_winner_inputs;

                #pragma omp parallel for
             for (const auto& from_area_name : from_areas) {
                auto& connectome = connectomes[from_area_name][target_area_name].con;
                auto& from_area = area_by_name[from_area_name];
                for (int w : from_area.winners) {
                    for (size_t i = 0; i < target_area.w; ++i) {
                        #pragma omp atomic
                        previous_winners_input[i] += connectome[w][i];
                    }
                }
            }

            if (!target_area.explicitArea) {
                float effective_neurons, quantile_value, alpha_value, mean, stddev, lower_bound, upper_bound;
                vector<float> potential_new_winner_inputs(target_area.k);
                total_source_areas = 0;
                total_winners = 0;
                cumulative_winners_per_source_area.push_back(total_winners);
        
                for (const auto& from_area_name : from_areas) {
                    int active_winners = area_by_name[from_area_name].winners.size(); // effective_k
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
                #pragma omp parallel for
                for (auto& input : potential_new_winner_inputs)
                    input = min<float>(total_winners, round(TruncatedNorm(lower_bound, _rng) * stddev + mean));

                all_potential_winner_inputs.resize(previous_winners_input.size() + potential_new_winner_inputs.size());
                copy(previous_winners_input.begin(), previous_winners_input.end(), all_potential_winner_inputs.begin());
                copy(potential_new_winner_inputs.begin(), potential_new_winner_inputs.end(), all_potential_winner_inputs.begin() + previous_winners_input.size());
            }
            else {
                all_potential_winner_inputs = previous_winners_input;
            }

            std::vector<int> new_winner_indices = getNLargestIndices(all_potential_winner_inputs, target_area.k);
            vector<float> initial_winner_inputs; 
            num_inputs_processed = 0;
            if (!target_area.explicitArea) {
                int new_winner_indices_len = new_winner_indices.size();
                for (int i = 0; i < new_winner_indices_len; i++) {
                    if (new_winner_indices[i] >= target_area.w) {
                        initial_winner_inputs.push_back(all_potential_winner_inputs[new_winner_indices[i]]);
                        new_winner_indices[i] = target_area.w + num_inputs_processed;
                        ++num_inputs_processed;
                    }
                }
            }

            target_area._new_winners = new_winner_indices;
            target_area._new_w = target_area.w + num_inputs_processed;
            inputs_by_first_winner_index = vector<vector<int>>(num_inputs_processed, vector<int>());

            for (auto i = 0; i < num_inputs_processed; i++) {
                vector<int> input_indices = uniqueRandomChoices(total_winners, int(initial_winner_inputs[i]),_rng);
                vector<int> connections_per_source_area(total_source_areas, 0); 
                for (auto j = 0; j < total_source_areas; j++) {
                    #pragma omp parallel for
                    for (const auto& winner : input_indices) {
                        if (cumulative_winners_per_source_area[j + 1] > winner && winner >= cumulative_winners_per_source_area[j])
                              #pragma omp atomic
                              connections_per_source_area[j] += 1;
                    }
                }
                inputs_by_first_winner_index[i] = connections_per_source_area;
            }
        }

        processed_input_count = 0;
        for(auto&from_area_name : from_areas) {
            int&  from_area_w = area_by_name[from_area_name].w;
            vector<int>& from_area_winners = area_by_name[from_area_name].winners;
            Connectome& the_connectomes = connectomes[from_area_name][target_area_name];

            the_connectomes.colpadding(num_inputs_processed, 0);
             #pragma omp parallel for
            for (int i = 0; i < num_inputs_processed; ++i) {
                int total_in = inputs_by_first_winner_index[i][processed_input_count]; 
                vector<int>sample_indices = random_choice( total_in,from_area_winners); 
                for(auto& j : sample_indices)
                    the_connectomes.con[j][target_area.w + i] = 1.0;

                for (int j = 0; j <  from_area_w; ++j)
                    if (find(from_area_winners.begin(), from_area_winners.end(), j) == from_area_winners.end())
                        the_connectomes.con[j][target_area.w + i] = std::binomial_distribution<int>(1, p)(_rng);
            }

            float area_to_area_beta = disable_plasticity ? 0 : target_area.beta_by_area[from_area_name]; // area_to_area_beta
             #pragma omp parallel for
            for (const auto& i : target_area._new_winners)
                for (const auto& j : from_area_winners)
                     #pragma omp atomic
                    the_connectomes.con[j][i] *= 1.0 + area_to_area_beta;
            processed_input_count++;
        }

        for (auto& kv : area_by_name) {
            const std::string& other_area_name = kv.first;
            Area& other_area = kv.second;
            if (find(from_areas.begin(), from_areas.end(), other_area_name) == from_areas.end()) {
                connectomes[other_area_name][target_area_name].colpadding(num_inputs_processed, 0);
                for (int i = 0; i < connectomes[other_area_name][target_area_name].row; ++i) {
                    for (int j = target_area.w; j < target_area._new_w; ++j) {
                        connectomes[other_area_name][target_area_name].con[i][j] = std::binomial_distribution<int>(1, p)(_rng);
                    }
                }
            }
            
            connectomes[target_area_name][other_area_name].rowpadding(num_inputs_processed, 0);
            #pragma omp parallel for
            for (int i = target_area.w; i < target_area._new_w; ++i) {
                for (int j = 0; j < connectomes[target_area_name][other_area_name].col; ++j) {
                    connectomes[target_area_name][other_area_name].con[i][j] = std::binomial_distribution<int>(1, p)(_rng);
                }
            }
        }
        return num_inputs_processed;
    }

private:

// 从输入向量中随机选择指定数量的元素索引
std::vector<int> random_choice(int num_choices,const std::vector<int>& input) {
    std::vector<int> indices(input.size());
    std::iota(indices.begin(), indices.end(), 0); 

    std::random_device rd;
    std::mt19937 _rng(rd());
    
    std::shuffle(indices.begin(), indices.end(), _rng);
    
    std::vector<int> result;
    for (int i = 0; i < num_choices; ++i) {
        result.push_back(input[indices[i]]);
    }
    
    return result;
}

};   
