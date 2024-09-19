#include "brain.h"

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

namespace nemo {
namespace {
// 二项分布的分位数(Binomial Quantile)
float BinomQuantile(uint32_t k, float p, float percent) {
  // 计算二项分布的概率
  double pi = std::pow(1.0 - p, k);
  // 计算系数
  double mul = (1.0 * p) / (1.0 - p);
  // 初始化总概率
  double total_p = pi;
  // 初始化计数器
  uint32_t i = 0;
  // 循环直到累积概率大于等于给定的百分比
  while (total_p < percent) {
    // 更新当前概率
    pi *= ((k - i) * mul) / (i + 1);
    // 累积总概率
    total_p += pi;
    // 更新计数器
    ++i;
  }
  // 返回计数器 i，即分位数
  return i;
}

// 截断正态分布的生成函数 TruncatedNorm，根据给定的参数 a 和随机数生成器对象 rng，生成满足截断正态分布的随机数
template<typename Trng>
float TruncatedNorm(float a, Trng& rng) {
  if (a <= 0.0f) {
    std::normal_distribution<float> norm(0.0f, 1.0f);  // 创建均值为0，标准差为1的正态分布对象
    for (;;) {
      const float x = norm(rng);  // 从正态分布中生成一个随机数
      if (x >= a) return x;  // 如果生成的随机数大于等于a，则返回该随机数
    }
  } else {
    // Robert提出的指数接受-拒绝算法，参考文献：https://arxiv.org/pdf/0907.4010.pdf
    const float alpha = (a + std::sqrt(a * a + 4)) * 0.5f;  // 计算指数分布的参数 alpha
    std::exponential_distribution<float> d(alpha);  // 创建参数为 alpha 的指数分布对象
    std::uniform_real_distribution<float> u(0.0f, 1.0f);  // 创建 [0, 1] 上均匀分布的随机数对象
    for (;;) {
      const float z = a + d(rng);  // 从指数分布中生成一个随机数 z
      const float dz = z - alpha;  // 计算 z 和 alpha 的差值
      const float rho = std::exp(-0.5f * dz * dz);  // 计算接受概率 rho
      if (u(rng) < rho) return z;  // 根据接受概率决定是否接受生成的随机数 z
    }
  }
}

// 生成突触的函数 GenerateSynapses，根据给定的支持度 support 和概率 p，使用几何分布进行抽样生成突触
template<typename Trng>
std::vector<Synapse> GenerateSynapses(uint32_t support, float p, Trng& rng) {
  std::vector<Synapse> synapses;  // 存储突触的向量
  // 从几何分布（geometric(p) distribution）中抽样，方法是从 floor(log(U[0, 1])/log(1-p)) 中抽样。
  // U[0, 1] 表示 [0, 1] 上均匀分布的随机数。

  std::uniform_real_distribution<float> u(0.0, 1.0);  // 创建 [0, 1] 上均匀分布的随机数对象
  const float scale = 1.0f / std::log(1 - p);  // 计算缩放因子 scale
  uint32_t last = std::floor(std::log(u(rng)) * scale);  // 从几何分布中抽样初始值 last
  synapses.reserve(support * p * 1.05);  // 预先分配 synapses 的容量，留有一定的冗余空间
  while (last < support) {
    synapses.push_back({last, 1.0f});  // 将新的突触添加到 synapses 中，权重为 1.0f
    last += 1 + std::floor(std::log(u(rng)) * scale);  // 更新 last 的值，继续抽样
  }
  return synapses;  // 返回生成的突触向量
}

// 选择前 k 个最高激活的突触的函数 SelectTopK
void SelectTopK(std::vector<Synapse>& activations, uint32_t k) {
  // 使用 std::nth_element 函数将 activations 中的元素按照权重降序排列，并选择前 k 个元素。
  std::nth_element(activations.begin(), activations.begin() + k - 1,
                   activations.end(),
                   [](const Synapse& a, const Synapse& b) {
                     // Lambda 表达式定义了比较函数，首先按照权重降序排列，如果权重相等则按照 neuron 升序排列。
                     if (a.weight != b.weight) return a.weight > b.weight;
                     return a.neuron < b.neuron;
                   });
  activations.resize(k);  // 调整 activations 的大小为 k，保留前 k 个元素。
}


}  // namespace

Brain::Brain(float p, float beta, float max_weight, uint32_t seed)
    : rng_(seed), p_(p), beta_(beta), learn_rate_(1.0f + beta_),
      max_weight_(max_weight), areas_(1, Area(0, 0, 0)),
      fibers_(1, Fiber(0, 0)), incoming_fibers_(1), outgoing_fibers_(1),
      area_name_(1, "INVALID") {}

// 增加区域
Area& Brain::AddArea(const std::string& name, uint32_t n, uint32_t k,
                     bool recurrent, bool is_explicit) {
  // 增加区域 区域的index为未加入前的区域数
  uint32_t area_i = areas_.size();
  areas_.push_back(Area(area_i, n, k));
  // 判断是否有重复区域名,有则报错
  if (area_by_name_.find(name) != area_by_name_.end()) {
    fprintf(stderr, "Duplicate area name %s\n", name.c_str());
  }
  // 若显示,则将support设为n
  if (is_explicit) {
    areas_[area_i].support = n;
  }
  // 构建index和区域名的映射 和 in out fibers
  area_by_name_[name] = area_i;
  area_name_.push_back(name);
  incoming_fibers_.push_back({});
  outgoing_fibers_.push_back({});
  // 是否子循环,即有该区域有fiber出去又连到自己
  if (recurrent) {
    AddFiber(name, name);
  }
  return areas_.back();
}


void Brain::AddStimulus(const std::string& name, uint32_t k) {
  // 调用 AddArea 函数添加一个新的区域，并设置其为显式区域（is_explicit=true），非递归区域（recurrent=false）。
  AddArea(name, k, k, /*recurrent=*/false, /*is_explicit=*/true);
  
  // 激活刚刚添加的区域中的第一个结合（assembly），集合的索引为 0。
  ActivateArea(name, 0);
}

//增加fiber
void Brain::AddFiber(const std::string& from, const std::string& to,
                     bool bidirectional) {
  const Area& area_from = GetArea(from);
  const Area& area_to = GetArea(to);
  uint32_t fiber_i = fibers_.size();
  Fiber fiber(area_from.index, area_to.index);
  incoming_fibers_[area_to.index].push_back(fiber_i);
  outgoing_fibers_[area_from.index].push_back(fiber_i);
  for (uint32_t i = 0; i < area_from.support; ++i) {
    std::vector<Synapse> synapses =
        GenerateSynapses(area_to.support, p_, rng_);
    fiber.outgoing_synapses.emplace_back(std::move(synapses));
  }
  fibers_.emplace_back(std::move(fiber));
  // 是否是双向的
  if (bidirectional) {
    AddFiber(to, from);
  }
}

// 获取对应区域名的Area
Area& Brain::GetArea(const std::string& name) {
  std::map<std::string, uint32_t>::iterator it = area_by_name_.find(name);
  if (it != area_by_name_.end()) {
    return areas_[it->second];
  }
  fprintf(stderr, "Invalid area name %s\n", name.c_str());
  return areas_[0];
}

// 获取对应区域名的Area
const Area& Brain::GetArea(const std::string& name) const {
  std::map<std::string, uint32_t>::const_iterator it = area_by_name_.find(name);
  if (it != area_by_name_.end()) {
    return areas_[it->second];
  }
  fprintf(stderr, "Invalid area name %s\n", name.c_str());
  return areas_[0];
}

// 获得从from到to的fiber
Fiber& Brain::GetFiber(const std::string& from, const std::string& to) {
  const Area& from_area = GetArea(from);
  const Area& to_area = GetArea(to);
  for (auto fiber_i : outgoing_fibers_[from_area.index]) {
    Fiber& fiber = fibers_[fiber_i];
    if (fiber.to_area == to_area.index) {
      return fiber;
    }
  }
  fprintf(stderr, "No fiber found from %s to %s\n", from.c_str(), to.c_str());
  return fibers_[0];
}

// 获得从from到to的fiber
const Fiber& Brain::GetFiber(const std::string& from,
                             const std::string& to) const{
  const Area& from_area = GetArea(from);
  const Area& to_area = GetArea(to);
  for (auto fiber_i : outgoing_fibers_[from_area.index]) {
    const Fiber& fiber = fibers_[fiber_i];
    if (fiber.to_area == to_area.index) {
      return fiber;
    }
  }
  fprintf(stderr, "No fiber found from %s to %s\n", from.c_str(), to.c_str());
  return fibers_[0];
}

// 抑制所有fiber
void Brain::InhibitAll() {
  for (Fiber& fiber : fibers_) {
    fiber.is_active = false;
  }
}

// 抑制特定的fiber
void Brain::InhibitFiber(const std::string& from, const std::string& to) {
  GetFiber(from, to).is_active = false;
}

// 激活特定的fiber
void Brain::ActivateFiber(const std::string& from, const std::string& to) {
  GetFiber(from, to).is_active = true;
}

// 激活指定区域中的特定集合(assembly)
void Brain::ActivateArea(const std::string& name, uint32_t assembly_index) {
  // 如果日志级别大于0，打印激活信息
  if (log_level_ > 0) {
    printf("Activating %s assembly %u\n", name.c_str(), assembly_index);
  }
  
  // 获取指定名称的区域
  Area& area = GetArea(name);
  
  // 计算集合在支持范围内的偏移量
  uint32_t offset = assembly_index * area.k;
  
  // 如果集合的结束位置超过了区域的支持上限，则打印错误信息并返回
  if (offset + area.k > area.support) {
    fprintf(stderr, "[Area %s] Could not activate assembly index %u "
            "(not enough support: %u vs %u)\n", name.c_str(), assembly_index,
            area.support, offset + area.k);
    return;
  }
  
  // 调整区域的激活向量大小为 k，并设置激活的神经元索引
  area.activated.resize(area.k);
  for (uint32_t i = 0; i < area.k; ++i) {
    area.activated[i] = offset + i;
  }
  
  // 将区域标记为固定（is_fixed=true）
  area.is_fixed = true;
}

// 模拟一个时间步长的神经网络激活和可塑性更新过程
void Brain::SimulateOneStep(bool update_plasticity) {
  // 如果日志级别大于0，打印当前步数信息
  if (log_level_ > 0) {
    if (step_ == 0 && log_level_ > 2) {
      LogGraphStats();  // 如果是第一步且日志级别大于2，记录图的统计信息
    }
    printf("Step %u%s\n", step_, update_plasticity ? "" : " (readout)");
  }

  // 用于存储新激活的神经元索引
  std::vector<std::vector<uint32_t>> new_activated(areas_.size());

  // 遍历所有区域
  for (uint32_t area_i = 0; area_i < areas_.size(); ++area_i) {
    Area& to_area = areas_[area_i];
    uint32_t total_activated = 0;

    // 计算投射到该区域的总激活神经元数
    for (uint32_t fiber_i : incoming_fibers_[to_area.index]) {
      const Fiber& fiber = fibers_[fiber_i];
      const uint32_t num_activated = areas_[fiber.from_area].activated.size();
      if (!fiber.is_active || num_activated == 0) continue;
      if (log_level_ > 0) {
        printf("%s%s", total_activated == 0 ? "Projecting " : ",",
               area_name_[fiber.from_area].c_str());
      }
      total_activated += num_activated;
    }

    // 如果没有激活神经元，跳过该区域
    if (total_activated == 0) {
      continue;
    }

    if (log_level_ > 0) {
      printf(" into %s\n", area_name_[area_i].c_str());
    }

    // 如果该区域不是固定的，则计算新的激活
    if (!to_area.is_fixed) {
      std::vector<Synapse> activations;
      if (to_area.support > 0) {
        ComputeKnownActivations(to_area, activations);  // 计算已知激活
        SelectTopK(activations, to_area.k);  // 选择前k个激活
      }
      if (activations.empty() ||
          activations[to_area.k - 1].weight < total_activated) {
        GenerateNewCandidates(to_area, total_activated, activations);  // 生成新的候选激活
        SelectTopK(activations, to_area.k);  // 再次选择前k个激活
      }
      if (log_level_ > 1) {
        printf("[Area %s] Cutoff weight for best %d activations: %f\n",
               area_name_[area_i].c_str(), to_area.k,
               activations[to_area.k - 1].weight);
      }

      // 初始化新的激活神经元索引
      new_activated[area_i].resize(to_area.k);
      const uint32_t K = to_area.support;
      uint32_t num_new = 0;
      uint32_t total_from_activated = 0;
      uint32_t total_from_non_activated = 0;

      // 处理新的激活神经元
      for (uint32_t i = 0; i < to_area.k; ++i) {
        const Synapse& s = activations[i];
        if (s.neuron >= K) {
          new_activated[area_i][i] = K + num_new;
          ConnectNewNeuron(to_area, std::round(s.weight), total_from_non_activated);
          total_from_activated += std::round(s.weight);
          num_new++;
        } else {
          new_activated[area_i][i] = s.neuron;
        }
      }

      if (log_level_ > 1) {
        printf("[Area %s] Num new activations: %u, "
               "new synapses (from activated / from non-activated): %u / %u\n",
               area_name_[area_i].c_str(), num_new, total_from_activated,
               total_from_non_activated);
      }

      std::sort(new_activated[area_i].begin(), new_activated[area_i].end());
    } else {
      new_activated[area_i] = to_area.activated;  // 如果是固定的，使用已有的激活神经元
    }

    if (update_plasticity) {
      UpdatePlasticity(to_area, new_activated[area_i]);  // 更新可塑性
    }
  }

  // 更新所有区域的激活神经元
  for (uint32_t area_i = 0; area_i < areas_.size(); ++area_i) {
    Area& area = areas_[area_i];
    if (!area.is_fixed) {
      std::swap(area.activated, new_activated[area_i]);
    }
  }

  if (log_level_ > 2) {
    LogGraphStats();  // 如果日志级别大于2，记录图的统计信息
  }

  if (update_plasticity) {
    ++step_;  // 更新步数
  }
}


void Brain::InitProjection(const ProjectMap& graph) {
  InhibitAll();
  for (const auto& [from, edges] : graph) {
    for (const auto& to : edges) {
      ActivateFiber(from, to);
    }
  }
}

void Brain::Project(const ProjectMap& graph, uint32_t num_steps,
                    bool update_plasticity) {
  InitProjection(graph);
  for (uint32_t i = 0; i < num_steps; ++i) {
    SimulateOneStep(update_plasticity);
  }
}

// 计算某个目标区域的已知激活神经元的权重
void Brain::ComputeKnownActivations(const Area& to_area,
                                    std::vector<Synapse>& activations) {
  // 初始化激活向量，大小为区域的支持度
  activations.resize(to_area.support);
  for (uint32_t i = 0; i < activations.size(); ++i) {
    activations[i].neuron = i;
    activations[i].weight = 0;  // 初始化权重为0
  }

  // 遍历投射到目标区域的所有纤维
  for (uint32_t fiber_i : incoming_fibers_[to_area.index]) {
    const Fiber& fiber = fibers_[fiber_i];
    if (!fiber.is_active) continue;  // 如果纤维不活跃，跳过

    const Area& from_area = areas_[fiber.from_area];
    // 遍历源区域的所有激活神经元
    for (uint32_t from_neuron : from_area.activated) {
      const auto& synapses = fiber.outgoing_synapses[from_neuron];
      // 遍历每个激活神经元的突触
      for (size_t i = 0; i < synapses.size(); ++i) {
        // 累加权重到目标神经元
        activations[synapses[i].neuron].weight += synapses[i].weight;
      }
    }
  }
}

// 生成新的候选突触（神经元之间的连接），这些突触会影响神经元的激活状态
void Brain::GenerateNewCandidates(const Area& to_area, uint32_t total_k,
                                  std::vector<Synapse>& activations) {
  // 计算投射到该区域的总神经元数。
   // Compute the total number of neurons firing into this area.
  const uint32_t remaining_neurons = to_area.n - to_area.support;
  if (remaining_neurons <= 2 * to_area.k) {
    // 直接从 binomial(total_k, p_) 分布中生成所有剩余神经元的突触数量。
     // Generate number of synapses for all remaining neurons directly from the
    // binomial(total_k, p_) distribution.
    std::binomial_distribution<> binom(total_k, p_);
    for (uint32_t i = 0; i < remaining_neurons; ++i) {
      activations.push_back({to_area.support + i, binom(rng_) * 1.0f});
    }
  } else {
    // 从近似 binomial(total_k, p_) 分布的正态分布尾部生成突触数量的前 k 个值。
    // TODO(szabadka): 为了正态近似生效，均值应至少为9。如果不满足这个条件，需要找到更好的近似方法。
    // Generate top k number of synapses from the tail of the normal
    // distribution that approximates the binomial(total_k, p_) distribution.
    // TODO(szabadka): For the normal approximation to work, the mean should be
    // at least 9. Find a better approximation if this does not hold.
    const float percent =
        (remaining_neurons - to_area.k) * 1.0f / remaining_neurons;
    const float cutoff = BinomQuantile(total_k, p_, percent);
    const float mu = total_k * p_;
    const float stddev = std::sqrt(total_k * p_ * (1.0f - p_));
    const float a = (cutoff - mu) / stddev;
    if (log_level_ > 1) {
      printf("[Area %s] Generating candidates: percent=%f cutoff=%.0f "
             "mu=%f stddev=%f a=%f\n", area_name_[to_area.index].c_str(),
             percent, cutoff, mu, stddev, a);
    }
    float max_d = 0;
    float min_d = total_k;
    for (uint32_t i = 0; i < to_area.k; ++i) {
      const float x = TruncatedNorm(a, rng_);
      const float d = std::min<float>(total_k, std::round(x * stddev + mu));
      max_d = std::max(d, max_d);
      min_d = std::min(d, min_d);
      activations.push_back({to_area.support + i, d});
    }
    if (log_level_ > 1) {
      printf("[Area %s] Range of %d new candidate connections: %.0f .. %.0f\n",
             area_name_[to_area.index].c_str(), to_area.k, min_d, max_d);
    }
  }
}

// 将新的神经元连接到指定的区域内，具体是从已激活和未激活的神经元中选择突触，并增加区域内支持的神经元数量
void Brain::ConnectNewNeuron(Area& area,
                             uint32_t num_synapses_from_activated,
                             uint32_t& total_synapses_from_non_activated) {
  // 从已激活的神经元中选择突触
  ChooseSynapsesFromActivated(area, num_synapses_from_activated);
  // 从未激活的神经元中选择突触
  ChooseSynapsesFromNonActivated(area, total_synapses_from_non_activated);
  // 选择当前区域的输出突触
  ChooseOutgoingSynapses(area);
  // 增加支持的神经元数量
  ++area.support;
}

// 从已激活的神经元中选择指定数量的突触，并将这些突触连接到当前区域的支持神经元上
void Brain::ChooseSynapsesFromActivated(const Area& area,
                                        uint32_t num_synapses) {
  // 获取当前支持的神经元数量
  const uint32_t neuron = area.support;
  // 初始化总激活神经元数量
  uint32_t total_k = 0;
  std::vector<uint32_t> offsets;
  // 获取当前区域的所有输入纤维
  const auto& incoming_fibers = incoming_fibers_[area.index];
  // 遍历所有输入纤维，计算总激活神经元数量，并记录每个纤维的偏移量
  for (uint32_t fiber_i : incoming_fibers) {
    const Fiber& fiber = fibers_[fiber_i];
    const Area& from_area = areas_[fiber.from_area];
    const uint32_t from_k = from_area.activated.size();
    offsets.push_back(total_k);
    if (fiber.is_active) {
      total_k += from_k;
    }
  }
  offsets.push_back(total_k);
  // 初始化随机数分布，用于从激活神经元中随机选择突触
  std::uniform_int_distribution<> u(0, total_k - 1);
  std::vector<uint8_t> selected(total_k);
  // 选择指定数量的突触
  for (uint32_t j = 0; j < num_synapses; ++j) {
    uint32_t next_i;
    // 随机选择一个未被选择的激活神经元
    while (selected[next_i = u(rng_)]) {}
    selected[next_i] = 1;
    // 找到该神经元所属的输入纤维
    auto it = std::upper_bound(offsets.begin(), offsets.end(), next_i);
    const uint32_t fiber_i = (it - offsets.begin()) - 1;
    Fiber& fiber = fibers_[incoming_fibers[fiber_i]];
    const Area& from_area = areas_[fiber.from_area];
    // 获取对应的激活神经元
    uint32_t from = from_area.activated[next_i - offsets[fiber_i]];
    // 将新的突触添加到相应的输出突触列表中
    fiber.outgoing_synapses[from].push_back({neuron, 1.0f});
  }
}

// 从未激活的神经元中选择突触，并将这些突触连接到当前区域的支持神经元上
void Brain::ChooseSynapsesFromNonActivated(const Area& area,
                                           uint32_t& total_synapses) {
  // 获取当前支持的神经元数量
  const uint32_t neuron = area.support;
  // 遍历当前区域的所有输入纤维
  for (uint32_t fiber_i : incoming_fibers_[area.index]) {
    Fiber& fiber = fibers_[fiber_i];
    const Area& from_area = areas_[fiber.from_area];
    // 创建一个向量，用于标记已激活的神经元
    std::vector<uint8_t> selected(from_area.support);
    size_t num_activated = fiber.is_active ? from_area.activated.size() : 0;
    // 如果纤维是激活的，标记所有激活的神经元
    if (fiber.is_active) {
      for (uint32_t i : from_area.activated) {
        selected[i] = 1;
      }
    }
    // 如果支持的神经元数小于等于2倍的激活神经元数，直接生成二项分布突触
    if (from_area.support <= 2 * num_activated) {
      std::binomial_distribution<> binom(1, p_);
      for (size_t from = 0; from < from_area.support; ++from) {
        if (!selected[from] && binom(rng_)) {
          fiber.outgoing_synapses[from].push_back({neuron, 1.0f});
          ++total_synapses;
        }
      }
    } else {
      // 否则，从未激活的神经元中选择突触
      uint32_t population = from_area.support - num_activated;
      std::binomial_distribution<> binom(population, p_);
      std::uniform_int_distribution<> u(0, from_area.support - 1);
      size_t num_synapses = binom(rng_);
      
      // 选择未激活的神经元
      for (size_t i = 0; i < num_synapses; ++i) {
        for (;;) {
          uint32_t from = u(rng_);
          if (selected[from]) {
            continue;
          }
          selected[from] = 1;
          fiber.outgoing_synapses[from].push_back({neuron, 1.0f});
          ++total_synapses;
          break;
        }
      }
    }
  }
}



// 为一个区域选择并生成其所有输出突触
void Brain::ChooseOutgoingSynapses(const Area& area) {
  // 遍历从当前区域出发的所有纤维
  for (uint32_t fiber_i : outgoing_fibers_[area.index]) {
    Fiber& fiber = fibers_[fiber_i];
    // 获取目标区域
    const Area& to_area = areas_[fiber.to_area];
    uint32_t support = to_area.support;
    // 如果目标区域和当前区域相同，支持数量加一
    if (area.index == to_area.index) ++support;
    // 生成突触
    std::vector<Synapse> synapses = GenerateSynapses(support, p_, rng_);
    // 将生成的突触移动到纤维的输出突触列表中
    fiber.outgoing_synapses.emplace_back(std::move(synapses));
  }
}


// 根据新激活的神经元调整突触的权重
void Brain::UpdatePlasticity(Area& to_area,
                             const std::vector<uint32_t>& new_activated) {
  // 创建一个向量用于标记新激活的神经元
  std::vector<uint8_t> is_new_activated(to_area.support);
  // 将新激活的神经元标记为1
  for (uint32_t neuron : new_activated) {
    is_new_activated[neuron] = 1;
  }
  // 遍历目标区域的所有输入纤维
  for (uint32_t fiber_i : incoming_fibers_[to_area.index]) {
    Fiber& fiber = fibers_[fiber_i];
    // 跳过不活跃的纤维
    if (!fiber.is_active) continue;
    // 获取当前纤维来源区域
    const Area& from_area = areas_[fiber.from_area];
    // 遍历来源区域中所有激活的神经元
    for (uint32_t from_neuron : from_area.activated) {
      auto& synapses = fiber.outgoing_synapses[from_neuron];
      // 遍历该神经元的所有突触
      for (size_t j = 0; j < synapses.size(); ++j) {
        // 如果突触连接到新激活的神经元
        if (is_new_activated[synapses[j].neuron]) {
          // 更新突触权重，确保不超过最大权重
          synapses[j].weight =
              std::min(synapses[j].weight * learn_rate_, max_weight_);
        }
      }
    }
  }
}


// 读取并确定在指定区域中哪个（assembly）与当前激活的神经元重叠最多，并返回该组的索引及其重叠数量
void Brain::ReadAssembly(const std::string& name,
                         size_t& index, size_t& overlap) {
  // 获取指定名称的区域
  const Area& area = GetArea(name);
  // 计算该区域内的组数
  const size_t num_assemblies = area.n / area.k;
  // 创建一个向量用于存储每个组的重叠数
  std::vector<size_t> overlaps(num_assemblies);
  // 计算每个神经元所属的组，并增加相应组的重叠数
  for (auto neuron : area.activated) {
    ++overlaps[neuron / area.k];
  }
  // 找出重叠最多的组的索引
  index = std::max_element(overlaps.begin(), overlaps.end()) - overlaps.begin();
  // 获取该组的重叠数
  overlap = overlaps[index];
}


//  打印激活日志
void Brain::LogActivated(const std::string& area_name) {
  const Area& area = GetArea(area_name);
  printf("[%s] activated: ", area_name.c_str());
  for (auto n : area.activated) printf(" %u", n);
  printf("\n");
}

// 打印日志信息
void Brain::LogGraphStats() {
  printf("Graph Stats after %u update steps\n", step_);
  for (const auto& area : areas_) {
    if (area.support == 0) continue;
    printf("Area %d [%s] has %d neurons\n",
           area.index, area_name_[area.index].c_str(), area.support);
    if (log_level_ > 3) {
      std::set<uint32_t> tmp(area.activated.begin(), area.activated.end());
      printf("   %s active:", area_name_[area.index].c_str());
      for (auto n : tmp) printf(" %u", n);
      printf("\n");
    }
  }
  const float kThresLow = std::pow(learn_rate_, 10);
  for (const Fiber& fiber : fibers_) {
    if (fiber.outgoing_synapses.empty()) continue;
    size_t num_synapses = 0;
    size_t num_low_weights = 0;
    size_t num_mid_weights = 0;
    size_t num_sat_weights = 0;
    float max_w = 0.0;
    for (uint32_t i = 0; i < fiber.outgoing_synapses.size(); ++i) {
      const auto& synapses = fiber.outgoing_synapses[i];
      num_synapses += synapses.size();
      for (size_t j = 0; j < synapses.size(); ++j) {
        const float w = synapses[j].weight;
        max_w = std::max(w, max_w);
        if (w < kThresLow) ++num_low_weights;
        else if (w < max_weight_) ++num_mid_weights;
        else ++num_sat_weights;
      }
    }
    printf("Fiber %s -> %s has %zu synapses (low/mid/sat: %zu/%zu/%zu), "
           "max w: %f\n", area_name_[fiber.from_area].c_str(),
           area_name_[fiber.to_area].c_str(), num_synapses, num_low_weights,
           num_mid_weights, num_sat_weights, max_w);
  }
}

}  // namespace nemo
