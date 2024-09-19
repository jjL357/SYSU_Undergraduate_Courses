#ifndef NEMO_BRAIN_H_
#define NEMO_BRAIN_H_

#include <stdint.h>

#include <map>      // 使用标准映射容器
#include <random>   // 使用随机数库
#include <string>   // 使用字符串类
#include <vector>   // 使用向量容器

namespace nemo {

// 突触结构体
struct Synapse {
  uint32_t neuron;  // 神经元索引
  float weight;     // 突触权重
};

// 区域结构体
struct Area {
  Area(uint32_t index, uint32_t n, uint32_t k) : index(index), n(n), k(k) {}

  const uint32_t index;         // 区域索引
  const uint32_t n;             // 区域中神经元数量
  const uint32_t k;             // 每个集合中的神经元数量
  uint32_t support = 0;         // 支持数
  bool is_fixed = false;        // 是否固定
  std::vector<uint32_t> activated;  // 激活的神经元索引向量
};

// 纤维结构体
struct Fiber {
  Fiber(uint32_t from, uint32_t to) : from_area(from), to_area(to) {}

  const uint32_t from_area;         // 起始区域索引
  const uint32_t to_area;           // 目标区域索引
  bool is_active = true;            // 是否激活
  std::vector<std::vector<Synapse>> outgoing_synapses;  // 出站突触的二维向量
};

// 项目映射类型定义
typedef std::map<std::string, std::vector<std::string>> ProjectMap;

// 大脑类声明
class Brain {
 public:
  Brain(float p, float beta, float max_weight, uint32_t seed);  // 构造函数

  // 添加区域
  Area& AddArea(const std::string& name, uint32_t n, uint32_t k,
                bool recurrent = true, bool is_explicit = false);
  
  // 添加刺激
  void AddStimulus(const std::string& name, uint32_t k);
  
  // 添加纤维
  void AddFiber(const std::string& from, const std::string& to,
                bool bidirectional = false);

  // 获取区域（可变版本和常量版本）
  Area& GetArea(const std::string& name);
  const Area& GetArea(const std::string& name) const;

  // 获取纤维（可变版本和常量版本）
  Fiber& GetFiber(const std::string& from, const std::string& to);
  const Fiber& GetFiber(const std::string& from, const std::string& to) const;

  // 抑制所有区域
  void InhibitAll();
  
  // 抑制指定纤维
  void InhibitFiber(const std::string& from, const std::string& to);
  
  // 激活指定纤维
  void ActivateFiber(const std::string& from, const std::string& to);
  
  // 初始化投影
  void InitProjection(const ProjectMap& graph);

  // 激活区域中的集合
  void ActivateArea(const std::string& name, uint32_t assembly_index);

  // 模拟一个时间步
  void SimulateOneStep(bool update_plasticity = true);
  
  // 投影到指定步数
  void Project(const ProjectMap& graph, uint32_t num_steps,
               bool update_plasticity = true);

  // 读取集合信息
  void ReadAssembly(const std::string& name, size_t& index, size_t& overlap);

  // 设置日志级别
  void SetLogLevel(int log_level) { log_level_ = log_level; }
  
  // 记录图统计信息
  void LogGraphStats();
  
  // 记录激活信息
  void LogActivated(const std::string& area_name);

 private:
  // 计算已知激活
  void ComputeKnownActivations(const Area& to_area,
                               std::vector<Synapse>& activations);
  
  // 生成新候选者
  void GenerateNewCandidates(const Area& to_area, uint32_t total_k,
                             std::vector<Synapse>& activations);
  
  // 连接新神经元
  void ConnectNewNeuron(Area& area,
                        uint32_t num_synapses_from_activated,
                        uint32_t& total_synapses_from_non_activated);
  
  // 从已激活中选择突触
  void ChooseSynapsesFromActivated(const Area& area,
                                   uint32_t num_synapses);
  
  // 从未激活中选择突触
  void ChooseSynapsesFromNonActivated(const Area& area,
                                      uint32_t& total_synapses);
  
  // 选择出站突触
  void ChooseOutgoingSynapses(const Area& area);
  
  // 更新可塑性
  void UpdatePlasticity(Area& to_area,
                        const std::vector<uint32_t>& new_activated);

 protected:
  std::mt19937 rng_;      // 随机数生成器
  int log_level_ = 0;     // 日志级别

  const float p_;         // 激活概率
  const float beta_;      // 贝塔值
  const float learn_rate_;// 学习率
  const float max_weight_;// 最大权重
  std::vector<Area> areas_;       // 区域向量
  std::vector<Fiber> fibers_;     // 纤维向量
  std::vector<std::vector<uint32_t>> incoming_fibers_;   // 入站纤维
  std::vector<std::vector<uint32_t>> outgoing_fibers_;   // 出站纤维
  std::map<std::string, uint32_t> area_by_name_;          // 区域名称映射
  std::vector<std::string> area_name_;                    // 区域名称向量
  uint32_t step_ = 0;      // 步数
};

}  // namespace nemo

#endif // NEMO_BRAIN_H_
