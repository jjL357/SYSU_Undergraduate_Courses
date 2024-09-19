#ifndef NEMO_BRAIN_UTIL_H_
#define NEMO_BRAIN_UTIL_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <iterator>
#include <vector>

namespace nemo {

// 计算两个向量的交集元素个数
inline size_t NumCommon(const std::vector<uint32_t>& a_in,
                        const std::vector<uint32_t>& b_in) {
  // 复制输入向量，避免修改原始数据
  std::vector<uint32_t> a = a_in;
  std::vector<uint32_t> b = b_in;
  
  // 对两个向量进行排序，确保 set_intersection 可以正常工作
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  
  // 用于存储交集结果的向量
  std::vector<uint32_t> c;
  
  // 计算 a 和 b 的交集，并将结果存储在 c 中
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(c));
  
  // 返回交集的大小，即共同元素的个数
  return c.size();
}

}  // namespace nemo
#endif  // NEMO_BRAIN_UTIL_H_
