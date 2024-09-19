import numpy as np

# 假设 target_area.w, target_area._new_w, 和 self.p 已经被定义
target_area_w = 5  # 示例：已经使用的权重的索引
target_area_new_w = 10  # 示例：新的总权重数
self_p = 0.5  # 示例：成功概率

# 假设 the_target_area_connectome 是一个已经定义的数组
# 我们需要知道它的列数来正确地初始化新权重
the_target_area_connectome = np.zeros((target_area_new_w, 5))  # 示例：初始化一个全0的数组
print(the_target_area_connectome)
# 使用 np.random.binomial 生成服从二项分布的随机数数组
# 我们生成 (target_area_new_w - target_area_w, the_target_area_connectome.shape[1]) 形状的数组
new_weights = np.random.binomial(n=1, p=self_p, size=(target_area_new_w - target_area_w, the_target_area_connectome.shape[1]))

# 将新生成的权重赋值给数组的相应部分
the_target_area_connectome[target_area_w:, :] = new_weights

# 打印结果
print(the_target_area_connectome)