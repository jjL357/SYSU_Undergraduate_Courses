import matplotlib.pyplot as plt
import numpy as np
import math

# 核心数列表
p_values = [2**i for i in range(1, 11)]

# 计算线性结构算法的操作次数
linear_receives = [p-1 for p in p_values]
linear_adds = [p-1 for p in p_values]

# 计算树形结构算法的操作次数
tree_receives = [math.ceil(math.log2(p)) for p in p_values]  # 使用上确界
tree_adds = [math.ceil(math.log2(p)) for p in p_values]  # 使用上确界

# 绘制接收操作次数比较图
plt.figure(figsize=(10, 5))
plt.plot(p_values, linear_receives, label='Linear - Receives', marker='o')
plt.plot(p_values, tree_receives, label='Tree - Receives', marker='x')
plt.xlabel('Total Cores (p)')
plt.ylabel('Number of Receives by Core 0')
plt.title('Comparison of Receives in Linear vs. Tree Algorithms')
plt.xscale('log', base=2)  # 使用'base'设置对数刻度
plt.yscale('log', base=2)  # 使用'base'修正
plt.xticks(p_values, labels=p_values)
plt.grid(True, which="both", ls="--")
plt.legend()

# 绘制加法操作次数比较图
plt.figure(figsize=(10, 5))
plt.plot(p_values, linear_adds, label='Linear - Adds', marker='o')
plt.plot(p_values, tree_adds, label='Tree - Adds', marker='x')
plt.xlabel('Total Cores (p)')
plt.ylabel('Number of Adds by Core 0')
plt.title('Comparison of Adds in Linear vs. Tree Algorithms')
plt.xscale('log', base=2)  # 一致使用'base'
plt.yscale('log', base=2)  # 这里也正确使用'base'
plt.xticks(p_values, labels=p_values)
plt.grid(True, which="both", ls="--")
plt.legend()

plt.show()
