import matplotlib.pyplot as plt

# 线程块大小
block_sizes = [4, 8, 16, 32, 64]

# 全局内存时间（ms）
global_memory_times = [4.45, 2.31, 1.59, 1.48, 0.01]

# 共享内存时间（ms）
shared_memory_times = [2.94, 1.61, 1.63, 2.08, 0.01]

# 创建图表
plt.figure(figsize=(15, 6))

# 绘制全局内存时间
plt.plot(block_sizes, global_memory_times, marker='o', linestyle='-', color='b', label='Global Memory')

# 绘制共享内存时间
plt.plot(block_sizes, shared_memory_times, marker='o', linestyle='-', color='r', label='Shared Memory')

# 添加标题和标签
plt.title('Convolution Time vs. Block Size')
plt.xlabel('Thread Block Size (TILESIZE x TILESIZE)')
plt.ylabel('Time (ms)')
plt.grid(True, which="both", ls="--")
plt.legend()

# 调整 x 轴标签显示格式
plt.xticks(block_sizes, [f'{size}x{size}' for size in block_sizes])

# 显示图表
plt.show()
