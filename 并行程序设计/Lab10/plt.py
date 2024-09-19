import matplotlib.pyplot as plt

# 数据：矩阵维度、全局内存运行时间、共享内存运行时间
matrix_sizes = ["128 x 128", "256 x 256", "512 x 512", "1024 x 1024", "2048 x 2048"]
global_memory_times = [
    [0.04, 0.16, 1.10, 8.64, 84.26],  # Block size (2, 2)
    [0.03, 0.06, 0.38, 2.78, 23.22],  # Block size (4, 4)
    [0.02, 0.03, 0.15, 1.02, 8.46],   # Block size (16, 16)
    [0.03, 0.03, 0.15, 0.93, 7.44]    # Block size (32, 32)
]
shared_memory_times = [
    [0.07, 0.41, 3.14, 24.88, 192.24],  # Block size (2, 2)
    [0.02, 0.08, 0.47, 3.61, 31.15],    # Block size (4, 4)
    [0.02, 0.03, 0.11, 0.61, 4.66],     # Block size (16, 16)
    [0.02, 0.02, 0.09, 0.53, 4.15]      # Block size (32, 32)
]

# 绘图
plt.figure(figsize=(10, 6))

# 绘制全局内存性能
for i, block_size in enumerate([(2, 2), (4, 4), (16, 16), (32, 32)]):
    plt.plot(matrix_sizes, global_memory_times[i], label=f'Global Memory (Block {block_size})', marker='o')

# 绘制共享内存性能
for i, block_size in enumerate([(2, 2), (4, 4), (16, 16), (32, 32)]):
    plt.plot(matrix_sizes, shared_memory_times[i], label=f'Shared Memory (Block {block_size})', marker='o')

plt.title('Performance Comparison')
plt.xlabel('Matrix Dimensions')
plt.ylabel('Time (ms)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
