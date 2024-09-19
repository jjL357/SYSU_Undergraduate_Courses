import matplotlib.pyplot as plt

# 数据
processes = [1, 2, 4]
matrix_sizes = [128, 256, 512, 1024, 2048]
data = [
    [0.015423, 0.127436, 1.092201, 10.044151, 89.042888],
    [0.007757, 0.084623, 0.714060, 5.965219, 55.334119],
    [0.006912, 0.043915, 0.408103, 3.375850, 32.002999]
]  # 将其余的数据补充完整

# 绘制图表
plt.figure(figsize=(10, 6))

for i in range(len(matrix_sizes)):
    plt.plot(processes, [row[i] for row in data], marker='o', label=f'Matrix Size {matrix_sizes[i]}')

plt.title('Execution Time vs. Number of Processes')
plt.xlabel('Number of Processes')
plt.ylabel('Execution Time (s)')
plt.legend()
plt.grid(True)
plt.xscale('log')  # 使用对数坐标
plt.yscale('log')  # 使用对数坐标
plt.xticks(processes, processes)  # 设置x轴刻度为进程数
plt.show()
