import matplotlib.pyplot as plt

# 数据
threads = [1, 2, 4, 8, 16]
iterations = [16978, 16978, 14838, 13515, 16863]
runtimes = [59.589935, 40.103887, 30.738437, 39.187043, 72.638829]

# 绘制折线图
plt.plot(threads, runtimes, marker='o', label='running time')

# 添加标题和标签
plt.title('thread_num - running time')
plt.xlabel('thread_num')
plt.ylabel('running time(s)')
plt.xticks(threads)  # 设置x轴刻度

# 添加网格和图例
plt.grid(True)
plt.legend()

# 显示图形
plt.show()
