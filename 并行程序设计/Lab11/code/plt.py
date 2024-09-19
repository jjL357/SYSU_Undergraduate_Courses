import matplotlib.pyplot as plt

# 数据
input_dims = ['3x128x128', '3x256x256', '3x512x512']
global_memory_times = [0.03, 0.04, 0.05]
shared_memory_times = [0.03, 0.03, 0.05]
im2col_times = [0.03, 0.25, 1.07]
cuDNN_times = [0.039844, 0.039543, 0.041879]

# 绘制图表
plt.figure(figsize=(10, 6))

plt.plot(input_dims, global_memory_times, label='Global Memory Time (ms)', marker='o')
plt.plot(input_dims, shared_memory_times, label='Shared Memory Time (ms)', marker='o')
plt.plot(input_dims, im2col_times, label='im2col Time (ms)', marker='o')
plt.plot(input_dims, cuDNN_times, label='cuDNN Time (ms)', marker='o')

plt.xlabel('Input Dimensions')
plt.ylabel('Time (ms)')
plt.title('Convolution Time Comparison (Input Size: 128x128, 256x256, 512x512)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
