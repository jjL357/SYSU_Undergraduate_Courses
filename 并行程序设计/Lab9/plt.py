import matplotlib.pyplot as plt

# 数据
matrix_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# 不同块大小的时间数据
block_sizes = {
    (2, 2): [0.19661, 0.00819, 0.00819, 0.01126, 0.03846, 0.11923, 0.44579, 1.74307, 6.95862, 27.84317, 107.90051],
    (4, 4): [0.01024, 0.00614, 0.00614, 0.00614, 0.01478, 0.03837, 0.12170, 0.44384, 1.74899, 6.96794, 23.70880],
    (8, 8): [0.00922, 0.00614, 0.00614, 0.00611, 0.00982, 0.01680, 0.04051, 0.11632, 0.44707, 1.75478, 6.14333],
    (16, 16): [0.00614, 0.00717, 0.00614, 0.00512, 0.00816, 0.01200, 0.02109, 0.06387, 0.22794, 1.41088, 5.21613]
}

# 绘图
plt.figure(figsize=(10, 6))

for block_size, times in block_sizes.items():
    plt.plot(matrix_sizes, times, marker='o', label=f'Block Size {block_size}')

plt.xlabel('Matrix Size')
plt.ylabel('Time (milliseconds)')
plt.title('Matrix Transpose Execution Time vs. Matrix Size for Different Block Sizes')
plt.xscale('log')
plt.yscale('log')
plt.xticks(matrix_sizes, matrix_sizes)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
