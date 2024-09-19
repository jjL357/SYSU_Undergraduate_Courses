import matplotlib.pyplot as plt

# Data
num_threads = [1, 2, 4, 8, 16]
num_of_nodes = [128, 256, 512, 1024, 2048]
computing_time = [
    [0.017671, 0.092672, 0.705672, 6.108950, 27.934701],
    [0.005877, 0.043937, 0.264180, 2.157002, 16.602200],
    [0.005432, 0.022412, 0.165022, 1.348110, 10.071274],
    [0.004384, 0.020871, 0.173033, 1.251947, 11.966243],
    [0.007342, 0.041568, 0.333765, 2.459176, 18.564995]
]

# Plot
plt.figure(figsize=(10, 6))
for i in range(len(num_threads)):
    plt.plot(num_of_nodes, computing_time[i], label=f'{num_threads[i]} threads')

plt.title('Computing Time vs. Number of Nodes')
plt.xlabel('Number of Nodes')
plt.ylabel('Computing Time (seconds)')
plt.xscale('log')  # Log scale for better visualization
plt.yscale('log')  # Log scale for better visualization
plt.grid(True, which="both", ls="--", color='gray', alpha=0.5)
plt.xticks(num_of_nodes, num_of_nodes)  # Set x ticks to match the number of nodes
plt.legend()
plt.show()
