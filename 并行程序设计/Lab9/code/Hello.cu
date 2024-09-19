#include <stdio.h>

// Kernel函数，每个线程块中的每个线程执行一次
__global__ void helloWorld(int m, int k) {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x; // 计算线程块编号
    // int threadId = threadIdx.y * blockDim.x + threadIdx.x; // 计算线程在块内的编号
    
    // 输出线程块编号和线程在块内的编号
    printf("Hello World from Thread (%d, %d) in Block %d!\n", threadIdx.x, threadIdx.y, blockId);
}

int main() {
    int m = 2; // 线程块维度 m
    int k = 3; // 线程块维度 k
  	int n = 4; // 线程块数目 n
    dim3 gridDim(n); // 线程块网格维度
    dim3 blockDim(m, k); // 线程块维度
    
    // 启动内核函数
    helloWorld<<<gridDim, blockDim>>>(m, k);
    
    // 同步线程，等待内核函数执行完成
    cudaDeviceSynchronize();
    
    // 主线程输出信息
    printf("Hello World from the host!\n");
        
    return 0;
}
