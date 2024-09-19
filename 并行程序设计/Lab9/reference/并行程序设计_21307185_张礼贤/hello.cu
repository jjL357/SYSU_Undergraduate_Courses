#include <stdio.h>
#include <stdlib.h>

__global__ void helloWorld(int n, int m, int k) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;

    printf("Hello World from Thread (%d, %d) in Block %d!\n", threadIdx.x, threadIdx.y, blockId);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <n> <m> <k>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);  // 线程块数量
    int m = atoi(argv[2]);  // 每个线程块的维度
    int k = atoi(argv[3]);  // 每个线程块的维度

    helloWorld<<<n, dim3(m, k)>>>(n, m, k);

    cudaDeviceSynchronize();

    printf("Hello World from the host!\n");

    return 0;
}
