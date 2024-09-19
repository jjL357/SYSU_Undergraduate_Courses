#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define ROW 16
#define COL 16

// Kernel函数，每个线程块中的每个线程执行一次矩阵转置操作
// 每个线程处理一个位置的转置
__global__ void matrixTranspose_global(float *A, float *AT, int n) {
    // 计算线程在矩阵中的索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查索引是否超出矩阵的范围
    if (i < n && j < n) {
        // 计算转置后的索引并进行赋值
        AT[j * n + i] = A[i * n + j];
    }
}


// Kernel函数，每个线程块中的每个线程执行一次矩阵转置操作
// 每个线程块处理一个数据块的转置
__global__ void matrixTranspose_shared(float *A, float *AT, int n) {
    // 声明共享内存数组，用于存储输入矩阵的部分数据
    __shared__ float sharedMemory[ROW][COL];

    // 计算当前线程在输入矩阵中的索引
    int i1 = blockIdx.x * ROW + threadIdx.x;
    int j1 = blockIdx.y * COL + threadIdx.y;

    // 将输入矩阵中的数据复制到共享内存中
    if (i1 < n && j1 < n) {
        sharedMemory[threadIdx.y][threadIdx.x] = A[j1 * n + i1];
    }

    // 等待所有线程将数据加载到共享内存中
    __syncthreads();

    // 计算当前线程在输出矩阵中的索引，以进行转置操作
    int i2 = blockIdx.y * COL + threadIdx.x;
    int j2 = blockIdx.x * ROW + threadIdx.y;

    // 将共享内存中的数据写回到输出矩阵中，完成转置操作
    if (i2 < n && j2 < n) {
        AT[j2 * n + i2] = sharedMemory[threadIdx.x][threadIdx.y];
    }
}



// 打印矩阵
void printMatrix(float* A,int n){
    for(int i = 0 ;i < n ;i++){
        for(int j = 0 ;j < n;j++){
            printf("%lf ",A[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int n; // 矩阵大小

    // 检查是否提供了命令行参数
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return -1;
    }

    // 将命令行参数转换为整数作为矩阵大小
    n = atoi(argv[1]);

    size_t size = n * n * sizeof(float); // 矩阵数据大小

    printf("Matrix Size: %d x %d\n", n, n);

    // 在主机上分配内存并生成随机矩阵A
    float *h_A = (float*)malloc(size);
    for (int i = 0; i < n * n; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
    }
    
    // 在设备上分配内存
    float *d_A, *d_AT;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_AT, size);

    

    // 定义线程块和网格维度
    dim3 blockSize(ROW, COL);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    //打印转置前的矩阵
    printf("The matrix before transposing:\n");
    printMatrix(h_A,n);

    // 创建 CUDA 事件计时器
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start);

    // 调用Kernel函数执行矩阵转置
    matrixTranspose_global<<<gridSize, blockSize>>>(d_A, d_AT, n);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算转置操作的执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 打印网格维度和线程维度
    printf("Grid Size (global): (%d, %d)\n", gridSize.x, gridSize.y);
    printf("Block Size (global): (%d, %d)\n", blockSize.x, blockSize.y);
    printf("Time to compute matrix transpose(global): %.5f milliseconds\n", milliseconds);

    // 将结果从设备复制回主机
    cudaMemcpy(h_A, d_AT, size, cudaMemcpyDeviceToHost); // 将转置后的矩阵复制回主机内存

    // 打印转置后的矩阵
    printf("The matrix transpose:\n");
    printMatrix(h_A,n);

    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost); // 将转置前的矩阵复制回主机内存

    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    // 打印转置前的矩阵
    printf("The matrix before transposing:\n");
    printMatrix(h_A,n);

    // 记录开始时间
    cudaEventRecord(start);

    // 调用Kernel函数执行矩阵转置
    matrixTranspose_shared<<<gridSize, blockSize>>>(d_A, d_AT, n);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算转置操作的执行时间
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 打印网格维度和线程维度
    printf("Grid Size (shared): (%d, %d)\n", gridSize.x, gridSize.y);
    printf("Block Size (shared): (%d, %d)\n", blockSize.x, blockSize.y);
    printf("Time to compute matrix transpose(shared): %.5f milliseconds\n", milliseconds);

    // 将结果从设备复制回主机
    cudaMemcpy(h_A, d_AT, size, cudaMemcpyDeviceToHost); // 将转置后的矩阵复制回主机内存

    // 打印转置后的矩阵
    printf("The matrix transpose:\n");
    printMatrix(h_A,n);
    
    // 释放内存
    free(h_A);
    cudaFree(d_A);
    cudaFree(d_AT);

    return 0;
}
