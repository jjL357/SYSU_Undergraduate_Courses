#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16 // 定义线程块大小

// Kernel函数，执行矩阵乘法运算
__global__ void matrixMultiply_global(float *A, float *B, float *C, int N) {
    // 计算线程在矩阵C中的位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查是否超出矩阵大小
    if (row < N && col < N) {
        // 计算C[row][col]的值
        float sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel函数，执行矩阵乘法运算
__global__ void matrixMultiply_shared(float *A, float *B, float *C, int N) {
    // 声明共享内存数组，用于存储部分矩阵数据
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    // 计算线程在矩阵C中的位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;

    // 循环计算每个子矩阵的乘积并累加到结果中
    for (int t = 0; t < N / TILE_SIZE; ++t) {
        // 将子矩阵A和B从全局内存复制到共享内存中
        shared_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        shared_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        // 等待所有线程将数据加载到共享内存中
        __syncthreads();

        // 计算矩阵乘积的部分和
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        // 等待所有线程完成当前子矩阵的计算
        __syncthreads();
    }

    // 将计算结果写入矩阵C
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}


int main() {
    int N = 4; // 矩阵大小为 N x N
    int size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // 初始化矩阵A和B
    srand(time(NULL)); // 使用当前时间作为随机数种子
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX; // 生成随机数并归一化到 [0, 1]
        h_B[i] = (float)rand() / RAND_MAX; // 生成随机数并归一化到 [0, 1]
    }

    // 将矩阵A和B复制到设备
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义线程块和网格大小
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE); // 每个线程块大小为 16x16
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE); // 网格维度根据矩阵大小和线程块维度计算得出

    // 创建 CUDA 事件计时器
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start);

    // 调用Kernel函数进行矩阵乘法运算
    matrixMultiply_shared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

      // 同步线程，等待内核函数执行完成
    cudaDeviceSynchronize();

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 将结果复制回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 输出矩阵A
    printf("Matrix A:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.2f ", h_A[i * N + j]);
        }
        printf("\n");
    }

    // 输出矩阵B
    printf("\nMatrix B:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.2f ", h_B[i * N + j]);
        }
        printf("\n");
    }

    // 输出矩阵C
    printf("\nMatrix C:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.2f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // 输出计算时间
    printf("\nTime for computing C: %.2f ms\n", milliseconds);


    

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
