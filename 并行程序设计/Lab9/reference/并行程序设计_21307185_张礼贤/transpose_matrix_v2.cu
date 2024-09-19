#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

// 全局内存访存的CUDA核函数
__global__ void matrixTransposeGlobal(float *d_out, float *d_in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index_in = y * width + x;
        int index_out = x * height + y;
        d_out[index_out] = d_in[index_in];
    }
}

// 共享内存访存的CUDA核函数
__global__ void matrixTransposeShared(float *d_out, float *d_in, int width, int height, int tile_dim, int block_rows) {
    extern __shared__ float tile[];

    int x = blockIdx.x * tile_dim + threadIdx.x;
    int y = blockIdx.y * tile_dim + threadIdx.y;
    int local_index = threadIdx.y * tile_dim + threadIdx.x;

    if (x < width && y < height) {
        int index_in = y * width + x;

        for (int i = 0; i < tile_dim; i += block_rows) {
            if (y + i < height) {
                tile[(threadIdx.y + i) * (tile_dim + 1) + threadIdx.x] = d_in[index_in + i * width];
            }
        }
    }

    __syncthreads(); // 等待所有线程加载完毕

    x = blockIdx.y * tile_dim + threadIdx.x;
    y = blockIdx.x * tile_dim + threadIdx.y;

    if (x < height && y < width) {
        int index_out = y * height + x;

        for (int i = 0; i < tile_dim; i += block_rows) {
            if (y + i < width) {
                d_out[index_out + i * height] = tile[threadIdx.x * (tile_dim + 1) + (threadIdx.y + i)];
            }
        }
    }
}

// 主函数
void testTranspose(int m, int n, dim3 grid, dim3 threads, bool useShared, int tile_dim, int block_rows) {
    size_t bytes = m * n * sizeof(float);

    // 分配内存并初始化矩阵
    float *h_matrix = (float*)malloc(bytes);
    float *h_transposed = (float*)malloc(bytes);
    if (h_matrix == NULL || h_transposed == NULL) {
        printf("Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    srand(time(NULL));
    for (int i = 0; i < m * n; i++) {
        h_matrix[i] = (float)rand() / RAND_MAX;
    }

    // 分配GPU内存
    float *d_matrix, *d_transposed;
    if (cudaMalloc(&d_matrix, bytes) != cudaSuccess || cudaMalloc(&d_transposed, bytes) != cudaSuccess) {
        printf("Device memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // 将矩阵数据从主机内存复制到GPU内存
    if (cudaMemcpy(d_matrix, h_matrix, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Memory copy to device failed\n");
        exit(EXIT_FAILURE);
    }

    // 启动CUDA核函数，进行矩阵转置
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (useShared) 
    {
        matrixTransposeShared<<<grid, threads, (tile_dim * (tile_dim + 1) * sizeof(float))>>>(d_transposed, d_matrix, n, m, tile_dim, block_rows);
    } 
    else 
    {
        matrixTransposeGlobal<<<grid, threads>>>(d_transposed, d_matrix, n, m);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 将转置后的矩阵数据从GPU内存复制回主机内存
    if (cudaMemcpy(h_transposed, d_transposed, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("Memory copy to host failed\n");
        exit(EXIT_FAILURE);
    }

    // 检验结果
    bool correct = true;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (h_transposed[j * m + i] != h_matrix[i * n + j]) {
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }

    if (correct) {
        printf("Matrix Transpose is correct!\n");
    } else {
        printf("Matrix Transpose is incorrect!\n");
    }

    printf("Matrix size: %dx%d, Grid: (%d,%d), Threads: (%d,%d), Shared Memory: %s\n", 
           m, n, grid.x, grid.y, threads.x, threads.y, useShared ? "Yes" : "No");
    printf("Time taken for matrix transpose: %f milliseconds\n", milliseconds);

    // 释放内存
    free(h_matrix);
    free(h_transposed);
    cudaFree(d_matrix);
    cudaFree(d_transposed);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <m> <n>\n", argv[0]);
        return -1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);

    // 设置不同的CUDA网格和线程块的维度进行测试
    dim3 threads[] = {dim3(16, 16), dim3(32, 8), dim3(32, 32)};
    dim3 grids[] = {
        dim3((n + 15) / 16, (m + 15) / 16),
        dim3((n + 31) / 32, (m + 7) / 8),
        dim3((n + 31) / 32, (m + 31) / 32)
    };
    int tile_dims[] = {16, 32, 32};
    int block_rowss[] = {16, 8, 32};

    for (int i = 0; i < 3; i++) {
        printf("Testing Global Memory Transpose:\n");
        testTranspose(m, n, grids[i], threads[i], false, tile_dims[i], block_rowss[i]);
        printf("Testing Shared Memory Transpose:\n");
        testTranspose(m, n, grids[i], threads[i], true, tile_dims[i], block_rowss[i]);
    }

    return 0;
}
