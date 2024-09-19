#include <iostream>
#include <cuda_runtime.h>
#include <random>
using namespace std;

#define ROW 16
#define COL 16
#define N 5

//全局内存访存
__global__ void matrixTranspose(float * matrix, float * out){
    int idex = blockIdx.x * ROW + threadIdx.x;
    int idey = blockIdx.y * COL + threadIdx.y;
    if(idex < N && idey < N){
        out[idex * N + idey] = matrix[idey * N + idex];
    }
}

//共享内存访存
__global__ void matrixTranspose2(float * matrix, float * out){
    __shared__ float sharedMem[ROW][COL+1];

    int idex = blockIdx.x * ROW + threadIdx.x;
    int idey = blockIdx.y * COL + threadIdx.y;
    if(idex < N && idey < N){
        sharedMem[threadIdx.y][threadIdx.x] = matrix[idey * N + idex];
    }

    __syncthreads();

    int idex1 = blockIdx.y * COL + threadIdx.x;
    int idey1 = blockIdx.x * ROW + threadIdx.y;

    if (idex1 < N && idey1 < N) {
        out[idey1 * N + idex1] = sharedMem[threadIdx.x][threadIdx.y];
    }
}


int main(){
    int size = N*N*sizeof(float);
    float * matrix = (float*)malloc(size);
    float * matrixT = (float*)malloc(size);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            matrix[i*N + j] = i * N + j + 1;
        }
    } 
    
    cout << "Input Matrix:" << endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            cout << matrix[i*N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    
    float * outMatrix, * inMatrix;
    cudaMalloc((void **)&outMatrix, size);
    cudaMalloc((void **)&inMatrix, size);
    cudaMemcpy(inMatrix, matrix, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 块划分方式
    dim3 dimGrid1((N + ROW - 1) / ROW, (N + COL - 1) / COL);
    dim3 dimBlock1(ROW, COL);

    // 全局内存访存，块划分方式
    cudaEventRecord(start, 0);
    matrixTranspose<<<dimGrid1, dimBlock1>>>(inMatrix, outMatrix);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start, stop);
    cout << "Global memory access (block division) time: " << milliseconds1 << " ms" << endl;

    cudaMemcpy(matrixT, outMatrix, size, cudaMemcpyDeviceToHost);

    
    cout << "Transposed Matrix (block division):" << endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            cout << matrixT[i*N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    
    // 列划分方式
    dim3 dimGrid2(1, (N + COL - 1) / COL);
    dim3 dimBlock2(N, COL);

    // 全局内存访存，列划分方式
    cudaEventRecord(start, 0);
    matrixTranspose<<<dimGrid2, dimBlock2>>>(inMatrix, outMatrix);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start, stop);
    cout << "Global memory access (column division) time: " << milliseconds2 << " ms" << endl;

    cudaMemcpy(matrixT, outMatrix, size, cudaMemcpyDeviceToHost);
    
    cout << "Transposed Matrix (column division):" << endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            cout << matrixT[i*N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    
    // 行划分方式
    dim3 dimGrid3((N + ROW - 1) / ROW, 1);
    dim3 dimBlock3(ROW, N);

    // 全局内存访存，行划分方式
    cudaEventRecord(start, 0);
    matrixTranspose<<<dimGrid3, dimBlock3>>>(inMatrix, outMatrix);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds3 = 0;
    cudaEventElapsedTime(&milliseconds3, start, stop);
    cout << "Global memory access (row division) time: " << milliseconds3 << " ms" << endl;

    cudaMemcpy(matrixT, outMatrix, size, cudaMemcpyDeviceToHost);
    
    cout << "Transposed Matrix (row division):" << endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            cout << matrixT[i*N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    
    // 共享内存，块划分
    cudaEventRecord(start, 0);
    matrixTranspose2<<<dimGrid1, dimBlock1>>>(inMatrix, outMatrix);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds4 = 0;
    cudaEventElapsedTime(&milliseconds4, start, stop);
    cout << "Shared memory access time: " << milliseconds4 << " ms" << endl;

    cudaMemcpy(matrixT, outMatrix, size, cudaMemcpyDeviceToHost);
    
    cout << "Transposed Matrix (shared memory):" << endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            cout << matrixT[i*N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    
    cudaFree(outMatrix);
    cudaFree(inMatrix);
    free(matrix);
    free(matrixT);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
