#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "parallel_for.c"

// 结构体定义任务参数
struct matrix_multiply_args {
    int row_start;
    int row_end;
    int col_start;
    int col_end;
    float *A;
    float *B;
    float *C;
    int num_cols;
};

// 矩阵乘法的functor
void *matrix_multiply_functor(int idx, void *args) {
    struct matrix_multiply_args *mma = (struct matrix_multiply_args *)args;
    int row = idx / mma->num_cols;
    int col = idx % mma->num_cols;

    float sum = 0.0f;
    for (int k = mma->col_start; k < mma->col_end; k++) {
        sum += mma->A[row * mma->num_cols + k] * mma->B[k * mma->num_cols + col];
    }
    mma->C[row * mma->num_cols + col] = sum;

    return NULL;
}

int main()
{

    for(int num_thread=1;num_thread<=16;num_thread *= 2)
    {
        for(int matrix_size=128;matrix_size<=2048;matrix_size*=2)
        {
            // 分配矩阵内存
            float *A = (float *)malloc(matrix_size * matrix_size * sizeof(float));
            float *B = (float *)malloc(matrix_size * matrix_size * sizeof(float));
            float *C = (float *)malloc(matrix_size * matrix_size * sizeof(float));

            // 初始化矩阵数据
            for (int i = 0; i < matrix_size * matrix_size; i++) {
                A[i] = (float)(rand() % 100) / 100.0f;
                B[i] = (float)(rand() % 100) / 100.0f;
            }

            // 定义functor参数
            struct matrix_multiply_args args = {0, matrix_size, 0, matrix_size, A, B, C, matrix_size};

            // 计时开始
            struct timespec start, end;

            clock_gettime(CLOCK_REALTIME, &start);

            // 并行计算矩阵乘法
            parallel_for(0, matrix_size * matrix_size, 1, matrix_multiply_functor, (void *)&args, num_thread);

            // 计时结束
            clock_gettime(CLOCK_REALTIME, &end);

            double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
            printf("Matrix_size: %d, Num_thread: %d, Time_Spent: %f seconds\n",matrix_size, num_thread, time_spent);

            // 释放内存
            free(A);
            free(B);
            free(C);
        }
    }

    

    return 0;
}
