#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// 随机生成矩阵
void generate_matrix(int rows, int cols, double *matrix) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

// 矩阵乘法
void matrix_multiply(int m, int n, int k, double *A, double *B, double *C) {
	//选择调度方式
    //#pragma omp parallel for  schedule(static) num_threads(1)
    //#pragma omp parallel for  schedule(dynamic) num_threads(8)
    //#pragma omp parallel for  schedule(guided) num_threads(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            for (int l = 0; l < n; ++l) {
                sum += A[i * n + l] * B[l * k + j];
            }
            // 使用 #pragma omp critical 声明临界区
            #pragma omp critical
            {
                C[i * k + j] = sum;
            }
        }
      
    }
}

int main() {
    int m = 128, n = 128, k = 128; // 矩阵规模
    double *A = (double *)malloc(m * n * sizeof(double));
    double *B = (double *)malloc(n * k * sizeof(double));
    double *C = (double *)malloc(m * k * sizeof(double));

    // 设置随机数种子为当前时间
    srand(time(NULL));

    // 生成随机矩阵
    generate_matrix(m, n, A);
    generate_matrix(n, k, B);

    // 输出矩阵 A
    printf("Matrix A:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.2f ", A[i * n + j]);
        }
        printf("\n");
    }

    // 输出矩阵 B
    printf("Matrix B:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            printf("%.2f ", B[i * k + j]);
        }
        printf("\n");
    }

    // 计时
    double start_time = omp_get_wtime();

    // 矩阵乘法
    matrix_multiply(m, n, k, A, B, C);

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    // 输出矩阵 C
    printf("Matrix C:\n");
    /for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            printf("%.2f ", C[i * k + j]);
        }
        printf("\n");
    }

    // 输出消耗时间
    printf("Time elapsed: %.6f seconds\n", elapsed_time);
    //printf("Max Number of threads: %d\n", omp_get_max_threads());
    
    

    free(A);
    free(B);
    free(C);
    return 0;
}
