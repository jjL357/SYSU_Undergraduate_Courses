#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void generate_matrix(int n, double* matrix) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = (double)rand() / RAND_MAX;
        }
    }
}

int main() {
    for (int num_threads = 1; num_threads <= 16; num_threads*=2) {
        for (int n = 128; n <= 2048; n *= 2) {
            double* A = (double*)malloc(n * n * sizeof(double));
            double* B = (double*)malloc(n * n * sizeof(double));
            double* C = (double*)malloc(n * n * sizeof(double));
            double start_time, end_time, elapsed_time;

            // 生成随机矩阵
            generate_matrix(n, A);
            generate_matrix(n, B);

            // 设置线程数量
            omp_set_num_threads(num_threads);

            // 默认调度
            start_time = omp_get_wtime();
            #pragma omp parallel for
            for (int i = 0; i < n; i++) 
            {
                for (int j = 0; j < n; j++) 
                {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++) 
                    {
                        sum += A[i * n + k] * B[k * n + j];
                    }
                    C[i * n + j] = sum;
                }
            }
            end_time = omp_get_wtime();
            elapsed_time = end_time - start_time;
            printf("Threads: %d, Matrix Size: %d, Schedule: Default, Elapsed Time: %.6f seconds\n", num_threads, n, elapsed_time);

            // 静态调度
            start_time = omp_get_wtime();
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) 
            {
                for (int j = 0; j < n; j++) 
                {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++) 
                    {
                        sum += A[i * n + k] * B[k * n + j];
                    }
                    C[i * n + j] = sum;
                }
            }
            end_time = omp_get_wtime();
            elapsed_time = end_time - start_time;
            printf("Threads: %d, Matrix Size: %d, Schedule: Static, Elapsed Time: %.6f seconds\n", num_threads, n, elapsed_time);

            // 动态调度
            start_time = omp_get_wtime();
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n; i++) 
            {
                for (int j = 0; j < n; j++) 
                {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++) 
                    {
                        sum += A[i * n + k] * B[k * n + j];
                    }
                    C[i * n + j] = sum;
                }
            }
            end_time = omp_get_wtime();
            elapsed_time = end_time - start_time;
            printf("Threads: %d, Matrix Size: %d, Schedule: Dynamic, Elapsed Time: %.6f seconds\n", num_threads, n, elapsed_time);

            free(A);
            free(B);
            free(C);
        }
    }

    return 0;
}
