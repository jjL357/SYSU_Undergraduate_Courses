#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
int num_of_threads = 0;
// 随机生成矩阵
void generate_matrix(int rows, int cols, double *matrix) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

// 矩阵乘法
void matrix_multiply_static(int m, int n, int k, double *A, double *B, double *C) {
	
    #pragma omp parallel for  schedule(static) num_threads(num_of_threads)
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

// 矩阵乘法
void matrix_multiply_dynamic(int m, int n, int k, double *A, double *B, double *C) {
	
    //#pragma omp parallel for  schedule(static) num_threads(1)
    #pragma omp parallel for  schedule(dynamic) num_threads(num_of_threads)
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

// 矩阵乘法
void matrix_multiply_guided(int m, int n, int k, double *A, double *B, double *C) {
	
    //#pragma omp parallel for  schedule(static) num_threads(1)
    //#pragma omp parallel for  schedule(dynamic) num_threads(num_of_threads)
    #pragma omp parallel for  schedule(guided) num_threads(num_of_threads)
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
    for(int dimension = 128; dimension <= 2048 ;dimension*=2){
    printf("------------------------------\n");
    printf("The dimension of matrices:%d\n",dimension);
    int m = dimension, n = dimension, k = dimension; // 矩阵规模
    double *A = (double *)malloc(m * n * sizeof(double));
    double *B = (double *)malloc(n * k * sizeof(double));
    double *C = (double *)malloc(m * k * sizeof(double));

    // 设置随机数种子为当前时间
    srand(time(NULL));

    // 生成随机矩阵
    generate_matrix(m, n, A);
    generate_matrix(n, k, B);

    // // 输出矩阵 A
    // printf("Matrix A:\n");
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         printf("%.2f ", A[i * n + j]);
    //     }
    //     printf("\n");
    // }

    // // 输出矩阵 B
    // printf("Matrix B:\n");
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < k; ++j) {
    //         printf("%.2f ", B[i * k + j]);
    //     }
    //     printf("\n");
    // }
    double start_time,end_time,elapsed_time;
    for(num_of_threads = 1; num_of_threads <=16 ; num_of_threads *=2){
    	
	    // 计时
	    //double start_time = omp_get_wtime();
	    printf("The num of threads:%d\n",num_of_threads);
	    // 矩阵乘法
	    start_time = omp_get_wtime();
	    matrix_multiply_static(m, n, k, A, B, C);
	    end_time = omp_get_wtime();
	    elapsed_time = end_time - start_time;
	    printf("Time elapsed[static]: %.6f seconds\n", elapsed_time);
	    
	    start_time = omp_get_wtime();
	    matrix_multiply_dynamic(m, n, k, A, B, C);
	    end_time = omp_get_wtime();
	    elapsed_time = end_time - start_time;
	    printf("Time elapsed[dynamic]: %.6f seconds\n", elapsed_time);
	    
	    start_time = omp_get_wtime();
	    matrix_multiply_guided(m, n, k, A, B, C);
	    end_time = omp_get_wtime();
	    elapsed_time = end_time - start_time;
	    printf("Time elapsed[guided]: %.6f seconds\n", elapsed_time);
	    
	    //double end_time = omp_get_wtime();
	    //double elapsed_time = end_time - start_time;

	//     // 输出矩阵 C
	//     printf("Matrix C:\n");
	//     /for (int i = 0; i < m; ++i) {
	//         for (int j = 0; j < k; ++j) {
	//             printf("%.2f ", C[i * k + j]);
	//         }
	//         printf("\n");
	//     }

	    // 输出消耗时间
	    
	    //printf("Max Number of threads: %d\n", omp_get_max_threads());
    	    printf("\n");
    }
    
    
    

    free(A);
    free(B);
    free(C);
    printf("\n");
    }
    return 0;
}
