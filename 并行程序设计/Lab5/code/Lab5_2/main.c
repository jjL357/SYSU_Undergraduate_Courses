#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

extern void parallel_for(int start, int end, int inc, void *(*func)(void*), void* arg, int num_threads,pthread_mutex_t*mutex);


pthread_mutex_t mutex;
// 随机生成矩阵
void generate_matrix(int rows, int cols, double *matrix) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

//函数参数的结构体
struct functor_args{
    double*A;
    double *B;
    double *C;
    int m;
    int n;
    int k;
};

struct parallel_args {
    void *functor_args;
    int start, end, inc; 
    pthread_mutex_t *mutex;
    void *(*functor)(void *);
};

// 矩阵乘法
void *matrix_multiply(void *args) {
    struct parallel_args *para_args = (struct parallel_args *)args;
    int start = para_args->start;
    int end = para_args->end;
    int inc = para_args->inc;
    pthread_mutex_t *mutex = para_args->mutex; 

    struct functor_args* func_args = (struct functor_args *)para_args->functor_args;
    double *A = func_args->A;
    double *B = func_args->B;
    double *C = func_args->C;
    int m = func_args->m;
    int n = func_args->n;
    int k = func_args->k;

    for (int i = start; i < end; i += inc) {
        for (int j = 0; j < k; j += inc) {
            double sum = 0.0;
            for (int l = 0; l < n; ++l) {
                sum += A[i * n + l] * B[l * k + j];
            }
            // 加锁
            //pthread_mutex_lock(mutex);
            C[i * k + j] = sum;
            // 解锁
            //pthread_mutex_unlock(mutex);
        }
      
    }
    return NULL;
}

int main() {
    // 初始化互斥锁
    pthread_mutex_init(&mutex, NULL);
    for(int dimension = 128 ; dimension <= 2048 ;dimension *=2){
        printf("----------------------------\n");
        printf("The dimension of matrices: %d\n",dimension);
    for(int thread_num = 1; thread_num <= 16 ; thread_num *= 2){
        printf("The num of threads: %d\n",thread_num);
    int m = dimension, n = dimension, k = dimension; // 矩阵规模
    double *A = (double *)malloc(m * n * sizeof(double));
    double *B = (double *)malloc(n * k * sizeof(double));
    double *C = (double *)malloc(m * k * sizeof(double));

    // 设置随机数种子为当前时间
    srand(time(NULL));

    // 生成随机矩阵
    generate_matrix(m, n, A);
    generate_matrix(n, k, B);

    // 输出矩阵 A
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

    struct functor_args fun_args;
    fun_args.A = A;
    fun_args.B = B;
    fun_args.C = C;
    fun_args.m = m;
    fun_args.n = n;
    fun_args.k = k;
    

    // 计时开始
    struct timespec start, end;

    clock_gettime(CLOCK_REALTIME, &start);

    // 矩阵乘法
    parallel_for(0,m,1,matrix_multiply,&fun_args,thread_num,&mutex);

    // 计时结束
    clock_gettime(CLOCK_REALTIME, &end);

  double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    

    // 输出矩阵 C
    // printf("Matrix C:\n");
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < k; ++j) {
    //         printf("%.2f ", C[i * k + j]);
    //     }
    //     printf("\n");
    // }

    // 输出消耗时间
    printf("Time elapsed: %.6f seconds\n\n", time_spent);
    //printf("Max Number of threads: %d\n", omp_get_max_threads());
    
    free(A);
    free(B);
    free(C);
    }
    printf("\n");
    }
    // 销毁互斥锁
    pthread_mutex_destroy(&mutex);
    return 0;
}
