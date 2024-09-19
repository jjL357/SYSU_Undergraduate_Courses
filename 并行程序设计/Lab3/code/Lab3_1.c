#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <time.h>

#define MATRIX_SIZE_MIN 128
#define MATRIX_SIZE_MAX 2048

int **A,**B,**C;
int matrix_size;
int num_threads;


// 分配矩阵内存的函数
int **allocate_matrix(int size) {
    int **matrix = (int **)malloc(size * sizeof(int *));
    for (int i = 0; i < size; i++) {
        matrix[i] = (int *)malloc(size * sizeof(int));
    }
    return matrix;
}

// 生成矩阵随机值的函数
void generate_random_matrix(int **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = rand() % 10; 
        }
    }
}

// 释放矩阵内存的函数
void free_matrix(int **matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// 在指定行范围内对矩阵A和矩阵B进行乘法运算的函数
void *multiply_matrices(void* rank) {
    int my_rank = (long)rank;
    int rows = matrix_size / num_threads;

    for (int i = my_rank * rows; i < (my_rank + 1)* rows; i++) {
        for (int j = 0; j < matrix_size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < matrix_size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return NULL;
}
void print_matrix(int**matrix){
    for(int i = 0;i < matrix_size; i++){
        for(int j = 0;j < matrix_size ;j++ ){
            printf("%d ",matrix[i][j]);
        }
        printf("\n");
    }
}
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("用法: %s <矩阵大小> <线程数量> \n", argv[0]);
        return 1;
    }

    matrix_size = atoi(argv[1]);
    num_threads = atoi(argv[2]);
    


    // 设置随机数种子
    srand(time(NULL));

    // 分配矩阵内存
    A = allocate_matrix(matrix_size);
    B = allocate_matrix(matrix_size);
    C = allocate_matrix(matrix_size);

    // 生成矩阵随机值
    generate_random_matrix(A, matrix_size);
    generate_random_matrix(B, matrix_size);

    // 获取开始时间
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // 创建线程
    pthread_t threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, multiply_matrices,(void*)i);
    }

    // 等待线程结束
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // 获取结束时间
    gettimeofday(&end_time, NULL);

    // 计算消耗的时间
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0; // 将秒转换为毫秒
    elapsed_time += (end_time.tv_usec - start_time.tv_usec) / 1000.0; // 将微秒转换为毫秒

    // 输出矩阵计算所消耗的时间
    printf("Computing time:  %.2f ms\n", elapsed_time);
    // printf("A:\n");
    // print_matrix(A);
    // printf("B\n");
    // print_matrix(B);
    // printf("C\n");
    // print_matrix(C);
    // 释放矩阵内存
    free_matrix(A, matrix_size);
    free_matrix(B, matrix_size);
    free_matrix(C, matrix_size);

    return 0;
}
