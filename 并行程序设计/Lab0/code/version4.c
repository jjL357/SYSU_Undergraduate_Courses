// C/C++版本 + 调整循环顺序 + 编译时优化

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 为大小为 rows x cols 的矩阵分配内存空间
double** allocateMatrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; ++i) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    return matrix;
}

// 生成一个大小为 rows x cols 的随机矩阵
void generateRandomMatrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = rand() % 10 + 1; // 生成介于 1 和 10 之间的随机数
        }
    }
}

// 初始化矩阵为0
void initializeMatrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = 0;
        }
    }
}

// 执行矩阵乘法(调整循环优化)
void matrixMultiplication(double** A, double** B, double** C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int l = 0; l < n; ++l) {
            for (int j = 0; j < k; ++j) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

// 释放矩阵占用的内存空间
void freeMatrix(double** matrix, int rows) {
    for (int i = 0; i < rows; ++i) {
        free(matrix[i]);
    }
    free(matrix);
}

int main() {
    int m, n, k;
    printf("Enter the dimensions of matrices (m n k [512, 2048], where A is m x n and B is n x k): ");
    scanf("%d %d %d", &m, &n, &k);

    double** A = allocateMatrix(m, n);
    double** B = allocateMatrix(n, k);
    double** C = allocateMatrix(m, k);

    srand(time(NULL)); // 初始化随机种子

    generateRandomMatrix(A, m, n);
    generateRandomMatrix(B, n, k);

    // 初始化矩阵 C 为 0
    initializeMatrix(C, m, k);

    // 执行矩阵乘法并测量时间
    clock_t start = clock();
    matrixMultiplication(A, B, C, m, n, k);
    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    // 输出计算时间
    printf("Time taken for computation: %f milliseconds\n", time_spent);

    // 释放内存空间
    freeMatrix(A, m);
    freeMatrix(B, n);
    freeMatrix(C, m);

    return 0;
}
