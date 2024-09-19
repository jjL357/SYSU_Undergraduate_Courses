#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>

void generate_random_matrix(int rows, int cols, double *matrix) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() % 10;
    }
}

int main() {
    int m, n, k;
    printf("Enter the dimensions of matrices (m n k [512, 2048], where A is m x n and B is n x k): ");
    scanf("%d %d %d", &m, &n, &k);


    double *A = (double *)malloc(m * n * sizeof(double));
    double *B = (double *)malloc(n * k * sizeof(double));
    double *C = (double *)malloc(m * k * sizeof(double));

    generate_random_matrix(m, n, A);
    generate_random_matrix(n, k, B);

    clock_t start = clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1.0, A, n, B, k, 0.0, C, k);
    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    // 输出计算时间
    printf("Time taken for computation: %f milliseconds\n", time_spent);

    free(A);
    free(B);
    free(C);

    return 0;
}
