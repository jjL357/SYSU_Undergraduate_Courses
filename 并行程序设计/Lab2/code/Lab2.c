#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// 每个进程执行局部的矩阵乘法
void matrixMultiplication(int *A, int *B, int *C, int rows, int cols, int common, int my_rank, int num_procs, int extra_rows ,int *D) {
    if (my_rank < extra_rows) { // 判断是否前extra_rows进程是否要多负责一个进程
        for (int i = my_rank * (rows + 1); i < (my_rank + 1) * (rows + 1); i++) { // 每个进程负责矩阵 A 的对应部分
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < common; k++) {
                    C[i * cols + j] += A[i * common + k] * B[k * cols + j];
                }
                D[i * cols + j] = C[i * cols + j]; 
            }
        }
    } else {
        for (int i = my_rank * rows + extra_rows; i < (my_rank + 1) * rows + extra_rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < common; k++) {
                    C[i * cols + j] += A[i * common + k] * B[k * cols + j];
                }
                D[i * cols + j] = C[i * cols + j]; 
            }
        }
    }
}

int main(int argc, char **argv) {
    int my_rank, num_procs;
    int A_rows, A_cols, B_rows, B_cols;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double start_time, end_time;

    // 只有0号进程接收输入
    if (my_rank == 0) {
        if (argc != 5) { // 运行参数出错
            printf("Usage: %s <A_rows> <A_cols> <B_rows> <B_cols>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(1);
        }

        A_rows = atoi(argv[1]);
        A_cols = atoi(argv[2]);
        B_rows = atoi(argv[3]);
        B_cols = atoi(argv[4]);

        if (A_cols != B_rows) { // A B 维度无法进行矩阵乘法
            printf("Error: Number of columns in matrix A must be equal to number of rows in matrix B.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(1);
        }
    }

    // 广播矩阵维度
    MPI_Bcast(&A_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 计算每个进程负责的行数
    int rows_per_proc = A_rows / num_procs;
    int extra_rows = A_rows % num_procs;

    // 动态分配矩阵内存
    int *A, *B, *C, *D;
    A = (int *)malloc(A_rows * A_cols * sizeof(int));
    B = (int *)malloc(B_rows * B_cols * sizeof(int));
    C = (int *)calloc(A_rows * B_cols, sizeof(int));
    D = (int *)malloc(A_rows * B_cols * sizeof(int));
    //total_D = (int *)malloc(A_rows * B_cols * sizeof(int));

    // 0号进程初始化矩阵 A 和 B
    if (my_rank == 0) {
        srand(time(NULL));
        for (int i = 0; i < A_rows; i++) {
            for (int j = 0; j < A_cols; j++) {
                A[i * A_cols + j] = rand() % 10 + 1; // 生成介于 1 和 10 之间的随机数
            }
        }
        for (int i = 0; i < B_rows; i++) {
            for (int j = 0; j < B_cols; j++) {
                B[i * B_cols + j] = rand() % 10 + 1; // 生成介于 1 和 10 之间的随机数
            }
        }
    }
    for(int i = 0 ;i < A_rows;i++ ){
        for(int j = 0;j< B_cols ;j++){
            C[i * B_cols +j ]=0;
            D[i * B_cols +j ]=0;
            //total_D[i * B_cols +j ]=0;
        }
    }

    // 分发矩阵 A 的数据给各个进程
    int *sendbuf = NULL;
    if (my_rank == 0) {
        sendbuf = (int *)malloc(A_rows * A_cols * sizeof(int));
        for (int i = 0; i < A_rows; i++) {
            for (int j = 0; j < A_cols; j++) {
                sendbuf[i * A_cols + j] = A[i * A_cols + j];
            }
        }
    }

    // 接收缓冲区
    int *recvbuf = (int *)malloc((rows_per_proc + extra_rows) * A_cols * sizeof(int));

    // 分发矩阵 A 的数据
    MPI_Scatter(sendbuf, rows_per_proc  * A_cols, MPI_INT, recvbuf, (rows_per_proc ) * A_cols, MPI_INT, 0, MPI_COMM_WORLD);

    // 将接收到的数据复制到本地的 A 数组
    for(int i = my_rank * rows_per_proc ;i < (my_rank + 1) * rows_per_proc ;i++){
        for(int j = 0;j < A_cols;j++  ){
            A[i * A_cols + j] = recvbuf[(i-my_rank*rows_per_proc) * A_cols + j];
        }
    }
    

    // 释放发送缓冲区和接收缓冲区
    free(sendbuf);
    free(recvbuf);

    // 广播矩阵 B 的数据
    MPI_Bcast(B, B_rows * B_cols, MPI_INT, 0, MPI_COMM_WORLD);

    // 计时开始
    if (my_rank == 0) {
        start_time = MPI_Wtime();
    }
    
    // 计算局部矩阵乘法
    matrixMultiplication(A, B, C, rows_per_proc, B_cols, A_cols, my_rank, num_procs, extra_rows, D);

    // 0号进程收集局部结果
    //MPI_Reduce(D, C, A_rows * B_cols, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // 使用 MPI_Gather 将各个进程的局部结果收集到根进程中
    MPI_Gather(D + rows_per_proc * my_rank*4 , rows_per_proc * B_cols, MPI_INT, C, rows_per_proc  * B_cols, MPI_INT, 0, MPI_COMM_WORLD);
    // 输出矩阵 A、B 和 C
    if (my_rank == 0) {
        printf("A: %dX%d    B: %dX%d    C: %dX%d\n",A_rows,A_cols,B_rows,B_cols,A_rows,A_cols);
       printf("The num of process : %d\n",num_procs);
    //     printf("Matrix A:\n");
    //     printMatrix(A, A_rows, A_cols);

    //     printf("Matrix B:\n");
    //     printMatrix(B, B_rows, B_cols);

    //     printf("Result matrix:\n");
    //     printMatrix(C, A_rows, B_cols);

        end_time = MPI_Wtime();
        printf("Computation time: %f seconds\n", end_time - start_time);
    }

    // 释放内存
    free(A);
    free(B);
    free(C);
    free(D);
    //free(total_D);

    MPI_Finalize();
    return 0;
}
