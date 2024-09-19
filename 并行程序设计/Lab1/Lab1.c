#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
// 每个进程执行局部的矩阵乘法
void matrixMultiplication(int **A, int **B, int **C, int rows, int cols, int common, int my_rank, int num_procs,int extra_rows) {
    if(my_rank < extra_rows){ // 判断是否前extra_rows进程是否要多负责一个进程
        for (int i = my_rank * (rows + 1); i < (my_rank + 1) * (rows + 1); i++) { // 每个进程负责矩阵 A 的对应部分
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < common; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
    else {
        for (int i = my_rank*rows + extra_rows; i < (my_rank + 1) * rows + extra_rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < common; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
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
        if (argc != 5) {//运行参数出错
            printf("Usage: %s <A_rows> <A_cols> <B_rows> <B_cols>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(1);
        }

        A_rows = atoi(argv[1]);
        A_cols = atoi(argv[2]);
        B_rows = atoi(argv[3]);
        B_cols = atoi(argv[4]);

        if (A_cols != B_rows) {// A B 维度无法进行矩阵乘法
            printf("Error: Number of columns in matrix A must be equal to number of rows in matrix B.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(1);
        }
    }

    // 利用点对点和循环实现 广播矩阵维度
    if (my_rank == 0) {
        int dimensions[4] = {A_rows, A_cols, B_rows, B_cols};
        for (int dest = 1; dest < num_procs; dest++) {
            MPI_Send(dimensions, 4, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
    } else {
        int dimensions[4];
        MPI_Recv(dimensions, 4, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        A_rows = dimensions[0];
        A_cols = dimensions[1];
        B_rows = dimensions[2];
        B_cols = dimensions[3];
    }

    // 计算每个进程负责的行数
    int rows_per_proc = A_rows / num_procs;
    int extra_rows = A_rows % num_procs;

    // 动态分配矩阵内存
    int **A, **B, **C;
    A = (int **)malloc(A_rows * sizeof(int *));
    B = (int **)malloc(B_cols * sizeof(int *));
    C = (int **)malloc(A_rows * sizeof(int *));
    for (int i = 0; i < A_rows; i++) {
        A[i] = (int *)malloc(A_cols * sizeof(int));
        C[i] = (int *)calloc(B_cols, sizeof(int));
    }
    for (int i = 0; i < B_cols; i++) {
        B[i] = (int *)malloc(B_cols * sizeof(int));
    }

    // 0号进程初始化矩阵 A 和 B
    if (my_rank == 0) {
        srand(time(NULL));
        //printf("Matrix A:\n");
        for (int i = 0; i < A_rows; i++) {
            for (int j = 0; j < A_cols; j++) {
                A[i][j] = rand() % 10 + 1; // 生成介于 1 和 10 之间的随机数
            }
        }

        //printf("Matrix B:\n");
        for (int i = 0; i < B_rows; i++) {
            for (int j = 0; j < B_cols; j++) {
                B[i][j] = rand() % 10 + 1; // 生成介于 1 和 10 之间的随机数
            }
        }
    }

    

    // 发送矩阵 A 的部分数据给其他进程
    if (my_rank == 0) {
        for (int dest = 1; dest < num_procs; dest++) {
            if(dest < extra_rows){// 判断是否前extra_rows进程是否要多负责一个进程
                for (int i = dest * (rows_per_proc + 1) ; i < (dest + 1) * (rows_per_proc + 1); i++) {
                    MPI_Send(A[i], A_cols, MPI_INT, dest, 0, MPI_COMM_WORLD);
                }
            }
            else{
                for (int i = dest * rows_per_proc + extra_rows; i < (dest + 1) * rows_per_proc + extra_rows; i++) {
                    MPI_Send(A[i], A_cols, MPI_INT, dest, 0, MPI_COMM_WORLD);
                }
            }
        }
        
    } else {
        
            if(my_rank < extra_rows){// 判断是否前extra_rows进程是否要多负责一个进程
                for (int i = my_rank * (rows_per_proc + 1) ; i < (my_rank + 1) * (rows_per_proc + 1); i++) {
                    MPI_Recv(A[i], A_cols, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            else{
                for (int i = my_rank * rows_per_proc + extra_rows; i < (my_rank + 1) * rows_per_proc + extra_rows; i++) {
                    MPI_Recv(A[i], A_cols, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        

    }
    

    // 发送矩阵 B 的数据给其他进程
    if (my_rank == 0) {
        for (int dest = 1; dest < num_procs; dest++) {
            for (int i = 0; i < B_rows; i++) {
                MPI_Send(B[i], B_cols, MPI_INT, dest, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        for (int i = 0; i < B_rows; i++) {
            MPI_Recv(B[i], B_cols, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    
    // 计时开始
    if (my_rank == 0) {
        start_time = MPI_Wtime();
    }

    // 计算局部矩阵乘法
    matrixMultiplication(A, B, C, rows_per_proc, B_cols, A_cols, my_rank, num_procs,extra_rows);

    // 发送局部结果给0号进程
    if (my_rank != 0) {
        if(my_rank < extra_rows){ // 判断是否前extra_rows进程是否要多负责一个进程
            for (int i = my_rank * (rows_per_proc + 1); i < (my_rank + 1) * (rows_per_proc + 1); i++) {
                MPI_Send(C[i], B_cols, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
        else {
            for (int i = my_rank * rows_per_proc + extra_rows; i < (my_rank + 1) * rows_per_proc + extra_rows; i++) {
                MPI_Send(C[i], B_cols, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
    } else {  // 判断是否前extra_rows进程是否要多负责一个进程
        for (int source = 1; source < num_procs; source++) {
            if(source < extra_rows){
                for (int i = source * (rows_per_proc + 1); i < (source + 1) * (rows_per_proc + 1); i++) {
                    MPI_Recv(C[i], B_cols, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            else {
                for (int i = source * rows_per_proc + extra_rows; i < (source + 1) * rows_per_proc + extra_rows; i++) {
                    MPI_Recv(C[i], B_cols, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
         }
    }

    // // // 每个进程输出负责计算的 C 的结果
    // printf("Process %d calculated C:\n", my_rank);
    // for (int i = my_rank * rows_per_proc; i < (my_rank + 1) * rows_per_proc; i++) {
    //     printf("Row %d:\n",i);
    //     for (int j = 0; j < B_cols; j++) {
    //         printf("%d ", C[i][j]);
    //     }
    //     printf("\n");
    // }

    // 0号进程输出结果和运算时间
    if (my_rank == 0) {
        end_time = MPI_Wtime();

        // printf("Matrix A:\n");
        // for (int i = 0; i < A_rows; i++) {
        //     for (int j = 0; j < A_cols; j++) {
        //         printf("%d ", A[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");

        // printf("Matrix B:\n");
        // for (int i = 0; i < B_rows; i++) {
        //     for (int j = 0; j < B_cols; j++) {
        //         printf("%d ", B[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");

        // printf("Result matrix:\n");
        // for (int i = 0; i < A_rows; i++) {
        //     for (int j = 0; j < B_cols; j++) {
        //         printf("%d ", C[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");

        printf("Computation time: %f seconds\n", end_time - start_time);
    }

    // 释放内存
    for (int i = 0; i < A_rows; i++) {
        free(A[i]);
        free(C[i]);
    }
    for (int i = 0; i < B_cols; i++) {
        free(B[i]);
    }
    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return 0;
}
