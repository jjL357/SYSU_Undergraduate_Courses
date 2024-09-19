# python版本

import random
import time

# 生成一个大小为 rows x cols 的随机矩阵
def generate_random_matrix(rows, cols):
    matrix = [[random.randint(1, 10) for _ in range(cols)] for _ in range(rows)]
    return matrix

# 执行矩阵乘法
def matrix_multiplication(A, B):
    m = len(A)
    n = len(A[0])
    k = len(B[0])
    C = [[0] * k for _ in range(m)]
    for i in range(m):
        for j in range(k):
            for l in range(n):
                C[i][j] += A[i][l] * B[l][j]
    return C

if __name__ == "__main__":
    m, n, k = map(int, input("Enter the dimensions of matrices (m n k [512, 2048], where A is m x n and B is n x k):").split())

    random.seed() # 初始化随机种子

    A = generate_random_matrix(m, n)
    B = generate_random_matrix(n, k)

    # 执行矩阵乘法并测量时间
    start = time.time()
    C = matrix_multiplication(A, B)
    end = time.time()
    time_taken = (end - start) * 1000

    # 输出计算时间
    print("Time taken for computation: {:.2f} milliseconds".format(time_taken))
