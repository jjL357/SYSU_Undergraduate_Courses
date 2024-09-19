#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <limits.h>
#define INF INT_MAX

void floyd_warshall(double **graph, int n);
void print_graph_to_csv(double **graph, int n, const char *filename);
void read_graph_from_csv(double **graph, int n, const char *filename);
void free_graph(double **graph, int n);
double calculate_average_degree(double **graph, int n);
void generate_random_graph_to_csv(int n, const char *filename);

// Floyd-Warshall算法实现，计算所有节点对之间的最短路径
void floyd_warshall(double **dist, int n) {
    // 创建一个距离矩阵，用于存储最短路径


    // Floyd-Warshall算法
    for (int k = 0; k < n; k++) {
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }


}

// 将邻接矩阵输出到CSV文件中
void print_graph_to_csv(double **graph, int n, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "fail to open %s\n", filename);
        return;
    }

    // 写入标题行
    fprintf(file, "source,target,distance\n");

    // 写入邻接矩阵数据
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (graph[i][j] != INF) {
                fprintf(file, "%d,%d,%.2lf\n", i, j, graph[i][j]);
            }
        }
    }

    fclose(file);
    printf("saved %s\n", filename);
}

// 释放邻接矩阵的动态分配内存
void free_graph(double **graph, int n) {
    for (int i = 0; i < n; i++) {
        free(graph[i]);
    }
    free(graph);
}

// 计算节点的平均度数
double calculate_average_degree(double **graph, int n) {
    double sum_degree = 0;
    int num_edges = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && graph[i][j] != INF) {
                sum_degree += graph[i][j];
                num_edges++;
            }
        }
    }

    return sum_degree / num_edges;
}

// 生成随机邻接矩阵并输出到CSV文件
void generate_random_graph_to_csv(int n, const char *filename) {
    srand(time(NULL)); // 使用当前时间作为随机数种子

    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Fail to open %s\n", filename);
        return;
    }

    // 写入标题行
    fprintf(file, "source,target,distance\n");

    // 生成随机邻接矩阵并将其写入文件
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double distance = 0 + ((double)rand() / RAND_MAX) * (10 - 0);; // 生成1到10的随机距离
                fprintf(file, "%d,%d,%.2lf\n", i, j, distance);
            }
            else fprintf(file, "%d,%d,%.2lf\n", i, j, 0.0);
        }
    }

    fclose(file);
    printf("%s saved\n", filename);
}

// 从CSV文件中读取邻接矩阵
void read_graph_from_csv(double **graph, int n, const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "fail to open %s\n", filename);
        return;
    }

    // 跳过第一行（标题行）
    char line[256];
    fgets(line, sizeof(line), file);

    // 读取邻接矩阵数据
    while (fgets(line, sizeof(line), file) != NULL) {
        int source, target;
        double distance;
        sscanf(line, "%d,%d,%lf", &source, &target, &distance);
        graph[source][target] = distance;
    }

    fclose(file);
}

int main() {
    int n = 2000; // 节点数量0
    int generate_flag = 1;

    // 生成随机邻接矩阵并输出到CSV文件
    if(generate_flag)generate_random_graph_to_csv(n, "random_adjacency_matrix.csv");

    // 从CSV文件中读取邻接矩阵
    double **graph = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        graph[i] = (double *)malloc(n * sizeof(double));
        // 初始化邻接矩阵，将距离设置为无穷大
        for (int j = 0; j < n; j++) {
            graph[i][j] = INF;
        }
    }
    read_graph_from_csv(graph, n, "random_adjacency_matrix.csv");

   // 获取开始时间
    clock_t start, end;
    double cpu_time_used;

    // 记录开始时间
    start = clock();
    // 将读取的邻接矩阵进行Floyd-Warshall算法
    floyd_warshall(graph, n);
    // 记录结束时间
    end = clock();
    // 计算经过的时间（以秒为单位）
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Computing Time : %lf s\n", cpu_time_used);

    // 将结果输出到CSV文件中
    print_graph_to_csv(graph, n, "shortest_paths.csv");

    // 释放动态分配的内存
    free_graph(graph, n);

    return 0;
}
