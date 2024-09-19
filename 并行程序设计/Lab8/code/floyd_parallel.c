#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>
#include <limits.h>
#define INF INT_MAX

double **graph ;
int n;
double **dist;
int num_threads;


void *floyd_warshall(void *arg) ;
void print_graph_to_csv(double **graph, int n, const char *filename);
void read_graph_from_csv(double **graph, int n, const char *filename);
void free_graph(double **graph, int n);
double calculate_average_degree(double **graph, int n);
void generate_random_graph_to_csv(int n, const char *filename);
void floyd_warshall_parallel(int num_threads);

// Floyd-Warshall算法实现，计算所有节点对之间的最短路径
void *floyd_warshall(void *arg) {
    int thread_id = *(int *)arg;
    int per = n / num_threads;
    int extra = n % num_threads;
    int start = thread_id < extra ? (1 + per) * thread_id : extra + per * thread_id ;
    int end = thread_id < extra ? (1 + per) * (thread_id + 1) : extra + per * (thread_id + 1);
    for (int k = 0; k < n; k++) {
        for (int i = start; i < end; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
return NULL;
}

void floyd_warshall_parallel(int num_threads) {
    
    pthread_t threads[num_threads];
    int thread_ids[num_threads];

    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, floyd_warshall, &thread_ids[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
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
            else fprintf(file, "%d,%d,%.2lf\n", i, j, 0.0);
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

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <n> <num_threads>\n", argv[0]);
        return 1;
    }

    n = atoi(argv[1]); // 从命令行参数获取节点数量
    num_threads = atoi(argv[2]); // 从命令行参数获取线程数量

    int generate_flag = 1;

    // 生成随机邻接矩阵并输出到CSV文件
    if(generate_flag)generate_random_graph_to_csv(n, "random_adjacency_matrix.csv");

    // 从CSV文件中读取邻接矩阵
    graph = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        graph[i] = (double *)malloc(n * sizeof(double));
        // 初始化邻接矩阵，将距离设置为无穷大
        for (int j = 0; j < n; j++) {
            graph[i][j] = INF;
        }
    }
    dist = graph;
    read_graph_from_csv(graph, n, "random_adjacency_matrix.csv");

   // 计时开始
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    floyd_warshall_parallel(num_threads);
    // 计时结束
    clock_gettime(CLOCK_REALTIME, &end);

    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf ( "  Computing time = %fs\n", time_spent );
    // 将结果输出到CSV文件中
    print_graph_to_csv(graph, n, "shortest_paths.csv");

    // 释放动态分配的内存
    free_graph(graph, n);

    return 0;
}
