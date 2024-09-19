#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

double pi; // 估计出的pi值
long long num_threads; // 线程数目
long long num_total_points; // 随机产生点的个数
long long num_points_inside_circle = 0; // 随机产生点落在圆内的个数
pthread_mutex_t sum_mutex; // 定义互斥锁

// 线程函数，计算pi
void *calculate_pi(void *rank) {
    int my_rank = (long)rank;
    long long per_calculate_num = num_total_points / num_threads;
    long long extra_num = num_total_points % num_threads;
    long long start = my_rank < extra_num ? (per_calculate_num + 1) * my_rank : per_calculate_num * my_rank + extra_num;
    long long end = my_rank < extra_num ? (per_calculate_num + 1) * (my_rank + 1) : per_calculate_num * (my_rank + 1) + extra_num;
    
    unsigned int seed = (unsigned int)time(NULL) + my_rank; // 使用当前时间和线程编号作为种子
    for (long long i = start; i < end; i++) {
        // 使用rand_r保证多线程随机数的安全
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1) {
            // 使用互斥锁保护临界区
            pthread_mutex_lock(&sum_mutex);
            num_points_inside_circle += 1;
            pthread_mutex_unlock(&sum_mutex);
        }
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("用法: %s <产生点的个数> <线程数量>\n", argv[0]);
        return 1;
    }
    
    num_total_points = atoll(argv[1]);
    num_threads = atoll(argv[2]);

    printf("产生点的个数：%lld\n", num_total_points);
    printf("线程数量：%lld\n", num_threads);

    // 初始化互斥锁
    pthread_mutex_init(&sum_mutex, NULL);

    // 获取开始时间
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // 创建线程
    pthread_t threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, calculate_pi, (void*)i);
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

    // 计算估计的π值
    pi = (double)num_points_inside_circle / num_total_points * 4; 
    
    // 输出π的误差值
    double error = fabs(pi - M_PI); 
    printf("The pi: %lf\n", pi);
    printf("Error: %lf\n", error);
    printf("Computing time: %.2f ms\n", elapsed_time);

    // 销毁互斥锁
    pthread_mutex_destroy(&sum_mutex);

    return 0;
}
