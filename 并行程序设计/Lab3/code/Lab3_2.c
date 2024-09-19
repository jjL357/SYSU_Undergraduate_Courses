#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <time.h>

#define ARRAY_SIZE_MIN 4
#define ARRAY_SIZE_MAX 134217728 // 128M

int *array;
long long array_size;
long long total_sum = 0;
int num_threads;
pthread_mutex_t sum_mutex; // 定义互斥锁

// 分配数组内存的函数
int *allocate_array(long long size) {
    int *arr = (int *)malloc(size * sizeof(int));
    return arr;
}

// 生成随机整数数组的函数
void generate_random_array(int *arr, long long size) {
    for (long long i = 0; i < size; i++) {
        arr[i] = rand() % 10; // 生成0到99之间的随机整数
    }
}

// 释放数组内存的函数
void free_array(int *arr) {
    free(arr);
}

// 线程函数，计算数组部分的和
void *calculate_partial_sum(void *rank) {
    int my_rank = (long)rank;
    long long rows = array_size / num_threads;
    long long start = rows * my_rank;
    long long end = rows * (my_rank + 1);

    long long partial_sum = 0;
    for (long long i = start; i < end; i++) {
        partial_sum += array[i];
    }

    // 将局部和累加到总和中
    // 使用互斥锁保护临界区
    pthread_mutex_lock(&sum_mutex);
    total_sum += partial_sum;
    pthread_mutex_unlock(&sum_mutex);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("用法: %s <数组大小> <线程数量>\n", argv[0]);
        return 1;
    }
    
    array_size = atoll(argv[1]);
    num_threads = atoi(argv[2]);

    if (array_size < ARRAY_SIZE_MIN || array_size > ARRAY_SIZE_MAX) {
        printf("数组大小必须在%d和%d之间\n", ARRAY_SIZE_MIN, ARRAY_SIZE_MAX);
        return 1;
    }

    // 使用当前时间作为随机数种子
    srand(time(NULL));
    // 初始化互斥锁
    pthread_mutex_init(&sum_mutex, NULL);

    // 分配数组内存并生成随机数组
    array = allocate_array(array_size);
    generate_random_array(array, array_size);

    // 获取开始时间
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // 创建线程
    pthread_t threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, calculate_partial_sum, (void*)i);
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

    // 输出数组和及消耗的时间
    printf("The size of array: %lld\n",array_size);
    //for(int i = 0 ;i < array_size ;i++)printf("%d ",array[i]);
    printf("\n");
    printf("sum: %lld\n", total_sum);
    printf("Computing time: %.2f ms\n", elapsed_time);

    // 销毁互斥锁
    pthread_mutex_destroy(&sum_mutex);


    // 释放数组内存
    free_array(array);

    return 0;
}
