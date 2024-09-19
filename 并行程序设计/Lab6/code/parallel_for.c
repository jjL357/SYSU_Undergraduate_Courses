#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// 定义结构体来传递参数给函数
struct parallel_args {
    void *functor_args; // 执行函数所需参数
    int start, end, inc;  // 开始位置 结束位置 增量
    pthread_mutex_t *mutex; // 互斥锁
};

// 打包成动态链接库的入口函数
void parallel_for(int start, int end, int inc, 
                        void *(*functor)(void*), void *arg, int num_threads,pthread_mutex_t*mutex) {
    
    pthread_t threads[num_threads];
    struct parallel_args args_data[num_threads];

    int total_task = end-start;
    int per_task = total_task / num_threads;

    // 创建线程并执行并行for循环
    for (int i = 0; i < num_threads; ++i) {
        args_data[i].functor_args = arg;
        args_data[i].start = start + i * per_task;
        args_data[i].end = start + (i + 1) * per_task;
        args_data[i].inc = inc;
        args_data[i].mutex = mutex;
        pthread_create(&threads[i], NULL, functor, (void *)&args_data[i]);
    }

    // 等待所有线程结束
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
}
