#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// 结构体定义任务参数
struct parallel_for_args {
    int start;
    int end;
    int inc;
    void *(*functor)(int, void *);
    void *arg;
};

// 线程执行函数
void *thread_function(void *args) {
    struct parallel_for_args *pf_args = (struct parallel_for_args *)args;
    void *(*functor)(int, void *) = pf_args->functor;
    void *arg = pf_args->arg;

    for (int i = pf_args->start; i < pf_args->end; i += pf_args->inc) {
        (*functor)(i, arg);
    }

    return NULL;
}

// parallel_for函数
void parallel_for(int start, int end, int inc,
                  void *(*functor)(int, void *), void *arg, int num_threads) {
    pthread_t threads[num_threads];
    struct parallel_for_args pf_args[num_threads];

    // 计算每个线程的任务范围
    int block_size = (end - start + num_threads - 1) / num_threads;
    for (int i = 0; i < num_threads; i++) {
        pf_args[i].start = start + i * block_size;
        pf_args[i].end = (i == num_threads - 1) ? end : start + (i + 1) * block_size;
        pf_args[i].inc = inc;
        pf_args[i].functor = functor;
        pf_args[i].arg = arg;
    }

    // 创建并启动线程
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, thread_function, (void *)&pf_args[i]);
    }

    // 等待所有线程执行完毕
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}
