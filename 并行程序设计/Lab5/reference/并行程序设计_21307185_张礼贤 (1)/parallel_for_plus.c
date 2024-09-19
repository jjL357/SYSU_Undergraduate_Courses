#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct parallel_for_args {
    int start;
    int end;
    int inc;
    void *(*functor)(int, void *);
    void *arg;
};

pthread_mutex_t task_mutex; // 任务分配锁

void *thread_function(void *args) {
    struct parallel_for_args *pf_args = (struct parallel_for_args *)args;
    void *(*functor)(int, void *) = pf_args->functor;
    void *arg = pf_args->arg;
    int i;

    while (1) 
    {
        pthread_mutex_lock(&task_mutex); // 加锁
        i = pf_args->start; // 获取任务起始位置
        pf_args->start += pf_args->inc; // 更新任务起始位置
        pthread_mutex_unlock(&task_mutex); // 解锁
        if (i >= pf_args->end) // 所有任务执行完毕，退出循环
            break;
        (*functor)(i, arg); // 执行任务
    }
    return NULL;
}

void parallel_for(int start, int end, int inc,
                  void *(*functor)(int, void *), void *arg, int num_threads) {
    pthread_t threads[num_threads];
    struct parallel_for_args pf_args[num_threads];

    pthread_mutex_init(&task_mutex, NULL); // 初始化任务分配锁

    // 分配任务给各个线程
    int block_size = (end - start + num_threads - 1) / num_threads;
    for (int i = 0; i < num_threads; i++) {
        pf_args[i].start = start + i * block_size;
        pf_args[i].end = (i == num_threads - 1) ? end : start + (i + 1) * block_size;
        pf_args[i].inc = inc;
        pf_args[i].functor = functor;
        pf_args[i].arg = arg;

        pthread_create(&threads[i], NULL, thread_function, (void *)&pf_args[i]);
    }

    // 等待所有线程执行完毕
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&task_mutex); // 销毁任务分配锁
}
