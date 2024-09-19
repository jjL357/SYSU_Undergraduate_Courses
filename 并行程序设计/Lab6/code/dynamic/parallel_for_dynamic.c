#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// 定义结构体来传递参数给函数
struct parallel_args {
    void *functor_args; // 执行函数所需参数
    int start, end, inc,task_completed_count;  // 开始位置 结束位置 增量
    pthread_mutex_t *mutex; // 互斥锁
    pthread_mutex_t *task_mutex; // 互斥锁
    void *(*functor)(void *);
};

//函数参数的结构体
struct functor_args{
    int m; // 维度M
    int n; // 维度N 
    double *w; // 指向矩阵w
    double *mean; //指向平均值mean
    double *diff; // 指向判断收敛的diff
    double *u; //指向辅助矩阵u
};
void *thread_function(void *args) {
    struct parallel_args *thread_args_prt = (struct parallel_args *)args;
    struct parallel_args thread_args;
    thread_args.functor = thread_args_prt->functor;
    thread_args.functor_args = thread_args_prt->functor_args;
    thread_args.inc = thread_args_prt->inc;
    thread_args.mutex = thread_args_prt->mutex;
    thread_args.task_mutex = thread_args_prt->task_mutex;

    int start = thread_args_prt->start;
    int end = thread_args_prt->end;

    void *(*functor)(void *) = thread_args_prt->functor;
    while (1) 
    {
        pthread_mutex_lock(thread_args.task_mutex); // 加锁
        thread_args.start = thread_args_prt->task_completed_count;
        thread_args.end = thread_args.start + thread_args.inc;
        if(thread_args.end > end)thread_args.end = end;
        thread_args_prt->task_completed_count = thread_args.end;
        pthread_mutex_unlock(thread_args.task_mutex); // 解锁
        if (thread_args.start >= end) // 所有任务执行完毕，退出循环
            break;
        (*functor)((void *)&thread_args); // 执行任务
    }
    return NULL;
}

// 打包成动态链接库的入口函数
void parallel_for(int start, int end, int inc, 
                        void *(*functor)(void*), void *arg, int num_threads,pthread_mutex_t*mutex) {
    
    pthread_t threads[num_threads];
    pthread_mutex_t task_mutex;
    // 初始化互斥锁
    pthread_mutex_init(&task_mutex, NULL);

    struct parallel_args args_data;
    args_data.functor_args = (struct functor_args*)arg;
    args_data.start = start;
    args_data.end = end;
    args_data.inc = inc;
    args_data.mutex = mutex;
    args_data.task_mutex = &task_mutex;
    args_data.functor = functor;
    args_data.task_completed_count = 0;

    // 创建线程并执行并行for循环
    for (int i = 0; i < num_threads; ++i) {
        pthread_create(&threads[i], NULL, thread_function, (void *)&args_data);
    }

    // 等待所有线程结束
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
    // 销毁互斥锁
    pthread_mutex_destroy(&task_mutex);
}
