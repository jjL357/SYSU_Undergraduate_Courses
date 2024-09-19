# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <pthread.h>
extern void parallel_for(int start, int end, int step, void (*func)(int, void*), void* arg, int num_threads,pthread_mutex_t *mutex;);

//函数参数的结构体
struct functor_args{
    int m; // 维度M
    int n; // 维度N 
    double *w; // 指向矩阵w
    double *mean; //指向平均值mean
    double *diff; // 指向判断收敛的diff
    double *u; //指向辅助矩阵u
};
// parallel_for参数结构体
struct parallel_args {
    void *functor_args;
    int start, end, inc; 
    pthread_mutex_t *mutex; // 锁
};
// 初始化w矩阵
void *set_w(void *args){
  struct parallel_args *para_args = (struct parallel_args *)args;
  int start = para_args->start;
  int end = para_args->end;
  int inc = para_args->inc;
  pthread_mutex_t *mutex = para_args->mutex; 

  struct functor_args* func_args = (struct functor_args *)para_args->functor_args;
  int m = func_args->m;
  int n = func_args->n;
  double (*w)[n] = (double (*)[n])func_args->w;
    // 给w矩阵赋值
    for ( int i = start; i < end; i++ )
    {
      w[i][0] = 100.0;
    }

    for ( int i = start; i < end; i++ )
    {
      w[i][n-1] = 100.0;
    }

    for (int j = start; j < end; j++ )
    {
      w[m-1][j] = 100.0;
    }

    for (int j = start; j < end; j++ )
    {
      w[0][j] = 0.0;
    }

  return NULL;
}

// 计算矩阵元素和,用mean记录
void *calculate_sum(void *args){
  struct parallel_args *para_args = (struct parallel_args *)args;
  int start = para_args->start;
  int end = para_args->end;
  int inc = para_args->inc;
  pthread_mutex_t *mutex = para_args->mutex; 

  struct functor_args* func_args = (struct functor_args *)para_args->functor_args;
  int m = func_args->m;
  int n = func_args->n;
  double (*w)[n] = (double (*)[n])func_args->w;
  double*mean = func_args->mean;
  
    for (int i = start ; i < end; i++ )
    {
      pthread_mutex_lock(mutex); // 上锁
      *mean = *mean + w[i][0] + w[i][n-1];
      pthread_mutex_unlock(mutex); // 解锁
    }

    for (int j = start; j < end; j++ )
    {
      pthread_mutex_lock(mutex); // 上锁
      *mean = *mean + w[m-1][j] + w[0][j];
      pthread_mutex_unlock(mutex);// 解锁
    }
  
  return NULL;
}
// 给矩阵w中间元素赋值平均值mean
void *set_mean(void *args){
  struct parallel_args *para_args = (struct parallel_args *)args;
  int start = para_args->start;
  int end = para_args->end;
  int inc = para_args->inc;
  pthread_mutex_t *mutex = para_args->mutex; 

  struct functor_args* func_args = (struct functor_args *)para_args->functor_args;
  int m = func_args->m;
  int n = func_args->n;
  double (*w)[n] = (double (*)[n])func_args->w;
  double*mean = func_args->mean;
   
     for (int i = start; i < end; i++ )
    {         
      if(i == m - 1||i==0)continue;
      for (int j = 1; j < n - 1; j++ )
      {
        w[i][j] = *mean;
      }
    }
  
  return NULL;
}
// 将矩阵w的值赋值给u
void *w_to_u(void *args){
  struct parallel_args *para_args = (struct parallel_args *)args;
  int start = para_args->start;
  int end = para_args->end;
  int inc = para_args->inc;
  pthread_mutex_t *mutex = para_args->mutex; 

  struct functor_args* func_args = (struct functor_args *)para_args->functor_args;
  int m = func_args->m;
  int n = func_args->n;
  double (*w)[n] = (double (*)[n])func_args->w;
  double*mean = func_args->mean;
 
  double (*u)[n] = (double (*)[n])func_args->u;

     for (int i = start; i < end; i++ )
    {        
      for (int j = 0; j < n ; j++ )
      {
        u[i][j] = w[i][j];
      }
    }
  
  return NULL;
}
// 模拟热传导
void *heat_transfer(void *args){
  struct parallel_args *para_args = (struct parallel_args *)args;
  int start = para_args->start;
  int end = para_args->end;
  int inc = para_args->inc;
  pthread_mutex_t *mutex = para_args->mutex; 

  struct functor_args* func_args = (struct functor_args *)para_args->functor_args;
  int m = func_args->m;
  int n = func_args->n;
   double (*w)[n] = (double (*)[n])func_args->w;
  double*mean = func_args->mean;
 
   double (*u)[n] = (double (*)[n])func_args->u;

     for (int i = start; i < end; i++ )
    {         
      if(i == m - 1||i==0)continue;
      for (int j = 1; j < n - 1; j++ )
      {
        w[i][j] = ( u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] ) / 4.0;
      }
    }
  
  return NULL;
}
//  计算diff
void *calculate_diff(void *args){
  struct parallel_args *para_args = (struct parallel_args *)args;
  int start = para_args->start;
  int end = para_args->end;
  int inc = para_args->inc;
  pthread_mutex_t *mutex = para_args->mutex; 

  struct functor_args* func_args = (struct functor_args *)para_args->functor_args;
  int m = func_args->m;
  int n = func_args->n;
   double (*w)[n] = (double (*)[n])func_args->w;
   double (*u)[n] = (double (*)[n])func_args->u;
  double*mean = func_args->mean;
  
  double *diff = func_args->diff;
  
    
     for (int i = start; i < end; i++ )
    {         
      if(i == m - 1||i==0)continue;
      for (int j = 1; j < n - 1; j++ )
      { 
        if(*diff<fabs(w[i][j] - u[i][j])){
           pthread_mutex_lock(mutex);
          *diff = fabs(w[i][j] - u[i][j]);
           pthread_mutex_unlock(mutex);
      }
    }
    }
  return NULL;
}

int main ( int argc, char *argv[] );

/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for HEATED_PLATE_PTHREAD.

  Discussion:

    This code solves the steady state heat equation on a rectangular region.

    The sequential version of this program needs approximately
    18/epsilon iterations to complete. 


    The physical region, and the boundary conditions, are suggested
    by this diagram;

                   W = 0
             +------------------+
             |                  |
    W = 100  |                  | W = 100
             |                  |
             +------------------+
                   W = 100

    The region is covered with a grid of M by N nodes, and an N by N
    array W is used to record the temperature.  The correspondence between
    array indices and locations in the region is suggested by giving the
    indices of the four corners:

                  I = 0
          [0][0]-------------[0][N-1]
             |                  |
      J = 0  |                  |  J = N-1
             |                  |
        [M-1][0]-----------[M-1][N-1]
                  I = M-1

    The steady state solution to the discrete heat equation satisfies the
    following condition at an interior grid point:

      W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    where "Central" is the index of the grid point, "North" is the index
    of its immediate neighbor to the "north", and so on.
   
    Given an approximate solution of the steady state heat equation, a
    "better" solution is given by replacing each interior point by the
    average of its 4 neighbors - in other words, by using the condition
    as an ASSIGNMENT statement:

      W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    If this process is repeated often enough, the difference between successive 
    estimates of the solution will go to zero.

    This program carries out such an iteration, using a tolerance specified by
    the user, and writes the final estimate of the solution to a file that can
    be used for graphic processing.

  Licensing:

    This code is distributed under the MIT license. 

  Modified:

    18 October 2011

  Author:

    Original C version by Michael Quinn.
    This C version by John Burkardt.

  Reference:

    Michael Quinn,
    Parallel Programming in C with MPI and PTHREAD,
    McGraw-Hill, 2004,
    ISBN13: 978-0071232654,
    LC: QA76.73.C15.Q55.

  Local parameters:

    Local, double DIFF, the norm of the change in the solution from one iteration
    to the next.

    Local, double MEAN, the average of the boundary values, used to initialize
    the values of the solution in the interior.

    Local, double U[M][N], the solution at the previous iteration.

    Local, double W[M][N], the solution computed at the latest iteration.
*/
{
# define M 500
# define N 500

  double diff;
  double epsilon = 0.001;
  int i;
  int iterations;
  int iterations_print;
  int j;
  double mean;
  double my_diff;
  double u[M][N];
  double w[M][N];
  double wtime;
  int num_threads = atoi(argv[1]);
  printf ( "\n" );
  printf ( "HEATED_PLATE_PTHREAD\n" );
  printf ( "  C/PTHREAD version\n" );
  printf ( "  A program to solve for the steady state temperature distribution\n" );
  printf ( "  over a rectangular plate.\n" );
  printf ( "\n" );
  printf ( "  Spatial grid of %d by %d points.\n", M, N );
  printf ( "  The iteration will be repeated until the change is <= %e\n", epsilon ); 
  //printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads =              %d\n", num_threads);
/*
  Set the boundary values, which don't change. 
*/
  
  mean = 0.0;

  pthread_mutex_t mutex;
  // 初始化互斥锁
  pthread_mutex_init(&mutex, NULL);

  struct functor_args fun_args;
 
  fun_args.m = M;
  fun_args.n = N;
  // return 0;
  fun_args.mean = &mean;
  
  fun_args.w = (double*)w;
  fun_args.u = (double*)u;
  //fun_args.diff = 0;


  //printf("SET_W\n");
  parallel_for(0,M,1,set_w,&fun_args,num_threads,&mutex);
  
  // omp version
  // #pragma omp parallel shared ( w ) private ( i, j )
  //   {
  // #pragma omp for
  //     for ( i = 1; i < M - 1; i++ )
  //     {
  //       w[i][0] = 100.0;
  //     }
  // #pragma omp for
  //     for ( i = 1; i < M - 1; i++ )
  //     {
  //       w[i][N-1] = 100.0;
  //     }
  // #pragma omp for
  //     for ( j = 0; j < N; j++ )
  //     {
  //       w[M-1][j] = 100.0;
  //     }
  // #pragma omp for
  //     for ( j = 0; j < N; j++ )
  //     {
  //       w[0][j] = 0.0;
  //     }



/*
  Average the boundary values, to come up with a reasonable
  initial value for the interior.
*/
//printf("CALCULATE_SUM\n");
  parallel_for(0,M,1,calculate_sum,&fun_args,num_threads,&mutex);
//omp version
// #pragma omp for reduction ( + : mean )
//     for ( i = 1; i < M - 1; i++ )
//     {
//       mean = mean + w[i][0] + w[i][N-1];
//     }
// #pragma omp for reduction ( + : mean )
//     for ( j = 0; j < N; j++ )
//     {
//       mean = mean + w[M-1][j] + w[0][j];
//     }
//   }

/*
  PTHREAD note:
  You cannot normalize MEAN inside the parallel region.  It
  only gets its correct value once you leave the parallel region.
  So we interrupt the parallel region, set MEAN, and go back in.
*/
  mean = mean / ( double ) ( 2 * M + 2 * N - 4 );
  printf ( "\n" );
  printf ( "  MEAN = %f\n", mean );
    /* 
      Initialize the interior solution to the mean value.
    */
    //printf("SET_MEAN\n");
      parallel_for(0,M,1,set_mean,&fun_args,num_threads,&mutex);
    // omp version
    // #pragma omp parallel shared ( mean, w ) private ( i, j )
    //   {
    // #pragma omp for
    //     for ( i = 1; i < M - 1; i++ )
    //     {
    //       for ( j = 1; j < N - 1; j++ )
    //       {
    //         w[i][j] = mean;
    //       }
    //     }
    //   }


    /*
      iterate until the  new solution W differs from the old solution U
      by no more than EPSILON.
    */
  iterations = 0;
  iterations_print = 1;
  printf ( "\n" );
  printf ( " Iteration  Change\n" );
  printf ( "\n" );
  

  // 计时开始
  struct timespec start, end;

  clock_gettime(CLOCK_REALTIME, &start);
  diff = epsilon;
  fun_args.diff = &diff;
  while ( epsilon <= diff )
  {
  //printf("W_TO_U&&HEAT_TRANSFER\n");
  parallel_for(0,M,1,w_to_u,&fun_args,num_threads,&mutex);
  parallel_for(0,M,1,heat_transfer,&fun_args,num_threads,&mutex);
    // omp version
    // # pragma omp parallel shared ( u, w ) private ( i, j )
    //     {
    // /*
    //   Save the old solution in U.
    // */
    // # pragma omp for
    //       for ( i = 0; i < M; i++ ) 
    //       {
    //         for ( j = 0; j < N; j++ )
    //         {
    //           u[i][j] = w[i][j];
    //         }
    //       }
    // /*
    //   Determine the new estimate of the solution at the interior points.
    //   The new solution W is the average of north, south, east and west neighbors.
    // */
    // # pragma omp for
    //       for ( i = 1; i < M - 1; i++ )
    //       {
    //         for ( j = 1; j < N - 1; j++ )
    //         {
    //           w[i][j] = ( u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] ) / 4.0;
    //         }
    //       }
    //     }

    /*
      C and C++ cannot compute a maximum as a reduction operation.

      Therefore, we define a private variable MY_DIFF for each thread.
      Once they have all computed their values, we use a CRITICAL section
      to update DIFF.
    */
    diff = 0.0;
    //printf("CALCULATE_DIFF\n");
    parallel_for(0,M,1,calculate_diff,&fun_args,num_threads,&mutex);
    //omp version
    // # pragma omp parallel shared ( diff, u, w ) private ( i, j, my_diff )
    //     {
    //       my_diff = 0.0;
    // # pragma omp for
    //       for ( i = 1; i < M - 1; i++ )
    //       {
    //         for ( j = 1; j < N - 1; j++ )
    //         {
    //           if ( my_diff < fabs ( w[i][j] - u[i][j] ) )
    //           {
    //             my_diff = fabs ( w[i][j] - u[i][j] );
    //           }
    //         }
    //       }
    // # pragma omp critical
    //       {
    //         if ( diff < my_diff )
    //         {
    //           diff = my_diff;
    //         }
    //       }
    //     }

    iterations++;
    if ( iterations == iterations_print )
    {
      printf ( "  %8d  %f\n", iterations, diff );
      iterations_print = 2 * iterations_print;
    }
  } 
 // 计时结束
  clock_gettime(CLOCK_REALTIME, &end);

  double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf ( "\n" );
  printf ( "  %8d  %f\n", iterations, diff );
  printf ( "\n" );
  printf ( "  Error tolerance achieved.\n" );
  printf ( "  Wallclock time = %f\n", time_spent );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "HEATED_PLATE_PTHREAD:\n" );
  printf ( "  Normal end of execution.\n" );
  
  // 销毁互斥锁
  pthread_mutex_destroy(&mutex);
  return 0;

# undef M
# undef N
}
