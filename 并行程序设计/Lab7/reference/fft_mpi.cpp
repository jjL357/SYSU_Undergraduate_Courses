# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <cstring>
# include <omp.h>
# include <mpi.h>

using namespace std;

int main ( int argc, char * argv[] );
void ccopy ( int n, double x[], double y[] );
void cfft2 ( int n, double x[], double y[], double w[], double sgn, int num_transpose[], int my_rank, int process_num);
void cffti ( int n, double w[] );
double cpu_time ( void );
double ggl ( double *ds );
void step ( int n, int mj, double a[], double b[], double c[], double d[], 
  double w[], double sgn, int num_transpose[], int start, int end);
void timestamp ( );


int main ( int argc, char * argv[] )

//  Purpose:
//
//    MAIN is the main program for FFT_SERIAL.
//
//  Discussion:
//
//    The complex data in an N vector is stored as pairs of values in a
//    real vector of length 2*N.

{
  MPI_Init(&argc, &argv);
  int my_rank;
  int process_num;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_num);	
  
  double ctime;
  double ctime1;
  double ctime2;
  double error;
  int first;
  double flops;
  double fnm1;
  int i;
  int icase;
  int it;
  int ln2;
  double mflops;
  int n;
  int nits = 10000;
  static double seed;
  double sgn;
  double *w;
  double *x;
  double *y;
  double *z;
  double z0;
  double z1;
  
  int *num_transpose;

  if (my_rank == 0) {
  	timestamp ( );
    cout << "\n";
    cout << "FFT_SERIAL\n";
    cout << "  C++ version\n";
    cout << "\n";
    cout << "  Demonstrate an implementation of the Fast Fourier Transform\n";
    cout << "  of a complex data vector.\n";
  //
  //  Prepare for tests.
  //
    cout << "\n";
    cout << "  Accuracy check:\n";
    cout << "\n";
    cout << "    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n";
    cout << "\n";
    cout << "             N      NITS    Error         Time          Time/Call     MFLOPS\n";
    cout << "\n";

    seed  = 331.0;
    n = 1;
  }

  MPI_Bcast(&seed, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

//
//  LN2 is the log base 2 of N.  Each increase of LN2 doubles N.
//
  for ( ln2 = 1; ln2 <= 20; ln2++ )
  {
    n = 2 * n;
//
//  Allocate storage for the complex arrays W, X, Y, Z.  
//
//  We handle the complex arithmetic,
//  and store a complex number as a pair of doubles, a complex vector as a doubly
//  dimensioned array whose second dimension is 2. 
//
    w = new double[  n];
    x = new double[2*n];
    y = new double[2*n];
    z = new double[2*n];

    memset(w, 0, sizeof(double)*n);
    memset(x, 0, sizeof(double)*n*2);
    memset(y, 0, sizeof(double)*n*2);
    memset(z, 0, sizeof(double)*n*2);

	  num_transpose = new int[n];

    if (my_rank == 0) first = 1;
    MPI_Bcast(&first, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for ( icase = 0; icase < 2; icase++ )
    {
      
      if (my_rank == 0) {
        if ( first )
        {
          for ( i = 0; i < 2 * n; i = i + 2 )
          {
            z0 = ggl ( &seed );
            z1 = ggl ( &seed );
            x[i] = z0;
            z[i] = z0;
            x[i+1] = z1;
            z[i+1] = z1;
          }
        } 
        else
        {
          for ( i = 0; i < 2 * n; i = i + 2 )
          {
            z0 = 0.0;
            z1 = 0.0;
            x[i] = z0;
            z[i] = z0;
            x[i+1] = z1;
            z[i+1] = z1;
          }
        }

        cffti ( n, w );
	  

        for (int i = 0; i < n; i++) 
			num_transpose[i] = i;
        int i, j, k;
        for(i = 1, j = n / 2; i < n-1; i++, j = j ^ k) {
	    	if(i < j) {
	    	swap(num_transpose[i], num_transpose[j]);
        }
          for(k = n / 2; j&k; j = j ^ k, k /= 2) 
		  	continue;
        }
      }
      
//
//  Initialize the sine and cosine tables.
//

       // pack 与 unpack 方式的消息传递 
	   int pos =0;
       int buffer_size = 5 * n * sizeof(double) + 100;
       void * buffer = malloc(buffer_size);
       if (my_rank == 0) {
         MPI_Pack(x, 2*n, MPI_DOUBLE, buffer, buffer_size, &pos, MPI_COMM_WORLD);
         MPI_Pack(z, 2*n, MPI_DOUBLE, buffer, buffer_size, &pos, MPI_COMM_WORLD);
         MPI_Pack(w, n, MPI_DOUBLE, buffer, buffer_size, &pos, MPI_COMM_WORLD);
         for (int id = 1; id < process_num; id++) {
           MPI_Send(buffer, pos, MPI_PACKED, id, 0, MPI_COMM_WORLD);
         }
       } else {
         MPI_Status status;
         MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
         int size;
         MPI_Get_count(&status, MPI_PACKED, &size);
         MPI_Recv(buffer, size, MPI_PACKED, 0, 0, MPI_COMM_WORLD, &status);
         MPI_Unpack(buffer, size, &pos, x, 2*n, MPI_DOUBLE, MPI_COMM_WORLD);
         MPI_Unpack(buffer, size, &pos, z, 2*n, MPI_DOUBLE, MPI_COMM_WORLD);
         MPI_Unpack(buffer, size, &pos, w, n, MPI_DOUBLE, MPI_COMM_WORLD);
       }
	  
//      MPI_Bcast(x, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//      MPI_Bcast(z, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(num_transpose, n, MPI_INT, 0, MPI_COMM_WORLD);
//      MPI_Bcast(w, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//
//  Transform forward, back 
//
      int i, j, k;
      if ( first )
      {
        sgn = + 1.0;
        if(my_rank == 0){
        for(i = 1, j = n/2, k; i < n-1; i++, j = k ^ j) {
		    if(i < j) {
		      swap(x[2*i], x[2*j]);
			  swap(x[2*i+1], x[2*j+1]);
		  	}
		    for(k = n/2; j&k; j = k ^ j, k /= 2) 
			  continue;
		  }
	    }
	    MPI_Bcast(x, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        cfft2 ( n, x, y, w, sgn, num_transpose, my_rank, process_num);
        sgn = - 1.0;
        if(my_rank == 0){
        for(i = 1, j = n/2, k; i < n-1; i++, j = k ^ j) {
		    if(i < j) {
		      swap(y[2*i], y[2*j]);
			  swap(y[2*i+1], y[2*j+1]);
		  	}
		    for(k = n/2; j&k; j = k ^ j, k /= 2) 
			  continue;
		  }
	    }
	    MPI_Bcast(y, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        cfft2 ( n, y, x, w, sgn, num_transpose, my_rank, process_num);
// 
//  Results should be same as initial multiplied by N.
//
        if (my_rank == 0) {
          fnm1 = 1.0 / ( double ) n;
          error = 0.0;
          for ( i = 0; i < 2 * n; i = i + 2 )
          {
            error = error 
            + pow ( z[i]   - fnm1 * x[i], 2 )
            + pow ( z[i+1] - fnm1 * x[i+1], 2 );
          }
          error = sqrt ( fnm1 * error );
          cout << "  " << setw(12) << n
              << "  " << setw(8) << nits
              << "  " << setw(12) << error;
          first = 0;
        }
        MPI_Bcast(&first, 1, MPI_INT, 0, MPI_COMM_WORLD);
      }
      else
      {
        ctime1 = cpu_time ( );
        for ( it = 0; it < nits; it++ )
        {
          sgn = + 1.0;

          for(i = 1, j = n/2, k; i < n-1; i++, j = k ^ j) {
		    if(i < j) {
		      swap(x[2*i], x[2*j]);
			  swap(x[2*i+1], x[2*j+1]);
		  	}
		    for(k = n/2; j&k; j = k ^ j, k /= 2) 
			  continue;
		  }
	      MPI_Bcast(x, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          MPI_Bcast(w, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          cfft2 ( n, x, y, w, sgn, num_transpose, my_rank, process_num);
          for(i = 1, j = n/2, k; i < n-1; i++, j = k ^ j) {
		    if(i < j) {
		      swap(y[2*i], y[2*j]);
			  swap(y[2*i+1], y[2*j+1]);
		  	}
		    for(k = n/2; j&k; j = k ^ j, k /= 2) 
			  continue;
		  }
	      MPI_Bcast(y, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          
          sgn = - 1.0;
          cfft2 ( n, y, x, w, sgn, num_transpose, my_rank, process_num);
        }
        ctime2 = cpu_time ( );
        ctime = ctime2 - ctime1;

        flops = 2.0 * ( double ) nits * ( 5.0 * ( double ) n * ( double ) ln2 );

        mflops = flops / 1.0E+06 / ctime;
        if (my_rank == 0) {
          cout << "  " << setw(12) << ctime
               << "  " << setw(12) << ctime / ( double ) ( 2 * nits )
               << "  " << setw(12) << mflops << "\n";
        }
      }
    }
    if ( ( ln2 % 4 ) == 0 ) 
    {
      nits = nits / 10;
    }
    if ( nits < 1 ) 
    {
      nits = 1;
    }
    
    delete [] num_transpose;
    
    delete [] w;
    delete [] x;
    delete [] y;
    delete [] z;
  }

  if (my_rank == 0) {
    cout << "\n";
    cout << "FFT_SERIAL:\n";
    cout << "  Normal end of execution.\n";
    cout << "\n";
    timestamp ( );
  }

  MPI_Finalize();
  return 0;
}
//****************************************************************************80

void ccopy ( int n, double x[], double y[] )

//****************************************************************************80
//
//  Purpose:
//
//    CCOPY copies a complex vector.
//
//  Discussion:
//
//    The "complex" vector A[N] is actually stored as a double vector B[2*N].
//
//    The "complex" vector entry A[I] is stored as:
//
//      B[I*2+0], the real part,
//      B[I*2+1], the imaginary part.
//
//  Parameters:
//
//    Input, int N, the length of the "complex" array.
//
//    Input, double X[2*N], the array to be copied.
//
//    Output, double Y[2*N], a copy of X.
//
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    y[i*2+0] = x[i*2+0];
    y[i*2+1] = x[i*2+1];
   }
  return;
}
//****************************************************************************80

void receiveData(int source, int multiplier, double *data_array, MPI_Status &status) {
    MPI_Recv(data_array + multiplier, multiplier, MPI_DOUBLE, source, 21307077, MPI_COMM_WORLD, &status);
}

void sendDataToParent(int rank, int multiplier, double *data_array) {
    MPI_Send(data_array + multiplier * rank, multiplier, MPI_DOUBLE, rank / 2, 21307077, MPI_COMM_WORLD);
}

void receiveDataFromChildren(int rank, int multiplier, double *data_array, MPI_Status &status) {
    MPI_Recv(data_array + 2 * rank * multiplier, multiplier, MPI_DOUBLE, 2 * rank, 21307077, MPI_COMM_WORLD, &status);
    MPI_Recv(data_array + (2 * rank + 1) * multiplier, multiplier, MPI_DOUBLE, 2 * rank + 1, 21307077, MPI_COMM_WORLD, &status);
}

// 返回一个bool，指示是否应该终止调用方的执行
bool processData(int total_procs, int current_rank, int level_size, int data_multiplier, double *data_array) {
    MPI_Status process_status;

    // 如果进程数量大于当前层级的处理单元数
    // 如果是主进程
    if (current_rank == 0) {
        receiveData(1, data_multiplier, data_array, process_status);
    }
    // 如果是中间层级的进程
    else if (current_rank < 2 * level_size) {
        sendDataToParent(current_rank, data_multiplier, data_array);

        // 如果是非叶子节点
        if (current_rank < level_size) {
            receiveDataFromChildren(current_rank, data_multiplier, data_array, process_status);
        } 
		else {
            // 非叶子节点但不需要进一步处理
            return true;
        }
    } 
	else {
        // 处理器超出当前层级范围，不需要继续处理
        return true;
    }
    // 继续执行后续逻辑
    return false;
}


void cfft2 ( int n, double x[], double y[], double w[], double sgn, int num_transpose[], int my_rank, int process_num)

//****************************************************************************80
//
//  Purpose:
//
//    CFFT2 performs a complex Fast Fourier Transform.
//
//  Parameters:
//
//    Input, int N, the size of the array to be transformed.
//
//    Input/output, double X[2*N], the data to be transformed.  
//    On output, the contents of X have been overwritten by work information.
//
//    Output, double Y[2*N], the forward or backward FFT of X.
//
//    Input, double W[N], a table of sines and cosines.
//
//    Input, double SGN, is +1 for a "forward" FFT and -1 for a "backward" FFT.
//
{
  int j;
  int m;
  int mj;
  int tgle;

  int i, k, lj, div, start, end;

	
   m = ( int ) ( log ( ( double ) n ) / log ( 1.99 ) );
   mj   = 1;
//
//  Toggling switch for work array.
//
  tgle = 1;

  lj = n / (mj * 2);
  
  if (process_num > lj){
  	bool ret_flag = processData(process_num, my_rank, lj, mj*2, x);

    // 如果函数返回true，结束当前函数的执行
    if (ret_flag) {
     return;
    }
  }
  
  
  div = lj / process_num;
  if (div == 0) div = 1;
  start = div * my_rank;
  end = div * (my_rank + 1);

  step ( n, mj, &x[0*2+0], &x[mj*2+0], &y[0*2+0], &y[mj*2+0], w, sgn, num_transpose, start, end);

  if ( n == 2 )
  { 
    return;
  }

  for ( j = 0; j < m - 2; j++ )
  {
    mj = mj * 2;
    if ( tgle )
    {
      lj = n / (mj * 2);

      if (process_num > lj){
  	    bool ret_flag = processData(process_num, my_rank, lj, mj*2, y);

        // 如果函数返回true，结束当前函数的执行
        if (ret_flag) {
         return;
        }
      }
      
      div = lj / process_num;
      if (div == 0) div = 1;
      start = div * my_rank;
      end = div * (my_rank + 1);

	  step ( n, mj, &y[0*2+0], &y[mj*2+0], &x[0*2+0], &x[mj*2+0], w, sgn, num_transpose, start, end);
      tgle = 0;
    }
    else
    {
	    
      lj = n / (mj * 2);
      if (process_num > lj){
  	    bool ret_flag = processData(process_num, my_rank, lj, mj*2, x);

        // 如果函数返回true，结束当前函数的执行
        if (ret_flag) {
         return;
        }
      }
      
      div = lj / process_num;
      if (div == 0) div = 1;
      start = div * my_rank;
      end = div * (my_rank + 1);
      
      step ( n, mj, &x[0*2+0], &x[mj*2+0], &y[0*2+0], &y[mj*2+0], w, sgn, num_transpose, start, end);
      tgle = 1;
    }
  }
//
//  Last pass thru data: move y to x if needed 
//
  if ( tgle ) 
  {
    ccopy ( n, y, x );
  }

  mj = n / 2;

  lj = n / (mj * 2);
  
  if (process_num > lj){
  	bool ret_flag = processData(process_num, my_rank, lj, mj*2, x);

    // 如果函数返回true，结束当前函数的执行
    if (ret_flag) {
     return;
    }
  }
  
  div = lj / process_num;
  if (div == 0) div = 1;
  start = div * my_rank;
  end = div * (my_rank + 1);


//  step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
  step ( n, mj, &x[0*2+0], &x[mj*2+0], &y[0*2+0], &y[mj*2+0], w, sgn, num_transpose, start, end);


  return;
}
//****************************************************************************80

void cffti ( int n, double w[] )

//****************************************************************************80
//
//  Purpose:
//
//    CFFTI sets up sine and cosine tables needed for the FFT calculation.
//
//  Parameters:
//
//    Input, int N, the size of the array to be transformed.
//
//    Output, double W[N], a table of sines and cosines.
//
{
  double arg;
  double aw;
  int i;
  int n2;
  const double pi = 3.141592653589793;

  n2 = n / 2;
  aw = 2.0 * pi / ( ( double ) n );

  for ( i = 0; i < n2; i++ )
  {
    arg = aw * ( ( double ) i );
    w[i*2+0] = cos ( arg );
    w[i*2+1] = sin ( arg );
  }
  return;
}
//****************************************************************************80

double cpu_time ( void )

//****************************************************************************80
//
//  Purpose:
// 
//    CPU_TIME reports the elapsed CPU time.
//
//
//  Parameters:
//
//    Output, double CPU_TIME, the current total elapsed CPU time in second.
//
{
  double value;

  value = ( double ) clock ( ) / ( double ) CLOCKS_PER_SEC;

  return value;
}
//****************************************************************************80

double ggl ( double *seed )

//****************************************************************************80
//
//  Purpose:
//
//    GGL generates uniformly distributed pseudorandom numbers. 
//
//  Parameters:
//
//    Input/output, double *SEED, used as a seed for the sequence.
//
//    Output, double GGL, the next pseudorandom value.
//
{
  double d2 = 0.2147483647e10;
  double t;
  double value;

  t = *seed;
  t = fmod ( 16807.0 * t, d2 );
  *seed = t;
  value = ( t - 1.0 ) / ( d2 - 1.0 );

  return value;
}
//****************************************************************************80
 
void step ( int n, int mj, double a[], double b[], double c[],
  double d[], double w[], double sgn, int num_transpose[], int start, int end)

//****************************************************************************80
//
//  Purpose:
//
//    STEP carries out one step of the workspace version of CFFT2.
//
{
  double ambr;
  double ambu;
  int j;
  int ja;
  int jb;
  int jc;
  int jd;
  int jw;
  int k;
  int lj;
  int mj2;
  double wjw[2];

  mj2 = 2 * mj;
  lj = n / mj2;


  for ( j = start; j < end; j++ )
  {
    jw = num_transpose[j * mj2] * mj;
	ja = j * mj2;
	jb = ja;
    jc  = j * mj2;
    jd  = jc;

    wjw[0] = w[jw*2+0]; 
    wjw[1] = w[jw*2+1];

    if ( sgn < 0.0 ) 
    {
      wjw[1] = - wjw[1];
    }
	
    for ( k = 0; k < mj; k++ ) 
	{
      c[(jc+k)*2+0] = a[(ja+k)*2+0] + b[(jb+k)*2+0];
      c[(jc+k)*2+1] = a[(ja+k)*2+1] + b[(jb+k)*2+1];

      ambr = a[(ja+k)*2+0] - b[(jb+k)*2+0];
      ambu = a[(ja+k)*2+1] - b[(jb+k)*2+1];

      d[(jd+k)*2+0] = wjw[0] * ambr - wjw[1] * ambu;
      d[(jd+k)*2+1] = wjw[1] * ambr + wjw[0] * ambu;
    }
  }
  return;
}
//****************************************************************************80

void timestamp ( )

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}
