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
void cfft2 ( int n, double x[], double y[], double w[], double sgn, int nt[], int myid, int numprocs);
void cffti ( int n, double w[] );
double cpu_time ( void );
double ggl ( double *ds );
void step ( int n, int mj, double a[], double b[], double c[], double d[], 
  double w[], double sgn, int nt[], int s, int e);
void timestamp ( );

//****************************************************************************80

int main ( int argc, char * argv[] )

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main program for FFT_SERIAL.
//
//  Discussion:
//
//    The complex data in an N vector is stored as pairs of values in a
//    real vector of length 2*N.
//
//  Modified:
//
//    23 March 2009
//
//  Author:
//
//    Original C version by Wesley Petersen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
//
{
  MPI_Init(&argc, &argv);
  int myid;
  int numprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);	
  
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
  
  int *nt;

  if (myid == 0) {
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

	  nt = new int[n];

    if (myid == 0) first = 1;
    MPI_Bcast(&first, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for ( icase = 0; icase < 2; icase++ )
    {
      
      if (myid == 0) {
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
	  

        for (int i = 0; i < n; i++) nt[i] = i;
        int i, j, k;
        for(i = 1, j = n>>1, k; i < n-1; ++i, j ^= k) {
          if(i < j) {
          int tmp = nt[i];
          nt[i] = nt[j];
          nt[j] = tmp;
        }
          for(k = n>>1; j&k; j ^= k, k >>= 1) continue;
        }
      }
      
//
//  Initialize the sine and cosine tables.
//

      MPI_Bcast(x, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(z, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(nt, n, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(w, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//
//  Transform forward, back 
//
      if ( first )
      {
        sgn = + 1.0;
        cfft2 ( n, x, y, w, sgn, nt, myid, numprocs);
        MPI_Bcast(y, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        sgn = - 1.0;
        cfft2 ( n, y, x, w, sgn, nt, myid, numprocs);
// 
//  Results should be same as initial multiplied by N.
//
        
        if (myid == 0) {
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
          // int position =0;
          // int buffer_size = 3 * n * sizeof(double) + 100;
          // void * buffer = malloc(buffer_size);
          // if (myid == 0) {
          //   MPI_Pack(x, 2*n, MPI_DOUBLE, buffer, buffer_size, &position, MPI_COMM_WORLD);
          //   MPI_Pack(w, n, MPI_DOUBLE, buffer, buffer_size, &position, MPI_COMM_WORLD);
          //   for (int id = 1; id < numprocs; id++) {
          //     MPI_Send(buffer, position, MPI_PACKED, id, 0, MPI_COMM_WORLD);
          //   }
          // } else {
          //   MPI_Status status;
          //   MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
          //   int size;
          //   MPI_Get_count(&status, MPI_PACKED, &size);
          //   MPI_Recv(buffer, size, MPI_PACKED, 0, 0, MPI_COMM_WORLD, &status);
          //   MPI_Unpack(buffer, size, &position, x, 2*n, MPI_DOUBLE, MPI_COMM_WORLD);
          //   MPI_Unpack(buffer, size, &position, w, n, MPI_DOUBLE, MPI_COMM_WORLD);
          // }

          MPI_Bcast(x, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          MPI_Bcast(w, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          cfft2 ( n, x, y, w, sgn, nt, myid, numprocs);
          MPI_Bcast(y, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          sgn = - 1.0;
          cfft2 ( n, y, x, w, sgn, nt, myid, numprocs);
        }
        ctime2 = cpu_time ( );
        ctime = ctime2 - ctime1;

        flops = 2.0 * ( double ) nits * ( 5.0 * ( double ) n * ( double ) ln2 );

        mflops = flops / 1.0E+06 / ctime;
        if (myid == 0) {
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
    
    delete [] nt;
    
    delete [] w;
    delete [] x;
    delete [] y;
    delete [] z;
  }

  if (myid == 0) {
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
//  Modified:
//
//    23 March 2009
//
//  Author:
//
//    Original C version by Wesley Petersen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
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

void cfft2 ( int n, double x[], double y[], double w[], double sgn, int nt[], int myid, int numprocs)

//****************************************************************************80
//
//  Purpose:
//
//    CFFT2 performs a complex Fast Fourier Transform.
//
//  Modified:
//
//    23 March 2009
//
//  Author:
//
//    Original C version by Wesley Petersen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
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

  int i, k, lj, div, s, e;

  for(i = 1, j = n>>1, k; i < n-1; ++i, j ^= k) {
    if(i < j) {
      double tmp = x[2*i];
      x[2*i] = x[2*j];
      x[2*j] = tmp;
      tmp = x[2*i + 1];
      x[2*i + 1] = x[2*j + 1];
      x[2*j + 1] = tmp;
  }
    for(k = n>>1; j&k; j ^= k, k >>= 1) continue;
  }

	
   m = ( int ) ( log ( ( double ) n ) / log ( 1.99 ) );
   mj   = 1;
//
//  Toggling switch for work array.
//
  tgle = 1;

  lj = n / (mj * 2);
  if (numprocs > lj) {
    if (myid == 0) {
      MPI_Status status;
      MPI_Recv(x + mj*2, mj*2, MPI_DOUBLE, 1, 110, MPI_COMM_WORLD, &status);
    }
    else if (myid < 2 * lj) {
      MPI_Send(x + mj*2*myid, mj*2, MPI_DOUBLE, myid/2, 110, MPI_COMM_WORLD);

      if (myid < lj) {
        MPI_Status status;
        MPI_Recv(x + 2*myid*mj*2, mj*2, MPI_DOUBLE, 2*myid, 110, MPI_COMM_WORLD, &status);
        MPI_Recv(x + (2 * myid + 1)*mj*2, mj*2, MPI_DOUBLE, 2*myid+1, 110, MPI_COMM_WORLD, &status);
      } else {
        return;
      }
    } else {
      return;
    }
  } 
  
  div = lj / numprocs;
  if (div == 0) div = 1;
  s = div * myid;
  e = div * (myid + 1);

//  step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
  step ( n, mj, &x[0*2+0], &x[mj*2+0], &y[0*2+0], &y[mj*2+0], w, sgn, nt, s, e);

  if ( n == 2 )
  { 
    return;
  }

  for ( j = 0; j < m - 2; j++ )
  {
    mj = mj * 2;
    if ( tgle )
    {
//      step ( n, mj, &y[0*2+0], &y[(n/2)*2+0], &x[0*2+0], &x[mj*2+0], w, sgn );
      lj = n / (mj * 2);

      if (numprocs > lj) {
      if (myid == 0) {
        MPI_Status status;
        MPI_Recv(y + mj*2, mj*2, MPI_DOUBLE, 1, 110, MPI_COMM_WORLD, &status);
      }
      else if (myid < 2 * lj) {
        MPI_Send(y + mj*2*myid, mj*2, MPI_DOUBLE, myid/2, 110, MPI_COMM_WORLD);

        if (myid < lj) {
          MPI_Status status;
          MPI_Recv(y + 2*myid*mj*2, mj*2, MPI_DOUBLE, 2*myid, 110, MPI_COMM_WORLD, &status);
          MPI_Recv(y + (2 * myid + 1)*mj*2, mj*2, MPI_DOUBLE, 2*myid+1, 110, MPI_COMM_WORLD, &status);
        } else {
          return;
        }
      } else {
        return;
      }
    } 
      
      div = lj / numprocs;
      if (div == 0) div = 1;
      s = div * myid;
      e = div * (myid + 1);

	    step ( n, mj, &y[0*2+0], &y[mj*2+0], &x[0*2+0], &x[mj*2+0], w, sgn, nt, s, e);
      tgle = 0;
    }
    else
    {
//      step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
	    
      lj = n / (mj * 2);
      if (numprocs > lj) {
      if (myid == 0) {
        MPI_Status status;
        MPI_Recv(x + mj*2, mj*2, MPI_DOUBLE, 1, 110, MPI_COMM_WORLD, &status);
      }
      else if (myid < 2 * lj) {
        MPI_Send(x + mj*2*myid, mj*2, MPI_DOUBLE, myid/2, 110, MPI_COMM_WORLD);

        if (myid < lj) {
          MPI_Status status;
          MPI_Recv(x + 2*myid*mj*2, mj*2, MPI_DOUBLE, 2*myid, 110, MPI_COMM_WORLD, &status);
          MPI_Recv(x + (2 * myid + 1)*mj*2, mj*2, MPI_DOUBLE, 2*myid+1, 110, MPI_COMM_WORLD, &status);
        } else {
          return;
        }
      } else {
        return;
      }
    } 
      
      div = lj / numprocs;
      if (div == 0) div = 1;
      s = div * myid;
      e = div * (myid + 1);
      
      step ( n, mj, &x[0*2+0], &x[mj*2+0], &y[0*2+0], &y[mj*2+0], w, sgn, nt, s, e);
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
  if (numprocs > lj) {
    if (myid == 0) {
      MPI_Status status;
      MPI_Recv(x + mj*2, mj*2, MPI_DOUBLE, 1, 110, MPI_COMM_WORLD, &status);
    }
    else if (myid < 2 * lj) {
      MPI_Send(x + mj*2*myid, mj*2, MPI_DOUBLE, myid/2, 110, MPI_COMM_WORLD);

      if (myid < lj) {
        MPI_Status status;
        MPI_Recv(x + 2*myid*mj*2, mj*2, MPI_DOUBLE, 2*myid, 110, MPI_COMM_WORLD, &status);
        MPI_Recv(x + (2 * myid + 1)*mj*2, mj*2, MPI_DOUBLE, 2*myid+1, 110, MPI_COMM_WORLD, &status);
      } else {
        return;
      }
    } else {
      return;
    }
  } 
  
  div = lj / numprocs;
  if (div == 0) div = 1;
  s = div * myid;
  e = div * (myid + 1);


//  step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
  step ( n, mj, &x[0*2+0], &x[mj*2+0], &y[0*2+0], &y[mj*2+0], w, sgn, nt, s, e);


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
//  Modified:
//
//    23 March 2009
//
//  Author:
//
//    Original C version by Wesley Petersen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
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
//  Modified:
//
//    27 September 2005
//
//  Author:
//
//    John Burkardt
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
//  Modified:
//
//    23 March 2009
//
//  Author:
//
//    Original C version by Wesley Petersen, M Troyer, I Vattulainen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
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
  double d[], double w[], double sgn, int nt[], int s, int e)

//****************************************************************************80
//
//  Purpose:
//
//    STEP carries out one step of the workspace version of CFFT2.
//
//  Modified:
//
//    23 March 2009
//
//  Author:
//
//    Original C version by Wesley Petersen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
//
//  Parameters:
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


  for ( j = s; j < e; j++ )
  {
    jw = nt[j * mj2] * mj;
//    ja  = jw;
//    jb  = ja;
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
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    24 September 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
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