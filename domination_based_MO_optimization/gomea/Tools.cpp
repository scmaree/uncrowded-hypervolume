/**
 *
 * MO-RV-GOMEA
 *
 * If you use this software for any purpose, please cite the most recent publication:
 * A. Bouter, N.H. Luong, C. Witteveen, T. Alderliesten, P.A.N. Bosman. 2017.
 * The Multi-Objective Real-Valued Gene-pool Optimal Mixing Evolutionary Algorithm.
 * In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2017).
 * DOI: 10.1145/3071178.3071274
 *
 * Copyright (c) 1998-2017 Peter A.N. Bosman
 *
 * The software in this file is the proprietary information of
 * Peter A.N. Bosman.
 *
 * IN NO EVENT WILL THE AUTHOR OF THIS SOFTWARE BE LIABLE TO YOU FOR ANY
 * DAMAGES, INCLUDING BUT NOT LIMITED TO LOST PROFITS, LOST SAVINGS, OR OTHER
 * INCIDENTIAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR THE INABILITY
 * TO USE SUCH PROGRAM, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGES, OR FOR ANY CLAIM BY ANY OTHER PARTY. THE AUTHOR MAKES NO
 * REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. THE
 * AUTHOR SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY ANYONE AS A RESULT OF
 * USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
 *
 * The software in this file is the result of (ongoing) scientific research.
 * The following people have been actively involved in this research over
 * the years:
 * - Peter A.N. Bosman
 * - Dirk Thierens
 * - Jörn Grahl
 * - Anton Bouter
 * 
 */

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "Tools.h"
#include  <iostream>
#include <iomanip>
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace gomea
{
  
  int64_t    random_seed,                      /* The seed used for the random-number generator. */
  random_seed_changing;             /* Internally used variable for randomly setting a random seed. */
  
  long  timestamp_start,                       /* The time stamp in milliseconds for when the program was started. */
  timestamp_start_after_init;            /* The time stamp in milliseconds for when the algorithm was started */
  
  double haveNextNextGaussian,             /* Internally used variable for sampling the normal distribution. */
  nextNextGaussian;                     /* Internally used variable for sampling the normal distribution. */

  
  /*-=-=-=-=-=-=-=-=-=-=-= Section Elementary Operations -=-=-=-=-=-=-=-=-=-=-*/
  /**
   * Allocates memory and exits the program in case of a memory allocation failure.
   */
  void *Malloc( long size )
  {
      void *result;

      result = (void *) malloc( size );
      if( !result )
      {
          printf("\n");
          printf("Error while allocating memory in Malloc( %ld ), aborting program.", size);
          printf("\n");

          exit( 0 );
      }

      return( result );
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Matrix -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Creates a new matrix with dimensions n x m.
   */
  double **matrixNew( int n, int m )
  {
      int      i;
      double **result;

      result = (double **) malloc( n*( sizeof( double * ) ) );
      for( i = 0; i < n; i++ )
          result[i] = (double *) malloc( m*( sizeof( double ) ) );

      return( result );
  }

  /**
   * Computes the dot product of two vectors of the same dimensionality n0.
   */
  double vectorDotProduct( double *vector0, double *vector1, int n0 )
  {
      int    i;
      double result;

      result = 0.0;
      for( i = 0; i < n0; i++ )
          result += vector0[i]*vector1[i];
      
      return( result );
  }

  /**
   * Computes the Euclidean norm of a given vector.
   */
  double vectorNorm( double *vector0, int n0 )
  {
      return( sqrt(vectorDotProduct( vector0, vector0, n0 )) );
  }

  /**
   * Computes the multiplication Av of a matrix A and a vector v
   * where matrix A has dimensions n0 x n1 and vector v has
   * dimensionality n1.
   */
  double *matrixVectorMultiplication( double **matrix, double *vector, int n0, int n1 )
  {
      int     i;
      double *result;

      result = (double *) malloc( n0*sizeof( double ) );
      for( i = 0; i < n0; i++ )
          result[i] = vectorDotProduct( matrix[i], vector, n1 );

      return( result );
  }

  double *matrixVectorPartialMultiplication( double **matrix, double *vector, int n0, int n1, int number_of_elements, int *element_indices )
  {
      int i,j,index;
      double *result;

      result = (double *) malloc( n0*sizeof( double ) );
      for( i = 0; i < n0; i++ )
          result[i] = 0;

      for( j = 0; j < number_of_elements; j++)
      {
          index = element_indices[j];
          for( i = 0; i < n0; i++ )
              result[i] += ( vector[index] * matrix[i][index] );
      }

      return result;
  }
  /**
   * Computes the matrix multiplication of two matrices A and B
   * of dimensions A: n0 x n1 and B: n1 x n2.
   */
  double **matrixMatrixMultiplication( double **matrix0, double **matrix1, int n0, int n1, int n2 )
  {
      int     i, j, k;
      double **result;

      result = (double **) malloc( n0*sizeof( double * ) );
      for( i = 0; i < n0; i++ )
          result[i] = (double *) malloc( n2*sizeof( double ) );

      for( i = 0; i < n0; i++ )
      {
          for( j = 0; j < n2; j++ )
          {
              result[i][j] = 0;
              for( k = 0; k < n1; k++ )
                  result[i][j] += matrix0[i][k]*matrix1[k][j];
          }
      }

      return( result );
  }

  /**
   * BLAS subroutine.
   */
  int blasDSWAP( int n, double *dx, int incx, double *dy, int incy )
  {
      double dtmp;

      if (n > 0)
      {
          incx *= sizeof( double );
          incy *= sizeof( double );

          dtmp  = (*dx);
          *dx   = (*dy);
          *dy   = dtmp;

          while( (--n) > 0 )
          {
              dx = (double *) ((char *) dx + incx);
              dy = (double *) ((char *) dy + incy);
              dtmp = (*dx); *dx = (*dy); *dy = dtmp;
          }
      }

      return( 0 );
  }

  /**
   * BLAS subroutine.
   */
  int blasDAXPY(int n, double da, double *dx, int incx, double *dy, int incy)
  {
      double dtmp0, dtmp, *dx0, *dy0;

      if( n > 0 && da != 0. )
      {
          incx *= sizeof(double);
          incy *= sizeof(double);
          *dy  += da * (*dx);

          if( (n & 1) == 0 )
          {
              dx   = (double *) ((char *) dx + incx);
              dy   = (double *) ((char *) dy + incy);
              *dy += da * (*dx);
              --n;
          }
          n = n >> 1;
          while( n > 0 )
          {
              dy0   = (double *) ((char *) dy + incy);
              dy    = (double *) ((char *) dy0 + incy);
              dtmp0 = (*dy0);
              dtmp  = (*dy);
              dx0   = (double *) ((char *) dx + incx);
              dx    = (double *) ((char *) dx0 + incx);
              *dy0  = dtmp0 + da * (*dx0);
              *dy   = dtmp + da * (*dx);
              --n;
          }
      }

      return( 0 );
  }

  /**
   * BLAS subroutine.
   */
  void blasDSCAL( int n, double sa, double x[], int incx )
  {
      int i, ix, m;

      if( n <= 0 )
      {
      }
      else if( incx == 1 )
      {
          m = n % 5;

          for( i = 0; i < m; i++ )
          {
              x[i] = sa * x[i];
          }

          for( i = m; i < n; i = i + 5 )
          {
              x[i]   = sa * x[i];
              x[i+1] = sa * x[i+1];
              x[i+2] = sa * x[i+2];
              x[i+3] = sa * x[i+3];
              x[i+4] = sa * x[i+4];
          }
      }
      else
      {
          if( 0 <= incx )
          {
              ix = 0;
          }
          else
          {
              ix = ( - n + 1 ) * incx;
          }

          for( i = 0; i < n; i++ )
          {
              x[ix] = sa * x[ix];
              ix = ix + incx;
          }
      }
  }

  /**
   * LINPACK subroutine.
   */
  int linpackDCHDC( double a[], int lda, int p, double work[], int ipvt[] )
  {
      int    info, j, jp, k, l, maxl, pl, pu;
      double maxdia, temp;

      pl   = 1;
      pu   = 0;
      info = p;
      for( k = 1; k <= p; k++ )
      {
          maxdia = a[k-1+(k-1)*lda];
          maxl   = k;
          if( pl <= k && k < pu )
          {
              for( l = k+1; l <= pu; l++ )
              {
                  if( maxdia < a[l-1+(l-1)*lda] )
                  {
                      maxdia = a[l-1+(l-1)*lda];
                      maxl   = l;
                  }
              }
          }

          if( maxdia <= 0.0 )
          {
              info = k - 1;

              return( info );
          }

          if( k != maxl )
          {
              blasDSWAP( k-1, a+0+(k-1)*lda, 1, a+0+(maxl-1)*lda, 1 );

              a[maxl-1+(maxl-1)*lda] = a[k-1+(k-1)*lda];
              a[k-1+(k-1)*lda]       = maxdia;
              jp                     = ipvt[maxl-1];
              ipvt[maxl-1]           = ipvt[k-1];
              ipvt[k-1]              = jp;
          }
          work[k-1]        = sqrt( a[k-1+(k-1)*lda] );
          a[k-1+(k-1)*lda] = work[k-1];

          for( j = k+1; j <= p; j++ )
          {
              if( k != maxl )
              {
                  if( j < maxl )
                  {
                      temp                = a[k-1+(j-1)*lda];
                      a[k-1+(j-1)*lda]    = a[j-1+(maxl-1)*lda];
                      a[j-1+(maxl-1)*lda] = temp;
                  }
                  else if ( maxl < j )
                  {
                      temp                = a[k-1+(j-1)*lda];
                      a[k-1+(j-1)*lda]    = a[maxl-1+(j-1)*lda];
                      a[maxl-1+(j-1)*lda] = temp;
                  }
              }
              a[k-1+(j-1)*lda] = a[k-1+(j-1)*lda] / work[k-1];
              work[j-1]        = a[k-1+(j-1)*lda];
              temp             = -a[k-1+(j-1)*lda];

              blasDAXPY( j-k, temp, work+k, 1, a+k+(j-1)*lda, 1 );
          }
      }

      return( info );
  }

  /**
   * Computes the lower-triangle Cholesky Decomposition
   * of a square, symmetric and positive-definite matrix.
   * Subroutines from LINPACK and BLAS are used.
   */
  double **choleskyDecomposition( double **matrix, int n )
  {
    bool success;
    return choleskyDecomposition(matrix, n, success);
  }
  
  /**
   * Computes the lower-triangle Cholesky Decomposition
   * of a square, symmetric and positive-definite matrix.
   * Subroutines from LINPACK and BLAS are used.
   */
  double **choleskyDecomposition( double **matrix, int n, bool & success )
  {
    success = false;
    int     i, j, k, info, *ipvt;
    double *a, *work, **result;
    
    a    = (double *) Malloc( n*n*sizeof( double ) );
    work = (double *) Malloc( n*sizeof( double ) );
    ipvt = (int *) Malloc( n*sizeof( int ) );
    
    result = matrixNew( n, n );
    
    k = 0;
    for( i = 0; i < n; i++ )
    {
      for( j = 0; j < n; j++ )
      {
        a[k] = matrix[i][j];
        k++;
      }
      ipvt[i] = 0;
    }
    
    info = linpackDCHDC( a, n, n, work, ipvt );
    
    // if matrix is Positive Definite
    if (info == n)
    {
      success = true;
      k = 0;
      for( i = 0; i < n; i++ )
      {
        for( j = 0; j < n; j++ )
        {
          result[i][j] = i < j ? 0.0 : a[k];
          k++;
        }
      }
    }
    
    // Univariate approximation
    if (info != n)
    {
      success = false;
      // std::cout << "Univariate Choleksy for FOS of length " << n << "." << std::endl;
      for( i = 0; i < n; i++ ) {
        for( j = 0; j < n; j++ ) {
          result[i][j] = i != j ? 0.0 : sqrt( matrix[i][j] );
        }
      }
    }
    
    free( ipvt );
    free( work );
    free( a );
    
    return( result );
    
  }

  /**
   * LINPACK subroutine.
   */
  int linpackDTRDI( double t[], int ldt, int n )
  {
      int    j, k, info;
      double temp;

      info = 0;
      for( k = n; 1 <= k; k-- )
      {
          if ( t[k-1+(k-1)*ldt] == 0.0 )
          {
              info = k;
              break;
          }

          t[k-1+(k-1)*ldt] = 1.0 / t[k-1+(k-1)*ldt];
          temp = -t[k-1+(k-1)*ldt];

          if ( k != n )
          {
              blasDSCAL( n-k, temp, t+k+(k-1)*ldt, 1 );
          }

          for( j = 1; j <= k-1; j++ )
          {
              temp = t[k-1+(j-1)*ldt];
              t[k-1+(j-1)*ldt] = 0.0;
              blasDAXPY( n-k+1, temp, t+k-1+(k-1)*ldt, 1, t+k-1+(j-1)*ldt, 1 );
          }
      }

      return( info );
  }

  /**
   * Computes the inverse of a matrix that is of
   * lower triangular form.
   */
  double **matrixLowerTriangularInverse( double **matrix, int n )
  {
      int     i, j, k, info;
      double *t, **result;

      t = (double *) Malloc( n*n*sizeof( double ) );

      k = 0;
      for( i = 0; i < n; i++ )
      {
          for( j = 0; j < n; j++ )
          {
              t[k] = matrix[j][i];
              k++;
          }
      }

      info = linpackDTRDI( t, n, n );

      result = matrixNew( n, n );
      k = 0;
      for( i = 0; i < n; i++ )
      {
          for( j = 0; j < n; j++ )
          {
              result[j][i] = i > j ? 0.0 : t[k];
              k++;
          }
      }

      free( t );

      return( result );
  }

  void eigenDecomposition( double **matrix, int n, double **D, double **Q )
  {
      int     i, j;
      double *rgtmp, *diag;

      rgtmp = (double *) Malloc( n*sizeof( double ) );
      diag  = (double *) Malloc( n*sizeof( double ) );

      for( i = 0; i < n; i++ )
      {
          for( j = 0; j <= i; j++ )
          {
              Q[j][i] = matrix[j][i];
              Q[i][j] = Q[j][i];
          }
      }

      eigenDecompositionHouseholder2( n, Q, diag, rgtmp );
      eigenDecompositionQLalgo2( n, Q, diag, rgtmp );

      for( i = 0; i < n; i++ )
      {
          for( j = 0; j < n; j++ )
          {
              D[i][j] = 0.0;
          }
          D[i][i] = diag[i];
      }

      free( diag );
      free( rgtmp );
  }


  void eigenDecompositionQLalgo2( int n, double **V, double *d, double *e )
  {
      int i, k, l, m;
      double f = 0.0;
      double tst1 = 0.0;
      double eps = 2.22e-16; /* Math.pow(2.0,-52.0);  == 2.22e-16 */

      /* shift input e */
      for (i = 1; i < n; i++) {
          e[i-1] = e[i];
      }
      e[n-1] = 0.0; /* never changed again */

      for (l = 0; l < n; l++) {

          /* Find small subdiagonal element */

          if (tst1 < fabs(d[l]) + fabs(e[l]))
              tst1 = fabs(d[l]) + fabs(e[l]);
          m = l;
          while (m < n) {
              if (fabs(e[m]) <= eps*tst1) {
                  /* if (fabs(e[m]) + fabs(d[m]+d[m+1]) == fabs(d[m]+d[m+1])) { */
                  break;
              }
              m++;
          }

          /* If m == l, d[l] is an eigenvalue, */
          /* otherwise, iterate. */

          if (m > l) {
              int iter = 0;
              do { /* while (fabs(e[l]) > eps*tst1); */
                  double dl1, h;
                  double g = d[l];
                  double p = (d[l+1] - g) / (2.0 * e[l]);
                  double r = myhypot(p, 1.);

                  iter = iter + 1;  /* Could check iteration count here */

                  /* Compute implicit shift */

                  if (p < 0) {
                      r = -r;
                  }
                  d[l] = e[l] / (p + r);
                  d[l+1] = e[l] * (p + r);
                  dl1 = d[l+1];
                  h = g - d[l];
                  for (i = l+2; i < n; i++) {
                      d[i] -= h;
                  }
                  f = f + h;

                  /* Implicit QL transformation. */

                  p = d[m];
                  {
                      double c = 1.0;
                      double c2 = c;
                      double c3 = c;
                      double el1 = e[l+1];
                      double s = 0.0;
                      double s2 = 0.0;
                      for (i = m-1; i >= l; i--) {
                          c3 = c2;
                          c2 = c;
                          s2 = s;
                          g = c * e[i];
                          h = c * p;
                          r = myhypot(p, e[i]);
                          e[i+1] = s * r;
                          s = e[i] / r;
                          c = p / r;
                          p = c * d[i] - s * g;
                          d[i+1] = h + s * (c * g + s * d[i]);

                          /* Accumulate transformation. */

                          for (k = 0; k < n; k++) {
                              h = V[k][i+1];
                              V[k][i+1] = s * V[k][i] + c * h;
                              V[k][i] = c * V[k][i] - s * h;
                          }
                      }
                      p = -s * s2 * c3 * el1 * e[l] / dl1;
                      e[l] = s * p;
                      d[l] = c * p;
                  }

                  /* Check for convergence. */

              } while (fabs(e[l]) > eps*tst1);
          }
          d[l] = d[l] + f;
          e[l] = 0.0;
      }

      /* Sort eigenvalues and corresponding vectors. */
  #if 1
      /* TODO: really needed here? So far not, but practical and only O(n^2) */
      {
          int j;
          double p;
          for (i = 0; i < n-1; i++) {
              k = i;
              p = d[i];
              for (j = i+1; j < n; j++) {
                  if (d[j] < p) {
                      k = j;
                      p = d[j];
                  }
              }
              if (k != i) {
                  d[k] = d[i];
                  d[i] = p;
                  for (j = 0; j < n; j++) {
                      p = V[j][i];
                      V[j][i] = V[j][k];
                      V[j][k] = p;
                  }
              }
          }
      }
  #endif
  } /* QLalgo2 */

  double myhypot( double a, double b )
  {
      double r = 0;
      if( fabs(a) > fabs(b) )
      {
          r = b/a;
          r = fabs(a)*sqrt(1+r*r);
      }
      else if (b != 0)
      {
          r = a/b;
          r = fabs(b)*sqrt(1+r*r);
      }

      return r;
  }

  void eigenDecompositionHouseholder2( int n, double **V, double *d, double *e )
  {
      int i,j,k;

      for (j = 0; j < n; j++) {
          d[j] = V[n-1][j];
      }

      /* Householder reduction to tridiagonal form */

      for (i = n-1; i > 0; i--) {

          /* Scale to avoid under/overflow */

          double scale = 0.0;
          double h = 0.0;
          for (k = 0; k < i; k++) {
              scale = scale + fabs(d[k]);
          }
          if (scale == 0.0) {
              e[i] = d[i-1];
              for (j = 0; j < i; j++) {
                  d[j] = V[i-1][j];
                  V[i][j] = 0.0;
                  V[j][i] = 0.0;
              }
          } else {

              /* Generate Householder vector */

              double f, g, hh;

              for (k = 0; k < i; k++) {
                  d[k] /= scale;
                  h += d[k] * d[k];
              }
              f = d[i-1];
              g = sqrt(h);
              if (f > 0) {
                  g = -g;
              }
              e[i] = scale * g;
              h = h - f * g;
              d[i-1] = f - g;
              for (j = 0; j < i; j++) {
                  e[j] = 0.0;
              }

              /* Apply similarity transformation to remaining columns */

              for (j = 0; j < i; j++) {
                  f = d[j];
                  V[j][i] = f;
                  g = e[j] + V[j][j] * f;
                  for (k = j+1; k <= i-1; k++) {
                      g += V[k][j] * d[k];
                      e[k] += V[k][j] * f;
                  }
                  e[j] = g;
              }
              f = 0.0;
              for (j = 0; j < i; j++) {
                  e[j] /= h;
                  f += e[j] * d[j];
              }
              hh = f / (h + h);
              for (j = 0; j < i; j++) {
                  e[j] -= hh * d[j];
              }
              for (j = 0; j < i; j++) {
                  f = d[j];
                  g = e[j];
                  for (k = j; k <= i-1; k++) {
                      V[k][j] -= (f * e[k] + g * d[k]);
                  }
                  d[j] = V[i-1][j];
                  V[i][j] = 0.0;
              }
          }
          d[i] = h;
      }

      /* Accumulate transformations */

      for (i = 0; i < n-1; i++) {
          double h;
          V[n-1][i] = V[i][i];
          V[i][i] = 1.0;
          h = d[i+1];
          if (h != 0.0) {
              for (k = 0; k <= i; k++) {
                  d[k] = V[k][i+1] / h;
              }
              for (j = 0; j <= i; j++) {
                  double g = 0.0;
                  for (k = 0; k <= i; k++) {
                      g += V[k][i+1] * V[k][j];
                  }
                  for (k = 0; k <= i; k++) {
                      V[k][j] -= g * d[k];
                  }
              }
          }
          for (k = 0; k <= i; k++) {
              V[k][i+1] = 0.0;
          }
      }
      for (j = 0; j < n; j++) {
          d[j] = V[n-1][j];
          V[n-1][j] = 0.0;
      }
      V[n-1][n-1] = 1.0;
      e[0] = 0.0;

  }

  /**
   * Writes the contents of a matrix of dimensions n0 x n1 to a file.
   */
  void matrixWriteToFile( FILE *file, double **matrix, int n0, int n1 )
  {
      int  i, j;
      char line_for_output[10000];

      sprintf( line_for_output, "[" );
      fputs( line_for_output, file );
      for( i = 0; i < n0; i++ )
      {
          sprintf( line_for_output, "[" );
          fputs( line_for_output, file );
          for( j = 0; j < n1; j++ )
          {
              sprintf( line_for_output, "%lf", matrix[i][j] );
              fputs( line_for_output, file );
              if( j < n1-1 )
              {
                  sprintf( line_for_output, ", " );
                  fputs( line_for_output, file );
              }
          }
          if( i == n0-1 )
              sprintf( line_for_output, "]" );
          else
              sprintf( line_for_output, "];" );
          fputs( line_for_output, file );
      }
      sprintf( line_for_output, "]\n" );
      fputs( line_for_output, file );
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  /*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Merge Sort -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Sorts an array of doubles and returns the sort-order (small to large).
   */
  int *mergeSort( double *array, int array_size )
  {
      int i, *sorted, *tosort;

      sorted = (int *) Malloc( array_size * sizeof( int ) );
      tosort = (int *) Malloc( array_size * sizeof( int ) );
      for( i = 0; i < array_size; i++ )
          tosort[i] = i;

      if( array_size == 1 )
          sorted[0] = 0;
      else
          mergeSortWithinBounds( array, sorted, tosort, 0, array_size-1 );

      free( tosort );

      return( sorted );
  }

  /**
   * Subroutine of merge sort, sorts the part of the array between p and q.
   */
  void mergeSortWithinBounds( double *array, int *sorted, int *tosort, int p, int q )
  {
      int r;

      if( p < q )
      {
          r = (p + q) / 2;
          mergeSortWithinBounds( array, sorted, tosort, p, r );
          mergeSortWithinBounds( array, sorted, tosort, r+1, q );
          mergeSortMerge( array, sorted, tosort, p, r+1, q );
      }
  }
  void mergeSortWithinBoundsInt( int *array, int *sorted, int *tosort, int p, int q )
  {
      int r;

      if( p < q )
      {
          r = (p + q) / 2;
          mergeSortWithinBoundsInt( array, sorted, tosort, p, r );
          mergeSortWithinBoundsInt( array, sorted, tosort, r+1, q );
          mergeSortMergeInt( array, sorted, tosort, p, r+1, q );
      }
  }
  /**
   * Subroutine of merge sort, merges the results of two sorted parts.
   */
  void mergeSortMerge( double *array, int *sorted, int *tosort, int p, int r, int q )
  {
      int i, j, k, first;

      i = p;
      j = r;
      for( k = p; k <= q; k++ )
      {
          first = 0;
          if( j <= q )
          {
              if( i < r )
              {
                  if( array[tosort[i]] < array[tosort[j]] )
                      first = 1;
              }
          }
          else
              first = 1;

          if( first )
          {
              sorted[k] = tosort[i];
              i++;
          }
          else
          {
              sorted[k] = tosort[j];
              j++;
          }
      }

      for( k = p; k <= q; k++ )
          tosort[k] = sorted[k];
  }

  int *mergeSortInt( int *array, int array_size )
  {
      int i, *sorted, *tosort;

      sorted = (int *) Malloc( array_size * sizeof( int ) );
      tosort = (int *) Malloc( array_size * sizeof( int ) );
      for( i = 0; i < array_size; i++ )
          tosort[i] = i;

      if( array_size == 1 )
          sorted[0] = 0;
      else
          mergeSortWithinBoundsInt( array, sorted, tosort, 0, array_size-1 );

      free( tosort );

      return( sorted );
  }


  void mergeSortMergeInt( int *array, int *sorted, int *tosort, int p, int r, int q )
  {
      int i, j, k, first;

      i = p;
      j = r;
      for( k = p; k <= q; k++ )
      {
          first = 0;
          if( j <= q )
          {
              if( i < r )
              {
                  if( array[tosort[i]] < array[tosort[j]] )
                      first = 1;
              }
          }
          else
              first = 1;

          if( first )
          {
              sorted[k] = tosort[i];
              i++;
          }
          else
          {
              sorted[k] = tosort[j];
              j++;
          }
      }

      for( k = p; k <= q; k++ )
          tosort[k] = sorted[k];
  }

  int *getRanks( double *array, int array_size )
  {
      int i, *sorted, *ranks;

      sorted = mergeSort( array, array_size );
      ranks = (int *) Malloc( array_size * sizeof( int ) );
      for( i = 0; i < array_size; i++ ) ranks[sorted[i]] = i;

      free( sorted );
      return( ranks );
  }

  int *getRanksFromSorted( int *sorted, int array_size )
  {
      int i, *ranks;

      ranks = (int *) Malloc( array_size * sizeof( int ) );
      for( i = 0; i < array_size; i++ ) ranks[sorted[i]] = i;

      return( ranks );
  }

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Time -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
  long getMilliSecondsRunning()
  {
      return( getMilliSecondsRunningSinceTimeStamp( timestamp_start ) );
  }

  long getMilliSecondsRunningAfterInit()
  {
      return( getMilliSecondsRunningSinceTimeStamp( timestamp_start_after_init ) );
  }

  long getMilliSecondsRunningSinceTimeStamp( long timestamp )
  {
      long timestamp_now, difference;

      timestamp_now = getCurrentTimeStampInMilliSeconds();

      difference = timestamp_now-timestamp;

      return( difference );
  }

  long getCurrentTimeStampInMilliSeconds()
  {
      struct timeval tv;
      //struct tm *timep;
      long   result;

      gettimeofday( &tv, NULL );
      //timep = localtime( &tv.tv_sec );
      //result = timep->tm_hour * 3600 * 1000 + timep->tm_min * 60 * 1000 + timep->tm_sec * 1000 + tv.tv_usec / 1000;
      result = (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
      return( result );
  }

  void startTimer( void )
  {
      timestamp_start = getCurrentTimeStampInMilliSeconds();
  }

  double getTimer( void )
  {
      return ( (double) (getMilliSecondsRunningSinceTimeStamp( timestamp_start )/1000.0) );
  }

  void printTimer( void )
  {
      double cpu_time_used;

      cpu_time_used = (double) (getMilliSecondsRunningSinceTimeStamp( timestamp_start )/1000.0);
      printf("%.3f\n",cpu_time_used);
  }

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


  /*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Random Numbers -=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Returns a random double, distributed uniformly between 0 and 1.
   */
  double randomRealUniform01( void )
  {
      int64_t n26, n27;
      double  result;

      random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
      n26                  = (int64_t)(random_seed_changing >> (48 - 26));
      random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
      n27                  = (int64_t)(random_seed_changing >> (48 - 27));
      result               = (((int64_t)n26 << 27) + n27) / ((double) (1LLU << 53));

      return( result );
  }

  /**
   * Returns a random integer, distributed uniformly between 0 and maximum.
   */
  int randomInt( int maximum )
  {
      int result;

      result = (int) (((double) maximum)*randomRealUniform01());

      return( result );
  }

  /**
   * Returns a random double, distributed normally with mean 0 and variance 1.
   */
  double random1DNormalUnit( void )
  {
      double v1, v2, s, multiplier, value;

      if( haveNextNextGaussian )
      {
          haveNextNextGaussian = 0;

          return( nextNextGaussian );
      }
      else
      {
          do
          {
              v1 = 2 * (randomRealUniform01()) - 1;
              v2 = 2 * (randomRealUniform01()) - 1;
              s = v1 * v1 + v2 * v2;
          } while (s >= 1);

          value                = -2 * log(s)/s;
          multiplier           = value <= 0.0 ? 0.0 : sqrt( value );
          nextNextGaussian     = v2 * multiplier;
          haveNextNextGaussian = 1;

          return( v1 * multiplier );
      }
  }

  /**
   * Returns a random double, distributed normally with given mean and variance.
   */
  double random1DNormalParameterized( double mean, double variance )
  {
      double result;

      result = mean + sqrt( variance )*random1DNormalUnit();

      return( result );
  }

  /**
   * Initializes the random number generator.
   */
  void initializeRandomNumberGenerator( void )
  {
      struct timeval tv;
      
      while( random_seed_changing == 0 )
      {
          gettimeofday(&tv, NULL);
          random_seed_changing = (int64_t) tv.tv_usec;
          random_seed_changing = (random_seed_changing/((int) (9.99*randomRealUniform01())+1))*(((int) (randomRealUniform01()*1000000.0))%10);
      }

      random_seed = random_seed_changing;
      
      FILE *file;
      file = fopen( "random_seed.dat", "w");
    fprintf( file, "%lld\n", random_seed );
      fclose(file);

  }

  /**
   * Returns a random compact (using integers 0,1,...,n-1) permutation
   * of length n using the Fisher-Yates shuffle.
   */
  int *randomPermutation( int n )
  {
      int i, j, dummy, *result;

      result = (int *) Malloc( n*sizeof( int ) );
      for( i = 0; i < n; i++ )
          result[i] = i;

      for( i = n-1; i > 0; i-- )
      {
          j         = randomInt( i+1 );
          dummy     = result[j];
          result[j] = result[i];
          result[i] = dummy;
      }

      return( result );
  }

  /*
   * Returns all compact integer permutations of
   * a specified length, sorted in ascending
   * radix-sort order.
   */
  int **allPermutations( int length, int *numberOfPermutations )
  {
      int **result;

      result = allPermutationsSubroutine( 0, length, numberOfPermutations );

      return( result );
  }

  /*
   * Subroutine of allPermutations.
   */
  int **allPermutationsSubroutine( int from, int length, int *numberOfPermutations )
  {
      int i, j, k, q, **result, **smallerResult, smallerNumberOfPermutations;

      (*numberOfPermutations) = 1;
      for( i = 2; i <= length; i++ )
          (*numberOfPermutations) *= i;

      result = (int **) Malloc( (*numberOfPermutations)*sizeof( int * ) );
      for( i = 0; i < *numberOfPermutations; i++ )
          result[i] = (int *) Malloc( length*sizeof( int ) );

      if( length == 1 )
      {
          result[0][0] = from;
      }
      else
      {
          smallerResult = allPermutationsSubroutine( from+1, length-1, &smallerNumberOfPermutations );

          k = 0;
          for( i = from; i < from+length; i++ )
          {
              for( j = 0; j < smallerNumberOfPermutations; j++ )
              {
                  result[k][0] = i;
                  for( q = 1; q < length; q++ )
                      result[k][q] = smallerResult[j][q-1] <= i ? smallerResult[j][q-1]-1 : smallerResult[j][q-1];
                  k++;
              }
          }

          for( i = 0; i < smallerNumberOfPermutations; i++ )
              free( smallerResult[i] );
          free( smallerResult );
      }

      return( result );
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  double min( double x, double y )
  {
      if( x <= y )
          return x;
      return y;
  }

  double max( double x, double y )
  {
      if( x >= y )
          return x;
      return y;
  }

  /**
   * Computes the distance between two solutions a and b as
   * the Euclidean distance in parameter space.
   */
  double distanceEuclidean( double *x, double *y, int number_of_dimensions )
  {
      int    i;
      double value, result;

      result = 0.0;
      for( i = 0; i < number_of_dimensions; i++ )
      {
          value   = y[i] - x[i];
          result += value*value;
      }
      result = sqrt( result );

      return( result );
  }

  /**
   * Computes the Euclidean distance between two points.
   */
  double distanceEuclidean2D( double x1, double y1, double x2, double y2 )
  {
      double result;

      result = (y1 - y2)*(y1-y2) + (x1-x2)*(x1-x2);
      result = sqrt( result );

      return( result );
  }
  
}
