

/*
 
 HICAM
 
 By S.C. Maree, 2018
 s.c.maree[at]amc.uva.nl
 smaree.com
 
 */

#include "mathfunctions.h"
#include "solution.h"
#include "population.h"

namespace hicam 
{
 
  double round(double x)
  {
    return x >= 0.0 ? (double) floorf((float) (x + 0.5f)) : (double) ceilf((float) (x - 0.5f));
  }

  // Cholesky decomposition
  //---------------------------------------------------------------------------
  
  void principalCholeskyDecomposition(const matrix_t & cov, matrix_t & chol, std::vector<int> & permutation_order)
  {
    assert(cov.rows() == cov.cols());
    int n = (int)cov.rows();
    
    double **cholesky_factor_lower_triangle;
    
    cholesky_factor_lower_triangle = principalCholeskyDecomposition( cov.toArray(), n, permutation_order);
    chol.setRaw(cholesky_factor_lower_triangle, n,n);

  }
  
  void choleskyDecomposition(const matrix_t & cov, matrix_t & chol)
  {

    assert(cov.rows() == cov.cols());
    int n = (int)cov.rows();
    
    double **cholesky_factor_lower_triangle;
    cholesky_factor_lower_triangle = choleskyDecomposition( cov.toArray(), n );
    
    chol.setRaw(cholesky_factor_lower_triangle, n,n);
  }
  
  void choleskyDecomposition_univariate(const vec_t & univariate_covariance, vec_t & univariate_cholesky)
  {

    int n = (int)univariate_covariance.size();

    univariate_cholesky.reset(n, 0.0);

    for (int i = 0; i < n; ++i) {
      univariate_cholesky[i] = sqrt(univariate_covariance[i]);
    }
  }

  // Normal pdf
  // returns the value of the pdf at a given point.
  // assumes all input is of the same dimension as mean.
  //---------------------------------------------------------------------------

  
  // does chol + returns it.
  double normpdf(const vec_t & mean, const matrix_t & cov, const vec_t & x, matrix_t & chol, matrix_t & inverse_chol)
  {
    
    choleskyDecomposition(cov, chol);
    int n = (int)cov.rows();
    inverse_chol.setRaw(matrixLowerTriangularInverse(chol.toArray(), n),n,n);
    
    return normpdf(mean,chol,inverse_chol,x);
    
  }
  
  // use given chol
  double normpdf(const vec_t & mean, const matrix_t & chol, const matrix_t & inverse_chol, const vec_t & x)
  {
    
    double value = 0.0,
           chol_determinant = 1.0,
           dim = (double)mean.size();
    
    
    // compute the determinant of the cholesky factor.
    for(size_t i = 0; i < dim; ++i)
      chol_determinant *= chol[i][i];
    
    vec_t diff = mean - x;
    
    //  exp(-0.5*diff'*inv(cov)*diff) = exp(-0.5*squarednorm(inv(L)*diff) );
    value = exp(-0.5*inverse_chol.lowerProduct(diff).squaredNorm());
    
    // equal to sqrt((2pi)^d * det(cov))
    value /= pow(2*PI,dim*0.5)*fabs(chol_determinant);
    
    return value;
  }

  // use given chol
  double normpdf(const vec_t & mean, const matrix_t & inverse_chol, const double normal_factor, const vec_t & x, const bool univariate)
  {

    double value = 0.0;
    //  dim = (double)mean.size();

    vec_t diff = mean - x;

    //  exp(-0.5*diff'*inv(cov)*diff) = exp(-0.5*squarednorm(inv(L)*diff) );
    if (univariate) {
      value = exp(-0.5*inverse_chol.diagProduct(diff).squaredNorm());
    } 
    else {
      value = exp(-0.5*inverse_chol.lowerProduct(diff).squaredNorm());
    }

    // equal to sqrt((2pi)^d * det(cov))
    // value /= pow(2 * PI, dim*0.5)*fabs(cholesky_determinant);
    value /= normal_factor;

    return value;
  }
  
  // uses diag(cov) only
  // obsolete (scm 20170705)
  double normpdf_diagonal(const vec_t & mean, const vec_t & cov_diagonal, const vec_t & x)
  {
  
    double value = 0.0,
           dim = (double)mean.size(),
           determinant = 1.0;
    
    // difference from mean
    vec_t diff = x - mean;
    
    // create product inv(L)*diff
    for (size_t i = 0; i < dim; ++i) {
      diff[i] /= sqrt(cov_diagonal[i]);
      determinant *= sqrt(cov_diagonal[i]);
    }
    
    value = exp(-0.5*(diff.squaredNorm()));
    value /= sqrt(pow(2 * PI, dim))*determinant;
    
    return value;

    
  }
  
  double normcdf(double x)
  {
    // constants
    double a1 = 0.254829592;
    double a2 = -0.284496736;
    double a3 = 1.421413741;
    double a4 = -1.453152027;
    double a5 = 1.061405429;
    double p = 0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0) {
      sign = -1;
    }
    x = fabs(x) / sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0 / (1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return 0.5*(1.0 + sign*y);
  }


  // sample the parameter from a normal distribution
  // make sure it is within the parameter domain.
  //-------------------------------------------------------------------------
  unsigned int sample_normal(vec_t & sample, vec_t & sample_transformed, const size_t problem_size, const vec_t & mean, const matrix_t & MatrixRoot, const double Multiplier, bool use_boundary_repair, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng)
  {

    // Sample independent standard normal variables Z = N(0,1)
    // std::normal_distribution<double> std_normal(0.0, 1.0);
    vec_t z(problem_size);

    // try to sample within bounds
    bool sample_in_range = false;
    unsigned int attempts = 0;


    std::normal_distribution<double> std_normal(0.0, 1.0);

    // try using the normal distribution
    while (!sample_in_range && attempts < 100)
    {

      // sample a new solution
      for (size_t i = 0; i < problem_size; ++i) {
        z[i] = std_normal(*rng);
      }

      sample_transformed = z;
      z *= Multiplier;
      sample = mean + MatrixRoot.product(z);

      if (use_boundary_repair) {
        boundary_repair(sample, lower_param_range, upper_param_range);
      }

      sample_in_range = in_range(sample, lower_param_range, upper_param_range);
      attempts++;

    }
    // if that fails, fall back to uniform from the initial user-defined range
    if (!sample_in_range) {
      sample_uniform(sample, problem_size, lower_param_range, upper_param_range, rng);
      sample_transformed = vec_t(problem_size, 0.0);
      // std::cout << "Warning: Too many sample attempts. Sampling uniform." << std::endl;
    }

    return attempts;

  }

  
  // sample the parameter from a normal distribution
  // make sure it is within the parameter domain.
  //-------------------------------------------------------------------------
  unsigned int sample_normal(vec_t & sample, const size_t problem_size, const vec_t & mean, const matrix_t & chol, bool use_boundary_repair, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng)
  {
    vec_t sample_transformed;
    double multiplier = 1.0;
    return sample_normal(sample, sample_transformed, problem_size, mean, chol, multiplier, use_boundary_repair, lower_param_range, upper_param_range, rng);
  }

  unsigned int sample_normal_univariate(vec_t & sample, const size_t problem_size, const vec_t & mean, const vec_t & univariate_cholesky, bool use_boundary_repair, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng)
  {

    // Sample independent standard normal variables Z = N(0,1)
    std::normal_distribution<double> std_normal(0.0, 1.0);
    sample.resize(problem_size);

    // try to sample within bounds
    bool sample_in_range = false;
    unsigned int attempts = 0;

    // try using the normal distribution
    while (!sample_in_range && attempts < 100)
    {

      // sample a new solution
      for (size_t i = 0; i < problem_size; ++i) {
        sample[i] = mean[i] + univariate_cholesky[i]*std_normal(*rng);
      }

      if (use_boundary_repair) {
        boundary_repair(sample, lower_param_range, upper_param_range);
      }
      sample_in_range = in_range(sample, lower_param_range, upper_param_range);
      attempts++;

    }
    // if that fails, fall back to uniform from the initial user-defined range
    if (!sample_in_range) {
      sample_uniform(sample, problem_size, lower_param_range, upper_param_range, rng);
      // std::cout << "Too many sample attempts. Sample uniform. (mathfunctions.cpp:105)" << std::endl;
    }

    return attempts;

  }
  
  // sample the solution from a uniform distribution
  //-------------------------------------------------------------------------
  void sample_uniform(vec_t & sample, const size_t problem_size, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng)
  {
    
    // resize the parameter vector (in case it is larger)
    sample.resize(problem_size);
    
    assert(lower_param_range.size() == problem_size);
    assert(upper_param_range.size() == problem_size);

    std::uniform_real_distribution<double> unif(0, 1);
    for (size_t i = 0; i < problem_size; ++i)
    {
     
      double r = unif(*rng);
      sample[i] = r * (upper_param_range[i] - lower_param_range[i]) + lower_param_range[i];
      
    }
    
  }
  
  
  // check if a solution is within the parameter bounds
  //-------------------------------------------------------------------------------------
  bool in_range(const vec_t & sample, const vec_t & lower_param_range, const vec_t & upper_param_range)
  {
    
    assert(lower_param_range.size() == sample.size());
    assert(upper_param_range.size() == sample.size());
    
    
    // check each dimension
    for (size_t i = 0; i < (unsigned int) sample.size(); ++i)
    {
      
      // assert(isfinite(sample[i]));

      if (sample[i] < lower_param_range[i] || sample[i] > upper_param_range[i])
        return false;
      
    }
    
    return true;
    
  }
  
  // returns true if the boundary is repaired
  //-------------------------------------------------------------------------------------
  bool boundary_repair(vec_t & sample, const vec_t & lower_param_range, const vec_t & upper_param_range)
  {

    assert(lower_param_range.size() == sample.size());
    assert(upper_param_range.size() == sample.size());
    bool repaired = false;

    // check each dimension
    for (size_t i = 0; i < (unsigned int)sample.size(); ++i)
    {

      if (sample[i] < lower_param_range[i]) {
        sample[i] = lower_param_range[i];
        repaired = true;
      }
        
      if (sample[i] > upper_param_range[i]) {
        sample[i] = upper_param_range[i];
        repaired = true;
      }

    }

    return repaired;

  }


  
  /*-=-=-=-=-=-=-=-=-=-=-= Section Elementary Operations -=-=-=-=-=-=-=-=-=-=-*/
  /**
   * Allocates memory and exits the program in case of a memory allocation failure.
   */
  void *Malloc(long size)
  {
    void *result;
    
    result = (void *)malloc(size);
    
    assert(result);
    
    return(result);
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Matrix -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Creates a new matrix with dimensions n x m.
   */
  double **matrixNew(int n, int m)
  {
    int      i;
    double **result;
    
    result = (double **)malloc(n*(sizeof(double *)));
    for (i = 0; i < n; i++)
      result[i] = (double *)malloc(m*(sizeof(double)));
    
    return(result);
  }
  
  /**
   * Computes the dot product of two vectors of the same dimensionality n0.
   */
  double vectorDotProduct(const double *vector0, const double *vector1, int n0)
  {
    int    i;
    double result;
    
    result = 0.0;
    for (i = 0; i < n0; i++)
      result += vector0[i] * vector1[i];
    
    return(result);
  }
  
  /**
   * Computes the multiplication Av of a matrix A and a vector v
   * where matrix A has dimensions n0 x n1 and vector v has
   * dimensionality n1.
   */
  double *matrixVectorMultiplication(const double **matrix, const double *vector, int n0, int n1)
  {
    int     i;
    double *result;
    
    result = (double *)malloc(n0*sizeof(double));
    for (i = 0; i < n0; i++)
      result[i] = vectorDotProduct(matrix[i], vector, n1);
    
    return(result);
  }
  
  /**
   * Computes the matrix multiplication of two matrices A and B
   * of dimensions A: n0 x n1 and B: n1 x n2.
   */
  double **matrixMatrixMultiplication(double **matrix0, double **matrix1, int n0, int n1, int n2)
  {
    int     i, j, k;
    double **result;
    
    result = (double **)malloc(n0*sizeof(double *));
    for (i = 0; i < n0; i++)
      result[i] = (double *)malloc(n2*sizeof(double));
    
    for (i = 0; i < n0; i++)
    {
      for (j = 0; j < n2; j++)
      {
        result[i][j] = 0;
        for (k = 0; k < n1; k++)
          result[i][j] += matrix0[i][k] * matrix1[k][j];
      }
    }
    
    return(result);
  }


  /**
   * BLAS subroutine.
   */
  int blasDSWAP(int n, double *dx, int incx, double *dy, int incy)
  {
    double dtmp;
    
    if (n > 0)
    {
      incx *= sizeof(double);
      incy *= sizeof(double);
      
      dtmp = (*dx);
      *dx = (*dy);
      *dy = dtmp;
      
      while ((--n) > 0)
      {
        dx = (double *)((char *)dx + incx);
        dy = (double *)((char *)dy + incy);
        dtmp = (*dx); *dx = (*dy); *dy = dtmp;
      }
    }
    
    return(0);
  }
  
  /**
   * BLAS subroutine.
   */
  int blasDAXPY(int n, double da, double *dx, int incx, double *dy, int incy)
  {
    double dtmp0, dtmp, *dx0, *dy0;
    
    if (n > 0 && da != 0.)
    {
      incx *= sizeof(double);
      incy *= sizeof(double);
      *dy += da * (*dx);
      
      if ((n & 1) == 0)
      {
        dx = (double *)((char *)dx + incx);
        dy = (double *)((char *)dy + incy);
        *dy += da * (*dx);
        --n;
      }
      n = n >> 1;
      while (n > 0)
      {
        dy0 = (double *)((char *)dy + incy);
        dy = (double *)((char *)dy0 + incy);
        dtmp0 = (*dy0);
        dtmp = (*dy);
        dx0 = (double *)((char *)dx + incx);
        dx = (double *)((char *)dx0 + incx);
        *dy0 = dtmp0 + da * (*dx0);
        *dy = dtmp + da * (*dx);
        --n;
      }
    }
    
    return(0);
  }
  
  /**
   * BLAS subroutine.
   */
  void blasDSCAL(int n, double sa, double x[], int incx)
  {
    int i, ix, m;
    
    if (n <= 0)
    {
    }
    else if (incx == 1)
    {
      m = n % 5;
      
      for (i = 0; i < m; i++)
      {
        x[i] = sa * x[i];
      }
      
      for (i = m; i < n; i = i + 5)
      {
        x[i] = sa * x[i];
        x[i + 1] = sa * x[i + 1];
        x[i + 2] = sa * x[i + 2];
        x[i + 3] = sa * x[i + 3];
        x[i + 4] = sa * x[i + 4];
      }
    }
    else
    {
      if (0 <= incx)
      {
        ix = 0;
      }
      else
      {
        ix = (-n + 1) * incx;
      }
      
      for (i = 0; i < n; i++)
      {
        x[ix] = sa * x[ix];
        ix = ix + incx;
      }
    }
  }
  
  /**
   * LINPACK subroutine.
   */
  int linpackDCHDC(double a[], int lda, int p, double work[], int ipvt[])
  {
    int    info, j, jp, k, l, maxl, pl, pu;
    double maxdia, temp;
    
    pl = 1;
    pu = 0;
    info = p;
    for (k = 1; k <= p; k++)
    {
      maxdia = a[k - 1 + (k - 1)*lda];
      maxl = k;
      if (pl <= k && k < pu)
      {
        for (l = k + 1; l <= pu; l++)
        {
          if (maxdia < a[l - 1 + (l - 1)*lda])
          {
            maxdia = a[l - 1 + (l - 1)*lda];
            maxl = l;
          }
        }
      }
      
      if (maxdia <= 0.0)
      {
        info = k - 1;
        
        return(info);
      }
      
      if (k != maxl)
      {
        blasDSWAP(k - 1, a + 0 + (k - 1)*lda, 1, a + 0 + (maxl - 1)*lda, 1);
        
        a[maxl - 1 + (maxl - 1)*lda] = a[k - 1 + (k - 1)*lda];
        a[k - 1 + (k - 1)*lda] = maxdia;
        jp = ipvt[maxl - 1];
        ipvt[maxl - 1] = ipvt[k - 1];
        ipvt[k - 1] = jp;
      }
      work[k - 1] = sqrt(a[k - 1 + (k - 1)*lda]);
      a[k - 1 + (k - 1)*lda] = work[k - 1];
      
      for (j = k + 1; j <= p; j++)
      {
        if (k != maxl)
        {
          if (j < maxl)
          {
            temp = a[k - 1 + (j - 1)*lda];
            a[k - 1 + (j - 1)*lda] = a[j - 1 + (maxl - 1)*lda];
            a[j - 1 + (maxl - 1)*lda] = temp;
          }
          else if (maxl < j)
          {
            temp = a[k - 1 + (j - 1)*lda];
            a[k - 1 + (j - 1)*lda] = a[maxl - 1 + (j - 1)*lda];
            a[maxl - 1 + (j - 1)*lda] = temp;
          }
        }
        a[k - 1 + (j - 1)*lda] = a[k - 1 + (j - 1)*lda] / work[k - 1];
        work[j - 1] = a[k - 1 + (j - 1)*lda];
        temp = -a[k - 1 + (j - 1)*lda];
        
        blasDAXPY(j - k, temp, work + k, 1, a + k + (j - 1)*lda, 1);
      }
    }
    
    return(info);
  }
  
  /**
   * Computes the lower-triangle Cholesky Decomposition
   * of a square, symmetric and positive-definite matrix.
   * Subroutines from LINPACK and BLAS are used.
   */
  
  double **choleskyDecomposition( double **matrix, int n )
  {
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
    
    // if matrix is PSD
    if (info == n)
    {
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
    
    // Matrix is non-PSD
    if (info != n)
    {
      // std::cout << " Cholesky-Non-PSD ";
      for( i = 0; i < n; i++ )
      {
        for( j = 0; j < n; j++ )
        {
          result[i][j] = i != j ? 0.0 : sqrt( matrix[i][j] );
        }
      }
    }
    
    free( ipvt );
    free( work );
    free( a );
    
    return( result );
    
  }
  
  // make sure matrix is allocated (nxn) and permutation_order (n)
  double **principalCholeskyDecomposition( double **matrix, int n, std::vector<int> & permutation_order )
  {
    int     i, j, k, info, *ipvt;
    double *a, *work, **result;
    
    a    = (double *) Malloc( n*n*sizeof( double ) );
    work = (double *) Malloc( n*sizeof( double ) );
    ipvt = (int *) Malloc( n*sizeof( int ) );
    
    result = matrixNew( n, n );
    
    double *variances;
    variances = (double *) Malloc( n*sizeof( double ) );
    for( i = 0; i < n; i++ ) {
      variances[i] = -matrix[i][i]; // minus cuz mergeSort sort small to large.
    }
    int * p;
    p = mergeSort( variances, n ); // sorts small to large.
    
    permutation_order.resize(n);
    for(size_t i = 0; i < n; ++i) {
      permutation_order[i] = p[i];
    }
    
    k = 0;
    for( i = 0; i < n; i++ )
    {
      for( j = 0; j < n; j++ )
      {
        a[k] = matrix[p[i]][p[j]]; // permute 'a', the working-copy of 'matrix'
        k++;
      }
      ipvt[i] = 0;
    }
    
    info = linpackDCHDC( a, n, n, work, ipvt );
    
    // copy a back to the matrix, if PSD
    if (info == n)
    {
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
    
    // Matrix is non-PSD
    if( info != n )
    {
      // std::cout << " PrincipalCholesky-NonPSD ";
      k = 0;
      for( i = 0; i < n; i++ )
      {
        for( j = 0; j < n; j++ )
        {
          if (i < info - 1 && j < info - 1) {
            result[i][j] = i < j ? 0.0 : a[k];
          }
          else
          {
            result[i][j] = i != j ? 0.0 : sqrt(matrix[p[i]][p[j]]);
          }
          k++;
        }
      }
    }
    
    free( ipvt );
    free( work );
    free( a );
    free( p );
    
    return( result );
    
  }
  
  
  /**
   * LINPACK subroutine.
   */
  int linpackDTRDI(double t[], int ldt, int n)
  {
    int    j, k, info;
    double temp;
    
    info = 0;
    for (k = n; 1 <= k; k--)
    {
      if (t[k - 1 + (k - 1)*ldt] == 0.0)
      {
        info = k;
        break;
      }
      
      t[k - 1 + (k - 1)*ldt] = 1.0 / t[k - 1 + (k - 1)*ldt];
      temp = -t[k - 1 + (k - 1)*ldt];
      
      if (k != n)
      {
        blasDSCAL(n - k, temp, t + k + (k - 1)*ldt, 1);
      }
      
      for (j = 1; j <= k - 1; j++)
      {
        temp = t[k - 1 + (j - 1)*ldt];
        t[k - 1 + (j - 1)*ldt] = 0.0;
        blasDAXPY(n - k + 1, temp, t + k - 1 + (k - 1)*ldt, 1, t + k - 1 + (j - 1)*ldt, 1);
      }
    }
    
    return(info);
  }
  
  /**
   * Computes the inverse of a matrix that is of
   * lower triangular form.
   */
  double **matrixLowerTriangularInverse(double **matrix, int n)
  {
    int     i, j, k;
    double *t, **result;
    
    t = (double *)Malloc(n*n*sizeof(double));
    
    k = 0;
    for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
      {
        t[k] = matrix[j][i];
        k++;
      }
    }
    
    linpackDTRDI(t, n, n);
    
    result = matrixNew(n, n);
    k = 0;
    for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
      {
        result[j][i] = i > j ? 0.0 : t[k];
        k++;
      }
    }
    
    free(t);
    
    return(result);
  }

  int *mergeSort(double *array, int array_size)
  {
	  int i, *sorted, *tosort;

	  sorted = (int *)Malloc(array_size * sizeof(int));
	  tosort = (int *)Malloc(array_size * sizeof(int));
	  for (i = 0; i < array_size; i++)
		  tosort[i] = i;

	  if (array_size == 1)
		  sorted[0] = 0;
	  else
		  mergeSortWithinBounds(array, sorted, tosort, 0, array_size - 1);

	  free(tosort);

	  return(sorted);
  }

  /**
  * Subroutine of merge sort, sorts the part of the array between p and q.
  */
  void mergeSortWithinBounds(double *array, int *sorted, int *tosort, int p, int q)
  {
	  int r;

	  if (p < q)
	  {
		  r = (p + q) / 2;
		  mergeSortWithinBounds(array, sorted, tosort, p, r);
		  mergeSortWithinBounds(array, sorted, tosort, r + 1, q);
		  mergeSortMerge(array, sorted, tosort, p, r + 1, q);
	  }
  }

  /**
  * Subroutine of merge sort, merges the results of two sorted parts.
  */
  void mergeSortMerge(double *array, int *sorted, int *tosort, int p, int r, int q)
  {
	  int i, j, k, first;

	  i = p;
	  j = r;
	  for (k = p; k <= q; k++)
	  {
		  first = 0;
		  if (j <= q)
		  {
			  if (i < r)
			  {
				  if (array[tosort[i]] < array[tosort[j]])
					  first = 1;
			  }
		  }
		  else
			  first = 1;

		  if (first)
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

	  for (k = p; k <= q; k++)
		  tosort[k] = sorted[k];
  }

  /**
  * Sorts an array of doubles and returns the sort-order (small to large).
  */
  int *mergeSortInt(int *array, int array_size)
  {
	  int i, *sorted, *tosort;

	  sorted = (int *)Malloc(array_size * sizeof(int));
	  tosort = (int *)Malloc(array_size * sizeof(int));
	  for (i = 0; i < array_size; i++)
		  tosort[i] = i;

	  if (array_size == 1)
		  sorted[0] = 0;
	  else
		  mergeSortWithinBoundsInt(array, sorted, tosort, 0, array_size - 1);

	  free(tosort);

	  return(sorted);
  }

  /**
  * Subroutine of merge sort, sorts the part of the array between p and q.
  */
  void mergeSortWithinBoundsInt(int *array, int *sorted, int *tosort, int p, int q)
  {
	  int r;

	  if (p < q)
	  {
		  r = (p + q) / 2;
		  mergeSortWithinBoundsInt(array, sorted, tosort, p, r);
		  mergeSortWithinBoundsInt(array, sorted, tosort, r + 1, q);
		  mergeSortMergeInt(array, sorted, tosort, p, r + 1, q);
	  }
  }

  /**
  * Subroutine of merge sort, merges the results of two sorted parts.
  */
  void mergeSortMergeInt(int *array, int *sorted, int *tosort, int p, int r, int q)
  {
	  int i, j, k, first;

	  i = p;
	  j = r;
	  for (k = p; k <= q; k++)
	  {
		  first = 0;
		  if (j <= q)
		  {
			  if (i < r)
			  {
				  if (array[tosort[i]] < array[tosort[j]])
					  first = 1;
			  }
		  }
		  else
			  first = 1;

		  if (first)
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

	  for (k = p; k <= q; k++)
		  tosort[k] = sorted[k];
  }

  /*
  * Returns all compact integer permutations of
  * a specified length, sorted in ascending
  * radix-sort order.
  */
  int **allPermutations(int length, int *numberOfPermutations)
  {
	  int **result;

	  result = allPermutationsSubroutine(0, length, numberOfPermutations);

	  return(result);
  }



  /*
  * Subroutine of allPermutations.
  */
  int **allPermutationsSubroutine(int from, int length, int *numberOfPermutations)
  {
	  int i, j, k, q, **result, **smallerResult, smallerNumberOfPermutations;

	  (*numberOfPermutations) = 1;
	  for (i = 2; i <= length; i++)
		  (*numberOfPermutations) *= i;

	  result = (int **)Malloc((*numberOfPermutations) * sizeof(int *));
	  for (i = 0; i < *numberOfPermutations; i++)
		  result[i] = (int *)Malloc(length * sizeof(int));

	  if (length == 1)
	  {
		  result[0][0] = from;
	  }
	  else
	  {
		  smallerResult = allPermutationsSubroutine(from + 1, length - 1, &smallerNumberOfPermutations);

		  k = 0;
		  for (i = from; i < from + length; i++)
		  {
			  for (j = 0; j < smallerNumberOfPermutations; j++)
			  {
				  result[k][0] = i;
				  for (q = 1; q < length; q++)
					  result[k][q] = smallerResult[j][q - 1] <= i ? smallerResult[j][q - 1] - 1 : smallerResult[j][q - 1];
				  k++;
			  }
		  }

		  for (i = 0; i < smallerNumberOfPermutations; i++)
			  free(smallerResult[i]);
		  free(smallerResult);
	  }

	  return(result);
  }

  // end BLAS / LINPACK library functions
  
  // Eigen Decomposition
  
  void eigenDecomposition(matrix_t & mat, matrix_t & D, matrix_t & Q)
  {
    
    assert(mat.rows() == mat.cols());
    int n = (int)mat.rows();
    
    double **mat_raw = mat.toArray(); // super nasty code. Allows a const to give a non-const pointer to its data..
    double **D_raw = D.toArray();
    double **Q_raw = Q.toArray();
    
    eigenDecomposition(mat_raw, n, D_raw, Q_raw);

    
  }
  
  
  
  void eigenDecomposition(double **matrix, int n, double **D, double **Q)
  {
    int     i, j;
    double *rgtmp, *diag;

    rgtmp = (double *)Malloc(n*sizeof(double));
    diag = (double *)Malloc(n*sizeof(double));

    for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
      {
        Q[j][i] = matrix[j][i];
        Q[i][j] = Q[j][i];
      }
    }

    eigenDecompositionHouseholder2(n, Q, diag, rgtmp);
    eigenDecompositionQLalgo2(n, Q, diag, rgtmp);

    for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
      {
        D[i][j] = 0.0;
      }
      D[i][i] = diag[i];
    }

    free(diag);
    free(rgtmp);
  }

  vec_t max(const vec_t & v1, const vec_t & v2)
  {

    assert(v1.size() == v2.size());
    vec_t result = vec_t(v1.size());


    for (size_t i = 0; i < v1.size(); ++i) {
      result[i] = std::max(v1[i], v2[i]);
    }

    return result;
  }

  vec_t min(const vec_t & v1, const vec_t & v2)
  {

    assert(v1.size() == v2.size());
    vec_t result = vec_t(v1.size());


    for (size_t i = 0; i < v1.size(); ++i) {
      result[i] = std::min(v1[i], v2[i]);
    }

    return result;
  }

  // computes the rank correlation O(N)
  double rank_correlation(const vec_t & v1, const vec_t & v2)
  {
   
    assert(v1.size() == v2.size());

    double corr = 0.0;
    double rankdifference;
    double N = (double)v1.size();

    if (N <= 1)
      return corr;

    for (size_t i = 0; i < N; ++i) {
      rankdifference = (double)(v1[i]- v2[i]);
      corr += rankdifference*rankdifference;
    }

    // compute the predictive value
    // https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    corr = 1.0 - corr / (N*(N*N - 1.0) / 6.0);

    return corr;
  }

  
  // computes the correlation O(N)
  double correlation(const vec_t & v1, const vec_t & v2)
  {
    
    assert(v1.size() == v2.size());
    
    
    // find the mean
    double mean_v1 = 0;
    double mean_v2 = 0;
    double N = (double) v1.size();
    
    if (N<= 1)
      return 0.0;
    
    for(size_t i = 0; i < N; ++i)
    {
      mean_v1 += v1[i];
      mean_v2 += v2[i];
    }
    
    mean_v1 /= N;
    mean_v2 /= N;
    
    double cov = 0.0;
    double var_v1 = 0.0;
    double var_v2 = 0.0;
    for(size_t i = 0; i < N; ++i)
    {
      cov += (v1[i] - mean_v1) * (v2[i] - mean_v2);
      var_v1 += (v1[i] - mean_v1) * (v1[i] - mean_v1);
      var_v2 += (v2[i] - mean_v2) * (v2[i] - mean_v2);
    }
    
    return cov / (sqrt(var_v1) * sqrt(var_v2));
    
  }

  // largest first
  void compute_ranks_desc(const vec_t & vec, vec_t & ranks)
  {

    size_t N = vec.size();

    ranks.resize(N);

    for (size_t i = 0; i < N; ++i) {
      ranks[i] = (double) i;
    }

    std::sort(std::begin(ranks), std::end(ranks), [&vec](double idx, double idy) { return vec[(size_t) idx] > vec[(size_t) idy]; });

  }

  // smallest first
  void compute_ranks_asc(const vec_t & vec, vec_t & ranks)
  {

    size_t N = vec.size();

    ranks.resize(N);

    for (size_t i = 0; i < N; ++i) {
      ranks[i] = (double)i;
    }

    std::sort(std::begin(ranks), std::end(ranks), [&vec](double idx, double idy) { return vec[(size_t)idx] < vec[(size_t)idy]; });

  }

  // smallest first
  void compute_ranks_asc(const vec_t & vec, std::vector<size_t> & ranks)
  {

    size_t N = vec.size();

    ranks.resize(N);

    for (size_t i = 0; i < N; ++i) {
      ranks[i] = i;
    }

    std::sort(std::begin(ranks), std::end(ranks), [&vec](double idx, double idy) { return vec[(size_t)idx] < vec[(size_t)idy]; });

  }


  // smallest first
  void compute_ranks_asc(double * vec, size_t vec_size, std::vector<size_t> & ranks)
  {
    
    size_t N = vec_size;
    
    ranks.resize(N);
    
    for (size_t i = 0; i < N; ++i) {
      ranks[i] = i;
    }
    
    std::sort(std::begin(ranks), std::end(ranks), [&vec](double idx, double idy) { return vec[(size_t)idx] < vec[(size_t)idy]; });
    
  }



  void eigenDecompositionQLalgo2(int n, double **V, double *d, double *e)
  {
    int i, k, l, m;
    double f = 0.0;
    double tst1 = 0.0;
    double eps = 2.22e-16; /* Math.pow(2.0,-52.0);  == 2.22e-16 */

                           /* shift input e */
    for (i = 1; i < n; i++) {
      e[i - 1] = e[i];
    }
    e[n - 1] = 0.0; /* never changed again */

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
          double p = (d[l + 1] - g) / (2.0 * e[l]);
          double r = myhypot(p, 1.);

          iter = iter + 1;  /* Could check iteration count here */

                            /* Compute implicit shift */

          if (p < 0) {
            r = -r;
          }
          d[l] = e[l] / (p + r);
          d[l + 1] = e[l] * (p + r);
          dl1 = d[l + 1];
          h = g - d[l];
          for (i = l + 2; i < n; i++) {
            d[i] -= h;
          }
          f = f + h;

          /* Implicit QL transformation. */

          p = d[m];
          {
            double c = 1.0;
            double c2 = c;
            double c3 = c;
            double el1 = e[l + 1];
            double s = 0.0;
            double s2 = 0.0;
            for (i = m - 1; i >= l; i--) {
              c3 = c2;
              c2 = c;
              s2 = s;
              g = c * e[i];
              h = c * p;
              r = myhypot(p, e[i]);
              e[i + 1] = s * r;
              s = e[i] / r;
              c = p / r;
              p = c * d[i] - s * g;
              d[i + 1] = h + s * (c * g + s * d[i]);

              /* Accumulate transformation. */

              for (k = 0; k < n; k++) {
                h = V[k][i + 1];
                V[k][i + 1] = s * V[k][i] + c * h;
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
      for (i = 0; i < n - 1; i++) {
        k = i;
        p = d[i];
        for (j = i + 1; j < n; j++) {
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

  double myhypot(double a, double b)
  {
    double r = 0;
    if (fabs(a) > fabs(b))
    {
      r = b / a;
      r = fabs(a)*sqrt(1 + r*r);
    }
    else if (b != 0)
    {
      r = a / b;
      r = fabs(b)*sqrt(1 + r*r);
    }

    return r;
  }

  void eigenDecompositionHouseholder2(int n, double **V, double *d, double *e)
  {
    int i, j, k;

    for (j = 0; j < n; j++) {
      d[j] = V[n - 1][j];
    }

    /* Householder reduction to tridiagonal form */

    for (i = n - 1; i > 0; i--) {

      /* Scale to avoid under/overflow */

      double scale = 0.0;
      double h = 0.0;
      for (k = 0; k < i; k++) {
        scale = scale + fabs(d[k]);
      }
      if (scale == 0.0) {
        e[i] = d[i - 1];
        for (j = 0; j < i; j++) {
          d[j] = V[i - 1][j];
          V[i][j] = 0.0;
          V[j][i] = 0.0;
        }
      }
      else {

        /* Generate Householder vector */

        double f, g, hh;

        for (k = 0; k < i; k++) {
          d[k] /= scale;
          h += d[k] * d[k];
        }
        f = d[i - 1];
        g = sqrt(h);
        if (f > 0) {
          g = -g;
        }
        e[i] = scale * g;
        h = h - f * g;
        d[i - 1] = f - g;
        for (j = 0; j < i; j++) {
          e[j] = 0.0;
        }

        /* Apply similarity transformation to remaining columns */

        for (j = 0; j < i; j++) {
          f = d[j];
          V[j][i] = f;
          g = e[j] + V[j][j] * f;
          for (k = j + 1; k <= i - 1; k++) {
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
          for (k = j; k <= i - 1; k++) {
            V[k][j] -= (f * e[k] + g * d[k]);
          }
          d[j] = V[i - 1][j];
          V[i][j] = 0.0;
        }
      }
      d[i] = h;
    }

    /* Accumulate transformations */

    for (i = 0; i < n - 1; i++) {
      double h;
      V[n - 1][i] = V[i][i];
      V[i][i] = 1.0;
      h = d[i + 1];
      if (h != 0.0) {
        for (k = 0; k <= i; k++) {
          d[k] = V[k][i + 1] / h;
        }
        for (j = 0; j <= i; j++) {
          double g = 0.0;
          for (k = 0; k <= i; k++) {
            g += V[k][i + 1] * V[k][j];
          }
          for (k = 0; k <= i; k++) {
            V[k][j] -= g * d[k];
          }
        }
      }
      for (k = 0; k <= i; k++) {
        V[k][i + 1] = 0.0;
      }
    }
    for (j = 0; j < n; j++) {
      d[j] = V[n - 1][j];
      V[n - 1][j] = 0.0;
    }
    V[n - 1][n - 1] = 1.0;
    e[0] = 0.0;

  }

  /**
  * Selects n points from a set of points. A greedy heuristic is used to find a good
  * scattering of the selected points. First, a point is selected with a maximum value
  * in a randomly selected dimension. The remaining points are selected iteratively.
  * In each iteration, the point selected is the one that maximizes the minimal distance
  * to thepoints selected so far.
  */
  void greedyScatteredSubsetSelection(std::vector<vec_t> & points, size_t number_to_select, std::vector<size_t> & result, rng_pt & rng)
  {
    if (points.size() == 0 || number_to_select == 0)  {
      return;
    }
    
    size_t number_of_points = points.size();
    size_t number_of_dimensions = points[0].size();

    std::vector<size_t> indices_left(number_of_points, 0);
    for (size_t i = 0; i < number_of_points; i++) {
      indices_left[i] = i;
    }

    // the original code gave an error and died. I think this is acceptable as well without exiting.
    if (number_to_select > number_of_points) {
      result = indices_left;
      return;
    }

    result.resize(number_to_select);

    // Find the first point: maximum value in a randomly chosen dimension 
    std::uniform_real_distribution<double> unif(0, 1);
    size_t random_dimension_index = (size_t)(unif(*rng) * number_of_dimensions);

    size_t index_of_farthest = 0;
    double distance_of_farthest = points[indices_left[index_of_farthest]][random_dimension_index];

    for (size_t i = 1; i < number_of_points; i++)
    {
      if (points[indices_left[i]][random_dimension_index] > distance_of_farthest)
      {
        index_of_farthest = i;
        distance_of_farthest = points[indices_left[i]][random_dimension_index];
      }
    }

    size_t number_selected_so_far = 0;
    result[number_selected_so_far] = indices_left[index_of_farthest];
    indices_left[index_of_farthest] = indices_left[number_of_points - number_selected_so_far - 1];
    number_selected_so_far++;

    /* Then select the rest of the solutions: maximum minimum
    * (i.e. nearest-neighbour) distance to so-far selected points */
    
    vec_t nn_distances(number_of_points, 0.0);

    for (size_t i = 0; i < number_of_points - number_selected_so_far; i++) {
      nn_distances[i] = (points[indices_left[i]] - points[result[number_selected_so_far - 1]]).norm();
    }

    while (number_selected_so_far < number_to_select)
    {
      index_of_farthest = 0;
      distance_of_farthest = nn_distances[0];
      for (size_t i = 1; i < number_of_points - number_selected_so_far; i++)
      {
        if (nn_distances[i] > distance_of_farthest)
        {
          index_of_farthest = i;
          distance_of_farthest = nn_distances[i];
        }
      }

      result[number_selected_so_far] = indices_left[index_of_farthest];
      indices_left[index_of_farthest] = indices_left[number_of_points - number_selected_so_far - 1];
      nn_distances[index_of_farthest] = nn_distances[number_of_points - number_selected_so_far - 1];
      number_selected_so_far++;

      double value;
      for (size_t i = 0; i < number_of_points - number_selected_so_far; i++)
      {
        value = (points[indices_left[i]] - points[result[number_selected_so_far - 1]]).norm();
        if (value < nn_distances[i]) {
          nn_distances[i] = value;
        }
      }
    }

  }

  // push_back exactly 'number_of_solutions_to_select' from 'solutions' to 'selected_solutions', based on a greedy diversity selection
  void selectSolutionsBasedOnObjectiveDiversity(const std::vector<solution_pt> & solutions, size_t number_of_solutions_to_select, std::vector<solution_pt> & selected_solutions, const vec_t & obj_ranges, std::vector<solution_pt> & non_selected_solutions, rng_pt & rng)
  {

    if(solutions.size() == 0) {
      return;
    }
    
    // we scale the objectives to the objective ranges before performing subset selection
    // we also filter out the potential nullptr solutions
    std::vector<vec_t> scaled_objectives;
    scaled_objectives.reserve(solutions.size());
    std::vector<size_t> non_nullptr_solution_index;
    non_nullptr_solution_index.reserve(solutions.size());
    
    for (size_t i = 0; i < solutions.size(); ++i)
    {
      
      if(solutions[i] != nullptr)
      {
        vec_t scaled_objective(solutions[i]->number_of_objectives(), 0.0);

        for (size_t j = 0; j < scaled_objective.size(); ++j) {
          scaled_objective[j] = solutions[i]->obj[j] / obj_ranges[j];
        }
        scaled_objectives.push_back(scaled_objective);
        non_nullptr_solution_index.push_back(i);
      }
    }
    
    // Subset Selection
    std::vector<size_t> selected_indices;
    selected_indices.reserve(number_of_solutions_to_select);

    greedyScatteredSubsetSelection(scaled_objectives, (int)number_of_solutions_to_select, selected_indices, rng);

    // Copy to selection 
    std::vector<bool> non_selected_indices(solutions.size(), true);
    selected_solutions.reserve(selected_solutions.size() + number_of_solutions_to_select);

    for (size_t i = 0; i < selected_indices.size(); i++) {
      selected_solutions.push_back(solutions[non_nullptr_solution_index[selected_indices[i]]]);
      non_selected_indices[non_nullptr_solution_index[selected_indices[i]]] = false;
    }

    // Copy non_selected_solutions to the non_selection 
    for (size_t i = 0; i < non_selected_indices.size(); ++i) {
      if (non_selected_indices[i]) {
        non_selected_solutions.push_back(solutions[i]);
      }
    }

  }


  void selectSolutionsBasedOnObjectiveDiversity(const std::vector<solution_pt> & solutions, size_t number_of_solutions_to_select, std::vector<solution_pt> & selected_solutions, const vec_t & obj_ranges, rng_pt & rng)
  {

    std::vector<solution_pt> non_selected_solutions;
    selectSolutionsBasedOnObjectiveDiversity(solutions, number_of_solutions_to_select, selected_solutions, obj_ranges, non_selected_solutions, rng);

  }


  // push_back exactly 'number_of_solutions_to_select' from 'solutions' to 'selected_solutions', based on a greedy diversity selection
  // non-const because the random number generator is used.
  // does not use any members of optimizer_t, and doesn't have to be a member-function therefore.
  void selectSolutionsBasedOnParameterDiversity(const std::vector<solution_pt> & solutions, size_t number_of_solutions_to_select, std::vector<solution_pt> & selected_solutions, std::vector<solution_pt> & non_selected_solutions, rng_pt & rng)
  {

    if(solutions.size() == 0) {
      return;
    }
    
    // we scale the objectives to the objective ranges before performing subset selection
    // we also filter out the potential nullptr solutions
    std::vector<vec_t> parameters;
    parameters.reserve(solutions.size());
    std::vector<size_t> non_nullptr_solution_index;
    non_nullptr_solution_index.reserve(solutions.size());
    
    for (size_t i = 0; i < solutions.size(); ++i)
    {
      
      if(solutions[i] != nullptr)
      {
        vec_t parameter(solutions[i]->number_of_parameters(), 0.0);
        
        for (size_t j = 0; j < parameter.size(); ++j) {
          parameter[j] = solutions[i]->param[j];
        }
        parameters.push_back(parameter);
        non_nullptr_solution_index.push_back(i);
      }
    }
    
    // Subset Selection
    std::vector<size_t> selected_indices;
    selected_indices.reserve(number_of_solutions_to_select);
    
    greedyScatteredSubsetSelection(parameters, (int)number_of_solutions_to_select, selected_indices, rng);
    
    // Copy to selection
    std::vector<bool> non_selected_indices(solutions.size(), true);
    selected_solutions.reserve(selected_solutions.size() + number_of_solutions_to_select);
    
    for (size_t i = 0; i < selected_indices.size(); i++) {
      selected_solutions.push_back(solutions[non_nullptr_solution_index[selected_indices[i]]]);
      non_selected_indices[non_nullptr_solution_index[selected_indices[i]]] = false;
    }
    
    // Copy non_selected_solutions to the non_selection
    for (size_t i = 0; i < non_selected_indices.size(); ++i) {
      if (non_selected_indices[i]) {
        non_selected_solutions.push_back(solutions[i]);
      }
    }

  }

  void selectSolutionsBasedOnParameterDiversity(const std::vector<solution_pt> & solutions, size_t number_of_solutions_to_select, std::vector<solution_pt> & selected_solutions, rng_pt & rng)
  {

    std::vector<solution_pt> non_selected_solutions;

    selectSolutionsBasedOnParameterDiversity(solutions, number_of_solutions_to_select, selected_solutions, non_selected_solutions, rng);

  }


  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


  double distanceEuclidean(double *x, double *y, int number_of_dimensions)
  {
	  int    i;
	  double value, result;

	  result = 0.0;
	  for (i = 0; i < number_of_dimensions; i++)
	  {
		  value = y[i] - x[i];
		  result += value*value;
	  }
	  result = sqrt(result);

	  return(result);
  }
  
  double distanceAbsolute(double *x, double *y, int number_of_dimensions)
  {
    int    i;
    double value, result;
    
    result = 0.0;
    for (i = 0; i < number_of_dimensions; i++)
    {
      value = y[i] - x[i];
      result += std::fabs(value); // value*value;
    }
    // result = sqrt(result);
    
    return(result);
  }
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  
  /*     References
   ----------
   .. [Wessing2015] Wessing, Simon (2015). Two-stage Methods for Multimodal
   Optimization. PhD Thesis, Technische Universität Dortmund.
   http://hdl.handle.net/2003/34148
   
   re-implemented by S.C. Maree in C++
   
   
   existing_samples: array_like, optional
   samples that cannot be modified anymore, but should be considered in
   the distance computations.
   
   */
  void maximin_reconstruction(const size_t number_of_samples, const size_t dimension, std::vector<std::shared_ptr<vec_t>> & samples, const std::vector<std::shared_ptr<vec_t>> & existing_samples, rng_pt rng)
  {
    
    samples.clear();
    
    if (number_of_samples <= 0)
      return;
    
    size_t number_of_steps = 100 * number_of_samples;
    
    // initial set of samples
    stratified_sampling(number_of_samples, dimension, samples, rng);
    
    vec_t distances, dists_to_existing_points;
    
    // list(range(num_points));
    std::vector<size_t> remaining_indices(number_of_samples);
    for (size_t i = 0; i < remaining_indices.size(); ++i) {
      remaining_indices[i] = i;
    }
    std::random_shuffle(std::begin(remaining_indices), std::end(remaining_indices));
    
    size_t removal_candidate_index = remaining_indices.back();
    remaining_indices.pop_back(); // python .pop() does these two things.
    
    std::shared_ptr<vec_t> removal_candidate = samples[removal_candidate_index];
    
    dist_matrix_function(*removal_candidate, samples, distances); // [0]; // what does that [0] do??
    distances[removal_candidate_index] = 1e308; // np.inf
    
    double current_dist = distances.min();
    if (existing_samples.size() > 0)
    {
      dist_matrix_function(*removal_candidate, existing_samples, dists_to_existing_points); // [0]; // what does that [0] do??
      current_dist = std::min(current_dist, dists_to_existing_points.min());
    }
    
    
    std::shared_ptr<vec_t> new_sample = std::make_shared<vec_t>(dimension);
    double new_dist;
    std::uniform_real_distribution<double> unif(0, 1);
    
    
    for (size_t step = 0; step < number_of_steps; ++step)
    {
      
      for (size_t i = 0; i < dimension; ++i) {
        (*new_sample)[i] = unif(*rng);
      }
      
      new_dist = 1e308; // new_dist = np.inf
      
      if (new_dist >= current_dist)
      {
        dist_matrix_function(*new_sample, samples, distances); // [0]; // what does that [0] do??
        distances[removal_candidate_index] = 1e308; // np.inf
        new_dist = std::min(new_dist, distances.min());
        
        if (new_dist >= current_dist && existing_samples.size() > 0)
        {
          dist_matrix_function(*new_sample, existing_samples, dists_to_existing_points); // [0]; // what does that [0] do??
          new_dist = std::min(new_dist, dists_to_existing_points.min());
        }
      }
      
      if (new_dist >= current_dist)
      {
        // accept new point
        // std::cout << "+";
        samples[removal_candidate_index] = new_sample;
        current_dist = new_dist;
        removal_candidate = new_sample;
        // removal_candidate_index stays the same, but reset other indices
        
        remaining_indices.clear();
        remaining_indices.reserve(number_of_samples - 1);
        
        for (size_t j = 0; j < number_of_samples; ++j)
        {
          if (j != removal_candidate_index) {
            remaining_indices.push_back(j);
          }
        }
        
        // random.shuffle(remaining_indices)
        std::random_shuffle(std::begin(remaining_indices), std::end(remaining_indices));
      }
      else
      {
        // std::cout << "-";
        // failed to find better point in this iteration
        if (remaining_indices.size() > 0)
        {
          // carry out one attempt to find new removal candidate
          size_t removal_candidate_candidate_index = remaining_indices.back(); //  .pop() in python does these two things.
          remaining_indices.pop_back();
          
          std::shared_ptr<vec_t> removal_candidate_candidate = samples[removal_candidate_candidate_index];
          
          // calculate minimal distance
          dist_matrix_function(*removal_candidate_candidate, samples, distances); // [0]; // what does that [0] do??
          distances[removal_candidate_candidate_index] = 1e308; //  np.inf
          double candidate_candidate_dist = distances.min();
          if (existing_samples.size() > 0)
          {
            dist_matrix_function(*removal_candidate_candidate, existing_samples, dists_to_existing_points); // [0]; // what does that [0] do??
            candidate_candidate_dist = std::min(candidate_candidate_dist, dists_to_existing_points.min());
          }
          if (candidate_candidate_dist <= current_dist)
          {
            // found new removal candidate
            removal_candidate = removal_candidate_candidate;
            current_dist = candidate_candidate_dist;
            removal_candidate_index = removal_candidate_candidate_index;
          }
          
        }
      }
    }
    
    
  } // end maximin
  
  void dist_matrix_function(const vec_t & sample, const std::vector<std::shared_ptr<vec_t>> & ref_samples, vec_t & distances)
  {
    
    distances.resize(ref_samples.size(), 0.0);
    
    for (size_t i = 0; i < ref_samples.size(); ++i)
    {
      dist_matrix_function(sample, *ref_samples[i], distances[i]);
    }
    
  }
  
  void dist_matrix_function(const vec_t & sample, const vec_t & ref_sample, double & distance)
  {
    
    distance = 0.0;
    
    double diff;
    for (size_t i = 0; i < sample.size(); ++i)
    {
      diff = std::abs(sample[i] - ref_sample[i]);
      distance += std::min(diff, 1.0-diff);
    }
    
  }
  
  
  
  void stratified_sampling(const size_t number_of_samples, const size_t dimension, std::vector<std::shared_ptr<vec_t>> & samples, rng_pt rng)
  {
    
    // int bates_param = 1;
    bool avoid_odd_numbers = true;
    
    // init a unit cube.
    std::shared_ptr<std::vector<vec_t>> cuboid = std::make_shared<std::vector<vec_t>>(dimension);
    
    for (size_t i = 0; i < dimension; ++i)
    {
      (*cuboid)[i].resize(2);
      (*cuboid)[i][0] = 0.0;
      (*cuboid)[i][1] = 1.0;
    }
    
    std::vector<size_t> dimensions(dimension, 0);
    for (size_t i = 0; i < dimension; ++i) {
      dimensions[i] = i;
    }
    
    samples.clear();
    samples.resize(number_of_samples);
    
    
    std::vector<size_t> remaining_strata_number_of_samples(1, number_of_samples);
    std::vector<std::shared_ptr<std::vector<vec_t>>> remaining_strata_cuboid(1, cuboid);
    std::vector<std::shared_ptr<std::vector<vec_t>>> final_strata(0);
    
    std::uniform_real_distribution<double> unif(0, 1);
    
    vec_t diffs(dimension, 0.0);
    size_t current_num_points;
    while (remaining_strata_number_of_samples.size() > 0)
    {
      
      current_num_points = remaining_strata_number_of_samples.back();
      remaining_strata_number_of_samples.pop_back();
      
      std::shared_ptr<std::vector<vec_t>> current_bounds = remaining_strata_cuboid.back();
      remaining_strata_cuboid.pop_back();
      
      if (current_num_points == 1)
      {
        final_strata.push_back(current_bounds);
        continue;
      }
      
      
      for (size_t i = 0; i < dimension; ++i) {
        diffs[i] = (*current_bounds)[i][1] - (*current_bounds)[i][0];
      }
      
      double max_extent = diffs.max();
      
      std::vector<size_t> max_extent_dims(0);
      
      for (size_t i = 0; i < dimension; ++i)
      {
        if (diffs[i] == max_extent) {
          max_extent_dims.push_back(i);
        }
      }
      
      size_t num1 = (size_t) (current_num_points * 0.5);
      
      bool do_subtract_one = (avoid_odd_numbers && current_num_points >= 6) && num1 % 2 != 0 && current_num_points % 2 == 0;
      if (do_subtract_one) {
        num1--;
      }
      
      size_t num2 = current_num_points - num1;
      
      if (unif(*rng) < 0.5)
      {
        size_t num_temp = num2;
        num2 = num1;
        num1 = num_temp;
      }
      
      size_t random_index = (size_t) (unif(*rng) * max_extent_dims.size());
      size_t split_dim = max_extent_dims[random_index];
      
      double split_pos = num1 / ((double) current_num_points);
      split_pos = (*current_bounds)[split_dim][0] + max_extent * split_pos;
      
      std::shared_ptr<std::vector<vec_t>> new_bounds1 = std::make_shared<std::vector<vec_t>>(*current_bounds);
      std::shared_ptr<std::vector<vec_t>> new_bounds2 = std::make_shared<std::vector<vec_t>>(*current_bounds);
      
      (*new_bounds1)[split_dim][1] = split_pos;
      (*new_bounds2)[split_dim][0] = split_pos;
      
      remaining_strata_cuboid.push_back(new_bounds1);
      remaining_strata_number_of_samples.push_back(num1);
      
      remaining_strata_cuboid.push_back(new_bounds2);
      remaining_strata_number_of_samples.push_back(num2);
    }
    
    // bates_param = 1
    for (size_t i = 0; i < final_strata.size(); ++i)
    {
      samples[i] = std::make_shared<vec_t>(dimension, 0.0);
      
      for (size_t j = 0; j < dimension; ++j)
      {
        (*samples[i])[j] = unif(*rng) * ((*final_strata[i])[j][1] - (*final_strata[i])[j][0]) + (*final_strata[i])[j][0];
        
      }
    }
    
  }


}


