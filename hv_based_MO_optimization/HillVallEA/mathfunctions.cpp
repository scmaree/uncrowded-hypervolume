/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

*/

#include "mathfunctions.hpp"
#include "solution.hpp"

namespace hillvallea 
{

  
  
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
  
  // uses diag(cov) only
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
  int sample_normal(vec_t & sample, const size_t problem_size, const vec_t & mean, const matrix_t & MatrixRoot, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng)
  {
    vec_t sample_transformed;
    return sample_normal(sample, sample_transformed, problem_size, mean, MatrixRoot, lower_param_range, upper_param_range, rng);
  }
  
  int sample_normal(vec_t & sample, vec_t & z, const size_t problem_size, const vec_t & mean, const matrix_t & MatrixRoot, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng)
  {

    // Sample independent standard normal variables Z = N(0,1)
    std::normal_distribution<double> std_normal(0.0, 1.0);
    // vec_t z(problem_size);
    z.resize(problem_size); // param_transformed
    
    // try to sample within bounds
    bool sample_in_range = false;
    int attempts = 0;

    // try using the normal distribution
    while (!sample_in_range && attempts < 100)
    {

      // sample a new solution
      for (size_t i = 0; i < problem_size; ++i) {
        z[i] = std_normal(*rng);
      }

      sample = mean + MatrixRoot.lowerProduct(z);
      boundary_repair(sample, lower_param_range, upper_param_range);

      sample_in_range = in_range(sample, lower_param_range, upper_param_range);
      attempts++;

    }

    // if that fails, fall back to uniform from the initial user-defined range
    if (!sample_in_range) {
      sample_uniform(sample, problem_size, lower_param_range, upper_param_range, rng);
    }

    return attempts;

  }

  int sample_normal_univariate(vec_t & sample, const size_t problem_size, const vec_t & mean, const vec_t & chol, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng)
  {

    // Sample independent standard normal variables Z = N(0,1)
    std::normal_distribution<double> std_normal(0.0, 1.0);
    sample.resize(problem_size);

    // try to sample within bounds
    bool sample_in_range = false;
    int attempts = 0;

    // try using the normal distribution
    while (!sample_in_range && attempts < 100)
    {

      // sample a new solution
      for (size_t i = 0; i < problem_size; ++i) {
        sample[i] = mean[i] + chol[i]*std_normal(*rng);
      }

      boundary_repair(sample, lower_param_range, upper_param_range);

      sample_in_range = in_range(sample, lower_param_range, upper_param_range);
      attempts++;

    }
    // if that fails, fall back to uniform from the initial user-defined range
    if (!sample_in_range) {
      sample_uniform(sample, problem_size, lower_param_range, upper_param_range, rng);
      std::cout << "Too many sample attempts. Sample uniform. (mathfunctions.cpp:105)" << std::endl;
    }

    return attempts;

  }

  
  int sample_normal_univariate(vec_t & sample, vec_t & untransformed_sample, const size_t problem_size, const vec_t & mean, const vec_t & chol, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng)
  {
    
    // Sample independent standard normal variables Z = N(0,1)
    std::normal_distribution<double> std_normal(0.0, 1.0);
    sample.resize(problem_size);
    untransformed_sample.resize(problem_size);
    
    // try to sample within bounds
    bool sample_in_range = false;
    int attempts = 0;
    
    // try using the normal distribution
    while (!sample_in_range && attempts < 100)
    {
      
      // sample a new solution
      for (size_t i = 0; i < problem_size; ++i) {
        untransformed_sample[i] = std_normal(*rng);
        sample[i] = mean[i] + chol[i]*untransformed_sample[i];
      }
      
      boundary_repair(sample, lower_param_range, upper_param_range);
      
      sample_in_range = in_range(sample, lower_param_range, upper_param_range);
      attempts++;
      
    }
    // if that fails, fall back to uniform from the initial user-defined range
    if (!sample_in_range) {
      sample_uniform(sample, problem_size, lower_param_range, upper_param_range, rng);
      std::cout << "Too many sample attempts. Sample uniform. (mathfunctions.cpp:105)" << std::endl;
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
    for (size_t i = 0; i < (int) sample.size(); ++i)
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
    /* for (size_t i = 0; i < (int)sample.size(); ++i)
    {

      if (sample[i] < lower_param_range[i]) {
        sample[i] = lower_param_range[i];
        repaired = true;
      }
        
      if (sample[i] > upper_param_range[i]) {
        sample[i] = upper_param_range[i];
        repaired = true;
      }

    }*/

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
  double *matrixVectorMultiplication(double **matrix, double *vector, int n0, int n1)
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
   // Cholesky decomposition
   //---------------------------------------------------------------------------
  void choleskyDecomposition(const matrix_t & cov, matrix_t & chol)
  {
    bool success = true;
    choleskyDecomposition(cov, chol, success);
  }
  
  void choleskyDecomposition(const matrix_t & cov, matrix_t & chol, bool & success)
  {
    
    assert(cov.rows() == cov.cols());
    int n = (int)cov.rows();
    
    double **cholesky_factor_lower_triangle;
    cholesky_factor_lower_triangle = choleskyDecomposition(cov.toArray(), n, success);
    
    chol.setRaw(cholesky_factor_lower_triangle, n, n);
  }

  void choleskyDecomposition_univariate(const matrix_t & cov, matrix_t & chol)
  {

    assert(cov.rows() == cov.cols());
    int n = (int)cov.rows();

    chol.reset(n, n, 0.0);

    for (int i = 0; i < n; ++i) {
      chol[i][i] = sqrt(cov[i][i]);
    }
  }

  void choleskyDecomposition_univariate(const vec_t & cov, vec_t & chol)
  {
    chol.resize(cov.size());
    
    for (size_t i = 0; i < cov.size(); ++i) {
      chol[i] = sqrt(cov[i]);
    }
    
  }
  
  double **choleskyDecomposition(double **matrix, int n)
  {
    bool success;
    return choleskyDecomposition(matrix, n, success);
  }
  
  double **choleskyDecomposition(double **matrix, int n, bool & success)
  {
    success = false;

    int     i, j, k, info, *ipvt;
    double *a, *work, **result;
    
    a = (double *)Malloc(n*n*sizeof(double));
    work = (double *)Malloc(n*sizeof(double));
    ipvt = (int *)Malloc(n*sizeof(int));
    
    k = 0;
    for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
      {
        a[k] = matrix[i][j];
        k++;
      }
      ipvt[i] = 0;
    }
    
    info = linpackDCHDC(a, n, n, work, ipvt);
    
    result = matrixNew(n, n);
    if (info != n) /* Matrix is not positive definite */
    {
      success = false;
      // std::cout << " Univariate Cholesky decomposition performed" << std::endl;
      k = 0;
      for (i = 0; i < n; i++)
      {
        for (j = 0; j < n; j++)
        {
          result[i][j] = i != j ? 0.0 : sqrt(matrix[i][j]);
          k++;
        }
      }
    }
    else
    {
      success = true;
      k = 0;
      for (i = 0; i < n; i++)
      {
        for (j = 0; j < n; j++)
        {
          result[i][j] = i < j ? 0.0 : a[k];
          k++;
        }
      }
    }
    
    free(ipvt);
    free(work);
    free(a);
    
    return(result);
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

  // end BLAS / LINPACK library functions

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
        vec_t parameter(solutions[i]->param.size(), 0.0);
        
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
  void compute_ranks_asc(const vec_t & vec, std::vector<size_t> & ranks)
  {
    
    size_t N = vec.size();
    
    ranks.resize(N);
    
    for (size_t i = 0; i < N; ++i) {
      ranks[i] = i;
    }
    
    std::sort(std::begin(ranks), std::end(ranks), [&vec](double idx, double idy) { return vec[(size_t)idx] < vec[(size_t)idy]; });
    
  }
  
  void compute_ranks_asc(const vec_t & vec, vec_t & ranks)
  {
    
    size_t N = vec.size();
    
    ranks.resize(N);
    
    for (size_t i = 0; i < N; ++i) {
      ranks[i] = (double)i;
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
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Random Numbers -=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Returns a random double, distributed uniformly between 0 and 1.
   */
  double randomRealUniform01( rng_t & rng )
  {
    std::uniform_real_distribution<double> unif(0, 1);
    return( unif(rng) );
  }
  
  /**
   * Returns a random integer, distributed uniformly between 0 and maximum.
   */
  int randomInt( int maximum, rng_t & rng )
  {
    int result;
    
    result = (int) (((double) maximum)*randomRealUniform01(rng));
    
    return( result );
  }

  /**
   * Returns a random compact (using integers 0,1,...,n-1) permutation
   * of length n using the Fisher-Yates shuffle.
   */
  int *randomPermutation( int n, rng_t & rng )
  {
    int i, j, dummy, *result;
    
    result = (int *) Malloc( n*sizeof( int ) );
    for( i = 0; i < n; i++ )
      result[i] = i;
    
    for( i = n-1; i > 0; i-- )
    {
      j         = randomInt( i+1, rng );
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


