#pragma once

/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

*/

#include "hillvallea_internal.hpp"
#include "param.hpp"

namespace hillvallea
{

  // Normal pdf
  // returns the value of the pdf at a given point.
  // assumes all input is of the same dimension as mean.
  //---------------------------------------------------------------------------
  double normpdf(const vec_t & mean, const matrix_t & cov, const vec_t & x, matrix_t & chol, matrix_t & inverse_chol);
  double normpdf(const vec_t & mean, const matrix_t & chol, const matrix_t & inverse_chol, const vec_t & x);
  double normpdf_diagonal(const vec_t & mean, const vec_t & cov_diagonal, const vec_t & x);           // uses diag(cov) only
  double normcdf(const double x);
  
  // sample parameters using normal distribution
  // returns the number of trials before an in-range sample is found.
  // on too many fails, sample uniform.
  //----------------------------------------------
  int sample_normal(vec_t & sample, const size_t problem_size, const vec_t & mean, const matrix_t & chol, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng);
  int sample_normal_univariate(vec_t & sample, const size_t problem_size, const vec_t & mean, const vec_t & chol, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng);
  // with storing untransformed sample.
  int sample_normal(vec_t & sample, vec_t & untransformed_sample, const size_t problem_size, const vec_t & mean, const matrix_t & chol, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng);
  int sample_normal_univariate(vec_t & sample, vec_t & untransformed_sample, const size_t problem_size, const vec_t & mean, const vec_t & chol, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng);

  void sample_uniform(vec_t & sample, const size_t problem_size, const vec_t & lower_user_range, const vec_t & upper_user_range,  rng_pt rng);
  
  // check if the parameter is in range
  //-----------------------------------------
  bool in_range(const vec_t & sample, const vec_t & lower_param_range, const vec_t & upper_param_range);
  bool boundary_repair(vec_t & sample, const vec_t & lower_param_range, const vec_t & upper_param_range);

  // Cholesky decomposition
  // based on BLAS / LINPACK library functions
  //-----------------------------------------
  void choleskyDecomposition(const matrix_t & cov, matrix_t & chol);
  void choleskyDecomposition(const matrix_t & cov, matrix_t & chol, bool & success);
  void choleskyDecomposition_univariate(const matrix_t & cov, matrix_t & chol);
  void choleskyDecomposition_univariate(const vec_t & cov, vec_t & chol);
  void *Malloc(long size);
  double **matrixNew(int n, int m);
  double vectorDotProduct(const double *vector0, const double *vector1, int n0);
  double *matrixVectorMultiplication(double **matrix, double *vector, int n0, int n1);
  double **matrixMatrixMultiplication(double **matrix0, double **matrix1, int n0, int n1, int n2);
  int blasDSWAP(int n, double *dx, int incx, double *dy, int incy);
  int blasDAXPY(int n, double da, double *dx, int incx, double *dy, int incy);
  void blasDSCAL(int n, double sa, double x[], int incx);
  int linpackDCHDC(double a[], int lda, int p, double work[], int ipvt[]);
  double **choleskyDecomposition(double **matrix, int n);
  double **choleskyDecomposition(double **matrix, int n, bool & success);
  int linpackDTRDI(double t[], int ldt, int n);
  double **matrixLowerTriangularInverse(double **matrix, int n);
  
  
  /**
   * Selects n points from a set of points. A greedy heuristic is used to find a good
   * scattering of the selected points. First, a point is selected with a maximum value
   * in a randomly selected dimension. The remaining points are selected iteratively.
   * In each iteration, the point selected is the one that maximizes the minimal distance
   * to the points selected so far.
   */
  void greedyScatteredSubsetSelection(std::vector<vec_t> & points, size_t number_to_select, std::vector<size_t> & selected_indices, rng_pt & rng);

  void selectSolutionsBasedOnParameterDiversity(const std::vector<solution_pt> & solutions, size_t number_of_solutions_to_select, std::vector<solution_pt> & selected_solutions, std::vector<solution_pt> & non_selected_solutions, rng_pt & rng);
  void selectSolutionsBasedOnParameterDiversity(const std::vector<solution_pt> & solutions, size_t number_of_solutions_to_select, std::vector<solution_pt> & selected_solutions, rng_pt & rng);

  // Eigenvalue functions (for cma-es / rs-cmsa)
  void eigenDecompositionHouseholder2(int n, double **V, double *d, double *e);
  double myhypot(double a, double b);
  void eigenDecompositionQLalgo2(int n, double **V, double *d, double *e);
  
  void eigenDecomposition(matrix_t & mat, matrix_t & D, matrix_t & Q);
  void eigenDecomposition(double **matrix, int n, double **D, double **Q);
  
  void compute_ranks_desc(const vec_t & vec, vec_t & ranks);
  void compute_ranks_asc(const vec_t & vec, vec_t & ranks);
  void compute_ranks_asc(const vec_t & vec, std::vector<size_t> & ranks);
  
  // Tools.h
  //----------------------------------------------
  int *mergeSort( double *array, int array_size );
  void mergeSortWithinBounds( double *array, int *sorted, int *tosort, int p, int q );
  void mergeSortWithinBoundsInt( int *array, int *sorted, int *tosort, int p, int q );
  
  void mergeSortMerge( double *array, int *sorted, int *tosort, int p, int r, int q );
  int *mergeSortInt( int *array, int array_size );
  void mergeSortMergeInt( int *array, int *sorted, int *tosort, int p, int r, int q );
  
  int *getRanks(double *array, int array_size );
  int *getRanksFromSorted(int *sorted, int array_size );
  
  double randomRealUniform01( rng_t & rng );
  int randomInt( int maximum, rng_t & rng );
  int *randomPermutation( int n, rng_t & rng );
  int **allPermutations( int length, int *numberOfPermutations );
  int **allPermutationsSubroutine( int from, int length, int *numberOfPermutations );
  
  double max( double x, double y );
  double min( double x, double y );
  double distanceEuclidean( double *solution_a, double *solution_b, int n );
  double distanceEuclidean2D( double x1, double y1, double x2, double y2 );
  
  double *matrixVectorPartialMultiplication( double **matrix, double *vector, int n0, int n1, int number_of_elements, int *element_indices );
  
  
}
