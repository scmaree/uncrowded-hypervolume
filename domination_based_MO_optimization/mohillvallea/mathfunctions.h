#pragma once

/*
 
 HICAM Multi-objective
 
 By S.C. Maree, 2018
 s.c.maree[at]amc.uva.nl
 smaree.com
 
 */

#include "hicam_internal.h"
#include "param.h"

namespace hicam
{
  
  class population_t;
  
  double round(double x);

  // cholesky decomposition
  void principalCholeskyDecomposition(const matrix_t & covariance, matrix_t & cholesky, std::vector<int> & permutation_order);
  void choleskyDecomposition(const matrix_t & covariance, matrix_t & cholesky);
  void choleskyDecomposition_univariate(const vec_t & univariate_covariance, vec_t & univariate_cholesky);

  // Normal pdf
  // returns the value of the pdf at a given point.
  // assumes all input is of the same dimension as mean.
  //---------------------------------------------------------------------------
  double normpdf(const vec_t & mean, const matrix_t & cov, const vec_t & x, matrix_t & chol, matrix_t & inverse_chol);
  double normpdf(const vec_t & mean, const matrix_t & chol, const matrix_t & inverse_chol, const vec_t & x);
  double normpdf_diagonal(const vec_t & mean, const vec_t & cov_diagonal, const vec_t & x);           // uses diag(cov) only
  double normpdf(const vec_t & mean, const matrix_t & inverse_chol, const double normal_factor, const vec_t & x, const bool univariate);

  // one-dimensional cdf of the standard normal distribution.
  double normcdf(const double x);

  // sample parameters using normal distribution
  // returns the number of trials before an in-range sample is found.
  // on fail, sample uniform.
  //----------------------------------------------
  unsigned int sample_normal(vec_t & sample, vec_t & sample_transformed, const size_t problem_size, const vec_t & mean, const matrix_t & MatrixRoot, const double multiplier, bool use_boundary_repair, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng);
  unsigned int sample_normal(vec_t & sample, const size_t problem_size, const vec_t & mean, const matrix_t & chol, bool use_boundary_repair, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng);
  unsigned int sample_normal_univariate(vec_t & sample, const size_t problem_size, const vec_t & mean, const vec_t & univariate_cholesky, bool use_boundary_repair, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng);

  void sample_uniform(vec_t & sample, const size_t problem_size, const vec_t & lower_user_range, const vec_t & upper_user_range,  rng_pt rng);
  
  // check if the parameter is in range
  //-----------------------------------------
  bool in_range(const vec_t & sample, const vec_t & lower_param_range, const vec_t & upper_param_range);
  bool boundary_repair(vec_t & sample, const vec_t & lower_param_range, const vec_t & upper_param_range);

  // BLAS / LINPACK library functions
  //-----------------------------------------
  void *Malloc(long size);
  double **matrixNew(int n, int m);
  double vectorDotProduct(const double *vector0, const double *vector1, int n0);
  double *matrixVectorMultiplication(const double **matrix, const double *vector, int n0, int n1);
  double **matrixMatrixMultiplication(const double **matrix0, const double **matrix1, int n0, int n1, int n2);
  int blasDSWAP(int n, double *dx, int incx, double *dy, int incy);
  int blasDAXPY(int n, double da, double *dx, int incx, double *dy, int incy);
  void blasDSCAL(int n, double sa, double x[], int incx);
  int linpackDCHDC(double a[], int lda, int p, double work[], int ipvt[]);
  double **choleskyDecomposition(double **matrix, int n);
  double **principalCholeskyDecomposition(double **matrix, int n, std::vector<int> & permutation_order);
  int linpackDTRDI(double t[], int ldt, int n);
  double **matrixLowerTriangularInverse(double **matrix, int n);
  int *mergeSort(double *array, int array_size);
  void mergeSortWithinBounds(double *array, int *sorted, int *tosort, int p, int q);
  void mergeSortMerge(double *array, int *sorted, int *tosort, int p, int r, int q);
  int *mergeSortInt(int *array, int array_size);
  void mergeSortWithinBoundsInt(int *array, int *sorted, int *tosort, int p, int q);
  void mergeSortMergeInt(int *array, int *sorted, int *tosort, int p, int r, int q);
  int **allPermutations(int length, int *numberOfPermutations);
  int **allPermutationsSubroutine(int from, int length, int *numberOfPermutations);

  // Eigenvalue functions (for cma-es)
  void eigenDecompositionHouseholder2(int n, double **V, double *d, double *e);
  double myhypot(double a, double b);
  void eigenDecompositionQLalgo2(int n, double **V, double *d, double *e);
  
  void eigenDecomposition(matrix_t & mat, matrix_t & D, matrix_t & Q);
  void eigenDecomposition(double **matrix, int n, double **D, double **Q);

  vec_t max(const vec_t & v1, const vec_t & v2);
  vec_t min(const vec_t & v1, const vec_t & v2);

  // computes the rank correlation O(N)
  double rank_correlation(const vec_t & v1, const vec_t & v2);
  double correlation(const vec_t & v1, const vec_t & v2);

  void compute_ranks_desc(const vec_t & vec, vec_t & ranks);
  void compute_ranks_asc(const vec_t & vec, vec_t & ranks);
  void compute_ranks_asc(const vec_t & vec, std::vector<size_t> & ranks);
  void compute_ranks_asc(double * vec, size_t vec_size, std::vector<size_t> & ranks);

  /**
  * Selects n points from a set of points. A greedy heuristic is used to find a good
  * scattering of the selected points. First, a point is selected with a maximum value
  * in a randomly selected dimension. The remaining points are selected iteratively.
  * In each iteration, the point selected is the one that maximizes the minimal distance
  * to the points selected so far.
  */
  void greedyScatteredSubsetSelection(std::vector<vec_t> & points, size_t number_to_select, std::vector<size_t> & selected_indices, rng_pt & rng);
  
  void selectSolutionsBasedOnObjectiveDiversity(const std::vector<solution_pt> & solutions, size_t number_of_solutions_to_select, std::vector<solution_pt> & selected_solutions, const vec_t & obj_ranges, std::vector<solution_pt> & non_selected_solutions, rng_pt & rng);
  void selectSolutionsBasedOnObjectiveDiversity(const std::vector<solution_pt> & solutions, size_t number_of_solutions_to_select, std::vector<solution_pt> & selected_solutions, const vec_t & obj_ranges, rng_pt & rng);

  
  void selectSolutionsBasedOnParameterDiversity(const std::vector<solution_pt> & solutions, size_t number_of_solutions_to_select, std::vector<solution_pt> & selected_solutions, std::vector<solution_pt> & non_selected_solutions, rng_pt & rng);
  void selectSolutionsBasedOnParameterDiversity(const std::vector<solution_pt> & solutions, size_t number_of_solutions_to_select, std::vector<solution_pt> & selected_solutions, rng_pt & rng);

  // mamalgam's original clustering is way too messy to re-implement..
  //------------------------------
  double distanceEuclidean(double *x, double *y, int number_of_dimensions);
  double distanceAbsolute(double *x, double *y, int number_of_dimensions);
  
  // Simon Wessing's Maximin reconstruction
  void maximin_reconstruction(const size_t number_of_samples, const size_t dimension, std::vector<std::shared_ptr<vec_t>> & samples, const std::vector<std::shared_ptr<vec_t>> & existing_samples, rng_pt rng);
  void stratified_sampling(const size_t number_of_samples, const size_t dimension, std::vector<std::shared_ptr<vec_t>> & samples, rng_pt rng);
  void dist_matrix_function(const vec_t & sample, const std::vector<std::shared_ptr<vec_t>> & ref_samples, vec_t & distances);
  void dist_matrix_function(const vec_t & sample, const vec_t & ref_sample, double & distance);
  
}
