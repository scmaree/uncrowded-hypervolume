#pragma once

/* GOMEA.HPP
 
 AMaLGaM as part of HillVallEA
 
 Implementation by S.C. Maree
 s.c.maree[at]amc.uva.nl
 github.com/SCMaree/HillVallEA
 
 
 */

#include "hillvallea_internal.hpp"
#include "optimizer.hpp"

namespace hillvallea
{
  
  class FOS_element_t
  {
    
  public:
    
    FOS_element_t() {
      distribution_multiplier = 1.0;
      samples_drawn_from_normal = 0;
      out_of_bounds_draws = 0;
      enable_regularization = false;
      sample_conditionally = false;
      sample_bayesian_factorization = false;
    }
    
    std::vector<size_t> params;
    bool enable_regularization;
    matrix_t cholesky_factor_lower_triangle;
    matrix_t covariance_matrix;
    int samples_drawn_from_normal;
    int out_of_bounds_draws;
    double distribution_multiplier;
    
    bool sample_conditionally;
    bool sample_bayesian_factorization;
    matrix_t mean_shift_matrix_to_condition_on;
    std::vector<size_t> params_to_condition_on;
    matrix_t bayesian_factorization_indicator_matrix;
    
    size_t length() const { return params.size(); }
    void copySettings(const FOS_element_t & other)
    {
      enable_regularization = other.enable_regularization;
      sample_conditionally = other.sample_conditionally;
      sample_bayesian_factorization = other.sample_bayesian_factorization;
      samples_drawn_from_normal = other.samples_drawn_from_normal;
      out_of_bounds_draws = other.out_of_bounds_draws;
      distribution_multiplier = other.distribution_multiplier;
    }
  };
  
  class FOS_t
  {
  public:
    std::vector<FOS_element_pt> sets;
    size_t length() const { return sets.size(); }
  };
  
  class gomea_t : public optimizer_t
  {
    
  public:
    
    // HillVallEA Framework functions
    //-------------------------------------------
    gomea_t(const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, int version, fitness_pt fitness_function, rng_pt rng);
    ~gomea_t();
    optimizer_pt clone() const;
    
    std::string name() const;
    size_t recommended_popsize(const size_t problem_dimension) const;
    void initialize_from_population(population_pt pop, size_t target_popsize);
    bool checkTerminationCondition();
    void generation(size_t sample_size, int & number_of_evaluations);
    
    // GOMEA Member Variables: Settings
    //--------------------------------------------
    int version;
    bool population_initialized;
    bool dynamic_filter_large_FOS_elements;
    bool always_keep_largest_FOS_element;
    bool sample_conditionally;
    bool sample_bayesian_factorization;
    bool gradient_step;
    int FOS_element_lb;                       /* Cut-off value for bounded fixed linkage tree (BFLT). */
    int FOS_element_ub;                       /* Cut-off value for bounded fixed linkage tree (BFLT). */
    bool learn_linkage_tree_from_distance_matrix;
    bool static_linkage;                  /* Whether the FOS is fixed throughout optimization. */
    bool random_linkage;                  /* Whether the fixed linkage tree is learned based on a random distance measure. */
    
    // GOMEA algorithm settings
    //--------------------------------------------
    double distribution_multiplier_increase;                    /* The multiplicative distribution multiplier increase. */
    double distribution_multiplier_decrease;                    /* The multiplicative distribution multiplier decrease. */
    double st_dev_ratio_threshold;                              /* The maximum ratio of the distance of the average improvement to the mean compared to the distance of one standard deviation before triggering AVS (SDR mechanism). */
    double eta_ams;
    double eta_cov;
    int no_improvement_stretch;                        /* The number of subsequent generations without an improvement while the distribution multiplier is <= 1.0, for each population separately. */
    int population_size;                              /* The size of the population. */
    size_t initial_population_size;

    // GOMEA stuff thats per FOS element
    //--------------------------------------------
    FOS_pt linkage_model;

    // Other data members
    vec_t mean_vector;
    vec_t old_mean_vector;
    vec_t mean_shift_vector;
    double weighted_number_of_evaluations; /* The current number of times a function evaluation was performed. */
    
    // GOMEA Member Functions
    //--------------------------------------------
    short checkFitnessVarianceTerminationSinglePopulation(  );
    short checkDistributionMultiplierTerminationCondition( );
    void evaluateCompletePopulation(  );
    void generateAndEvaluateNewSolutionsToFillPopulation( const solution_t & elite );
    void computeParametersForSampling( const population_t & selection);
    short generateNewSolutionFromFOSElement( FOS_element_pt FOS_element, int individual_index, short apply_AMS, bool force_accept_new_solutions );
    short applyAMS( solution_pt & sol, double & weighted_number_of_evaluations, solution_t & best ) const;
    void applyForcedImprovements(int individual_index, const solution_t & donor_sol );
    double *generateNewPartialSolutionFromFOSElement( FOS_element_pt FOS_element );
    double *generateNewConditionalPartialSolutionFromFOSElement( FOS_element_pt FOS_element, const solution_t & conditional_sol  );
    void adaptDistributionMultiplier( double & multiplier, bool improvement, double st_dev_ratio, size_t out_of_bounds_draws, size_t samples_drawn_from_normal  ) const;
    size_t generationalImprovementForFOSElement( FOS_element_pt FOS_element_pt, double *st_dev_ratio, const solution_t & elite ) const;
    double getStDevRatioForFOSElement( const vec_t & parameters, FOS_element_pt FOS_element_pt ) const;
    void ezilaitini( void );
    void ezilaitiniMemory( void );
    
    short isParameterInRangeBounds( double & parameter, int dimension ) const;
    
    // Gradient stuff
    //---------------------------------------------
    std::vector<adam_pt> gradient_methods;
    
  };
  
  // Assisting functions
  int *mergeSortFitness( double *objectives, double *constraints, int number_of_solutions );
  void mergeSortFitnessWithinBounds( double *objectives, double *constraints, int *sorted, int *tosort, int p, int q );
  void mergeSortFitnessMerge( double *objectives, double *constraints, int *sorted, int *tosort, int p, int r, int q );
  short betterFitness( double objective_value_x, double constraint_value_x, double objective_value_y, double constraint_value_y );
  
  void computeRanksForInitialPopulation(const population_t & pop, size_t initial_population_size, vec_t & ranks );
  void computeRanks(const population_t & pop, vec_t & ranks);
  
  void getBestInPopulation(const population_t & pop, int *individual_index );
  
  solution_pt copySolution(const solution_t & sol);
  solution_pt copySolution(const solution_pt sol);
  
  // Selection
  void makeSelection(const population_t & pop, double selection_fraction, population_t & selection, bool initial_population, int initial_popsize );
  void makeSelection(const population_t & pop, double selection_fraction, population_t & selection );
  void makeSelectionsUsingDiversityOnRank0(const population_t & pop, double selection_fraction, vec_t & ranks, population_t & selection );
  
  
  // parameter estimation
  double estimateCovariance(size_t vara, size_t varb, const std::vector<solution_pt> & sols, const vec_t & mean_vector, double init_univariate_bandwidth);
  void estimateFullCovarianceMatrixML( matrix_t & full_covariance_matrix, const population_t & selection, const vec_t & mean_vector, double init_univariate_bandwidth );
  
  
  void applyDistributionMultiplier(FOS_element_pt FOS_element);
  
  void estimateMeanVectorML_partial(vec_t & mean_vector, FOS_element_pt FOS_element, const population_t & selection );
  void estimateCovarianceMatrix( FOS_element_pt FOS_element, const population_t & selection, const vec_t & mean_vector, double init_univariate_bandwidth );
  
  void initializeCovarianceMatrices( FOS_pt linkage_model );
  
  
  void estimateMeanVectorML( const population_t & selection, vec_t & mean_vector, vec_t & old_mean_vector, vec_t & mean_shift_vector );
  void estimateMeanVectorML( const population_t & selection, vec_t & mean_vector);
  
  bool regularizeCovarianceMatrix(matrix_t & covariance, const std::vector<solution_pt> & sols, const std::vector<double> & mean, const std::vector<size_t> & parameters);
  
  // FOS.h
  double getSimilarity( size_t a, size_t b, int FOS_element_ub, size_t number_of_parameters, const std::vector<size_t> & mpm_number_of_indices, bool random_linkage_tree, const matrix_t & S_matrix, const vec_t & S_vector );
  void getSimilarityMatrix(const matrix_t & similarity_matrix, bool invert_input_matrix, matrix_t & S_matrix, int * index_order);
  size_t determineNearestNeighbour( size_t index, const matrix_t & S_matrix, const std::vector<size_t> & mpm_number_of_indices, int mpm_length, int FOS_element_ub, size_t number_of_parameters, bool random_linkage_tree, const vec_t & S_vector );
  void filterFOS( FOS_pt input_FOS, int lb, int ub, bool always_keep_largest_FOS_element );
  
  FOS_pt learnLinkageTree(
     const matrix_t & similarity_matrix,
     const bool invert_input_matrix, // the input matrix should be a similarty matrix, i.e.,
     const bool random_linkage_tree, // ignores the input matrix
     const size_t number_of_parameters,
     rng_pt rng,
     const int FOS_element_ub // for efficiency
  );
  
  FOS_pt updateLinkageTree(
     const matrix_t & input_matrix,
     FOS_pt old_FOS,
     const bool learn_from_distance_matrix,
     const bool random_linkage,
     const bool static_linkage,
     const int number_of_generations,
     int & FOS_element_lb,
     int & FOS_element_ub,
     bool always_keep_largest_FOS_element,
     const size_t number_of_parameters,
     bool & sample_bayesian_factorization,
     rng_pt rng
    );
  
  void inheritSettings( FOS_pt new_FOS, const FOS_pt prev_FOS );
  
  int *matchFOSElements( FOS_pt new_FOS, const FOS_pt prev_FOS );
  int *hungarianAlgorithm( int** similarity_matrix, int dim );
  void hungarianAlgorithmAddToTree(int x, int prevx, short *S, int *prev, int *slack, int *slackx, int* lx, int *ly, int** similarity_matrix, int dim);
  void computeMIMatrix( matrix_t & MI_matrix, const matrix_t & covariance_matrix );
  
  void printFOS( FOS_pt fos );
  
  
}
