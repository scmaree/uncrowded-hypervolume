#pragma once

/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

*/

#include "solution.hpp"

namespace hillvallea
{
  
  // a population, basically a list of individuals
  //-----------------------------------------
  class population_t{

  public:

    // constructor & destructor
    //------------------------------------------
    population_t();
    ~population_t();
    
    // essential data members
    //------------------------------------------
    std::vector<solution_pt > sols; // solutions
    
    // Dimension accessors
    //------------------------------------------
    size_t size() const; // population size
    size_t problem_size() const; // problem size

    // initialization
    //------------------------------------------
    void fill_uniform(const size_t sample_size, const size_t problem_size, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng);
    void fill_greedy_uniform(const size_t sample_size, const size_t problem_size, double sample_ratio, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng);
    void fill_with_rejection(const size_t sample_size, const size_t problem_size, double sample_ratio, const std::vector<solution_pt> & previous_sols, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng);
    int fill_normal(const size_t sample_size, const size_t problem_size, const vec_t & mean, const matrix_t & MatrixRoot, const vec_t & lower_param_range, const vec_t & upper_param_range, const size_t number_of_elites, rng_pt rng);
    int fill_normal_univariate(const size_t sample_size, const size_t problem_size, const vec_t & mean, const vec_t & cholesky, const vec_t & lower_param_range, const vec_t & upper_param_range, const size_t number_of_elites, rng_pt rng);

    // Sorting and ranking
    //------------------------------------------
    void sort_on_fitness();

    // Distribution parameter estimation
    // Maximum likelihood estimation of mean and covariance
    //-------------------------------------------
    void mean(vec_t & mean) const;
    void weighted_mean(vec_t & mean, const vec_t & weights) const;
    void weighted_transformed_mean(vec_t & mean, const vec_t & weights) const;
    void weighted_mean_of_selection(vec_t & mean, const vec_t & weights, const size_t selection_size) const;
    void covariance(const vec_t & mean, matrix_t & covariance) const;
    void covariance_univariate(const vec_t & mean, matrix_t & covariance) const;
    void covariance_univariate(const vec_t & mean, vec_t & covariance) const;
    
    double compute_DFC();
    void sort_on_probability();
    void set_fitness_rank();
    void set_probability_rank();
    
    // evaluate all solution in the population
    // returns the number of evaluations
    //------------------------------------------
    int evaluate(const fitness_pt fitness_function, const size_t skip_number_of_elites);
    int evaluate_with_gradients(const fitness_pt fitness_function, const size_t skip_number_of_elites);

    // Selection
    //------------------------------------------
    void truncation_percentage(population_t & selection, double selection_percentage) const;
    void truncation_size(population_t & selection, size_t selection_size) const;
    
    // Combine two populations
    //------------------------------------------
    void addSolutions(const population_t & pop);

    // For use in the SDR (AMaLGaM)
    //-------------------------------------------
    bool improvement_over(const double objective) const;

    // Population statistics
    //--------------------------------------------
    solution_pt first() const;
    solution_pt last() const;
    double average_fitness() const;
    double fitness_variance() const;
    double relative_fitness_std() const;

    double average_constraint() const;
    double relative_constraint_std() const;

  };
  


}
