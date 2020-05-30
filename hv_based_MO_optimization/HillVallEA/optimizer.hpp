#pragma once

/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

*/

#include "population.hpp"

namespace hillvallea
{
  
  class population_t;
  
  // abstract class (generation() is virtual)
  class optimizer_t {
    
  public:
    
    // constructor & destructor
    //--------------------------------------------------------------------------------
    optimizer_t(const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, fitness_pt fitness_function, rng_pt rng);
    ~optimizer_t();

    // Virtual Functions
    //---------------------------------------------------------------------------------
    virtual std::string name() const; 
    virtual size_t recommended_popsize(const size_t problem_dimension) const;
    virtual void initialize_from_population(population_pt pop, size_t target_popsize);
    virtual void generation(size_t sample_size, int & number_of_evaluations);
    virtual bool checkTerminationCondition();
    virtual void estimate_sample_parameters();
    virtual size_t sample_new_population(const size_t sample_size);

    // Data members
    //--------------------------------------------------------------------------------
    bool active;
    size_t number_of_parameters;
    vec_t  lower_param_bounds, upper_param_bounds;
    bool use_boundary_repair;
    fitness_pt fitness_function;
    int number_of_generations;
    std::shared_ptr<std::mt19937> rng;
    population_pt pop;
    solution_t best;
    vec_t average_fitness_history;
    double selection_fraction;
    double init_univariate_bandwidth; 
    int maximum_no_improvement_stretch;
    double param_std_tolerance;
    double fitness_std_tolerance;
    double penalty_std_tolerance;

  };

  // initialized optimizers of different types
  optimizer_pt init_optimizer(const int local_optimizer_index, const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, fitness_pt fitness_function, rng_pt rng);
}
