#pragma once

/*

CMA-ES

Implementation by S.C. Maree, 2017
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "optimizer.hpp"

namespace hillvallea
{

  class sep_cmaes_t : public optimizer_t
  {

  public:
    
    sep_cmaes_t(const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, fitness_pt fitness_function, rng_pt rng);
    ~sep_cmaes_t();
    optimizer_pt clone() const;

    // CMA-ES Parameters
    //---------------------------------------------------------------------------------
    vec_t weights;              // weight vector for weighted means
    size_t lambda;              // Population size
    size_t mu;                  // Selection size
    double mueff;               // CMA-ES mueff.
    double cc;                  // CMA-ES cc.
    double ccov;
    double mucov;               // CMA-ES mucov.
    double cs;                  // CMA-ES cs.
    double damps;               // CMA-ES damping factor.
    double sigma;               // sigma.
    double chiN;                // Chi_N
    vec_t pc;                   // pc
    vec_t ps;                   // ps
   
    
    vec_t mean;
    vec_t mean_z;
    vec_t old_mean;             // old sample mean
    vec_t covariance;
    vec_t fitnesshist;

    bool flginiphase;
    // double number_of_generations; // a double as we average it a lot; // counts the number of generations before a covariance update 
    double no_improvement_stretch; // a double as we average it a lot

    int minimum_cluster_size;
    
    // Initialization
    //---------------------------------------------------------------------------------
    virtual void initialize_from_population(population_pt pop, size_t target_popsize);
    virtual size_t recommended_popsize(const size_t problem_dimension) const;
    void initStrategyParameters(const size_t selection_size);

    // Run-time control
    //---------------------------------------------------------------------------------
    bool checkTerminationCondition();
    void estimate_sample_parameters();
    void estimate_covariance_univariate(vec_t & pc, vec_t & ps, vec_t & fitnesshist, double & sigma, vec_t & covariance, bool & flginiphase) const;
    size_t sample_new_population(const size_t sample_size);
    void generation(size_t sample_size, int & number_of_evaluations);

    // debug info
    //---------------------------------------------------------------------------------
    std::string name() const;



  };

}
