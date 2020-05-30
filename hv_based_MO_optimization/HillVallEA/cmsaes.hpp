#pragma once

/*

CMA-ES

Implementation by S.C. Maree, 2017
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "hillvallea_internal.hpp"
#include "optimizer.hpp"

namespace hillvallea
{

  class cmsaes_t : public optimizer_t
  {

  public:
    
    cmsaes_t(const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, fitness_pt fitness_function, rng_pt rng);
    ~cmsaes_t();
    optimizer_pt clone() const;

    // CMA-ES Parameters
    //---------------------------------------------------------------------------------
    size_t lambda;              // Population size
    size_t mu;                  // Selection size
    double sigma;               // sigma / multiplier.
    double tau, tau_c;
    
    int minimum_cluster_size;
    vec_t bestf_NE;
    
    vec_t weights;
    vec_t mean;         
    matrix_t covariance;
    matrix_t cholesky;                // product matrix cholesky such that C = cholesky * (cholesky)^(-T)

    double no_improvement_stretch; // a double as we average it a lot

    // Initialization
    //---------------------------------------------------------------------------------
    virtual void initialize_from_population(population_pt pop, size_t target_popsize);
    virtual size_t recommended_popsize(const size_t problem_dimension) const;

    // Run-time control
    //---------------------------------------------------------------------------------
    bool checkTerminationCondition();
    void estimate_sample_parameters();
    void estimate_covariance(matrix_t & covariance, matrix_t & BD) const;
    size_t sample_new_population(const size_t sample_size);
    void initStrategyParameters(const size_t selection_size);
    void generation(size_t sample_size, int & number_of_evaluations);

    // Parameter transfer 
    //---------------------------------------------------------------------------------
    const double getParamDouble(const std::string & param_name) const;
    const vec_t getParamVec(const std::string & param_name) const;
    const matrix_t getParamMatrix(const std::string & param_name) const;

    // debug info
    //---------------------------------------------------------------------------------
    std::string name() const;



  };

}
