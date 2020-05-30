#pragma once

/*

iAMaLGaM as part of HillVallEA

Implementation by S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA


*/

#include "optimizer.hpp"

namespace hillvallea
{

  class iamalgam_t : public optimizer_t
  {

  public: 

    // C++ Rule of Three
    //-------------------------------------------
    iamalgam_t(const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, fitness_pt fitness_function, rng_pt rng);
    ~iamalgam_t();
    optimizer_pt clone() const;

    // Essential data members
    //-------------------------------------------
    vec_t mean;                        // sample mean
    matrix_t aggregated_covariance;    //  aggregated over multiple generations
    matrix_t generational_covariance;  // estimated in this generation

    matrix_t covariance;              //  covariance used for sampling
    matrix_t cholesky;                // decomposed covariance matrix C = LL^T
    matrix_t inverse_cholesky;        // inverse of the cholesky decomposition

    vec_t ams_direction;

    double eta_p, eta_s; 

    // Transferrable parameters
    //-------------------------------------------
    double no_improvement_stretch;
    double multiplier;
    vec_t old_mean;

    // Stopping criteria
    //-------------------------------------------
    double st_dev_ratio_threshold;
    double distribution_multiplier_decrease;
    double sample_succes_ratio_threshold;  

    double delta_ams;
    bool apply_ams;

    // Run-time control
    //---------------------------------------------------------------------------------
    bool checkTerminationCondition();
    void estimate_sample_parameters();
    size_t sample_new_population(const size_t sample_size);
    void generation(size_t sample_size, int & number_of_evaluations);

    // Initialization
    //---------------------------------------------------------------------------------
    void initialize_from_population(population_pt pop, size_t target_popsize);
    size_t recommended_popsize(const size_t problem_dimension) const;

    // AMS & SDR
    //-------------------------------------------
    void apply_ams_to_population(const size_t number_of_ams_solutions, const double ams_factor, const vec_t & ams_direction);
    double getSDR(const solution_t & best, const vec_t & mean, matrix_t & cholesky_factor) const;
    void update_distribution_multiplier(double & multiplier, const bool improvement, double & no_improvement_stretch, const double & sample_success_ratio, const double sdr) const;
    
    // Debug info
    //---------------------------------------------------------------------------------
    std::string name() const;

  };

}
