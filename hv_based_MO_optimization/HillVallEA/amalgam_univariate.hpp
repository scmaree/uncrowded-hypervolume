#pragma once

/*

AMaLGaM-Univariate as part of HillVallEA

Implementation by S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA


*/

#include "optimizer.hpp"

namespace hillvallea
{

  class amalgam_univariate_t : public optimizer_t
  {

  public: 

    // C++ Rule of Three
    //-------------------------------------------
    amalgam_univariate_t(const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, fitness_pt fitness_function, rng_pt rng, bool enable_multiplier_vec);
    ~amalgam_univariate_t();

    optimizer_pt clone() const;

    // Essential data members
    //-------------------------------------------
    vec_t mean;                    // sample mean
    vec_t covariance;              // sample covariance matrix C
    vec_t cholesky;                // decomposed covariance matrix C = LL^T

    // Transferrable parameters
    //-------------------------------------------
    double no_improvement_stretch;
    double multiplier;
    vec_t multiplier_vec;
    vec_t old_mean;
    bool use_multiplier_vec;

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
    double getSDR(const solution_t & best, const vec_t & mean, vec_t & choleksy) const;
    vec_t getSDR_vec(const solution_t & best, const vec_t & mean, vec_t & choleksy) const;
    void update_distribution_multiplier(double & multiplier, const bool improvement, double & no_improvement_stretch, const double sample_success_ratio, const double sdr) const;
    void update_distribution_multiplier_vec(vec_t & multiplier, const bool improvement, double & no_improvement_stretch, const vec_t & sample_success_ratio, const vec_t & sdr) const;
    
    // Debug info
    //---------------------------------------------------------------------------------
    std::string name() const;

  };

}
