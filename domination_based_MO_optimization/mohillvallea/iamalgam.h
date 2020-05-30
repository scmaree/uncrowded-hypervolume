#pragma once

/*

AMaLGaM

Implementation by S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "hicam_internal.h"
#include "cluster.h"

namespace hicam
{

  class iamalgam_t : public cluster_t
  {

  public: 

    // C++ Rule of Three
    //-------------------------------------------
    iamalgam_t(fitness_pt fitness_function, rng_pt rng, bool use_univariate);
    iamalgam_t(const iamalgam_t & other);
    ~iamalgam_t();

    // Run-time control
    //---------------------------------------------------------------------------------
    bool checkTerminationCondition();
    void estimateParameters();
    std::string name() const;

    bool updateStrategyParameters(const elitist_archive_t & elitist_archive, size_t & no_improvement_stretch, const size_t maximum_no_improvement_stretch);
    void computeParametersForSampling();
    void generateNewSolutions(std::vector<solution_pt> & solutions, size_t number_of_solutions, size_t number_of_ams_solutions, rng_pt & rng);
    size_t recommended_popsize(size_t number_of_parameters) const;

    vec_t getVec(std::string variable_name);
    double getDouble(std::string variable_name);
    matrix_t getMatrix(std::string variable_name);

  private:


    // Essential data members
    //-------------------------------------------
    vec_t mean;                       // sample mean
    double multiplier;                // distribution multiplier
    
    // full matrices
    matrix_t covariance;              // sample covariance matrix C
    matrix_t cholesky;                // decomposed covariance matrix C = LL^T
    matrix_t inverse_cholesky;        // inverse of the cholesky decomposition

    bool use_boundary_repair;
    bool use_univariate;

    matrix_t aggregated_covariance;    //  aggregated over multiple generations
    matrix_t generational_covariance;  // estimated in this generation
    
    double eta_p;
    double eta_s;
    
    vec_t ams_direction;
    
    // Stopping criteria
    //-------------------------------------------
    bool terminated;
    bool improvement;
    double out_of_bounds_sample_ratio;

    //// AMS & SDR
    ////-------------------------------------------
    void apply_ams(std::vector<solution_pt> & solutions, const size_t number_of_elites, const size_t number_of_ams_solutions, const double ams_factor, const vec_t & ams_direction, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds) const;
    double getSDR(const vec_t & params) const;
    
    
    bool adaptDistributionMultiplier(const elitist_archive_t & elitist_archive, size_t & no_improvement_stretch, const size_t maximum_no_improvement_stretch);
    bool generationalImprovementForOneCluster(double & st_dev_ratio, const elitist_archive_t & elitist_archive) const;
  };

}
