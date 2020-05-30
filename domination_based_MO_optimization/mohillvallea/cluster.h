#pragma once

/*
 
 General framework of EDA's such as AMaLGaM and CMA-ES
 
 By S.C. Maree, 2018
 s.c.maree[at]amc.uva.nl
 smaree.com
 
 */

#include "hicam_internal.h"
#include "fitness.h"
#include "population.h"


namespace hicam
{
  
  
  // abstract class 
  class cluster_t : public population_t
  {
    
  public:
    
    // constructor & destructor
    //--------------------------------------------------------------------------------
    cluster_t(fitness_pt fitness_function, rng_pt rng);
    cluster_t(const cluster_t & other);
    ~cluster_t();

    fitness_pt fitness_function;
    rng_pt rng;

    vec_t parameter_mean;
    vec_t objective_mean; 
    cluster_pt previous;
    double init_bandwidth;
    
    size_t objective_number; // to remember which clusters are SO clusters
    
    virtual bool checkTerminationCondition();
    virtual void estimateParameters();
    virtual bool updateStrategyParameters(const elitist_archive_t & elitist_archive, size_t & no_improvement_stretch, const size_t maximum_no_improvement_stretch);

    virtual void computeParametersForSampling();
    virtual void generateNewSolutions(std::vector<solution_pt> & solutions, size_t number_of_solutions, size_t number_of_ams_solutions, rng_pt & rng);
    virtual size_t recommended_popsize(size_t number_of_parameters) const;

    virtual vec_t getVec(std::string variable_name);
    virtual double getDouble(std::string variable_name);
    virtual matrix_t getMatrix(std::string variable_name);
    
    virtual std::string name() const;

  };

  // initialized optimizers of different types
  cluster_pt init_cluster(const int cluster_index, fitness_pt fitness_function, rng_pt rng);
}
