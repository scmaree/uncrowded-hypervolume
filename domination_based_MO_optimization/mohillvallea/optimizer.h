#pragma once

/*

HICAM Multi-objective

By S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "hicam_internal.h"
#include "fitness.h"
#include "cluster.h"

namespace hicam
{

  class optimizer_t
  {


  public:

    optimizer_t(
      fitness_pt fitness_function,
      const vec_t & lower_init_ranges,
      const vec_t & upper_init_ranges,
      int local_optimizer_index,
      double HL_tol,
      size_t population_size,
      size_t number_of_mixing_components,
      int optimizer_number,
      unsigned int maximum_number_of_evaluations,
      unsigned int maximum_number_of_seconds,
      double vtr,
      bool use_vtr,
      rng_pt rng
    );

    ~optimizer_t();

    bool terminated;
    size_t cluster_size;
    int local_optimizer_index;
    int optimizer_number;
    double HL_tol;

    // run-time control
    void initialize(elitist_archive_t & elitist_archive, unsigned int & number_of_evaluations);
    void initialize_mm(elitist_archive_t & elitist_archive, unsigned int & number_of_evaluations);
    void generation(elitist_archive_t & elitist_archive, unsigned int & number_of_evaluations);
    void generation_mm(elitist_archive_t & elitist_archive, unsigned int & number_of_evaluations, bool largest_population);
    
    // termination
    //------------------------------------------------------------------------------
    bool checkTerminationCondition();

    // public data members
    //------------------------------------------------------
    population_pt population; 
    size_t population_size;
    size_t number_of_mixing_components;
    unsigned int number_of_generations;
    
    // for niching
    std::vector<population_pt> subpopulations;
    
    vec_t objective_ranges;
    size_t new_elites_added;
    
    bool global_selection;
    

    
  private:
    
    // data members : user settings
    //-------------------------------------------------------------------------------
    fitness_pt fitness_function;
    size_t  number_of_parameters;
    size_t number_of_objectives;
    vec_t  lower_init_ranges;
    vec_t  upper_init_ranges;
    vec_t  lower_param_bounds;
    vec_t  upper_param_bounds;

    unsigned int maximum_number_of_evaluations;
    unsigned int maximum_number_of_seconds;
    double vtr;
    bool use_vtr;

    double tau; // selection pressure
    size_t maximum_no_improvement_stretch;
    size_t no_improvement_stretch;
    unsigned int number_of_evaluations;
    double average_edge_length;

    rng_pt rng;

    void generateOffspring(population_t & subpopulation, size_t number_of_solutions);
    void linkSubpopulations(std::vector<population_pt> & subpopulations, std::vector<population_pt> & previous_subpopulations, const std::vector<vec_t> & previous_means) const;
    
    void generateAndEvaluateNewSolutionsToFillPopulation(population_t & population, size_t population_size, const std::vector<cluster_pt> & clusters, size_t & number_of_elites, unsigned int & number_of_evaluations, rng_pt & rng) const;
    void generateAndEvaluateNewSolutionsToFillPopulationNoElites(population_t & population, size_t number_of_solutions_to_generate, const std::vector<cluster_pt> & clusters, unsigned int & number_of_evaluations, rng_pt & rng) const;

    void updateStrategyParameters(const population_t & population, std::vector<cluster_pt> & clusters, const elitist_archive_t & previous_elitist_archive, size_t & no_improvement_stretch, const size_t maximum_no_improvement_stretch, const vec_t & objective_ranges) const;

    
    
    // Clustering
    //---------------------------------------------------------------------------------------------
    void mamalgam_cluster_registration(std::vector<cluster_pt> & clusters, const std::vector<cluster_pt> & previous_clusters, const vec_t & obj_ranges);
    void mamalgam_clustering(const population_t & pop, std::vector<cluster_pt> & clusters, size_t number_of_clusters, size_t cluster_size, size_t selection_size, rng_pt & rng) const;
    void mamalgam_bugfix_clustering(const population_t & pop, size_t selection_size, std::vector<cluster_pt> & clusters, size_t number_of_clusters, size_t cluster_size, double average_edge_length, rng_pt & rng) const;
    void mamalgam_MO_clustering(const population_t & pop, std::vector<cluster_pt> & clusters, size_t number_of_clusters, size_t cluster_size, size_t selection_size, rng_pt & rng) const;
    void mamalgam_SO_clustering(const population_t & pop, std::vector<cluster_pt> & clusters, size_t cluster_size) const;
    
    void direct_cluster_registration_on_mean(std::vector<cluster_pt> & clusters, const std::vector<cluster_pt> & previous_clusters, const vec_t & obj_ranges) const;
  
    void removeElitesFromSubpopulations(size_t max_elites_per_subpop, std::vector<population_pt> & subpopulations, rng_pt & rng) const;
    
  };



}
