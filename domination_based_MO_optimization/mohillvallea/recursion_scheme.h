#pragma once

/*

HICAM Multi-objective

By S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "hicam_internal.h"
#include "fitness.h"

namespace hicam
{

  class recursion_scheme_t
  {

  public:

    recursion_scheme_t(
      fitness_pt fitness_function,
      const vec_t & lower_init_ranges,
      const vec_t & upper_init_ranges,
      double vtr,
      bool use_vtr,
      int version,
      int local_optimizer_index,
      double HL_tol,
      size_t elitist_archive_size_target,
      size_t approximation_set_size,
      size_t maximum_number_of_populations,
      int base_population_size,
      unsigned int number_of_subgenerations_per_population_factor,
      unsigned int maximum_number_of_evaluations,
      unsigned int maximum_number_of_seconds,
      int random_seed,
      bool write_generational_solutions,
      bool write_generational_statistics,
      bool print_generational_statistics,
      const std::string & write_directory,
      const std::string & file_appendix,
      bool print_verbose_overview
    );

    ~recursion_scheme_t();

    // run the thing
    void run();
    clock_t run_time;
    unsigned int number_of_evaluations;
    
    // Elitist Archive
    // a single elitist archive for all populations combined
    //------------------------------------------------------------------------
    elitist_archive_pt elitist_archive;
    elitist_archive_pt approximation_set;


  private:


    // input data members
    //------------------------------------------------------------------------
    fitness_pt fitness_function;
    vec_t lower_init_ranges;
    vec_t upper_init_ranges;
    double vtr;
    bool use_vtr;
    int version;
    int local_optimizer_index;
    double HL_tol;
    size_t elitist_archive_size_target;
    size_t approximation_set_size;
    size_t maximum_number_of_populations;
    unsigned int number_of_subgenerations_per_population_factor;
    unsigned int maximum_number_of_evaluations;
    unsigned int maximum_number_of_seconds;
    int random_seed;
    bool write_generational_solutions;
    bool write_generational_statistics;
    bool print_generational_statistics;
    std::string write_directory;
    std::string file_appendix;
    bool print_verbose_overvie;


    // essential data members
    //------------------------------------------------------------------------
    std::vector<optimizer_pt> populations;
    int base_number_of_mixing_components;
    int base_population_size;
    rng_pt rng;

    // size_t  number_of_parameters;
    // size_t number_of_objectives;
    vec_t  lower_param_bounds;
    vec_t  upper_param_bounds;

    // Stopping Criteria
    //------------------------------------------------------------------------
    clock_t start_time;
    unsigned int total_number_of_generations;
    double reached_vtr; // we save it when writing as it is expensive to evaluate
    unsigned int number_of_evaluations_hillvalley_clustering;
    unsigned int number_of_evaluations_hillvalley_archive;
    unsigned int number_of_evaluations_start;

    unsigned int number_of_solution_sets_written;
    bool print_verbose_overview;

    // Runtime control
    //------------------------------------------------------------------------
    void generationalStepAllPopulations();
    void generationalStepAllPopulationsRecursiveFold(size_t population_index_smallest, size_t population_index_biggest);
    
    void initialize();

    // terminating
    //------------------------------------------------------------------------
    bool checkTerminationConditionAllPopulations(const elitist_archive_t & approximation_set);
    bool checkTerminationConditionOnePopulation(optimizer_t & population);
    bool checkNumberOfEvaluationsTerminationCondition() const;
    bool checkVTRTerminationCondition(const elitist_archive_t & approximation_set);
    bool checkTimeLimitTerminationCondition();
    bool checkGlobalTerminationCondition(const elitist_archive_t & approximation_set);
    double getTimer() const;

    // writing
    //------------------------------------------------------------------------
    void writeGenerationalStatisticsForOnePopulation(const optimizer_t & population, size_t population_number, bool & use_vtr, double & reached_vtr, bool print_output, bool write_header) const;
    void writeGenerationalSolutions(elitist_archive_t & approximation_set, bool final, unsigned int & number_of_solution_sets_written);
    void printVerboseOverview() const;
    void writePopulation(size_t population_index, const optimizer_t & population) const;

  };

}

