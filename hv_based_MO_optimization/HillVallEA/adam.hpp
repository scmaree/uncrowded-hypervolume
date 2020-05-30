#pragma once

/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

*/

#include "hillvallea_internal.hpp"
#include "optimizer.hpp"
#include "fitness.h"

namespace hillvallea
{
  
  class adam_t
  {
    
    
  public:
    
    adam_t
    (
     fitness_pt fitness_function,
     int version,
     const vec_t & lower_init_ranges,
     const vec_t & upper_init_ranges,
     int maximum_number_of_evaluations,
     int maximum_number_of_seconds,
     double vtr,
     int use_vtr,
     int random_seed,
     bool write_generational_solutions,
     bool write_generational_statistics,
     const std::string & write_directory,
     const std::string & file_appendix,
     double gamma_weight,
     double finite_differences_multiplier
     );

    ~adam_t();
    
    fitness_pt fitness_function;
    
    // ADAM settings
    int version;
    bool use_momentum_with_nag;
    int maximum_no_improvement_stretch;
    int no_improvement_stretch;
    bool use_boundary_repair;
    bool accept_only_improvements;
    double gamma;
    double b1;
    double b2;
    double epsilon;
    double finite_differences_multiplier;
    
    // Run it!
    //--------------------------------------------------------------------------------
    // Runs minimizer.
    void run();
    double gradientOffspring(solution_pt & sol, const std::vector<std::vector<size_t>> & touched_parameter_idx, vec_t & gamma);
    double HIGAMOgradientOffspring(solution_pt & sol, const std::vector<std::vector<size_t>> & touched_parameter_idx, vec_t & gamma);
    
    bool isParameterInRangeBounds( double & parameter, size_t dimension ) const;
    solution_t best;
    solution_t sol_current;
    
    double evaluateWithfiniteDifferences(solution_pt & sol, double h, bool use_central_difference) const;
    double partialEvaluateWithfiniteDifferences(solution_pt & sol, double h, bool use_central_difference, const std::vector<size_t> & touched_parameter_idx, const solution_pt & old_sol) const;
    
    // data members : optimization results
    //--------------------------------------------------------------------------------
    bool terminated;
    bool success;
    int number_of_evaluations;
    double weighted_number_of_evaluations;
    int number_of_generations;
    clock_t starting_time;
    
    // Random number generator
    // Mersenne twister
    //------------------------------------
    std::shared_ptr<std::mt19937> rng;
    
    // data members : user settings
    //-------------------------------------------------------------------------------
    vec_t lower_init_ranges;
    vec_t upper_init_ranges;
    vec_t lower_param_bounds;
    vec_t upper_param_bounds;
    int maximum_number_of_evaluations;
    int maximum_number_of_seconds;
    double vtr;
    int use_vtr;
    int random_seed; 
    bool write_generational_solutions;
    bool write_generational_statistics;
    std::string write_directory;
    std::string file_appendix;
    std::string file_appendix_generation;
    
    // Termination criteria
    //-------------------------------------------------------------------------------
    bool terminate_on_runtime() const;
    bool checkTerminationCondition(double old_fitness, solution_t & sol, int & no_improvement_stretch);

    
    // Output to file
    //-------------------------------------------------------------------------------
    std::ofstream statistics_file;
    void new_statistics_file();
    void close_statistics_file();
    void write_statistics_line(const solution_t & sol, size_t number_of_generations, const solution_t & best);
  };
  
  
  
}
