#pragma once

/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

*/

#include "sofomore_fitness.hpp"
#include "hillvallea_internal.hpp"
#include "optimizer.hpp"
#include "fitness.h"

namespace hillvallea
{
  
  class sofomore_t
  {
    
    
  public:
    
    sofomore_t
    (
     hicam::fitness_pt mo_fitness_function,
     size_t number_of_mo_solutions,
     size_t number_of_so_solutions_per_shadow_population,
     int local_optimizer_index,
     bool collect_all_mo_sols_in_archive,
     size_t elitist_archive_size_target,
     size_t approximation_set_size,
     const hicam::vec_t & lower_init_ranges,
     const hicam::vec_t & upper_init_ranges,
     int maximum_number_of_evaluations,
     int maximum_number_of_seconds,
     double vtr,
     int use_vtr,
     int random_seed,
     bool write_generational_solutions,
     bool write_generational_statistics,
     const std::string & write_directory,
     const std::string & file_appendix
     );

    ~sofomore_t();
    
    fitness_pt fitness_function;
    
    int local_optimizer_index;

    // Run it!
    //--------------------------------------------------------------------------------
    // Runs minimizer.
    void run();

    // data members : optimization results
    //--------------------------------------------------------------------------------
    bool terminated;
    bool success;
    int number_of_evaluations;
    int number_of_generations;
    clock_t starting_time;
    
    // Random number generator
    // Mersenne twister
    //------------------------------------
    std::shared_ptr<std::mt19937> rng;

    // MO stuff
    //-------------------------------------------------------------------------------
    hicam::fitness_pt mo_fitness_function;
    size_t number_of_mo_solutions;
    hicam::population_pt mo_population;
    
    bool collect_all_mo_sols_in_archive;
    size_t elitist_archive_size_target;
    size_t approximation_set_size;
    hicam::elitist_archive_pt elitist_archive;
    
    // data members : user settings
    //-------------------------------------------------------------------------------
    size_t number_of_so_solutions_per_shadow_population;
    hicam::vec_t lower_init_ranges;
    hicam::vec_t upper_init_ranges;
    hicam::vec_t lower_param_bounds;
    hicam::vec_t upper_param_bounds;
    hillvallea::vec_t so_lower_param_bounds;
    hillvallea::vec_t so_upper_param_bounds;
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
    
    // Output to file
    //-------------------------------------------------------------------------------
    std::ofstream statistics_file;
    void new_statistics_file();
    void close_statistics_file();
    void write_statistics_line(size_t number_of_generations, const hicam::population_pt & mo_population, const hicam::elitist_archive_pt & elitist_archive);
  };
  
  
  
}
