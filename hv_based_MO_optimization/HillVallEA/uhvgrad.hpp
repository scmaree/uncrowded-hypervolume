#pragma once

/*
 
 HillVallEA
 
 By S.C. Maree
 s.c.maree[at]amc.uva.nl
 github.com/SCMaree/HillVallEA
 
 */

#include "hillvallea_internal.hpp"
#include "optimizer.hpp"
#include "../../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hillvallea
{
  
  class uhvgrad_t
  {
    
    
  public:
    
    uhvgrad_t
    (
     hicam::fitness_pt mo_fitness_function,
     size_t number_of_mo_solutions,
     const hicam::vec_t & lower_init_ranges,
     const hicam::vec_t & upper_init_ranges,
     bool collect_all_mo_sols_in_archive,
     size_t elitist_archive_size_target,
     double gamma_weight,
     bool use_finite_differences,
     double finite_differences_multiplier,
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
    
    ~uhvgrad_t();
    
    // data members : user settings
    //-------------------------------------------------------------------------------
    hicam::fitness_pt mo_fitness_function;
    size_t number_of_mo_solutions;
    hicam::vec_t lower_init_ranges;
    hicam::vec_t upper_init_ranges;
    bool collect_all_mo_sols_in_archive;
    size_t elitist_archive_size_target;
    double gamma_weight;
    bool use_finite_differences;
    double finite_differences_multiplier;
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
    
    // other settings
    hicam::vec_t lower_param_bounds;
    hicam::vec_t upper_param_bounds;
    int maximum_no_improvement_stretch;
    int no_improvement_stretch;
    bool use_boundary_repair;
    double gamma;
    double b1;
    double b2;
    double epsilon;
    std::vector<vec_t> gradient_weights;
    std::vector<vec_t> mt;
    std::vector<vec_t> vt;
    
    // Run it!
    //--------------------------------------------------------------------------------
    // Runs minimizer.
    void run();
    double gradientOffspring(hicam::population_pt & mo_population, double & current_hypervolume, hicam::population_pt & best_mo_population, double & best_hypervolume, double & gamma);
    bool isParameterInRangeBounds( double & parameter, size_t dimension ) const;
    double evaluateWithfiniteDifferences(hicam::solution_pt & sol, double h) const;
    
    // data members : optimization results
    //--------------------------------------------------------------------------------
    bool terminated;
    bool success;
    int number_of_evaluations;
    int number_of_generations;
    clock_t starting_time;
    double current_hypervolume;
    double best_hypervolume;
    hicam::population_pt mo_population;
    hicam::population_pt best_mo_population;
    hicam::elitist_archive_pt elitist_archive;
    
    
    // Random number generator
    // Mersenne twister
    //------------------------------------
    std::shared_ptr<std::mt19937> rng;
    
    // Termination criteria
    //-------------------------------------------------------------------------------
    bool terminate_on_runtime() const;
    bool checkTerminationCondition(const hicam::population_pt & mo_population, double current_hypervolume, int no_improvement_stretch);
    
    // Output to file
    //-------------------------------------------------------------------------------
    std::ofstream statistics_file;
    void new_statistics_file();
    void close_statistics_file();
    void write_statistics_line(const hicam::population_pt & mo_population, double current_hypervolume, size_t number_of_generations, const hicam::population_pt & best_mo_population, double best_hypervolume);
  
    // UHV related stuff
    // Adapted from the uncrowded hypervolume improvement by the Inria group
    //-----------------------------------
    double uhv(const hicam::population_pt & mo_population) const;
    double uhv(const hicam::population_pt & mo_population, bool compute_gradient_weights, std::vector<vec_t> & gradient_weights) const;
    double compute2DHyperVolume(const vec_t & obj0, const vec_t & obj1, std::vector<size_t> & sorted_obj, double max_0, double max_1) const;
    void get_front(const std::vector<hicam::solution_pt> & mo_sols, std::vector<bool> & part_of_front, vec_t & front_x, vec_t & front_y, double r_x, double r_y) const;
    double distance_to_box(double ref_x, double ref_y, double p_x, double p_y);
    double distance_to_box(double ref_x, double ref_y, double p_x, double p_y, double & nearest_x, double & nearest_y, bool & shares_x, bool & shares_y) const;
    double distance_to_front_without_corner_boxes(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y, vec_t & nearest_point_on_front, size_t & nearest_x_idx, size_t & nearest_y_idx) const;
    double distance_to_front(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y, vec_t & nearest_point_on_front, size_t & nearest_x_idx, size_t & nearest_y_idx) const;
    double distance_to_front(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y) const;
    
  };
  
  
  
}
