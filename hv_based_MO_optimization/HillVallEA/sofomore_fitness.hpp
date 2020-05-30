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
  
  class sofomore_fitness_t : public fitness_t
  {
  public:
    
    hicam::fitness_pt mo_fitness_function;
    std::vector<hicam::solution_pt> * mo_population;
    
    sofomore_fitness_t(hicam::fitness_pt mo_fitness_function, std::vector<hicam::solution_pt> * mo_population, bool collect_all_mo_sols_in_archive, hicam::elitist_archive_pt elitist_archive);
    ~sofomore_fitness_t();
    
    // overloading
    void get_param_bounds(vec_t & lower, vec_t & upper) const;
    void init_solutions_randomly(population_t & population, size_t sample_size, const vec_t & lower_init_ranges, const vec_t & upper_init_ranges, size_t number_of_elites, rng_pt rng);
    std::string name() const;
    
    void get_front(const std::vector<hicam::solution_pt> & mo_sols, std::vector<bool> & part_of_front, vec_t & front_x, vec_t & front_y, double r_x, double r_y);
    void sort_population_parameters(population_t & pop, FOS_t & FOS);
    
    // as in the uncrowded hypervolume improvement by the Inria group
    double distance_to_box(double ref_x, double ref_y, double p_x, double p_y);
    double distance_to_front(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y);
    double compute2DHyperVolume(const vec_t & obj0, const vec_t & obj1, std::vector<size_t> & sorted_obj, double max_0, double max_1);
    
    // fitness!
    void define_problem_evaluation(solution_t & sol);

    
    // MO-solution logging
    hicam::elitist_archive_pt elitist_archive;
    bool collect_all_mo_sols_in_archive;
    
  };
  
  
}
