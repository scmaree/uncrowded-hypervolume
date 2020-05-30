/*

HillVallEA 

Real-valued Multi-Modal Evolutionary Optimization

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

Example script to demonstrate the usage of HillVallEA
 on the well-known 2D Six Hump Camel Back function

*/

// SO stuff
#include "HillVallEA/hillvallea.hpp"
#include "HillVallEA/fitness.h"
#include "HillVallEA/mathfunctions.hpp"

// for MO problems
#include "../domination_based_MO_optimization/mohillvallea/hicam_external.h"

namespace hillvallea
{
  
  class bezierUHV_t : public fitness_t
  {
  public:
    
    size_t bezier_degree;
    size_t number_of_test_points;
    hicam::fitness_pt mo_fitness_function;
    hicam::elitist_archive_pt elitist_archive;
    
    bool collect_all_mo_sols_in_archive;
    size_t elitist_archive_size;
    
    bezierUHV_t
    (
     hicam::fitness_pt mo_fitness_function,
     size_t bezier_degree,
     size_t number_of_test_points,
     bool collect_all_mo_sols_in_archive,
     size_t elitist_archive_size,
     hicam::elitist_archive_pt initial_archive
     );
    
    ~bezierUHV_t();
    
    // overloading
    void get_param_bounds(vec_t & lower, vec_t & upper) const;
    void init_solutions_randomly(population_t & population, size_t sample_size, const vec_t & lower_init_ranges, const vec_t & upper_init_ranges, size_t number_of_elites, rng_pt rng);
    void linkage_learning_distance_matrix(matrix_t & M);
    void dynamic_linkage_learning_distance_matrix(matrix_t & M, const population_t & pop);
    std::string write_solution_info_header(bool niching_enabled);
    std::string write_additional_solution_info(const solution_t & best,  const std::vector<solution_pt> & elitist_archive, bool niching_enabled);
    std::string name() const;
    void write_solution(const solution_t & sol, const std::string & filename);
    void sort_population_parameters(population_t & pop, FOS_t & FOS);
    bool vtr_reached(solution_t & sol, double vtr);

    double number_of_mo_evaluations;
    
    // assingisting fitness-computation
    void flip_line_direction(solution_t & sol);
    
    // creates mo_sols from the parameters of sol.
    void set_reference_points(solution_t & sol, size_t mo_number_of_parameters);
    void update_reference_points_partial(solution_t & sol, size_t mo_number_of_parameters, const std::vector<std::vector<size_t>> & mo_touched_parameter_idx);
    
    void set_and_evaluate_mo_solutions(solution_t & sol, size_t number_of_test_points, hicam::fitness_pt mo_fitness_function);
    void update_and_evaluate_mo_solutions(solution_t & sol, size_t number_of_test_points, hicam::fitness_pt mo_fitness_function, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol);
    
    void set_test_points(solution_t & sol, size_t number_of_test_points, size_t mo_number_of_parameters);
    void update_test_points_partial(solution_t & sol, size_t number_of_test_points, size_t mo_number_of_parameters, const std::vector<size_t> & mo_touched_parameter_idx_bezier);
    
    
    void get_front(const std::vector<hicam::solution_pt> & mo_sols, std::vector<bool> & part_of_front, vec_t & front_x, vec_t & front_y, double r_x, double r_y);
    void get_mo_touched_parameter_idx(std::vector<std::vector<size_t>> & mo_touched_parameter_idx, const std::vector<size_t> & touched_parameter_idx, size_t mo_number_of_parameters);
    void get_mo_touched_parameter_idx_bezier(std::vector<size_t> & mo_touched_parameter_idx_bezier, const std::vector<std::vector<size_t>> & mo_touched_parameter_idx, size_t mo_number_of_parameters);
    
    // fitness!
    void define_problem_evaluation(solution_t & sol);
    void define_partial_problem_evaluation(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol);
    void compute_fitness(bool partial_evaluation, solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol);
    
    
    double bezier_curve(double d, const vec_t & p, int p_start, int p_end);
    // set the test points based on the reference points of sol.
    
    double compute2DHyperVolume(const vec_t & obj0, const vec_t & obj1, std::vector<size_t> & sorted_obj, double max_0, double max_1);
    
    // distance to a box defined by [-infty, ref_x, -infty, ref_y]
    double distance_to_box(double ref_x, double ref_y, double p_x, double p_y);
    
    // as in the uncrowded hypervolume improvement by the Inria group
    double distance_to_front(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y);
    double distance_to_front_without_corner_boxes(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y);
    
  };
  
}
