#pragma once

/*

hillvallea Multi-objective

By S.C. Maree, 2016
s.c.maree[at]amc.uva.nl
smaree.com

*/


#include <functional>
#include "hillvallea_internal.hpp"
#include "population.hpp"


// Defines the fitness function of our choice
// implementation idea from libcmaes by Emmanuel Benazera.
//------------------------------------------------

namespace hillvallea
{

  class FOS_t;
  
  class fitness_t
  {

  public:

    fitness_t();
    ~fitness_t();

    size_t number_of_parameters;
    double number_of_evaluations;
    size_t covariance_block_size;
    size_t maximum_number_of_evaluations;

    size_t get_number_of_parameters() const;
    
    bool dynamic_objective; // re-evaluate best every generation etc.
    bool has_round_off_errors_in_partial_evaluations;
    bool partial_evaluations_available;
    bool linkage_learning_distance_matrix_available;
    bool dynamic_linkage_learning_distance_matrix_available;
    size_t fos_element_size_lower_bound;
    size_t fos_element_size_upper_bound;
    bool use_boundary_repair;
    double local_optima_tolerance;

  
  
    // evaluates the function
    // for new functions, define problem_evaluation in "define_problem_evaluation".
    // evaluate covers the evaluation itself and can be set to cover other stuff
    // such as counting the number of evaluations or printing
    void evaluate(solution_t & sol);
    void evaluate(solution_pt & sol);
    void evaluate_with_gradients(solution_t & sol);
    void evaluate_with_gradients(solution_pt & sol);
    void evaluate_with_finite_differences(solution_t & sol, double step_size);
    void evaluate_with_finite_differences(solution_pt & sol, double step_size);
    
    void partial_evaluate(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol);
    void partial_evaluate(solution_pt & sol, const std::vector<size_t> & touched_parameter_idx, const solution_pt & old_sol);
    void partial_evaluate_with_gradients(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol);
    void partial_evaluate_with_gradients(solution_pt & sol, const std::vector<size_t> & touched_parameter_idx, const solution_pt & old_sol);
    void partial_evaluate_with_finite_differences(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol, double step_size);
    void partial_evaluate_with_finite_differences(solution_pt & sol, const std::vector<size_t> & touched_parameter_idx, const solution_pt & old_sol, double step_size);

    virtual void sort_population_parameters(population_t & pop, FOS_t & FOS);
    
    // Placeholders for user-defined objective functions
    //----------------------------------------------------------------------------------------
    virtual void set_number_of_parameters(size_t & number_of_parameters);
    virtual void get_param_bounds(vec_t & lower, vec_t & upper) const;
    
    virtual void define_problem_evaluation(solution_t & sol);
    virtual void define_partial_problem_evaluation(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol);

    virtual void define_problem_evaluation_with_gradients(solution_t & sol);
    virtual void define_partial_problem_evaluation_with_gradients(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol);

    
    // finite difference approximation of the gradient
    bool use_finite_differences;
    virtual void define_problem_evaluation_with_finite_differences(solution_t & sol, double step_size);
    virtual void define_partial_problem_evaluation_with_finite_differences(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol, double step_size);
    
    
    virtual std::string name() const;
    
    // redefine stuff
    bool redefine_random_initialization;
    bool redefine_boundary_repair;
    bool redefine_vtr;
    
    virtual bool vtr_reached(solution_t & sol, double vtr);
    
    virtual void init_solutions_randomly(population_t & population, size_t sample_size, const vec_t & lower_init_ranges, const vec_t & upper_init_ranges, size_t number_of_elites, rng_pt rng);
    virtual void boundary_repair(solution_t & sol);
    
    virtual std::string write_solution_info_header(bool niching_enabled);
    virtual std::string write_additional_solution_info(const solution_t & best,  const std::vector<solution_pt> & elitist_archive, bool niching_enabled);

    virtual void linkage_learning_distance_matrix(matrix_t & M);
    virtual void dynamic_linkage_learning_distance_matrix(matrix_t & M, const population_t & pop);
    
    virtual void write_solution(const solution_t & sol, const std::string & filename);
    
    virtual void set_conditional_dependencies(FOS_t & FOS, population_t & pop);
  };


}
