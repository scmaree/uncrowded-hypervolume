#pragma once

/*

HICAM Multi-objective

By S.C. Maree, 2016
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "hicam_internal.h"
#include "population.h"
#include <functional>

// Defines the fitness function of our choice
// implementation idea from libcmaes by Emmanuel Benazera.
//------------------------------------------------

namespace hicam
{

  class fitness_t
  {

  public:

    fitness_t();
    ~fitness_t();

    size_t number_of_parameters;
    size_t number_of_objectives;
    unsigned int number_of_evaluations;

    size_t get_number_of_objectives() const;
    size_t get_number_of_parameters() const;
    
    bool use_lex; // Enable lexicographic optimization
    bool partial_evaluations_available;
    bool analytical_gradient_available;
    bool linkage_learning_distance_matrix_available;
    size_t fos_element_size_lower_bound;
    size_t fos_element_size_upper_bound;
    bool use_boundary_repair;
    
    // evaluates the function
    // for new functions, define problem_evaluation in "define_problem_evaluation".
    // evaluate covers the evaluation itself and can be set to cover other stuff
    // such as counting the number of evaluations or printing
    void evaluate(solution_t & sol);
    void evaluate(solution_pt & sol);
    void partial_evaluate(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol);
    void partial_evaluate(solution_pt & sol, const std::vector<size_t> & touched_parameter_idx, const solution_pt & old_sol);

    void evaluate_with_gradients(solution_t & sol);
    void evaluate_with_gradients(solution_pt & sol);
    void partial_evaluate_with_gradients(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol);
    void partial_evaluate_with_gradients(solution_pt & sol, const std::vector<size_t> & touched_parameter_idx, const solution_pt & old_sol);

    
    // Bezier stuff
    bool do_evaluate_bezier_controlpoints;
    void evaluate_bezier_controlpoint(solution_t & sol);
    void partial_evaluate_bezier_controlpoint(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol);
    virtual void define_problem_evaluation_for_bezier_controlpoint(solution_t & sol);
    virtual void define_partial_problem_evaluation_for_bezier_controlpoint(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol);
    
    // Placeholders for user-defined objective functions
    //----------------------------------------------------------------------------------------
    virtual void set_number_of_objectives(size_t & number_of_objectives);
    virtual void set_number_of_parameters(size_t & number_of_parameters);
    virtual void get_param_bounds(vec_t & lower, vec_t & upper) const;
    virtual void define_problem_evaluation(solution_t & sol);
    virtual void define_partial_problem_evaluation(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol);
    virtual void define_problem_evaluation_with_gradients(solution_t & sol);
    virtual void define_partial_problem_evaluation_with_gradients(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol);

    virtual std::string name() const;

    // Pareto_set, for performance measuring
    population_t pareto_set;
    std::vector<population_pt> pareto_sets;
    vec_t pareto_sets_max_igdx;
    
    bool igd_available;
    bool igdx_available;
    bool sr_available;
    virtual bool get_pareto_set();
    bool analytical_igd_avialable;
    bool analytical_gd_avialable;
    
    double hypervolume_max_f0;
    double hypervolume_max_f1;
    
    // redefine initialization
    bool redefine_random_initialization;
    
    virtual void init_solutions_randomly(population_pt & population, size_t sample_size, const vec_t & lower_init_ranges, const vec_t & upper_init_ranges, size_t number_of_elites, rng_pt rng);
    virtual void linkage_learning_distance_matrix(matrix_t & M);
    
    virtual double distance_to_front(const solution_t & sol);
  };


}
