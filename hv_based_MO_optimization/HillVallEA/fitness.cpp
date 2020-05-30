
#include "fitness.h"
#include "mathfunctions.hpp"
#include "gomea.hpp"


hillvallea::fitness_t::fitness_t()
{
  number_of_evaluations = 0;
  number_of_parameters = 0;
  redefine_random_initialization = false;
  redefine_boundary_repair = false;
  covariance_block_size = number_of_parameters;
  maximum_number_of_evaluations = 0;
  
  has_round_off_errors_in_partial_evaluations = true;
  partial_evaluations_available = false;
  linkage_learning_distance_matrix_available = false;
  dynamic_linkage_learning_distance_matrix_available = false;
  fos_element_size_lower_bound = 1;
  fos_element_size_upper_bound = 0;
  dynamic_objective = false;
  use_boundary_repair = false;
  redefine_vtr = false;
  local_optima_tolerance = -1; // invalid value, falls back to default
  use_finite_differences = false;
}

hillvallea::fitness_t::~fitness_t() {}


void hillvallea::fitness_t::set_number_of_parameters(size_t & number_of_parameters)
{
  std::cout << "fitness_function error 'set_number_of_parameters' not implemented" << std::endl;
  assert(false);
  return;
}

void hillvallea::fitness_t::get_param_bounds(vec_t & lower, vec_t & upper) const
{
  std::cout << "fitness_function error 'get_param_bounds' not implemented" << std::endl;
  assert(false);
  return;
}

size_t hillvallea::fitness_t::get_number_of_parameters() const
{
  return number_of_parameters;
}

void hillvallea::fitness_t::evaluate(solution_t & sol)
{
  assert(sol.param.size() == number_of_parameters);

  define_problem_evaluation(sol);
  
  assert(!isnan(sol.f));
  
  number_of_evaluations++;
}

void hillvallea::fitness_t::evaluate(solution_pt & sol)
{ 
  evaluate(*sol); 
}

void hillvallea::fitness_t::evaluate_with_gradients(solution_t & sol)
{
  assert(sol.param.size() == number_of_parameters);
  
  define_problem_evaluation_with_gradients(sol);
  
  assert(!isnan(sol.f));
  
  number_of_evaluations++;
}

void hillvallea::fitness_t::evaluate_with_gradients(solution_pt & sol)
{
  evaluate_with_gradients(*sol);
}

void hillvallea::fitness_t::evaluate_with_finite_differences(solution_t & sol, double step_size)
{
  assert(sol.param.size() == number_of_parameters);
  
  define_problem_evaluation_with_finite_differences(sol, step_size);
  
  assert(!isnan(sol.f));
  
  number_of_evaluations++;
}

void hillvallea::fitness_t::evaluate_with_finite_differences(solution_pt & sol, double step_size)
{
  evaluate_with_finite_differences(*sol, step_size);
}


void hillvallea::fitness_t::partial_evaluate(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
{
  assert(sol.param.size() == number_of_parameters);
  
  define_partial_problem_evaluation(sol, touched_parameter_idx, old_sol);
  
  assert(!isnan(sol.f));
  
  number_of_evaluations += touched_parameter_idx.size() / (double) number_of_parameters;
}

void hillvallea::fitness_t::partial_evaluate(solution_pt & sol, const std::vector<size_t> & touched_parameter_idx, const solution_pt & old_sol)
{
  partial_evaluate(*sol, touched_parameter_idx, *old_sol);
}

void hillvallea::fitness_t::partial_evaluate_with_gradients(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
{
  assert(sol.param.size() == number_of_parameters);
  
  define_partial_problem_evaluation_with_gradients(sol, touched_parameter_idx, old_sol);
  
  assert(!isnan(sol.f));
  
  number_of_evaluations += touched_parameter_idx.size() / (double) number_of_parameters;
}

void hillvallea::fitness_t::partial_evaluate_with_gradients(solution_pt & sol, const std::vector<size_t> & touched_parameter_idx, const solution_pt & old_sol)
{
  partial_evaluate_with_gradients(*sol, touched_parameter_idx, *old_sol);
}

void hillvallea::fitness_t::partial_evaluate_with_finite_differences(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol, double step_size)
{
  assert(sol.param.size() == number_of_parameters);
  
  define_partial_problem_evaluation_with_finite_differences(sol, touched_parameter_idx, old_sol, step_size);
  
  assert(!isnan(sol.f));
  
  number_of_evaluations += touched_parameter_idx.size() / (double) number_of_parameters;
}

void hillvallea::fitness_t::partial_evaluate_with_finite_differences(solution_pt & sol, const std::vector<size_t> & touched_parameter_idx, const solution_pt & old_sol, double step_size)
{
  partial_evaluate_with_finite_differences(*sol, touched_parameter_idx, *old_sol, step_size);
}

// evaluates the function
// for new functions, set problem_evaluation.
// evaluate covers the evaluation itself and can be set to cover other stuff
// such as counting the number of evaluations or printing

void hillvallea::fitness_t::define_problem_evaluation(solution_t & sol)
{
  std::cout << "fitness_function error 'problem_evaluation' not implemented" << std::endl;
  assert(false);
  return;
}

void hillvallea::fitness_t::define_partial_problem_evaluation(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
{
  partial_evaluations_available = false;
  
  std::cout << "Warning, partial evaluations not implemented. Disabled, re-evaluate this solution." << std::endl;
  
  define_problem_evaluation(sol);
  
}

void hillvallea::fitness_t::define_problem_evaluation_with_gradients(solution_t & sol)
{
  std::cout << "fitness_function error 'define_problem_evaluation_with_gradients' not implemented" << std::endl;
  assert(false);
  return;
}

void hillvallea::fitness_t::define_partial_problem_evaluation_with_gradients(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
{
  partial_evaluations_available = false;
  
  std::cout << "Warning, partial_evaluations_with_gradients not implemented. Disabled, re-evaluate this solution." << std::endl;
  
  define_problem_evaluation_with_gradients(sol);
  
}

void hillvallea::fitness_t::define_problem_evaluation_with_finite_differences(solution_t & sol, double step_size)
{
  use_finite_differences = false;
  std::cout << "fitness_function error 'define_problem_evaluation_with_gradients' not implemented" << std::endl;
  assert(false);
  return;
}

void hillvallea::fitness_t::define_partial_problem_evaluation_with_finite_differences(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol, double step_size)
{
  partial_evaluations_available = false;
  
  std::cout << "Warning, partial_evaluations_with_FD not implemented. Disabled, re-evaluate this solution." << std::endl;
  
  define_problem_evaluation_with_finite_differences(sol, step_size);
  
}

std::string hillvallea::fitness_t::name() const
{
  std::cout << "fitness_function warning 'name' not implemented" << std::endl;
  return "no name";
}

void hillvallea::fitness_t::init_solutions_randomly(population_t & population, size_t sample_size, const vec_t & lower_init_ranges, const vec_t & upper_init_ranges, size_t number_of_elites, rng_pt rng)
{
  std::cout << "fitness_function warning 'init_solutions_randomly' not implemented" << std::endl;
  redefine_random_initialization = false;
}

void hillvallea::fitness_t::boundary_repair(solution_t & sol)
{
  std::cout << "fitness_function warning 'boundary_repair' not implemented" << std::endl;
  redefine_boundary_repair = false;
}

std::string hillvallea::fitness_t::write_solution_info_header(bool niching_enabled) {
  return "";
}
std::string hillvallea::fitness_t::write_additional_solution_info(const solution_t & best,  const std::vector<solution_pt> & elitist_archive, bool niching_enabled)
{
  return "";
}



void hillvallea::fitness_t::linkage_learning_distance_matrix(matrix_t & M)
{
  std::cout << "fitness_function warning 'linkage_learning_distance_matrix' not implemented" << std::endl;
  linkage_learning_distance_matrix_available = false;
}

void hillvallea::fitness_t::dynamic_linkage_learning_distance_matrix(matrix_t & M, const population_t & pop)
{
  std::cout << "fitness_function warning 'dynamic_linkage_learning_distance_matrix' not implemented" << std::endl;
  dynamic_linkage_learning_distance_matrix_available = false;
  linkage_learning_distance_matrix(M);
}

void hillvallea::fitness_t::write_solution(const solution_t & sol, const std::string & filename)
{
  // std::cout << "fitness_function warning 'write_solution' not implemented" << std::endl;
}


void hillvallea::fitness_t::sort_population_parameters(population_t & pop, FOS_t & FOS)
{
  // std::cout << "fitness_function warning 'sort_population_parameters' not implemented" << std::endl;

}

bool hillvallea::fitness_t::vtr_reached(solution_t & sol, double vtr)
{
  std::cout << "fitness_function warning 'vtr_reached' not implemented. Custum vtr disabled now." << std::endl;
  redefine_vtr = false;
  return false;
}


void hillvallea::fitness_t::set_conditional_dependencies(FOS_t & FOS, population_t & pop)
{
  std::cout << "fitness_function warning 'set_conditional_dependencies' not implemented." << std::endl;
  
  for(size_t i = 0; i < FOS.length(); ++i) {
    FOS.sets[i]->sample_conditionally = false;
  }
}
