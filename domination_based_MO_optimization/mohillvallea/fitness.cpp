
#include "fitness.h"
#include "mathfunctions.h"



hicam::fitness_t::fitness_t() 
{
  number_of_evaluations = 0;
  number_of_objectives = 0;
  number_of_parameters = 0;
  igd_available = false;
  igdx_available = false;
  sr_available = false;
  redefine_random_initialization = false;
  
  hypervolume_max_f0 = 1.1;
  hypervolume_max_f1 = 1.1;
  use_lex = false;
  partial_evaluations_available = false;
  analytical_gradient_available = false;
  
  partial_evaluations_available = false;
  linkage_learning_distance_matrix_available = false;
  fos_element_size_lower_bound = 1;
  fos_element_size_upper_bound = number_of_parameters;
  
  analytical_igd_avialable = false;
  analytical_gd_avialable = false;
  use_boundary_repair = false;
  
  // bezier stuff
  do_evaluate_bezier_controlpoints = false;
  
}
hicam::fitness_t::~fitness_t() {}


void hicam::fitness_t::set_number_of_objectives(size_t & number_of_objectives)
{
  std::cout << "fitness_function error 'set_number_of_objectives' not implemented" << std::endl;
  assert(false);
  return;
}

void hicam::fitness_t::set_number_of_parameters(size_t & number_of_parameters)
{
  std::cout << "fitness_function error 'set_number_of_parameters' not implemented" << std::endl;
  assert(false);
  return;
}

void hicam::fitness_t::get_param_bounds(vec_t & lower, vec_t & upper) const
{
  std::cout << "fitness_function error 'get_param_bounds' not implemented" << std::endl;
  assert(false);
  return;
}

size_t hicam::fitness_t::get_number_of_objectives() const
{
  return number_of_objectives;
}

size_t hicam::fitness_t::get_number_of_parameters() const
{
  return number_of_parameters;
}

void hicam::fitness_t::evaluate(solution_t & sol)
{
  assert(sol.number_of_parameters() == number_of_parameters);
  
  if(use_lex) { sol.use_lex = true; }
  sol.set_number_of_objectives(number_of_objectives);

  define_problem_evaluation(sol);
  
  for(size_t i = 0; i < sol.obj.size(); ++i) {
    assert(!isnan(sol.obj[i]));
  }
  
  number_of_evaluations++;
}

void hicam::fitness_t::evaluate(solution_pt & sol) 
{ 
  evaluate(*sol); 
}

void hicam::fitness_t::partial_evaluate(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
{
  assert(sol.param.size() == number_of_parameters);
  
  if(use_lex) { sol.use_lex = true; }
  sol.set_number_of_objectives(number_of_objectives);
  
  // define_problem_evaluation(sol);
  define_partial_problem_evaluation(sol, touched_parameter_idx, old_sol);
  
  number_of_evaluations++;
}

void hicam::fitness_t::partial_evaluate(solution_pt & sol, const std::vector<size_t> & touched_parameter_idx, const solution_pt & old_sol)
{
  partial_evaluate(*sol, touched_parameter_idx, *old_sol);
}


void hicam::fitness_t::evaluate_with_gradients(solution_t & sol)
{
  assert(sol.number_of_parameters() == number_of_parameters);
  
  if(use_lex) { sol.use_lex = true; }
  sol.set_number_of_objectives(number_of_objectives);
  
  define_problem_evaluation_with_gradients(sol);
  
  for(size_t i = 0; i < sol.obj.size(); ++i) {
    assert(!isnan(sol.obj[i]));
  }
  
  number_of_evaluations++;
}

void hicam::fitness_t::evaluate_with_gradients(solution_pt & sol)
{
  evaluate_with_gradients(*sol);
}

void hicam::fitness_t::partial_evaluate_with_gradients(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
{
  assert(sol.param.size() == number_of_parameters);
  
  if(use_lex) { sol.use_lex = true; }
  sol.set_number_of_objectives(number_of_objectives);
  
  // define_problem_evaluation(sol);
  define_partial_problem_evaluation_with_gradients(sol, touched_parameter_idx, old_sol);
  
  number_of_evaluations++;
}

void hicam::fitness_t::partial_evaluate_with_gradients(solution_pt & sol, const std::vector<size_t> & touched_parameter_idx, const solution_pt & old_sol)
{
  partial_evaluate_with_gradients(*sol, touched_parameter_idx, *old_sol);
}


// evaluates the function
// for new functions, set problem_evaluation.
// evaluate covers the evaluation itself and can be set to cover other stuff
// such as counting the number of evaluations or printing

void hicam::fitness_t::define_problem_evaluation(solution_t & sol)
{
  std::cout << "fitness_function error 'problem_evaluation' not implemented" << std::endl;
  assert(false);
  return;
}

void hicam::fitness_t::define_partial_problem_evaluation(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
{
  partial_evaluations_available = false;
  
  std::cout << "Warning, partial evaluations not implemented. Disabled, re-evaluate this solution." << std::endl;
  
  define_problem_evaluation(sol);
  
}


void hicam::fitness_t::define_problem_evaluation_with_gradients(solution_t & sol)
{
  std::cout << "fitness_function error 'problem_evaluation_with_gradients' not implemented" << std::endl;
  assert(false);
  return;
}

void hicam::fitness_t::define_partial_problem_evaluation_with_gradients(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
{
  partial_evaluations_available = false;
  
  std::cout << "Warning, partial evaluations _with_gradients not implemented. Disabled, re-evaluate this solution." << std::endl;
  
  define_problem_evaluation_with_gradients(sol);
  
}

void hicam::fitness_t::evaluate_bezier_controlpoint(solution_t & sol)
{
  assert(sol.number_of_parameters() == number_of_parameters);
  
  if(use_lex) { sol.use_lex = true; }
  sol.set_number_of_objectives(number_of_objectives);
  
  define_problem_evaluation_for_bezier_controlpoint(sol);
  
  for(size_t i = 0; i < sol.obj.size(); ++i) {
    assert(!isnan(sol.obj[i]));
  }
  
  number_of_evaluations++;
}

void hicam::fitness_t::partial_evaluate_bezier_controlpoint(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
{
  assert(sol.number_of_parameters() == number_of_parameters);
  
  if(use_lex) { sol.use_lex = true; }
  sol.set_number_of_objectives(number_of_objectives);
  
  define_partial_problem_evaluation_for_bezier_controlpoint(sol, touched_parameter_idx, old_sol);
  
  for(size_t i = 0; i < sol.obj.size(); ++i) {
    assert(!isnan(sol.obj[i]));
  }
  
  number_of_evaluations++;
}

void hicam::fitness_t::define_problem_evaluation_for_bezier_controlpoint(solution_t & sol)
{
  do_evaluate_bezier_controlpoints = false;
  
  std::cout << "Warning, evaluations for Bezier control points not implemented. Disabled do_evaluate_bezier_controlpoints. Control point is not evaluated" << std::endl;
  
}
void hicam::fitness_t::define_partial_problem_evaluation_for_bezier_controlpoint(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
{
  do_evaluate_bezier_controlpoints = false;
  
  std::cout << "Warning, partial evaluations for Bezier control points not implemented. Disabled partial evaluation, attempting to perform a full evaluation" << std::endl;
  
  define_problem_evaluation_for_bezier_controlpoint(sol);
}


double hicam::fitness_t::distance_to_front(const solution_t & sol)
{
  std::cout << "Warning, 'distance_to_front' not implemented." << std::endl;
  return -1.0;
}



std::string hicam::fitness_t::name() const
{
  std::cout << "fitness_function warning 'name' not implemented" << std::endl;
  return "no name";
}

bool hicam::fitness_t::get_pareto_set()
{
  std::cout << "fitness_function warning 'get_pareto_set' not implemented" << std::endl;
  igdx_available = false;
  igd_available = false;
  return false;
}

void hicam::fitness_t::init_solutions_randomly(population_pt & population, size_t sample_size, const vec_t & lower_init_ranges, const vec_t & upper_init_ranges, size_t number_of_elites, rng_pt rng)
{
  std::cout << "fitness_function warning 'init_solutions_randomly' not implemented" << std::endl;
  redefine_random_initialization = false;
}
void hicam::fitness_t::linkage_learning_distance_matrix(matrix_t & M)
{
  std::cout << "fitness_function warning 'linkage_learning_distance_matrix' not implemented" << std::endl;
  linkage_learning_distance_matrix_available = false;
}
