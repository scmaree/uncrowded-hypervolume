/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA


*/

#include "solution.hpp"
#include "optimizer.hpp"
#include "amalgam.hpp"
#include "amalgam_univariate.hpp"
#include "iamalgam.hpp"
#include "iamalgam_univariate.hpp"
#include "cmsaes.hpp"
#include "sepcmaes.hpp"
#include "cmaes.hpp"
#include "gomea.hpp"

hillvallea::optimizer_pt hillvallea::init_optimizer(const int local_optimizer_index, const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, fitness_pt fitness_function, rng_pt rng)
{

  // parse settings
  switch (local_optimizer_index)
  {
    case 0: return std::make_shared<amalgam_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng, false); break; // AMaLGaM full
    case 1: return std::make_shared<amalgam_univariate_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng, false); break; // AMaLGaM Univariate
    case 10: return std::make_shared<cmsaes_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng); break; // CMSA-ES (full)
    case 20: return std::make_shared<iamalgam_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng); break; // iAMaLGaM full
    case 21: return std::make_shared<iamalgam_univariate_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng); break; // iAMaLGaM univariate
    case 30: return std::make_shared<cmaes_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng); break; // CMA-ES (full) - probably implementation bug
    case 31: return std::make_shared<sep_cmaes_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng); break; // sep-CMA-ES (univariate)
    case 40: return std::make_shared<amalgam_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng, true); break; // AMaLGaM full -- multiplier-vector
    case 41: return std::make_shared<amalgam_univariate_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng, true); break; // AMaLGaM univariate -- multiplier-vector
    case 50: return std::make_shared<gomea_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, 50, fitness_function, rng); break; // GOMEA full
    case 59: return std::make_shared<gomea_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, 59, fitness_function, rng); break; // GOMEA full
    case 61: return std::make_shared<gomea_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, 61, fitness_function, rng); break; // GOMEA univariate
    case 62: return std::make_shared<gomea_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, 62, fitness_function, rng); break; // GOMEA learn linkage tree & intermediate_updates
    case 63: return std::make_shared<gomea_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, 63, fitness_function, rng); break; // GOMEA static linkage tree (from distance matrix) & intermediate_updates
    case 64: return std::make_shared<gomea_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, 64, fitness_function, rng); break; // GOMEA static linkage tree & with UB & (from distance matrix) & intermediate_updates
    case 65: return std::make_shared<gomea_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, 65, fitness_function, rng); break; // GOMEA random linkage tree & intermediate_updates
    case 66: return std::make_shared<gomea_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, 66, fitness_function, rng); break; // GOMEA dynamic linkage tree & intermediate_updates
    case 74: return std::make_shared<gomea_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, 74, fitness_function, rng); break; // GOMEA static conditional linkage tree & with UB & (from distance matrix) & intermediate_updates
    case 76: return std::make_shared<gomea_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, 76, fitness_function, rng); break; // GOMEA learn conditional marginal linkage (from distance matrix) & intermediate_updates
    case 84: return std::make_shared<gomea_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, 84, fitness_function, rng); break; // GOMEA with gradient descent step
    default: return std::make_shared<amalgam_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng, true); break;
  }

}

// initialization of the general parameters of the EDAs
hillvallea::optimizer_t::optimizer_t(const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, fitness_pt fitness_function, rng_pt rng)
{

  active = true;
  this->number_of_parameters = number_of_parameters;
  this->lower_param_bounds = lower_param_bounds;
  this->upper_param_bounds = upper_param_bounds;
  this->fitness_function = fitness_function;
  number_of_generations = 0;
  this->rng = rng;
  pop = std::make_shared<population_t>();
  // best;
  average_fitness_history.resize(0); 
  selection_fraction = 0; // this will definitely cause weird stuff.
  this->init_univariate_bandwidth = init_univariate_bandwidth;
  maximum_no_improvement_stretch = 1000000;
  use_boundary_repair = false;
  
  param_std_tolerance = 1e-20;// TODO: Make this fitness-function dependent
  fitness_std_tolerance = 1e-20;//
  penalty_std_tolerance = 1e-20;//

}

hillvallea::optimizer_t::~optimizer_t() {}

 // Initialization
 //---------------------------------------------------------------------------------
void hillvallea::optimizer_t::initialize_from_population(population_pt pop, size_t target_popsize)
{
  std::cout << "initialize_from_population not implemented" << std::endl;
  assert(false);
  return;
}


size_t hillvallea::optimizer_t::recommended_popsize(const size_t problem_dimension) const
{
  std::cout << "recommended_popsize not implemented" << std::endl;
  assert(false);
  return 0;
}

void hillvallea::optimizer_t::generation(size_t sample_size, int & number_of_evaluations)
{
  std::cout << "generation not implemented" << std::endl;
  assert(false);
  return;
}

bool hillvallea::optimizer_t::checkTerminationCondition()
{
  std::cout << "checkTerminiationCriteria not implemented" << std::endl;
  assert(false);
  return true;
}

size_t hillvallea::optimizer_t::sample_new_population(const size_t sample_size)
{
  std::cout << "sample_new_population not implemented" << std::endl;
  assert(false);
  return 0;
}

std::string hillvallea::optimizer_t::name() const
{
  std::cout << "name not implemented" << std::endl;
  assert(false);
  return "name not implemented"; 
}

void hillvallea::optimizer_t::estimate_sample_parameters()
{
  std::cout << "estimate_sample_parameters not implemented" << std::endl;
  assert(false);
}

