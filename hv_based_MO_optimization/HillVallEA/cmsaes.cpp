

/*

CMA-ES

Implementation by S.C. Maree, 2017
s.c.maree[at]amc.uva.nl
smaree.com

*/
#include "population.hpp"
#include "mathfunctions.hpp"
#include "cmsaes.hpp"
#include "fitness.h"


// init cma-es default parameters
hillvallea::cmsaes_t::cmsaes_t(const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, fitness_pt fitness_function, rng_pt rng) : optimizer_t(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng)
{

  minimum_cluster_size = (int) (number_of_parameters) + 1;
  maximum_no_improvement_stretch = (int)(25 + number_of_parameters);
  selection_fraction = 0.5;

  cholesky.setIdentity(number_of_parameters, number_of_parameters);
  covariance.setIdentity(number_of_parameters, number_of_parameters);

  size_t stallsize = (size_t)(10 + floor(30.0 * number_of_parameters / recommended_popsize(number_of_parameters)));  // Stall time(for termination criterion)
  bestf_NE.resize(stallsize, 0.0);

  for (size_t i = 0; i < stallsize; ++i)
  {
    bestf_NE[i] = (stallsize - i)*(1e140);
  }

}

hillvallea::cmsaes_t::~cmsaes_t() {};

// Algorithm name
std::string hillvallea::cmsaes_t::name() const { return "CMSA"; }

hillvallea::optimizer_pt hillvallea::cmsaes_t::clone() const
{

  cmsaes_pt opt = std::make_shared<cmsaes_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng);

  // Optimizer data members
  //-------------------------------------------
  opt->active = active;
  opt->number_of_parameters = number_of_parameters;
  opt->lower_param_bounds = lower_param_bounds;
  opt->upper_param_bounds = upper_param_bounds;
  opt->fitness_function = fitness_function;
  opt->number_of_generations = number_of_generations;
  opt->rng = rng;
  opt->pop = std::make_shared<population_t>(); //!! A copy of the contents, not the pointer
  opt->pop->addSolutions(*pop);
  opt->best = best;
  // opt->number = number;
  // opt->average_fitness = average_fitness;
  // opt->previous_average_fitness = previous_average_fitness;
  // opt->time_to_opt = time_to_opt;
  opt->average_fitness_history = average_fitness_history;
  // opt->previous_average_fitness_generation = previous_average_fitness_generation;
  opt->selection_fraction = selection_fraction;
  opt->minimum_cluster_size = minimum_cluster_size;
  opt->init_univariate_bandwidth = init_univariate_bandwidth;
  
  // Stopping criteria
  //----------------------------------------------------------------------------------
  opt->maximum_no_improvement_stretch = maximum_no_improvement_stretch;
  opt->param_std_tolerance = param_std_tolerance;
  opt->fitness_std_tolerance = fitness_std_tolerance;

  // CMA-ES Parameters
  //---------------------------------------------------------------------------------
  opt->weights = weights;
  opt->lambda = lambda;           
  opt->mu = mu;              
  opt->sigma = sigma;           
  opt->tau = tau;
  opt->tau_c = tau_c;
  opt->mean = mean;
  opt->covariance = covariance;
  opt->cholesky = cholesky;
  opt->no_improvement_stretch = no_improvement_stretch;


  return opt;
}


// returns true if any of the termination criteria is satisfied
bool hillvallea::cmsaes_t::checkTerminationCondition()
{

  // check the cluster size
  if (pop->size() == 0) {

    // std::cout << " terminate_on_popsize_zero";

    active = false;
    return !active;
  }

  // check the maximum parameter variance
  // we scale the fitness std by it.
  double max_param_variance = 0.0;
  for (size_t i = 0; i < covariance.rows(); ++i) {
    if (covariance[i][i] > max_param_variance) {
      max_param_variance = covariance[i][i];
    }
  }

  // 3. Check imp over time
  size_t stallsize = (size_t)(10 + floor(30.0 * number_of_parameters / recommended_popsize(number_of_parameters)));  // Stall time(for termination criterion)
  //double TolHistFun = 1e-5;
  double bestf_NEmin = 1e308;
  double bestf_NEmax = -1e308;
  for (size_t i = this->bestf_NE.size() - stallsize; i < this->bestf_NE.size(); ++i)
  {
    if (this->bestf_NE[i] < bestf_NEmin)
      bestf_NEmin = this->bestf_NE[i];

    if (this->bestf_NE[i] > bestf_NEmax)
      bestf_NEmax = this->bestf_NE[i];
  }
  //double max_diff = bestf_NEmax - bestf_NEmin;





  vec_t mean;
  pop->mean(mean);

  // if the mean equals zero, we can't didivide by it, so terminate it when it is kinda small
  bool terminate_for_param_std_mean_zero = (mean.infinitynorm() <= 0 && sqrt(max_param_variance) < param_std_tolerance);
  bool terminate_on_parameter_std = sqrt(max_param_variance) / mean.infinitynorm() < param_std_tolerance;
  bool terminate_on_fitness_std = (pop->average_constraint() == 0) && (pop->size() > 1) && (pop->relative_fitness_std() < fitness_std_tolerance);
  bool terminate_on_huge_multiplier = (sigma > 1e+300);
  bool terminate_on_slow_improvement = false; // max_diff < TolHistFun;
  bool terminate_on_penalty_std = (pop->average_constraint() > 0) && (pop->size() > 1) && (pop->relative_constraint_std() < penalty_std_tolerance);

  if (terminate_for_param_std_mean_zero || terminate_on_parameter_std || terminate_on_fitness_std || terminate_on_huge_multiplier || terminate_on_slow_improvement || terminate_on_penalty_std)
  {
    // elitist_archive.push_back(sols[0]);
    active = false;
    return !active;
  }
 

  // if we have not terminated so far, the cluster is active.
  // if, due to selection, the cluster is shrunken, it is set to active again.
  active = true;
  return !active;

}

void hillvallea::cmsaes_t::initStrategyParameters(size_t selection_size)
{

  double TCovCoeff = 1.0; // rs-cmsa uses this. 

  mu = selection_size;
  tau = 1.0 / sqrt(2.0 * number_of_parameters);
  tau_c = 1 + number_of_parameters * (number_of_parameters + 1.0) / (selection_size) * TCovCoeff;

  // Weights
  weights.resize(mu);
  double sum_weights = 0.0;
  for (size_t i = 0; i < weights.size(); ++i)
  {
    weights[i] = log(mu + 1.0) - log(i + 1.0);
    sum_weights += weights[i];
  }

  // normalize the weights
  // double sum_weights_squared = 0.0;
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] /= sum_weights;
  }
}


void hillvallea::cmsaes_t::generation(size_t sample_size, int & number_of_evaluations)
{
  pop->sort_on_fitness();
  pop->truncation_percentage(*pop, selection_fraction);
  estimate_sample_parameters();
  number_of_evaluations += sample_new_population(sample_size);
  // pop->truncation_percentage(*pop, selection_fraction);
  average_fitness_history.push_back(pop->average_fitness());
}

void hillvallea::cmsaes_t::initialize_from_population(population_pt pop, size_t target_popsize)
{
  this->pop = pop;
  // pop->setOrigin(this);

  // pop is the selection (size_t)(selection_fraction * pop->size());
  size_t selection_size = pop->size(); 
  
  // Population & selection size
  mu = selection_size; 
  
  initStrategyParameters(mu);

  no_improvement_stretch = 0;
  number_of_generations = 0;

  pop->mean(mean);
  best = solution_t(*pop->first());

  if(pop->size() == 1)
  {
    covariance.setIdentity(number_of_parameters, number_of_parameters);
    cholesky.setIdentity(number_of_parameters, number_of_parameters);

    covariance.multiply(init_univariate_bandwidth*0.01);
    cholesky.multiply(sqrt(init_univariate_bandwidth*0.01));
  }
  else
  {
    // pop->covariance(mean, covariance);
    // choleskyDecomposition(covariance, cholesky);
    pop->covariance_univariate(mean, covariance);
    choleskyDecomposition_univariate(covariance, cholesky);
  }

  sigma = 1;
  
}

size_t hillvallea::cmsaes_t::recommended_popsize(const size_t problem_dimension) const
{
  return (size_t)std::max((double)((size_t)((2.0 / selection_fraction) + 1)), (3.0 * sqrt((double)(problem_dimension))));
}



void hillvallea::cmsaes_t::estimate_sample_parameters()
{

  if (weights.size() != pop->size()) { // pop is selection
    initStrategyParameters(pop->size());
  }


  pop->weighted_mean(mean,weights);

  if (number_of_generations == 0)
  {
    sigma = 1;
  }
  else
  {
    // get sigma
    //-------------------------------------------
    double sum_logsigma = 0.0;
    double weighted_sum_logsigma = 0.0;

    for (size_t i = 0; i < pop->size(); ++i) {
      weighted_sum_logsigma += weights[i] * log(pop->sols[i]->multiplier);
      sum_logsigma += log(pop->sols[i]->multiplier); // actually, this should be of the whole pop before selection TODO
    }

    sigma *= exp(weighted_sum_logsigma) / exp(sum_logsigma / pop->size());
    assert(!isnan(sigma));
  }

  // sample covariance
  estimate_covariance(covariance, cholesky);

}



void hillvallea::cmsaes_t::estimate_covariance(matrix_t & covariance, matrix_t & cholesky) const
{

  // init WG matrix by zeros
  matrix_t wgMatrix = matrix_t(number_of_parameters, number_of_parameters, 0.0);

  // compute the wgMatrix
  for (size_t i = 0; i < number_of_parameters; i++) {
    for (size_t j = i; j < number_of_parameters; j++) {
      for (size_t k = 0; k < pop->size(); ++k) {
        wgMatrix[i][j] += weights[k] * (pop->sols[k]->param_transformed[i])*(pop->sols[k]->param_transformed[j]);
      }
      wgMatrix[j][i] = wgMatrix[i][j];
    }
  }
  

  // compute the covariance matrix
  // C <- (1-1/tau_c) * C + 1/tc <ss^T>
  for (size_t i = 0; i < number_of_parameters; i++) {
    for (size_t j = i; j < number_of_parameters; j++) {
      covariance[i][j] = (1.0 - 1.0 / tau_c) * covariance[i][j] + ( 1.0 / tau_c) * wgMatrix[i][j];
      covariance[j][i] = covariance[i][j];
    }
  }
  
  choleskyDecomposition(covariance, cholesky);

}

const double hillvallea::cmsaes_t::getParamDouble(const std::string & param_name) const
{
  if (param_name.compare("nis") == 0) {
    return no_improvement_stretch;
  }

  if (param_name.compare("nog") == 0) {
    return number_of_generations;
  }

  if (param_name.compare("sigma") == 0) {
    return sigma;
  }

  std::cout << "Unknown Parameter requested (" << param_name << ")" << std::endl;
  assert(0);

  return -1.0;
}

const hillvallea::vec_t hillvallea::cmsaes_t::getParamVec(const std::string & param_name) const
{
  if (param_name.compare("mean") == 0) {
    return mean;
  }

  std::cout << "Unknown Parameter requested (" << param_name << ")" << std::endl;
  assert(0);

  return vec_t();
}

const hillvallea::matrix_t hillvallea::cmsaes_t::getParamMatrix(const std::string & param_name) const
{

  if (param_name.compare("covariance") == 0)
  {
    return covariance;
  }

  if (param_name.compare("cholesky") == 0) {
    return cholesky;
  }

  std::cout << "Unknown Parameter requested (" << param_name << ")" << std::endl;
  assert(0);

  return matrix_t();
}

size_t hillvallea::cmsaes_t::sample_new_population(const size_t sample_size)
{
  // this is a new generation!
  //---------------------------------------------------------------------------------------
  number_of_generations++;

  // Sample a new population
  //--------------------------------------------
  size_t number_of_elites = 1;
  pop->sols.resize(sample_size);

  // for each sol in the pop, sample.
  for (size_t i = 0; i < pop->sols.size(); ++i)
  {

    // save the elite (if it is defined)
    if (i < number_of_elites && pop->sols[i] != nullptr)
      continue;

    // if the solution is not yet initialized, do it now.
    if (pop->sols[i] == nullptr)
    {
      solution_pt sol = std::make_shared<solution_t>(number_of_parameters);
      pop->sols[i] = sol;
    }

    // Sample independent standard normal variables Z = N(0,1)
    // std::normal_distribution<double> std_normal(0.0, 1.0);
    vec_t z(number_of_parameters);

    // try to sample within bounds
    bool sample_in_range = false;
    int attempts = 0;
    
    // sample a new solution
    std::normal_distribution<double> std_normal(0.0, 1.0); // move this inside the for-loop below to match it with the CMA-ES code by Peter

    // try using the normal distribution
    while (!sample_in_range && attempts < 100)
    {


      for (size_t j = 0; j < number_of_parameters; ++j) {
        z[j] = std_normal(*rng);
      }

      // pop->sols[i]->param_transformed = z;
      // z *= sigma;

      pop->sols[i]->multiplier = sigma * exp(tau *std_normal(*rng));
      pop->sols[i]->param_transformed = cholesky.product(z); // param_transformed = s_l in the CMSA paper. 

      pop->sols[i]->param = mean + pop->sols[i]->multiplier * pop->sols[i]->param_transformed;
      
      if(fitness_function->redefine_boundary_repair) {
        fitness_function->boundary_repair(*pop->sols[i]);
      } else {
        boundary_repair(pop->sols[i]->param, lower_param_bounds, upper_param_bounds);
      }
      
      sample_in_range = in_range(pop->sols[i]->param, lower_param_bounds, upper_param_bounds);
      attempts++;

    }
    // if that fails, fall back to uniform from the initial user-defined range
    if (!sample_in_range) {
      sample_uniform(pop->sols[i]->param, number_of_parameters, lower_param_bounds, upper_param_bounds, rng);
      pop->sols[i]->param_transformed = vec_t(number_of_parameters, 0.0);
      std::cout << "Too many sample attempts. Sample uniform. (mathfunctions.cpp:105)" << std::endl;
    }

  }


  // evaluate the population
  //---------------------------------------------------------------------------------------
  size_t number_of_evaluations = pop->evaluate(fitness_function, 1);
  pop->sort_on_fitness();
  // pop->setOrigin(this);

  // Update Params
  //---------------------------------------------------------------------------------------
  // average_fitness = pop->average_fitness();
  bool improvement = pop->improvement_over(best.f);

  if (improvement)  {
    no_improvement_stretch = 0;
  }
  else {
    no_improvement_stretch++;
  }

  best = solution_t(*pop->first());
  bestf_NE.push_back(best.f);

  return number_of_evaluations;
}

