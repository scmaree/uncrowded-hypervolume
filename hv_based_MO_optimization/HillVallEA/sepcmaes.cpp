

/*

CMA-ES

Implementation by S.C. Maree, 2017
s.c.maree[at]amc.uva.nl
smaree.com

*/
#include "population.hpp"
#include "mathfunctions.hpp"
#include "sepcmaes.hpp"



// init cma-es default parameters
hillvallea::sep_cmaes_t::sep_cmaes_t(const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, fitness_pt fitness_function, rng_pt rng) : optimizer_t(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng)
{

  minimum_cluster_size = (int) (number_of_parameters) + 1;
  maximum_no_improvement_stretch = (int)(25 + number_of_parameters);
  selection_fraction = 0.5;

  flginiphase = 1;

  // fitness history
  fitnesshist = vec_t(3, 1e+308);

  pc.resize(number_of_parameters);
  pc.fill(0.0);

  ps.resize(number_of_parameters);
  ps.fill(0.0);
  
  covariance.resize(number_of_parameters);
  covariance.fill(1.0);
}

hillvallea::sep_cmaes_t::~sep_cmaes_t() {};


hillvallea::optimizer_pt hillvallea::sep_cmaes_t::clone() const
{

  sep_cmaes_pt opt = std::make_shared<sep_cmaes_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng);

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
  // opt->gen_span = gen_span;
  opt->selection_fraction = selection_fraction;
  opt->minimum_cluster_size = minimum_cluster_size;
  opt->init_univariate_bandwidth = init_univariate_bandwidth;

  // Stopping criteria
  //----------------------------------------------------------------------------------
  opt->maximum_no_improvement_stretch = maximum_no_improvement_stretch; ;
  opt->param_std_tolerance = param_std_tolerance;
  opt->fitness_std_tolerance = fitness_std_tolerance;

  // CMA-ES parameters
  //---------------------------------------------------------------------------------
  opt->weights = weights;              // weight vector for weighted means
  opt->lambda = lambda;              // Population size
  opt->mu = mu;                  // Selection size
  opt->mueff = mueff;               // CMA-ES mueff.
  opt->cc = cc;                  // CMA-ES cc.
  opt->ccov = ccov;
  opt->mucov = mucov;               // CMA-ES mucov.
  opt->cs = cs;                  // CMA-ES cs.
  opt->damps = damps;               // CMA-ES damping factor.
  opt->sigma = sigma;               // sigma.
  opt->chiN = chiN;                // Chi_N
  opt->pc = pc;                   // pc
  opt->ps = ps;                   // ps
  opt->mean = mean;
  opt->mean_z = mean_z;
  opt->old_mean = old_mean;             // old sample mean
  opt->covariance = covariance;
  opt->fitnesshist = fitnesshist;
  opt->flginiphase = flginiphase;
  opt->no_improvement_stretch = no_improvement_stretch; // a double as we average it


  return opt;
}

// Algorithm name
std::string hillvallea::sep_cmaes_t::name() const { return "sep-CMA-ES"; }


// returns true if any of the termination criteria is satisfied
bool hillvallea::sep_cmaes_t::checkTerminationCondition()
{

  // check the cluster size
  if (pop->size() == 0) {

    // std::cout << " terminate_on_popsize_zero";

    active = false;
    return !active;
  }

  // check the maximum parameter variance
  // we scale the fitness std by it.
  double max_param_variance = covariance.max_elem();
  vec_t mean;
  pop->mean(mean);

  // if the mean equals zero, we can't didivide by it, so terminate it when it is kinda small
  bool terminate_for_param_std_mean_zero = (mean.infinitynorm() <= 0 && (max_param_variance) < 1e-100);
  bool terminate_on_parameter_std = (max_param_variance) / mean.infinitynorm() < param_std_tolerance;
  bool terminate_on_fitness_std = (pop->average_constraint() == 0) && (pop->size() > 1) && (pop->relative_fitness_std() < fitness_std_tolerance);
  bool terminate_on_huge_multiplier = (sigma > 1e+300);
  bool terminate_on_penalty_std = (pop->average_constraint() > 0) && (pop->size() > 1) && (pop->relative_constraint_std() < penalty_std_tolerance);

  if (terminate_for_param_std_mean_zero || terminate_on_parameter_std || terminate_on_fitness_std || terminate_on_huge_multiplier || terminate_on_penalty_std)
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

void hillvallea::sep_cmaes_t::initialize_from_population(population_pt pop, size_t target_popsize)
{
  this->pop = pop;
  size_t selection_size = pop->size(); // pop is the selection (size_t)(selection_fraction * pop->size());
  initStrategyParameters(selection_size);

  no_improvement_stretch = 0;
  number_of_generations = 0;
  // number_of_evaluations = 0;

  pop->weighted_mean_of_selection(mean,weights,selection_size);
  best = solution_t(*pop->first());
  
  covariance.resize(number_of_parameters);
  covariance.fill(1.0);
  
  sigma = 1;
  
}

size_t hillvallea::sep_cmaes_t::recommended_popsize(const size_t problem_dimension) const
{
  return (4 + (size_t)(3 * log((double)(problem_dimension))));
  
  //return (size_t) std::max((double)((size_t)((2.0 / selection_fraction) + 1)), (4.0 + (size_t)(3.0 * log((double)(problem_dimension)))));
  // return (size_t)std::max((double)((size_t)((2.0 / selection_fraction) + 1)), 17.0 + 3.0*pow((double)problem_dimension, 1.5));
  // return (size_t)std::max((double)((size_t)((2.0 / selection_fraction) + 1)), 10.0*pow((double)problem_dimension, 0.5));
}

// Population size N
// strategy params remain constant over generations (as long as the popsize is). 
void hillvallea::sep_cmaes_t::initStrategyParameters(const size_t selection_size)
{

  // Population & selection size
  mu = selection_size; // (size_t)(population_size * selection_fraction); // selection
  lambda = (size_t) (mu / ((double)selection_size));
  // size_t N = lambda;

  // Weights
  weights.resize(mu);
  double sum_weights = 0.0;
  for (size_t i = 0; i < weights.size(); ++i)
  {
    weights[i] = log(mu + 1.0) - log(i+1.0);
    sum_weights += weights[i];
  }

  // normalize the weights
  double sum_weights_squared = 0.0;
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] /= sum_weights;
    sum_weights_squared += weights[i]*weights[i];
  }

  // Effective Mu
  mueff = 1.0 / sum_weights_squared; 

  // Strategy parameters for adaptation
  cc = 4.0 / (number_of_parameters + 4.0); // should be <= 1. is always true.
  // cc = (4 + mueff/number_of_parameters) / (number_of_parameters + 4 + 2*mueff/number_of_parameters); // new value

  mucov = mueff;
  ccov = 2.0 / pow(number_of_parameters + 1.41, 2.0);

  ccov = std::max(ccov, (1.0 / mucov)*ccov + (1.0 - 1.0 / mucov)*std::min(1.0, (2.0*mucov - 1.0) / (pow(number_of_parameters + 2.0, 2.0) + mucov)));

  // sep_cma
  // ccov *= (number_of_parameters + 2.0) / 3.0;

  cs = 10.0 / (number_of_parameters + 20.0);
  damps = std::max(1.0, 3.0*mueff / (number_of_parameters + 10.0)) / cs + 1.0;
  // redefined such that maximum_number_of_evaluations = inf;
  assert(ccov < 1);

  // chiN
  chiN = sqrt((double)(number_of_parameters))*(1.0 - 1.0 / (4.0*number_of_parameters) + 1.0 / (21.0*number_of_parameters*number_of_parameters));
  
}



void hillvallea::sep_cmaes_t::estimate_sample_parameters()
{

  // if the popsize changes for some reason, update the strategy params.
  if (weights.size() != pop->size()) { // pop is selection
    initStrategyParameters(pop->size());
  }

  // copy the old mean
  if (number_of_generations > 0) {
    old_mean = mean;
  }

  // sample mean
  // only the selection is given to the population. so pop->size() == selection_size and we convert it back.
  pop->weighted_mean(mean, weights);

  // sample meanz
  pop->weighted_transformed_mean(mean_z, weights);

  // copy the old mean
  if (number_of_generations == 0) {
    old_mean = mean;
  }

  // sample covariance
  estimate_covariance_univariate(pc, ps, fitnesshist, sigma, covariance, flginiphase);
}



void hillvallea::sep_cmaes_t::estimate_covariance_univariate(vec_t & pc, vec_t & ps, vec_t & fitnesshist, double & sigma, vec_t & covariance, bool & flginiphase) const
{
  if (number_of_generations > 1)
  {
    pc = (1.0 - cc)*pc + (sqrt(cc*(2.0 - cc)*mueff) / sigma) * (mean - old_mean);
    ps = (1.0 - cs)*ps + (sqrt(cs*(2.0 - cs)*mueff)) * mean_z;
  }

  fitnesshist[2] = fitnesshist[1];
  fitnesshist[1] = fitnesshist[0];
  fitnesshist[0] = pop->sols[0]->f;

  double ps_AvgSquaredNorm = ps.squaredNorm() / number_of_parameters;

  if (ps_AvgSquaredNorm > (1.5 + 10.0*sqrt(2.0 / number_of_parameters)) && (fitnesshist[1] < fitnesshist[0]) && (fitnesshist[2] < fitnesshist[1])) {
    ps *= sqrt((1.0 +std::max(0.0, log(ps_AvgSquaredNorm))) / ps_AvgSquaredNorm);
    std::cout << "ps correction applied" << std::endl;
  }

  // compute the wgMatrix
  if (!flginiphase)
  {
    double value = (1.0 - 1.0 / mucov)*pow(sigma, -2.0);
    double wg;
    
    for (size_t i = 0; i < number_of_parameters; i++)
    {
      wg = 0.0;
      for (size_t k = 0; k < pop->size(); ++k) {
        wg += value * (pop->sols[k]->param[i] - old_mean[i])*(pop->sols[k]->param[i] - old_mean[i])*weights[k];
      }

      covariance[i] = (1.0 - ccov) * covariance[i] + ( ccov / mucov )*pc[i]*pc[i] + ccov * wg;
    }
  }

  sigma *= exp((ps.norm() / chiN - 1.0) / damps);
  
  if (flginiphase && ((number_of_generations -1) > std::min(1.0 / cs, 1.0 + number_of_parameters / mucov)))
  {
    double value = ps_AvgSquaredNorm / (1.0 - pow(1.0 - cs, (double)number_of_generations));
    if (value < 1.05) {
      flginiphase = 0;
    }
  }
}

void hillvallea::sep_cmaes_t::generation(size_t sample_size, int & number_of_evaluations)
{
  estimate_sample_parameters();
  number_of_evaluations += sample_new_population(sample_size);
  pop->truncation_percentage(*pop, selection_fraction);
  average_fitness_history.push_back(pop->average_fitness());
}

size_t hillvallea::sep_cmaes_t::sample_new_population(const size_t sample_size)
{
  // this is a new generation!
  //---------------------------------------------------------------------------------------
  number_of_generations++;
  // previous_average_fitness = average_fitness;
  // Sample new population
  //---------------------------------------------------------------------------------------
  vec_t Ds = covariance;
  
  for(size_t i = 0; i < number_of_parameters; ++i) {
    Ds[i] = sqrt(Ds[i]) * sigma;
  }
  
  pop->fill_normal_univariate(sample_size, number_of_parameters, mean, Ds, lower_param_bounds, upper_param_bounds, 0, rng);

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

  return number_of_evaluations;
}

