/*

AMaLGaM as part of HillVallEA

Implementation by S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA


*/

#include "population.hpp"
#include "mathfunctions.hpp"
#include "amalgam.hpp"
#include "fitness.h"

// init amalgam default parameters
hillvallea::amalgam_t::amalgam_t(const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, fitness_pt fitness_function, rng_pt rng, bool enable_multiplier_vec) : optimizer_t(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng)
{

  // default values for AMaLGaM
  maximum_no_improvement_stretch = (int)(number_of_parameters + 25);
  selection_fraction = 0.35;
  st_dev_ratio_threshold = 1.0; 
  distribution_multiplier_decrease = 0.9;
  sample_succes_ratio_threshold = 0.1;


  apply_ams = true;
  delta_ams = 2.0;
  
  use_multiplier_vec = enable_multiplier_vec;

}

hillvallea::amalgam_t::~amalgam_t(){};



hillvallea::optimizer_pt hillvallea::amalgam_t::clone() const
{

  amalgam_pt opt = std::make_shared<amalgam_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng, use_multiplier_vec);

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
  opt->average_fitness_history = average_fitness_history;
  opt->selection_fraction= selection_fraction;
  opt->init_univariate_bandwidth= init_univariate_bandwidth;

  // Stopping criteria
  //----------------------------------------------------------------------------------
  opt->maximum_no_improvement_stretch = maximum_no_improvement_stretch;
  opt->param_std_tolerance = param_std_tolerance;     
  opt->fitness_std_tolerance = fitness_std_tolerance; 

  // AMaLGaM data members
  //-------------------------------------------
  opt->mean = mean;
  opt->covariance = covariance;       
  opt->cholesky = cholesky;             
  opt->inverse_cholesky = inverse_cholesky;    
  opt->no_improvement_stretch = no_improvement_stretch;
  opt->multiplier = multiplier;
  opt->multiplier_vec = multiplier_vec;
  opt->use_multiplier_vec = use_multiplier_vec;
  opt->old_mean = old_mean;
  opt->st_dev_ratio_threshold = st_dev_ratio_threshold;      
  opt->distribution_multiplier_decrease = distribution_multiplier_decrease; 
  opt->maximum_no_improvement_stretch = maximum_no_improvement_stretch;   
  opt->sample_succes_ratio_threshold = sample_succes_ratio_threshold; 
  opt->delta_ams = delta_ams;
  opt->apply_ams = apply_ams;

  return opt;
}



// Algorithm Name
std::string hillvallea::amalgam_t::name() const { return "AMaLGaM-full"; }

// Initial initialization of the algorithm
// Population should be sorted on fitness (fittest first)
void hillvallea::amalgam_t::initialize_from_population(population_pt pop, size_t target_popsize)
{
  this->pop = pop;

  multiplier = 1.0;
  multiplier_vec.resize(pop->problem_size());
  multiplier_vec.fill(1.0);
  no_improvement_stretch = 0;
  pop->mean(old_mean);
  mean = old_mean;
  pop->sort_on_fitness();
  best = solution_t(*pop->sols[0]);

}


void hillvallea::amalgam_t::generation(size_t sample_size, int & number_of_evaluations)
{
  estimate_sample_parameters();
  number_of_evaluations += sample_new_population(sample_size);
  // pop->truncation_percentage(*pop, selection_fraction);
  average_fitness_history.push_back(pop->average_fitness());
  // pop->sort_on_fitness();
  pop->truncation_percentage(*pop, selection_fraction);
}

size_t hillvallea::amalgam_t::recommended_popsize(const size_t problem_dimension) const
{
  //return (4 + (size_t)(3 * log((double)(problem_dimension))));
  return (size_t)std::max((double)((size_t)((2.0 / selection_fraction) + 1)), 17.0 + 3.0*pow((double)problem_dimension, 1.5));
  // return (size_t)std::max((double)((size_t)((2.0 / selection_fraction) + 1)), 10.0*pow((double)problem_dimension, 0.5));
}

void hillvallea::amalgam_t::update_distribution_multiplier_vec(vec_t & multiplier, const bool improvement, int & no_improvement_stretch, const vec_t & sample_success_ratio, const vec_t & sdr) const
{
  
  assert(sample_success_ratio.size() == multiplier.size());
  
  
  // default variables;
  double sample_succes_ratio_threshold = 0.10;
  
  for(size_t i = 0; i < multiplier.size(); ++i)
  {
    // if >90% of the samples is out of bounds, multiplier *0.5
    if (sample_success_ratio[i] < sample_succes_ratio_threshold) {
      multiplier[i] *= 0.5;
    }
  }
  
  
  if (improvement)
  {
    no_improvement_stretch = 0.0;
    
    for(size_t i = 0; i < multiplier.size(); ++i)
    {
      if (multiplier[i] < 1.0) {
        multiplier[i] = 1.0;
      }
      
      if (sdr[i] > st_dev_ratio_threshold) { // to get a single multiplier back, replace sdr[i] here with sdr.max_elem()
        multiplier[i] /= distribution_multiplier_decrease;
      }
    }
  }
  else
  {
    // choice: increase no_improvement_stretch when all multipliers are 1.0, or when one of them is 1.0 ?
    if(multiplier.max_elem() <= 1.0) {
      no_improvement_stretch++;
    }
    
    for(size_t i = 0; i < multiplier.size(); ++i)
    {
      if (multiplier[i] > 1.0 || no_improvement_stretch >= maximum_no_improvement_stretch) {
        multiplier[i] *= distribution_multiplier_decrease;
      }
      
      if (multiplier[i] < 1.0 && no_improvement_stretch < maximum_no_improvement_stretch) {
        multiplier[i] = 1.0;
      }
      
    }
    
  }
}


void hillvallea::amalgam_t::update_distribution_multiplier(double & multiplier, const bool improvement, int & no_improvement_stretch, const double sample_success_ratio, const double sdr) const
{

  // default variables;
  double sample_succes_ratio_threshold = 0.10;

  // if >90% of the samples is out of bounds, multiplier *0.5
  if (sample_success_ratio < sample_succes_ratio_threshold)
    multiplier *= 0.5;

  if (improvement)
  {
    no_improvement_stretch = 0;

    if (multiplier < 1.0)
      multiplier = 1.0;

    if (sdr > st_dev_ratio_threshold)
      multiplier /= distribution_multiplier_decrease;

  }
  else
  {

    if (multiplier <= 1.0)
      no_improvement_stretch++;

    if (multiplier > 1.0 || no_improvement_stretch >= maximum_no_improvement_stretch)
      multiplier *= distribution_multiplier_decrease;

    if (multiplier < 1.0 && no_improvement_stretch < maximum_no_improvement_stretch)
      multiplier = 1.0;

  }
  
  // std::cout << multiplier << " ";
}

// returns true if any of the termination criteria is satisfied
bool hillvallea::amalgam_t::checkTerminationCondition()
{

  if (number_of_generations == 0) {
    active = true;
    return !active;
  }

  // 1. if the cluster is empty, deactivate it.
  if (pop->size() == 0) {

    // std::cout << " terminate_on_popsize_zero";

    active = false;
    return !active;
  }

  // 2. check the maximum parameter variance
  // we scale the fitness std by it.
  double max_param_variance = 0.0;
  for (size_t i = 0; i < covariance.rows(); ++i) {
    if (covariance[i][i] > max_param_variance) {
      max_param_variance = covariance[i][i];
    }
  }

  vec_t mean;
  pop->mean(mean);

  // if the mean equals zero, we can't didivide by it, so terminate it when it is kinda small
  bool terminate_for_param_std_mean_zero = (mean.infinitynorm() <= 0 && sqrt(max_param_variance) < param_std_tolerance);
  bool terminate_on_parameter_std = sqrt(max_param_variance) / mean.infinitynorm() < param_std_tolerance;
  bool terminate_on_fitness_std = (pop->average_constraint() == 0) && (pop->size() > 1) && (pop->relative_fitness_std() < fitness_std_tolerance);
  bool terminate_on_constraint_std = (pop->average_constraint() > 0) && (pop->size() > 1) && (pop->relative_constraint_std() < penalty_std_tolerance);
  bool terminate_on_distribution_multiplier;
  if(use_multiplier_vec) {
    terminate_on_distribution_multiplier = (multiplier_vec.max_elem() < 1e-10);
  }
  else
  {
     terminate_on_distribution_multiplier = (multiplier < 1e-10);
  }

  if (terminate_for_param_std_mean_zero || terminate_on_parameter_std || terminate_on_fitness_std || terminate_on_distribution_multiplier || terminate_on_constraint_std)
  {
    active = false;
    return !active;
  }
  
  // if we have not terminated so far, the cluster is active.
  // if, due to selection, the cluster is shrunken, it is set to active again.
  active = true;
  return !active;

}

void hillvallea::amalgam_t::estimate_sample_parameters()
{

  // Compute sample mean and sample covariance
  old_mean = mean;

  // Change the focus of the search to the best solution 
  if((use_multiplier_vec && multiplier_vec.max_elem() < 1.0) || multiplier < 1.0) {
      mean = pop->sols[0]->param;
  } else {
    pop->mean(mean);
  }
  
  // if the population size is too small,
  // estimate a univariate covariance matrix
  if (pop->size() == 1)
  {
    covariance.setIdentity(mean.size(), mean.size());
    covariance.multiply(init_univariate_bandwidth*0.01);
  }
  else 
  {
    // pop->covariance(mean, covariance);
    pop->covariance(mean, covariance);
  }
  // Cholesky decomposition
  choleskyDecomposition(covariance, cholesky);
  // choleskyDecomposition_univariate(covariance, cholesky); // this makes AM-full exactly equal to AM-uni
  
  // apply the multiplier
  if(use_multiplier_vec)
  {
    // maybe do this more efficient?
    matrix_t multiplier_mat(mean.size(), mean.size(), 0.0);
    
    for(size_t i = 0; i < multiplier_vec.size(); ++i) {
      multiplier_mat[i][i] = sqrt(multiplier_vec[i]);
    }
    cholesky = cholesky * multiplier_mat;
    
  } else {
    cholesky.multiply(sqrt(multiplier));
  }
  
  // invert the cholesky decomposition
  int n = (int)covariance.rows();
  inverse_cholesky.setRaw(matrixLowerTriangularInverse(cholesky.toArray(), n), n, n);
  
}

// sample a new population
size_t hillvallea::amalgam_t::sample_new_population(const size_t sample_size)
{

  // Sample new population
  //----------------------------------------------------------------------------------------
  int number_of_samples = pop->fill_normal(sample_size, number_of_parameters, mean, cholesky, lower_param_bounds, upper_param_bounds, 1, rng);

  // apply the AMS
  vec_t ams_direction;
  size_t number_of_ams_solutions = 0;
  if (apply_ams)
  {
    ams_direction = mean - old_mean;

    number_of_ams_solutions = (size_t)(0.5*selection_fraction*sample_size); // alpha ams
    apply_ams_to_population(number_of_ams_solutions, delta_ams, ams_direction); // ams, not to elite
  }

  // evaluate the population
  //---------------------------------------------------------------------------------------
  size_t number_of_evaluations = pop->evaluate(fitness_function, 1);
  pop->sort_on_fitness();

  // Update Params
  //---------------------------------------------------------------------------------------
  bool improvement = pop->improvement_over(best.f);
  if(use_multiplier_vec) {
    vec_t sdr = getSDR_vec(best, mean, inverse_cholesky);
    // TODO actually count sample_succes ratio per problem variable!
    vec_t sample_success_ratio(mean.size(), (double)(sample_size - 1) / number_of_samples); // -1 cuz we do not sample the best.
    update_distribution_multiplier_vec(multiplier_vec, improvement, no_improvement_stretch, sample_success_ratio, sdr);
    
    //std::cout << std::endl << covariance << std::endl;
    
    // std::cout << std::fixed << std::setprecision(2) << multiplier_vec << std::endl;
  }
  else
  {
    double sdr = getSDR(best, mean, inverse_cholesky);
    double sample_success_ratio = (double)(sample_size - 1) / number_of_samples; // we do not sample the best.
    update_distribution_multiplier(multiplier, improvement, no_improvement_stretch, sample_success_ratio, sdr);
  }
  best = solution_t(*pop->first());

  number_of_generations++;

  return number_of_evaluations;
}





// Compute the SDR
//--------------------------------------------------------------------------------------
double hillvallea::amalgam_t::getSDR(const solution_t & best, const vec_t & mean, matrix_t & inverse_chol) const
{
  size_t i;

  // find improvements over the best.
  vec_t average_params(number_of_parameters, 0.0);
  for (i = 0; (i < pop->size()) && (pop->sols[i]->f < best.f); ++i) {
    average_params += pop->sols[i]->param;
  }

  if (i == 0)
    return 0.0;

  average_params /= (double)i;

  vec_t diff = average_params - mean;

  return inverse_chol.lowerProduct(diff).infinitynorm();

}

hillvallea::vec_t hillvallea::amalgam_t::getSDR_vec(const solution_t & best, const vec_t & mean, matrix_t & inverse_chol) const
{
  size_t i;
  
  // find improvements over the best.
  vec_t average_params(number_of_parameters, 0.0);
  for (i = 0; (i < pop->size()) && (pop->sols[i]->f < best.f); ++i) {
    average_params += pop->sols[i]->param;
  }
  
  if (i == 0)
  {
    vec_t diff(number_of_parameters, 0.0);
    return diff;
  }
  
  average_params /= (double)i;
  
  vec_t diff = average_params - mean;
  //vec_t scaled_diff(diff.size());
  //for(size_t i = 0; i < scaled_diff.size(); ++i) {
  //  scaled_diff[i] = fabs(scaled_diff[i]) / cholesky[i][i];
  //}
  
  vec_t scaled_diff = inverse_chol.lowerProduct(diff);
  for(size_t i = 0; i < scaled_diff.size(); ++i) {
    scaled_diff[i] = fabs(scaled_diff[i]);
  }
  
  return scaled_diff;
  
}

// Apply the Anticipated Mean Shift (AMS) to the first solutions (not the elite)
//------------------------------------------------------------------------
void hillvallea::amalgam_t::apply_ams_to_population(const size_t number_of_ams_solutions, const double ams_factor, const vec_t & ams_direction)
{

  // we have a shrink factor in case the shifts are
  // outside of the parameter bounds
  double shrink_factor;
  int attempts = 0;

  // loop over the first solutions to shift them,
  // but we save the elite.
  for (size_t i = 1; i < std::min(number_of_ams_solutions + 1, pop->sols.size()); ++i)
  {

    // try to sample within bounds
    shrink_factor = 1;

    // shift x.
    vec_t ams_params = pop->sols[i]->param;
    if(use_multiplier_vec)
    {
      for(size_t j = 0; j < ams_params.size(); ++j) {
        ams_params[j] += shrink_factor * ams_factor * multiplier_vec[j] * ams_direction[j]; // not entirely sure this is correct yet
      }
    }
    else
    {
      ams_params += shrink_factor * ams_factor * multiplier * ams_direction;
    }
    /*
    if(fitness_function->redefine_boundary_repair) {
      fitness_function->boundary_repair(*pop->sols[i]);
    } else {
      boundary_repair(pop->sols[i]->param, lower_param_bounds, upper_param_bounds);
    }
*/
    // try smaller shifts until the sol is within range
    while (attempts < 100 && !in_range(ams_params, lower_param_bounds, upper_param_bounds))
    {

      // if not, decrease the shrink_factor
      attempts++;
      shrink_factor *= 0.5;
      if(use_multiplier_vec)
      {
        for(size_t j = 0; j < ams_params.size(); ++j) {
          ams_params[j] -= shrink_factor * ams_factor * multiplier_vec[j] * ams_direction[j]; // not entirely sure this is correct yet
        }
      }
      else
      {
        ams_params -= shrink_factor * ams_factor * multiplier * ams_direction;
      }

    }

    if (attempts < 100) {
      pop->sols[i]->param = ams_params;
    }

  }
}
