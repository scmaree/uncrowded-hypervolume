

/*

AMaLGaM

Implementation by S.C. Maree, 2017
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "population.h"
#include "mathfunctions.h"
#include "iamalgam.h"
#include "elitist_archive.h"

// init amalgam default parameters
hicam::iamalgam_t::iamalgam_t(fitness_pt fitness_function, rng_pt rng, bool use_univariate) : cluster_t(fitness_function, rng)
{
  terminated = false;
  multiplier = 1.0;
  out_of_bounds_sample_ratio = 1.0;
  use_boundary_repair = true;

  this->use_univariate = use_univariate;
  
  double selection_fraction = 0.35;
  size_t number_of_parameters = fitness_function->get_number_of_parameters();
  
  // for univariate
  if(use_univariate)
  {
    eta_p = 1.0 - exp(-0.31*pow((int)(selection_fraction*recommended_popsize(number_of_parameters)), 0.27) / pow((double)number_of_parameters, 0.067));
    eta_s = 1.0 - exp(-0.40*pow((int)(selection_fraction*recommended_popsize(number_of_parameters)), 0.15) / pow((double)number_of_parameters, -0.034));
  }
  else
  {
    eta_p = 1.0 - exp(-1.2*pow((int)(selection_fraction*recommended_popsize(number_of_parameters)), 0.31) / pow((double)number_of_parameters, 0.50));
    eta_s = 1.0 - exp(-1.1*pow((int)(selection_fraction*recommended_popsize(number_of_parameters)), 1.20) / pow((double)number_of_parameters, 1.60));
  }
}

hicam::iamalgam_t::iamalgam_t(const iamalgam_t & other) : cluster_t(other)
{

  // Essential data members
  //-------------------------------------------
  this->mean = other.mean;
  this->multiplier = other.multiplier;                       // distribution multiplier

  this->covariance = other.covariance;                      // sample covariance matrix C
  this->cholesky = other.cholesky;                         // decomposed covariance matrix C = LL^T
  this->inverse_cholesky = other.inverse_cholesky;        // inverse of the cholesky decomposition

  this->use_boundary_repair = other.use_boundary_repair;
  this->use_univariate = other.use_univariate;

  this->eta_p = other.eta_p;
  this->eta_s = other.eta_s;
  
  this->aggregated_covariance = other.aggregated_covariance;
  this->generational_covariance = other.generational_covariance;
  
  this->ams_direction = other.ams_direction;
  
  // Stopping criteria
  //-------------------------------------------
  this->improvement = other.improvement;
  this->out_of_bounds_sample_ratio = other.out_of_bounds_sample_ratio;
  this->terminated = other.terminated;

}

size_t hicam::iamalgam_t::recommended_popsize(size_t number_of_parameters) const
{
  
  double selection_fraction = 0.35;
  if(use_univariate)
  {
    return (size_t) std::max((double) ((size_t) ((2.0 / selection_fraction) + 1)), 4.0*pow((double)number_of_parameters, 0.5));
  }
  else
  {
    return (size_t)std::max((double)((size_t)((2.0 / selection_fraction) + 1)), 10.0*pow((double)number_of_parameters, 0.5));
  }
}


hicam::iamalgam_t::~iamalgam_t(){};

// Algorithm Name
std::string hicam::iamalgam_t::name() const
{ 
  if (use_univariate)
    return "iAMaLGaM-Univariate";
  else
    return "iAMaLGaM";
}

bool hicam::iamalgam_t::adaptDistributionMultiplier(const elitist_archive_t & elitist_archive, size_t & no_improvement_stretch, const size_t maximum_no_improvement_stretch)
{

  double distribution_multiplier_decrease = 0.9;
  double out_of_bounds_sample_threshold = 0.9;
  double st_dev_ratio_threshold = 1.0;

  double st_dev_ratio;

  bool cluster_failure = true;
  // size_t number_of_cluster_failures = 0;
 
  improvement = generationalImprovementForOneCluster(st_dev_ratio, elitist_archive);
  
  if (out_of_bounds_sample_ratio > out_of_bounds_sample_threshold) {
    multiplier *= 0.5;
  }

  if (improvement)
  {
    cluster_failure = false;
    no_improvement_stretch = 0;

    if (multiplier < 1.0) {
      multiplier = 1.0;
    }

    if (st_dev_ratio > st_dev_ratio_threshold) {
      multiplier /= distribution_multiplier_decrease;
    }

  }
  else
  {
    if (multiplier > 1.0) {
      cluster_failure = false;
    }

    if ((multiplier > 1.0) || (no_improvement_stretch >= maximum_no_improvement_stretch)) {
      multiplier *= distribution_multiplier_decrease;
    }

    if ((no_improvement_stretch < maximum_no_improvement_stretch) && (multiplier < 1.0))
      multiplier = 1.0;
  }
  
  //if (cluster_failure)
  //  number_of_cluster_failures++;
  

  return cluster_failure;

}



//// returns true if any of the termination criteria is satisfied
bool hicam::iamalgam_t::checkTerminationCondition()
{
  // 1. if the cluster is empty, deactivate it.
  if (size() == 0) {
    terminated = true;
    return terminated;
  }

  // if the mean equals zero, we can't didivide by it, so terminate it when it is kinda small
  bool terminate_on_distribution_multiplier = (multiplier < 1e-10);

  if ( terminate_on_distribution_multiplier)
  {
     terminated = true;
     return terminated;
  }

  // if we have not terminated so far, the cluster is active.
  // if, due to selection, the cluster is shrunken, it is set to active again.
  terminated = false;
  return terminated;

}





void hicam::iamalgam_t::estimateParameters()
{
  
  // inherit the multiplier
  if (previous != nullptr) {
    multiplier = previous->getDouble("amalgam_multiplier");
  }
  
  vec_t true_mean;
  compute_mean(true_mean);
  
  if (multiplier < 1.0)
  {
    if (elites.size() > 0) {
      mean = elites[0]->param;
    }
    else {
      mean = sols[0]->param;
    }
  }
  else {
    mean = true_mean;
  }

  // covariance matrix
  if(size() == 1)
  {
    // kernel
    generational_covariance.setIdentity(mean.size(), mean.size());
    generational_covariance.multiply(init_bandwidth);
    
  }
  else
  {
    if(use_univariate) // || size() <= mean.size())
    {
      // univariate
      vec_t temp_univariate_covariance;
      compute_covariance_univariate(true_mean, temp_univariate_covariance);
      
      generational_covariance.reset(mean.size(), mean.size(), 0.0);
      
      for (size_t i = 0; i < mean.size(); i++) {
        generational_covariance[i][i] = temp_univariate_covariance[i];
      }
      
    }
    else
    {  // full
      bool enable_regularization = false;
      compute_covariance(true_mean, generational_covariance, enable_regularization);
    }
  }
  
  size_t number_of_parameters = fitness_function->get_number_of_parameters();
  
  if (previous == nullptr)
  {
    ams_direction.resize(number_of_parameters);
    ams_direction.fill(0.0);
    
    aggregated_covariance.reset(number_of_parameters, number_of_parameters, 0.0);
    
    for (size_t i = 0; i < number_of_parameters; i++) {
      aggregated_covariance[i][i] = generational_covariance[i][i];
    }
  }
  else
  {
    
    vec_t old_mean = previous->getVec("amalgam_mean");
    if (previous->previous == nullptr)  { // oh men i love programming.
      ams_direction = mean - old_mean;
    }
    else {
      ams_direction = previous->getVec("ams_direction");
      ams_direction = (1.0 - eta_p)*ams_direction + eta_p*(mean - old_mean);
    }
    
    aggregated_covariance = previous->getMatrix("amalgam_covariance_matrix");
    
    for (size_t i = 0; i < number_of_parameters; i++) {
      for (size_t j = 0; j < number_of_parameters; j++) {
        aggregated_covariance[i][j] = (1.0 - eta_s)*aggregated_covariance[i][j] + eta_s*generational_covariance[i][j];
      }
    }
    
  }
  
  covariance = aggregated_covariance;
  
  // Cholesky decomposition
  choleskyDecomposition(covariance, cholesky);
  
  // apply the multiplier
  cholesky.multiply(sqrt(multiplier));
  
  // invert the cholesky decomposition
  int n = (int)covariance.rows();
  inverse_cholesky.setRaw(matrixLowerTriangularInverse(cholesky.toArray(), n), n, n);
  
}




/**
* Determines whether an improvement has resulted this
* cluster. Returns 1 in case of an improvement, 0 otherwise. The
* standard-deviation ratio required by the SDR-AVS mechanism is computed
* and returned.
*/
bool hicam::iamalgam_t::generationalImprovementForOneCluster(double & st_dev_ratio, const elitist_archive_t & elitist_archive) const
{

  size_t number_of_improvements = 0;
  st_dev_ratio = 0.0;

  for (size_t i = 0; i < this->size(); i++)
  {
    if (elitist_archive.solutionHasImproved(*sols[i]))
    {
      number_of_improvements++;
      st_dev_ratio += getSDR(sols[i]->param);
    }
  }

  if (number_of_improvements > 0)
    st_dev_ratio /= number_of_improvements;

  return(number_of_improvements > 0);
}

// Compute the SDR
//--------------------------------------------------------------------------------------
double hicam::iamalgam_t::getSDR(const vec_t & params) const
{
  return inverse_cholesky.lowerProduct(params - mean).infinitynorm();
}


//
//
//// Apply the Anticipated Mean Shift (AMS) to the first solutions (not the elites)
////------------------------------------------------------------------------
void hicam::iamalgam_t::apply_ams(std::vector<solution_pt> & solutions, const size_t number_of_elites, const size_t number_of_ams_solutions, const double ams_factor, const vec_t & ams_direction, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds) const
{

  // we have a shrink factor in case the shifts are
  // outside of the parameter bounds
  double shrink_factor;
  unsigned int attempts = 0;

  // loop over the first solutions to shift them,
  // but we save the elite.
  for (size_t i = number_of_elites; i < std::min(number_of_ams_solutions + number_of_elites, solutions.size()); ++i)
  {

    // try to sample within bounds
    shrink_factor = 1.0;

    // shift x.
    vec_t ams_params = solutions[i]->param;
    ams_params += shrink_factor * ams_factor * ams_direction;

    if (use_boundary_repair) {
      boundary_repair(solutions[i]->param, lower_param_bounds, upper_param_bounds);
    }

    // try smaller shifts until the sol is within range
    while (attempts < 100 && !in_range(ams_params, lower_param_bounds, upper_param_bounds))
    {

      // if not, decrease the shrink_factor
      attempts++;
      shrink_factor *= 0.5;
      ams_params -= shrink_factor * ams_factor * ams_direction;

    }

    if (attempts < 100) {
      solutions[i]->param = ams_params;
    }

  }
}


bool hicam::iamalgam_t::updateStrategyParameters(const elitist_archive_t & elitist_archive, size_t & no_improvement_stretch, const size_t maximum_no_improvement_stretch)
{
  bool cluster_failure = adaptDistributionMultiplier(elitist_archive, no_improvement_stretch, maximum_no_improvement_stretch);
  return cluster_failure;

}



void hicam::iamalgam_t::computeParametersForSampling()
{

  // collect covariance
  covariance = aggregated_covariance;
  
  // FULL COVARIANCE
  choleskyDecomposition(covariance, cholesky);

  // sanity check if cholesky decomposition is a full matrix. Else,
  // do a univariate estimate of it.
  double diag_det = cholesky.determinantDiag();
  
  // if det(chol) = 0, then det(cov) = 0, and the matrix is not full. Then, estimate
  // a univariate covariance matrix.
  if(fabs(diag_det) < 1e-8)
  {
    vec_t diag = covariance.get_diagonal();
    vec_t uni_chol;
    choleskyDecomposition_univariate(diag, uni_chol);
    cholesky.set_diagonal_matrix(uni_chol);
  }
  
  // apply the multiplier
  cholesky.multiply(sqrt(multiplier));

  // invert the cholesky decomposition
  int n = (int)covariance.rows();
  inverse_cholesky.setRaw(matrixLowerTriangularInverse(cholesky.toArray(), n), n, n);

}



void hicam::iamalgam_t::generateNewSolutions(std::vector<solution_pt> & solutions, size_t number_of_solutions, size_t number_of_ams_solutions, rng_pt & rng)
{

  // Sample new population
  //----------------------------------------------------------------------------------------
  vec_t lower_param_bounds, upper_param_bounds;
  fitness_function->get_param_bounds(lower_param_bounds, upper_param_bounds);
  double delta_ams = 2.0;

  unsigned int out_of_bounds_draws = 0;
  unsigned int samples_drawn_from_normal = 0;

  out_of_bounds_draws = this->fill_vector_normal(solutions, number_of_solutions, fitness_function->number_of_parameters, mean, cholesky, use_boundary_repair, lower_param_bounds, upper_param_bounds, 0, rng);

  // apply the AMS
  // 
  if (number_of_ams_solutions > 0 && solutions.size() > 0)
  {

    vec_t old_mean;

    assert(previous != nullptr); // warning: setting AMS but previous is not set.

    old_mean = previous->getVec("amalgam_mean");
 
    apply_ams(solutions, 0, number_of_ams_solutions, delta_ams*multiplier, ams_direction, lower_param_bounds, upper_param_bounds); // ams, not to elite
  }

  samples_drawn_from_normal = out_of_bounds_draws + (unsigned int) number_of_solutions;
  this->out_of_bounds_sample_ratio = out_of_bounds_draws / (double)(samples_drawn_from_normal);
 
}


hicam::vec_t hicam::iamalgam_t::getVec(std::string variable_name)
{
  if (variable_name.compare("amalgam_mean") == 0)
  {
    return mean;
  }
  else
  {
    if (variable_name.compare("ams_direction") == 0)
    {
      return ams_direction;
    }
    else
    {
      assert(false);
      return vec_t();
    }
  }
}

double hicam::iamalgam_t::getDouble(std::string variable_name)
{
  
  if (variable_name.compare("amalgam_multiplier") == 0)
  {
    return multiplier;
  }
  else
  {
    assert(false);
    return -1.0;
  }
}

hicam::matrix_t hicam::iamalgam_t::getMatrix(std::string variable_name)
{
  
  if (variable_name.compare("amalgam_covariance_matrix") == 0)
  {
    return aggregated_covariance;
  }
  else
  {
    assert(false);
    return matrix_t();
  }
}

