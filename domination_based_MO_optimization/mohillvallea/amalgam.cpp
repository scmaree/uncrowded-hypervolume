

/*

AMaLGaM

Implementation by S.C. Maree, 2017
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "population.h"
#include "mathfunctions.h"
#include "amalgam.h"
#include "elitist_archive.h"

// init amalgam default parameters
hicam::amalgam_t::amalgam_t(fitness_pt fitness_function, rng_pt rng, bool use_univariate, bool use_multiplier_vec) : cluster_t(fitness_function, rng)
{
  terminated = false;
  multiplier = 1.0;
  out_of_bounds_sample_ratio = 1.0;
  use_boundary_repair = true;

  this->use_univariate = use_univariate;
  this->use_multiplier_vec = use_multiplier_vec;
  // this->use_multiplier_vec = use_multiplier_vec;
  this->multiplier_vec.reset(fitness_function->number_of_parameters,1.0);
  
  this->use_principal_cholesky_decomposition = true; // use_multiplier_vec;
}

hicam::amalgam_t::amalgam_t(const amalgam_t & other) : cluster_t(other)
{

  // Essential data members
  //-------------------------------------------
  this->mean = other.mean;
  this->multiplier = other.multiplier;                       // distribution multiplier
  this->use_multiplier_vec = other.use_multiplier_vec;
  this->multiplier_vec = other.multiplier_vec;

  this->covariance = other.covariance;                      // sample covariance matrix C
  this->cholesky = other.cholesky;                         // decomposed covariance matrix C = LL^T
  this->inverse_cholesky = other.inverse_cholesky;        // inverse of the cholesky decomposition

  this->use_boundary_repair = other.use_boundary_repair;
  this->use_univariate = other.use_univariate;
 
  this->univariate_covariance = other.univariate_covariance;                      // sample univariate_covariance matrix C
  this->univariate_cholesky = other.univariate_cholesky;                         // decomposed univariate_covariance matrix C = LL^T
  this->univariate_inverse_cholesky = other.univariate_inverse_cholesky;        // inverse of the univariate_cholesky decomposition

  // Stopping criteria
  //-------------------------------------------
  this->improvement = other.improvement;
  this->out_of_bounds_sample_ratio = other.out_of_bounds_sample_ratio;
  this->out_of_bounds_sample_ratio_vec = other.out_of_bounds_sample_ratio_vec;
  this->terminated = other.terminated;

}


hicam::amalgam_t::~amalgam_t(){};

// Algorithm Name
std::string hicam::amalgam_t::name() const 
{ 
  if (use_univariate) {
    if(use_multiplier_vec) {
      return "AMaLGaM-Univariate-vec";
    } else {
      return "AMaLGaM-Univariate";
    }
  } else {
    if(use_multiplier_vec) {
      return "AMaLGaM-vec";
    } else {
      return "AMaLGaM";
    }
  }
}

bool hicam::amalgam_t::adaptDistributionMultiplier(const elitist_archive_t & elitist_archive, size_t & no_improvement_stretch, const size_t maximum_no_improvement_stretch)
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


bool hicam::amalgam_t::adaptDistributionMultiplier_vec(const elitist_archive_t & elitist_archive, size_t & no_improvement_stretch, const size_t maximum_no_improvement_stretch)
{
  
  double distribution_multiplier_decrease = 0.9;
  double out_of_bounds_sample_threshold = 0.9;
  double st_dev_ratio_threshold = 1.0;
  
  vec_t st_dev_ratio_vec;
  
  bool cluster_failure = true;
  
  improvement = generationalImprovementForOneCluster_vec(st_dev_ratio_vec, elitist_archive);
  
  for(size_t i = 0; i < multiplier_vec.size(); ++i)
  {
    if (out_of_bounds_sample_ratio_vec[i] > out_of_bounds_sample_threshold) {
      multiplier_vec[i] *= 0.5;
    }
  }
    
  if (improvement)
  {
    cluster_failure = false;
    no_improvement_stretch = 0;
    
    for(size_t i = 0; i < multiplier_vec.size(); ++i)
    {
      if (multiplier_vec[i] < 1.0) {
        multiplier_vec[i] = 1.0;
      }
    
      if (st_dev_ratio_vec[i] > st_dev_ratio_threshold) {
        multiplier_vec[i] /= distribution_multiplier_decrease;
      }
    }
  }
  else
  {
    if (multiplier_vec.max() > 1.0) {
      cluster_failure = false;
    }
    
    for(size_t i = 0; i < multiplier_vec.size(); ++i)
    {
      if ((multiplier_vec[i] > 1.0) || (no_improvement_stretch >= maximum_no_improvement_stretch)) {
        multiplier_vec[i] *= distribution_multiplier_decrease;
      }
      
      if ((no_improvement_stretch < maximum_no_improvement_stretch) && (multiplier_vec[i] < 1.0)) {
        multiplier_vec[i] = 1.0;
      }
    }
  }
  
  //if (cluster_failure)
  //  number_of_cluster_failures++;
  
  
  return cluster_failure;
  
}


//// returns true if any of the termination criteria is satisfied
bool hicam::amalgam_t::checkTerminationCondition()
{
  // 1. if the cluster is empty, deactivate it.
  if (size() == 0) {
    terminated = true;
    return terminated;
  }

  // if the mean equals zero, we can't didivide by it, so terminate it when it is kinda small
  bool terminate_on_distribution_multiplier = (multiplier_vec.max() < 1e-10 || multiplier < 1e-10);

  if (terminate_on_distribution_multiplier)
  {
     terminated = true;
     return terminated;
  }

  // if we have not terminated so far, the cluster is active.
  // if, due to selection, the cluster is shrunken, it is set to active again.
  terminated = false;
  return terminated;

}





void hicam::amalgam_t::estimateParameters()
{

  // Change the focus of the search to the best solution
  vec_t true_mean;
  compute_mean(true_mean);
  
  if (multiplier < 1.0) {
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


  if (use_univariate)
  {
    // UNIVARIATE COVARIANCE
    // if the population size is too small,
    // estimate a univariate covariance matrix
    if (size() == 1) {
      univariate_covariance.reset(mean.size(),init_bandwidth);
    }
    else {
      compute_covariance_univariate(true_mean, univariate_covariance);
    }
  }
  else
  {
    // FULL COVARIANCE
    // if the population size is too small,
    // estimate a univariate covariance matrix
    if (size() == 1)
    {
      covariance.setIdentity(mean.size(), mean.size());
      covariance.multiply(init_bandwidth);
    }
    else
    {
      bool enable_regularization = false;
      compute_covariance(true_mean, covariance, enable_regularization);
    }
  }

  // inherit the multiplier
 if (previous != nullptr) {
     multiplier = previous->getDouble("amalgam_multiplier");
     multiplier_vec = previous->getVec("multiplier_vec");
 }
}

/**
* Determines whether an improvement has resulted this
* cluster. Returns 1 in case of an improvement, 0 otherwise. The
* standard-deviation ratio required by the SDR-AVS mechanism is computed
* and returned.
*/
bool hicam::amalgam_t::generationalImprovementForOneCluster(double & st_dev_ratio, const elitist_archive_t & elitist_archive) const
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


bool hicam::amalgam_t::generationalImprovementForOneCluster_vec(vec_t & st_dev_ratio, const elitist_archive_t & elitist_archive) const
{
  
  size_t number_of_improvements = 0;
  st_dev_ratio.reset(fitness_function->number_of_parameters, 0.0);
  
  for (size_t i = 0; i < this->size(); i++)
  {
    if (elitist_archive.solutionHasImproved(*sols[i]))
    {
      number_of_improvements++;
      st_dev_ratio += getSDR_vec(sols[i]->param);
    }
  }
  
  if (number_of_improvements > 0)
    st_dev_ratio /= number_of_improvements;
  
  return(number_of_improvements > 0);
}

// Compute the SDR
//--------------------------------------------------------------------------------------
double hicam::amalgam_t::getSDR(const vec_t & params) const
{
  if (use_univariate)
  {
    vec_t diff(univariate_inverse_cholesky.size(), 0.0);

    for (size_t i = 0; i < univariate_inverse_cholesky.size(); ++i) {
      diff[i] = univariate_inverse_cholesky[i] * (params[i] - mean[i]);
    }

    return diff.infinitynorm();
  }
  else
  {
    if(!use_principal_cholesky_decomposition) {
      return inverse_cholesky.lowerProduct(params - mean).infinitynorm();
    } else {
      return inverse_cholesky.product(params - mean).infinitynorm();
    }
  }
}

hicam::vec_t hicam::amalgam_t::getSDR_vec(const vec_t & params) const
{
  if (use_univariate)
  {
    vec_t diff(univariate_inverse_cholesky.size(), 0.0);
    
    for (size_t i = 0; i < univariate_inverse_cholesky.size(); ++i) {
      diff[i] = univariate_inverse_cholesky[i] * (params[i] - mean[i]);
    }
    
    return diff;
  }
  else
  {
    //permute
    vec_t scaled_diff;
    if(!use_principal_cholesky_decomposition) {
      scaled_diff = inverse_cholesky.lowerProduct(params - mean);
    } else {
      scaled_diff = inverse_cholesky.product(params - mean);
    }
    
    for(size_t i = 0; i < scaled_diff.size(); ++i) {
      scaled_diff[i] = fabs(scaled_diff[i]);
    }
    
    return scaled_diff;
  }
}


//
//
//// Apply the Anticipated Mean Shift (AMS) to the first solutions (not the elites)
////------------------------------------------------------------------------
void hicam::amalgam_t::apply_ams(std::vector<solution_pt> & solutions, const size_t number_of_elites, const size_t number_of_ams_solutions, const double ams_factor, const vec_t & ams_direction, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds) const
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


bool hicam::amalgam_t::updateStrategyParameters(const elitist_archive_t & elitist_archive, size_t & no_improvement_stretch, const size_t maximum_no_improvement_stretch)
{
  bool cluster_failure = adaptDistributionMultiplier_vec(elitist_archive, no_improvement_stretch, maximum_no_improvement_stretch);
  cluster_failure = adaptDistributionMultiplier(elitist_archive, no_improvement_stretch, maximum_no_improvement_stretch);
  
  return cluster_failure;

}



void hicam::amalgam_t::computeParametersForSampling()
{


  if (use_univariate) 
  {
    // UNIVARIATE 'CO'VARIANCE

    // Cholesky decomposition
    choleskyDecomposition_univariate(univariate_covariance, univariate_cholesky);

    if(use_multiplier_vec) {
      for(size_t i = 0; i < multiplier_vec.size(); ++i) {
        univariate_cholesky[i] *= sqrt(multiplier_vec[i]);
      }
    }
    else
    {
      univariate_cholesky *= sqrt(multiplier);
    }
    univariate_inverse_cholesky.resize(univariate_cholesky.size(), 0.0);

    for (size_t i = 0; i < univariate_inverse_cholesky.size(); ++i) {
      univariate_inverse_cholesky[i] = 1.0 / univariate_cholesky[i];
    }

  }
  else 
  {
    
    // todo: multiplier_vec

    std::vector<int> p;// permutation order
    if(!use_principal_cholesky_decomposition) {
      choleskyDecomposition(covariance, cholesky);
      cholesky.multiply(sqrt(multiplier));
      int n = (int)covariance.rows();
      inverse_cholesky.setRaw(matrixLowerTriangularInverse(cholesky.toArray(), n), n, n);
    }
    else
    {
      principalCholeskyDecomposition(covariance, cholesky, p);
      cholesky.multiply(sqrt(multiplier));
      int n = (int)covariance.rows();
      inverse_cholesky.setRaw(matrixLowerTriangularInverse(cholesky.toArray(), n), n, n);
      
      matrix_t temp(n,n);
      
      for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < n; ++j) {
          temp[p[i]][p[j]] = cholesky[i][j];
        }
      }
      
      cholesky = temp;
      
      // permute the inverse cholesky back
      for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < n; ++j) {
          temp[p[i]][p[j]] = inverse_cholesky[i][j];
        }
      }
      
      inverse_cholesky = temp;
    }
  }



}



void hicam::amalgam_t::generateNewSolutions(std::vector<solution_pt> & solutions, size_t number_of_solutions, size_t number_of_ams_solutions, rng_pt & rng)
{

  // Sample new population
  //----------------------------------------------------------------------------------------
  vec_t lower_param_bounds, upper_param_bounds;
  fitness_function->get_param_bounds(lower_param_bounds, upper_param_bounds);
  double delta_ams = 2.0;

  unsigned int out_of_bounds_draws = 0;
  unsigned int samples_drawn_from_normal = 0;

  if (use_univariate)
  {
    out_of_bounds_draws = this->fill_vector_normal_univariate(solutions, number_of_solutions, fitness_function->number_of_parameters, mean, univariate_cholesky, use_boundary_repair, lower_param_bounds, upper_param_bounds, 0, rng);
  }
  else
  {
    // todo permute
    out_of_bounds_draws = this->fill_vector_normal(solutions, number_of_solutions, fitness_function->number_of_parameters, mean, cholesky, use_boundary_repair, lower_param_bounds, upper_param_bounds, 0, rng);
  }


  // apply the AMS
  // 
  if (number_of_ams_solutions > 0 && solutions.size() > 0)
  {

    vec_t old_mean;

    assert(previous != nullptr); // warning: setting AMS but previous is not set.

    old_mean = previous->getVec("amalgam_mean");
 
    
    vec_t ams_direction = mean - old_mean;

    apply_ams(solutions, 0, number_of_ams_solutions, delta_ams*multiplier, ams_direction, lower_param_bounds, upper_param_bounds); // ams, not to elite
  }

  samples_drawn_from_normal = out_of_bounds_draws + (unsigned int) number_of_solutions;
  this->out_of_bounds_sample_ratio = out_of_bounds_draws / (double)(samples_drawn_from_normal);
  
  this->out_of_bounds_sample_ratio_vec.reset(fitness_function->number_of_parameters, out_of_bounds_sample_ratio); // todo, implement this per dimension.
 
}


hicam::vec_t hicam::amalgam_t::getVec(std::string variable_name)
{
  if (variable_name.compare("amalgam_mean") == 0) {
    return mean;
  }

  if (variable_name.compare("multiplier_vec") == 0) {
    return multiplier_vec;
  }
  
  assert(false);
  return vec_t();
}

double hicam::amalgam_t::getDouble(std::string variable_name)
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

size_t hicam::amalgam_t::recommended_popsize(size_t number_of_parameters) const
{
  double selection_fraction = 0.35;

  if(use_univariate)
  {
    return (size_t)std::max((double)((size_t)((2.0 / selection_fraction) + 1)), 10.0*pow((double)number_of_parameters, 0.5));
  }
  else
  {
    return (size_t)std::max((double)((size_t)((2.0 / selection_fraction) + 1)), 17.0 + 3.0*pow((double)number_of_parameters, 1.5));
  }
}

