

/*

CMA-ES

Implementation by S.C. Maree, 2017
s.c.maree[at]amc.uva.nl
smaree.com

*/
#include "population.hpp"
#include "mathfunctions.hpp"
#include "cmaes.hpp"



// init cma-es default parameters
hillvallea::cmaes_t::cmaes_t(const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, fitness_pt fitness_function, rng_pt rng) : optimizer_t(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng)
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

  B.setIdentity(number_of_parameters, number_of_parameters);
  D.setIdentity(number_of_parameters, number_of_parameters);
  BD.setIdentity(number_of_parameters, number_of_parameters);
  covariance.setIdentity(number_of_parameters, number_of_parameters);

}

hillvallea::cmaes_t::~cmaes_t() {};

// Algorithm name
std::string hillvallea::cmaes_t::name() const { return "CMA-ES"; }


hillvallea::optimizer_pt hillvallea::cmaes_t::clone() const
{

  cmaes_pt opt = std::make_shared<cmaes_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng);

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
  opt-> weights = weights;              // weight vector for weighted means
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
  opt->BD = BD;                // product matrix BD such that C = BD * (BD)^(-1)
  opt->B = B;                 // matrix B
  opt->D = D;                 // matrix D    
  opt->mean = mean;
  opt->mean_z = mean_z;
  opt->old_mean = old_mean;             // old sample mean
  opt->covariance = covariance;
  opt->fitnesshist = fitnesshist;
  opt->flginiphase = flginiphase;
  opt->no_improvement_stretch = no_improvement_stretch; // a double as we average it a lot


  return opt;
}

// returns true if any of the termination criteria is satisfied
bool hillvallea::cmaes_t::checkTerminationCondition()
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

void hillvallea::cmaes_t::initialize_from_population(population_pt pop, size_t target_popsize)
{
    this->pop = pop;
    // pop->setOrigin(this);

    size_t selection_size = pop->size(); // pop is the selection (size_t)(selection_fraction * pop->size());
    initStrategyParameters(selection_size);

    no_improvement_stretch = 0;
    number_of_generations = 0;
    // number_of_evaluations = 0;

    pop->weighted_mean_of_selection(mean,weights,selection_size);
  best = solution_t(*pop->first());

    // Init sigma
  /*
   sigma = 0.0;
    vec_t params_min = vec_t(number_of_parameters, 1e308);
    vec_t params_max = vec_t(number_of_parameters, -1e308);
    for (auto sol = pop->sols.begin(); sol != pop->sols.end(); ++sol)
    {
      params_min = min(params_min, (*sol)->param);
      params_max = max(params_max, (*sol)->param);
    }

    for (size_t i = 0; i < number_of_parameters; i++) {
      sigma += (params_max[i] - params_min[i]) / 2.0;
    }
    sigma /= (double)number_of_parameters;
   sigma *= 1.0;
  */
  /*
  if(pop->size() == 1)
  {
    covariance.setIdentity(number_of_parameters, number_of_parameters);
    covariance.multiply(init_univariate_bandwidth*0.01);

    BD.setIdentity(number_of_parameters, number_of_parameters);
    BD.multiply(sqrt(init_univariate_bandwidth*0.01));
  }
  else if(pop->size() < number_of_parameters)
  {
    pop->covariance_univariate(mean, covariance);
    choleskyDecomposition_univariate(covariance, BD);

  } else {
    pop->covariance(mean, covariance);
    choleskyDecomposition(covariance, BD);

  }
  */
  
  B.setIdentity(number_of_parameters, number_of_parameters);
  D.setIdentity(number_of_parameters, number_of_parameters);
  BD.setIdentity(number_of_parameters, number_of_parameters);
  covariance.setIdentity(number_of_parameters, number_of_parameters);
  sigma = 1;
}

size_t hillvallea::cmaes_t::recommended_popsize(const size_t problem_dimension) const
{
  // return (size_t)(17.0 + 3.0*pow((double)problem_dimension, 1.5));
  return (4 + (size_t)(3 * log((double)(problem_dimension))));
  
  // return (size_t)std::max((double)((size_t)((2.0 / selection_fraction) + 1)), 17.0 + 3.0*pow((double)problem_dimension, 1.5));
  // return (size_t)std::max((double)((size_t)((2.0 / selection_fraction) + 1)), 10.0*pow((double)problem_dimension, 0.5));
  
}

// Population size N
// strategy params remain constant over generations (as long as the popsize is). 
void hillvallea::cmaes_t::initStrategyParameters(const size_t selection_size)
{

  // Population & selection size
  mu = selection_size; // (size_t)(population_size * selection_fraction); // selection
  lambda = (size_t) (mu / ((double)selection_fraction)); 
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
  cs = 10.0 / (number_of_parameters + 20.0);
  damps = std::max(1.0, 3.0*mueff / (number_of_parameters + 10.0)) / cs + 1.0;
  // redefined such that maximum_number_of_evaluations = inf;
  // assert(ccov < 1);

  // chiN
  chiN = sqrt((double)(number_of_parameters))*(1.0 - 1.0 / (4.0*number_of_parameters) + 1.0 / (21.0*number_of_parameters*number_of_parameters));
  
}



void hillvallea::cmaes_t::estimate_sample_parameters()
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
  estimate_covariance(pc, ps, fitnesshist, sigma, covariance, flginiphase, B, D, BD);

}



void hillvallea::cmaes_t::estimate_covariance(vec_t & pc, vec_t & ps, vec_t & fitnesshist, double & sigma, matrix_t & covariance, bool & flginiphase, matrix_t & B, matrix_t & D, matrix_t & BD) const
{

  // updates the evolution path pc, ps
  // sigma


  if (number_of_generations > 1)
  {
    pc = (1.0 - cc)*pc + (sqrt(cc*(2.0 - cc)*mueff) / sigma) * (mean - old_mean);
    ps = (1.0 - cs)*ps + (sqrt(cs*(2.0 - cs)*mueff)) * (B * mean_z);
  }

  fitnesshist[2] = fitnesshist[1];
  fitnesshist[1] = fitnesshist[0];
  fitnesshist[0] = pop->sols[0]->f;

  double ps_AvgSquaredNorm = ps.squaredNorm() / number_of_parameters;

  if (ps_AvgSquaredNorm > (1.5 + 10.0*sqrt(2.0 / number_of_parameters)) && (fitnesshist[1] < fitnesshist[0]) && (fitnesshist[2] < fitnesshist[1])) {
    ps *= sqrt((1.0 +std::max(0.0, log(ps_AvgSquaredNorm))) / ps_AvgSquaredNorm);
    // std::cout << "ps correction applied" << std::endl;
  }

  if (!flginiphase)
  {

    // init WG matrix by zeros
    matrix_t wgMatrix = matrix_t(number_of_parameters, number_of_parameters, 0.0);
    double value = (1.0 - 1.0 / mucov)*pow(sigma, -2.0);

    // compute the wgMatrix
    for (size_t i = 0; i < number_of_parameters; i++) {
      for (size_t j = i; j < number_of_parameters; j++) {
        for (size_t k = 0; k < pop->size(); ++k) {
          wgMatrix[i][j] += value * (pop->sols[k]->param[i] - old_mean[i])*(pop->sols[k]->param[j] - old_mean[j])*weights[k];
        }
        wgMatrix[j][i] = wgMatrix[i][j];
      }
    }

    // compute the covariance matrix
    for (size_t i = 0; i < number_of_parameters; i++) {
      for (size_t j = i; j < number_of_parameters; j++) {
        covariance[i][j] = (1.0 - ccov) * covariance[i][j] + ( ccov / mucov )*pc[i]*pc[j] + ccov * wgMatrix[i][j];
        covariance[j][i] = covariance[i][j];
        // assert(isfinite(covariance[i][j]));
      }
    }

  }

  sigma *= exp((ps.norm() / chiN - 1.0) / damps);

  // only once every so-many generations, do a eigenvalue decomposition to update the covariance
  if ((!flginiphase) && (ccov > 0) && (fmod(number_of_generations * 2, 1.0 / ccov / number_of_parameters / 3.0) < 1.0))
  {

    // do an eigenvalue decomposition of the covariance matrix
    eigenDecomposition(covariance, D, B);

    // find the min and max diagonal elements of D
    double maxdiag = D[0][0];
    double mindiag = D[0][0];
    for (size_t i = 0; i < number_of_parameters; i++)
    {

      // assert(isfinite(D[i][i]) && D[i][i] > 0);

      if (D[i][i] > maxdiag) {
        maxdiag = D[i][i];
      }
      if (D[i][i] < mindiag) {
        mindiag = D[i][i];
      }
    }

    // this is some sort of regularization
    // check if the values are not too different from each other
    if (maxdiag > 1e14*mindiag)
    {
      double value = maxdiag / 1e14 - mindiag; // value is greater than zero. 
      for (size_t i = 0; i < number_of_parameters; i++)
      {
        covariance[i][i] += value;
        D[i][i] += value;
        // std::cout << "D min/max correction applied" << std::endl;
      }
    }

    // take the square root of D. 
    // assert(isfinite(D[0][0]));
    D.diagonalMatrixSquareRoot();
    // assert(isfinite(D[0][0]));

    // Compute BD such that C = BD * (BD)^(-1)
    BD = B*D;
    // assert(isfinite(BD[0][0]));
  }

  // this checks if any of the diagonal entries equals zero
  int any = 0;
  for (size_t i = 0; i < number_of_parameters; i++) {
    if ((sigma*sqrt(covariance[i][i]) < 0) || (mean[i] == (mean[i] + 0.2*sigma*sqrt(covariance[i][i]))))
    {
      any = 1;
      break;
    }
  }

  if (any) 
  {
    for (size_t i = 0; i < number_of_parameters; i++)
    {
      if (mean[i] == (mean[i] + 0.2*sigma*sqrt(covariance[i][i]))) {
        covariance[i][i] += ccov*covariance[i][i];
        // assert(isfinite(covariance[i][i]));
        // std::cout << "ccov correction applied" << std::endl;
      }
    }
    sigma *= exp(0.05 + 1.0 / damps);
  }


  // According to Anton, this code is useless.
  // indeed it is weird. It does something like checking if a value in the BD matrix equals 0.
  // Then, it damps the sigma.
  // also, it damps the sigma if the fitness landscape is flat within 25% percent of the best solutions.
  // CMAES says : Adjust step size in case of (numerical) precision problem 
  int all = 1;
  for (int i = 0; i < number_of_parameters; i++)
  {
    // (number_of_evaluations/number_of_populations)/population_size) = number_of_generations
    if (mean[i] != (mean[i] + 0.1*sigma*BD[i][(int)(number_of_generations-1) % number_of_parameters]))
    {
      all = 0;
      break;
    }
  }
  // changed this such that fmin(lambda/4,mu-1) >= 1
  if( all || ( pop->size() >= 2 && (pop->sols[0]->f == pop->sols[(size_t)std::max(1.0,std::min(lambda/4.0,(mu-1.0)))]->f)) ) 
  {
    sigma *= exp( 0.2 + 1.0/damps );
    // std::cout << "sigma correction applied" << std::endl;
  }

  if (flginiphase && ((number_of_generations -1) > std::min(1.0 / cs, 1.0 + number_of_parameters / mucov)))
  {
    double value = ps_AvgSquaredNorm / (1.0 - pow(1.0 - cs, (double)number_of_generations));
    if (value < 1.05) {
      flginiphase = 0;
      // std::cout << "Number of generations" << number_of_generations << " : ini off" << std::endl;
    }
  }
}


void hillvallea::cmaes_t::generation(size_t sample_size, int & number_of_evaluations)
{
  pop->sort_on_fitness();
  pop->truncation_percentage(*pop, selection_fraction);
  estimate_sample_parameters();
  number_of_evaluations += sample_new_population(sample_size);
  // pop->truncation_percentage(*pop, selection_fraction);
  average_fitness_history.push_back(pop->average_fitness());
}


size_t hillvallea::cmaes_t::sample_new_population(const size_t sample_size)
{
  // this is a new generation!
  //---------------------------------------------------------------------------------------
  number_of_generations++;
  // previous_average_fitness = average_fitness;
  // Sample new population
  //---------------------------------------------------------------------------------------
  matrix_t BDs = BD;
  BDs.multiply(sigma);
  pop->fill_normal(sample_size, number_of_parameters, mean, BDs, lower_param_bounds, upper_param_bounds, 1, rng);


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

