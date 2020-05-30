#define _CRT_SECURE_NO_WARNINGS

/*
 
 HICAM Multi-objective
 
 By S.C. Maree, 2018
 s.c.maree[at]amc.uva.nl
 smaree.com
 
 */

#include "population.h"
#include "mathfunctions.h"
#include "fitness.h"
#include "cluster.h"

namespace hicam
{

  // Constructor
  population_t::population_t() {
    
    this->elitist_archive = nullptr;
    this->number = 1000;
    this->new_elites_added = 0;
    
  }
  
  // Copy constructor
  population_t::population_t(const population_t & other)
  {
    this->sols = other.sols;
    this->best = other.best;
    this->worst = other.worst;
    this->elitist_archive = other.elitist_archive;
    this->elites = other.elites;
    this->number = other.number;
    this->clusters = other.clusters;
    this->previous = other.previous;
    this->new_elites_added = other.new_elites_added;
  }

  // Destructor
  population_t::~population_t() { }

  // dimensions
  //------------------------------------------
  size_t population_t::size() const { return sols.size(); }    // pop.size()
  size_t population_t::popsize() const { return sols.size(); } // popsize(), sometimes, this reads nicer.
  size_t population_t::problem_size() const { return sols[0]->number_of_parameters(); }


  // Population mean
  void population_t::compute_mean(vec_t & mean) const
  {
    // Compute the sample mean
    //-------------------------------------------
    // set the mean to zero.
    mean.resize(problem_size());
    mean.fill(0);
    size_t count = 0;
    for (size_t i = 0 ; i < sols.size(); ++i) {
      if(sols[i] != nullptr) {
        mean += sols[i]->param;
        count++;
      }
    }
    
    mean /= (double) count;
    
  }

  // Population mean
  void population_t::compute_mean_of_selection(vec_t & mean, size_t selection_size) const
  {
    if(selection_size > sols.size()) {
      selection_size = sols.size();
    }
    
    // Compute the sample mean
    //-------------------------------------------
    // set the mean to zero.
    mean.resize(problem_size());
    mean.fill(0);
    size_t count = 0;
    for (size_t i = 0 ; i < selection_size; ++i) {
      if(sols[i] != nullptr) {
        mean += sols[i]->param;
        count++;
      }
    }
    
    mean /= (double) count;
    
  }
  
  // population covariance
  void population_t::compute_covariance(const vec_t & mean, matrix_t & covariance, bool enable_regularization) const
  {
    // Compute the sample covariance
    // use the maximum likelihood estimate (see e.g. wikipedia)
    //-------------------------------------------
    covariance.reset(problem_size(),problem_size(), 0.0);
    
    /* First do the maximum-likelihood estimate from data */
    for(size_t i = 0; i < problem_size(); i++ )
    {
      for(size_t j = i; j < problem_size(); j++ )
      {
        for(size_t k = 0; k < sols.size(); k++ ) {
          covariance[i][j] += (sols[k]->param[i]-mean[i])*(sols[k]->param[j]-mean[j]);
        }
        
        covariance[i][j] /= (double) sols.size();
      }
    }
    
    for(size_t i = 0; i < problem_size(); i++ ) {
      for(size_t j = 0; j < i; j++ ) {
        covariance[i][j] = covariance[j][i];
      }
    }
    
    if(enable_regularization)
    {
      // regularization for small populations
      double number_of_samples = (double) sols.size();
      size_t n = problem_size();
      //if(number_of_samples < n + 1)
      {
        //double meanvar = 0.0;
        //for(size_t i = 0; i < n; ++i) {
        //  meanvar += covariance[i][i];
        //}
        //meanvar /= (double) n;

        double phi = 0.0;

        // y = x.^2
        // phiMat = y'*y/t-sample.^2
        // phi = sum(sum(phiMat))
        matrix_t squared_cov(n,n,0.0);
        double temp;
        for(size_t i = 0; i < n; ++i)
        {
          for(size_t j = 0; j < n; ++j)
          {
            squared_cov[i][j] = 0.0;
            
            for(size_t k = 0; k < sols.size(); ++k)
            {
              temp = (sols[k]->param[i]-mean[i])*(sols[k]->param[j]-mean[j]);
              squared_cov[i][j] += temp*temp;
            }
            squared_cov[i][j] /= number_of_samples;
          }
        }

        // this can be implemented faster by considering only half this matrix,
        // and we dont need to store square_cov actually.
        for(size_t i = 0; i < n; ++i)
        {
          for(size_t j = 0; j < n; ++j)
          {
            phi += squared_cov[i][j] - covariance[i][j] * covariance[i][j];
          }
        }

        // Frobenius norm, i.e.,
        // gamma = norm(sample - prior,'fro')^2;
        double gamma = 0.0;

        for(size_t i = 0; i < n; ++i)
        {
          for(size_t j = 0; j < n; ++j)
          {
            // temp = fabs(covariance[i][j] - (( i == j ) ? meanvar : 0.0));
            temp = fabs(covariance[i][j] - (( i == j ) ? covariance[i][i] : 0.0));
            gamma += temp*temp;
          }
        }

        double kappa = phi/gamma;
        double shrinkage = std::max(0.0,std::min(1.0,kappa/number_of_samples));

        //if(shrinkage > 0.0 && shrinkage < 1.0)
        {
          //std::cout << "Using a shrinkage factor of " << shrinkage << "." << std::endl;
          
          for(size_t i = 0; i < n; ++i) {
            for(size_t j = 0; j < n; ++j) {
              // covariance[i][j] = (1.0 - shrinkage) * covariance[i][j] + ((i==j) ? shrinkage*meanvar : 0.0);
              covariance[i][j] = (1.0 - shrinkage) * covariance[i][j] + ((i==j) ? shrinkage*covariance[i][i] : 0.0);
            }
          }
        }
        //else
        //{
        //  std::cout << "Shrinking not applied, kappa/t = " << kappa/number_of_samples << "." << std::endl;
        //}
      }
    } // end regularization
    
  }
  
  // population covariance
  void population_t::compute_covariance(const vec_t & mean, matrix_t & covariance) const
  {
    compute_covariance(mean, covariance, false);
  }

  // population covariance
  void population_t::compute_covariance_univariate(const vec_t & mean, vec_t & univariate_covariance) const
  {
    // Compute the sample covariance
    // use the maximum likelihood estimate (see e.g. wikipedia)
    //-------------------------------------------
    univariate_covariance.reset(problem_size(), 0.0);

    /* First do the maximum-likelihood estimate from data */
    for (size_t i = 0; i < problem_size(); i++)
    {
      for (size_t k = 0; k < sols.size(); k++) {
        univariate_covariance[i] += (sols[k]->param[i] - mean[i])*(sols[k]->param[i] - mean[i]);
      }

      univariate_covariance[i] /= (double)sols.size();
    }
  }
  

  // evaluate the population
  //-------------------------------------------------------------------------------------
  void population_t::evaluate(fitness_pt fitness_function, const size_t skip_number_of_elites, unsigned int & number_of_evaluations)
  {
    
    for(size_t i = skip_number_of_elites; i < sols.size(); ++i) {
      fitness_function->evaluate(sols[i]);
      number_of_evaluations++;
    }
    
  }

  void hicam::population_t::compute_fitness_ranks()
  {

    if (sols.size() == 0) {
      return;
    }

    size_t number_of_objectives = sols[0]->number_of_objectives();

    // set objective values to 1e308 if they are NaN.
    for (size_t i = 0; i < sols.size(); i++)
    {
      bool is_illegal = false;
      for (size_t j = 0; j < number_of_objectives; j++)
      {

        if (isnan(sols[i]->obj[j]))
        {
          is_illegal = true;
          break;
        }

      }

      if (isnan(sols[i]->constraint)) {
        is_illegal = true;
      }

      if (is_illegal)
      {
        for (size_t j = 0; j < number_of_objectives; j++) {
          sols[i]->obj[j] = 1e+308;
        }
        sols[i]->constraint = 1e+308;
      }
    }

    // The domination matrix stores for each solution i whether it dominates solution j, i.e. domination[i][j] = 1
    std::vector<std::vector<int>> domination_matrix(sols.size());
    for(size_t i = 0; i < sols.size(); ++i) {
      domination_matrix[i].resize(sols.size(), 0);
    }
    
    std::vector<int> being_dominated_count(sols.size(), 0);

    for (size_t i = 0; i < sols.size(); i++)
    {
      for (size_t j = 0; j < sols.size(); j++)
      {
        if (i != j)
        {
          if (sols[i]->better_than(*sols[j]))
          {
            assert(domination_matrix[i][j] == 0);
            domination_matrix[i][j] = 1;
            assert(domination_matrix[j][i] == 0);
            
            being_dominated_count[j]++;
          }
        }
      }
    }
    
    /* Compute ranks from the domination matrix */
    size_t rank = 0;
    size_t number_of_solutions_ranked = 0;
    std::vector<size_t> indices_in_this_rank(sols.size(), 0);
    while (number_of_solutions_ranked < sols.size())
    {
      size_t k = 0;
      for (size_t i = 0; i < sols.size(); i++)
      {
        if (being_dominated_count[i] == 0)
        {
          sols[i]->rank = rank;
          indices_in_this_rank[k] = i;
          k++;
          being_dominated_count[i]--;
          number_of_solutions_ranked++;
        }
      }

      // for all other indices in this rank
      size_t new_0 = false;
      for (size_t i = 0; i < k; i++)
      {
        for (size_t j = 0; j < sols.size(); j++)
        {
          if (domination_matrix[indices_in_this_rank[i]][j] > 0) { // or == 1
            being_dominated_count[j]--;
            if(being_dominated_count[j] == 0)
            {
              new_0 = true;
            }
          }
        }
      }

      // circles can exist between three or more solutions because we consider better with a margin
      if(number_of_solutions_ranked < sols.size() && !new_0)
      {
        std::vector<size_t> remaining;
        for(size_t i = 0; i < being_dominated_count.size(); ++i)
        {
          if(being_dominated_count[i] >= 0 ) {
            remaining.push_back(i);
          }
        }
        
        matrix_t d(remaining.size(),remaining.size(),0);
        
        for(size_t i = 0; i < remaining.size(); ++i)
        {
          for(size_t j = 0; j < remaining.size(); ++j)
          {
            d[i][j] = domination_matrix[remaining[i]][remaining[j]];
          }
          
          d[i][i] = 1;
        }
        
        for (size_t i = 0; pow(2,i) < remaining.size() - 1; ++i) {
          d = d * d;
        }
        
        // this does assume that there is a single cycle of (any length) of non-dominated solutions (i.e., with rank1.)
        // if there are multiple non-dominated cycles, this fails.
        // if there are
        // there are definitely better algorithms around to do this.
        bool found_a_circle = false;
        for(size_t i = 0; i < remaining.size(); ++i)
        {
          bool all1 = true;
          for(size_t j =0; j < remaining.size(); ++j)
          {
            if (d[i][j] == 0) {
              all1 = false;
              break;
            }
          }
          
          if(all1) {
            being_dominated_count[remaining[i]]--;
            found_a_circle = true;
          }
          
        }
        
        // the above doesn't always work, just rank everything remaining the same
        if(!found_a_circle)
        {
          for(size_t i = 0; i < remaining.size(); ++i)
          {
            being_dominated_count[remaining[i]] = 0;
          }
        }
        
        rank--;
      }
      
      rank++;
      
      // a failsafe, which should never happen. idk why it does now?
      if (rank > sols.size()) {
        std::cout << "still a rank eror??";
        assert(false);
        break;
      }
    }
    
  }

  void population_t::sort_on_ranks()
  {
    std::sort(sols.begin(), sols.end(), solution_t::better_rank_via_pointers);
  }

  void population_t::getSingleObjectiveRanks(std::vector<size_t> & fitness_ranks, size_t objective_number) const
  {
    
    fitness_ranks.resize(sols.size());
    for(size_t i = 0; i < fitness_ranks.size(); ++i) {
      fitness_ranks[i] = i;
    }
    
    const std::vector<solution_pt>& sols = this->sols;
    
    auto SOsort = [objective_number, sols](const size_t & r1, const size_t & r2) -> bool {
      return sols[r1]->better_than_per_objective(*sols[r2], objective_number);
    };
    
    std::sort(fitness_ranks.begin(), fitness_ranks.end(), SOsort);
  }
  

  // Fill the given population by uniform initialization in the range [min,max),
  // for all dimensions equal
  //----------------------------------------------------------------------------------------
  void population_t::fill_uniform(size_t sample_size, size_t problem_size, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng)
  {
    fill_uniform(sample_size, problem_size, lower_param_range, upper_param_range, 0, rng);
  }

  void population_t::fill_uniform(size_t sample_size, size_t problem_size, const vec_t & lower_param_range, const vec_t & upper_param_range, size_t number_of_elites, rng_pt rng)
  {
    
    // resize the solutions vector.
    sols.resize(sample_size);
    
    // sample solutions and evaluate them
    for(size_t i = number_of_elites; i < sols.size(); ++i)
    {
      
      // if the solution is not yet initialized, do it now.
      if (sols[i] == nullptr)
      {
        solution_pt sol = std::make_shared<solution_t>(problem_size);
        sols[i] = sol;
      }
      
      // sample a new solution ...
      sample_uniform(sols[i]->param, problem_size,lower_param_range,upper_param_range,rng);
      
    }
  }

  
  // Fill the given population by normal sampling
  //-------------------------------------------------------------------------------------------------------------------------------
  unsigned int population_t::fill_normal_univariate(size_t sample_size, size_t problem_size, const vec_t & mean, const vec_t & univariate_cholesky, bool use_boundary_repair, const vec_t & lower_param_range, const vec_t & upper_param_range, size_t number_of_elites, rng_pt rng)
  {

    unsigned int number_of_samples = fill_vector_normal_univariate(sols, sample_size, problem_size, mean, univariate_cholesky, use_boundary_repair, lower_param_range, upper_param_range, number_of_elites, rng);
    return number_of_samples;
  }

  unsigned int population_t::fill_vector_normal_univariate(std::vector<solution_pt> & solutions, size_t sample_size, size_t problem_size, const vec_t & mean, const vec_t & univariate_cholesky, bool use_boundary_repair, const vec_t & lower_param_range, const vec_t & upper_param_range, size_t number_of_elites, rng_pt rng) const
  {

    // Resize the population vector
    //--------------------------------------------
    solutions.resize(sample_size);

    unsigned int number_of_samples = 0;

    // for each sol in the pop, sample.
    for (size_t i = 0; i < solutions.size(); ++i)
    {

      // save the elite (if it is defined)
      if (i < number_of_elites && solutions[i] != nullptr)
        continue;

      // if the solution is not yet initialized, do it now.
      if (solutions[i] == nullptr)
      {
        solution_pt sol = std::make_shared<solution_t>(problem_size);
        solutions[i] = sol;
      }


      number_of_samples += sample_normal_univariate(solutions[i]->param, problem_size, mean, univariate_cholesky, use_boundary_repair, lower_param_range, upper_param_range, rng);

    }

    return number_of_samples;
  }


  unsigned int population_t::fill_vector_normal(std::vector<solution_pt> & solutions, size_t sample_size, size_t problem_size, const vec_t & mean, const matrix_t & cholesky, bool use_boundary_repair, const vec_t & lower_param_range, const vec_t & upper_param_range, size_t number_of_elites, rng_pt rng) const
  {

    // Resize the population vector
    //--------------------------------------------
    solutions.resize(sample_size);

    unsigned int number_of_samples = 0;

    // for each sol in the pop, sample.
    for (size_t i = 0; i < solutions.size(); ++i)
    {

      // save the elite (if it is defined)
      if (i < number_of_elites && solutions[i] != nullptr)
        continue;

      // if the solution is not yet initialized, do it now.
      if (solutions[i] == nullptr)
      {
        solution_pt sol = std::make_shared<solution_t>(problem_size);
        solutions[i] = sol;
      }


      number_of_samples += sample_normal(solutions[i]->param, problem_size, mean, cholesky, use_boundary_repair, lower_param_range, upper_param_range, rng);
      
      for(size_t j = 0; j < solutions[i]->param.size(); ++j) {
        if(isnan(solutions[i]->param[j])) {
          std::cout << "NaN param sampled." << std::endl;
        }
      }

    }

    return number_of_samples;
  }
  
  // Fill equally distributed sample points using Wessing's Maximin.
  //----------------------------------------------------------------------------------------
  void population_t::fill_maximin(size_t sample_size, size_t problem_size, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng)
  {
    fill_maximin(sample_size, problem_size, lower_param_range, upper_param_range, 0, rng);
  }
  
  void population_t::fill_maximin(size_t sample_size, size_t problem_size, const vec_t & lower_param_range, const vec_t & upper_param_range, const size_t number_of_elites, rng_pt rng)
  {
    
    // resize the solutions vector.
    assert(sample_size >= number_of_elites);
    
    sols.resize(sample_size);
    std::vector<std::shared_ptr<vec_t>> samples, existing_samples;

    vec_t range = upper_param_range - lower_param_range;
    bool good_sample = true;
    for (size_t i = 0; i < number_of_elites; ++i)
    {
      
      if (sols[i] == nullptr) {
        continue;
      }
      
      std::shared_ptr<vec_t> scaled_params = std::make_shared<vec_t>(problem_size);
      
      for(size_t j = 0; j < problem_size; ++j)
      {
        (*scaled_params)[j] = (sols[i]->param[j] / range[j]) - lower_param_range[j];
        
        if( (*scaled_params)[j] < 0.0 || (*scaled_params)[j] > 1.0) {
          good_sample = false;
          break;
        }
      }
      
      // the init range doesn't have to be the boundary ranges, so samples can exist outside of
      // [0,1]^n. These are left out of the sampling to reduce complexity
      if(good_sample) {
        existing_samples.push_back(scaled_params);
      }
      
    }
    
    size_t new_sample_size = sample_size - number_of_elites;
    
    maximin_reconstruction(new_sample_size, problem_size, samples, existing_samples, rng);
    

    // sample  solutions and evaluate them
    for (size_t i = number_of_elites; i < sample_size; ++i)
    {
      
      // sample a new solution ...
      sols[i] = std::make_shared<solution_t>(problem_size);
      
      for (size_t j = 0; j < problem_size; ++j){
        sols[i]->param[j] = range[j] * (*samples[i-number_of_elites])[j] + lower_param_range[j];
      }
      
    }
  }
  
  


  void population_t::objectiveRanges(vec_t & objective_ranges)
  {

    if (size() == 0)
      return;

    size_t start_index = 0;
    for (start_index = 0; start_index < sols.size(); ++start_index)
    {
      if (sols[start_index] != nullptr) {
        break;
      }
    }

    size_t number_of_objectives = sols[start_index]->number_of_objectives();
    bool use_lex = sols[start_index]->use_lex;
    
    
    // hack (hardcoded 1e300, do i need it?)
    // the naming best/worst is kinda off, its better to say lowest/highest.
    worst = solution_t(*sols[start_index]);
    best = solution_t(*sols[start_index]);
    
    for (size_t i = start_index + 1; i < size(); i++)
    {

      if (sols[i] == nullptr) {
        continue;
      }

      for (size_t j = 0; j < number_of_objectives; j++)
      {

        if (sols[i]->better_than_unconstraint_per_objective(best, j))
        {
          best.obj[j] = sols[i]->obj[j];
          
          if(use_lex) {
            best.lex_obj[j] = sols[i]->lex_obj[j];
          }
        }

        if (!sols[i]->better_than_unconstraint_per_objective(worst, j))
        {
          worst.obj[j] = sols[i]->obj[j];
          
          if(use_lex) {
            worst.lex_obj[j] = sols[i]->lex_obj[j];
          }
        }
      }
    }

    objective_ranges.resize(number_of_objectives);
    for(size_t i = 0; i < number_of_objectives; ++i) {
      objective_ranges[i] = worst.obj[i] - best.obj[i];
    }
    
  }

  void population_t::computeParameterRanges(vec_t & parameter_ranges) const
  {
    
    if (size() == 0)
      return;
    
    size_t start_index = 0;
    for (start_index = 0; start_index < sols.size(); ++start_index)
    {
      if (sols[start_index] != nullptr) {
        break;
      }
    }
    
    size_t number_of_parameters = sols[start_index]->number_of_parameters();
    
    vec_t largest_param(number_of_parameters, -1e+308);
    vec_t smallest_param(number_of_parameters, 1e+308);
    
    for (size_t i = start_index; i < size(); i++)
    {
      
      if (sols[i] == nullptr) {
        continue;
      }
      
      for (size_t j = 0; j < number_of_parameters; j++)
      {
        
        if (sols[i]->param[j] < smallest_param[j]) {
          smallest_param[j] = sols[i]->param[j];
        }
        
        if (sols[i]->param[j] > largest_param[j]) {
          largest_param[j] = sols[i]->param[j];
        }
      }
    }
    
    parameter_ranges = largest_param - smallest_param;
    
  }

  // Truncation selection (selection percentage)
  // select the selection_percentage*population_size best individuals in the population
  //-------------------------------------------------------------------------------------
  void population_t::truncation_percentage(population_t & selection, double selection_percentage) const
  {
    
    // Get the parent population size to compute the selection fraction.
    // Then, call truncation by number.
    truncation_size(selection,(size_t) (selection_percentage*sols.size()));
    
  }
  
  
  // Truncation selection (selection size)
  // select the selection_size best individuals in the population
  // the population is already sorted.
  //-------------------------------------------------------------------------------------
  void population_t::truncation_size(population_t & selection, size_t selection_size) const
  {
    
    selection.sols.resize(selection_size);
    
    // copy the pointers from the parents to the selection
    std::copy(sols.begin(),sols.begin() + selection_size,selection.sols.begin());
    
  }
  
  
  void population_t::truncation_percentage(population_t & selection, double selection_percentage, population_t & not_selected_solutions) const
  {
    
    // Get the parent population size to compute the selection fraction.
    // Then, call truncation by number.
    truncation_size(selection,(size_t) (selection_percentage*sols.size()), not_selected_solutions);
    
  }
  
  void population_t::truncation_size(population_t & selection, size_t selection_size, population_t & not_selected_solutions) const
  {
    
    selection.sols.resize(selection_size);
    size_t left_over_size = (size_t) std::max(0,(int)(sols.size() - selection_size));
    
    not_selected_solutions.sols.resize(left_over_size);
    
    // copy the pointers from the parents to the selection
    std::copy(sols.begin(),sols.begin() + selection_size,selection.sols.begin());
    
    // copy the pointers from the parents to the selection
    std::copy(sols.end()-left_over_size,sols.end(),not_selected_solutions.sols.begin());
    
  }
  

  void population_t::makeSelection(size_t selection_size, const hicam::vec_t &objective_ranges, rng_pt &rng)
  {
    makeSelection(selection_size,objective_ranges, true, rng);
  }
  
  void population_t::makeSelection(size_t selection_size, const vec_t & objective_ranges, bool use_objective_distances, rng_pt & rng)
  {

    // sort the population based on (already computed) ranks
    this->sort_on_ranks();
    
    if(selection_size == 0) {
      return;
    }

    // duplicate ranks may (will) occor. Add all solutions to the population
    // based on the last rank that would appear in the population,
    // and select the remaining solutions from the last_selected_rank
    // based on a greedy heuristic
    // size_t selection_size = (size_t)(tau*population->size());
    size_t last_selected_rank = this->sols[selection_size - 1]->rank;

    std::vector<solution_pt> selection;
    std::vector<solution_pt> non_selected_solutions;
    selection.reserve(selection_size);

    std::vector<solution_pt> selection_candidates;
    size_t last_index = 0;

    // add solutions to the selection and selection_candidates
    for (size_t i = 0; i < this->size(); ++i)
    {

      if (this->sols[i]->rank < last_selected_rank) {
        selection.push_back(this->sols[i]);
        continue;
      }

      if (this->sols[i]->rank == last_selected_rank) {
        selection_candidates.push_back(this->sols[i]);
        continue;
      }

      if (this->sols[i]->rank > last_selected_rank) {
        last_index = i;
        break;
      }
    }


    // perform greedy selection for the remainder
    size_t number_to_select = selection_size - selection.size();

    if (number_to_select > 0) {
      if(use_objective_distances) {
        selectSolutionsBasedOnObjectiveDiversity(selection_candidates, number_to_select, selection, objective_ranges, non_selected_solutions, rng);
      }
      else
      {
        selectSolutionsBasedOnParameterDiversity(selection_candidates, number_to_select, selection, non_selected_solutions, rng);
      }
    }

    for (size_t i = last_index; i < this->size(); ++i) {
      non_selected_solutions.push_back(this->sols[i]);
    }


    // replace the population by the selection
    // they have the same size, but are sorted such that the first tau*N solutions are the selection.
    this->sols = selection;
    this->addSolutions(non_selected_solutions);

  }
  
  // Add the solutions of another population to this one
  //---------------------------------------------------------------------
  void population_t::addSolutions(const population_t & pop)
  {
    this->addSolutions(pop.sols);
  }

  void population_t::addSolutions(const std::vector<solution_pt> & sols)
  {
    this->sols.insert(this->sols.end(), sols.begin(), sols.end());
  }
 
  void population_t::addSolution(const solution_pt & sol)
  {
    this->sols.push_back(sol);
  }

  // Add the solutions of another population to this one
  //---------------------------------------------------------------------
  void population_t::addCopyOfSolutions(const population_t & pop)
  {
    this->addCopyOfSolutions(pop.sols);
  }

  void population_t::addCopyOfSolutions(const std::vector<solution_pt> & sols)
  {
    for (size_t i = 0; i < sols.size(); ++i)
    {
      if(sols[i] != nullptr) {
        this->addCopyOfSolution(*sols[i]);
      }
    }
  }

  void population_t::addCopyOfSolution(const solution_t & sol)
  {
    solution_pt copy = std::make_shared<solution_t>(sol);
    this->sols.push_back(copy);
  }

  void population_t::setPopulationNumber(int population_number)
  {
    setPopulationNumber(population_number, 0);
  }
  
  void population_t::setPopulationNumber(int population_number, size_t number_of_elites)
  {
    for(size_t i = number_of_elites; i < sols.size(); ++i) {
      sols[i]->population_number = population_number;
    }
  }
  
  void population_t::setClusterNumber(int cluster_number)
  {
    setClusterNumber(cluster_number, 0);
  }
  
  void population_t::setClusterNumber(int cluster_number, size_t number_of_elites)
  {
    for(size_t i = number_of_elites; i < sols.size(); ++i) {
      sols[i]->cluster_number = cluster_number;
    }
  }
  
  
  void population_t::collectSolutions(const std::vector<population_pt> & subpopulations)
  {
    sols.clear();
    for(size_t j = 0; j < subpopulations.size(); ++j)
    {
      addSolutions(*subpopulations[j]);
    }
  }
  

  
  // Sort the population such that best = first
  //-------------------------------------------------------------------------------------
  void population_t::sort_on_fitness() {
    std::sort(sols.begin(),sols.end(),solution_t::better_solution_via_pointers);
  }

  void hicam::population_t::removeSolutionNullptrs()
  {
    std::vector<solution_pt> new_sols;
    new_sols.reserve(sols.size());
    
    for(size_t i = 0; i < sols.size(); ++i)
    {
      if(sols[i] != nullptr) {
        new_sols.push_back(sols[i]);
      }
    }
    
    sols = new_sols;
  }

  
  
  // Population Statistics
  //--------------------------------------------------------------------
  solution_pt population_t::first() const {

    if (sols.size() < 1)
      return nullptr;
    else
      return sols[0];
  }
  
  solution_pt population_t::last() const {
    if (sols.size() < 1)
      return nullptr;
    else
      return sols[sols.size()-1];
  }


  // Average fitness of the population
  //-------------------------------------------
  void population_t::average_fitness(vec_t & avg_fitness) const
  {

    avg_fitness.clear();

    if (sols.size() == 0) {
      return;
    }

    avg_fitness.resize(sols[0]->number_of_objectives(), 0.0);
    
    for (auto sol = sols.begin(); sol != sols.end(); ++sol)
    {
      avg_fitness += (*sol)->obj;
      // assert(isfinite(average_fitness));
      
    }
    
    avg_fitness /= size();

  }
  
  
  void population_t::fitness_variance(vec_t & mean, vec_t & var) const
  {

    average_fitness(mean);

    var.clear();
    var.resize(sols[0]->number_of_objectives(), 0.0);


    for (auto sol = sols.begin(); sol != sols.end(); ++sol)
    {
      for (size_t i = 0; i < mean.size(); ++i) {
        var[i] += ((*sol)->obj[i] - mean[i])*((*sol)->obj[i] - mean[i]);
      }
    }

    var /= (sols.size());

  }

  void population_t::fitness_std(vec_t & mean, vec_t & std) const
  {

    fitness_variance(mean, std);

    for (size_t i = 0; i < mean.size(); ++i)
    {
      if (mean[i] <= 0)
        std[i] = sqrt(std[i]) / fabs(mean[i]);
    }

  }

  
  double population_t::average_constraint() const
  {
    double mean = 0;
    for (auto sol = sols.begin(); sol != sols.end(); ++sol)
      mean += (*sol)->constraint;
    
    mean /= sols.size();
    
    return mean;
  }
  
  
  double population_t::constraint_variance() const
  {
    double mean = 0;
    double variance = 0;
    
    for (auto sol = sols.begin(); sol != sols.end(); ++sol)
      mean += (*sol)->constraint;
    
    mean /= (sols.size());
    
    for (auto sol = sols.begin(); sol != sols.end(); ++sol)
      variance += ((*sol)->constraint - mean)*((*sol)->constraint - mean);
    
    variance /= (sols.size());
    
    return variance;
  }
  
  
  double population_t::constraint_of_first() const {
    return sols[0]->constraint;
  }
  
  
  double population_t::constraint_of_last() const {
    return sols[0]->constraint;
  }



  // two reference points.
  double population_t::compute2DHyperVolume(double max_0, double max_1) const
  {

    size_t n = sols.size();

    if (n == 0) {
      return 0.0;
    }

    vec_t obj_0;
    std::vector<size_t> sol_index;
    
    obj_0.reserve(sols.size());
    sol_index.reserve(sols.size());
    
    for (size_t i = 0; i < n; i++) {
      if(sols[i] != nullptr) {
        obj_0.push_back(sols[i]->obj[0]);
        sol_index.push_back(i);
      }
    }
    
    n = sol_index.size();
    std::vector<size_t> sorted;
    compute_ranks_asc(obj_0, sorted);

    double area = (max_0 - fmin(max_0, obj_0[sorted[n - 1]])) * (max_1 - fmin(max_1, sols[sol_index[sorted[n - 1]]]->obj[1]));
    for (int i = (int) (n - 2); i >= 0; i--) {
      area += (fmin(max_0, obj_0[sorted[i + 1]]) - fmin(max_0, obj_0[sorted[i]])) * (max_1 - fmin(max_1, sols[sol_index[sorted[i]]]->obj[1]));
    }

    return area;
  }

 
}



double hicam::population_t::compute2DHyperVolumeAlreadySortedOnObj0(double max_0, double max_1) const
{
  
  size_t n = sols.size();
  
  if (n == 0) {
    return 0.0;
  }
  
  double area = (max_0 - fmin(max_0, sols[n - 1]->obj[0])) * (max_1 - fmin(max_1, sols[n - 1]->obj[1]));
  for (int i = (int) (n - 2); i >= 0; i--) {
    area += (fmin(max_0, sols[i + 1]->obj[0]) - fmin(max_0, sols[i]->obj[0])) * (max_1 - fmin(max_1, sols[i]->obj[1]));
  }
  
  return area;
}

double hicam::population_t::computeIGD(const population_t & pareto_set) const
{
  // this should warn you that something is wrong, an IGD of -1 is never possible.
  if(pareto_set.size() == 0) {
    return -1.0;
  }
  
  vec_t obj_ranges(pareto_set.sols[0]->number_of_objectives(), 1.0);
  return pareto_set.objective_distance(*this, obj_ranges);
}

double hicam::population_t::computeGD(const population_t & pareto_set) const
{
  // this should warn you that something is wrong, an IGD of -1 is never possible.
  if(pareto_set.size() == 0) {
    return -1.0;
  }
  
  vec_t obj_ranges(pareto_set.sols[0]->number_of_objectives(), 1.0);
  return this->objective_distance(pareto_set, obj_ranges);
}


double hicam::population_t::computeAnalyticGD(fitness_t & fitness_function) const
{
  
  // this should warn you that something is wrong, an IGD of -1 is never possible.
  if(!fitness_function.analytical_gd_avialable)  {
    return -1.0;
  }
 
  double sum_dist = 0;
  double count = 0;
  
  for(size_t i = 0; i < sols.size(); ++i)
  {
    if(sols[i] == nullptr) {
      continue;
    }
    
    sum_dist += fitness_function.distance_to_front(*sols[i]);
  
    count++;
  }
  
  // something wrong.
  if(count == 0) {
    return -1.0;
  }
  
  return sum_dist / count;
}

double hicam::population_t::computeIGDX(const population_t & pareto_set) const
{
  return pareto_set.param_distance(*this);
}


double hicam::population_t::computeSR(const std::vector<population_pt> & pareto_sets, double threshold, const vec_t & max_igd) const
{
  // this should warn you that something is wrong, an SR of -1 is never possible.
  if(pareto_sets.size() == 0) {
    return -1.0;
  }
  
  double successes = 0.0;
  double number_of_nonzero_sets = 0.0;
  vec_t achieved_igdx(pareto_sets.size(), 0.0);
  
  for(size_t i = 0; i < pareto_sets.size(); ++i)
  {
    
    if(pareto_sets[i]->size() == 0) {
      continue;
    }
    
    achieved_igdx[i] = this->computeIGDX(*pareto_sets[i]);
    successes += achieved_igdx[i]  <= (threshold * max_igd[i]);
    number_of_nonzero_sets++;
    
  }
  
  if(number_of_nonzero_sets == 0.0) {
    return -1.0;
  }
  
  
  // std::cout << std::endl << max_igd << std::endl;
  // std::cout << achieved_igdx << std::endl;
  
  return successes / number_of_nonzero_sets;
  
}

// note: population must be sorted already!
double hicam::population_t::computeSmoothness() const
{
  if(sols.size() <= 2) {
    return 1.0;
  }
  
  double smoothness = 0.0;

  double detour = 0;
  for(size_t i = 1; i < sols.size() - 1;++i) {
    detour = sols[i-1]->param_distance(*sols[i+1]) / ( sols[i]->param_distance(*sols[i-1]) + sols[i]->param_distance(*sols[i+1]));
    detour = std::max(0.0, detour);
    detour = std::min(1.0, detour);
    smoothness += detour;
  }
  
  smoothness /= (double) (sols.size() - 2.0);
  
  return smoothness;
}

void hicam::population_t::writeToFile(const char * filename) const
{
  char  string[1000];
  FILE *file;
  
  file = fopen(filename, "w");

  for (size_t i = 0; i < sols.size(); i++)
  {
    if(sols[i] != nullptr)
    {
      for (size_t j = 0; j < sols[i]->number_of_parameters(); j++)
      {
        sprintf(string, "%13e", sols[i]->param[j]);
        fputs(string, file);

        if (j <  sols[i]->number_of_parameters() - 1)
        {
          sprintf(string, " ");
          fputs(string, file);
        }
      }
    
      sprintf(string, "     ");
      fputs(string, file);

      for (size_t j = 0; j <  sols[i]->number_of_objectives(); j++)
      {
        sprintf(string, "%13e ", sols[i]->obj[j]);
        fputs(string, file);
      }

      sprintf(string, "%13e", sols[i]->constraint);
      fputs(string, file);
      
      sprintf(string, "%13d", sols[i]->cluster_number);
      fputs(string, file);

      for (size_t j = 0; j <  sols[i]->dvis.size(); j++)
      {
        sprintf(string, "%13e ", sols[i]->dvis[j]);
        fputs(string, file);
      }
      
      
      sprintf(string,"\n");
      fputs(string, file);
    }
  }

  fclose(file);

}


void hicam::population_t::writeObjectivesToFile(const char * filename) const
{
  char  string[1000];
  FILE *file;
  
  file = fopen(filename, "w");
  
  for (size_t i = 0; i < sols.size(); i++)
  {
    if(sols[i] != nullptr)
    {
      for (size_t j = 0; j <  sols[i]->number_of_objectives(); j++)
      {
        sprintf(string, "%13e ", sols[i]->obj[j]);
        fputs(string, file);
      }
      
      sprintf(string,"\n");
      fputs(string, file);
    }
  }
  
  fclose(file);
  
}

void hicam::population_t::read2DObjectivesFromFile(const char *filename, size_t number_of_lines)
{

  // read file
  FILE* f = fopen(filename, "r");

  if (f == NULL) {
    return;
  }

  vec_t obj(2, 0.0);

  sols.clear();
  sols.reserve(5000);

  for (size_t i = 0; i < number_of_lines; i++)
  {
    solution_pt sol = std::make_shared<solution_t>(0, 2);
    fscanf(f, "%lf %lf\n", &sol->obj[0], &sol->obj[1]);
    sols.push_back(sol);
  }

  fclose(f);
}

// distance between population is the average of the minimal distance to the other population. 
double hicam::population_t::objective_distance(const population_t & other_pop, const vec_t & obj_ranges) const
{

  if (this->size() == 0 || other_pop.size() == 0) {
    return 1e308;
  }

  double summed_distance = 0.0;
  double number_of_solutions = 0;
  for (size_t i = 0; i < this->size(); i++) {
    if(this->sols[i] != nullptr) {
      summed_distance += this->sols[i]->minimal_objective_distance(other_pop.sols, obj_ranges);
      number_of_solutions++;
    }
  }

  return (summed_distance / number_of_solutions);

}

// distance between population is the average of the minimal distance to the other population. 
double hicam::population_t::param_distance(const population_t & other_pop) const
{

  if (this->size() == 0 || other_pop.size() == 0) {
    return 1e308;
  }

  double summed_distance = 0.0;
  for (size_t i = 0; i < this->size(); i++) {
    summed_distance += this->sols[i]->minimal_param_distance(other_pop.sols);
  }

  return (summed_distance / this->size());

}
