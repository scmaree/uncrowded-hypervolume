/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA


*/

#include "adam.hpp"
#include "population.hpp"
#include "mathfunctions.hpp"
#include "gomea.hpp" // this is only here cuz i want a

namespace hillvallea
{
  // Constructor
  adam_t::adam_t
    (
     fitness_pt fitness_function,
     int version,
     const vec_t & lower_init_ranges,
     const vec_t & upper_init_ranges,
     int maximum_number_of_evaluations,
     int maximum_number_of_seconds,
     double vtr,
     int use_vtr,
     int random_seed,
     bool write_generational_solutions,
     bool write_generational_statistics,
     const std::string & write_directory,
     const std::string & file_appendix,
     double gamma_weight,
     double finite_differences_multiplier
     )
  {

    // copy all settings
    this->fitness_function = fitness_function;
    this->version = version;
    this->lower_init_ranges = lower_init_ranges;
    this->upper_init_ranges = upper_init_ranges;
    this->maximum_number_of_evaluations = maximum_number_of_evaluations;
    this->maximum_number_of_seconds = maximum_number_of_seconds;
    this->vtr = vtr;
    this->use_vtr = use_vtr;
    this->random_seed = random_seed;
    this->write_generational_solutions = write_generational_solutions;
    this->write_generational_statistics = write_generational_statistics;
    this->write_directory = write_directory;
    this->file_appendix = file_appendix;

    rng = std::make_shared<std::mt19937>((unsigned long)(random_seed));
    std::uniform_real_distribution<double> unif(0, 1);

    fitness_function->get_param_bounds(lower_param_bounds, upper_param_bounds);
    
    for(size_t i = 0; i < lower_param_bounds.size(); ++i) {
      if(this->lower_init_ranges[i] < lower_param_bounds[i]) { this->lower_init_ranges[i] = lower_param_bounds[i]; }
      if(this->upper_init_ranges[i] > upper_param_bounds[i]) { this->upper_init_ranges[i] = upper_param_bounds[i]; }
    }
    
    maximum_no_improvement_stretch = 25 + (int) fitness_function->number_of_parameters;
    use_boundary_repair = true;
    use_momentum_with_nag = false; // nesterov accelerated gradient
    accept_only_improvements = false;
    
    // Adam parameters;
    b1 = 0.9;
    b2 = 0.999;
    epsilon = 1e-16;
    
    double init_gamma = 0.0;
    for(size_t i = 0; i < fitness_function->number_of_parameters; ++i) {
      init_gamma += (upper_init_ranges[i] - lower_init_ranges[i]);
    }
    init_gamma /= fitness_function->number_of_parameters;
    
    assert(init_gamma > 0);
    
    gamma = init_gamma * gamma_weight;
    this->finite_differences_multiplier = finite_differences_multiplier;
    
  }

  adam_t::~adam_t() { }
  
  
  
  // Write statistic Files
  //----------------------------------------------------------------------------------
  void adam_t::new_statistics_file()
  {
    std::string filename;
    if (file_appendix.empty()) {
      filename = write_directory + "statistics.dat";
    }
    else {
      filename = write_directory + "statistics" + file_appendix + ".dat";
    }
    
    statistics_file.open(filename, std::ofstream::out | std::ofstream::trunc);
    assert(statistics_file.is_open());
    
    statistics_file
    << "Gen    Evals          Time                   Best-f    Best-constr               Current-obj       Std-obj    Cur-constr    Std-constr "
    << fitness_function->write_solution_info_header(false) << std::endl;
  }
  
  void adam_t::close_statistics_file() {
    statistics_file.close();
  }
  
  void adam_t::write_statistics_line(const solution_t & sol, size_t number_of_generations, const solution_t & best)
  {
    clock_t current_time = clock();
    double runtime = double(current_time - starting_time) / CLOCKS_PER_SEC;
    std::vector<solution_pt> niching_archive;
    
    statistics_file
    << std::setw(3) << number_of_generations
    << " " << std::setw(8) << number_of_evaluations
    << " " << std::setw(13) << std::scientific << std::setprecision(3) << runtime
    << " " << std::setw(25) << std::scientific << std::setprecision(16) << best.f
    << " " << std::setw(13) << std::scientific << std::setprecision(3) << best.constraint
    << " " << std::setw(25) << std::scientific << std::setprecision(16) << sol.f
    << " " << std::setw(13) << std::scientific << std::setprecision(3) << 0
    << " " << std::setw(13) << std::scientific << std::setprecision(3) << sol.constraint
    << " " << std::setw(13) << std::scientific << std::setprecision(3) << 0
    << " " << fitness_function->write_additional_solution_info(best, niching_archive, false) << std::endl;
  }
  
  // Termination Criteria
  //-------------------------------------------------------------------------------
  // returns true if we should terminate.
  bool adam_t::checkTerminationCondition(double old_fitness, solution_t & sol, int & no_improvement_stretch)
  {

    if(number_of_generations == 0) {
      return false;
    }
    
    if (maximum_number_of_evaluations > 0 && number_of_evaluations >= maximum_number_of_evaluations) {
      std::cout << "  Terminated core search algorithm because function evaluations limit reached" << std::endl;
      return true;
    }
    
    // stop if we run out of time.
    if (terminate_on_runtime()) {
      std::cout << "  Terminated core search algorithm because time limit reached" << std::endl;
      return true;
    }
    
    // stop if the vtr is hit
    if (use_vtr != 0)
    {
      bool vtr_hit = false;

      if(!fitness_function->redefine_vtr) {
        vtr_hit = (sol.constraint == 0) && (sol.f <= vtr);
      } else {
        vtr_hit = fitness_function->vtr_reached(sol, vtr);
      }
      
      if(vtr_hit)
      {
        std::cout << "  Terminated because VTR reached! Yay" << std::endl;
        success = true;
        return true;
      }
    }
    
    // Terminate
    //-----------------------------------------------------
    if(no_improvement_stretch > 2 * maximum_no_improvement_stretch) {
      std::cout << "Terminated because there are " << no_improvement_stretch << " generations without improvement (gamma = " << gamma << ") ." << std::endl;
      return true;
    }
    
    // Else, just continue;
    return false;
  }
  
  
  bool adam_t::terminate_on_runtime() const
  {
    // stop if we run out of time.
    if (maximum_number_of_seconds > 0)
    {
      clock_t current_time = clock();
      double runtime = double(current_time - starting_time) / CLOCKS_PER_SEC;
      
      if (runtime > maximum_number_of_seconds) {
        return true;
      }
    }
    
    return false;
  }
  
  
  void adam_t::run()
  {
    //---------------------------------------------
    // reset all runlogs (in case hillvallea is run multiple time)
    starting_time = clock();
    success = false;
    terminated = false;
    number_of_evaluations = 0;
    weighted_number_of_evaluations = 0;
    number_of_generations = 0;
    no_improvement_stretch = 0;
    
    //---------------------------------------------
    // output
    if(write_generational_statistics) {
      new_statistics_file();
    }
    
    // Initialize a first solution
    population_pt pop = std::make_shared<population_t>();
    size_t population_size = 1;
    size_t number_of_parameters = fitness_function->number_of_parameters;
    
    if(fitness_function->redefine_random_initialization) {
      fitness_function->init_solutions_randomly(*pop, population_size, lower_init_ranges, upper_init_ranges, 0, rng);
    }
    else {
      pop->fill_uniform(population_size, number_of_parameters, lower_init_ranges, upper_init_ranges, rng);
    }
    
    // Init ADAM parameters and settings
    solution_pt sol = pop->sols[0];
    if(!fitness_function->use_finite_differences) {
      fitness_function->evaluate_with_gradients(sol);
      weighted_number_of_evaluations++;
    } else {
      fitness_function->evaluate_with_finite_differences(sol, finite_differences_multiplier * gamma);
      weighted_number_of_evaluations++;
    }
    
    best = solution_t(*sol);
    
    double old_fitness = 1e300; // minimization!
    
    std::vector<std::vector<size_t>> touched_parameter_idx;
    vec_t gammas;
    
    // if there are no reference sols, we do full evaluations
    size_t p = sol->mo_test_sols.size();
    if(p == 0) { version = 50; }
    
    if(version == 50) {
      touched_parameter_idx.resize(1);
      gammas.resize(1,gamma);
      
      touched_parameter_idx[0].resize(number_of_parameters);
      for(size_t i = 0; i < number_of_parameters; ++i) {
        touched_parameter_idx[0][i] = i;
      }
    }
    
    if(version == 64)
    {
      size_t n = sol->mo_test_sols[0]->number_of_parameters();
      touched_parameter_idx.resize(p);
      gammas.resize(p, gamma);
      size_t ki = 0;
      for(size_t FOS_idx = 0; FOS_idx < p; ++FOS_idx)
      {
        touched_parameter_idx[FOS_idx].resize(n);
        for(size_t i = 0; i < n; ++i) {
          touched_parameter_idx[FOS_idx][i] = ki;
          ki++;
        }
      }
      assert(ki == number_of_parameters);
      
      maximum_no_improvement_stretch *= n;
    }
    
    if(version == 66)
    {
      size_t n = sol->mo_test_sols[0]->number_of_parameters();
      touched_parameter_idx.resize(p);
      gammas.resize(p+1, gamma);
      
      size_t ki = 0;
      for(size_t FOS_idx = 0; FOS_idx < p; ++FOS_idx)
      {
        touched_parameter_idx[FOS_idx].resize(n);
        for(size_t i = 0; i < n; ++i) {
          touched_parameter_idx[FOS_idx][i] = ki;
          ki++;
        }
      }
      assert(ki == number_of_parameters);
      
      touched_parameter_idx.push_back(std::vector<size_t>(number_of_parameters));
      for(size_t i = 0; i < number_of_parameters; ++i) {
        touched_parameter_idx[touched_parameter_idx.size()-1][i] = i;
      }
      maximum_no_improvement_stretch *= n+1;
    }
    
    vec_t lower, upper;
    fitness_function->get_param_bounds(lower, upper);
    
    // optimization loop
    //---------------------------------------------
    while(!checkTerminationCondition(old_fitness, *sol, no_improvement_stretch))
    {
      
      // this keeps the elitist archive tractable in UHV GOMEA
      if(number_of_generations % 50 == 0) {
        FOS_t dummy_FOS;
        population_t dummy_pop;
        fitness_function->sort_population_parameters(dummy_pop, dummy_FOS);
      }
      
      if (write_generational_statistics)
      {
        int digits = (int) round(log10((double)number_of_generations)) - 1;
        if(number_of_generations < 50 || number_of_generations % (int) std::pow(10.0,digits) == 0 )  {
          write_statistics_line(*sol, number_of_generations, best);
        }
      }
      
      // write elitist_archive of this generation.
      if (write_generational_solutions) {
        std::stringstream ss;
        ss << write_directory << "best_generation" << std::setw(5) << std::setfill('0') << number_of_generations << file_appendix << ".dat";
        fitness_function->write_solution(best, ss.str() );
        std::stringstream ss_current;
        ss_current << write_directory << "current_generation" << std::setw(5) << std::setfill('0') << number_of_generations << file_appendix << ".dat";
        sol_current = solution_t(*sol);
        fitness_function->write_solution(sol_current, ss_current.str() );
      }
      
      
      // for termination only
      old_fitness = sol->f;
      
      // this is it.
      if(version == 64) {
        weighted_number_of_evaluations += HIGAMOgradientOffspring(sol, touched_parameter_idx, gammas);
      } else {
        weighted_number_of_evaluations += gradientOffspring(sol, touched_parameter_idx, gammas);
      }
      
      number_of_generations++;
      number_of_evaluations = (int) round(weighted_number_of_evaluations);
      
    }

   if (write_generational_statistics)
   {
     int digits = ((int) round(log10((double)number_of_generations))) - 1;
     if(number_of_generations > 50 && number_of_generations % (int) std::pow(10.0,digits) != 0 )  { // prevents duplicate final line
       write_statistics_line(*sol, number_of_generations, best);
     }
   }
    
    // write final best
    std::stringstream ss;
    ss << write_directory << "best_final" << file_appendix << ".dat";
    fitness_function->write_solution(best, ss.str() );
    
    if (write_generational_statistics) {
      close_statistics_file();
    }
    
  }
  
  double adam_t::gradientOffspring(solution_pt &sol,  const std::vector<std::vector<size_t>> & touched_parameter_idx, vec_t & gamma)
  {
    
    size_t number_of_parameters = fitness_function->number_of_parameters;
    double weighted_number_of_evaluations = 0;
    bool use_adamax = false;
    
    // Determine new parameters
    //-------------------------------------------------------
    if(sol->adam_mt.size() != sol->gradient.size() || sol->adam_vt.size() != sol->gradient.size()) {
      sol->adam_mt.clear();
      sol->adam_mt.resize(number_of_parameters,0.0);
      sol->adam_vt.clear();
      sol->adam_vt.resize(number_of_parameters,0.0);
    }
    
    int *fos_order = randomPermutation((int) touched_parameter_idx.size(), *rng);

    
    for(size_t oj = 0; oj < touched_parameter_idx.size(); ++oj)
    {
      int FOS_idx = fos_order[oj];
      
      vec_t mt_biased(touched_parameter_idx[FOS_idx].size(), 0.0);
      vec_t vt_biased(touched_parameter_idx[FOS_idx].size(), 0.0);
      vec_t step(touched_parameter_idx[FOS_idx].size(), 0.0);
      vec_t new_param(touched_parameter_idx[FOS_idx].size(), 0.0);
      
      for(size_t i = 0; i < touched_parameter_idx[FOS_idx].size(); ++i) {
        size_t param_i = touched_parameter_idx[FOS_idx][i];
        sol->adam_mt[param_i] = b1 * sol->adam_mt[param_i] + (1.0 - b1) * sol->gradient[param_i];
        
        mt_biased[i] = sol->adam_mt[param_i] / (1.0 - pow(b1,number_of_generations + 1.0));
        
        double gamma_factor = 0;
        if(use_adamax) {
          sol->adam_vt[param_i] = std::max(b2*sol->adam_vt[param_i], fabs(sol->gradient[param_i]));
          gamma_factor = sol->adam_vt[param_i];
        } else {
          sol->adam_vt[param_i] = b2 * sol->adam_vt[param_i] + (1.0 - b2) * (sol->gradient[param_i] * sol->gradient[param_i]);
          vt_biased[i] = sol->adam_vt[param_i] / (1.0 - pow(b2,number_of_generations + 1.0));
          gamma_factor = sqrt(vt_biased[i]) + epsilon;
        }
        
        if(use_momentum_with_nag) {
          step[i] = - (gamma[FOS_idx] * b1) * (mt_biased[i] / gamma_factor) - (gamma[FOS_idx] * (1.0-b1) / (1.0 - pow(b1,number_of_generations + 1.0))) * (sol->gradient[param_i] / gamma_factor);
        } else {
          step[i] = - gamma[FOS_idx] * (mt_biased[i] / gamma_factor);
        }
      }
      
      bool out_of_range = true;
      double shrink_factor = 2;
      while( out_of_range && (shrink_factor > epsilon) )
      {
        shrink_factor *= 0.5;
        out_of_range = false;
        for(size_t i = 0; i < touched_parameter_idx[FOS_idx].size(); i++ )
        {
          size_t param_i = touched_parameter_idx[FOS_idx][i];
          new_param[i] = sol->param[param_i]+shrink_factor*step[i];
          if( !isParameterInRangeBounds( new_param[i], param_i ) ) {
            out_of_range = true;
            break;
          }
        }
      }
      
      solution_pt old_sol;
      if( !out_of_range )
      {
        if(shrink_factor < 1) {
          std::cout << "Used a shrink factor of " << shrink_factor << "." << std::endl;
        }
        
        old_sol = std::make_shared<solution_t>(*sol); // this still copies the entire solution, which is probably not desirable
        for(size_t i = 0; i < touched_parameter_idx[FOS_idx].size(); i++ ) {
          size_t param_i = touched_parameter_idx[FOS_idx][i];
          sol->param[param_i] = new_param[i];
        }
      } else {
        // terminate the loop;
        std::cout << "Solution out of range, skip update." << std::endl;
        continue;
      }
      
      // end update parameters
      bool found_improvement = false;
      while( !found_improvement && (shrink_factor > epsilon) )
      {
        found_improvement = true; // is set to false if needed later.
        // evaluate solution
        if(touched_parameter_idx[FOS_idx].size() == number_of_parameters)
        {
          if(!fitness_function->use_finite_differences) {
            fitness_function->evaluate_with_gradients(sol);
            weighted_number_of_evaluations++;
          } else {
            fitness_function->evaluate_with_finite_differences(sol, finite_differences_multiplier * gamma[FOS_idx]);
            weighted_number_of_evaluations++;
          }
        }
        else
        {
          if(!fitness_function->use_finite_differences) {
            fitness_function->partial_evaluate_with_gradients(sol, touched_parameter_idx[FOS_idx], old_sol);
            weighted_number_of_evaluations += (touched_parameter_idx[FOS_idx].size() / (double) number_of_parameters);
          } else {
            fitness_function->partial_evaluate_with_finite_differences(sol, touched_parameter_idx[FOS_idx], old_sol, finite_differences_multiplier * gamma[FOS_idx]);
            weighted_number_of_evaluations += (touched_parameter_idx[FOS_idx].size() / (double) number_of_parameters);
          }
        }
        
        if(solution_t::better_solution(*sol, best)) {
          best = solution_t(*sol);
        }
        
        if(solution_t::better_solution(*old_sol, *sol))
        {
          no_improvement_stretch++;
          if(accept_only_improvements) {
            shrink_factor *= 0.5;
            found_improvement = false;
            
            for(size_t i = 0; i < touched_parameter_idx[FOS_idx].size(); i++ ) {
              size_t param_i = touched_parameter_idx[FOS_idx][i];
              sol->param[param_i] = old_sol->param[param_i]+shrink_factor*step[i];
            }
          } else {
            gamma[FOS_idx] *= 0.99; // 1 - 0.1 / (double) sol->mo_test_sols.size();
          }
        } else {
          no_improvement_stretch = 0;
          found_improvement = true;
        }
      }
      
      if(!found_improvement) {
        sol = old_sol;
      }
      
    }
    free( fos_order );
    
    return weighted_number_of_evaluations;
  }

  
  double adam_t::HIGAMOgradientOffspring(solution_pt &sol,  const std::vector<std::vector<size_t>> & touched_parameter_idx, vec_t & gamma)
  {
    
    size_t number_of_parameters = fitness_function->number_of_parameters;
    double weighted_number_of_evaluations = 0;
    
    // HIGA-MO parameters
    double alpha = 0.7; // or alpha = 0.5
    double c = 0.1;
    
    solution_pt old_sol = std::make_shared<solution_t>(*sol); // this still copies the entire solution, which is probably not desirable
    
    int *fos_order = randomPermutation((int) touched_parameter_idx.size(), *rng);
    
    // find parameter_space_distances
    // this is very specific UHV code
    double min_param_dist = 1e300;
    double max_param_dist = 0;
    double dist;
    for(size_t i = 0; i < touched_parameter_idx.size(); ++i)
    {
      vec_t params_i(touched_parameter_idx[i].size());
      for(size_t k = 0; k < touched_parameter_idx[i].size(); ++k) {
        params_i[k] = sol->param[touched_parameter_idx[i][k]];
      }
      
      for(size_t j = i+1; j < touched_parameter_idx.size(); ++j)
      {
        vec_t params_j(touched_parameter_idx[j].size());
        
        for(size_t k = 0; k < touched_parameter_idx[j].size(); ++k) {
          params_j[k] = sol->param[touched_parameter_idx[j][k]];
        }
        
        dist = (params_i - params_j).norm();
        
        if(dist < min_param_dist) {
          min_param_dist = dist;
        }
        if(dist > max_param_dist) {
          max_param_dist = dist;
        }
      }
    }
    
    
    // ADAM_mt = self.inner_product in HIGA-MO
    // ADAM_vt = self.path in HIGA-MO.
    if(sol->adam_mt.size() != touched_parameter_idx.size()) {
      sol->adam_mt.clear();
      sol->adam_mt.resize(touched_parameter_idx.size(),0.0);
      sol->adam_vt.clear();
      sol->adam_vt.resize(number_of_parameters,0.0);
    }
    
    for(size_t oj = 0; oj < touched_parameter_idx.size(); ++oj)
    {
      int FOS_idx = fos_order[oj];
      size_t n = touched_parameter_idx[FOS_idx].size();
      
      // Get gradient & normalize
      vec_t normalized_gradient(n,0.0);
      vec_t new_param(n,0.0);
      vec_t path(n,0.0);
      
      for(size_t i = 0; i < n; ++i) {
        size_t param_i = touched_parameter_idx[FOS_idx][i];
        normalized_gradient[i] = sol->gradient[param_i];
        path[i] = sol->adam_vt[param_i]; // old normalized gradient
      }
      double gradient_norm = normalized_gradient.norm();
      
      if(gradient_norm > 0) {
        normalized_gradient /= gradient_norm;
      }
      
      if(number_of_generations > 0) {
        sol->adam_mt[FOS_idx] = (1.0 - c) * sol->adam_mt[FOS_idx] + c * path.dot(normalized_gradient);
        gamma[FOS_idx] *= exp(alpha * sol->adam_mt[FOS_idx]);
        assert(!isnan(gamma[FOS_idx]));
        // std::cout << gamma[FOS_idx] << " ";
      }
       
      // exponential update with upper bound (which seems to be very important, or i have a bug at this point)
      double step_size_ub = 0.7 * 0.5 * (min_param_dist + max_param_dist);
      gamma[FOS_idx] = std::min(gamma[FOS_idx], step_size_ub);
      // std::cout << "step = " << gamma[FOS_idx] << "\n";
      
      for(size_t i = 0; i < n; i++ ){
        size_t param_i = touched_parameter_idx[FOS_idx][i];
        sol->param[param_i] -= gamma[FOS_idx] * normalized_gradient[i];
        
        // this does boundary repair, if enabled.
        if(!isParameterInRangeBounds(sol->param[param_i], i)) {
          std::cout << "parameter out of range \n";
        }
      }
      
      // backup the current (normalized) gradient.
      for(size_t i = 0; i < n; ++i) {
        size_t param_i = touched_parameter_idx[FOS_idx][i];
        sol->adam_vt[param_i] = normalized_gradient[i];
      }
    }
    free( fos_order );
    // std::cout << "\n";
    
    // check if in range.
    
    // evaluate solution
    if(!fitness_function->use_finite_differences) {
      fitness_function->evaluate_with_gradients(sol);
      weighted_number_of_evaluations++;
    } else {
      fitness_function->evaluate_with_finite_differences(sol, finite_differences_multiplier * gamma.mean());
      weighted_number_of_evaluations++;
    }
      
    if(solution_t::better_solution(*sol, best)) {
      best = solution_t(*sol);
    }
      
    if(solution_t::better_solution(*old_sol, *sol)) {
      no_improvement_stretch++;
    } else {
      no_improvement_stretch = 0;
    }
    
    
    return weighted_number_of_evaluations;
  }
  
  bool adam_t::isParameterInRangeBounds( double & parameter, size_t dimension ) const
  {
    
    if( parameter < lower_param_bounds[dimension] ||
       parameter > upper_param_bounds[dimension] ||
       isnan( parameter ) )
    {
      if(use_boundary_repair)
      {
        if( parameter < lower_param_bounds[dimension] ) {
          parameter = lower_param_bounds[dimension];
        }
        
        if( parameter > upper_param_bounds[dimension] ) {
          parameter = upper_param_bounds[dimension];
        }
        return true;
        
      }
      return false;
    }
    return true;
  }
  
  
  double adam_t::evaluateWithfiniteDifferences(solution_pt & sol, double h, bool use_central_difference) const
  {
    double number_of_evaluations = 0;
    
    //fitness_function->evaluate(sol);
    fitness_function->evaluate_with_gradients(sol);
    number_of_evaluations++;
    
    solution_pt sol_plus_h = std::make_shared<solution_t>(*sol);
    solution_pt sol_min_h = std::make_shared<solution_t>(*sol);
    
    sol->gradient.resize(sol->param.size());
    
    for(size_t i = 0; i < sol->param.size(); ++i)
    {
      
      // set the params for the step.
      if(use_central_difference) {
        sol_min_h->param[i] -= 0.5 * h;
        sol_plus_h->param[i] += 0.5 * h;
      } else {
        sol_plus_h->param[i] += h;
      }
      
      fitness_function->evaluate(sol_plus_h);
      number_of_evaluations++;
      
      if(use_central_difference) {
        fitness_function->evaluate(sol_min_h);
        number_of_evaluations++;
      }
      
      if(use_central_difference) {
        sol->gradient[i] = (sol_plus_h->f - sol_min_h->f) / h;
      } else {
        std::cout << sol->gradient[i];
        double old_gradient = sol->gradient[i];
        sol->gradient[i] = (sol_plus_h->f - sol->f) / h;
        std::cout << " --> " << sol->gradient[i] << ". Error = " << ( sol->gradient[i] - old_gradient ) / old_gradient << std::endl;
      }
      
      
      // reset the param we changed in this iteration
      sol_plus_h->param[i] = sol->param[i];
      sol_min_h->param[i] = sol->param[i];
      sol_plus_h->f = sol->f;
      sol_min_h->f = sol->f;
    }
    
   
    return number_of_evaluations;
  }
  
  
  double adam_t::partialEvaluateWithfiniteDifferences(solution_pt & sol, double h, bool use_central_difference, const std::vector<size_t> & touched_parameter_idx, const solution_pt & old_sol) const
  {
    double weighted_number_of_evaluations = 0;
    
    fitness_function->partial_evaluate_with_gradients(sol, touched_parameter_idx, old_sol);
    double n =  (double) sol->param.size();
    weighted_number_of_evaluations += (touched_parameter_idx.size() / n);
    
    solution_pt sol_plus_h = std::make_shared<solution_t>(*sol);
    solution_pt sol_min_h = std::make_shared<solution_t>(*sol);
    
    if (sol->gradient.size() != sol->param.size()) {
      sol->gradient.resize(sol->param.size());
    }
    
    std::vector<size_t> touched_idx(1,0);
    for(size_t ki = 0; ki < touched_parameter_idx.size(); ++ki)
    {
      size_t i = touched_parameter_idx[ki];
      touched_idx[0] = i;
      
      // set the params for the step.
      if(use_central_difference) {
        sol_min_h->param[i] -= 0.5 * h;
        sol_plus_h->param[i] += 0.5 * h;
      } else {
        sol_plus_h->param[i] += h;
      }
      
      fitness_function->partial_evaluate(sol_plus_h, touched_idx, old_sol);
      weighted_number_of_evaluations += 1.0 / n;
      
      if(use_central_difference) {
        fitness_function->partial_evaluate(sol_min_h, touched_idx, old_sol);
        weighted_number_of_evaluations += 1.0 / n;
      }
      
      if(use_central_difference) {
        sol->gradient[i] = (sol_plus_h->f - sol_min_h->f) / h;
      } else {
        sol->gradient[i] = (sol_plus_h->f - sol->f) / h;
      }
      
      
      // reset the param we changed in this iteration
      sol_plus_h->param[i] = sol->param[i];
      sol_min_h->param[i] = sol->param[i];
      sol_plus_h->f = sol->f;
      sol_min_h->f = sol->f;
    }
    
    
    return number_of_evaluations;
  }

}
