/*
 
 HillVallEA
 
 By S.C. Maree
 s.c.maree[at]amc.uva.nl
 github.com/SCMaree/HillVallEA
 
 
 */

#include "uhvgrad.hpp"
#include "population.hpp"
#include "mathfunctions.hpp"


namespace hillvallea
{
  // Constructor
  uhvgrad_t::uhvgrad_t
  (
   hicam::fitness_pt mo_fitness_function,
   size_t number_of_mo_solutions,
   const hicam::vec_t & lower_init_ranges,
   const hicam::vec_t & upper_init_ranges,
   bool collect_all_mo_sols_in_archive,
   size_t elitist_archive_size_target,
   double gamma_weight,
   bool use_finite_differences,
   double finite_differences_multiplier,
   int maximum_number_of_evaluations,
   int maximum_number_of_seconds,
   double vtr,
   int use_vtr,
   int random_seed,
   bool write_generational_solutions,
   bool write_generational_statistics,
   const std::string & write_directory,
   const std::string & file_appendix
   )
  {
    
    // copy all settings
    this->mo_fitness_function = mo_fitness_function;
    this->number_of_mo_solutions = number_of_mo_solutions;
    this->lower_init_ranges = lower_init_ranges;
    this->upper_init_ranges = upper_init_ranges;
    this->collect_all_mo_sols_in_archive = collect_all_mo_sols_in_archive;
    this->elitist_archive_size_target = elitist_archive_size_target;
    this->gamma_weight = gamma_weight;
    this->use_finite_differences = use_finite_differences;
    this->finite_differences_multiplier = finite_differences_multiplier;
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
    

    // param_bounds
    mo_fitness_function->get_param_bounds(lower_param_bounds, upper_param_bounds);
    
    for(size_t i = 0; i < lower_param_bounds.size(); ++i) {
      if(this->lower_init_ranges[i] < lower_param_bounds[i]) { this->lower_init_ranges[i] = lower_param_bounds[i]; }
      if(this->upper_init_ranges[i] > upper_param_bounds[i]) { this->upper_init_ranges[i] = upper_param_bounds[i]; }
    }
    // end param bounds
    
    
    maximum_no_improvement_stretch = 25 + (int) mo_fitness_function->number_of_parameters;
    use_boundary_repair = true; // i.e., projected gradient descent
    
    // Adam parameters;
    b1 = 0.9;
    b2 = 0.999;
    epsilon = 1e-16;
    
    // Adam set gamma
    double init_gamma = 0.0;
    for(size_t i = 0; i < mo_fitness_function->number_of_parameters; ++i) {
      init_gamma += (upper_init_ranges[i] - lower_init_ranges[i]);
    }
    init_gamma /= mo_fitness_function->number_of_parameters;
    
    assert(init_gamma > 0);
    
    gamma = init_gamma * gamma_weight;
    // end setting gamma
    
    // allocate archive
    if(this->collect_all_mo_sols_in_archive)
    {
      if(this->elitist_archive == nullptr) { //banaan
        rng_pt rng = std::make_shared<rng_t>(142391);
        elitist_archive = std::make_shared<hicam::elitist_archive_t>(elitist_archive_size_target, rng);
      }
      
      // this uses gHSS to reduce the archive size. Else, adaptive search space discretization is used.
      hicam::vec_t r(2);
      r[0] = mo_fitness_function->hypervolume_max_f0;
      r[1] = mo_fitness_function->hypervolume_max_f1;
      elitist_archive->set_use_hypervolume_for_size_control(false, r);
      
    }
    // end archive stuff
    
  }
  
  uhvgrad_t::~uhvgrad_t() { }
  
  
  
  // Write statistic Files
  //----------------------------------------------------------------------------------
  void uhvgrad_t::new_statistics_file()
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
    << "Gen    Evals          Time                  Best-UHV               Current-UHV            Appr.set HV           Appr.set IGD            Appr.set GD size            Elite.arch HV           Elite.arch IGD            Elite.arch GD size "
    << std::endl;
  }
  
  void uhvgrad_t::close_statistics_file() {
    statistics_file.close();
  }
  
  void uhvgrad_t::write_statistics_line(const hicam::population_pt & mo_population, double current_hypervolume, size_t number_of_generations, const hicam::population_pt & best_mo_population, double best_hypervolume)
  {
    clock_t current_time = clock();
    double runtime = double(current_time - starting_time) / CLOCKS_PER_SEC;
    std::vector<solution_pt> niching_archive;
    
    statistics_file
    << std::setw(3) << number_of_generations
    << " " << std::setw(8) << number_of_evaluations
    << " " << std::setw(13) << std::scientific << std::setprecision(3) << runtime
    << " " << std::setw(25) << std::scientific << std::setprecision(16) << best_hypervolume
    << " " << std::setw(25) << std::scientific << std::setprecision(16) << current_hypervolume;
    
    
    //------------------------------------------
    // create approximation set
    hicam::rng_pt rng = std::make_shared<hicam::rng_t>(1000);
    hicam::elitist_archive_t approximation_set(1000, rng);
    

    // collect all mo_solutions from all optima
    for(size_t i = 0; i < best_mo_population->size(); ++i) {
      approximation_set.updateArchive(best_mo_population->sols[i]);
    }
    approximation_set.removeSolutionNullptrs();
    std::sort(approximation_set.sols.begin(), approximation_set.sols.end(), hicam::solution_t::strictly_better_solution_via_pointers_obj0_unconstraint);
    
    // Allocate default values for parameters
    
    double HV = 0;
    double IGD = 0;
    double GD = 0;
    double size = 0;
    
    double archive_HV = 0;
    double archive_IGD = 0;
    double archive_GD = 0;
    double archive_size = 0;
    
    // IGD
    if (mo_fitness_function->igd_available) {
      IGD = approximation_set.computeIGD(mo_fitness_function->pareto_set);
    }
    
    // HV (2D Hypervolume)
    if(mo_fitness_function->number_of_objectives == 2) {
      HV = approximation_set.compute2DHyperVolume(mo_fitness_function->hypervolume_max_f0, mo_fitness_function->hypervolume_max_f1);
    }
    
    // GD
    if(mo_fitness_function->igd_available) {
      if(mo_fitness_function->analytical_gd_avialable) {
        GD = approximation_set.computeAnalyticGD(*mo_fitness_function);
      } else {
        GD = approximation_set.computeGD(mo_fitness_function->pareto_set);
      }
    }
    
    // Size
    size = approximation_set.actualSize();
    
    // Elitist Archive (=large_approximation_set)
    //-------------------------------------------
    if(collect_all_mo_sols_in_archive && elitist_archive != nullptr)
    {
      // Compute IGD
      if (mo_fitness_function->igd_available) {
        archive_IGD = elitist_archive->computeIGD(mo_fitness_function->pareto_set);
      }
      
      // HV
      if(mo_fitness_function->number_of_objectives == 2) {
        archive_HV = elitist_archive->compute2DHyperVolume(mo_fitness_function->hypervolume_max_f0, mo_fitness_function->hypervolume_max_f1);
      }
      
      // GD
      if (mo_fitness_function->igd_available) {
        
        if(mo_fitness_function->analytical_gd_avialable) {
          archive_GD = elitist_archive->computeAnalyticGD(*mo_fitness_function);
        } else {
          archive_GD = elitist_archive->computeGD(mo_fitness_function->pareto_set);
        }
      }
      
      // Size
      archive_size = elitist_archive->actualSize();
    }
    
    // Start writing stuff
    //-----------------------------------------------------------------------------------------
    statistics_file
    << " " << std::setw(14) << std::scientific << std::setprecision(16) << HV
    << " " << std::setw(14) << std::scientific << std::setprecision(16) << IGD
    << " " << std::setw(14) << std::scientific << std::setprecision(16) << GD
    << " " << std::setw(4) << std::fixed << (int) size;
    
    statistics_file
    << " " << std::setw(24) << std::scientific << std::setprecision(16) << archive_HV
    << " " << std::setw(24) << std::scientific << std::setprecision(16) << archive_IGD
    << " " << std::setw(24) << std::scientific << std::setprecision(16) << archive_GD
    << " " << std::setw(4) << std::fixed << (int) archive_size;
    
    statistics_file << std::endl;
  }
  
  // Termination Criteria
  //-------------------------------------------------------------------------------
  // returns true if we should terminate.
  bool uhvgrad_t::checkTerminationCondition(const hicam::population_pt & mo_population, double current_hypervolume, int no_improvement_stretch)
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
      
      if(use_vtr == 1) {
        vtr_hit = current_hypervolume >= vtr;
      }
      
      if(use_vtr == 2) {
        if(collect_all_mo_sols_in_archive) {
          vtr_hit = elitist_archive->computeIGD(mo_fitness_function->pareto_set) < vtr;
        } else {
          vtr_hit = mo_population->computeIGD(mo_fitness_function->pareto_set) < vtr;
        }
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
    if(no_improvement_stretch > maximum_no_improvement_stretch) {
      std::cout << "Terminated because there are " << no_improvement_stretch << " generations without improvement (gamma = " << gamma << ") ." << std::endl;
      return true;
    }
    
    // Else, just continue;
    return false;
  }
  
  
  bool uhvgrad_t::terminate_on_runtime() const
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
  
  
  void uhvgrad_t::run()
  {
    //---------------------------------------------
    // reset all runlogs (in case hillvallea is run multiple time)
    starting_time = clock();
    success = false;
    terminated = false;
    number_of_evaluations = 0;
    number_of_generations = 0;
    no_improvement_stretch = 0;
    
    //---------------------------------------------
    // output
    if(write_generational_statistics) {
      new_statistics_file();
    }
    
    // Allocate MO_population that is being optimized
    //----------------------------------------------
    mo_population = std::make_shared<hicam::population_t>();
    best_mo_population = std::make_shared<hicam::population_t>();
    
    if(mo_fitness_function->redefine_random_initialization) {
      mo_fitness_function->init_solutions_randomly(mo_population, number_of_mo_solutions, lower_init_ranges, upper_init_ranges, 0, rng);
    }
    else {
      mo_population->fill_uniform(number_of_mo_solutions, mo_fitness_function->number_of_parameters, lower_init_ranges, upper_init_ranges, rng);
    }
    
    // evaluate MO solutions
    for(size_t k = 0; k < mo_population->sols.size(); ++k)
    {
      if(use_finite_differences) {
        number_of_evaluations += evaluateWithfiniteDifferences(mo_population->sols[k], finite_differences_multiplier * gamma);
      } else {
        mo_fitness_function->evaluate_with_gradients(mo_population->sols[k]);
        number_of_evaluations++;
      }
      
      if(collect_all_mo_sols_in_archive) {
        elitist_archive->updateArchive(mo_population->sols[k],true);
      }
    }
    
    
    
    // Log best obtained population
    best_mo_population->sols.clear();
    best_mo_population->addCopyOfSolutions(*mo_population);
    double old_hypervolume = -1e300; // uhv(mo_population, true, gradient_weights);
    best_hypervolume = uhv(mo_population, true, gradient_weights);
    current_hypervolume = best_hypervolume;
    
    // optimization loop
    //---------------------------------------------
    while(!checkTerminationCondition(mo_population, current_hypervolume, no_improvement_stretch))
    {
      
      // this keeps the elitist archive tractable in UHV GOMEA
      if(number_of_generations % 50 == 0 && collect_all_mo_sols_in_archive && elitist_archive != nullptr) {
        elitist_archive->adaptArchiveSize();
      }
      
      if (write_generational_statistics)
      {
        int digits = (int) round(log10((double)number_of_generations)) - 1;
        if(number_of_generations < 50 || number_of_generations % (int) std::pow(10.0,digits) == 0 )  {
          write_statistics_line(mo_population, current_hypervolume, number_of_generations, best_mo_population, best_hypervolume);
        }
      }
      
      // write elitist_archive of this generation.
      if (write_generational_solutions)
      {
        // write best
        std::stringstream ss;
        ss << write_directory << "best_generation" << std::setw(5) << std::setfill('0') << number_of_generations << file_appendix << ".dat";
        best_mo_population->writeToFile(ss.str().c_str());

        // write current
        std::stringstream ss_current;
        ss_current << write_directory << "current_generation" << std::setw(5) << std::setfill('0') << number_of_generations << file_appendix << ".dat";
        mo_population->writeToFile(ss_current.str().c_str());
      }
      
      
      // for termination only
      old_hypervolume = current_hypervolume;
      
      // This is actually the interesting part
      number_of_evaluations += gradientOffspring(mo_population, current_hypervolume, best_mo_population, best_hypervolume, gamma);
      
      // end of the generational loop
      number_of_generations++;
    }
    
    if (write_generational_statistics)
    {
      int digits = ((int) round(log10((double)number_of_generations))) - 1;
      if(number_of_generations > 50 && number_of_generations % (int) std::pow(10.0,digits) != 0 )  { // prevents duplicate final line
        write_statistics_line(mo_population, current_hypervolume, number_of_generations, best_mo_population, best_hypervolume);
      }
    }
    
    // write elitist_archive of this generation.
    if (write_generational_solutions)
    {
      std::stringstream ss_current;
      ss_current << write_directory << "current_generation" << std::setw(5) << std::setfill('0') << number_of_generations << file_appendix << ".dat";
      mo_population->writeToFile(ss_current.str().c_str());
    }
    
    // write best
    std::stringstream ss;
    ss << write_directory << "best_generation" << std::setw(5) << std::setfill('0') << number_of_generations << file_appendix << ".dat";
    best_mo_population->writeToFile(ss.str().c_str());
    
    
    if (write_generational_statistics) {
      close_statistics_file();
    }
    
  }
  
  double uhvgrad_t::gradientOffspring(hicam::population_pt & mo_population, double & current_hypervolume, hicam::population_pt & best_mo_population, double & best_hypervolume, double & gamma)
  {

    
    size_t local_number_of_evaluations = 0;
    
    // Allocate solution memory for ADAM
    //-------------------------------------------------------
    size_t number_of_parameters = mo_fitness_function->number_of_parameters;
    size_t number_of_objectives = mo_fitness_function->number_of_objectives;
    
    if(mt.size() != mo_population->sols.size() || vt.size() != mo_population->sols.size()) {
      mt.clear();
      mt.resize(mo_population->sols.size());
      vt.clear();
      vt.resize(mo_population->sols.size());
    }
    
    for(size_t k = 0; k < mo_population->sols.size(); ++k)
    {

      if(mo_population->sols[k]->adam_mt.size() != number_of_objectives || mo_population->sols[k]->adam_vt.size() != number_of_objectives) {
        mo_population->sols[k]->adam_mt.clear();
        mo_population->sols[k]->adam_mt.resize(number_of_objectives,0.0);
        mo_population->sols[k]->adam_vt.clear();
        mo_population->sols[k]->adam_vt.resize(number_of_objectives,0.0);
      }
      
      for(size_t m = 0; m < number_of_objectives; ++m)
      {
        if(mo_population->sols[k]->adam_mt[m].size() != number_of_parameters || mo_population->sols[k]->adam_vt[m].size() != number_of_parameters) {
          mo_population->sols[k]->adam_mt[m].clear();
          mo_population->sols[k]->adam_mt[m].resize(number_of_parameters,0.0);
          mo_population->sols[k]->adam_vt[m].clear();
          mo_population->sols[k]->adam_vt[m].resize(number_of_parameters,0.0);
        }
      }
      
      // For SO-smoothing of gradients
      if(mt[k].size() != number_of_parameters || vt[k].size() != number_of_parameters) {
        mt[k].clear();
        mt[k].resize(number_of_parameters, 0.0);
        vt[k].clear();
        vt[k].resize(number_of_parameters, 0.0);
      }
    }
    

    
    double mt_biased = 0.0;
    double vt_biased = 0.0;
    vec_t step(number_of_parameters, 0.0);
    hicam::vec_t new_param(number_of_parameters, 0.0);

    // Update solution parameters
    //-------------------------------------------------------
    for(size_t k = 0; k < mo_population->sols.size(); ++k)
    {
      
      double gradient = 0.0;
      for(size_t n = 0; n < number_of_parameters; ++n)
      {
        gradient = 0.0;
        for(size_t m = 0; m < number_of_objectives; ++m) {
          gradient += gradient_weights[k][m] * mo_population->sols[k]->gradients[m][n];
        }
      
        mt[k][n] = b1 * mt[k][n] + (1.0 - b1) * gradient;
        vt[k][n] = b2 * vt[k][n] + (1.0 - b2) * gradient * gradient;
        mt_biased = mt[k][n] / (1.0 - pow(b1,number_of_generations + 1.0));
        vt_biased = vt[k][n] / (1.0 - pow(b2,number_of_generations + 1.0));

        step[n] = -gamma * (mt_biased / (sqrt(vt_biased) + epsilon));
      }
      //*/
      
      // scale the step to fit the domain
      bool out_of_range = true;
      double shrink_factor = 2;
      while( out_of_range && (shrink_factor > epsilon) )
      {
        shrink_factor *= 0.5;
        out_of_range = false;
        for(size_t n = 0; n < number_of_parameters; n++ )
        {
          new_param[n] = mo_population->sols[k]->param[n]+shrink_factor*step[n];
          if( !isParameterInRangeBounds( new_param[n], n ) ) {
            out_of_range = true;
            break;
          }
        }
      }
      
      if( !out_of_range ) {
        mo_population->sols[k]->param = new_param; // vector operation
      } else {
        // terminate the loop;
        std::cout << "Solution out of range, skip update." << std::endl;
        continue;
      }
    }
    
    // Evaluate Solutions
    //-------------------------------------------------------
    for(size_t k = 0; k < mo_population->sols.size(); ++k)
    {
      if(use_finite_differences) {
        local_number_of_evaluations += evaluateWithfiniteDifferences(mo_population->sols[k], finite_differences_multiplier * gamma);
      } else {
        mo_fitness_function->evaluate_with_gradients(mo_population->sols[k]);
        local_number_of_evaluations++;
      }
      
      if(collect_all_mo_sols_in_archive) {
        elitist_archive->updateArchive(mo_population->sols[k],true);
      }
    }
    
    
    // Compute HV of new population
    double old_hypervolume = current_hypervolume;
    current_hypervolume = uhv(mo_population, true, gradient_weights); // mo_population->compute2DHyperVolume(mo_fitness_function->hypervolume_max_f0, mo_fitness_function->hypervolume_max_f1);
    
    // replace best if iprovements were found.
    if(current_hypervolume > best_hypervolume) {
      best_hypervolume = current_hypervolume;
      best_mo_population->sols.clear();
      best_mo_population->addCopyOfSolutions(*mo_population);
    }
        
    if(old_hypervolume > current_hypervolume) {
      no_improvement_stretch++;
      gamma *= 0.99; // 1 - 0.1 / (double) sol->mo_test_sols.size();
    } else {
      no_improvement_stretch = 0;
    }
    
    return local_number_of_evaluations;
  }
  
  
  bool uhvgrad_t::isParameterInRangeBounds( double & parameter, size_t dimension ) const
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
  
  
  double uhvgrad_t::evaluateWithfiniteDifferences(hicam::solution_pt & sol, double h) const
  {
    double number_of_evaluations = 0;
    
    mo_fitness_function->evaluate(sol);
    number_of_evaluations++;
    
    hicam::solution_pt sol_plus_h = std::make_shared<hicam::solution_t>(*sol);
    
    sol->gradients.resize(mo_fitness_function->number_of_objectives);
    
    for(size_t m = 0; m < sol->gradients.size(); ++m) {
      sol->gradients[m].resize(mo_fitness_function->number_of_parameters);
    }
    
    for(size_t n = 0; n < sol->param.size(); ++n)
    {
      sol_plus_h->param[n] += h;
      
      mo_fitness_function->evaluate(sol_plus_h);
      number_of_evaluations++;
      
      for(size_t m = 0; m < mo_fitness_function->number_of_objectives; ++m) {
        sol->gradients[m][n] = (sol_plus_h->obj[m] - sol->obj[m]) / h;
      }
      
      // reset the param we changed in this iteration
      sol_plus_h->param[n] = sol->param[n];
      sol_plus_h->obj = sol->obj;
    }
    
    return number_of_evaluations;
  }
  
  
  
  // gets all nondominated solutions that are in the feasible domain, defined by (r_x,r_y)
  void uhvgrad_t::get_front(const std::vector<hicam::solution_pt> & mo_sols, std::vector<bool> & is_part_of_front, vec_t & front_x, vec_t & front_y, double r_x, double r_y) const
  {
    size_t K = mo_sols.size();
    is_part_of_front.clear();
    is_part_of_front.resize(K,true);
    size_t number_of_dominated_sols = 0;
    
    // for each solution
    for(size_t k = 0; k < K; ++k)
    {
      
      // if its not in the feasible region, skip it.
      if( mo_sols[k]->obj[0] >= r_x || mo_sols[k]->obj[1] >= r_y) {
        is_part_of_front[k] = false;
        number_of_dominated_sols++;
        continue;
      }
      
      // else, check if its dominated by other solutions, or dominates them.
      for(size_t j = 0; j < k; ++j)
      {
        if(is_part_of_front[j])
        {
          if(mo_sols[j]->better_than(*mo_sols[k])) {
            // if(mo_sols[j].obj[0] <= mo_sols[i].obj[0] && mo_sols[j].obj[1] <= mo_sols[i].obj[1]) {
            // j dominates i, so set i to dominated and stop inner for loop.
            is_part_of_front[k] = false;
            number_of_dominated_sols++;
            break;
          }
          if(mo_sols[k]->better_than(*mo_sols[j]) || mo_sols[k]->same_objectives(*mo_sols[j])) {
            // if(mo_sols[i].obj[0] <= mo_sols[j].obj[0] && mo_sols[i].obj[1] <= mo_sols[j].obj[1]) {
            // i dominated j, so set j to dominated
            is_part_of_front[j] = false;
            number_of_dominated_sols++;
          }
        }
      }
    }
    
    // construct non-dominated objective vectors
    front_x.resize(K - number_of_dominated_sols);
    front_y.resize(K - number_of_dominated_sols);
    size_t ki = 0;
    for(size_t k = 0; k < K; ++k)
    {
      if(is_part_of_front[k])
      {
        front_x[ki] = mo_sols[k]->obj[0];
        front_y[ki] = mo_sols[k]->obj[1];
        ki++;
      }
    }
  }
  
  // distance to a box defined by [-infty, ref_x, -infty, ref_y]
  double uhvgrad_t::distance_to_box(double ref_x, double ref_y, double p_x, double p_y)
  {
    double nearest_x = 0.0;
    double nearest_y = 0.0;
    bool nearest_x_idx, nearest_y_idx;
    return distance_to_box(ref_x, ref_y, p_x, p_y, nearest_x, nearest_y, nearest_x_idx, nearest_y_idx);
  }
  
  // distance to a box defined by [-infty, ref_x, -infty, ref_y] and return nearest point on boundary
  double uhvgrad_t::distance_to_box(double ref_x, double ref_y, double p_x, double p_y, double & nearest_x, double & nearest_y, bool & shares_x, bool & shares_y) const
  {
    double dx = max(0.0, p_x - ref_x );
    double dy = max(0.0, p_y - ref_y );
    
    nearest_x = min(p_x, ref_x);
    nearest_y = min(p_y, ref_y);
    // assert( sqrt((nearest_x - p_x) * (nearest_x - p_x)  + (nearest_y - p_y) * (nearest_y - p_y)) == sqrt(dx*dx + dy*dy));
    
    // returns 1 if the nearest point has an x/y coordinate of the reference point
    shares_x = (nearest_x == ref_x);
    shares_y = (nearest_y == ref_y);
    
    return sqrt(dx*dx + dy*dy);
  }
  
  // Based on the Uncrowded Hypervolume improvement by the Inria group,
  // but we extened the definition to points that are outside of the reference frame
  // we compute the distance to the non-dominated area, within the reference window (r_x,r_y)
  // define the area points a(P^(i)_x, P^(i-1)_y), for i = 0...n (n =  number of points in front, i.e., obj0.size())
  // and let P^(-1)_y = r_y,   and    P^(n)_x = r_x
  double uhvgrad_t::distance_to_front(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y) const
  {
    vec_t nearest_point_on_front(2,0);
    size_t nearest_x_idx, nearest_y_idx;
    return distance_to_front(p_x, p_y, obj_x, obj_y, sorted_obj, r_x, r_y, nearest_point_on_front, nearest_x_idx, nearest_y_idx);
  }
  
  double uhvgrad_t::distance_to_front(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y, vec_t & nearest_point_on_front, size_t & nearest_x_idx, size_t & nearest_y_idx) const
  {
    nearest_point_on_front.resize(2);
    // if the front is empty, use the reference point for the distance measure
    if(obj_x.size() == 0) {
      nearest_x_idx = 0; // = out of bounds of obj_x => reference point
      nearest_y_idx = 0; // = out of bounds of obj_x => reference point
      bool shares_x, shares_y;
      return distance_to_box(r_x, r_y, p_x, p_y, nearest_point_on_front[0], nearest_point_on_front[1], shares_x, shares_y);
    }
    
    size_t n = obj_x.size();
    
    // if not available, get the sorted front
    if(sorted_obj.size() != n)
    {
      sorted_obj.resize(n);
      for (size_t i = 0; i < n; ++i) {
        sorted_obj[i] = i;
      }
      
      std::sort(std::begin(sorted_obj), std::end(sorted_obj), [&obj_x](double idx, double idy) { return obj_x[(size_t)idx] < obj_x[(size_t)idy]; });
    }
    
    double dist;
    
    // distance to the 'end' boxes
    vec_t new_nearest_point_on_front(2,0.0);
    bool shares_x, shares_y;
    double min_dist = distance_to_box(obj_x[sorted_obj[0]], r_y, p_x, p_y, nearest_point_on_front[0], nearest_point_on_front[1], shares_x, shares_y);
    nearest_x_idx = shares_x ? sorted_obj[0] : obj_x.size();
    nearest_y_idx = obj_y.size();
    
    dist = distance_to_box(r_x, obj_y[sorted_obj[n-1]], p_x, p_y, new_nearest_point_on_front[0], new_nearest_point_on_front[1], shares_x, shares_y);
    
    if(dist < min_dist) {
      min_dist = dist;
      nearest_point_on_front = new_nearest_point_on_front;
      nearest_x_idx = obj_x.size();
      nearest_y_idx = shares_y ? sorted_obj[n-1] : obj_y.size();
    }
    
    {
      // distance to 'inner' boxes
      for(size_t k = 1; k < n; ++k)
      {
        dist = distance_to_box(obj_x[sorted_obj[k]], obj_y[sorted_obj[k-1]], p_x, p_y, new_nearest_point_on_front[0], new_nearest_point_on_front[1], shares_x, shares_y);
        
        if(dist < min_dist) {
          min_dist = dist;
          nearest_point_on_front = new_nearest_point_on_front;
          nearest_x_idx = shares_x ? sorted_obj[k] : obj_x.size();
          nearest_y_idx = shares_y ? sorted_obj[k-1] : obj_y.size();
        }
      }
    }
    
    assert(min_dist >= 0); // can be 0 if its at the front!
    return min_dist;
  }
  
  double uhvgrad_t::distance_to_front_without_corner_boxes(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y, vec_t & nearest_point_on_front, size_t & nearest_x_idx, size_t & nearest_y_idx) const
  {
    nearest_point_on_front.resize(2);
    // if the front is empty, use the reference point for the distance measure
    if(obj_x.size() == 0) {
      nearest_x_idx = 0; // = obj_x.size()
      nearest_y_idx = 0; // = obj_x.size()
      bool shares_x, shares_y;
      return distance_to_box(r_x, r_y, p_x, p_y, nearest_point_on_front[0], nearest_point_on_front[1], shares_x, shares_y);
    }
    
    size_t n = obj_x.size();
    
    // if not available, get the sorted front
    if(sorted_obj.size() != n)
    {
      sorted_obj.resize(n);
      for (size_t i = 0; i < n; ++i) {
        sorted_obj[i] = i;
      }
      
      std::sort(std::begin(sorted_obj), std::end(sorted_obj), [&obj_x](double idx, double idy) { return obj_x[(size_t)idx] < obj_x[(size_t)idy]; });
    }
    
    // distance to the 'end' boxes
    vec_t new_nearest_point_on_front(2,0.0);
    bool shares_x, shares_y;
    
    double dist;
    double min_dist = 1e300;
    if(obj_x.size() == 1)
    {
      nearest_point_on_front[0] = obj_x[0];
      nearest_point_on_front[1] = obj_y[0];
      nearest_x_idx = 0;
      nearest_y_idx = 0;
      min_dist = distanceEuclidean2D(obj_x[0], obj_y[0], p_x, p_y);
    }
    else
    {
      // distance to 'inner' boxes
      for(size_t k = 1; k < n; ++k)
      {
        dist = distance_to_box(obj_x[sorted_obj[k]], obj_y[sorted_obj[k-1]], p_x, p_y, new_nearest_point_on_front[0], new_nearest_point_on_front[1], shares_x, shares_y);
        
        if(dist < min_dist) {
          min_dist = dist;
          nearest_point_on_front = new_nearest_point_on_front;
          nearest_x_idx = shares_x ? sorted_obj[k] : obj_x.size();
          nearest_y_idx = shares_y ? sorted_obj[k-1] : obj_y.size();
        }
      }
    }
    
    assert(min_dist >= 0); // can be 0 if its at the front!
    return min_dist;
  }
  
  
  double uhvgrad_t::uhv(const hicam::population_pt & mo_population) const
  {
    std::vector<vec_t> dummy_weights;
    return uhv(mo_population, false, dummy_weights);
  }
  
  double uhvgrad_t::uhv(const hicam::population_pt & mo_population, bool compute_gradient_weights, std::vector<vec_t> & gradient_weights) const
  {
    
    std::vector<bool> is_part_of_front;
    vec_t front_x;
    vec_t front_y;
    double r0 = mo_fitness_function->hypervolume_max_f0;
    double r1 = mo_fitness_function->hypervolume_max_f1;
    get_front(mo_population->sols, is_part_of_front, front_x, front_y, r0, r1);
    
    std::vector<size_t> original_idx(front_x.size());
    size_t idx = 0;
    for(size_t i = 0; i < mo_population->sols.size(); ++i)
    {
      if(is_part_of_front[i]) {
        original_idx[idx] = i;
        idx++;
      }
    }
    
    //-----------------------------------------------
    // Set objective (Hyper Volume)
    std::vector<size_t> sorted_front_order;
    double HV = compute2DHyperVolume(front_x, front_y, sorted_front_order, r0, r1);
    double UHV = HV;
    
    double penalty = 0.0;
    double penalty_factor = 1.0 /((double) (mo_population->sols.size()));
    std::vector<vec_t> nearest_point_on_front(mo_population->sols.size());
    std::vector<size_t> nearest_x_idx(mo_population->sols.size());
    std::vector<size_t> nearest_y_idx(mo_population->sols.size());
    
    for(size_t k = 0; k < number_of_mo_solutions; ++k)
    {
      
      if(!is_part_of_front[k])
      {
        double dist = distance_to_front_without_corner_boxes(mo_population->sols[k]->obj[0], mo_population->sols[k]->obj[1], front_x, front_y, sorted_front_order, r0, r1, nearest_point_on_front[k], nearest_x_idx[k], nearest_y_idx[k]);
        // double dist = distance_to_front(mo_population->sols[k]->obj[0], mo_population->sols[k]->obj[1], front_x, front_y, sorted_front_order, r0, r1, nearest_point_on_front[k], nearest_x_idx[k], nearest_y_idx[k]);
        assert(dist >= 0);
        penalty += penalty_factor * pow(dist,mo_fitness_function->number_of_objectives);
      }
    }
    
    UHV -= penalty;
    
    
    // This is an experiment, where we included dominated solutions in the
    // gradient of non-dominated solutions (i.e., slightly pulling them back/sideways)
    // It wans't better seemingly, and sounds a bit off, so we disabled it.
    bool include_ud_in_grad = false;
    
    if(compute_gradient_weights)
    {
      gradient_weights.clear();
      gradient_weights.resize(mo_population->sols.size());
      for(size_t k = 0; k < mo_population->sols.size(); ++k) {
        gradient_weights[k].resize(mo_fitness_function->number_of_objectives, 0);
      }
      
      // gradient of non-dominated solutions
      for(size_t i = 0; i < front_x.size(); ++i)
      {
        // doube negative => positive cuz we minimize -HV!
        if(i == 0) {
          gradient_weights[original_idx[sorted_front_order[i]]][0] = ( r1 - front_y[sorted_front_order[i]] );
        } else {
          gradient_weights[original_idx[sorted_front_order[i]]][0] = ( front_y[sorted_front_order[i-1]] - front_y[sorted_front_order[i]] );
        }
        
        if(i == front_x.size()-1) {
          gradient_weights[original_idx[sorted_front_order[i]]][1] = ( r0 - front_x[sorted_front_order[i]] ); // [1] = hardcoded m = 1 here
        } else {
          gradient_weights[original_idx[sorted_front_order[i]]][1] = ( front_x[sorted_front_order[i+1]] - front_x[sorted_front_order[i]] );
        }
        
        if(include_ud_in_grad)
        {
          vec_t f_shift(mo_fitness_function->number_of_objectives, 0.0);
          
          // check for dominated points that have an UD that depends on
          // the current solution
          for(size_t k = 0; k < nearest_y_idx.size(); ++k)
          {
            if(!is_part_of_front[k])
            {
              if(nearest_x_idx[k] == sorted_front_order[i]) {
                f_shift[0] += penalty_factor * 2.0 * (nearest_point_on_front[k][0] - mo_population->sols[k]->obj[0]);
              }
              
              if(nearest_y_idx[k] == sorted_front_order[i]) {
                f_shift[1] += penalty_factor * 2.0 * (nearest_point_on_front[k][1] - mo_population->sols[k]->obj[1]);
              }
            }
          }
          
          if(f_shift[0] != 0 || f_shift[1] != 0) {
            for(int m = 0; m < mo_fitness_function->number_of_objectives; ++m) {
              gradient_weights[original_idx[sorted_front_order[i]]][m] += f_shift[m];
            }
          }
        }
      }
      
      for(size_t k = 0; k < number_of_mo_solutions; ++k)
      {
        if(!is_part_of_front[k])
        {
          for(int m = 0; m < mo_fitness_function->number_of_objectives; ++m) {
            gradient_weights[k][m] = penalty_factor * 2.0 * (mo_population->sols[k]->obj[m] - nearest_point_on_front[k][m]);
          }
        }
        
        // normalize the gradients
        double grad_length = 0.0;
        for(int m = 0; m < mo_fitness_function->number_of_objectives; ++m) {
          grad_length += gradient_weights[k][m] * gradient_weights[k][m];
          // grad_length += gradient_weights[k][m];
        }
        grad_length = sqrt(grad_length);
        
        if(grad_length != 0) {
          for(int m = 0; m < mo_fitness_function->number_of_objectives; ++m) {
            gradient_weights[k][m] /= grad_length;
          }
        }
      }
    }
    
    return UHV;
    
  }
  
  double uhvgrad_t::compute2DHyperVolume(const vec_t & obj0, const vec_t & obj1, std::vector<size_t> & sorted_obj, double max_0, double max_1) const
  {
    size_t n = obj0.size();
    if (n == 0) { return 0.0; }
    
    if(sorted_obj.size() != n)
    {
      sorted_obj.resize(n);
      for (size_t i = 0; i < n; ++i) {
        sorted_obj[i] = i;
      }
      
      std::sort(std::begin(sorted_obj), std::end(sorted_obj), [&obj0](double idx, double idy) { return obj0[(size_t)idx] < obj0[(size_t)idy]; });
    }
    
    double area = (max_0 - min(max_0, obj0[sorted_obj[n - 1]])) * (max_1 - min(max_1, obj1[sorted_obj[n - 1]]));
    for (int i = (int) (n - 2); i >= 0; i--) {
      
      assert(obj0[sorted_obj[i+1]] > obj0[sorted_obj[i]] );
      double d = (min(max_0, obj0[sorted_obj[i + 1]]) - min(max_0, obj0[sorted_obj[i]])) * (max_1 - min(max_1, obj1[sorted_obj[i]]));
      assert(d>0);
      area += d;
    }
    
    return area;
  }
  
  
}
