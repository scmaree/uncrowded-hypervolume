/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA


*/

#include "sofomore_fitness.hpp"
#include "sofomore.hpp"
#include "population.hpp"
#include "mathfunctions.hpp"
// #include "hgml.hpp"

namespace hillvallea
{
  // Constructor
  sofomore_t::sofomore_t
    (
     hicam::fitness_pt mo_fitness_function,
     size_t number_of_mo_solutions,
     size_t number_of_so_solutions_per_shadow_population,
     int local_optimizer_index,
     bool collect_all_mo_sols_in_archive,
     size_t elitist_archive_size_target,
     size_t approximation_set_size,
     const hicam::vec_t & lower_init_ranges,
     const hicam::vec_t & upper_init_ranges,
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
    this->number_of_so_solutions_per_shadow_population = number_of_so_solutions_per_shadow_population;
    this->local_optimizer_index = local_optimizer_index;
    this->collect_all_mo_sols_in_archive = collect_all_mo_sols_in_archive;
    this->elitist_archive_size_target = elitist_archive_size_target;
    this->approximation_set_size = approximation_set_size;
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

    // if the elitist archive size is set to zero, we (effectively) remove the limit.
    // similarly for the approximation_set_size, if its 0, its disabled and we return the entire elitist_archive.
    if(this->elitist_archive_size_target == 0) {
      this->elitist_archive_size_target = maximum_number_of_evaluations;
    }
    
    // allocate archive
    if(this->collect_all_mo_sols_in_archive) {
      rng_pt elite_rng = std::make_shared<rng_t>(1001032); //
      elitist_archive = std::make_shared<hicam::elitist_archive_t>(this->elitist_archive_size_target, elite_rng);
      
      hicam::vec_t r(2);
      r[0] = mo_fitness_function->hypervolume_max_f0;
      r[1] = mo_fitness_function->hypervolume_max_f1;
      elitist_archive->set_use_hypervolume_for_size_control(false, r);
    }
    
    // Initialize the MO-objective function
    mo_population = std::make_shared<hicam::population_t>();
    fitness_function = std::make_shared<sofomore_fitness_t>(this->mo_fitness_function, &mo_population->sols, this->collect_all_mo_sols_in_archive, elitist_archive);

    mo_fitness_function->get_param_bounds(lower_param_bounds, upper_param_bounds);
    
    for(size_t i = 0; i < lower_param_bounds.size(); ++i) {
      if(this->lower_init_ranges[i] < lower_param_bounds[i]) { this->lower_init_ranges[i] = lower_param_bounds[i]; }
      if(this->upper_init_ranges[i] > upper_param_bounds[i]) { this->upper_init_ranges[i] = upper_param_bounds[i]; }
    }
    
    so_lower_param_bounds.resize(lower_param_bounds.size());
    so_upper_param_bounds.resize(upper_param_bounds.size());
    
    for(size_t d = 0; d < so_lower_param_bounds.size(); ++d) {
      so_lower_param_bounds[d] = lower_param_bounds[d];
      so_upper_param_bounds[d] = upper_param_bounds[d];
    }
    
    // this is horrible impemented, but hey, it works.
    if (this->number_of_so_solutions_per_shadow_population <= 0){
      optimizer_pt dummy_opt = init_optimizer(local_optimizer_index, fitness_function->number_of_parameters, so_lower_param_bounds, so_upper_param_bounds, 1.0, fitness_function, rng);
      this->number_of_so_solutions_per_shadow_population = dummy_opt->recommended_popsize(fitness_function->number_of_parameters);
    }
    
    
  }

  sofomore_t::~sofomore_t() { }
  
  
  
  // Write statistic Files
  //----------------------------------------------------------------------------------
  void sofomore_t::new_statistics_file()
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
   // std::cout
    << "Gen    MO-Evals   Time      IGD                    GD                     HV                     size Archive_IGD            Archive_GD             Archive_HV             size" << std::endl;
    
  }
  
  void sofomore_t::close_statistics_file() {
    statistics_file.close();
  }
  
  void sofomore_t::write_statistics_line(size_t number_of_generations, const hicam::population_pt & mo_population, const hicam::elitist_archive_pt & elitist_archive)
  {
    clock_t current_time = clock();
    double runtime = double(current_time - starting_time) / CLOCKS_PER_SEC;
    
    //------------------------------------------
    // create approximation set
    hicam::rng_pt rng = std::make_shared<hicam::rng_t>(1000); // not used here
    hicam::elitist_archive_t approximation_set(1000, rng);
    
    for(size_t i = 0; i < mo_population->size(); ++i) {
      approximation_set.updateArchive(mo_population->sols[i]);
    }
    
    // Compute measures
    double IGD = 0.0;
    if (mo_fitness_function->igd_available) {
       IGD = approximation_set.computeIGD(mo_fitness_function->pareto_set);
    }

    // Compute measures
    double GD = 0.0;
    if (mo_fitness_function->igd_available) {
      if(mo_fitness_function->analytical_gd_avialable) {
        GD = approximation_set.computeAnalyticGD(*mo_fitness_function);
      } else {
        GD = approximation_set.computeGD(mo_fitness_function->pareto_set);
      }
    }
    
    double HV = 0.0;
    if(mo_fitness_function->number_of_objectives == 2) {
      HV = approximation_set.compute2DHyperVolume(mo_fitness_function->hypervolume_max_f0, mo_fitness_function->hypervolume_max_f1);
    }
    
    statistics_file
    //std::cout
    << std::setw(3) << number_of_generations
    << " " << std::setw(8) << number_of_evaluations
    << std::setw(12) << std::scientific << std::setprecision(3) << runtime
    // << std::setw(25) << std::scientific << std::setprecision(16) <<  best.f
    // << std::setw(12) << std::scientific << std::setprecision(3) <<  best.constraint
    << " " << std::setw(20) << std::scientific << std::setprecision(16) << IGD
    << " " << std::setw(20) << std::scientific << std::setprecision(16) << GD
    << " " << std::setw(20) << std::scientific << std::setprecision(16) << HV
    << " " << std::setw(4) << approximation_set.actualSize();
    
    
    if(collect_all_mo_sols_in_archive && elitist_archive != nullptr)
    {
      hicam::elitist_archive_t large_approximation_set = hicam::elitist_archive_t(approximation_set_size, rng);
      large_approximation_set.computeApproximationSet(approximation_set_size, elitist_archive, false);
      
      
      // Compute measures
      double large_IGD = 0.0;
      if (mo_fitness_function->igd_available) {
         large_IGD = large_approximation_set.computeIGD(mo_fitness_function->pareto_set);
      }
      
      // Compute measures
      double large_GD = 0.0;
      if (mo_fitness_function->igd_available) {
        
        if(mo_fitness_function->analytical_gd_avialable) {
          large_GD = large_approximation_set.computeAnalyticGD(*mo_fitness_function);
        } else {
          large_GD = large_approximation_set.computeGD(mo_fitness_function->pareto_set);
        }
      }
      
      double large_HV = 0.0;
      if(mo_fitness_function->number_of_objectives == 2) {
        large_HV = large_approximation_set.compute2DHyperVolume(mo_fitness_function->hypervolume_max_f0, mo_fitness_function->hypervolume_max_f1);
      }
      
      statistics_file
      << " " << std::setw(20) << std::scientific << std::setprecision(16) << large_IGD
      << " " << std::setw(20) << std::scientific << std::setprecision(16) << large_GD
      << " " << std::setw(20) << std::scientific << std::setprecision(16) << large_HV
          << " " << std::setw(4) << large_approximation_set.actualSize();
    }
    else
    {
      statistics_file
      << " " << std::setw(20) << std::scientific << std::setprecision(16) << 0
      << " " << std::setw(20) << std::scientific << std::setprecision(16) << 0
      << " " << std::setw(20) << std::scientific << std::setprecision(16) << 0
      << " " << std::setw(4) << 0;
    }
    
    
    
    statistics_file << std::endl;
  }
  
  // Termination Criteria
  //-------------------------------------------------------------------------------
  bool sofomore_t::terminate_on_runtime() const
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
  
  
  void sofomore_t::run()
  {
    //---------------------------------------------
    // reset all runlogs (in case hillvallea is run multiple time)
    starting_time = clock();
    success = false;
    terminated = false;
    number_of_evaluations = 0;
    number_of_generations = 0;
    
    //---------------------------------------------
    // output
    if(write_generational_statistics) {
      new_statistics_file();
    }
    
    // allocate MO_population
    // The MO population is evolved over time, i.e, it is something like a 1+lambda-ES
    //----------------------------------------------
    if(mo_fitness_function->redefine_random_initialization) {
      mo_fitness_function->init_solutions_randomly(mo_population, number_of_mo_solutions, lower_init_ranges, upper_init_ranges, 0, rng);
    }
    else
    {
      mo_population->fill_uniform(number_of_mo_solutions, mo_fitness_function->number_of_parameters, lower_init_ranges, upper_init_ranges, rng);
    }
    for(size_t k = 0; k < mo_population->sols.size(); ++k) {
      mo_fitness_function->evaluate(mo_population->sols[k]);
      number_of_evaluations++;
      
      if(collect_all_mo_sols_in_archive) {
        elitist_archive->updateArchive(mo_population->sols[k],true);
      }
    }
    
    // sort sols
    std::sort(mo_population->sols.begin(), mo_population->sols.end(), hicam::solution_t::strictly_better_solution_via_pointers_obj0_unconstraint);
    
    
    // allocate shadow populations
    //---------------------------------------------
    std::vector<population_pt> shadow_populations(number_of_mo_solutions);
    
    for(size_t k = 0; k < shadow_populations.size(); ++k)
    {
      
      shadow_populations[k] = std::make_shared<population_t>();
      
      // init a new SO sol, with params from the MO sol.
      solution_pt sol = std::make_shared<solution_t>(fitness_function->number_of_parameters);
      sol->mo_reference_sols.push_back(std::make_shared<hicam::solution_t>(*mo_population->sols[0])); // push a COPY of the reference sol to MO_reference_sols.
      
      for(size_t d = 0; d < fitness_function->number_of_parameters; ++d) {
        sol->param[d] = sol->mo_reference_sols[0]->param[d];
      }
      
      // push the new sol into the shadow population.
      shadow_populations[k]->sols.push_back(sol);
    }
    
    

    // fill the shadow populations uniformly with the remaining solutions
    //-----------------------------------------------
    
    // sample a set of solutions, sort them, and add them to the respective shadow population.
    // skip 1 because we already set it above.
    for(size_t i = 1; i < number_of_so_solutions_per_shadow_population; ++i)
    {
      // allocate temp pop
      hicam::population_pt temp_pop = std::make_shared<hicam::population_t>();
      
      // fill it.
      if(mo_fitness_function->redefine_random_initialization) {
        mo_fitness_function->init_solutions_randomly(temp_pop, shadow_populations.size(), lower_init_ranges, upper_init_ranges, 0, rng);
      }
      else {
        temp_pop->fill_uniform(shadow_populations.size(), mo_fitness_function->number_of_parameters, lower_init_ranges, upper_init_ranges, rng);
      }
      
      for(size_t k = 0; k < temp_pop->sols.size(); ++k) {
        mo_fitness_function->evaluate(temp_pop->sols[k]);
        number_of_evaluations++;
        
        if(collect_all_mo_sols_in_archive) {
          elitist_archive->updateArchive(temp_pop->sols[k],true);
        }
      }
      
      std::sort(temp_pop->sols.begin(), temp_pop->sols.end(), hicam::solution_t::strictly_better_solution_via_pointers_obj0_unconstraint);
      
      // make a SO-sol from the MO-sol and push it back to its corresponding shadow population
      for(size_t k = 0; k < shadow_populations.size(); ++k)
      {
        solution_pt sol = std::make_shared<solution_t>(fitness_function->number_of_parameters);
        sol->mo_reference_sols.push_back(std::make_shared<hicam::solution_t>(*temp_pop->sols[k]));
        
        for(size_t d = 0; d < fitness_function->number_of_parameters; ++d) {
          sol->param[d] = sol->mo_reference_sols[0]->param[d];
        }
        
       shadow_populations[k]->sols.push_back(sol);
      }
    }
    
    if(collect_all_mo_sols_in_archive) {
      elitist_archive->adaptArchiveSize();
    }
    // evaluate the shadow populations
    //--------------------------------------------
    for(size_t k = 0; k < shadow_populations.size(); ++k)
    {
      
      // hicam::solution_pt hold_out_sol = mo_population->sols[k];
      mo_population->sols[k] = nullptr;
      for(size_t i = 0; i < shadow_populations[k]->sols.size(); ++i) {
        fitness_function->evaluate(shadow_populations[k]->sols[i]);
      }
      
      shadow_populations[k]->sort_on_fitness();
      
      // UPDATE MO_population
      mo_population->sols[k] = std::make_shared<hicam::solution_t>(*shadow_populations[k]->sols[0]->mo_reference_sols[0]);
      
    }
    
    // Allocate optimizers
    //--------------------------------------------
    
    // Init Bandwidth: Not used in current implementation, but is a good value when init optimizers form a single solution.
    double scaled_search_volume = 1.0;
    for(size_t d = 0; d < upper_init_ranges.size(); ++d) {
      scaled_search_volume *= pow((upper_init_ranges[d] - lower_init_ranges[d]),1.0/fitness_function->number_of_parameters);
    }
    double init_bandwidth = sqrt(scaled_search_volume);
    // end init bandwidth
    
    std::vector<optimizer_pt> local_optimizers;
    
    for(size_t k = 0; k < shadow_populations.size(); ++k)
    {
      optimizer_pt opt = init_optimizer(local_optimizer_index, fitness_function->number_of_parameters, so_lower_param_bounds, so_upper_param_bounds, init_bandwidth, fitness_function, rng);
      opt->initialize_from_population(shadow_populations[k], number_of_so_solutions_per_shadow_population);
      local_optimizers.push_back(opt);
    }
    
    // optimization loop
    //---------------------------------------------
    bool restart = true;
    while(restart)
    {
     
      if (write_generational_statistics && (number_of_generations < 50 || number_of_generations % 50 == 0 ) ) {
        write_statistics_line(number_of_generations, mo_population, elitist_archive);
      }
      
      int n = (int) local_optimizers.size();
      int *order = randomPermutation(n, *rng);
      
      bool all_terminated = true;
      for (size_t ki = 0; ki < local_optimizers.size(); ++ki)
      {
        size_t k = order[ki];
        
        if (maximum_number_of_evaluations > 0 && number_of_evaluations >= maximum_number_of_evaluations) {
          std::cout << "  Terminated core search algorithm because function evaluations limit reached" << std::endl;
          restart = false;
          break;
        }
        
        // stop if we run out of time.
        if (terminate_on_runtime()) {
          std::cout << "  Terminated core search algorithm because time limit reached" << std::endl;
          restart = false;
          break;
        }
        
        // stop if the vtr is hit
        if (use_vtr != 0)
        {
          bool vtr_hit = false;
          
          if(use_vtr == 1) {
            vtr_hit = -mo_population->compute2DHyperVolume(mo_fitness_function->hypervolume_max_f0, mo_fitness_function->hypervolume_max_f1) < vtr;
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
            std::cout << "  Terminated core search algorithm because VTR reached! Yay" << std::endl;
            restart = false;
            success = true;
            break;
          }
        }
        
        hicam::solution_pt hold_out_sol = mo_population->sols[k];
        mo_population->sols[k] = nullptr;
        
        // update population
        // do not count these evalutations, as the MO-sol is not re-evaluated
        // re-evaluating the best is crucial, re-evaluating the remaining solutions only marginally improves the method.
        fitness_function->evaluate(local_optimizers[k]->best);
        for(size_t i = 0; i < local_optimizers[k]->pop->size(); ++i) {
          fitness_function->evaluate(local_optimizers[k]->pop->sols[i]);
        }
        
        if(collect_all_mo_sols_in_archive) {
          elitist_archive->adaptArchiveSize();
        }
        
        // if the cluster is active, and after checking it, it is terminated,
        // we add the best solution to the elitist archive
        bool already_terminated = !local_optimizers[k]->active;
        if (local_optimizers[k]->checkTerminationCondition())
        {
          if(!already_terminated) {
            std::cout << "  Terminated core search algorithm because of internal reasons (" << number_of_evaluations << " fevals)." << std::endl;
          }
          // UPDATE MO_population (elitist archive is updated in the fitness function itself
          mo_population->sols[k] = hold_out_sol;
          continue;
        } else 
        
        // if it is still active, run a generation of the local optimizer
        if (local_optimizers[k]->active)
        {
          all_terminated = false;
          
          local_optimizers[k]->generation(number_of_so_solutions_per_shadow_population, number_of_evaluations);
        }
        
        
        // UPDATE MO_population (elitist archive is updated in the fitness function itself
        mo_population->sols[k] = std::make_shared<hicam::solution_t>(*local_optimizers[k]->best.mo_reference_sols[0]);
      
      }
      // end optimizer for-loop
      free(order);
      
      if(all_terminated || !restart || success) {
        break;
      }
      
      number_of_generations++;
      
    }

   if (write_generational_statistics && (number_of_generations > 50 || number_of_generations % 50 != 0 ) ) {
      write_statistics_line(number_of_generations, mo_population, elitist_archive);
    }
    
    if (write_generational_statistics) {
      close_statistics_file();
    }
    
    std::stringstream ss;
    ss << write_directory << "best_final" << file_appendix << ".dat";
    mo_population->writeToFile(ss.str().c_str());
    
  }


}
