/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA


*/

#include "hillvallea.hpp"
#include "population.hpp"
#include "mathfunctions.hpp"
#include "fitness.h"
#include "hgml.hpp"

namespace hillvallea
{
  // Constructor
  hillvallea_t::hillvallea_t(
    fitness_pt fitness_function,
    const int local_optimizer_index,
    const int  number_of_parameters,
    const vec_t & lower_init_ranges,
    const vec_t & upper_init_ranges,
    const int maximum_number_of_evaluations,
    const int maximum_number_of_seconds,
    const double vtr,
    const bool use_vtr,
    const int random_seed,
    const bool write_generational_solutions,
    const bool write_generational_statistics,
    const std::string write_directory,
    const std::string file_appendix
  )
  {

    // copy all settings
    this->fitness_function = fitness_function;
    this->local_optimizer_index = local_optimizer_index;
    this->number_of_parameters = number_of_parameters;
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

    init_default_params();
  }


  // Quick Constructor
  hillvallea_t::hillvallea_t(
    fitness_pt fitness_function,
    const int  number_of_parameters,
    const vec_t & lower_init_ranges,
    const vec_t & upper_init_ranges,
    const int maximum_number_of_evaluations,
    const int random_seed
  )
  {

    // copy all settings
    this->fitness_function = fitness_function;
    this->cluster_alg = 0;
    this->local_optimizer_index = 0; // default: AMu
    this->number_of_parameters = number_of_parameters;
    this->lower_init_ranges = lower_init_ranges;
    this->upper_init_ranges = upper_init_ranges;
    this->maximum_number_of_evaluations = maximum_number_of_evaluations;
    this->maximum_number_of_seconds = 0;
    this->vtr = 0;
    this->use_vtr = false;
    this->random_seed = random_seed;
    this->write_generational_solutions = false;
    this->write_generational_statistics = false;
    this->write_directory = "";
    this->file_appendix = "";

    rng = std::make_shared<std::mt19937>((unsigned long)(random_seed));
    std::uniform_real_distribution<double> unif(0, 1);

    init_default_params();
  }


  hillvallea_t::~hillvallea_t()
  {

  }
  
  void hillvallea_t::init_default_params()
  {
    // Parameters of the recursion scheme
    //---------------------------------------------
    population_size_initializer = 6.0;
    population_size_incrementer = 2.0;
    cluster_size_initializer = 0.8;
    cluster_size_incrementer = 1.1;
    add_elites_max_trials = 5;
    selection_fraction_multiplier = 1.0; // this doesn't make things better, disabled it.
    
    TargetTolFun = 1e-5;
    
    scaled_search_volume = 1.0; // reduces round-off errors and +inf errors for large number_of_parameters;
    for(size_t i = 0; i < upper_init_ranges.size(); ++i) {
      scaled_search_volume *= pow((upper_init_ranges[i] - lower_init_ranges[i]),1.0/number_of_parameters);
    }
    write_elitist_archive = false;
    // clustering_max_number_of_neighbours = (size_t)(number_of_parameters + 1);

    // such that for d={1,2} still Nn = d+1
    if( number_of_parameters <= 3 ) {
      clustering_max_number_of_neighbours = number_of_parameters + 1;
    }
    else {
      clustering_max_number_of_neighbours = 3 + log(number_of_parameters);
    }

    
    fitness_function->get_param_bounds(lower_param_bounds, upper_param_bounds);
    
    for(size_t i = 0; i < lower_param_bounds.size(); ++i) {
      if(lower_init_ranges[i] < lower_param_bounds[i]) { lower_init_ranges[i] = lower_param_bounds[i]; }
      if(upper_init_ranges[i] > upper_param_bounds[i]) { upper_init_ranges[i] = upper_param_bounds[i]; }
    }
    
    evalute_with_gradients = false;
    
  }

  // Write statistic Files
  //----------------------------------------------------------------------------------
  void hillvallea_t::new_statistics_file()
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

    statistics_file << "Pop  Cluster    Gen   Evals        Time No.Elites  Best-elite   Average-obj       Std-obj" << std::endl;

  }
  
  void hillvallea_t::new_statistics_file_serial() {
    new_statistics_file_serial(false);
  }

  void hillvallea_t::new_statistics_file_serial(bool use_clustering)
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
    << "Gen    Evals          Time                    Best-f   Best-constr   Average-obj       Std-obj    Avg-constr    Std-constr "
    << fitness_function->write_solution_info_header(use_clustering) << std::endl;
  }
  
  void hillvallea_t::write_statistics_line_population(const population_t & pop, const std::vector<optimizer_pt> & local_optimizers, const std::vector<solution_pt> & elitist_archive)
  {
    
    
    clock_t current_time = clock();
    double runtime = double(current_time - starting_time) / CLOCKS_PER_SEC;
    
    solution_t best = solution_t(*pop.first());
    
    for (auto sol = elitist_archive.begin(); sol != elitist_archive.end(); ++sol)
    {
      if (solution_t::better_solution(**sol, best))
      {
        best = solution_t(**sol);
      }
    }
    
    statistics_file
    << std::setw(3) << number_of_generations
    << std::setw(9) << local_optimizers.size()
    << std::setw(7) << 0
    << std::setw(8) << number_of_evaluations
    << std::setw(12) << std::scientific << std::setprecision(3) << runtime
    << std::setw(10) << elitist_archive.size()
    << std::setw(12) << std::scientific << std::setprecision(3) <<  best.f
    << std::setw(14) << std::scientific << std::setprecision(3) << pop.average_fitness()
    << std::setw(14) << std::scientific << std::setprecision(3) << pop.relative_fitness_std()
    << std::endl;
  }
  
  
  void hillvallea_t::write_statistics_line_serial(const population_t & pop, size_t number_of_generations, const solution_t & best) {
    write_statistics_line_serial(pop, number_of_generations, best, false);
  }
  
  void hillvallea_t::write_statistics_line_serial(const population_t & pop, size_t number_of_generations, const solution_t & best, bool use_clustering)
  {
    
    clock_t current_time = clock();
    double runtime = double(current_time - starting_time) / CLOCKS_PER_SEC;
    
    statistics_file
    << std::setw(3) << number_of_generations
    << " " << std::setw(8) << number_of_evaluations
    << " " << std::setw(13) << std::scientific << std::setprecision(3) << runtime
    << " " << std::setw(25) << std::scientific << std::setprecision(16) << best.f
    << " " << std::setw(13) << std::scientific << std::setprecision(3) << best.constraint
    << " " << std::setw(13) << std::scientific << std::setprecision(3) << pop.average_fitness()
    << " " << std::setw(13) << std::scientific << std::setprecision(3) << pop.relative_fitness_std()
    << " " << std::setw(13) << std::scientific << std::setprecision(3) << pop.average_constraint()
    << " " << std::setw(13) << std::scientific << std::setprecision(3) << pop.relative_constraint_std()
    << " " << fitness_function->write_additional_solution_info(best, elitist_archive, use_clustering) << std::endl;
  }
  
  
  
  void hillvallea_t::write_statistics_line_cluster(const population_t & cluster_pop, int cluster_number, int cluster_generation, const std::vector<solution_pt> & elitist_archive)
  {
    
    
    clock_t current_time = clock();
    double runtime = double(current_time - starting_time) / CLOCKS_PER_SEC;
    
    solution_t best = solution_t(*cluster_pop.first());
    
    for (auto sol = elitist_archive.begin(); sol != elitist_archive.end(); ++sol)
    {
      if (solution_t::better_solution(**sol, best))
      {
        best = solution_t(**sol);
      }
    }
    
    statistics_file
    << std::setw(3) << ""
    << std::setw(9) << cluster_number
    << std::setw(7) << cluster_generation
    << std::setw(8) << number_of_evaluations
    << std::setw(12) << std::scientific << std::setprecision(3) << runtime
    << std::setw(10) << elitist_archive.size()
    << std::setw(12) << std::scientific << std::setprecision(3) << best.f
    << std::setw(14) << std::scientific << std::setprecision(3) << cluster_pop.average_fitness()
    << std::setw(14) << std::scientific << std::setprecision(3) << cluster_pop.relative_fitness_std()
    << std::endl;
  }

  
  void hillvallea_t::close_statistics_file()
  {
    statistics_file.close();
  }


  // Write population
  //-------------------------------------------------------------------------------
  void hillvallea_t::write_population_file(population_pt pop, std::vector<optimizer_pt> & local_optimizers) const
  {
    std::ofstream file;
    std::stringstream ss;
    ss << write_directory << "population" << std::setw(5) << std::setfill('0') << number_of_generations << "_inital_population" << file_appendix << ".dat";
    file.open(ss.str(), std::ofstream::out | std::ofstream::trunc);
    assert(file.is_open());

    for (auto sol = pop->sols.begin(); sol != pop->sols.end(); ++sol)
    {
      file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << (*sol)->param << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << (*sol)->f << " " << (*sol)->constraint << std::endl;
    }

    file.close();
  }

  // Write selection
  //-------------------------------------------------------------------------------
  void hillvallea_t::write_selection_file(population_pt pop, std::vector<optimizer_pt> & local_optimizers) const
  {

    // print clusters to file so that i can check them in matlab.
    std::ofstream file;
    std::stringstream ss;
    ss << write_directory << "population" << std::setw(5) << std::setfill('0') << number_of_generations << "_selection" << file_appendix << ".dat";
    file.open(ss.str(), std::ofstream::out | std::ofstream::trunc);
    assert(file.is_open());

    for (auto sol = pop->sols.begin(); sol != pop->sols.end(); ++sol)
    {
      file << (*sol)->param << (*sol)->f << " " << (*sol)->constraint << std::endl;
    }

    file.close();
  }


  // Write clusters
  //-------------------------------------------------------------------------------
  void hillvallea_t::write_cluster_population(int generation_nuber, size_t cluster_number, int cluster_generation, population_pt pop) const
  {
    std::ofstream file;
    std::stringstream ss;
    ss << write_directory << "population" << std::setw(5) << std::setfill('0') << generation_nuber << "_cluster" << std::setw(5) << std::setfill('0') << cluster_number << "_generation" << cluster_generation << file_appendix << ".dat";
    file.open(ss.str(), std::ofstream::out | std::ofstream::trunc);
    assert(file.is_open());

    for (auto sol = pop->sols.begin(); sol != pop->sols.end(); ++sol)
    {
      file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << (*sol)->param << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << (*sol)->f << " " << (*sol)->constraint << std::endl;
    }

    file.close();

  }
  
  void hillvallea_t::write_elitist_archive_file(const std::vector<solution_pt> & elitist_archive, bool final) const
  {
    std::ofstream file;
    std::string filename;
    
    if(final) {
      filename = write_directory + "elites" + file_appendix + ".dat";
    }
    else
    {
      std::stringstream ss;
      ss << write_directory << "population" << std::setw(5) << std::setfill('0') << number_of_generations << "_elites" << file_appendix << ".dat";
      filename = ss.str();
    }
    file.open(filename, std::ofstream::out | std::ofstream::trunc);
    assert(file.is_open());

    int prescision = std::numeric_limits<long double>::digits10 + 1;
    
    for (size_t i = 0; i < elitist_archive.size(); ++i) {
      file
      << std::fixed << std::setw(prescision + 5) << std::setprecision(prescision + 1) << elitist_archive[i]->param
      << std::fixed << std::setw(prescision + 5) << std::setprecision(prescision + 1) << elitist_archive[i]->f
      << std::fixed << std::setw(prescision + 5) << std::setprecision(prescision + 1) << elitist_archive[i]->constraint;

      if (i == 0 && use_vtr)
        file << " " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << vtr << " " << success;

      file << std::endl;
    }
  }
  
  void hillvallea_t::write_CEC2013_niching_file(bool final)
  {
    
    std::string filename;
    if(final) {
      std::stringstream ss;
      ss << write_directory << "niching_archive_core" << local_optimizer_index << "_cluster" << cluster_alg << "_" << fitness_function->name() << "_run" << std::setw(3) << std::setfill('0') << random_seed << "_final.dat";
      filename = ss.str();
      
      filename = write_directory + "niching_archive" + file_appendix + ".dat";
    }
    else
    {
      std::stringstream ss;
      ss << write_directory << "niching_archive_core" << local_optimizer_index << "_cluster" << cluster_alg << "_" << fitness_function->name() << "_run" << std::setw(3) << std::setfill('0') << random_seed << "_" << number_of_generations <<".dat";
      filename = ss.str();
    }
    
    std::ofstream file;
    file.open(filename, std::ofstream::out | std::ofstream::trunc);
    assert(file.is_open());
    
    int precision = std::numeric_limits<long double>::digits10 + 1;
    
    for (size_t i = 0; i < elitist_archive.size(); ++i) {
      file
      << std::fixed << std::setw(precision + 5) << std::setprecision(precision + 1) << elitist_archive[i]->param << " = "
      << std::fixed << std::setw(precision + 5) << std::setprecision(precision + 1) << elitist_archive[i]->f << " @ "
      << std::fixed << std::setw(precision + 5) << elitist_archive[i]->feval_obtained << " "
      << std::fixed << std::setw(precision + 5) << std::setprecision(precision + 1) << elitist_archive[i]->time_obtained;
      
      file << std::endl;
    }
    
  }

  // Termination Criteria
  //-------------------------------------------------------------------------------
  bool hillvallea_t::terminate_on_runtime() const
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

  bool hillvallea_t::terminate_on_approaching_elite(optimizer_t & local_optimizer, std::vector<solution_pt> & elite_candidates)
  {
    if(local_optimizer.pop->sols[0]->constraint > 0) {
      return false;
    }
    
    // find the nearest (candidate) elite that has similar or better fitness
    solution_pt nearest_elite;
    double distance_to_nearest_elite = 1e300;
    double distance;
    double best_fitness_so_far = local_optimizer.pop->sols[0]->f;
    // double TargetTolFun = 1e-5;
    if (elitist_archive.size() > 0)
    {
      // find the nearest elite
      for (size_t j = 0; j < elitist_archive.size(); ++j)
      {

        // only consider elites that have better fitness
        if (elitist_archive[j]->f < (local_optimizer.pop->sols[0]->f + TargetTolFun))
        {

          distance = elitist_archive[j]->param_distance(*local_optimizer.pop->sols[0]);
          if (distance < distance_to_nearest_elite)
          {
            distance_to_nearest_elite = distance;
            nearest_elite = elitist_archive[j];
          }

          // also find the best elite in the archive for later
          if (elitist_archive[j]->f < best_fitness_so_far) {
            best_fitness_so_far = elitist_archive[j]->f;
          }
        }

      }
    }

    for (size_t j = 0; j < elite_candidates.size(); ++j)
    {
      // only consider good elite candidates
      if (elite_candidates[j]->f < (best_fitness_so_far + TargetTolFun))
      {
        distance = elite_candidates[j]->param_distance(*local_optimizer.pop->sols[0]);

        if (distance < distance_to_nearest_elite) {
          distance_to_nearest_elite = distance;
          nearest_elite = elite_candidates[j];
        }
      }
    }

    if (nearest_elite != nullptr)
    {
      
      if(local_optimizer.number_of_generations < 0.5 * nearest_elite->generation_obtained) {
        return false;
      }
      
      if (check_edge(*nearest_elite, *local_optimizer.pop->sols[0], 5)) {
        // std::cout << "nearest elite = [" << nearest_elite->param[0] << "] and current sol = [ " << local_optimizer.pop->sols[0]->param[0] << "]" << std::endl;
        return true;
      }
    }

    return false;
  }

  bool hillvallea_t::terminate_on_converging_to_local_optimum(optimizer_t & local_optimizer, std::vector<solution_pt> & elite_candidates)
  {

    // if this is the first, never terminate.
    if (elite_candidates.size() == 0 && elitist_archive.size() == 0) {
      return false;
    }
    
    if(local_optimizer.pop->sols[0]->constraint != 0) {
      return false;
    }

    // find best solution in the archive and candidates
    //----------------------------------------------------------
    double best = 1e308;
    double max_number_of_generations_to_obtain_elite = 0;

    for (size_t i = 0; i < elitist_archive.size(); ++i) {
      if (elitist_archive[i]->f < best) {
        best = elitist_archive[i]->f;
      }

      if (elitist_archive[i]->generation_obtained > max_number_of_generations_to_obtain_elite) {
        max_number_of_generations_to_obtain_elite = elitist_archive[i]->generation_obtained;
      }
    }

    for (size_t i = 0; i < elite_candidates.size(); ++i) {
      if (elite_candidates[i]->f < best) {
        best = elite_candidates[i]->f;
      }

      if (elite_candidates[i]->generation_obtained > max_number_of_generations_to_obtain_elite) {
        max_number_of_generations_to_obtain_elite = elite_candidates[i]->generation_obtained;
      }
    }
    
    // Only terminate if the local optimizer if its best is significantly worse than the best elite
    if (local_optimizer.pop->sols[0]->f > best + TargetTolFun)
    {

      // compute the time to optimum
      //----------------------------------------------------------
      int lookback_window = std::min((int)local_optimizer.average_fitness_history.size(), 5);

      if (lookback_window < 5) {
        return false;
      }

      // terminate only if the recent averages are all decreasing
      // note, if this passes, it implies curr_dto < prev_dto,
      // and thus tto > 0
      for (size_t i = 0; i < lookback_window-1; ++i)
      {
        if (local_optimizer.average_fitness_history[local_optimizer.average_fitness_history.size() - lookback_window + i] 
            < local_optimizer.average_fitness_history[local_optimizer.average_fitness_history.size() - lookback_window + i + 1]) {
          return false;
        }
      }

      double curr_fitness = local_optimizer.average_fitness_history.back();
      double prev_fitness = local_optimizer.average_fitness_history[local_optimizer.average_fitness_history.size() - lookback_window];

      double curr_dto = curr_fitness - (best + TargetTolFun);
      double prev_dto = prev_fitness - (best + TargetTolFun);

      // this is the best so far, let it run
      if (curr_dto <= 0.0) {
        return false;
      }

      // normal case     
      double dto_vtr = 1e-12;
      double cr5 = (curr_dto - prev_dto) / prev_dto;
      double cr = cr5; // pow((1 + cr5), 1.0 / lookback_window) - 1;

      double tto = lookback_window * log(dto_vtr / curr_dto) / log(1.0 + cr);
      // std::cout << tto << "\n";
      if ((local_optimizer.number_of_generations + tto) > 50 * max_number_of_generations_to_obtain_elite) {
        return true;
      }

    }

    return false;

  }

  //----------------------------------------------------------------------------------------------
  // samples an initial population uniformly random, clusters it into a set of local_optimizers
  void hillvallea_t::initialize(population_pt pop, size_t population_size, double selection_fraction_multiplier, std::vector<optimizer_pt> & local_optimizers, const std::vector<solution_pt> & elitist_archive, bool enable_clustering, size_t target_clustersize)
  {

    // Initialize running parameters of hillvallea
    //-------------------------------------------------
    local_optimizers.clear();

    // initially, we create a single cluster that we initialize by uniform sampling
    //-------------------------------------------------------------------------------------------------------------
    std::vector<solution_pt> backup_sols = pop->sols;

    
    if(fitness_function->redefine_random_initialization)
    {
      fitness_function->init_solutions_randomly(*pop, population_size, lower_init_ranges, upper_init_ranges, 0, rng);
    }
    else
    {
      // double sample_ratio = 2.0;
      // pop->fill_with_rejection(population_size, number_of_parameters, sample_ratio, backup_sols, lower_init_ranges, upper_init_ranges, rng);
      pop->fill_uniform(population_size, number_of_parameters, lower_init_ranges, upper_init_ranges, rng);
      // pop->fill_greedy_uniform(population_size, number_of_parameters, sample_ratio, lower_init_ranges, upper_init_ranges, rng);
    }
    
    {
      int fevals;
      if(evalute_with_gradients) {
        fevals = pop->evaluate_with_gradients(this->fitness_function, 0); // no elite yet.
      } else {
        fevals = pop->evaluate(this->fitness_function, 0); // no elite yet.
      }
      number_of_evaluations += fevals;
      number_of_evaluations_init += fevals;
    }

    pop->sort_on_fitness();

    // create a dummy local_optimizer for the initial population so that we can perform selection and we can write it down.
    double init_univariate_bandwidth = scaled_search_volume * pow(pop->size(), -1.0/number_of_parameters);
    optimizer_pt local_optimizer = init_optimizer(local_optimizer_index, number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng);
    local_optimizer->initialize_from_population(pop, target_clustersize);

    population_pt selection = std::make_shared<population_t>();
    local_optimizer->pop->truncation_percentage(*selection, local_optimizer->selection_fraction * selection_fraction_multiplier);

    // Hill-Valley Clustering
    // note: i init the univariate bandwidth base don the average edge length.. so this is correct here :)
    //------------------------------------------------
    std::vector<population_pt> clusters;
    if(!enable_clustering)
    {
      clusters.push_back(selection);
    }
    else
    {
      if(cluster_alg == 1)
      {
        hgml_t hgml;
        hgml.hierarchical_clustering(*selection, clusters);
      }
      else
      if(cluster_alg == 2)
      {
        hgml_t hgml;
        hgml.nearest_better_clustering(*selection, clusters);
      }
      else
      {
        // add elites to the selection, and mark them as elite
        if (elitist_archive.size() > 0)
        {
          for (size_t i = 0; i < elitist_archive.size(); ++i)
          {
            // std::cout << "added elite;" << std::endl;
            elitist_archive[i]->elite = true;
            selection->sols.push_back(std::make_shared<solution_t>(*elitist_archive[i]));
          }
          selection->sort_on_fitness();
        }
        
        hillvalley_clustering(*selection, clusters);
      }
    }
    
    // Init local optimizers
    //---------------------------------------------------------------------------
    init_univariate_bandwidth = scaled_search_volume * pow(selection->size(), -1.0/number_of_parameters);
    for (auto cluster = clusters.begin(); cluster != clusters.end(); ++cluster)
    {
      if ((*cluster)->sols.size() > 0)
      {
        optimizer_pt opt = init_optimizer(local_optimizer_index, number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng);
        opt->initialize_from_population(*cluster, target_clustersize);
        opt->average_fitness_history.push_back(opt->pop->average_fitness());
        local_optimizers.push_back(opt);
        
        if(enable_clustering) {
          opt->param_std_tolerance = 1e-8;
          opt->fitness_std_tolerance = 1e-8;
          opt->penalty_std_tolerance = 1e-8;
        }
      }
    }

    //if (write_generational_statistics) {
    //  write_statistics_line_population(*pop, local_optimizers, elitist_archive);
    //}

    if (write_generational_solutions)
    {
      write_population_file(pop, local_optimizers);
      write_selection_file(selection, local_optimizers);
    }

  }

  void hillvallea_t::run()
  {
    
    // ugly hack to prevent CPU burning in CEC problems
    //---------------------------------------------------
    vec_t number_of_optima(20,0.0);
    number_of_optima[0] = 2;
    number_of_optima[1] = 5;
    number_of_optima[2] = 1;
    number_of_optima[3] = 4;
    number_of_optima[4] = 2;
    number_of_optima[5] = 18;
    number_of_optima[6] = 36;
    number_of_optima[7] = 81;
    number_of_optima[8] = 216;
    number_of_optima[9] = 12;
    number_of_optima[10] = 6;
    number_of_optima[11] = 8;
    number_of_optima[12] = 6;
    number_of_optima[13] = 6;
    number_of_optima[14] = 8;
    number_of_optima[15] = 6;
    number_of_optima[16] = 8;
    number_of_optima[17] = 6;
    number_of_optima[18] = 8;
    number_of_optima[19] = 8;
    
    

    //---------------------------------------------
    // reset all runlogs (in case hillvallea is run multiple time)
    starting_time = clock();
    success = false;
    terminated = false;
    number_of_evaluations = 0;
    number_of_evaluations_init = 0;
    number_of_evaluations_clustering = 0;
    number_of_generations = 0;
    bool restart = true;
    int number_of_generations_without_new_clusters = 0;
    elitist_archive.clear();


    // allocate population
    //---------------------------------------------
    pop = std::make_shared<population_t>();

    // Init population sizes
    //---------------------------------------------
    double current_population_size = pow(2.0, population_size_initializer);
    double current_cluster_size;
    double current_selection_fraction_multiplier = 1.0;
    
    {
      hillvallea::optimizer_pt dummy_optimizer = init_optimizer(local_optimizer_index, number_of_parameters, lower_param_bounds, upper_param_bounds, 1.0, fitness_function, rng);
      current_cluster_size = cluster_size_initializer *dummy_optimizer->recommended_popsize(number_of_parameters);
    }

    if(write_generational_statistics) {
      new_statistics_file();
    }
    
    // The big restart scheme
    //--------------------------------------------
    while (restart)
    {

      // each restart, collect the elite_candidates and add them to the archive at the end of the run
      // directly adding them to the archive is more expensive in terms of fevals, as you might find a bunch of local optima first
      // and adding solutions to the elitist archive costs fevals because we check them using the HillVallyTest
      std::vector<solution_pt> elite_candidates;
      std::vector<optimizer_pt> local_optimizers;

      // stop if the init popsize is too large. 
      // there is a bit to gain here by decreasing the popsize so that we can run some more local opts, but I don't think its worth it. 
      if (maximum_number_of_evaluations > 0 && current_population_size > (maximum_number_of_evaluations - number_of_evaluations)) {
        restart = false;
        break;
      }
      
      for (size_t problem = 1; problem <= 20; problem++)
      {
        std::stringstream ss;
        ss << "CEC2013_p" << problem;
        if(fitness_function->name().compare(ss.str()) == 0)
        {
          if(elitist_archive.size() == number_of_optima[problem-1]) {
            restart = false;
            break;
          }
        }
        
        if(!restart) {
          break;
        }
      }
      
      // compute initial population
      initialize(pop, (size_t) current_population_size, current_selection_fraction_multiplier, local_optimizers, elitist_archive, true, current_cluster_size);
      
      // we only create local optimizers from the global opts
      // so the local optimizer still inits new global opts
      // therefore, this is basically never hit.
      if (local_optimizers.size() == 0) 
      {
        current_population_size *= population_size_incrementer * population_size_incrementer;
        current_selection_fraction_multiplier *= selection_fraction_multiplier;
        number_of_generations_without_new_clusters++;
        
        if (number_of_generations_without_new_clusters == 3) {
          restart = false;
          std::cout << "unfinished. ";
          break;
        }
        else {
          continue;
        }
      }
      else {
        number_of_generations_without_new_clusters = 0;
      }

      // Run each of the local optimizers until convergence
      for (size_t i = 0; i < local_optimizers.size(); ++i)
      {

        // current local optimizer
        while (true)
        {

          // stop if the feval budget is reached
          size_t fevals_needed_to_check_elites = 1 + (size_t)(elite_candidates.size() * add_elites_max_trials * (elitist_archive.size() + elite_candidates.size() * 0.5));

          if (maximum_number_of_evaluations > 0 && number_of_evaluations + fevals_needed_to_check_elites + current_cluster_size >= maximum_number_of_evaluations) {
            restart = false;
            if (local_optimizers[i]->pop->size() > 0) {
              elite_candidates.push_back(local_optimizers[i]->pop->sols[0]);
            }
            break;
          }

          // stop if we run out of time.
          if (terminate_on_runtime()) {
            restart = false;
            if (local_optimizers[i]->pop->size() > 0) {
              elite_candidates.push_back(local_optimizers[i]->pop->sols[0]);
            }
            break;
          }

          // stop if the vtr is hit
          if (use_vtr && local_optimizers[i]->pop->size() != 0)
          {
            bool vtr_hit = false;
            
            if(!fitness_function->redefine_vtr) {
              vtr_hit = (local_optimizers[i]->pop->sols[0]->constraint == 0) && (local_optimizers[i]->pop->sols[0]->f <= vtr);
            } else {
              vtr_hit = fitness_function->vtr_reached(*local_optimizers[i]->pop->sols[0], vtr);
            }
            
            if(vtr_hit)
            {
              restart = false;
              elite_candidates.push_back(local_optimizers[i]->pop->sols[0]);
              best = solution_t(*local_optimizers[i]->pop->sols[0]);
              success = true;
              break;
            }
          }


          // stop this local optimizer if it approaches a previously obtained elite (candidate)
          if ((1 + local_optimizers[i]->number_of_generations) % 5 == 0)
          {
            if (terminate_on_approaching_elite(*local_optimizers[i], elite_candidates)) {
              local_optimizers[i]->active = false;
              break;
            }
          }

          // stop this local optimizer if it converges to a local optimum
          if (terminate_on_converging_to_local_optimum(*local_optimizers[i], elite_candidates)) {
            local_optimizers[i]->active = false;
            break;
          }

          // if the cluster is active, and after checking it, it is terminated, 
          // we add the best solution to the elitist archive
          if (local_optimizers[i]->active && local_optimizers[i]->checkTerminationCondition()) 
          {
            if (local_optimizers[i]->pop->size() > 0)
            {
              if (elitist_archive.size() == 0 || local_optimizers[i]->pop->sols[0]->f < elitist_archive[0]->f + TargetTolFun) {
                elite_candidates.push_back(local_optimizers[i]->pop->sols[0]);
                elite_candidates.back()->generation_obtained = local_optimizers[i]->number_of_generations;
              }
              
              break;
            }
          }

          // if it is still active, run a generation of the local optimizer
          if (local_optimizers[i]->active)
          {

            local_optimizers[i]->estimate_sample_parameters();

            int local_number_of_evaluations = (int)local_optimizers[i]->sample_new_population((size_t) current_cluster_size);
            number_of_evaluations += local_number_of_evaluations;

            if (write_generational_solutions) {
              write_cluster_population(number_of_generations, i, local_optimizers[i]->number_of_generations, local_optimizers[i]->pop);
            }

            local_optimizers[i]->pop->truncation_percentage(*local_optimizers[i]->pop, local_optimizers[i]->selection_fraction);
            local_optimizers[i]->average_fitness_history.push_back(local_optimizers[i]->pop->average_fitness());

            if (write_generational_statistics) {
              write_statistics_line_cluster(*local_optimizers[i]->pop, (int) i, local_optimizers[i]->number_of_generations, elitist_archive);
            }
          }
        }
      }

      // check if the elites are novel and add the to the archive. 
      int number_of_new_global_opts_found = -1;
      int number_of_global_opts_found = -1;
      add_elites_to_archive(elitist_archive, elite_candidates, number_of_global_opts_found, number_of_new_global_opts_found);

      
      // write elitist_archive of this generation.
      if (write_generational_solutions) {
        write_elitist_archive_file(elitist_archive, false);
      }
      
      if(write_elitist_archive) {
        write_CEC2013_niching_file(false);
      }
      
      // if we found no new global opt, this is either due to the fact that there are no new basins found, 
      // or cuz the cluser size is too small.  increase both
      if (number_of_new_global_opts_found == 0) {
        current_cluster_size *= cluster_size_incrementer;
        current_population_size *= population_size_incrementer;
        current_selection_fraction_multiplier *= current_selection_fraction_multiplier;
      }
      number_of_generations++;

    }

    // sort the archive s.t. the best is first
    if(elitist_archive.size() > 0) {
      best = solution_t(*elitist_archive[0]);
    }

    // write the final solution(s)
    // only if we care to output anything. Else, write nothing 
    if (write_generational_statistics || write_generational_solutions) {
      write_elitist_archive_file(elitist_archive, true);
    }
    
    if (write_elitist_archive) {
      write_CEC2013_niching_file(true);
    }
    
    if (write_generational_statistics) {
      close_statistics_file();
    }


  }

  
  
  void hillvallea_t::runSerial()
  {
    
    //---------------------------------------------
    // reset all runlogs (in case hillvallea is run multiple time)
    starting_time = clock();
    success = false;
    terminated = false;
    number_of_evaluations = 0;
    number_of_evaluations_init = 0;
    number_of_evaluations_clustering = 0;
    number_of_generations = 0;
    
    elitist_archive.clear(); // not really used here, contains only one solution.

    
    // Init population sizes
    //---------------------------------------------
    hillvallea::optimizer_pt optimizer = init_optimizer(local_optimizer_index, number_of_parameters, lower_param_bounds, upper_param_bounds, 1.0, fitness_function, rng);
    size_t population_size = optimizer->recommended_popsize(number_of_parameters); // Popsize

    if(write_generational_statistics) {
      new_statistics_file_serial();
    }
    
    // i was working on implementing the intialization
    // i also need to update the writing<blabla>_serial(
    
    // Initialize Population
    if(fitness_function->redefine_random_initialization) {
      fitness_function->init_solutions_randomly(*optimizer->pop, population_size, lower_init_ranges, upper_init_ranges, 0, rng);
    }
    else {
      optimizer->pop->fill_uniform(population_size, number_of_parameters, lower_init_ranges, upper_init_ranges, rng);
    }
    
    {
      int fevals = optimizer->pop->evaluate(this->fitness_function, 0); // no elite yet.
      number_of_evaluations += fevals;
      number_of_evaluations_init += fevals;
    }
    
    optimizer->pop->sort_on_fitness();
    
    population_pt selection = std::make_shared<population_t>();
    optimizer->pop->truncation_percentage(*selection, optimizer->selection_fraction * selection_fraction_multiplier);
    optimizer->initialize_from_population(selection, population_size);
    
    /*
    // create a dummy local_optimizer for the initial population so that we can perform selection and we can write it down.
    double init_univariate_bandwidth = scaled_search_volume * pow(pop->size(), -1.0/number_of_parameters);
    
    

    // Hill-Valley Clustering
    // note: i init the univariate bandwidth base don the average edge length.. so this is correct here :)
    //------------------------------------------------
    std::vector<population_pt> clusters;
    
    if(cluster_alg == 1)
    {
      hgml_t hgml;
      hgml.hierarchical_clustering(*selection, clusters);
    }
    else
    {
      if(cluster_alg == 2)
      {
        hgml_t hgml;
        hgml.nearest_better_clustering(*selection, clusters);
      }
      else
      {
        hillvalley_clustering(*selection, clusters);
      }
    }
    
    if (write_generational_statistics) {
      write_statistics_line_serial(*pop, local_optimizers, elitist_archive);
    }
    
    if (write_generational_solutions)
    {
      write_population_file(pop, local_optimizers);
      write_selection_file(selection, local_optimizers);
    }
    */

    // run optimizer
    while (optimizer->active)
    {
      // termination criteria
      if (number_of_evaluations >= maximum_number_of_evaluations) { break; }
      if (terminate_on_runtime()) { break; }
      
      // stop if the vtr is hit
      if (use_vtr && optimizer->pop->size() != 0)
      {
        bool vtr_hit = false;
        
        if(!fitness_function->redefine_vtr) {
          vtr_hit = (optimizer->pop->sols[0]->constraint == 0) && (optimizer->pop->sols[0]->f <= vtr);
        } else {
          vtr_hit = fitness_function->vtr_reached(*optimizer->pop->sols[0], vtr);
        }
        
        if(vtr_hit)
        {
          success = true;
          break;
        }
      }
      
      
      if (optimizer->active && optimizer->checkTerminationCondition()) { break; }
      
      // if it is still active, run a generation of the local optimizer
      if (optimizer->active)
      {
        
        optimizer->estimate_sample_parameters();
        
        // double forward_dfc = optimizer->pop->compute_DFC();
        //std::cout << std::fixed << std::setprecision(3) << std::setw(10) << forward_dfc << " ";
        number_of_evaluations += optimizer->sample_new_population(population_size);
        // double backward_dfc = optimizer->pop->compute_DFC();
        //std::cout << std::fixed << std::setprecision(3) << std::setw(10) << backward_dfc << " ";
        
        // write before selection is made.
        if (write_generational_solutions) {
          write_cluster_population(number_of_generations, 0, optimizer->number_of_generations, optimizer->pop);
        }
        
        optimizer->pop->truncation_percentage(*optimizer->pop, optimizer->selection_fraction);
        
        // check if the elites are novel and add the to the archive.
        if(optimizer->pop->size() > 0) {
          best = solution_t(*optimizer->pop->sols[0]);
        }
        
        if (write_generational_statistics) {
          write_statistics_line_serial(*optimizer->pop, optimizer->number_of_generations, best);
        }
      }

      // write elitist_archive
      if (write_generational_solutions) {
        write_elitist_archive_file(elitist_archive, false);
      }
      
      number_of_generations++;
      
    }
    
    // sort the archive s.t. the best is first
    if(elitist_archive.size() > 0) {
      best = solution_t(*elitist_archive[0]);
    }
    
    // write the final solution(s)
    // only if we care to output anything. Else, write nothing
    if (write_generational_statistics || write_generational_solutions) {
      write_elitist_archive_file(elitist_archive, true);
    }
    
    if (write_elitist_archive) {
      write_CEC2013_niching_file(true);
    }
    
    if (write_generational_statistics) {
      close_statistics_file();
    }
    
    
  }

  
  // local_optima_tolerance >= 0 fitness_tolerance for accecpting local optima
  // if its <0, we default to 1e-5
  void hillvallea_t::runSerial2(size_t popsize, bool enable_clustering, double local_optima_tolerance)
  {
    //---------------------------------------------
    // reset all runlogs (in case hillvallea is run multiple time)
    starting_time = clock();
    success = false;
    terminated = false;
    number_of_evaluations = 0;
    number_of_evaluations_init = 0;
    number_of_evaluations_clustering = 0;
    number_of_generations = 0;
    bool restart = true;
    int number_of_generations_without_new_clusters = 0;
    int number_of_generations_no_new_global_opts_found = 0;
    elitist_archive.clear();
    int best_written_prev_fevals = 0;
    
    if(local_optima_tolerance >= 0) {
      TargetTolFun = local_optima_tolerance;
    }
    
    // allocate population
    //---------------------------------------------
    pop = std::make_shared<population_t>();
    
    // Init population sizes
    //---------------------------------------------
    double current_population_size = pow(2.0, population_size_initializer);
    double current_cluster_size = popsize;
    double current_selection_fraction_multiplier = 1.0;
    
    if(popsize == 0)
    {
      hillvallea::optimizer_pt dummy_optimizer = init_optimizer(local_optimizer_index, number_of_parameters, lower_param_bounds, upper_param_bounds, 1.0, fitness_function, rng);
      current_cluster_size = 0.8 * dummy_optimizer->recommended_popsize(number_of_parameters);
    }
    
    if(!enable_clustering) {
      current_population_size = current_cluster_size;
    }
    
    if(write_generational_statistics) {
      // new_statistics_file();
      new_statistics_file_serial(enable_clustering);
    }
    
    // The big restart scheme
    //--------------------------------------------
    while (restart)
    {
      if(!enable_clustering) {
        restart = false;
      }
      
      // each restart, collect the elite_candidates and add them to the archive at the end of the run
      // directly adding them to the archive is more expensive in terms of fevals, as you might find a bunch of local optima first
      // and adding solutions to the elitist archive costs fevals because we check them using the HillVallyTest
      std::vector<solution_pt> elite_candidates;
      std::vector<optimizer_pt> local_optimizers;
      
      // stop if the init popsize is too large.
      // there is a bit to gain here by decreasing the popsize so that we can run some more local opts, but I don't think its worth it.
      if (maximum_number_of_evaluations > 0 && current_population_size > (maximum_number_of_evaluations - number_of_evaluations)) {
        restart = false;
        std::cout << "Terminated because of function evaluations" << std::endl;
        break;
      }

      // compute initial population
      initialize(pop, (size_t) current_population_size, current_selection_fraction_multiplier, local_optimizers, elitist_archive, enable_clustering, current_cluster_size);
      
      if (write_generational_statistics) {
        write_statistics_line_serial(*pop, number_of_generations, *pop->sols[0], enable_clustering);
      }
      
      std::cout << "Restart with a popsize of " << current_population_size << ", located " << local_optimizers.size() << " niche(s)." << std::endl;
      
      // we only create local optimizers from the global opts
      // so the local optimizer still inits new global opts
      // therefore, this is basically never hit.
      if (local_optimizers.size() == 0)
      {
        std::cout << "  No new lopts found " << std::endl;
        current_population_size *= population_size_incrementer * population_size_incrementer;
        current_selection_fraction_multiplier *= selection_fraction_multiplier;
        number_of_generations_without_new_clusters++;
        
        if (number_of_generations_without_new_clusters == 3) {
          std::cout << "  Terminated because Number_of_generations_without_new_clusters reached" << std::endl;
          restart = false;
          break;
        }
        else {
          continue;
        }
      }
      else {
        number_of_generations_without_new_clusters = 0;
      }
      
      // Run each of the local optimizers until convergence
      for (size_t i = 0; i < local_optimizers.size(); ++i)
      {
        
        // current local optimizer
        while (true)
        {
          if (maximum_number_of_evaluations > 0 && number_of_evaluations >= maximum_number_of_evaluations) {
            std::cout << "  Terminated core search algorithm because function evaluations limit reached" << std::endl;
            
            restart = false;
            if (local_optimizers[i]->pop->size() > 0) {
              elite_candidates.push_back(local_optimizers[i]->pop->sols[0]);
            }
            break;
          }
          
          // stop if we run out of time.
          if (terminate_on_runtime()) {
            std::cout << "  Terminated core search algorithm because time limit reached" << std::endl;
            restart = false;
            if (local_optimizers[i]->pop->size() > 0) {
              elite_candidates.push_back(local_optimizers[i]->pop->sols[0]);
            }
            break;
          }
          
          // stop if the vtr is hit
          if (use_vtr && local_optimizers[i]->pop->size() > 0)
          {
            
            bool vtr_hit = false;
            
            if(!fitness_function->redefine_vtr) {
              vtr_hit = (local_optimizers[i]->pop->sols[0]->constraint == 0) && (local_optimizers[i]->pop->sols[0]->f <= vtr);
            } else {
              vtr_hit = fitness_function->vtr_reached(*local_optimizers[i]->pop->sols[0], vtr);
            }
            
            if(vtr_hit)
            {
              bool true_hit = true;
              size_t hit_idx = 0;
              if(fitness_function->partial_evaluations_available && fitness_function->has_round_off_errors_in_partial_evaluations)
              {
                true_hit = false;
                for(hit_idx = 0; hit_idx < local_optimizers[i]->pop->size(); ++hit_idx)
                {
                  
                  // recheck
                  if(!fitness_function->redefine_vtr)
                  {
                    if( (local_optimizers[i]->pop->sols[hit_idx]->constraint != 0) || (local_optimizers[i]->pop->sols[hit_idx]->f > vtr) ) {
                      break;
                    }
                  } else {
                    if( !fitness_function->vtr_reached(*local_optimizers[i]->pop->sols[0], vtr) ) {
                      break;
                    }
                  }
                  
                  // re-evaluate solution when vtr is assumed to be hit.
                  fitness_function->evaluate(local_optimizers[i]->pop->sols[hit_idx]);
                  number_of_evaluations++;
                  
                  if(!fitness_function->redefine_vtr) {
                    true_hit = (local_optimizers[i]->pop->sols[0]->constraint == 0) && (local_optimizers[i]->pop->sols[0]->f <= vtr);
                  } else {
                    true_hit = fitness_function->vtr_reached(*local_optimizers[i]->pop->sols[0], vtr);
                  }
                  
                  if(true_hit) {
                    break;
                  }
                }
              }
              
              if(true_hit)
              {
                restart = false;
                std::cout << "  Terminated core search algorithm because VTR reached! Yay" << std::endl;
                elite_candidates.push_back(local_optimizers[i]->pop->sols[hit_idx]);
                best = solution_t(*local_optimizers[i]->pop->sols[0]);
                success = true;
                break;
              }
            }
          }
          
          // stop this local optimizer if it approaches a previously obtained elite (candidate)
          if ((1 + local_optimizers[i]->number_of_generations) % 5 == 0)
          {
            if (terminate_on_approaching_elite(*local_optimizers[i], elite_candidates)) {
              std::cout << "  Terminated core search algorithm because approaching elite (f = " << local_optimizers[i]->pop->sols[0]->f << ", x[0] = " << local_optimizers[i]->pop->sols[0]->param[0] << ")."  << std::endl;
              local_optimizers[i]->active = false;
              break;
            }
          }
          
          // stop this local optimizer if it converges to a local optimum
          if (terminate_on_converging_to_local_optimum(*local_optimizers[i], elite_candidates)) {
            std::cout << "  Terminated core search algorithm because approaching local optimum" << std::endl;
            local_optimizers[i]->active = false;
            break;
          }
          
          // if the cluster is active, and after checking it, it is terminated,
          // we add the best solution to the elitist archive
          if (local_optimizers[i]->active && local_optimizers[i]->checkTerminationCondition())
          {
            std::cout << "  Terminated core search algorithm because of internal reasons (f = " << local_optimizers[i]->pop->sols[0]->f << ", x[0] = " << local_optimizers[i]->pop->sols[0]->param[0] << ")."  << std::endl;
            if (local_optimizers[i]->pop->size() > 0)
            {
              if (elitist_archive.size() == 0 || local_optimizers[i]->pop->sols[0]->f < elitist_archive[0]->f + TargetTolFun) {
                elite_candidates.push_back(local_optimizers[i]->pop->sols[0]);
                elite_candidates.back()->generation_obtained = local_optimizers[i]->number_of_generations;
              }// else {
               // std::cout << "No new candidate: " << local_optimizers[i]->pop->sols[0]->f << "\n";
              //}
              
              break;
            }
          }
          
          // if it is still active, run a generation of the local optimizer
          if (local_optimizers[i]->active)
          {
            local_optimizers[i]->generation(current_cluster_size, number_of_evaluations);
            
            if (write_generational_solutions) {
              write_cluster_population(number_of_generations, i, local_optimizers[i]->number_of_generations, local_optimizers[i]->pop);
            }
            
            
            if (write_generational_statistics && (local_optimizers[i]->number_of_generations < 50 || local_optimizers[i]->number_of_generations % 50 == 0))
            {
              if(local_optimizers[i]->pop->size() == current_cluster_size)
              { // GOMEA keeps the population in pop, while AMaLGaM only maintains the selection. This is therefore a bit of a lousy hack.
                population_t selection;
                local_optimizers[i]->pop->truncation_percentage(selection, local_optimizers[i]->selection_fraction);
                // write_statistics_line_cluster(selection, (int) i, local_optimizers[i]->number_of_generations, elitist_archive);
                write_statistics_line_serial(selection, local_optimizers[i]->number_of_generations, *local_optimizers[i]->pop->sols[0], enable_clustering);
              }
              else
              {
                // write_statistics_line_cluster(*local_optimizers[i]->pop, (int) i, local_optimizers[i]->number_of_generations, elitist_archive);
                write_statistics_line_serial(*local_optimizers[i]->pop, local_optimizers[i]->number_of_generations, *local_optimizers[i]->pop->sols[0], enable_clustering);
              }
            }
            
            if( ((int) (number_of_evaluations / 10000)) !=  best_written_prev_fevals)
            {
              best_written_prev_fevals = ((int) (number_of_evaluations / 50000));
              std::stringstream ss;
              ss << write_directory << "best" << std::setw(5) << std::setfill('0') << best_written_prev_fevals << file_appendix << ".dat";
              fitness_function->write_solution(local_optimizers[i]->best, ss.str() );
            }
          }
        }
        
        if(!enable_clustering) {
          break;
        }
        
        if(!restart || success) {
          break;
        }
      }
      
      
      if (write_generational_statistics && !enable_clustering)
      {
        // if(local_optimizers[0]->number_of_generations > 50 && local_optimizers[0]->number_of_generations % 50 != 0)
        {
          if(local_optimizers[0]->pop->size() == current_cluster_size)
          { // GOMEA keeps the population in pop, while AMaLGaM only maintains the selection. This is therefore a bit of a lousy hack.
            population_t selection;
            local_optimizers[0]->pop->truncation_percentage(selection, local_optimizers[0]->selection_fraction);
            // write_statistics_line_cluster(selection, (int) i, local_optimizers[i]->number_of_generations, elitist_archive);
            write_statistics_line_serial(selection, local_optimizers[0]->number_of_generations, *local_optimizers[0]->pop->sols[0], enable_clustering);
          }
          else
          {
            // write_statistics_line_cluster(*local_optimizers[i]->pop, (int) i, local_optimizers[i]->number_of_generations, elitist_archive);
            write_statistics_line_serial(*local_optimizers[0]->pop, local_optimizers[0]->number_of_generations, *local_optimizers[0]->pop->sols[0], enable_clustering);
          }
        }
      }
      
      
      // check if the elites are novel and add the to the archive.
      int number_of_new_global_opts_found = -1;
      int number_of_global_opts_found = -1;
      add_elites_to_archive(elitist_archive, elite_candidates, number_of_global_opts_found, number_of_new_global_opts_found);
      std::cout << "  Of " << elite_candidates.size() << " elite candidates, found " << number_of_new_global_opts_found << ((number_of_new_global_opts_found != 1) ? " new optima." : " new optimum.") << " The elitist archive now contains " << elitist_archive.size() << (elitist_archive.size() == 1 ? " solution.":" solutions.") << std::endl;
      
      
      // write elitist_archive of this generation.
      if (write_generational_solutions) {
        write_elitist_archive_file(elitist_archive, false);
      }
      
      if(write_elitist_archive) {
        write_CEC2013_niching_file(false);
      }
      
      // if we found no new global opt, this is either due to the fact that there are no new basins found,
      // or cuz the cluser size is too small.  increase both
      if (number_of_new_global_opts_found == 0) {
        number_of_generations_no_new_global_opts_found++;
        
        if(number_of_generations_no_new_global_opts_found >= 1 + sqrt(number_of_parameters)) { //banaan
          number_of_generations_no_new_global_opts_found = 0;
          current_cluster_size *= cluster_size_incrementer;
          current_population_size *= population_size_incrementer;
          current_selection_fraction_multiplier *= current_selection_fraction_multiplier;
        }
      } else {
        number_of_generations_no_new_global_opts_found = 0;
      }
      number_of_generations++;
      
    }
    
    // sort the archive s.t. the best is first
    if(elitist_archive.size() > 0) {
      best = solution_t(*elitist_archive[0]);
    }
    
    // write the final solution(s)
    // only if we care to output anything. Else, write nothing
    if (write_generational_statistics || write_generational_solutions) {
      write_elitist_archive_file(elitist_archive, true);
    }
    
    if (write_elitist_archive) {
      write_CEC2013_niching_file(true);
    }
    
    if (write_generational_statistics) {
      close_statistics_file();
    }
    
    std::stringstream ss;
    ss << write_directory << "best_final" << file_appendix << ".dat";
    fitness_function->write_solution(best, ss.str() );
    
  }

  
  
  void hillvallea_t::runSerial3(size_t popsize, bool enable_clustering, int initial_number_of_evaluations)
  {
    //---------------------------------------------
    // reset all runlogs (in case hillvallea is run multiple time)
    starting_time = clock();
    success = false;
    terminated = false;
    number_of_evaluations = initial_number_of_evaluations;
    number_of_evaluations_init = initial_number_of_evaluations;
    number_of_evaluations_clustering = 0;
    number_of_generations = 0;
    bool restart = true;
    int number_of_generations_without_new_clusters = 0;
    elitist_archive.clear();
    int best_written_prev_fevals = 0;
    
    if (local_optimizer_index == 84) {
      evalute_with_gradients = true;
    }
    
    // allocate population
    //---------------------------------------------
    pop = std::make_shared<population_t>();
    
    // Init population sizes
    //---------------------------------------------
    double current_population_size = pow(2.0, population_size_initializer);
    double current_cluster_size = popsize;
    double current_selection_fraction_multiplier = 1.0;
    
    if(popsize == 0)
    {
      hillvallea::optimizer_pt dummy_optimizer = init_optimizer(local_optimizer_index, number_of_parameters, lower_param_bounds, upper_param_bounds, 1.0, fitness_function, rng);
      current_cluster_size = 0.8 * dummy_optimizer->recommended_popsize(number_of_parameters);
    }
    
    if(!enable_clustering) {
      current_population_size = current_cluster_size;
    }
    
    if(write_generational_statistics) {
      // new_statistics_file();
      new_statistics_file_serial();
    }
    
    // The big restart scheme
    //--------------------------------------------
    while (restart)
    {
      if(!enable_clustering) {
        restart = false;
      }
      
      // each restart, collect the elite_candidates and add them to the archive at the end of the run
      // directly adding them to the archive is more expensive in terms of fevals, as you might find a bunch of local optima first
      // and adding solutions to the elitist archive costs fevals because we check them using the HillVallyTest
      std::vector<solution_pt> elite_candidates;
      std::vector<optimizer_pt> local_optimizers;
      
      // stop if the init popsize is too large.
      // there is a bit to gain here by decreasing the popsize so that we can run some more local opts, but I don't think its worth it.
      if (maximum_number_of_evaluations > 0 && current_population_size > (maximum_number_of_evaluations - number_of_evaluations)) {
        restart = false;
        std::cout << "Terminated because of function evaluations" << std::endl;
        break;
      }
      
      // compute initial population
      // initialize(pop, (size_t) current_population_size, current_selection_fraction_multiplier, local_optimizers, elitist_archive, enable_clustering, current_cluster_size);
      /*
      if(fitness_function->redefine_random_initialization) {
        fitness_function->init_solutions_randomly(*pop, (size_t) current_population_size, lower_init_ranges, upper_init_ranges, 0, rng);
      } else {
        pop->fill_uniform((size_t) current_population_size, number_of_parameters, lower_init_ranges, upper_init_ranges, rng);
      }
      */
      
      
      if(fitness_function->redefine_random_initialization) {
        fitness_function->init_solutions_randomly(*pop, (size_t) current_population_size, lower_init_ranges, upper_init_ranges, 0, rng);
      } else {
        pop->fill_uniform((size_t) current_population_size, number_of_parameters, lower_init_ranges, upper_init_ranges, rng);
      }
      // TODO: try to init more locally!
      
      {
        int fevals;
        if(evalute_with_gradients) {
          fevals = pop->evaluate_with_gradients(this->fitness_function, 0); // no elite yet.
        } else {
          fevals = pop->evaluate(this->fitness_function, 0); // no elite yet.
        }
        number_of_evaluations += fevals;
        number_of_evaluations_init += fevals;
      }
      
      pop->sort_on_fitness();
      
      // Init local optimizers
      //---------------------------------------------------------------------------
      double init_univariate_bandwidth = scaled_search_volume * pow(pop->size(), -1.0/number_of_parameters);

      optimizer_pt opt = init_optimizer(local_optimizer_index, number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng);
      opt->initialize_from_population(pop, pop->size());
      opt->average_fitness_history.push_back(opt->pop->average_fitness());
      local_optimizers.push_back(opt);

      if(enable_clustering) {
        opt->param_std_tolerance = 1e-8;
        opt->fitness_std_tolerance = 1e-8;
        opt->penalty_std_tolerance = 1e-8;
      }
      
      if (write_generational_statistics) {
        write_statistics_line_serial(*pop, number_of_generations, *pop->sols[0]);
      }
      
      std::cout << "Restart with a popsize of " << current_population_size << ", located " << local_optimizers.size() << " niche(s)." << std::endl;
      
      // we only create local optimizers from the global opts
      // so the local optimizer still inits new global opts
      // therefore, this is basically never hit.
      if (local_optimizers.size() == 0)
      {
        current_population_size *= population_size_incrementer * population_size_incrementer;
        current_selection_fraction_multiplier *= selection_fraction_multiplier;
        number_of_generations_without_new_clusters++;
        
        if (number_of_generations_without_new_clusters == 3) {
          std::cout << "  Terminated because Number_of_generations_without_new_clusters reached" << std::endl;
          restart = false;
          break;
        }
        else {
          continue;
        }
      }
      else {
        number_of_generations_without_new_clusters = 0;
      }
      
      // Run each of the local optimizers until convergence
      for (size_t i = 0; i < local_optimizers.size(); ++i)
      {
        
        // current local optimizer
        while (true)
        {
          if (maximum_number_of_evaluations > 0 && number_of_evaluations >= maximum_number_of_evaluations) {
            std::cout << "  Terminated core search algorithm because function evaluations limit reached" << std::endl;
            
            restart = false;
            if (local_optimizers[i]->pop->size() > 0) {
              elite_candidates.push_back(local_optimizers[i]->pop->sols[0]);
            }
            break;
          }
          
          // stop if we run out of time.
          if (terminate_on_runtime()) {
            std::cout << "  Terminated core search algorithm because time limit reached" << std::endl;
            restart = false;
            if (local_optimizers[i]->pop->size() > 0) {
              elite_candidates.push_back(local_optimizers[i]->pop->sols[0]);
            }
            break;
          }
          
          // stop if the vtr is hit
          if (use_vtr && local_optimizers[i]->pop->size() > 0)
          {
            
            bool vtr_hit = false;
            
            if(!fitness_function->redefine_vtr) {
              vtr_hit = (local_optimizers[i]->pop->sols[0]->constraint == 0) && (local_optimizers[i]->pop->sols[0]->f <= vtr);
            } else {
              vtr_hit = fitness_function->vtr_reached(*local_optimizers[i]->pop->sols[0], vtr);
            }
            
            if(vtr_hit)
            {
              bool true_hit = true;
              size_t hit_idx = 0;
              if(fitness_function->partial_evaluations_available && fitness_function->has_round_off_errors_in_partial_evaluations)
              {
                true_hit = false;
                for(hit_idx = 0; hit_idx < local_optimizers[i]->pop->size(); ++hit_idx)
                {
                  
                  // recheck
                  if(!fitness_function->redefine_vtr)
                  {
                    if( (local_optimizers[i]->pop->sols[hit_idx]->constraint != 0) || (local_optimizers[i]->pop->sols[hit_idx]->f > vtr) ) {
                      break;
                    }
                  } else {
                    if( !fitness_function->vtr_reached(*local_optimizers[i]->pop->sols[0], vtr) ) {
                      break;
                    }
                  }
                  
                  // re-evaluate solution when vtr is assumed to be hit.
                  fitness_function->evaluate(local_optimizers[i]->pop->sols[hit_idx]);
                  number_of_evaluations++;
                  
                  if(!fitness_function->redefine_vtr) {
                    true_hit = (local_optimizers[i]->pop->sols[0]->constraint == 0) && (local_optimizers[i]->pop->sols[0]->f <= vtr);
                  } else {
                    true_hit = fitness_function->vtr_reached(*local_optimizers[i]->pop->sols[0], vtr);
                  }
                  
                  if(true_hit) {
                    break;
                  }
                }
              }
              
              if(true_hit)
              {
                restart = false;
                std::cout << "  Terminated core search algorithm because VTR reached! Yay" << std::endl;
                elite_candidates.push_back(local_optimizers[i]->pop->sols[hit_idx]);
                best = solution_t(*local_optimizers[i]->pop->sols[0]);
                success = true;
                break;
              }
            }
          }
          
          // stop this local optimizer if it approaches a previously obtained elite (candidate)
          if ((1 + local_optimizers[i]->number_of_generations) % 5 == 0)
          {
            if (terminate_on_approaching_elite(*local_optimizers[i], elite_candidates)) {
              std::cout << "  Terminated core search algorithm because approaching elite" << std::endl;
              local_optimizers[i]->active = false;
              break;
            }
          }
          
          // stop this local optimizer if it converges to a local optimum
          if (terminate_on_converging_to_local_optimum(*local_optimizers[i], elite_candidates)) {
            std::cout << "  Terminated core search algorithm because approaching local optimum" << std::endl;
            local_optimizers[i]->active = false;
            break;
          }
          
          // if the cluster is active, and after checking it, it is terminated,
          // we add the best solution to the elitist archive
          if (local_optimizers[i]->active && local_optimizers[i]->checkTerminationCondition())
          {
            std::cout << "  Terminated core search algorithm because of internal reasons" << std::endl;
            if (local_optimizers[i]->pop->size() > 0)
            {
              if (elitist_archive.size() == 0 || local_optimizers[i]->pop->sols[0]->f < elitist_archive[0]->f + TargetTolFun) {
                elite_candidates.push_back(local_optimizers[i]->pop->sols[0]);
                elite_candidates.back()->generation_obtained = local_optimizers[i]->number_of_generations;
              }
              
              break;
            }
          }
          
          // if it is still active, run a generation of the local optimizer
          if (local_optimizers[i]->active)
          {
            local_optimizers[i]->generation(current_cluster_size, number_of_evaluations);
            
            if (write_generational_solutions) {
              write_cluster_population(number_of_generations, i, local_optimizers[i]->number_of_generations, local_optimizers[i]->pop);
            }
            
            if (write_generational_statistics && (local_optimizers[i]->number_of_generations < 50 || local_optimizers[i]->number_of_generations % 100 == 0))
            {
              if(local_optimizers[i]->pop->size() == current_cluster_size)
              { // GOMEA keeps the population in pop, while AMaLGaM only maintains the selection. This is therefore a bit of a lousy hack.
                population_t selection;
                local_optimizers[i]->pop->truncation_percentage(selection, local_optimizers[i]->selection_fraction);
                // write_statistics_line_cluster(selection, (int) i, local_optimizers[i]->number_of_generations, elitist_archive);
                write_statistics_line_serial(selection, local_optimizers[i]->number_of_generations, *local_optimizers[i]->pop->sols[0]);
              }
              else
              {
                // write_statistics_line_cluster(*local_optimizers[i]->pop, (int) i, local_optimizers[i]->number_of_generations, elitist_archive);
                write_statistics_line_serial(*local_optimizers[i]->pop, local_optimizers[i]->number_of_generations, *local_optimizers[i]->pop->sols[0]);
              }
            }
            
            if( ((int) (number_of_evaluations / 10000)) !=  best_written_prev_fevals)
            {
              best_written_prev_fevals = ((int) (number_of_evaluations / 50000));
              std::stringstream ss;
              ss << write_directory << "best" << std::setw(5) << std::setfill('0') << best_written_prev_fevals << file_appendix << ".dat";
              fitness_function->write_solution(local_optimizers[i]->best, ss.str() );
            }
          }
        }
        
        if(!enable_clustering) {
          break;
        }
        
        if(!restart || success) {
          break;
        }
      }
      
      
      if (write_generational_statistics && !enable_clustering)
      {
        if(local_optimizers[0]->number_of_generations > 50 && local_optimizers[0]->number_of_generations % 100 != 0)
        {
          if(local_optimizers[0]->pop->size() == current_cluster_size)
          { // GOMEA keeps the population in pop, while AMaLGaM only maintains the selection. This is therefore a bit of a lousy hack.
            population_t selection;
            local_optimizers[0]->pop->truncation_percentage(selection, local_optimizers[0]->selection_fraction);
            // write_statistics_line_cluster(selection, (int) i, local_optimizers[i]->number_of_generations, elitist_archive);
            write_statistics_line_serial(selection, local_optimizers[0]->number_of_generations, *local_optimizers[0]->pop->sols[0]);
          }
          else
          {
            // write_statistics_line_cluster(*local_optimizers[i]->pop, (int) i, local_optimizers[i]->number_of_generations, elitist_archive);
            write_statistics_line_serial(*local_optimizers[0]->pop, local_optimizers[0]->number_of_generations, *local_optimizers[0]->pop->sols[0]);
          }
        }
      }
      
      
      // check if the elites are novel and add the to the archive.
      int number_of_new_global_opts_found = -1;
      int number_of_global_opts_found = -1;
      add_elites_to_archive(elitist_archive, elite_candidates, number_of_global_opts_found, number_of_new_global_opts_found);
      std::cout << "  Of " << elite_candidates.size() << " elite candidates, found " << number_of_new_global_opts_found << ((number_of_new_global_opts_found != 1) ? " new optima." : " new optimum.") << std::endl;
      
      
      // write elitist_archive of this generation.
      if (write_generational_solutions) {
        write_elitist_archive_file(elitist_archive, false);
      }
      
      if(write_elitist_archive) {
        write_CEC2013_niching_file(false);
      }
      
      // if we found no new global opt, this is either due to the fact that there are no new basins found,
      // or cuz the cluser size is too small.  increase both
      if (number_of_new_global_opts_found == 0) {
        current_cluster_size *= cluster_size_incrementer;
        current_population_size *= population_size_incrementer;
        current_selection_fraction_multiplier *= current_selection_fraction_multiplier;
      }
      number_of_generations++;
      
    }
    
    // sort the archive s.t. the best is first
    if(elitist_archive.size() > 0) {
      best = solution_t(*elitist_archive[0]);
    }
    
    // write the final solution(s)
    // only if we care to output anything. Else, write nothing
    if (write_generational_statistics || write_generational_solutions) {
      write_elitist_archive_file(elitist_archive, true);
    }
    
    if (write_elitist_archive) {
      write_CEC2013_niching_file(true);
    }
    
    if (write_generational_statistics) {
      close_statistics_file();
    }
    
    std::stringstream ss;
    ss << write_directory << "best_final" << file_appendix << ".dat";
    fitness_function->write_solution(best, ss.str() );
    
  }
  
  void hillvallea_t::add_elites_to_archive(std::vector<solution_pt> & elitist_archive, const std::vector<solution_pt> & elite_candidates, int & number_of_global_opts_found, int & number_of_new_global_opts_found)
  {

    number_of_global_opts_found = 0;
    number_of_new_global_opts_found = 0;

    // find best solution in the archive
    double best_archive = 1e308;
    for (size_t i = 0; i < elitist_archive.size(); ++i)
    {
      if (elitist_archive[i]->f < best_archive) {
        best_archive = elitist_archive[i]->f;
      }
    }

    // find best candidate
    double best_candidate = 1e308;
    for (size_t i = 0; i < elite_candidates.size(); ++i)
    {
      if (elite_candidates[i]->f < best_candidate) {
        best_candidate = elite_candidates[i]->f;
      }
    }

    // clear the achive if the best candidate is better
    // than the best in the archive
    double best = std::min(best_candidate, best_archive);
    if ((best_candidate + TargetTolFun) < best_archive) {
      elitist_archive.clear();
    }

    // potential candidates are only those that are global optima 
    std::vector<solution_pt> potential_candidates;
    for (size_t i = 0; i < elite_candidates.size(); ++i)
    {
      if (elite_candidates[i]->f < (best + TargetTolFun)) {
        potential_candidates.push_back(elite_candidates[i]);
      }
    }

    // for each candidate, check if it is a novel local optimum
    for (size_t i = 0; i < potential_candidates.size(); ++i)
    {

      // check if the potential global optima is novel
      bool novel = true;
      
      if (elitist_archive.size() > 0)
      {
        size_t nearest_elite = 0;
        double nearest_dist = 1e300;
        double current_dist = 0;
        
        for (size_t j = 0; j < elitist_archive.size(); ++j)
        {

          current_dist = elitist_archive[j]->param_distance(*potential_candidates[i]);
          
          if(current_dist < nearest_dist)
          {
            nearest_dist = current_dist;
            nearest_elite = j;
          }
        }
      
        size_t j = nearest_elite; // lazy : re-using code..
      
        // a valid edge (check_edge returns true) suggest that the two optima are the same.
        if (check_edge(*elitist_archive[j], *potential_candidates[i], add_elites_max_trials))
        {
          novel = false;
          number_of_global_opts_found++;

          // replace the elite with the candidate if it is better
          if (solution_t::better_solution_via_pointers(potential_candidates[i], elitist_archive[j])) {
            elitist_archive[j] = std::make_shared<solution_t>(*potential_candidates[i]);
            elitist_archive[j]->elite = true;
            elitist_archive[j]->time_obtained = ((double) (clock() - starting_time)) / CLOCKS_PER_SEC * 1000.0;
            elitist_archive[j]->feval_obtained = number_of_evaluations;
            elitist_archive[j]->generation_obtained = potential_candidates[i]->generation_obtained;
          }

          // break;
        }
      }

      // it is novel, add it to the archive
      if (novel) {
        elitist_archive.push_back(std::make_shared<solution_t>(*potential_candidates[i]));
        elitist_archive.back()->elite = true;
        elitist_archive.back()->time_obtained = ((double)(clock() - starting_time)) / CLOCKS_PER_SEC * 1000.0;
        elitist_archive.back()->feval_obtained = number_of_evaluations;
        elitist_archive.back()->generation_obtained = potential_candidates[i]->generation_obtained;
        number_of_new_global_opts_found++;
      }

    }
    
    std::sort(elitist_archive.begin(),elitist_archive.end(),solution_t::better_solution_via_pointers);
  }

}



// returns true if it is a valid edge. (if the solutions belong to the same basin)
bool hillvallea::hillvallea_t::check_edge(const hillvallea::solution_t &sol1, const hillvallea::solution_t &sol2, int max_trials)
{
  std::vector<solution_pt> test_points;
  return check_edge(sol1, sol2, max_trials, test_points);
}

bool hillvallea::hillvallea_t::check_edge(const hillvallea::solution_t &sol1, const hillvallea::solution_t &sol2, int max_trials, std::vector<solution_pt> & test_points)
{

  if (sol1.param_distance(sol2) == 0) {
    return true;
  }
  
  // elites are already checked to be in different basins
  if (sol1.elite && sol2.elite) {
    return false;
  }

  // check max_trials with number_of_evaluations remaining
  //if (maximum_number_of_evaluations > 0 && max_trials > maximum_number_of_evaluations - number_of_evaluations) {
  //  max_trials = maximum_number_of_evaluations - number_of_evaluations;
  //}

  // find the worst solution of the two. 
  solution_t worst;
  if (solution_t::better_solution(sol1, sol2)) {
    worst = sol2;
  }
  else {
    worst = sol1;
  }

  for (size_t k = 0; k < max_trials; k++)
  {

    solution_pt x_test = std::make_shared<solution_t>(sol1.param.size());

    x_test->param = sol1.param + ((k + 1.0) / (max_trials + 1.0)) * (sol2.param - sol1.param);

    if(evalute_with_gradients) {
      fitness_function->evaluate_with_gradients(x_test);
    } else {
      fitness_function->evaluate(x_test);
    }
    number_of_evaluations++;
    number_of_evaluations_clustering++;

    test_points.push_back(x_test);

    // if f[i] is better than f_test, we don't like the connection. So we stop.
    if (solution_t::better_solution(worst, *x_test)) {
      return false;
    }
  }

  return true;

}


void hillvallea::hillvallea_t::hillvalley_clustering(population_t & pop, std::vector<population_pt> & clusters)
{
  hillvalley_clustering(pop, clusters, clustering_max_number_of_neighbours, true, false, false);
}

void hillvallea::hillvallea_t::hillvalley_clustering(population_t & pop, std::vector<population_pt> & clusters, size_t max_number_of_neighbours, bool add_test_solutions, bool skip_check_for_worst_solutions, bool check_all_neighbors_from_same_clusters)
{

  // reset all clusters
  clusters.clear();

  // exclude the trivial case
  if (pop.size() == 0) {
    return;
  }


  // Initialize the first cluster as the cluster containing the best (=first) solution
  //-------------------------------------------------------------------------------------
  size_t number_of_clusters = 1;
  std::vector<size_t> cluster_index(pop.size(), -1);
  cluster_index[0] = 0;

  // remember how many solutions in the population created, so that we can allocate them later. 
  std::vector<solution_pt> test_points;
  std::vector<size_t> cluster_index_of_test_points;
  double average_edge_length = scaled_search_volume * pow(pop.size(), -1.0/number_of_parameters);
  double* dist = (double *)Malloc((long)pop.size() * sizeof(double));

  for (size_t i = 1; i < pop.size(); i++)
  {

    // compute the distance to all better solutions. 
    dist[i] = 0.0;
    size_t nearest_better_index = 0, worst_better_index = 0;
    for (size_t j = 0; j < i; j++) {
      dist[j] = pop.sols[i]->param_distance(*pop.sols[j]);

      if (dist[j] < dist[nearest_better_index]) {
        nearest_better_index = j;
      }

      if (dist[j] > dist[worst_better_index]) {
        worst_better_index = j;
      }
    }

    // Check neighbours
    bool edge_added = false;
    std::vector<size_t> does_not_belong_to(max_number_of_neighbours, -1);
    size_t old_nearest_better_index;
    std::vector<solution_pt> new_test_points_for_this_sol;

    for (size_t j = 0; j < std::min(i, max_number_of_neighbours); j++)
    {

      // find the next-to nearest index
      if (j > 0) 
      {
        old_nearest_better_index = nearest_better_index;
        nearest_better_index = worst_better_index;

        for (size_t k = 0; k < i; k++) {

          if (dist[k] > dist[old_nearest_better_index] && dist[k] < dist[nearest_better_index]) {
            nearest_better_index = k;
          }
        }
      }

      if (!check_all_neighbors_from_same_clusters)
      {
        bool skip_neighbour = false;
        for (size_t k = 0; k < does_not_belong_to.size(); ++k)
        {
          if (does_not_belong_to[k] == cluster_index[nearest_better_index])
          {
            skip_neighbour = true;
            break;
          }
        }

        if (skip_neighbour) {
          continue;
        }
      }

      int max_number_of_trial_solutions = 1 + ((int)(dist[nearest_better_index] / average_edge_length));
      max_number_of_trial_solutions = std::min((int) max_number_of_neighbours,max_number_of_trial_solutions);
      
      std::vector<solution_pt> new_test_points;
      bool force_accept = false;
      
      if(!skip_check_for_worst_solutions)
      {
        if(i > 0.5 * pop.size() && max_number_of_trial_solutions == 1) {
          force_accept = true;
        }
      }
      if (force_accept || check_edge(*pop.sols[i], *pop.sols[nearest_better_index], max_number_of_trial_solutions, new_test_points))
      {
        cluster_index[i] = cluster_index[nearest_better_index];
        edge_added = true;

        // if the edge is accepted, add all test_poitns to their cluster
        for (size_t k = 0; k < new_test_points.size(); ++k) {
          test_points.push_back(new_test_points[k]);
          cluster_index_of_test_points.push_back(cluster_index[nearest_better_index]);
        }

        break;
      }
      else
      {
        does_not_belong_to[j] = cluster_index[nearest_better_index];

        // if the edge is not accepted, add all solutions to that cluster
        // all but the last because that one caused the rejection
        if (new_test_points.size() > 0) {
          for (size_t k = 0; k < new_test_points.size() - 1; ++k) {
            new_test_points_for_this_sol.push_back(new_test_points[k]);
          }
        }

      }
    }

    // its a new clusters, label it like that. 
    if (!edge_added)
    {
      cluster_index[i] = number_of_clusters;
      number_of_clusters++;

      // if its a new cluster, add all its testpoints as well. 
      for (size_t k = 0; k < new_test_points_for_this_sol.size(); ++k)
      {
        test_points.push_back(new_test_points_for_this_sol[k]);
        cluster_index_of_test_points.push_back(cluster_index[i]);
      }

    }

  }

  // create & fill the clusters
  //---------------------------------------------------------------------------
  std::vector<population_pt> candidate_clusters(number_of_clusters);
  std::vector<bool> cluster_active(number_of_clusters, true);

  for (size_t i = 0; i < number_of_clusters; ++i) {
    candidate_clusters[i] = std::make_shared<population_t>();
  }

  for (size_t i = 0; i < cluster_index.size(); ++i) {
    candidate_clusters[cluster_index[i]]->sols.push_back(pop.sols[i]);
    pop.sols[i]->cluster_number = (int) cluster_index[i];

    // only if the elite is the best of that population, we do not run it again.
    // don't get confused by this. We disable the cluster as soon as the first solution is an elite. 
    if (candidate_clusters[cluster_index[i]]->sols.size() == 1 && pop.sols[i]->elite) {
      cluster_active[cluster_index[i]] = false;
    }
  }

  if(add_test_solutions)
  {
    for (size_t i = 0; i < test_points.size(); ++i) {
      candidate_clusters[cluster_index_of_test_points[i]]->sols.push_back(test_points[i]);
    }
  }
  for (size_t i = 0; i < candidate_clusters.size(); ++i)
  {
    if (cluster_active[i]) {
      clusters.push_back(candidate_clusters[i]);
    }
  }
  
  // if there are no clusters remaining, push back all clusters but without elites
  if(clusters.size() == 0 && candidate_clusters.size() > 0) {

    for(size_t i = 0; i < candidate_clusters.size(); ++i) {
      if(candidate_clusters[i]->size() > 1) {
        candidate_clusters[i]->sols[0] = candidate_clusters[i]->sols[candidate_clusters[i]->sols.size()-1];
        candidate_clusters[i]->sols.resize(candidate_clusters[i]->sols.size()-1);
        clusters.push_back(candidate_clusters[i]);
      }
    }
  }

}



