#pragma once

/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

*/

#include "hillvallea_internal.hpp"
#include "optimizer.hpp"

namespace hillvallea
{
  
  class hillvallea_t
  {
    
    
  public:
    
    hillvallea_t(
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
    );

    // quick constructor
    hillvallea_t(
      fitness_pt fitness_function,
      const int  number_of_parameters,
      const vec_t & lower_param_bounds,
      const vec_t & upper_param_bounds,
      const int maximum_number_of_evaluations,
      const int random_seed
    );
    
    ~hillvallea_t();
    
    fitness_pt fitness_function;
    int local_optimizer_index;
    int cluster_alg;
    void init_default_params();

    // Run it!
    //--------------------------------------------------------------------------------
    // Runs minimizer. 
    void run();
    void runSerial();
    void runSerial2(size_t popsize, bool enable_clustering, double local_optima_tolerance);
    void runSerial3(size_t popsize, bool enable_clustering, int initial_number_of_evaluations);

    // data members : optimization results
    //--------------------------------------------------------------------------------
    solution_t best;
    std::vector<solution_pt> elitist_archive;
    bool terminated;
    bool success;
    int number_of_evaluations;
    int number_of_evaluations_init;
    int number_of_evaluations_clustering;
    int number_of_generations;
    double selection_fraction_multiplier;
    clock_t starting_time;


    // Algorithm Parameters: initialized by their default values
    //-------------------------------------------------------------------------------
    double population_size_initializer;
    double population_size_incrementer;
    double cluster_size_initializer;
    double cluster_size_incrementer;
    double scaled_search_volume;
    size_t clustering_max_number_of_neighbours;
    double TargetTolFun;
    int add_elites_max_trials;
    
    // Hill-Valley Test and Clustering
    //--------------------------------------------------------------------------------
    void hillvalley_clustering(population_t & pop, std::vector<population_pt> & clusters, size_t max_number_of_neighbours, bool add_test_solutions, bool skip_check_for_worst_solutions, bool check_all_neighbors_from_same_clusters);
    void hillvalley_clustering(population_t & pop, std::vector<population_pt> & clusters);
    bool check_edge(const solution_t & sol1, const solution_t & sol2, int max_trials);
    bool check_edge(const solution_t & sol1, const solution_t & sol2, int max_trials, std::vector<solution_pt> & test_points);

    // Random number generator
    // Mersenne twister
    //------------------------------------
    std::shared_ptr<std::mt19937> rng;
    bool write_elitist_archive;

    
  private:

    // data members : user settings
    //-------------------------------------------------------------------------------
    int  number_of_parameters;
    vec_t  lower_init_ranges;
    vec_t  upper_init_ranges;
    vec_t  lower_param_bounds;
    vec_t  upper_param_bounds;
    int maximum_number_of_evaluations;
    int maximum_number_of_seconds;
    double vtr;
    bool use_vtr;
    int random_seed; 
    bool write_generational_solutions;
    bool write_generational_statistics;
    std::string write_directory;
    std::string file_appendix;
    std::string file_appendix_generation;
    bool evalute_with_gradients;


    // Run-time functions
    //-------------------------------------------------------------------------------
    void initialize(population_pt pop, size_t population_size, double selection_fraction_multiplier, std::vector<optimizer_pt> & local_optimizers, const std::vector<solution_pt> & elitist_archive, bool enable_clustering, size_t target_clustersize);
    void add_elites_to_archive(std::vector<solution_pt> & elitist_archive, const std::vector<solution_pt> & elite_candidates, int & global_opts_found, int & new_global_opts_found);
    
    // Termination criteria
    //-------------------------------------------------------------------------------
    bool terminate_on_runtime() const;
    bool terminate_on_approaching_elite(optimizer_t & local_optimizer, std::vector<solution_pt> & elite_candidates);
    bool terminate_on_converging_to_local_optimum(optimizer_t & local_optimizer, std::vector<solution_pt> & elite_candidates);
   
    // data members : populations
    //--------------------------------------------------------------------------------
    population_pt pop;

    // Output to file
    //-------------------------------------------------------------------------------
    std::ofstream statistics_file;
    void new_statistics_file();
    void new_statistics_file_serial();
    void new_statistics_file_serial(bool use_clustering);
    // void write_statistics_line_serial(const optimizer_pt & optimizer, const solution_t & best);
    void write_statistics_line_serial(const population_t & pop, size_t number_of_generations, const solution_t & best, bool use_clustering);
    void write_statistics_line_serial(const population_t & pop, size_t number_of_generations, const solution_t & best);
    void write_statistics_line_population(const population_t & pop, const std::vector<optimizer_pt> & local_optimizers, const std::vector<solution_pt> & elitist_archive);
    void write_statistics_line_cluster(const population_t & cluster_pop,int cluster_number,  int cluster_generation, const std::vector<solution_pt> & elitist_archive);
    void close_statistics_file();
    void write_population_file(population_pt pop, std::vector<optimizer_pt> & local_optimizers) const; 
    void write_selection_file(population_pt pop, std::vector<optimizer_pt> & local_optimizers) const;
    void write_cluster_population(int generation_nuber, size_t cluster_number, int cluster_generation, population_pt pop) const;
    void write_elitist_archive_file(const std::vector<solution_pt> & elitist_archive, bool final) const;
    void write_CEC2013_niching_file(bool final);

  };
  
  
  
}
