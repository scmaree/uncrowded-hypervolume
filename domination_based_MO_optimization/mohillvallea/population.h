#pragma once

/*
 
 HICAM Multi-objective
 
 By S.C. Maree, 2018
 s.c.maree[at]amc.uva.nl
 smaree.com
 
 */

#include "hicam_internal.h"
#include "solution.h"

namespace hicam
{
  
  // a population, basically a list of individuals
  //-----------------------------------------
  class population_t{

  public:

    // constructor & destructor
    //------------------------------------------
    population_t();
    population_t(const population_t & other);
    ~population_t();
    
    elitist_archive_pt elitist_archive;
    population_pt previous;
    
    // data members
    //------------------------------------------
    std::vector<solution_pt > sols; // solutions
    solution_t best;
    solution_t worst;
    std::vector<solution_pt> elites;
    size_t number;           // sometimes you need to number things
    std::vector<cluster_pt> clusters;
    size_t new_elites_added;

    // Dimension accessors
    //------------------------------------------
    size_t size() const; // population size
    size_t popsize() const; // also population size
    size_t problem_size() const; // problem size

    // initialization
    //------------------------------------------
    void fill_uniform(size_t sample_size, size_t problem_size, const vec_t & lower_param_range, const vec_t & upper_param_range, size_t number_of_elites, rng_pt rng);
    void fill_uniform(size_t sample_size, size_t problem_size, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng);
    unsigned int fill_normal_univariate(size_t sample_size, size_t problem_size, const vec_t & mean, const vec_t & univariate_cholesky, bool use_boundary_repair, const vec_t & lower_param_range, const vec_t & upper_param_range, size_t number_of_elites, rng_pt rng);

    unsigned int fill_vector_normal(std::vector<solution_pt> & solutions, size_t sample_size, size_t problem_size, const vec_t & mean, const matrix_t & cholesky, bool use_boundary_repair, const vec_t & lower_param_range, const vec_t & upper_param_range, size_t number_of_elites, rng_pt rng) const;
    unsigned int fill_vector_normal_univariate(std::vector<solution_pt> & solutions, size_t sample_size, size_t problem_size, const vec_t & mean, const vec_t & univariate_cholesky, bool use_boundary_repair, const vec_t & lower_param_range, const vec_t & upper_param_range, size_t number_of_elites, rng_pt rng) const;

    void fill_maximin(size_t sample_size, size_t problem_size, const vec_t & lower_param_range, const vec_t & upper_param_range, const size_t number_of_elites, rng_pt rng);
    void fill_maximin(size_t sample_size, size_t problem_size, const vec_t & lower_param_range, const vec_t & upper_param_range, rng_pt rng);

    

    // Sorting and ranking
    //------------------------------------------
    void sort_on_fitness();
    void removeSolutionNullptrs();
    
    // Distribution parameter estimation
    // Maximum likelihood estimation of mean and covariance
    //-------------------------------------------
    void compute_mean(vec_t & mean) const;
    void compute_covariance(const vec_t & mean, matrix_t & covariance) const;
    void compute_covariance(const vec_t & mean, matrix_t & covariance, bool enable_regularization) const;
    void compute_covariance_univariate(const vec_t & mean, vec_t & univariate_covariance) const;
    void compute_mean_of_selection(vec_t & mean, size_t selection_size) const;
    
    // evaluate all solution in the population
    // returns the number of evaluations
    //------------------------------------------
    void evaluate(fitness_pt fitness_function, const size_t skip_number_of_elites, unsigned int & number_of_evaluations);
    void compute_fitness_ranks();
    void sort_on_ranks();
    
    void getSingleObjectiveRanks(std::vector<size_t> & fitness_ranks, size_t objective_number) const;

    // objective normalization
    void objectiveRanges(vec_t & objective_ranges);
    void computeParameterRanges(vec_t & parameter_ranges) const;
    void setPopulationNumber(int population_number);
    void setPopulationNumber(int population_number, size_t number_of_elites);
    void setClusterNumber(int cluster_number);
    void setClusterNumber(int cluster_number, size_t number_of_elites);
    
    // Selection
    //------------------------------------------
    void truncation_percentage(population_t & selection, double selection_percentage) const;
    void truncation_size(population_t & selection, size_t selection_size) const;
    
    void truncation_percentage(population_t & selection, double selection_percentage, population_t & not_selected_solutions) const;
    void truncation_size(population_t & selection, size_t selection_size, population_t & not_selected_solutions) const;

    void makeSelection(size_t selection_size, const vec_t & objective_ranges, rng_pt & rng);
    void makeSelection(size_t selection_size, const vec_t & objective_ranges, bool use_objective_distances, rng_pt & rng);
    
    // Combine two populations
    //------------------------------------------
    void addSolutions(const population_t & pop);
    void addSolutions(const std::vector<solution_pt> & sols);
    void addSolution(const solution_pt & sol);

    void addCopyOfSolutions(const population_t & pop);
    void addCopyOfSolutions(const std::vector<solution_pt> & sols);
    void addCopyOfSolution(const solution_t & sol);

    void collectSolutions(const std::vector<population_pt> & subpopulations);
    
    // quality measures
    double compute2DHyperVolume(double max_f0, double max_f1) const;
    double compute2DHyperVolumeAlreadySortedOnObj0(double max_0, double max_1) const;
    double computeGD(const population_t & pareto_set) const;
    double computeIGD(const population_t & pareto_set) const;
    double computeIGDX(const population_t & pareto_set) const;
    double computeSR(const std::vector<population_pt> & pareto_sets, double threshold, const vec_t & max_igd) const;
    double computeSmoothness() const;
    
    double computeAnalyticGD(fitness_t & fitness_function) const;
    
    // Population statistics
    //--------------------------------------------
    solution_pt first() const;
    solution_pt last() const;
    void average_fitness(vec_t & mean) const;
    void fitness_variance(vec_t & mean, vec_t & var) const;
    void fitness_std(vec_t & mean, vec_t & std) const;
    double average_constraint() const;
    double constraint_variance() const;
    double constraint_of_first() const;
    double constraint_of_last() const;

    // read & write
    void writeToFile(const char * filename) const;
    void read2DObjectivesFromFile(const char * filename, size_t number_of_lines);
    void writeObjectivesToFile(const char * filename) const;


    // distance to another population
    // based on sol->obj_transformed
    double objective_distance(const population_t & pop, const vec_t & obj_ranges) const;
    double param_distance(const population_t & pop) const;

  };
  


}
