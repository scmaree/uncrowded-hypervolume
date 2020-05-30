#pragma once

/*

HillValleyClustering - adapted to MO


By S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
maree[at]cwi.nl

*/

#include "hicam_internal.h"
#include "fitness.h"
#include "elitist_archive.h"
namespace hicam
{

  class hvc_t
  {

  public:

    // (de)initialize
    //----------------------------------------------------------------------------------
    hvc_t(fitness_pt fitness_function);
    ~hvc_t();

    double get_average_edge_length(const population_t & pop, size_t number_of_parameters);
    double get_max_number_of_neighbours(size_t number_of_parameters);
    
    void cluster(const population_t & pop, std::vector<population_pt> & subpopulations,  unsigned int  & number_of_evaluations, double & average_edge_length, bool add_test_solutions, bool recheck_elites, int optimizer_number, rng_pt & rng);
    
    void HL_filter(const population_t & pop, population_t & new_population, double tolerance);
    
    void cluster_ObjParamDistanceRanks(const population_t & pop, std::vector<population_pt> & subpopulations,  unsigned int  & number_of_evaluations, double & average_edge_length, bool add_test_solutions, bool recheck_elites, int optimizer_number, rng_pt & rng);
    
    void nearest_solution(const population_t & pop, const size_t current_index, const std::vector<size_t> & candidate_indices, const std::vector<vec_t> & pairwise_distance_matrix, size_t number_of_neighbours, std::vector<size_t> & neighbours);
    void nearest_solution(const population_t & pop, const size_t current_index, const std::vector<size_t> & candidate_indices, const std::vector<vec_t> & d_param_ranks, const std::vector<vec_t> & d_obj_ranks, size_t number_of_neighbours, std::vector<size_t> & neighbours);
    void get_candidates(const population_t & pop, const size_t current_index, std::vector<size_t> & candidates);
    void get_pairwise_dist_matrix(const population_t & pop, std::vector<vec_t> & pairwise_distance_matrix, std::vector<size_t> & cluster_start_order, std::vector<vec_t> & d_param);
    
    void computeDVIRanges(const population_t & pop, vec_t & dvi_ranges) const;
    void cluster_on_dvis(const population_t & pop, std::vector<population_pt> & subpopulations,  unsigned int  & number_of_evaluations, double & average_edge_length, bool add_test_solutions, bool recheck_elites, int optimizer_number, rng_pt & rng);
    
    bool MO_HillValleyTest(const solution_t & sol1, const solution_t & sol2, const unsigned int max_trials, unsigned int & number_of_evaluations, std::vector<solution_pt> & test_solutions, bool consider_constraints, bool & rejected_because_out_of_domain);
    bool SO_HillValleyTest(const solution_t & sol1, const solution_t & sol2, size_t objective_number, unsigned int max_trials, unsigned int & number_of_evaluations, std::vector<solution_pt> & test_solutions, bool consider_constraints, bool & rejected_because_out_of_domain);
    
    bool SO_HillValleyTest_point(const solution_t & sol1, const solution_t & sol2, const solution_t &test, size_t objective_number, bool consider_constraints, bool & rejected_because_out_of_domain);

    
    
  private:

    // input parameters
    //---------------------------------------------------------------------------------
    fitness_pt fitness_function;


  };


}
