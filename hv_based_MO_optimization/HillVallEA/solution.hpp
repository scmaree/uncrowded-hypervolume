#pragma once

/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

*/

#include "param.hpp"
#include "../../domination_based_MO_optimization/mohillvallea/hicam_external.h"
namespace hillvallea
{
  
  class population_t;
  
  class solution_t {
    
  public:
    
    // constructor & destructor
    //-----------------------------------------
    solution_t();
    solution_t(size_t problem_size);
    solution_t(vec_t param);
    solution_t(const solution_t & other);
    ~solution_t();
    
    // For PSC & Indicator-based Multi-objective optimization
    //-----------------------------------------
    std::vector<hicam::solution_pt> mo_reference_sols;
    std::vector<hicam::solution_pt> mo_test_sols;


    // essential data members
    //-----------------------------------------
    vec_t param;            // position of the solution, i.e., the coordinate vector
    double f;               // fitness value
    double constraint;      // penalty value ( set to > 0 if the solution is infeasible )
    int cluster_number;
    
    vec_t param_transformed; // for CMSA-ES
    double multiplier;      // for CMSA-ES
    double NormTabDis;      // for RS-CMSA
    int NIS;                // for GOMEA, no improvement stretch
    
    // Register solution as elite in Hill-Valley Clustering
    //-----------------------------------------
    bool elite;
    
    // for HGML
    //-----------------------------------------
    double probability;    // for spearman rank correlation
    int fitness_rank;      // for spearman rank correlation & GOMEA
    int probability_rank;  // for spearman rank correlation

    // for performance logging of elites
    //-----------------------------------------
    double time_obtained;
    int feval_obtained;
    int generation_obtained;
    
    // for the Sofomore framework
    //-----------------------------------------
    vec_t sof_f;
    vec_t sof_constraint;

    // compare two solutions to see which is best
    //-----------------------------------------
    static bool better_solution_via_pointers(const solution_pt sol1, const solution_pt sol2);
    static bool better_solution(const solution_t & sol1, const solution_t & sol2);
    static bool better_solution_unconstraint(const solution_t & sol1, const solution_t & sol2);
    static bool higher_probability(const std::shared_ptr<solution_t> sol1, const std::shared_ptr<solution_t> sol2);
    
    // distance to another solution in parameter space
    //-----------------------------------------
    double param_distance(const solution_t & sol2) const;
    double param_distance(const vec_t & param2) const;
    
    // Gradients
    //-----------------------------------------
    vec_t gradient;
    vec_t adam_mt;
    vec_t adam_vt;
    
    
  };

}


