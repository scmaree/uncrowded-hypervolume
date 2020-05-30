#pragma once

/*
 
 WFG Benchmark set
 Interface implementation by S.C. Maree, 2020
 s.c.maree[at]amc.uva.nl
 smaree.com
 
 */

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{
  
  class wfg_t : public fitness_t
  {
    
  public:
    int fn;
    int k;
    int l;
    
    wfg_t(int function_number);
    ~wfg_t();
  
    void set_number_of_objectives(size_t & number_of_objectives);
    void set_number_of_parameters(size_t & number_of_parameters);
    void get_param_bounds(vec_t & lower, vec_t & upper) const;
    
    void define_problem_evaluation(solution_t & sol);
    std::string name() const;
    
    bool get_pareto_set();
    
  };
}
