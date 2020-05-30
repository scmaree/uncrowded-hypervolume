#pragma once

/*
 
 Implementation by S.C. Maree, 2018
 s.c.maree[at]amc.uva.nl
 smaree.com
 
 */

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{
  
  class elliSep1_t : public fitness_t
  {
    
  public:
    
    elliSep1_t()
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // default, can be adapted
      
      partial_evaluations_available = false;
      
      hypervolume_max_f0 = 1.1;
      hypervolume_max_f1 = 1.1;
      
    }
    ~elliSep1_t() {}
    
    // number of objectives
    void set_number_of_objectives(size_t & number_of_objectives)
    {
      this->number_of_objectives = 2;
      number_of_objectives = this->number_of_objectives;
    }
    
    // any positive value
    void set_number_of_parameters(size_t & number_of_parameters)
    {
      this->number_of_parameters = number_of_parameters;
    }
    
    
    void get_param_bounds(vec_t & lower, vec_t & upper) const
    {
      
      lower.clear();
      lower.resize(number_of_parameters, -1000);
      
      upper.clear();
      upper.resize(number_of_parameters, 1000);
      
    }
    
    void define_problem_evaluation(solution_t & sol)
    {
      assert(sol.param.size() >= 1);
      
      // f1
      sol.obj[0] = 0;
      
      for(size_t i = 0; i < sol.param.size(); ++i) {
        sol.obj[0] += pow(10,6.0*i/(number_of_parameters-1.0)) * sol.param[i] * sol.param[i];
      }
      
      // f2
      sol.obj[1] = (sol.param[0] - 1.0)*(sol.param[0] - 1.0);
      
      for(size_t i = 1; i < sol.param.size(); ++i) {
        sol.obj[1] += pow(10,6.0*i/(number_of_parameters-1.0)) * sol.param[i] * sol.param[i];
      }
      
      sol.constraint = 0;
    }
    
    void define_partial_problem_evaluation(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
    {
      size_t var;
      
      for(size_t i = 0; i < touched_parameter_idx.size(); ++i)
      {
        // f1
        var = touched_parameter_idx[i];
        
        sol.obj[0] += pow(10,6.0*var/(number_of_parameters-1.0)) * (sol.param[var] * sol.param[var] - old_sol.param[var] * old_sol.param[var]);
        
        // f2
        if(var == 0) {
          sol.obj[1] += (sol.param[0] - 1.0)*(sol.param[0] - 1.0) - (old_sol.param[0] - 1.0)*(old_sol.param[0] - 1.0) ;
        } else {
          sol.obj[1] += pow(10,6.0*var/(number_of_parameters-1.0)) * (sol.param[var] * sol.param[var] - old_sol.param[var] * old_sol.param[var]);
        }
      }
    }
    
    std::string name() const
    {
      return "elliSep1";
    }
    
    
    bool get_pareto_set()
    {
      size_t pareto_set_size = 5000;
      
      // generate default front
      if (pareto_set.size() != pareto_set_size)
      {
        
        pareto_set.sols.clear();
        pareto_set.sols.reserve(pareto_set_size);
        
        // the front
        for (size_t i = 0; i < pareto_set_size; ++i)
        {
          solution_pt sol = std::make_shared<solution_t>(number_of_parameters, number_of_objectives);
          
          sol->param.fill(0);
          sol->param[0] = (i / ((double)pareto_set_size - 1.0));
          define_problem_evaluation(*sol); // runs a feval without registering it.
          
          pareto_set.sols.push_back(sol);
        }
        
        igdx_available = true;
        igd_available = true;
        // pareto_set.writeToFile("./genMED.txt");
      }
      
      return true;
      
    }
    
  };
}
