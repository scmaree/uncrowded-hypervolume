#pragma once

/*
 
 Implementation by S.C. Maree, 2018
 s.c.maree[at]amc.uva.nl
 smaree.com
 
 */

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{
  
  class cigtabSep1_t : public fitness_t
  {
    
  public:
    
    cigtabSep1_t()
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // default, can be adapted
      
      partial_evaluations_available = false;
      
      hypervolume_max_f0 = 1.1;
      hypervolume_max_f1 = 1.1;
      
      analytical_gd_avialable = true;
    }
    ~cigtabSep1_t() {}
    
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
      lower.resize(number_of_parameters, -5);
      
      upper.clear();
      upper.resize(number_of_parameters, 5);
      
    }
    
    void define_problem_evaluation(solution_t & sol)
    {
      assert(sol.param.size() >= 2);
      
      // f1
      sol.obj[0] = sol.param[0] * sol.param[0] + 100000000 * sol.param[1] * sol.param[1];
      
      for(size_t i = 2; i < sol.param.size(); ++i) {
        sol.obj[0] += 10000 * sol.param[i] * sol.param[i];
      }
      
      // f2
      sol.obj[1] = (sol.param[0] - 1.0)*(sol.param[0] - 1.0) + 100000000 * sol.param[1] * sol.param[1];
      
      for(size_t i = 2; i < sol.param.size(); ++i) {
        sol.obj[1] += 10000 * sol.param[i] * sol.param[i];
      }
      
      sol.constraint = 0;
    }
    
    void define_partial_problem_evaluation(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
    {
      size_t var;
      
      for(size_t i = 0; i < touched_parameter_idx.size(); ++i)
      {
        var = touched_parameter_idx[i];
        
        // f1
        if(var == 0) {
          sol.obj[0] +=              (sol.param[var] * sol.param[var] - old_sol.param[var] * old_sol.param[var]);
        } else if(var == 1) {
          sol.obj[0] += 100000000  * (sol.param[var] * sol.param[var] - old_sol.param[var] * old_sol.param[var]);
        } else {
          sol.obj[0] += 10000      * (sol.param[var] * sol.param[var] - old_sol.param[var] * old_sol.param[var]);
        }
        
        // f2
        if(var == 0) {
          sol.obj[1] +=              ((sol.param[var] - 1.0) * (1.0 -sol.param[var]) - (old_sol.param[var] * old_sol.param[var]));
        } else if(var == 1) {
          sol.obj[1] += 100000000  * (sol.param[var] * sol.param[var] - old_sol.param[var] * old_sol.param[var]);
        } else {
          sol.obj[1] += 10000      * (sol.param[var] * sol.param[var] - old_sol.param[var] * old_sol.param[var]);
        }
      }
    }
    
    std::string name() const
    {
      return "cigtabSep1";
    }
    
    double distance_to_front(const solution_t & sol)
    {
      
      solution_t ref_sol(sol);
      vec_t obj_ranges(sol.param.size(), 1.0);
      
      for(size_t i = 1; i < ref_sol.param.size(); ++i) {
        ref_sol.param[i] = 0.0;
      }
      
      define_problem_evaluation(ref_sol);
      
      return ref_sol.transformed_objective_distance(sol, obj_ranges);
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
          // double t = i / ((double) pareto_set_size - 1.0);
          // sol->obj[0] = t;
          // sol->obj[1] = (1.0 - sqrt(t)) * (1.0 - sqrt(t));
          pareto_set.sols.push_back(sol);
        }
        
        igdx_available = false;
        igd_available = true;
        // pareto_set.writeToFile("./genMED.txt");
      }
      
      return true;
      
    }
    
  };
}
