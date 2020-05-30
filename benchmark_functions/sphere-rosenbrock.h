#pragma once

/*
 
 Implementation by S.C. Maree, 2018
 s.c.maree[at]amc.uva.nl
 smaree.com
 
 */

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{
  
  class sphereRosenbrock_t : public fitness_t
  {
  public:
    
    sphereRosenbrock_t()
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // default, can be adapted
      
      partial_evaluations_available = false;
      analytical_gradient_available = true;
      
      hypervolume_max_f0 = 11;
      hypervolume_max_f1 = 11;
    }
    ~sphereRosenbrock_t() {}
    
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
        sol.obj[0] += sol.param[i] * sol.param[i];
      }
      sol.obj[0] /= (double)  (sol.param.size());
      
      // f2
      sol.obj[1] = 0.0;
      
      for(size_t i = 0; i < sol.param.size()-1; ++i) {
        sol.obj[1] += 100.0 * (sol.param[i+1] - sol.param[i]*sol.param[i])*(sol.param[i+1] - sol.param[i]*sol.param[i]) + (1.0 - sol.param[i])*(1.0 - sol.param[i]);
      }
      sol.obj[1] /= (double) (sol.param.size()-1);
      
      
      sol.constraint = 0;
    }
    
    void define_problem_evaluation_with_gradients(solution_t & sol)
    {
      assert(sol.param.size() >= 1);
      
      // f1
      sol.obj[0] = 0;
      
      for(size_t i = 0; i < sol.param.size(); ++i) {
        sol.obj[0] += sol.param[i] * sol.param[i];
      }
      sol.obj[0] /= (double)  (sol.param.size());
      
      // f2
      sol.obj[1] = 0.0;
      
      for(size_t i = 0; i < sol.param.size()-1; ++i) {
        sol.obj[1] += 100.0 * (sol.param[i+1] - sol.param[i]*sol.param[i])*(sol.param[i+1] - sol.param[i]*sol.param[i]) + (1.0 - sol.param[i])*(1.0 - sol.param[i]);
      }
      sol.obj[1] /= (double) (sol.param.size()-1);
      
      
      sol.constraint = 0;
      
      
      // compute gradients
      sol.gradients.resize(number_of_objectives); // 2
      sol.gradients[0].resize(number_of_parameters);
      sol.gradients[1].resize(number_of_parameters);
      
      for(size_t i = 0; i < sol.param.size(); ++i) {
        sol.gradients[0][i] = 2 * sol.param[i] / ((double) sol.param.size());
      }
      
      // f2
      sol.gradients[1][0] = 400 * (sol.param[0]*sol.param[0]*sol.param[0] - sol.param[0] * sol.param[1]) - 2.0 * (1.0 - sol.param[0]);
      
      for(size_t i = 1; i < sol.param.size()-1; ++i) {
        sol.gradients[1][i] = 400 * (sol.param[i]*sol.param[i]*sol.param[i] - sol.param[i] * sol.param[i+1]) - 2.0 * (1.0 - sol.param[i]) + 200 * (sol.param[i] - sol.param[i-1] * sol.param[i-1]);
      }
      
      sol.gradients[1][sol.param.size()-1] = 200 * (sol.param[sol.param.size()-1] - sol.param[sol.param.size()-2] * sol.param[sol.param.size()-2]);
      
      for(size_t i = 0; i < sol.param.size(); ++i) {
        sol.gradients[1][i] /= (double) (sol.param.size()-1);
      }
      
    }
    
    std::string name() const
    {
      return "sphereRosenbrock";
    }
    
    
    // compute VTR in terms of the D_{\mathcal{P}_F}\rightarrow\mathcal{S}
    bool get_pareto_set()
    {
      
      if (pareto_set.size() == 0)
      {
        pareto_set.read2DObjectivesFromFile("../defaultFronts/BD2s.txt", 5000);
        
        // if we couldn't read the default front, disable the vtr.
        if (pareto_set.size() == 0)
        {
          std::cout << "Default front empty. VTR disabled." << std::endl;
          igd_available = false;
          igdx_available = false;
        }
        else
        {
          igd_available = true;
          igdx_available = false;
          return true;
        }
      }
      
      return true;
      
    }

    
  };
}
