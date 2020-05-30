#pragma once

/*

AMaLGaM

Implementation by S.C. Maree, 2017
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

  class BD2_t : public fitness_t
  {

  public:

    BD2_t()
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // default, can be adapted
    }
    ~BD2_t() {}

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
      lower.resize(number_of_parameters, -1e308);

      upper.clear();
      upper.resize(number_of_parameters, 1e308);
      
    }

    void define_problem_evaluation(solution_t & sol)
    {
    
      // f1
      double sum = 0.0;
      for (size_t i = 0; i < number_of_parameters; i++) {
        sum += sol.param[i] * sol.param[i];
      }

      double result = sum;
      result /= (double)number_of_parameters;

      sol.obj[0] = result;
    
      // f2

      sum = 0.0;
      for (size_t i = 0; i < number_of_parameters - 1; i++) {
        sum += 100.0*(sol.param[i + 1] - sol.param[i] * sol.param[i])*(sol.param[i + 1] - sol.param[i] * sol.param[i]) + (1.0 - sol.param[i])*(1.0 - sol.param[i]);
      }

      result = sum;
      result /= (double)(number_of_parameters - 1);

      sol.obj[1] = result;
      sol.constraint = 0;
    }

    std::string name() const
    {
      return "BD2s";
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
