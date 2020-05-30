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

  class SSUF1_t : public fitness_t
  {

  public:

    SSUF1_t()
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // default, can be adapted
    }
    ~SSUF1_t() {}

    // number of objectives 
    void set_number_of_objectives(size_t & number_of_objectives)
    {
      this->number_of_objectives = 2;
      number_of_objectives = this->number_of_objectives;
    }

    // any positive value
    void set_number_of_parameters(size_t & number_of_parameters)
    {
      this->number_of_parameters = 2;
      number_of_parameters = this->number_of_parameters;
    }


    void get_param_bounds(vec_t & lower, vec_t & upper) const
    {

      lower.clear();
      lower.resize(number_of_parameters, 0.0);
      lower[0] = 1.0;
      lower[1] = -1.0;
      
      upper.clear();
      upper.resize(number_of_parameters, 1.0);
      upper[0] = 3;
      upper[1] = 1.0;
    }

    void define_problem_evaluation(solution_t & sol)
    {
    
      // f1
      sol.obj[0] = fabs(sol.param[0]-2.0);
    
      // f2
      sol.obj[1] = 1.0 - sqrt(sol.obj[0]) + 2.0*pow(sol.param[1]-sin(6.0*PI*sol.obj[0]+PI),2.0);
      
      // constraint
      sol.constraint = 0.0;
    }

    std::string name() const
    {
      return "SSUF1";
    }

    // compute VTR in terms of the D_{\mathcal{P}_F}\rightarrow\mathcal{S}
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
          
          sol->param.fill(0.0);
          sol->param[0] = 1.0 + (3.0 - 1.0)*(i / (double) (pareto_set_size - 1.0));
          sol->param[1] = sin(6*PI*fabs(sol->param[0]-2.0) + PI);
          
          define_problem_evaluation(*sol); // runs a feval without registering it.
          
          pareto_set.sols.push_back(sol);
        }

        igdx_available = true;
        igd_available = true;
        
      }
      
      return true;
    }

  };
}
