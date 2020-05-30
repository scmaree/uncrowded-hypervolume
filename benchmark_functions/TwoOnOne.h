#pragma once

/*

 Benchmark function by
 
    Preuss, Naujoks, Rudolph. Pareto Set and EMOA Behavior for Simple
    multimodal multiobjective functions. In PPSN, pages 513-522, 2006.

 
Implementation by S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

  class TwoOnOne_t : public fitness_t
  {

  public:
    
    double c,d,k,l;
    
    TwoOnOne_t()
    {
      
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // fixed
      // implemented case 1
      this->c = 10;
      this->d = 0;
      this->k = 0;
      this->l = 0;
    }
    
    ~TwoOnOne_t() {}

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
      lower.resize(number_of_parameters, -3.0);
      
      upper.clear();
      upper.resize(number_of_parameters, 3.0);

    }

    void define_problem_evaluation(solution_t & sol)
    {
    
      // f1
      sol.obj[0] = pow(sol.param[0],4) + pow(sol.param[1],4) - pow(sol.param[0],2) + pow(sol.param[1],2) - c*sol.param[0]*sol.param[1] + d*sol.param[0] + 20;
      sol.obj[0] /= 10;
      // f2
      sol.obj[1] =  pow(sol.param[0]-k,2) + pow(sol.param[1]-l,2);
      sol.obj[1] /= 10;
      
      sol.constraint = 0;
    }


    std::string name() const
    {
      return "TwoOnOne";
    }

    bool get_pareto_set()
    {

      size_t pareto_set_size = 5000;

      // generate default front
      // note, this is an APPROXIMATION, where the distance from the true pareto set is less than 0.045! So the 
      // the true front is a solution of -xy^3-2.5y^2+(x^3-x)y-2.5x^2=0.
    
      if (pareto_set.size() != pareto_set_size)
      {

        pareto_set.sols.clear();
        pareto_set.sols.reserve(pareto_set_size);
        double x1;
        double range;
        // the front
        for (size_t i = 0; i < pareto_set_size; ++i)
        {
          solution_pt sol = std::make_shared<solution_t>(number_of_parameters, number_of_objectives);
          
          range = 0.5 * sqrt(sqrt(101.0) + 1.0);
          x1 = - range + (2.0*range)*(i / ((double)pareto_set_size - 1.0));
          sol->param[0] = x1;
          sol->param[1] = x1 * ((sqrt(101.0) - 1.0) / 10.0);
          
          define_problem_evaluation(*sol); // runs a feval without registering it. 

          pareto_set.sols.push_back(sol);

        }
        
        igd_available = true;
        igdx_available = true;

        // pareto_set.writeToFile("./TwoOnOne.txt");
      }

      return true;

    }

  };
}
