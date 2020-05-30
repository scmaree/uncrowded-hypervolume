#pragma once

/*

Implementation by S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

  class eye_t : public fitness_t
  {

  public:

    // data members
    vec_t center0, center1;

    void set_centers(size_t number_of_parameters)
    {
      center0.clear();
      center0.resize(number_of_parameters, 0.0);
      
      center1.clear();
      center1.resize(number_of_parameters, 0.0);
      center1[0] = 1;
      center1[1] = 1;
    }

    eye_t()
    {
      number_of_objectives = 2;
      number_of_parameters = 2; // default, can be adapted

      set_centers(number_of_parameters);

    }
    ~eye_t() {}

    // number of objectives 
    void set_number_of_objectives(size_t & number_of_objectives)
    {
      number_of_objectives = this->number_of_objectives;
    }

    // any positive value
    void set_number_of_parameters(size_t & number_of_parameters)
    {
      this->number_of_parameters = number_of_parameters;
      
      set_centers(number_of_parameters);
    }


    void get_param_bounds(vec_t & lower, vec_t & upper) const
    {

      lower.clear();
      lower.resize(number_of_parameters, -100);
      
      upper.clear();
      upper.resize(number_of_parameters, 100);

    }

    void define_problem_evaluation(solution_t & sol)
    {
      
      sol.obj[0] = pow( distanceEuclidean( sol.param.toArray(), center0.toArray(), (int) number_of_parameters ), 2.0 );
      sol.obj[1] = pow( distanceEuclidean( sol.param.toArray(), center1.toArray(), (int) number_of_parameters ), 2.0 );
      
      double mean_x = 0.0, std_x = 0.0;
      
      for(size_t i = 0; i < number_of_parameters; ++i){
        mean_x += sol.param[i];
      }
      mean_x /= (double) number_of_parameters;
      
      for(size_t i = 0; i < number_of_parameters; ++i) {
        std_x += (sol.param[i] - mean_x) * (sol.param[i] - mean_x);
      }
      std_x = sqrt(std_x / (double) (number_of_parameters-1));
      
      sol.obj[1] -= 2 * std_x * (1.0 - mean_x);
      
      /*
       
       if (number_of_parameters >= 2 && sol.param[0] > sol.param[1])
      {
        sol.obj[1] +=  (sol.param[0] - sol.param[1]);
      }
       */
      
      
      sol.constraint = 0;
      
    }

    bool get_pareto_set()
    {
      return false;
    }

    std::string name() const
    {
      return "eye";
    }

  };
}
