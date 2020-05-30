#pragma once

/*

Implementation by S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

  class circles_t : public fitness_t
  {

  public:

    size_t number_of_circles;

    circles_t(size_t number_of_circles)
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2 * number_of_circles;
      this->number_of_circles = number_of_circles;
    }
    circles_t() {}

    // number of objectives 
    void set_number_of_objectives(size_t & number_of_objectives)
    {
      this->number_of_objectives = 2;
      number_of_objectives = this->number_of_objectives;
    }

    // any positive value
    void set_number_of_parameters(size_t & number_of_parameters)
    {
      this->number_of_circles = (size_t) (number_of_parameters / 2.0);
      this->number_of_parameters = 2 * this->number_of_circles;
      number_of_parameters = this->number_of_parameters;
    }


    void get_param_bounds(vec_t & lower, vec_t & upper) const
    {

      lower.clear();
      lower.resize(number_of_parameters, 0);
      
      upper.clear();
      upper.resize(number_of_parameters, 1);

    }

    void define_problem_evaluation(solution_t & sol)
    {
    
      // f1
      vec_t min_dist(number_of_circles, 2.0);
      min_dist.min();
      min_dist.mean();
      
      double dist, dx, dy;
      
      for(size_t j = 1; j < number_of_circles; ++j)
      {
        for(size_t i = 0; i < j; ++i)
        {
          dx = sol.param[2*i] - sol.param[2*j];
          dy = sol.param[2*i+1] - sol.param[2*j+1];
          dist = sqrt(dx*dx + dy*dy);
          
          if(dist < min_dist[i]) {
            min_dist[i] = dist;
          }
          
          if(dist < min_dist[j]) {
            min_dist[j] = dist;
          }
        }
      }
      
      // the circles in a square packing problem aims to maximize the minimum distance between two circles
      // we perform minimization, so we minimize the negatation
      sol.obj[0] = - min_dist.min();
    
      // f2
      // we add a second objective, which is the average distance between two neighbouring solutions.
      sol.obj[1] = 0; //- min_dist.mean();
      
      sol.constraint = 0;
    }


    std::string name() const
    {
      return "CirclesInASquare";
    }

  };
}
