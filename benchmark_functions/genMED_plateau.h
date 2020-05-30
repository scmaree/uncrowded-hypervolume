#pragma once

/*

Implementation by S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

  class genMED_plateau_t : public fitness_t
  {

  public:

    // data members
    vec_t center0, center1;

    genMED_plateau_t()
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // default, can be adapted

      // set center0
      center0.clear();
      center0.resize(number_of_parameters, 0.0);
      center0[0] = 1.0;

      // set center1
      center1.clear();
      center1.resize(number_of_parameters, 0.0);
      center1[1] = 1.0;
      
      hypervolume_max_f0 = 2.1;
      hypervolume_max_f1 = 2.1;

    }
    ~genMED_plateau_t() {}

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
      
      // update center0
      center0.clear();
      center0.resize(number_of_parameters, 0.0);
      center0[0] = 1.0;

      // update center1
      center1.clear();
      center1.resize(number_of_parameters, 0.0);
      center1[1] = 1.0;
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
      // sol.obj[0] = pow(distanceEuclidean(sol.param.toArray(), center0.toArray(), (int) number_of_parameters)/sqrt(2.0),exponent);
      // sol.obj[1] = pow(distanceEuclidean(sol.param.toArray(), center1.toArray(), (int) number_of_parameters) / sqrt(2.0), exponent);
      
      
      sol.obj[0] = distanceAbsolute(sol.param.toArray(), center0.toArray(), (int) number_of_parameters);
      sol.obj[1] = distanceAbsolute(sol.param.toArray(), center1.toArray(), (int) number_of_parameters);
      sol.constraint = 0;
    }


    std::string name() const
    {
      return "genMED_plateau";
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

          sol->param = center0 + (center1 - center0)*(i / ((double)pareto_set_size - 1.0));
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
