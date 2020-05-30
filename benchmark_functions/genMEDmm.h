#pragma once

/*

Implementation by S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

  class genMEDmm_t : public fitness_t
  {

  public:

    // data members
    vec_t obj0_center0, obj0_center1, obj1_center0, obj1_center1;

    void set_centers(size_t number_of_parameters)
    {
      
      // update  obj1_center0
      obj0_center0.clear();
      obj0_center0.resize(number_of_parameters, 0.0);
      obj0_center0[0] = -2;
      obj0_center0[1] = -1;
      
      // update obj0_center1
      obj0_center1.clear();
      obj0_center1.resize(number_of_parameters, 0.0);
      obj0_center1[0] = 2;
      obj0_center1[1] = 1;
      
      // update  obj1_center0
      obj1_center0.clear();
      obj1_center0.resize(number_of_parameters, 0.0);
      obj1_center0[0] = -2;
      obj1_center0[1] = 1;
      
      // update  obj1_center1
      obj1_center1.clear();
      obj1_center1.resize(number_of_parameters, 0.0);
      obj1_center1[0] = 2;
      obj1_center1[1] = -1;
      
      hypervolume_max_f0 = 2.1;
      hypervolume_max_f1 = 2.1;
      
    }

    genMEDmm_t()
    {
      number_of_objectives = 2; // default, can be adapted
      number_of_parameters = 2; // default, can be adapted

      set_centers(number_of_parameters);

    }
    ~genMEDmm_t() {}

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
      lower.resize(number_of_parameters, -4);
      
      upper.clear();
      upper.resize(number_of_parameters, 4);

    }

    void define_problem_evaluation(solution_t & sol)
    {
      double exponent = 1;
      // f1
      sol.obj[0] = std::min(
                            pow(distanceEuclidean(sol.param.toArray(), obj0_center0.toArray(), (int) number_of_parameters)/sqrt(1.0),exponent),
                            pow(distanceEuclidean(sol.param.toArray(), obj0_center1.toArray(), (int) number_of_parameters)/sqrt(1.0),exponent)
                            );
      
      // f2
      sol.obj[1] = std::min(
                            pow(distanceEuclidean(sol.param.toArray(), obj1_center0.toArray(), (int) number_of_parameters)/sqrt(1.0),exponent),
                            pow(distanceEuclidean(sol.param.toArray(), obj1_center1.toArray(), (int) number_of_parameters)/sqrt(1.0),exponent)
                            );
      
      /*
      sol.obj[0] = std::min(
                            pow(distanceEuclidean(sol.param.toArray(), obj0_center0.toArray(), (int) number_of_parameters)/sqrt(2.0),exponent),
                            pow(distanceEuclidean(sol.param.toArray(), obj0_center1.toArray(), (int) number_of_parameters)/sqrt(2.0),exponent)
                            );
      
      // f2
      sol.obj[1] = std::min(
                            pow(distanceEuclidean(sol.param.toArray(), obj1_center0.toArray(), (int) number_of_parameters)/sqrt(2.0),exponent),
                            pow(distanceEuclidean(sol.param.toArray(), obj1_center1.toArray(), (int) number_of_parameters)/sqrt(2.0),exponent)
                            );
      */
      sol.constraint = 0;
      
    }


    std::string name() const
    {
      return "genMEDmm";
    }

    bool get_pareto_set()
    {

      size_t pareto_set_size = 5000;
      size_t halfsize = (size_t) (pareto_set_size / 2);
      pareto_set_size = 2 * halfsize; // makes it even

      // generate default front
      if (pareto_set.size() != pareto_set_size)
      {

        pareto_set.sols.clear();
        pareto_set.sols.reserve(pareto_set_size);

        // part 1 of the front
        for (size_t i = 0; i < halfsize; ++i)
        {
          solution_pt sol = std::make_shared<solution_t>(number_of_parameters, number_of_objectives);
          
          sol->param = obj0_center0 + (obj1_center0 - obj0_center0)*(i / ((double)halfsize - 1.0));
          define_problem_evaluation(*sol); // runs a feval without registering it. 

          pareto_set.sols.push_back(sol);
        }

        // part 2 of the front
        for (size_t i = 0; i < halfsize; ++i)
        {
          solution_pt sol = std::make_shared<solution_t>(number_of_parameters, number_of_objectives);

          sol->param = obj0_center1 + (obj1_center1 - obj0_center1)*(i / ((double) halfsize - 1.0));
          define_problem_evaluation(*sol); // runs a feval without registering it. 

          pareto_set.sols.push_back(sol);
          
        }
        
        igdx_available = true;
        igd_available = true;

        // pareto_set.writeToFile("./genMEDmm.txt");

      }

      return true;
    }

  };
}
