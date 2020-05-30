#pragma once

/*

Implementation by S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

  class genMED_t : public fitness_t
  {

  public:

    // data members
    double exponent; // <1 for a concave front and >1 for a convex front
    vec_t center0, center1;
    

    genMED_t(double exponent)
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // default, can be adapted
      this->exponent = exponent;

      // set center0
      center0.clear();
      center0.resize(number_of_parameters, 0.0);
      center0[0] = 1.0;

      // set center1
      center1.clear();
      center1.resize(number_of_parameters, 0.0);
      center1[1] = 1.0;
      
      hypervolume_max_f0 = 1.1*pow(distanceEuclidean(center1.toArray(), center0.toArray(), (int) number_of_parameters)/sqrt(2.0),exponent);
      hypervolume_max_f1 = 1.1*pow(distanceEuclidean(center1.toArray(), center0.toArray(), (int) number_of_parameters)/sqrt(2.0),exponent);
      
      partial_evaluations_available = false;

    }
    ~genMED_t() {}

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
    
      // f1
      sol.obj[0] = pow(distanceEuclidean(sol.param.toArray(), center0.toArray(), (int) number_of_parameters)/sqrt(2.0),exponent);
    
      // f2
      sol.obj[1] = pow(distanceEuclidean(sol.param.toArray(), center1.toArray(), (int) number_of_parameters) / sqrt(2.0), exponent);
      sol.constraint = 0;
    }

    void define_partial_problem_evaluation(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
    {
      sol.obj[0] = pow(sol.obj[0],1.0/exponent) * sqrt(2.0);
      sol.obj[0] *= sol.obj[0];
      
      sol.obj[1] = pow(sol.obj[1],1.0/exponent) * sqrt(2.0);
      sol.obj[1] *= sol.obj[1];
      
      size_t var = 0;
      for(size_t i = 0; i < touched_parameter_idx.size(); ++i) {
        var = touched_parameter_idx[i];
        sol.obj[0] += (center0[var] - sol.param[var]) * (center0[var] - sol.param[var]) - (center0[var] - old_sol.param[var]) * (center0[var] - old_sol.param[var]);
        sol.obj[1] += (center1[var] - sol.param[var]) * (center1[var] - sol.param[var]) - (center1[var] - old_sol.param[var]) * (center1[var] - old_sol.param[var]);
      }
      
      if(sol.obj[0] < 0 || sol.obj[1] < 0) {
        define_problem_evaluation(sol);
        return;
      }
      
      sol.obj[0] = pow(sqrt(sol.obj[0]) / sqrt(2.0),exponent);
      sol.obj[1] = pow(sqrt(sol.obj[1]) / sqrt(2.0),exponent);
      
      sol.constraint = 0.0;
      
    }

    std::string name() const
    {
      if(exponent > 1)
        return "genMEDconvex";
      else
        return "genMEDconcave";
    }


    bool get_pareto_set()
    {
      
      return false;

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
