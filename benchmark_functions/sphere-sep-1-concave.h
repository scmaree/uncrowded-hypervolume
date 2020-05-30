#pragma once

/*

Implementation by S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

  class sphereSepConcave_t : public fitness_t
  {

  public:

    sphereSepConcave_t()
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // default, can be adapted
      
      partial_evaluations_available = false;

      hypervolume_max_f0 = 11;
      hypervolume_max_f1 = 11;
      
      analytical_gd_avialable = true;
    }
    ~sphereSepConcave_t() {}

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
      sol.obj[0] = sqrt(sqrt(sol.obj[0]));
    
      // f2
      sol.obj[1] = (sol.param[0] - 1.0)*(sol.param[0] - 1.0);
      
      for(size_t i = 1; i < sol.param.size(); ++i) {
        sol.obj[1] += sol.param[i] * sol.param[i];
      }
      sol.obj[1] = sqrt(sqrt(sol.obj[1]));
      
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
     
      
      // f2
      sol.obj[1] = (sol.param[0] - 1.0)*(sol.param[0] - 1.0);
      
      for(size_t i = 1; i < sol.param.size(); ++i) {
        sol.obj[1] += sol.param[i] * sol.param[i];
      }

      sol.constraint = 0;
      
      // compute gradients
      sol.gradients.resize(number_of_objectives); // 2
      sol.gradients[0].resize(number_of_parameters);
      sol.gradients[0].fill(0.0);
      sol.gradients[1].resize(number_of_parameters);
      sol.gradients[0].fill(0.0);
      
      // f1
      if(sol.obj[0] != 0)
      {
        for(size_t i = 0; i < sol.param.size(); ++i) {
          sol.gradients[0][i] = 2 * sol.param[i];
          sol.gradients[0][i] *= 0.25 * pow(sol.obj[0],-0.75);
          assert(!isnan(sol.gradients[0][i]));
        }
      }
      
      // f2
      if(sol.obj[1] != 0)
      {
        sol.gradients[1][0] = 2 * (sol.param[0] - 1.0);
        sol.gradients[1][0] *= 0.25 * pow(sol.obj[1],-0.75);
        assert(!isnan(sol.gradients[1][0]));
        
        for(size_t i = 1; i < sol.param.size(); ++i) {
          sol.gradients[1][i] = 2 * sol.param[i];
          sol.gradients[1][i] *= 0.25 * pow(sol.obj[1],-0.75);
          assert(!isnan(sol.gradients[1][i]));
        }
      }
      
      sol.obj[0] = sqrt(sqrt(sol.obj[0]));
      sol.obj[1] = sqrt(sqrt(sol.obj[1]));
      
      assert(!isnan(sol.obj[0]));
      assert(!isnan(sol.obj[1]));
    }
    
    std::string name() const
    {
      return "sphereSepConcave";
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
          
          pareto_set.sols.push_back(sol);
        }
        
        igdx_available = true;
        igd_available = true;
        // pareto_set.writeToFile("./genMED.txt");
      }
      
      return true;
      
    }
    
    double distance_to_front(const solution_t & sol)
    {
      
      solution_t ref_sol(sol);
      vec_t obj_ranges(sol.param.size(), 1.0);
      
      for(size_t i = 1; i < ref_sol.param.size(); ++i) {
        ref_sol.param[i] = 0.0;
      }
      
      if(ref_sol.param[0] < 0.0) {
        ref_sol.param[0] = 0.0;
      }
      if(ref_sol.param[0] > 1.0) {
        ref_sol.param[0] = 1.0;
      }
      
      define_problem_evaluation(ref_sol);
      
      return ref_sol.transformed_objective_distance(sol, obj_ranges);
    }

  };
}
