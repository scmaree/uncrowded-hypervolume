#pragma once

/*

Implementation by S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

  class MinDistmm_t : public fitness_t
  {

  public:

  
    std::vector<vec_t> f0_opts;
    std::vector<vec_t> f1_opts;
    
    vec_t f0_opts_height;
    vec_t f1_opts_height;
    
    void set_centers(size_t number_of_parameters)
    {
      
      assert(number_of_parameters >= 2);
      
      // % x,y,delta_f
      /// f0_opts = [ -2 0.5 0.1 ; 0 0.5 0.06 ; 2 0.5 0];
      // f1_opts = [ -2 -0.5 0; 0 -0.5 0.06; 2 -0.5 0.1];

      f0_opts.clear();
      f0_opts.resize(3);
      f0_opts_height.resize(3);
      
      f0_opts[0].resize(number_of_parameters, -2.0);
      f0_opts[0][0] = -2;
      f0_opts[0][1] = 0.5;
      f0_opts_height[0] = 0.2;
      
      f0_opts[1].resize(number_of_parameters, 0.0);
      f0_opts[1][0] = 0;
      f0_opts[1][1] = 0.6;
      f0_opts_height[1] = 0.085; // 0.05;
      
      f0_opts[2].resize(number_of_parameters, 2.0);
      f0_opts[2][0] = 2;
      f0_opts[2][1] = 0.5;
      f0_opts_height[2] = 0;
      
      f1_opts.clear();
      f1_opts.resize(3);
      f1_opts_height.resize(3);
      
      f1_opts[0].resize(number_of_parameters, -2.0);
      f1_opts[0][0] = -2;
      f1_opts[0][1] = -0.5;
      f1_opts_height[0] = 0;
      
      f1_opts[1].resize(number_of_parameters, 0.0);
      f1_opts[1][0] = 0;
      f1_opts[1][1] = -0.6;
      f1_opts_height[1] = 0.085; // 0.05;
      
      f1_opts[2].resize(number_of_parameters, 2.0);
      f1_opts[2][0] = 2;
      f1_opts[2][1] = -0.5;
      f1_opts_height[2] = 0.2;
      
      hypervolume_max_f0 = 11;
      hypervolume_max_f1 = 11;
      
    }

    MinDistmm_t()
    {
      number_of_objectives = 2; // default, can be adapted
      number_of_parameters = 2; // default, can be adapted

      set_centers(number_of_parameters);

    }
    ~MinDistmm_t() {}

    // number of objectives 
    void set_number_of_objectives(size_t & number_of_objectives)
    {
      number_of_objectives = this->number_of_objectives;
    }

    // any positive value
    void set_number_of_parameters(size_t & number_of_parameters)
    {
      this->number_of_parameters = number_of_parameters;
      
      assert(number_of_parameters >= 2);
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
      sol.obj.resize(2);
      sol.obj.fill(1e300);
      
      double dist;
      double exponent = 2;
      
      for(size_t i = 0; i < f0_opts.size(); ++i) {
        //if(i == 1) { exponent = 1.4; } else {exponent = 1.9;}
        dist = f0_opts_height[i] + pow((f0_opts[i] - sol.param).norm(),exponent);
        if( dist < sol.obj[0] ) {
          sol.obj[0] = dist;
        }
      }
      
      for(size_t i = 0; i < f1_opts.size(); ++i) {
        //if(i == 1) { exponent = 1.4; } else {exponent = 1.9;}
        dist = f1_opts_height[i] + pow((f1_opts[i] - sol.param).norm(),exponent);
        if( dist < sol.obj[1] ) {
          sol.obj[1] = dist;
        }
      }
      
      sol.constraint = 0;
      
    }


    std::string name() const
    {
      return "MinDistmm";
    }

    bool get_pareto_set()
    {
      
      size_t pareto_set_size = 5000;
      size_t halfsize = (size_t) (pareto_set_size / f0_opts.size()); // haha its a third..
      pareto_set_size = 3 * f0_opts.size(); // makes it even

      // generate default front
      if (pareto_set.size() != pareto_set_size)
      {

        pareto_set.sols.clear();
        pareto_set.sols.reserve(pareto_set_size);

        // part 1 of the front
        for(size_t j = 0; j < f0_opts.size(); ++j)
        {
          for (size_t i = 0; i < halfsize; ++i)
          {
            solution_pt sol = std::make_shared<solution_t>(number_of_parameters, number_of_objectives);
            
            sol->param = f0_opts[j] + (f1_opts[j] - f0_opts[j])*(i / ((double)halfsize - 1.0));
            define_problem_evaluation(*sol); // runs a feval without registering it.

            pareto_set.sols.push_back(sol);
          }
        }
        
        igdx_available = true;
        igd_available = true;

        //std::ostringstream ss;
        //ss << name() << ".txt";
        //pareto_set.writeToFile(ss.str().c_str());

      }

      return true;

    }

  };
}
