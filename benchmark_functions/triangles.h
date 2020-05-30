#pragma once

/*

Implementation by S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com
 
 Ishibushi, Akedo, Nojima:
 many-objective test problems for visually examining diversity maintenance behavior in a decision space, GECCO11

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

  class triangles_t : public fitness_t
  {

  public:

    // data members
    std::vector<vec_t> obj1_centers;
    std::vector<vec_t> obj2_centers;
    std::vector<vec_t> obj3_centers;
    size_t number_of_optima;
    
    void set_centers(size_t number_of_parameters)
    {
      
      // Objective 1
      obj1_centers.resize(number_of_optima);
      obj2_centers.resize(number_of_optima);
      obj3_centers.resize(number_of_optima);
      
      if(number_of_optima > 0)
      {
        obj1_centers[0].reset(number_of_parameters, 0.0);
        obj1_centers[0][0] = -3;
        obj1_centers[0][1] = -2;
        if(number_of_parameters >= 3) { obj1_centers[0][2] = 1; }
      
        obj2_centers[0].reset(number_of_parameters, 0.0);
        obj2_centers[0][0] = -2;
        obj2_centers[0][1] = -4;
        if(number_of_parameters >= 3) { obj2_centers[0][2] = 1; }
        
        obj3_centers[0].reset(number_of_parameters, 0.0);
        obj3_centers[0][0] = -4;
        obj3_centers[0][1] = -4;
        if(number_of_parameters >= 3) { obj3_centers[0][2] = 1; }
      }
      
      if(number_of_optima > 1)
      {
        obj1_centers[1].reset(number_of_parameters, 0.0);
        obj1_centers[1][0] = 3;
        obj1_centers[1][1] = 4;
        if(number_of_parameters >= 3) { obj1_centers[0][2] = 1; }
        
        obj2_centers[1].reset(number_of_parameters, 0.0);
        obj2_centers[1][0] = 9 - 5;
        obj2_centers[1][1] = 7 - 5;
        if(number_of_parameters >= 3) { obj2_centers[0][2] = 1; }
        
        obj3_centers[1].reset(number_of_parameters, 0.0);
        obj3_centers[1][0] = 7 - 5;
        obj3_centers[1][1] = 7 - 5;
        if(number_of_parameters >= 4) { obj3_centers[0][2] = 1; }
      }
      
      if(number_of_optima > 2)
      {
        obj1_centers[2].reset(number_of_parameters, 0.0);
        obj1_centers[2][0] = 8 - 5;
        obj1_centers[2][1] = 3 - 5;
        if(number_of_parameters >= 3) { obj1_centers[0][2] = 1; }
        
        obj2_centers[2].reset(number_of_parameters, 0.0);
        obj2_centers[2][0] = 9 - 5;
        obj2_centers[2][1] = 1 - 5;
        if(number_of_parameters >= 3) { obj2_centers[0][2] = 1; }
        
        obj3_centers[2].reset(number_of_parameters, 0.0);
        obj3_centers[2][0] = 7 - 5;
        obj3_centers[2][1] = 1 - 5;
        if(number_of_parameters >= 3) { obj3_centers[0][2] = 1; }
        
      }
      
      if(number_of_optima > 3)
      {
        obj1_centers[3].reset(number_of_parameters, 0.0);
        obj1_centers[3][0] = 2 - 5;
        obj1_centers[3][1] = 9 - 5;
        if(number_of_parameters >= 3) { obj1_centers[0][2] = 1; }
        
        obj2_centers[3].reset(number_of_parameters, 0.0);
        obj2_centers[3][0] = 3 - 5;
        obj2_centers[3][1] = 7 - 5;
        if(number_of_parameters >= 3) { obj1_centers[0][2] = 1; }
        
        obj3_centers[3].reset(number_of_parameters, 0.0);
        obj3_centers[3][0] = 1 - 5;
        obj3_centers[3][1] = 7 - 5;
        if(number_of_parameters >= 3) { obj1_centers[0][2] = 1; }
      }
      
    }

    triangles_t(size_t number_of_optima)
    {
      number_of_objectives = 3;
      number_of_parameters = 2; // default, can be adapted
      
      this->number_of_optima = number_of_optima;
      if(this->number_of_optima <= 0 || this->number_of_optima > 4) {
        this->number_of_optima = 4;
      }

      set_centers(number_of_parameters);

    }
    
    ~triangles_t() {}

    // number of objectives 
    void set_number_of_objectives(size_t & number_of_objectives)
    {
      number_of_objectives = this->number_of_objectives;
    }

    // any positive value
    void set_number_of_parameters(size_t & number_of_parameters)
    {
      
      if(number_of_parameters < 2) {
        number_of_parameters = 2;
      }
      
      this->number_of_parameters = number_of_parameters;
      
      set_centers(number_of_parameters);
    }


    void get_param_bounds(vec_t & lower, vec_t & upper) const
    {

      lower.clear();
      lower.resize(number_of_parameters, -5);
      
      upper.clear();
      upper.resize(number_of_parameters, 5);

    }

    void define_problem_evaluation(solution_t & sol)
    {
     
      vec_t distances(number_of_optima,0.0);
      
      for(size_t i =0; i < number_of_optima; ++i) {
        distances[i] = (sol.param - obj1_centers[i]).norm();
      }
      
      sol.obj[0] = distances.min();
      
      for(size_t i =0; i < number_of_optima; ++i) {
        distances[i] = (sol.param - obj2_centers[i]).norm();
      }
      
      sol.obj[1] = distances.min();
      
      for(size_t i =0; i < number_of_optima; ++i) {
        distances[i] = (sol.param - obj3_centers[i]).norm();
      }
      
      sol.obj[2] = distances.min();
      
      sol.constraint = 0;
      
    }


    std::string name() const
    {
      if(number_of_optima == 1) {
        return "triangles1";
      }
      
      if(number_of_optima == 2) {
        return "triangles2";
      }
      
      if(number_of_optima == 3) {
        return "triangles3";
      }
      
      return "triangles";
    }

    bool get_pareto_set()
    {

      size_t pareto_set_size = 5000;
      size_t quarter_size = (size_t) (pareto_set_size / number_of_optima);
      pareto_set_size = number_of_optima * quarter_size; // makes it divisible by the number_of_optima

      rng_t rng = rng_t(100);
      std::uniform_real_distribution<double> unif(0, 1);
      
      // generate default front
      if (pareto_set.size() != pareto_set_size)
      {

        pareto_set.sols.clear();
        pareto_set.sols.reserve(pareto_set_size);

        double sqrt_r1, r2;
        for(size_t i = 0; i < number_of_optima; ++i)
        {
        
          for(size_t j = 0; j < quarter_size; ++j)
          {
            sqrt_r1 = sqrt(unif(rng));
            r2 = unif(rng);
            
            solution_pt sol = std::make_shared<solution_t>(number_of_parameters, number_of_objectives);
            
            sol->param = (1.0 - sqrt_r1) * obj1_centers[i] + sqrt_r1 * (1.0 - r2) * obj2_centers[i] + r2 * sqrt_r1 * obj3_centers[i];
            define_problem_evaluation(*sol); // runs a feval without registering it.
            
            pareto_set.sols.push_back(sol);
          }
          
        }
        
        igdx_available = true;
        igd_available = true;

        // pareto_set.writeToFile("./genMEDmm.txt");

      }

      return true;
    }

  };
}
