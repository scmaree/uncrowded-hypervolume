#pragma once

/*

AMaLGaM

Implementation by S.C. Maree, 2017
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"
#include "../domination_based_MO_optimization/mohillvallea/elitist_archive.h"

namespace hicam
{

  class ZDT3_t : public fitness_t
  {

  public:

    ZDT3_t()
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // default, can be adapted
      
      hypervolume_max_f0 = 11;
      hypervolume_max_f1 = 11;
      analytical_gd_avialable = true;
      
    }
    ~ZDT3_t() {}

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
      lower.resize(number_of_parameters, 0.0);
      
      upper.clear();
      upper.resize(number_of_parameters, 1.0);

    }

    void define_problem_evaluation(solution_t & sol)
    {
    
      // f1
      sol.obj[0] = sol.param[0];
    
      // f2
      double g = 0.0;
      for (size_t i = 1; i < number_of_parameters; i++) {
        g += sol.param[i];
      }
      g = 1.0 + 9.0 / (number_of_parameters - 1.0) * g;

      double h = 1.0 - sqrt(sol.obj[0] / g) - (sol.obj[0] / g)*sin(10.0*PI*sol.obj[0]);

      sol.obj[1] = g*h;
      sol.constraint = 0;
    }

    std::string name() const
    {
      return "ZDT3";
    }

    // compute VTR in terms of the D_{\mathcal{P}_F}\rightarrow\mathcal{S}
    bool get_pareto_set()
    {
      
      size_t pareto_set_size = 5000;
      
      // generate default front
      // this one is annyoing.
      // the front is only part of the described curve.
      // so we create an archive out of it and discard the dominated solutions.
      if (pareto_set.size() < 10)
      {
        rng_pt rng = std::make_shared<rng_t>(100); // not used anyways as the archive size is never adapted here
        elitist_archive_t temp_archive(5000, rng);
        
        size_t temp_pareto_set_size = 18975;
        
        pareto_set.sols.clear();
        pareto_set.sols.reserve(pareto_set_size);
        
        // the front
        for (size_t i = 0; i < temp_pareto_set_size; ++i)
        {
          solution_pt sol = std::make_shared<solution_t>(number_of_parameters, number_of_objectives);
          
          sol->param.fill(0.0);
          sol->param[0] = (i / ((double)temp_pareto_set_size - 1.0));
          
          define_problem_evaluation(*sol); // runs a feval without registering it.
          
          temp_archive.updateArchive(sol);
          
        }
        
        pareto_set.sols = temp_archive.sols;
        
        // pareto_set.writeToFile("./ZDT3.txt");
        igdx_available = true;
        igd_available = true;
      }
      
      return true;
    }
  
    
     double distance_to_front(const solution_t & sol)
     {
     
       solution_t ref_sol(sol);
       
       for(size_t i = 1; i < ref_sol.param.size(); ++i) {
         ref_sol.param[i] = 0.0;
       }
       
       if(ref_sol.param[0] < 0.0) {
         ref_sol.param[0] = 0.0;
       }
       if(ref_sol.param[0] > 1.0) {
         ref_sol.param[0] = 1.0;
       }
       
     
       vec_t left(5,0.0), right(5,0.0);
       left[0] = 0.0;
       right[0] = 0.0830015349269116;
       
       left[1] = 0.1822287280293998;
       right[1] = 0.2577623633878302;
       
       left[2] = 0.4093136748086568;
       right[2] = 0.4538821040888302;
       
       left[3] = 0.6183967944392658;
       right[3] = 0.6525117038046625;
       
       left[4] = 0.8233317983266328;
       right[4] = 0.8518328654364139;
       
       double x = ref_sol.param[0];
       bool found = false;
       for (size_t i = 0; i < 5; ++i)
       {
         // x is in the range of the PS
         if (x >= left[i] && x <= right[i]) {
           found = true;
           break;
         }
       }
     
       if(!found)
       {
         // x is not in the range of the PS, round it
         // to the nearest endpoint.
         double min_dist = 1e300;
         double dist = 0.0;
         double ref_x = 0.0;
         
         for (size_t i = 0; i < 5; ++i)
         {
           dist = fabs(left[i] - x);
           if(dist < min_dist) {
             min_dist = dist;
             ref_x = left[i];
           }
           
           dist = fabs(right[i] - x);
           if(dist < min_dist) {
             min_dist = dist;
             ref_x = right[i];
           }
         }
         
         ref_sol.param[0] = ref_x;
       }

     define_problem_evaluation(ref_sol);
     
       vec_t obj_ranges(sol.param.size(), 1.0);
     return ref_sol.transformed_objective_distance(sol, obj_ranges);
     }
    


  };
}
