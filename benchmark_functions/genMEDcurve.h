#pragma once

/*

Implementation by S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

  class genMEDcurve_t : public fitness_t
  {

  public:

    // data members
    double exponent; // <1 for a concave front and >1 for a convex front
    vec_t center0, center1;
    double scale_factor; // make an ellipsoid for one objective

    genMEDcurve_t(double exponent)
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // default, can be adapted
      this->exponent = exponent;
      scale_factor = 100;

      // set center0
      center0.clear();
      center0.resize(number_of_parameters, 0.0);
      center0[0] = 1.0;

      // set center1
      center1.clear();
      center1.resize(number_of_parameters, 0.0);
      center1[1] = 1.0;
      
      solution_t dummy0(center0); dummy0.obj.resize(2);
      define_problem_evaluation(dummy0);
      solution_t dummy1(center1); dummy1.obj.resize(2);
      define_problem_evaluation(dummy1);
      
      hypervolume_max_f0 = 11; // 1.1*dummy1.obj[0];
      hypervolume_max_f1 = 11; // 1.1*dummy0.obj[1];
      
      partial_evaluations_available = false;
      analytical_gradient_available = true;

    }
    ~genMEDcurve_t() {}

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
      sol.obj[0] = 0.0;
      sol.obj[1] = 0.0;
      
      for(size_t i = 0; i < number_of_parameters; ++i) {
        if(i == 0) {
          sol.obj[0] += scale_factor * (center0[i] - sol.param[i])*(center0[i] - sol.param[i]) / scale_factor;
        } else {
          sol.obj[0] += (center0[i] - sol.param[i])*(center0[i] - sol.param[i])  / scale_factor;
        }
        sol.obj[1] += (center1[i] - sol.param[i])*(center1[i] - sol.param[i]);
      }
      
      // sol.obj[0] = pow(sqrt(sol.obj[0]) / sqrt(2.0),exponent);
      // sol.obj[1] = pow(sqrt(sol.obj[1]) / sqrt(2.0),exponent);
      
      sol.constraint = 0.0;
      
    }

    void define_problem_evaluation_with_gradients(solution_t & sol)
    {
      sol.obj[0] = 0.0;
      sol.obj[1] = 0.0;
      
      for(size_t i = 0; i < number_of_parameters; ++i) {
        if(i == 0) {
          sol.obj[0] += scale_factor * (center0[i] - sol.param[i])*(center0[i] - sol.param[i])  / scale_factor;
        } else {
          sol.obj[0] += (center0[i] - sol.param[i])*(center0[i] - sol.param[i])  / scale_factor;
        }
        sol.obj[1] += (center1[i] - sol.param[i])*(center1[i] - sol.param[i]);
      }
      
      // sol.obj[0] = pow(sqrt(sol.obj[0]) / sqrt(2.0),exponent);
      // sol.obj[1] = pow(sqrt(sol.obj[1]) / sqrt(2.0),exponent);
      
      sol.gradients.resize(2);
      sol.gradients[0].resize(number_of_parameters);
      sol.gradients[1].resize(number_of_parameters);
      
      for(size_t i = 0; i < number_of_parameters; ++i) {
        if (i == 0) {
          sol.gradients[0][i] = -2 * scale_factor * (center0[i] - sol.param[i])  / scale_factor;
        } else {
          sol.gradients[0][i] = -2 * (center0[i] - sol.param[i])  / scale_factor;
        }
        sol.gradients[1][i] = -2 * (center1[i] - sol.param[i]);
      }
      
      sol.constraint = 0.0;
      
    }
    /*
    void define_partial_problem_evaluation(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
    {
      sol.obj[0] = pow(sol.obj[0],1.0/exponent) * sqrt(2.0);
      sol.obj[0] *= sol.obj[0];
      
      sol.obj[1] = pow(sol.obj[1],1.0/exponent) * sqrt(2.0);
      sol.obj[1] *= sol.obj[1];
      
      size_t var = 0;
      for(size_t i = 0; i < touched_parameter_idx.size(); ++i) {
        var = touched_parameter_idx[i];
        if(var == 0) {
          sol.obj[0] += scale_factor * (center0[var] - sol.param[var]) * (center0[var] - sol.param[var]) - scale_factor * (center0[var] - old_sol.param[var]) * (center0[var] - old_sol.param[var]);
        } else {
          sol.obj[0] += (center0[var] - sol.param[var]) * (center0[var] - sol.param[var]) - (center0[var] - old_sol.param[var]) * (center0[var] - old_sol.param[var]);
        }
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
*/
    std::string name() const
    {
      if(exponent > 1)
        return "genMEDcurve_convex";
      else
        return "genMEDcurve_concave";
    }


    bool get_pareto_set() {
      return false;
    }

  };
}
