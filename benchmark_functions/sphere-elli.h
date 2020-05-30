#pragma once

/*
 
 Implementation by S.C. Maree, 2018
 s.c.maree[at]amc.uva.nl
 smaree.com
 
 */

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{
  
  class sphereElli_t : public fitness_t
  {
    
    void initializeObjectiveRotationMatrix( double rotation_angle, size_t n, matrix_t & rotation_matrix )
    {
      int      i, j, index0, index1;
      double   theta, cos_theta, sin_theta;
      
      if( rotation_angle == 0.0 ) {
        rotation_matrix.setIdentity(n, n);
      }
      
      matrix_t matrix(n,n);
      rotation_matrix.resize(n,n);
      
      /* Initialize the rotation matrix to the identity matrix */
      for( i = 0; i < n; i++ )
      {
        for( j = 0; j < n; j++ )
          rotation_matrix[i][j] = 0.0;
        rotation_matrix[i][i] = 1.0;
      }
      
      /* Construct all rotation matrices (quadratic number) and multiply */
      theta     = (rotation_angle/180.0)*PI;
      cos_theta = cos( theta );
      sin_theta = sin( theta );
      for( index0 = 0; index0 < n-1; index0++ )
      {
        for( index1 = index0+1; index1 < n; index1++ )
        {
          for( i = 0; i < n; i++ )
          {
            for( j = 0; j < n; j++ )
              matrix[i][j] = 0.0;
            matrix[i][i] = 1.0;
          }
          matrix[index0][index0] = cos_theta;
          matrix[index0][index1] = -sin_theta;
          matrix[index1][index0] = sin_theta;
          matrix[index1][index1] = cos_theta;
          
          rotation_matrix = matrix * rotation_matrix;
        }
      }

    }
    
  public:
    
    
    matrix_t rotation_matrix;
    matrix_t sqrt_weight_matrix;
    
    void update_settings()
    {
      initializeObjectiveRotationMatrix(45, number_of_parameters, rotation_matrix);
      sqrt_weight_matrix.setIdentity(number_of_parameters, number_of_parameters);
      
      for(size_t i = 0; i < number_of_parameters; ++i) {
        sqrt_weight_matrix[i][i] = sqrt(pow(10,6.0*i/(number_of_parameters-1.0)));
      }

      hypervolume_max_f0 = 11;
      hypervolume_max_f1 = 11;
    }
    
    sphereElli_t()
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // default, can be adapted
      
      partial_evaluations_available = false;
      analytical_gradient_available = true;
      
      update_settings();
      
    }
    ~sphereElli_t() {}
    
    // number of objectives
    void set_number_of_objectives(size_t & number_of_objectives)
    {
      this->number_of_objectives = 2;
      number_of_objectives = this->number_of_objectives;
      
      update_settings();
    }
    
    // any positive value
    void set_number_of_parameters(size_t & number_of_parameters)
    {
      this->number_of_parameters = number_of_parameters;
      
      update_settings();
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
      
      sol.gradients.resize(2);
      
      sol.obj[0] = sol.param.dot(sol.param);
      
      vec_t optimum(sol.param.size(),0.0);
      optimum[0] = 1.0;
      
      vec_t rotated_parameters =  rotation_matrix * sol.param - optimum;
      vec_t transformed_parameters = sqrt_weight_matrix * rotated_parameters;
      sol.obj[1] = transformed_parameters.dot(transformed_parameters);
      
      
    }
    
    void define_problem_evaluation_with_gradients(solution_t & sol)
    {
      assert(sol.param.size() >= 1);
      
      sol.gradients.resize(2);
      
      sol.obj[0] = sol.param.dot(sol.param);
      sol.gradients[0] = 2 * sol.param;

      vec_t optimum(sol.param.size(),0.0);
      optimum[0] = 1.0;
      
      vec_t rotated_parameters =  rotation_matrix * sol.param - optimum;
      vec_t transformed_parameters = sqrt_weight_matrix * rotated_parameters;
      sol.obj[1] = transformed_parameters.dot(transformed_parameters);

      sol.gradients[1] = 2 * ((sqrt_weight_matrix * rotation_matrix).transpose() * rotated_parameters);
      
    }
    
    std::string name() const
    {
      return "sphereElli";
    }
    
    
    // compute VTR in terms of the D_{\mathcal{P}_F}\rightarrow\mathcal{S}
    bool get_pareto_set()
    {
      
      if (pareto_set.size() == 0)
      {
        pareto_set.read2DObjectivesFromFile("../defaultFronts/sphere-elli.txt", 5000);
        
        // if we couldn't read the default front, disable the vtr.
        if (pareto_set.size() == 0)
        {
          std::cout << "Default front empty. VTR disabled." << std::endl;
          igd_available = false;
          igdx_available = false;
        }
        else
        {
          igd_available = true;
          igdx_available = false;
          return true;
        }
      }
      
      return true;
      
    }
    
  };
}
