#pragma once

/*

AMaLGaM

Implementation by S.C. Maree, 2017
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

  class SYMPART_t : public fitness_t
  {

  public:

    size_t version; // 1,2,3
    double sympartA = 1.0;
    double sympartB = 10.0;
    double sympartC = 8.0;
    double rotationRadian = 0.25*PI;
    double sympart3EpsilonVar = 0.001;
    double lowerLimit = -20.0;
    double upperLimit = 20.0;
    
    SYMPART_t(size_t version)
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // default, can be adapted
      this->version = version;
      
      if (version != 1 && version != 2 && version != 3) {
        this->version = 3;
      }
      
      hypervolume_max_f0 = 11;
      hypervolume_max_f1 = 11;
      
    }
    ~SYMPART_t() {}

    // number of objectives 
    void set_number_of_objectives(size_t & number_of_objectives)
    {
      this->number_of_objectives = 2;
      number_of_objectives = this->number_of_objectives;
    }

    // any positive value
    void set_number_of_parameters(size_t & number_of_parameters)
    {
      this->number_of_parameters = 2;
      number_of_parameters = this->number_of_parameters;
    }


    void get_param_bounds(vec_t & lower, vec_t & upper) const
    {

      lower.clear();
      lower.resize(number_of_parameters, lowerLimit);
      
      upper.clear();
      upper.resize(number_of_parameters, upperLimit);
    }
    
    double getSignValue(double x) {
      if (x > 0) return 1;
      else if (x < 0) return -1;
      else return 0;
    }
    
    void define_problem_evaluation(solution_t & sol)
    {
      double x1, x2;
      
      if(version == 1)
      {
        x1 = sol.param[0];
        x2 = sol.param[1];
      }
      else if(version == 2)
      {
        x1 = sol.param[0] * cos(rotationRadian) - sol.param[1] * sin(rotationRadian);
        x2 = sol.param[0] * sin(rotationRadian) + sol.param[1] * cos(rotationRadian);
      }
      else // version 3
      {
        double tmpVar = (sol.param[1] - lowerLimit + sympart3EpsilonVar) / (upperLimit - lowerLimit);
        double distortedX1 = sol.param[0] * pow(tmpVar, -1.0);
        
        x1 = distortedX1 * cos(rotationRadian) - sol.param[1] * sin(rotationRadian);
        x2 = distortedX1 * sin(rotationRadian) + sol.param[1] * cos(rotationRadian);
      }
      
      double translatedX1,translatedX2;
      double objVar;
      double tmpTileIdentifier1;
      double tmpTileIdentifier2;
      double tileIdentifier1;
      double tileIdentifier2;
      double tmpVar;
      
      tmpVar = (fabs(x1) - (sympartA + (sympartC / 2.0)))  / ((2 * sympartA) + sympartC);
      tmpTileIdentifier1 = getSignValue(x1) * ceil(tmpVar);
      
      tmpVar = (fabs(x2) - (sympartB / 2.0)) / sympartB;
      tmpTileIdentifier2 = getSignValue(x2) * ceil(tmpVar);
      
      tileIdentifier1 = getSignValue(tmpTileIdentifier1) * std::min(fabs(tmpTileIdentifier1), 1.0);
      tileIdentifier2 = getSignValue(tmpTileIdentifier2) * std::min(fabs(tmpTileIdentifier2), 1.0);
      
      translatedX1 = x1 - (tileIdentifier1 * (sympartC + 2 * sympartA));
      translatedX2 = x2 - (tileIdentifier2 * sympartB);
      
      //f1
      objVar = (translatedX1 + sympartA) * (translatedX1 + sympartA) + translatedX2 * translatedX2;
      sol.obj[0] = objVar;
      
      //f2
      objVar = (translatedX1 - sympartA) * (translatedX1 - sympartA) + translatedX2 * translatedX2;
      sol.obj[1] = objVar;
      
      // constraint
      sol.constraint = 0.0;
    }

    std::string name() const
    {
      if(version == 1)
        return "SYMPART1";
      
      if(version == 2)
        return "SYMPART2";
      
      return "SYMPART3";
    }

    // compute VTR in terms of the D_{\mathcal{P}_F}\rightarrow\mathcal{S}
    bool get_pareto_set()
    {
      
      size_t pareto_set_size = 5000;
      
      // generate default front
      if (pareto_set.size() != pareto_set_size)
      {
        
        pareto_set.sols.clear();
        pareto_set.sols.reserve(pareto_set_size);
        
        rng_pt rng = std::make_shared<rng_t>(100);
        std::uniform_real_distribution<double> unif(0, 1);

        double centerX1 = 0;
        double centerX2 = 0;
        int tilePosition;
        double randomVar;
        double tmpX1;
        double tmpX2;
        double tmpVar;
        double distortedX1;
        
        double translatedX1 = 0;
        double translatedX2 = 0;
        
        // the front
        for (size_t i = 0; i < pareto_set_size; ++i)
        {
          solution_pt sol = std::make_shared<solution_t>(number_of_parameters, number_of_objectives);
          
          tilePosition = (int) (9*unif(*rng));
          
          if (tilePosition == 0) {
            centerX1 = -sympartA - sympartC - sympartA;
            centerX2 = sympartB;
          }
          else if (tilePosition == 1) {
            centerX1 = 0.0;
            centerX2 = sympartB;
          }
          else if (tilePosition == 2) {
            centerX1 = sympartA + sympartC + sympartA;
            centerX2 = sympartB;
          }
          else if (tilePosition == 3) {
            centerX1 = -sympartA - sympartC - sympartA;
            centerX2 = 0.0;
          }
          else if (tilePosition == 4) {
            centerX1 = 0.0;
            centerX2 = 0.0;
          }
          else if (tilePosition == 5) {
            centerX1 = sympartA + sympartC + sympartA;
            centerX2 = 0.0;
          }
          if (tilePosition == 6) {
            centerX1 = -sympartA - sympartC - sympartA;
            centerX2 = -sympartB;
          }
          else if (tilePosition == 7) {
            centerX1 = 0.0;
            centerX2 = -sympartB;
          }
          else if (tilePosition == 8) {
            centerX1 = sympartA + sympartC + sympartA;
            centerX2 = -sympartB;
          }
          
          randomVar = (2 * sympartA) * unif(*rng) - sympartA;
          
          tmpX1 = centerX1 + randomVar;
          tmpX2 = centerX2 + 0.0;
          
          if  (version == 1) {
            translatedX1 = tmpX1;
            translatedX2 = tmpX2;
          }
          else if  (version == 2) {
            translatedX1 = tmpX1 * cos(-rotationRadian) - tmpX2 * sin(-rotationRadian);
            translatedX2 = tmpX1 * sin(-rotationRadian) + tmpX2 * cos(-rotationRadian);
          }
          else { // version == 3
            
            translatedX1 = tmpX1 * cos(-rotationRadian) - tmpX2 * sin(-rotationRadian);
            translatedX2 = tmpX1 * sin(-rotationRadian) + tmpX2 * cos(-rotationRadian);

            tmpVar = (translatedX2 - lowerLimit + sympart3EpsilonVar) / (upperLimit - lowerLimit);
            distortedX1 = translatedX1 / pow(tmpVar, -1.0);
            
            translatedX1 = distortedX1;
          }
          
          sol->param[0] = translatedX1;
          sol->param[1] = translatedX2;
          
          define_problem_evaluation(*sol); // runs a feval without registering it.
          
          pareto_set.sols.push_back(sol);
        }
        
        igdx_available = true;
        igd_available = true;
        
      }
      
      return true;
    }

  };
}
