/*
 * Copyright � 2005 The Walking Fish Group (WFG).
 *
 * This material is provided "as is", with no warranty expressed or implied.
 * Any use is at your own risk. Permission to use or copy this software for
 * any purpose is hereby granted without fee, provided this notice is
 * retained on all copies. Permission to modify the code and to distribute
 * modified code is granted, provided a notice that the code was modified is
 * included with the above copyright notice.
 *
 * http://www.wfg.csse.uwa.edu.au/
 */


/*
 * main.cpp
 *
 * This file contains a simple driver for testing the WFG problems and
 * transformation functions from the WFG test problem toolkit.
 *
 * Changelog:
 *   2005.06.01 (Simon Huband)
 *     - Corrected commments to indicate k and l are the number of position
 *       and distance parameters, respectively (not the other way around).
 */


//// Standard includes. /////////////////////////////////////////////////////

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>
#include <cassert>
#include <cmath>


//// Toolkit includes. //////////////////////////////////////////////////////


#include "wfg.h"
#include "wfg_Toolkit/ExampleProblems.h"
#include "wfg_Toolkit/TransFunctions.h"

#include "../domination_based_MO_optimization/mohillvallea/elitist_archive.h"

//// Used namespaces. ///////////////////////////////////////////////////////

using namespace WFG::Toolkit;
using namespace WFG::Toolkit::Examples;
using std::vector;
using std::string;


//// Local functions. ///////////////////////////////////////////////////////

namespace
{
  
  //** Using a uniform random distribution, generate a number in [0,bound]. ***
  
  double next_double( const double bound = 1.0 )
  {
    assert( bound > 0.0 );
    
    return bound * rand() / static_cast< double >( RAND_MAX );
  }
  
  
  //** Create a random Pareto optimal solution for WFG1. **********************
  
  vector< double > WFG_1_random_soln( const int k, const int l )
  {
    vector< double > result;  // the result vector
    
    
    //---- Generate a random set of position parameters.
    
    for( int i = 0; i < k; i++ )
    {
      // Account for polynomial bias.
      result.push_back( pow( next_double(), 50.0 ) );
    }
    
    
    //---- Set the distance parameters.
    
    for( int i = k; i < k+l; i++ )
    {
      result.push_back( 0.35 );
    }
    
    
    //---- Scale to the correct domains.
    
    for( int i = 0; i < k+l; i++ )
    {
      result[i] *= 2.0*(i+1);
    }
    
    
    //---- Done.
    
    return result;
  }
  
  
  //** Create a random Pareto optimal solution for WFG2-WFG7. *****************
  
  vector< double > WFG_2_thru_7_random_soln( const int k, const int l )
  {
    vector< double > result;  // the result vector
    
    
    //---- Generate a random set of position parameters.
    
    for( int i = 0; i < k; i++ )
    {
      result.push_back( next_double() );
    }
    
    
    //---- Set the distance parameters.
    
    for( int i = k; i < k+l; i++ )
    {
      result.push_back( 0.35 );
    }
    
    
    //---- Scale to the correct domains.
    
    for( int i = 0; i < k+l; i++ )
    {
      result[i] *= 2.0*(i+1);
    }
    
    
    //---- Done.
    
    return result;
  }
  
  
  //** Create a random Pareto optimal solution for WFG8. **********************
  
  vector< double > WFG_8_random_soln( const int k, const int l )
  {
    vector< double > result;  // the result vector
    
    
    //---- Generate a random set of position parameters.
    
    for( int i = 0; i < k; i++ )
    {
      result.push_back( next_double() );
    }
    
    
    //---- Calculate the distance parameters.
    
    for( int i = k; i < k+l; i++ )
    {
      const vector< double >  w( result.size(), 1.0 );
      const double u = TransFunctions::r_sum( result, w  );
      
      const double tmp1 = fabs( floor( 0.5 - u ) + 0.98/49.98 );
      const double tmp2 = 0.02 + 49.98*( 0.98/49.98 - ( 1.0 - 2.0*u )*tmp1 );
      
      result.push_back( pow( 0.35, pow( tmp2, -1.0 ) ));
    }
    
    
    //---- Scale to the correct domains.
    
    for( int i = 0; i < k+l; i++ )
    {
      result[i] *= 2.0*(i+1);
    }
    
    
    //---- Done.
    
    return result;
  }
  
  
  //** Create a random Pareto optimal solution for WFG9. **********************
  
  vector< double > WFG_9_random_soln( const int k, const int l )
  {
    vector< double > result( k+l );  // the result vector
    
    
    //---- Generate a random set of position parameters.
    
    for( int i = 0; i < k; i++ )
    {
      result[i] = next_double();
    }
    
    
    //---- Calculate the distance parameters.
    
    result[k+l-1] = 0.35;  // the last distance parameter is easy
    
    for( int i = k+l-2; i >= k; i-- )
    {
      vector< double > result_sub;
      for( int j = i+1; j < k+l; j++ )
      {
        result_sub.push_back( result[j] );
      }
      
      const vector< double > w( result_sub.size(), 1.0 );
      const double tmp1 = TransFunctions::r_sum( result_sub, w  );
      
      result[i] = pow( 0.35, pow( 0.02 + 1.96*tmp1, -1.0 ) );
    }
    
    
    //---- Scale to the correct domains.
    
    for( int i = 0; i < k+l; i++ )
    {
      result[i] *= 2.0*(i+1);
    }
    
    
    //---- Done.
    
    return result;
  }
  
  
  //** Create a random Pareto optimal solution for I1. *****************
  
  vector< double > I1_random_soln( const int k, const int l )
  {
    vector< double > result;  // the result vector
    
    
    //---- Generate a random set of position parameters.
    
    for( int i = 0; i < k; i++ )
    {
      result.push_back( next_double() );
    }
    
    
    //---- Set the distance parameters.
    
    for( int i = k; i < k+l; i++ )
    {
      result.push_back( 0.35 );
    }
    
    
    //---- Done.
    
    return result;
  }
  
  
  //** Create a random Pareto optimal solution for I2. **********************
  
  vector< double > I2_random_soln( const int k, const int l )
  {
    vector< double > result( k+l );  // the result vector
    
    
    //---- Generate a random set of position parameters.
    
    for( int i = 0; i < k; i++ )
    {
      result[i] = next_double();
    }
    
    
    //---- Calculate the distance parameters.
    
    result[k+l-1] = 0.35;  // the last distance parameter is easy
    
    for( int i = k+l-2; i >= k; i-- )
    {
      vector< double > result_sub;
      for( int j = i+1; j < k+l; j++ )
      {
        result_sub.push_back( result[j] );
      }
      
      const vector< double > w( result_sub.size(), 1.0 );
      const double tmp1 = TransFunctions::r_sum( result_sub, w  );
      
      result[i] = pow( 0.35, pow( 0.02 + 1.96*tmp1, -1.0 ) );
    }
    
    
    //---- Done.
    
    return result;
  }
  
  
  //** Create a random Pareto optimal solution for I3. **********************
  
  vector< double > I3_random_soln( const int k, const int l )
  {
    vector< double > result;  // the result vector
    
    
    //---- Generate a random set of position parameters.
    
    for( int i = 0; i < k; i++ )
    {
      result.push_back( next_double() );
    }
    
    
    //---- Calculate the distance parameters.
    
    for( int i = k; i < k+l; i++ )
    {
      const vector< double >  w( result.size(), 1.0 );
      const double u = TransFunctions::r_sum( result, w  );
      
      const double tmp1 = fabs( floor( 0.5 - u ) + 0.98/49.98 );
      const double tmp2 = 0.02 + 49.98*( 0.98/49.98 - ( 1.0 - 2.0*u )*tmp1 );
      
      result.push_back( pow( 0.35, pow( tmp2, -1.0 ) ));
    }
    
    
    //---- Done.
    
    return result;
  }
  
  
  //** Create a random Pareto optimal solution for I4. **********************
  
  vector< double > I4_random_soln( const int k, const int l )
  {
    return I1_random_soln( k, l );
  }
  
  
  //** Create a random Pareto optimal solution for I5. **********************
  
  vector< double > I5_random_soln( const int k, const int l )
  {
    return I3_random_soln( k, l );
  }
  
  
  //** Generate a random solution for a given problem. ************************
  
  vector< double > problem_random_soln
  (
   const int k,
   const int l,
   const std::string fn
   )
  {
    if ( fn == "WFG1" )
    {
      return WFG_1_random_soln( k, l );
    }
    else if
      (
       fn == "WFG2" ||
       fn == "WFG3" ||
       fn == "WFG4" ||
       fn == "WFG5" ||
       fn == "WFG6" ||
       fn == "WFG7"
       )
    {
      return WFG_2_thru_7_random_soln( k, l );
    }
    else if ( fn == "WFG8" )
    {
      return WFG_8_random_soln( k, l );
    }
    else if ( fn == "WFG9" )
    {
      return WFG_9_random_soln( k, l );
    }
    else if ( fn == "I1" )
    {
      return I1_random_soln( k, l );
    }
    else if ( fn == "I2" )
    {
      return I2_random_soln( k, l );
    }
    else if ( fn == "I3" )
    {
      return I3_random_soln( k, l );
    }
    else if ( fn == "I4" )
    {
      return I4_random_soln( k, l );
    }
    else if ( fn == "I5" )
    {
      return I5_random_soln( k, l );
    }
    else
    {
      assert( false );
      return vector< double >();
    }
  }
  
  
  //** Calculate the fitness for a problem given some parameter set. **********
  
  vector< double > problem_calc_fitness
  (
   const vector< double >& z,
   const int k,
   const int M,
   const std::string fn
   )
  {
    if ( fn == "WFG1" )
    {
      return Problems::WFG1( z, k, M );
    }
    else if ( fn == "WFG2" )
    {
      return Problems::WFG2( z, k, M );
    }
    else if ( fn == "WFG3" )
    {
      return Problems::WFG3( z, k, M );
    }
    else if ( fn == "WFG4" )
    {
      return Problems::WFG4( z, k, M );
    }
    else if ( fn == "WFG5" )
    {
      return Problems::WFG5( z, k, M );
    }
    else if ( fn == "WFG6" )
    {
      return Problems::WFG6( z, k, M );
    }
    else if ( fn == "WFG7" )
    {
      return Problems::WFG7( z, k, M );
    }
    else if ( fn == "WFG8" )
    {
      return Problems::WFG8( z, k, M );
    }
    else if ( fn == "WFG9" )
    {
      return Problems::WFG9( z, k, M );
    }
    else if ( fn == "I1" )
    {
      return Problems::I1( z, k, M );
    }
    else if ( fn == "I2" )
    {
      return Problems::I2( z, k, M );
    }
    else if ( fn == "I3" )
    {
      return Problems::I3( z, k, M );
    }
    else if ( fn == "I4" )
    {
      return Problems::I4( z, k, M );
    }
    else if ( fn == "I5" )
    {
      return Problems::I5( z, k, M );
    }
    else
    {
      assert( false );
      return vector< double >();
    }
  }

}

//// Standard functions. ////////////////////////////////////////////////////


hicam::wfg_t::wfg_t(int function_number)
{
  fn = function_number;
  
  number_of_objectives = 2;
  
  hypervolume_max_f0 = 11;
  hypervolume_max_f1 = 11;
  
  partial_evaluations_available = false;
  analytical_gd_avialable = false;

  // k = k_factor*( (int) number_of_objectives-1 );  // k (# position parameters) = k_factor*( M-1 )
  // l = l_factor*2; // l (# distance parameters) = l_factor*2
  l = 20;
  
  if(number_of_objectives == 2) {
    k = 4;
  } else {
    k = 2 * ((int) number_of_objectives - 1);
  }
  
  number_of_parameters = k + l; // default, can be adapted
  
}

hicam::wfg_t::~wfg_t()
{
  
}
    
// number of objectives
void hicam::wfg_t::set_number_of_objectives(size_t & number_of_objectives) {
  number_of_objectives  = 2;
}

// any positive value
void hicam::wfg_t::set_number_of_parameters(size_t & number_of_parameters) {
  this->number_of_parameters = this->number_of_parameters;
}


void hicam::wfg_t::get_param_bounds(vec_t & lower, vec_t & upper) const
{
  
  lower.clear();
  lower.resize(number_of_parameters, 0);
  
  upper.clear();
  upper.resize(number_of_parameters, 1000);
  
  for(size_t i = 0; i < number_of_parameters; ++i) {
    upper[i] = 2.0*( i+1 );
  }
  
}

void hicam::wfg_t::define_problem_evaluation(solution_t & sol)
{
  assert(sol.param.size() >= 1);
  
  sol.obj.resize(number_of_objectives,0);
  sol.constraint = 0;
  
  std::string fn_str;
  
  std::ostringstream ss;
  ss << "WFG" << fn;
  fn_str = ss.str();
  
  const vector< double >& f = problem_calc_fitness( sol.param, k, (int) number_of_objectives, fn_str );

  for(size_t i = 0; i < f.size(); ++i) {
    sol.obj[i] = f[i];
  }
}

std::string hicam::wfg_t::name() const
{
  std::ostringstream ss;
  ss << "WFG" << fn;
  return ss.str();
}
    

// compute VTR in terms of the D_{\mathcal{P}_F}\rightarrow\mathcal{S}
bool hicam::wfg_t::get_pareto_set()
{
  // We can randomly sample the Pareto set (+front),
  // but this gives an irregular distribution in objective space, especially
  // for the endpoints of the fronts. Therefore, we can also use the parametric expression of the front,
  // but this lacks the decision variables, and the IGDX and MR(=SR) cannot be computed therefore.
  bool generate_pareto_set_from_random_sampling = false;
  
  if(number_of_objectives > 2) {
    generate_pareto_set_from_random_sampling = true; // the parametric form is only implemented for M = 2
  }
  
  if(fn == 1) {
    generate_pareto_set_from_random_sampling = false; // there is something wrong for this parametric reference front.
  }
  
  size_t pareto_set_size = 5000;
  
  // if the front is already computed.
  if(pareto_set.size() > 0) {
    return true;
  }
  
  if(generate_pareto_set_from_random_sampling)
  {
    std::string fn_str;
    
    std::ostringstream ss;
    ss << "WFG" << fn;
    fn_str = ss.str();
    
    rng_pt rng = std::make_shared<rng_t>(100); // not used anyways as the archive size is never adapted here
    elitist_archive_t temp_archive(pareto_set_size * 10, rng);
    
    srand( 0 );  // seed the random number generator
    
    // the front
    size_t sample_set_size = 4.504 * pareto_set_size;
    for (size_t i = 0; i < sample_set_size; ++i)
    {
      solution_pt sol = std::make_shared<solution_t>(number_of_parameters, number_of_objectives);
      
      const vector< double >& z = problem_random_soln( k, l, fn_str );
      const vector< double >& f = problem_calc_fitness( z, k, (int) number_of_objectives, fn_str );
      
      for(size_t j = 0; j < z.size(); ++j) {
        sol->param[j] = z[j];
      }
      
      for(size_t j = 0; j < f.size(); ++j) {
        sol->obj[j] = f[j];
      }
      
      if(fn == 2) {
        temp_archive.updateArchive(sol);
      } else {
        pareto_set.sols.push_back(sol);
      }
    }
    
    if(fn == 2) {
      pareto_set.sols = temp_archive.sols;
    }
    
    igdx_available = false;
    igd_available = true;
    
    return true;
  }
  
  //-----------------------------------------------------------------
  // Construct the PF from the parametric expression of the front.
  pareto_set.sols.clear();
  pareto_set.sols.reserve(pareto_set_size);
  
  igdx_available = false;
  igd_available = true;
  
  if (fn == 1)
  {
    // there is something off with this reference front
    // the font computed here is better than the one computed by randomly sampling it.
    // i guess this is wrong, but i got no clue how.?
    for (size_t i = 0; i < pareto_set_size; ++i)
    {
      solution_pt sol = std::make_shared<solution_t>(0, number_of_objectives);
      
      double t = (i / ((double)pareto_set_size - 1.0)); // from 0 to 2
      sol->obj[0] = 1-cos(t*PI*0.5);
      sol->obj[1] = 1-t-cos(2*5*PI*t + PI*0.5) / (2*5*PI);
      
      sol->obj[0] *= 2;
      sol->obj[1] *= 4;
      pareto_set.sols.push_back(sol);
    }
  }
  
  
  if (fn == 2)
  {
    // matlab: t = 0:0.01:1;
    // x = 2*(1-cos(t*pi/2));
    // y = 4 - 4*t .* cos(5*t*pi).^2; plot(x,y);

    // this is a multimodal front
    // so filter out the dominated solutions
    pareto_set_size *= 5;
    rng_pt rng = std::make_shared<rng_t>(100); // not used anyways as the archive size is never adapted here
    elitist_archive_t temp_archive(pareto_set_size * 10, rng);
    
    for (size_t i = 0; i < pareto_set_size; ++i)
    {
      solution_pt sol = std::make_shared<solution_t>(0, number_of_objectives);
      
      double t = (i / ((double)pareto_set_size - 1.0)); // from 0 to 1
      sol->obj[0] = 2.0 * (1.0 - cos(0.5 * t * PI));
      sol->obj[1] = 4.0 - 4.0 * cos(5*t*PI)*cos(5*t*PI);
      
      temp_archive.updateArchive(sol);
    }
    
    pareto_set.sols = temp_archive.sols;
  }
  
  if (fn == 3)
  {
    // the front
    for (size_t i = 0; i < pareto_set_size; ++i)
    {
      solution_pt sol = std::make_shared<solution_t>(0, number_of_objectives);
      
      double t = 2 * (i / ((double)pareto_set_size - 1.0)); // from 0 to 2
      sol->obj[0] = t; // center0 + (center1 - center0)*(i / ((double)pareto_set_size - 1.0));
      sol->obj[1] = 2.0 + 2.0 * (1.0-t);
      
      pareto_set.sols.push_back(sol);
    }
  }
  
  // concave
  if (fn >= 4)
  {
    for (size_t i = 0; i < pareto_set_size; ++i)
    {
      solution_pt sol = std::make_shared<solution_t>(0, number_of_objectives);
      
      double t = 2 * (i / ((double)pareto_set_size - 1.0)); // from 0 to 2
      sol->obj[0] = t; // center0 + (center1 - center0)*(i / ((double)pareto_set_size - 1.0));
      sol->obj[1] = 4.0 * sqrt(1.0 - (t*t)/4.0);
      
      pareto_set.sols.push_back(sol);
    }
  }
  
  // uncomment if you want to write the PS/PF for external use
  // its fixed-seed, so only need to write it once.
  // std::ostringstream ss2;
  // ss2 << "../WFG" << fn << ".txt";
  // pareto_set.writeObjectivesToFile(ss2.str().c_str());
  
  return true;
  
}


