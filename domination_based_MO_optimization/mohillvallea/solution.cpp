

/*

HICAM Multi-objective

By S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "solution.h"

namespace hicam
{

  // initialize solution
  //----------------------------------------------
  solution_t::solution_t()
  {
    this->param.resize(0);
    this->obj.resize(0);
    this->constraint = 0.0;
    this->rank = -1;
    this->elite_origin = nullptr;
    this->cluster_number = -1;
    this->population_number  = -1;
    this->dvis.resize(10,0);
    this->current_batch = 0;
    this->batch_size = -1;
    this->use_lex = false;
    this->doselist_update_required = true;
  }

  solution_t::solution_t(size_t number_of_parameters)
  {
    this->param.resize(number_of_parameters, 0.0);
    this->obj.resize(0);
    this->constraint = 0.0;
    this->rank = -1;
    this->elite_origin = nullptr;
    this->cluster_number = -1;
    this->population_number  = -1;
    this->dvis.resize(10,0);
    this->current_batch = 0;
    this->batch_size = -1;
    this->use_lex = false;
    this->doselist_update_required = true;
  }

  solution_t::solution_t(size_t number_of_parameters, size_t number_of_objectives )
  {
    this->param.resize(number_of_parameters, 0.0);
    this->obj.resize(number_of_objectives, 1e308);
    this->constraint = 0.0;
    this->rank = -1;
    this->elite_origin = nullptr;
    this->cluster_number = -1;
    this->population_number  = -1;
    this->dvis.resize(10,0);
    this->current_batch = 0;
    this->batch_size = -1;
    this->use_lex = false;
    this->doselist_update_required = true;
    this->gradients.resize(number_of_objectives);
    this->adam_mt.resize(number_of_objectives);
    this->adam_vt.resize(number_of_objectives);
    
    for(size_t m = 0; m < number_of_objectives; ++m) {
      this->gradients[m].resize(number_of_parameters, 0.0);
      this->adam_mt[m].resize(number_of_parameters, 0.0);
      this->adam_vt[m].resize(number_of_parameters, 0.0);
    }
  }

  solution_t::solution_t(vec_t & param)
  {
    this->param = param;
    this->obj.resize(0);
    this->constraint = 0.0;
    this->rank = -1;
    this->elite_origin = nullptr;
    this->cluster_number = -1;
    this->population_number  = -1;
    this->dvis.resize(10,0);
    this->current_batch = 0;
    this->batch_size = -1;
    this->use_lex = false;
    this->doselist_update_required = true;
  }

  solution_t::solution_t(vec_t & param, vec_t & obj)
  {
    this->param = param;
    this->obj = obj;
    this->constraint = 0.0;
    this->rank = -1;
    this->elite_origin = nullptr;
    this->cluster_number = -1;
    this->population_number  = -1;
    this->dvis.resize(10,0);
    this->current_batch = 0;
    this->batch_size = -1;
    this->use_lex = false;
    this->doselist_update_required = true;
    
    for(size_t m = 0; m < obj.size(); ++m) {
      this->gradients[m].resize(param.size(), 0.0);
      this->adam_mt[m].resize(param.size(), 0.0);
      this->adam_vt[m].resize(param.size(), 0.0);
    }
  }

  solution_t::solution_t(const solution_t & other)
  {
    this->param = other.param;
    this->obj = other.obj;
    this->constraint = other.constraint;
    this->rank = other.rank;
    this->elite_origin = other.elite_origin;
    this->cluster_number = other.cluster_number;
    this->population_number  = other.population_number;
    this->dvis = other.dvis;
    this->current_batch = other.current_batch;
    this->batch_size = other.batch_size;
    this->lex_obj = other.lex_obj;
    this->use_lex = other.use_lex;
    this->doselists = other.doselists;
    this->doselist_update_required = other.doselist_update_required;
    this->gradients = other.gradients;
    this->dvi_gradients = other.dvi_gradients;
    this->adam_mt = other.adam_mt;
    this->adam_vt = other.adam_vt;
  }

  // delete solution
  // nothing to be done
  //----------------------------------------------
  solution_t::~solution_t() {}

  // problem dimensions
  //-----------------------------------------
  size_t solution_t::number_of_parameters() const {
    return param.size();
  }
  size_t solution_t::number_of_objectives() const {
    return obj.size();
  }

  void solution_t::set_number_of_parameters(size_t number) {
    this->param.resize(number);
  }
  void solution_t::set_number_of_objectives(size_t number) {
    this->obj.resize(number);
    
    if (use_lex) {
      this->lex_obj.resize(number);
    }
  }


  // comparison for solution_t pointers
  // returns true if sol1 is strict better than sol2
  //-----------------------------------------------
  // defined as static!

  bool solution_t::better_rank_via_pointers(const solution_pt & sol1, const solution_pt & sol2)
  {
    return (sol1->rank < sol2->rank);
  }

  bool solution_t::better_solution_via_pointers(const solution_pt & sol1, const solution_pt & sol2)
  {
    return better_solution(*sol1, *sol2);
  }

  bool solution_t::better_solution_via_pointers_obj0(const solution_pt & sol1, const solution_pt & sol2)
  {
    return better_solution_per_objective(*sol1, *sol2, 0);
  }

  bool solution_t::better_solution_via_pointers_obj0_unconstraint(const solution_pt & sol1, const solution_pt & sol2)
  {
    return sol1->better_than_unconstraint_per_objective(*sol2, 0);
  }
  
  bool solution_t::strictly_better_solution_via_pointers_obj0_unconstraint(const solution_pt & sol1, const solution_pt & sol2)
  {
    return sol1->strictly_better_than_unconstraint_per_objective(*sol2, 0);
  }
  
  
  // defined as static!
  // returns true of the first solution is strict better than the second
  bool solution_t::better_solution(const solution_t & sol1, const solution_t & sol2)
  {
    return sol1.better_than(sol2);
  }

  bool solution_t::better_than(const solution_t & other) const
  {
    
    if (this->constraint > 0) // this is infeasible
    {
      if (other.constraint > 0) // Both are infeasible
      {
        if (this->constraint < other.constraint)
          return true;
      }
    }
    else // this is feasible
    {
      if (other.constraint > 0) { // this is feasible and other is not
        return true;
      }
      else { // Both are feasible */
        return this->better_than_unconstraint(other);
      }
    }
    
    return false;
  }
  
  
  bool solution_t::better_solution_per_objective_via_pointers(const solution_pt &sol1, const solution_pt &sol2, size_t objective_number)
  {
    return better_solution_per_objective(*sol1, *sol2, objective_number);
  }
  
  bool solution_t::better_solution_per_objective(const solution_t & sol1, const solution_t & sol2, size_t objective_number)
  {
    return sol1.better_than_per_objective(sol2, objective_number);
  }
  
  // defined as static!
  // returns true of the first solution is strict better than the second
  bool solution_t::better_solution_unconstraint(const solution_t & sol1, const solution_t & sol2)
  {
    return sol1.better_than_unconstraint(sol2);
  }

  bool solution_t::better_than_unconstraint(const solution_t & sol) const
  {
    assert(sol.obj.size() == this->obj.size());

    bool strict = false;

    for (size_t i = 0; i < this->obj.size(); ++i)
    {
      // if (std::abs(this->obj[i] - other_obj[i]) >= 0.00001) // not 'equal'
      {
        if (!this->better_than_unconstraint_per_objective(sol, i))
        {
          return false;
          // break;
        }
        
        if (this->strictly_better_than_unconstraint_per_objective(sol, i)) {
          strict = true;
        }

      }
    }

    if (strict == false) {
      return false;
    }

    return true;

  }
  
  bool solution_t::strictly_better_than(const solution_t & other) const
  {

    if (this->constraint > 0) // this is infeasible 
    {
      if (other.constraint > 0) // Both are infeasible
      {
        if (this->constraint < other.constraint)
          return true;
      }
    }
    else // this is feasible
    {
      if (other.constraint > 0) { // this is feasible and other is not
        return true;
      }
      else
      { // Both are feasible */
          
          for (size_t i = 0; i < this->obj.size(); ++i)
          {
            if (this->strictly_better_than_unconstraint_per_objective(other.obj[i], i)) {
              return false;
            }
          }
          
          return true;
        
      }
    }

    return false;

  }
  
  
  bool solution_t::better_than_per_objective(const solution_t & other, size_t objective_number) const
  {
    if (this->constraint > 0) // this is infeasible
    {
      if (other.constraint > 0) // Both are infeasible
      {
        if (this->constraint < other.constraint)
          return true;
      }
    }
    else // this is feasible
    {
      if (other.constraint > 0) { // this is feasible and other is not
        return true;
      }
      else { // Both are feasible */
        return this->strictly_better_than_unconstraint_per_objective(other, objective_number);
      }
    }
    
    return false;

  }

  bool solution_t::better_than_unconstraint_per_objective(const solution_t & sol, size_t objective_number) const
  {
    if(!use_lex)
    {
      assert(this->obj.size() > objective_number);
      assert(sol.obj.size() > objective_number);
      return (this->obj[objective_number] <= sol.obj[objective_number]);
    }
    else
    { // if use_lex
      assert(this->lex_obj.size() > objective_number);
      assert(this->lex_obj[objective_number].size() == sol.lex_obj[objective_number].size());
      for(size_t i = 0; i < this->lex_obj[objective_number].size(); ++i)
      {
        // if (fabs(this->lex_obj[objective_number][i] - sol.lex_obj[objective_number][i]) <= 0) {
        if (this->lex_obj[objective_number][i] == sol.lex_obj[objective_number][i]) {
          continue;
        }
        
        return (this->lex_obj[objective_number][i] <= sol.lex_obj[objective_number][i]);
        
      }
      return true; // equality for all lex objectives..
    }
    
  }
  
  bool solution_t::strictly_better_than_unconstraint_per_objective(const solution_t & sol, size_t objective_number) const
  {
    if(!use_lex)
    {
      assert(this->obj.size() > objective_number);
      assert(sol.obj.size() > objective_number);
      return (this->obj[objective_number] < sol.obj[objective_number]);
    }
    else
    {     // use_lex
      assert(this->lex_obj.size() > objective_number);
      assert(this->lex_obj[objective_number].size() == sol.lex_obj[objective_number].size());
      for(size_t i = 0; i < this->lex_obj[objective_number].size(); ++i)
      {
        // if (fabs(this->lex_obj[objective_number][i] - sol.lex_obj[objective_number][i]) <= 0) {
        if(this->lex_obj[objective_number][i] == sol.lex_obj[objective_number][i]) {
          continue;
        }
        
        return (this->lex_obj[objective_number][i] < sol.lex_obj[objective_number][i]);
        
      }
      return false; // equality for all lex_obj..
    }
  }
  
  bool solution_t::same_objectives(const solution_t & other) const
  {
    if(this->constraint != other.constraint) {
      return false;
    }
    
    if(!use_lex)
    {
      assert(this->obj.size() == other.obj.size());
      
      for (size_t i = 0; i < number_of_objectives(); i++)
      {
        if (this->obj[i] != other.obj[i])
        {
          return false;
        }
      }

      return true;
    }
    else
    {    // if use_lex
      assert(this->lex_obj.size() == other.lex_obj.size());

      for (size_t i = 0; i < number_of_objectives(); i++)
      {
        assert(this->lex_obj[i].size() == other.lex_obj[i].size());
        
        for(size_t j = 0; j < this->lex_obj[i].size(); ++j)
        {
          if (this->lex_obj[i][j] != other.lex_obj[i][j]) {
            return false;
          }
        }
      }
      
      return true;
    } // end use_lex
  }


  // computes the distance to another solution
  //------------------------------------------------------------------------------------
  double solution_t::param_distance(const solution_t & sol2) const {
    return param_distance(sol2.param);
  }

  double solution_t::param_distance(const vec_t & param2) const
  {
    assert(this->param.size() == param2.size());
    return (this->param - param2).norm();
  }


  double solution_t::minimal_objective_distance(const std::vector<solution_pt> & sols, const vec_t & obj_ranges) const
  {
    
    double distance, distance_smallest = -1.0;
    for (size_t j = 0; j < sols.size(); j++)
    {
      if(sols[j] != nullptr)
      {
        distance = transformed_objective_distance(*sols[j], obj_ranges);
        if ((distance_smallest < 0) || (distance < distance_smallest)) {
          distance_smallest = distance;
        }
      }
    }

    if (distance_smallest == -1.0) {
      distance_smallest = 1e308;
    }

    return distance_smallest;

  }

  double solution_t::minimal_param_distance(const std::vector<solution_pt> & sols) const
  {

    double distance, distance_smallest = -1.0;
    for (size_t j = 0; j < sols.size(); j++)
    {
      if(sols[j] != nullptr)
      {
        distance = param_distance(*sols[j]);
        if ((distance_smallest < 0) || (distance < distance_smallest)) {
          distance_smallest = distance;
        }
      }
    }

    if (distance_smallest == -1.0) {
      distance_smallest = 1e308;
    }

    return distance_smallest;

  }

  double solution_t::transformed_objective_distance(const solution_t & other, const vec_t & obj_ranges) const {
    return transformed_objective_distance(other.obj, obj_ranges);
  }

  double solution_t::transformed_objective_distance(const vec_t & other_obj, const vec_t & obj_ranges) const
  {
    
    return this->obj.scaled_euclidean_distance(other_obj, obj_ranges);

  }

}







