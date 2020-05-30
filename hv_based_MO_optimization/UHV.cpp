/*

HillVallEA 

Real-valued Multi-Modal Evolutionary Optimization

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

Example script to demonstrate the usage of HillVallEA
 on the well-known 2D Six Hump Camel Back function

*/

// SO stuff
#include "HillVallEA/hillvallea.hpp"
#include "HillVallEA/fitness.h"
#include "HillVallEA/mathfunctions.hpp"
#include "UHV.hpp"
#include "HillVallEA/gomea.hpp"

// for MO problems
#include "../domination_based_MO_optimization/mohillvallea/hicam_external.h"

namespace hillvallea
{
  
  UHV_t::UHV_t
  (
   hicam::fitness_pt mo_fitness_function,
   size_t number_of_mo_solutions,
   bool collect_all_mo_sols_in_archive,
   size_t elitist_archive_size,
   hicam::elitist_archive_pt initial_archive,
   bool use_finite_differences
   )
  {
    
    this->mo_fitness_function = mo_fitness_function;
    this->number_of_mo_solutions = number_of_mo_solutions;
    this->collect_all_mo_sols_in_archive = collect_all_mo_sols_in_archive;
    this->elitist_archive_size = elitist_archive_size;
    this->elitist_archive = initial_archive;
    this->use_finite_differences = use_finite_differences;
    this->write_smoothness = false;
    
    init();
  };
  
  
  UHV_t::UHV_t
  (
   hicam::fitness_pt mo_fitness_function,
   size_t number_of_mo_solutions,
   bool collect_all_mo_sols_in_archive,
   size_t elitist_archive_size,
   hicam::elitist_archive_pt initial_archive,
   bool use_finite_differences,
   bool write_smoothness
   )
  {
    
    this->mo_fitness_function = mo_fitness_function;
    this->number_of_mo_solutions = number_of_mo_solutions;
    this->collect_all_mo_sols_in_archive = collect_all_mo_sols_in_archive;
    this->elitist_archive_size = elitist_archive_size;
    this->elitist_archive = initial_archive;
    this->use_finite_differences = use_finite_differences;
    this->write_smoothness = write_smoothness;
    
    init();
  };
  
  
  
  void UHV_t::init()
  {
    // validate MO Objective Function
    if(this->mo_fitness_function->get_number_of_objectives() != 2) {
      std::cout << "Error: not yet implemented for more than 2 objectives" << std::endl;
    }
    
    // Validate number of reference points
    if(this->number_of_mo_solutions < 1) {
      this->number_of_mo_solutions = 1;
    }
    
    this->number_of_parameters = this->number_of_mo_solutions * this->mo_fitness_function->number_of_parameters;
    maximum_number_of_evaluations = 0;// quite sure this is unused.
    
    // redefine initialization
    redefine_random_initialization = true; // this->mo_fitness_function->redefine_random_initialization;
    partial_evaluations_available = true; // this->mo_fitness_function->partial_evaluations_available;
    linkage_learning_distance_matrix_available = true; // this->mo_fitness_function->linkage_learning_custom_distance_matrix_available;
    dynamic_linkage_learning_distance_matrix_available = true;
    
    // the upper bound is always all solutions
    fos_element_size_upper_bound = this->number_of_mo_solutions * this->mo_fitness_function->fos_element_size_upper_bound;
    
    // if the mo function has partial evaluations, we set the lower bound to the one of the
    // mo function. If not, the lower bound is one mo_sol.
    if(this->mo_fitness_function->partial_evaluations_available) {
      fos_element_size_lower_bound = this->mo_fitness_function->fos_element_size_lower_bound;
    } else {
      fos_element_size_lower_bound = this->mo_fitness_function->number_of_parameters;
      // fos_element_size_lower_bound = this->mo_fitness_function->number_of_parameters;
      has_round_off_errors_in_partial_evaluations = false; // if the mo-sols have no partial evaluations, we do not have to re-evaluate the HV every once in a while
    }
    
    covariance_block_size = mo_fitness_function->number_of_parameters;
    
    // fos_element_size_upper_bound = fos_element_size_lower_bound;
    if(this->elitist_archive_size == 0) {
      this->elitist_archive_size = maximum_number_of_evaluations;
    }
    
    
    // allocate archive
    if(this->collect_all_mo_sols_in_archive)
    {
      if(this->elitist_archive == nullptr) { //banaan
        rng_pt rng = std::make_shared<rng_t>(142391);
        elitist_archive = std::make_shared<hicam::elitist_archive_t>(this->elitist_archive_size, rng);
      }
      
      hicam::vec_t r(2);
      r[0] = mo_fitness_function->hypervolume_max_f0;
      r[1] = mo_fitness_function->hypervolume_max_f1;
      elitist_archive->set_use_hypervolume_for_size_control(false, r);
      
    }
    
    number_of_mo_evaluations = 0;
    

    
    // Set param bounds
    hicam::vec_t mo_lower, mo_upper;
    mo_fitness_function->get_param_bounds(mo_lower, mo_upper);
    size_t mo_nop = mo_fitness_function->number_of_parameters;
    
    lower_param_bounds.resize(number_of_parameters, 0.0);
    upper_param_bounds.resize(number_of_parameters, 0.0);
    
    size_t ki = 0;
    for(size_t k = 0; k < number_of_mo_solutions; ++k)
    {
      for(size_t i = 0; i < mo_nop; ++i) {
        lower_param_bounds[ki] = mo_lower[i];
        upper_param_bounds[ki] = mo_upper[i];
        ki++;
      }
    }
    assert(ki == number_of_parameters);
    // end set param bounds;
    
    
    // Finite differences approximation of the gradient
    //----------------------------------------------------------------------------------
    use_central_finite_differences = false; // if false, use forward. (=cheaper)
    
    if(!mo_fitness_function->analytical_gradient_available) {
      use_finite_differences = true;
    }
    
    // end of FD settings
    
  }
    
  UHV_t::~UHV_t() {}
    
  void UHV_t::get_param_bounds(vec_t & lower, vec_t & upper) const
  {
    lower = lower_param_bounds;
    upper = upper_param_bounds;
    return;
  }
  
  // gets all nondominated solutions that are in the feasible domain, defined by (r_x,r_y)
  void UHV_t::get_front(const std::vector<hicam::solution_pt> & mo_sols, std::vector<bool> & is_part_of_front, vec_t & front_x, vec_t & front_y, double r_x, double r_y)
  {
    size_t K = mo_sols.size();
    is_part_of_front.clear();
    is_part_of_front.resize(K,true);
    size_t number_of_dominated_sols = 0;
    
    // for each solution
    for(size_t k = 0; k < K; ++k)
    {
      
      // if its not in the feasible region, skip it.
      if( mo_sols[k]->obj[0] >= r_x || mo_sols[k]->obj[1] >= r_y) {
        is_part_of_front[k] = false;
        number_of_dominated_sols++;
        continue;
      }
      
      // else, check if its dominated by other solutions, or dominates them.
      for(size_t j = 0; j < k; ++j)
      {
        if(is_part_of_front[j])
        {
          if(mo_sols[j]->better_than(*mo_sols[k])) {
            // if(mo_sols[j].obj[0] <= mo_sols[i].obj[0] && mo_sols[j].obj[1] <= mo_sols[i].obj[1]) {
            // j dominates i, so set i to dominated and stop inner for loop.
            is_part_of_front[k] = false;
            number_of_dominated_sols++;
            break;
          }
          if(mo_sols[k]->better_than(*mo_sols[j]) || mo_sols[k]->same_objectives(*mo_sols[j])) {
            // if(mo_sols[i].obj[0] <= mo_sols[j].obj[0] && mo_sols[i].obj[1] <= mo_sols[j].obj[1]) {
            // i dominated j, so set j to dominated
            is_part_of_front[j] = false;
            number_of_dominated_sols++;
          }
        }
      }
    }
    
    // construct non-dominated objective vectors
    front_x.resize(K - number_of_dominated_sols);
    front_y.resize(K - number_of_dominated_sols);
    size_t ki = 0;
    for(size_t k = 0; k < K; ++k)
    {
      if(is_part_of_front[k])
      {
        front_x[ki] = mo_sols[k]->obj[0];
        front_y[ki] = mo_sols[k]->obj[1];
        ki++;
      }
    }
  }
  
  // distance to a box defined by [-infty, ref_x, -infty, ref_y]
  double UHV_t::distance_to_box(double ref_x, double ref_y, double p_x, double p_y)
  {
    double nearest_x = 0.0;
    double nearest_y = 0.0;
    bool nearest_x_idx, nearest_y_idx;
    return distance_to_box(ref_x, ref_y, p_x, p_y, nearest_x, nearest_y, nearest_x_idx, nearest_y_idx);
  }
  
  // distance to a box defined by [-infty, ref_x, -infty, ref_y] and return nearest point on boundary
  double UHV_t::distance_to_box(double ref_x, double ref_y, double p_x, double p_y, double & nearest_x, double & nearest_y, bool & shares_x, bool & shares_y)
  {
    double dx = max(0.0, p_x - ref_x );
    double dy = max(0.0, p_y - ref_y );
    
    nearest_x = min(p_x, ref_x);
    nearest_y = min(p_y, ref_y);
    // assert( sqrt((nearest_x - p_x) * (nearest_x - p_x)  + (nearest_y - p_y) * (nearest_y - p_y)) == sqrt(dx*dx + dy*dy));
    
    // returns 1 if the nearest point has an x/y coordinate of the reference point
    shares_x = (nearest_x == ref_x);
    shares_y = (nearest_y == ref_y);
    
    return sqrt(dx*dx + dy*dy);
  }
  
  // Based on the Uncrowded Hypervolume improvement by the Inria group,
  // but we extened the definition to points that are outside of the reference frame
  // we compute the distance to the non-dominated area, within the reference window (r_x,r_y)
  // define the area points a(P^(i)_x, P^(i-1)_y), for i = 0...n (n =  number of points in front, i.e., obj0.size())
  // and let P^(-1)_y = r_y,   and    P^(n)_x = r_x
  double UHV_t::distance_to_front(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y)
  {
    vec_t nearest_point_on_front(2,0);
    size_t nearest_x_idx, nearest_y_idx;
    return distance_to_front(p_x, p_y, obj_x, obj_y, sorted_obj, r_x, r_y, nearest_point_on_front, nearest_x_idx, nearest_y_idx);
  }
  
  double UHV_t:: distance_to_front(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y, vec_t & nearest_point_on_front, size_t & nearest_x_idx, size_t & nearest_y_idx)
  {
    nearest_point_on_front.resize(2);
    // if the front is empty, use the reference point for the distance measure
    if(obj_x.size() == 0) {
      nearest_x_idx = 0; // = out of bounds of obj_x => reference point
      nearest_y_idx = 0; // = out of bounds of obj_x => reference point
      bool shares_x, shares_y;
      return distance_to_box(r_x, r_y, p_x, p_y, nearest_point_on_front[0], nearest_point_on_front[1], shares_x, shares_y);
    }
    
    size_t n = obj_x.size();
    
    // if not available, get the sorted front
    if(sorted_obj.size() != n)
    {
      sorted_obj.resize(n);
      for (size_t i = 0; i < n; ++i) {
        sorted_obj[i] = i;
      }
      
      std::sort(std::begin(sorted_obj), std::end(sorted_obj), [&obj_x](double idx, double idy) { return obj_x[(size_t)idx] < obj_x[(size_t)idy]; });
    }
    
    double dist;
    
    // distance to the 'end' boxes
    vec_t new_nearest_point_on_front(2,0.0);
    bool shares_x, shares_y;
    double min_dist = distance_to_box(obj_x[sorted_obj[0]], r_y, p_x, p_y, nearest_point_on_front[0], nearest_point_on_front[1], shares_x, shares_y);
    nearest_x_idx = shares_x ? sorted_obj[0] : obj_x.size();
    nearest_y_idx = obj_y.size();
    
    dist = distance_to_box(r_x, obj_y[sorted_obj[n-1]], p_x, p_y, new_nearest_point_on_front[0], new_nearest_point_on_front[1], shares_x, shares_y);
    
    if(dist < min_dist) {
      min_dist = dist;
      nearest_point_on_front = new_nearest_point_on_front;
      nearest_x_idx = obj_x.size();
      nearest_y_idx = shares_y ? sorted_obj[n-1] : obj_y.size();
    }
    
    {
      // distance to 'inner' boxes
      for(size_t k = 1; k < n; ++k)
      {
        dist = distance_to_box(obj_x[sorted_obj[k]], obj_y[sorted_obj[k-1]], p_x, p_y, new_nearest_point_on_front[0], new_nearest_point_on_front[1], shares_x, shares_y);
        
        if(dist < min_dist) {
          min_dist = dist;
          nearest_point_on_front = new_nearest_point_on_front;
          nearest_x_idx = shares_x ? sorted_obj[k] : obj_x.size();
          nearest_y_idx = shares_y ? sorted_obj[k-1] : obj_y.size();
        }
      }
    }
    
    assert(min_dist >= 0); // can be 0 if its at the front!
    return min_dist;
  }
  
  double UHV_t:: distance_to_front_without_corner_boxes(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y, vec_t & nearest_point_on_front, size_t & nearest_x_idx, size_t & nearest_y_idx)
  {
    nearest_point_on_front.resize(2);
    // if the front is empty, use the reference point for the distance measure
    if(obj_x.size() == 0) {
      nearest_x_idx = 0; // = obj_x.size()
      nearest_y_idx = 0; // = obj_x.size()
      bool shares_x, shares_y;
      return distance_to_box(r_x, r_y, p_x, p_y, nearest_point_on_front[0], nearest_point_on_front[1], shares_x, shares_y);
    }
    
    size_t n = obj_x.size();
    
    // if not available, get the sorted front
    if(sorted_obj.size() != n)
    {
      sorted_obj.resize(n);
      for (size_t i = 0; i < n; ++i) {
        sorted_obj[i] = i;
      }
      
      std::sort(std::begin(sorted_obj), std::end(sorted_obj), [&obj_x](double idx, double idy) { return obj_x[(size_t)idx] < obj_x[(size_t)idy]; });
    }
    
    // distance to the 'end' boxes
    vec_t new_nearest_point_on_front(2,0.0);
    bool shares_x, shares_y;
    
    double dist;
    double min_dist = 1e300;
    if(obj_x.size() == 1)
    {
      nearest_point_on_front[0] = obj_x[0];
      nearest_point_on_front[1] = obj_y[0];
      nearest_x_idx = 0;
      nearest_y_idx = 0;
      min_dist = distanceEuclidean2D(obj_x[0], obj_y[0], p_x, p_y);
    }
    else
    {
      // distance to 'inner' boxes
      for(size_t k = 1; k < n; ++k)
      {
        dist = distance_to_box(obj_x[sorted_obj[k]], obj_y[sorted_obj[k-1]], p_x, p_y, new_nearest_point_on_front[0], new_nearest_point_on_front[1], shares_x, shares_y);
        
        if(dist < min_dist) {
          min_dist = dist;
          nearest_point_on_front = new_nearest_point_on_front;
          nearest_x_idx = shares_x ? sorted_obj[k] : obj_x.size();
          nearest_y_idx = shares_y ? sorted_obj[k-1] : obj_y.size();
        }
      }
    }
    
    assert(min_dist >= 0); // can be 0 if its at the front!
    return min_dist;
  }
  
  double UHV_t::compute2DHyperVolume(const vec_t & obj0, const vec_t & obj1, std::vector<size_t> & sorted_obj, double max_0, double max_1)
  {
    size_t n = obj0.size();
    if (n == 0) { return 0.0; }
    
    if(sorted_obj.size() != n)
    {
      sorted_obj.resize(n);
      for (size_t i = 0; i < n; ++i) {
        sorted_obj[i] = i;
      }
      
      std::sort(std::begin(sorted_obj), std::end(sorted_obj), [&obj0](double idx, double idy) { return obj0[(size_t)idx] < obj0[(size_t)idy]; });
    }
    
    double area = (max_0 - min(max_0, obj0[sorted_obj[n - 1]])) * (max_1 - min(max_1, obj1[sorted_obj[n - 1]]));
    for (int i = (int) (n - 2); i >= 0; i--) {
      
      assert(obj0[sorted_obj[i+1]] > obj0[sorted_obj[i]] );
      double d = (min(max_0, obj0[sorted_obj[i + 1]]) - min(max_0, obj0[sorted_obj[i]])) * (max_1 - min(max_1, obj1[sorted_obj[i]]));
      assert(d>0);
      area += d;
    }
    
    return area;
  }
  
  void UHV_t::define_problem_evaluation(solution_t & sol){
    std::vector<size_t> dummy_touched_parameter_idx;
    solution_t dummy_old_sol;
    bool partial_evaluation = false;
    bool compute_gradients = false;
    double finite_differences_step_size = 0;
    compute_fitness(partial_evaluation, compute_gradients, sol, dummy_touched_parameter_idx, dummy_old_sol, finite_differences_step_size);
  }
  
  void UHV_t::define_partial_problem_evaluation(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol) {
    bool partial_evaluation = true;
    bool compute_gradients = false;
    double finite_differences_step_size = 0;
    compute_fitness(partial_evaluation, compute_gradients, sol, touched_parameter_idx, old_sol, finite_differences_step_size);
  }
  
  void UHV_t::define_problem_evaluation_with_gradients(solution_t & sol){
    std::vector<size_t> dummy_touched_parameter_idx;
    solution_t dummy_old_sol;
    bool partial_evaluation = false;
    bool compute_gradients = true;
    double finite_differences_step_size = 0;
    compute_fitness(partial_evaluation, compute_gradients, sol, dummy_touched_parameter_idx, dummy_old_sol, finite_differences_step_size);
  }
  
  void UHV_t::define_partial_problem_evaluation_with_gradients(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol) {
    bool partial_evaluation = true;
    bool compute_gradients = true;
    double finite_differences_step_size = 0;
    compute_fitness(partial_evaluation, compute_gradients, sol, touched_parameter_idx, old_sol, finite_differences_step_size);
  }
  
  void UHV_t::define_problem_evaluation_with_finite_differences(solution_t & sol, double finite_differences_step_size){
    std::vector<size_t> dummy_touched_parameter_idx;
    solution_t dummy_old_sol;
    bool partial_evaluation = false;
    bool compute_gradients = true;
    compute_fitness(partial_evaluation, compute_gradients, sol, dummy_touched_parameter_idx, dummy_old_sol, finite_differences_step_size);
  }
  
  void UHV_t::define_partial_problem_evaluation_with_finite_differences(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol, double finite_differences_step_size) {
    bool partial_evaluation = true;
    bool compute_gradients = true;
    compute_fitness(partial_evaluation, compute_gradients, sol, touched_parameter_idx, old_sol, finite_differences_step_size);
  }
  
  void UHV_t::compute_fitness(bool partial_evaluation, bool compute_gradients, solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol, double finite_differences_step_size)
  {
    // reset solution.
    sol.f = 0.0;
    sol.constraint = 0.0;

    //-----------------------------------------
    // reconstruct MO_params
    if(partial_evaluation) {
      update_mo_solutions(sol, number_of_mo_solutions, compute_gradients, mo_fitness_function, touched_parameter_idx, old_sol, finite_differences_step_size);
    }
    else {
      set_mo_solutions(sol, number_of_mo_solutions, compute_gradients, mo_fitness_function, finite_differences_step_size);
    }
    
    std::vector<bool> is_part_of_front;
    vec_t front_x;
    vec_t front_y;
    double r0 = mo_fitness_function->hypervolume_max_f0;
    double r1 = mo_fitness_function->hypervolume_max_f1;
    get_front(sol.mo_test_sols, is_part_of_front, front_x, front_y, r0, r1);

    std::vector<size_t> original_idx(front_x.size());
    size_t idx = 0;
    for(size_t i = 0; i < sol.mo_test_sols.size(); ++i)
    {
      if(is_part_of_front[i]) {
        original_idx[idx] = i;
        idx++;
      }
    }
    
    //-----------------------------------------------
    // Set objective (Hyper Volume)
    std::vector<size_t> sorted_front_order;
    double HV = compute2DHyperVolume(front_x, front_y, sorted_front_order, r0, r1);
    sol.f = - HV;
    
    double penalty = 0.0;
    double penalty_factor = 1.0 /((double) (sol.mo_test_sols.size()));
    std::vector<vec_t> nearest_point_on_front(sol.mo_test_sols.size());
    std::vector<size_t> nearest_x_idx(sol.mo_test_sols.size());
    std::vector<size_t> nearest_y_idx(sol.mo_test_sols.size());
    
    for(size_t k = 0; k < number_of_mo_solutions; ++k)
    {
      
      if(!is_part_of_front[k])
      {
        double dist;
        if(compute_gradients) {
          dist = distance_to_front_without_corner_boxes(sol.mo_test_sols[k]->obj[0], sol.mo_test_sols[k]->obj[1], front_x, front_y, sorted_front_order, r0, r1, nearest_point_on_front[k], nearest_x_idx[k], nearest_y_idx[k]);
        } else {
          dist = distance_to_front(sol.mo_test_sols[k]->obj[0], sol.mo_test_sols[k]->obj[1], front_x, front_y, sorted_front_order, r0, r1, nearest_point_on_front[k], nearest_x_idx[k], nearest_y_idx[k]);
        }
        assert(dist >= 0);
        penalty += penalty_factor * pow(dist,mo_fitness_function->number_of_objectives);
      }
    }

    // the UHV is now complete.
    sol.f += penalty;
    bool include_ud_in_grad = false;
    if(compute_gradients)
    {
   
      vec_t UHV_grad_f0(number_of_mo_solutions,0);
      vec_t UHV_grad_f1(number_of_mo_solutions,0);
      
      // gradient of non-dominated solutions
      for(size_t i = 0; i < front_x.size(); ++i)
      {
        // doube negative => positive cuz we minimize -HV!
        if(i == 0) {
          UHV_grad_f0[original_idx[sorted_front_order[i]]] = ( r1 - front_y[sorted_front_order[i]] );
        } else {
          UHV_grad_f0[original_idx[sorted_front_order[i]]] = ( front_y[sorted_front_order[i-1]] - front_y[sorted_front_order[i]] );
        }
        
        if(i == front_x.size()-1) {
          UHV_grad_f1[original_idx[sorted_front_order[i]]] = ( r0 - front_x[sorted_front_order[i]] );
        } else {
          UHV_grad_f1[original_idx[sorted_front_order[i]]] = ( front_x[sorted_front_order[i+1]] - front_x[sorted_front_order[i]] );
        }
        
        
        if(include_ud_in_grad)
        {
          double f0_shift = 0;
          double f1_shift = 0;
          // check for dominated points that have an UD that depends on
          // the current solution
          for(size_t k = 0; k < nearest_y_idx.size(); ++k)
          {
            if(!is_part_of_front[k])
            {
              if(nearest_x_idx[k] == sorted_front_order[i]) {
                // std::cout << "Shared corner point (" << front_x[sorted_front_order[i]] << " , " << nearest_point_on_front[k][0] << ") \n";
                 f0_shift += penalty_factor * 2.0 * (nearest_point_on_front[k][0] - sol.mo_test_sols[k]->obj[0]);
              }
              
              if(nearest_y_idx[k] == sorted_front_order[i]) {
                f1_shift += penalty_factor * 2.0 * (nearest_point_on_front[k][1] - sol.mo_test_sols[k]->obj[1]);
              }
            }
          }
          
          if(f1_shift != 0 || f0_shift != 0) {
            // std::cout << "(" << UHV_grad_f0[original_idx[sorted_front_order[i]]] << ", " << UHV_grad_f1[original_idx[sorted_front_order[i]]] << ")";
            UHV_grad_f0[original_idx[sorted_front_order[i]]] += f0_shift;
            UHV_grad_f1[original_idx[sorted_front_order[i]]] += f1_shift;
            
            // std::cout << " --> (" << UHV_grad_f0[original_idx[sorted_front_order[i]]] << ", " << UHV_grad_f1[original_idx[sorted_front_order[i]]] << ")\n";
          }
        }
      }
      
      //double overshoot = 0; // >= 0
      for(size_t k = 0; k < number_of_mo_solutions; ++k)
      {
        if(!is_part_of_front[k])
        {
          UHV_grad_f0[k] = penalty_factor * 2.0 * (sol.mo_test_sols[k]->obj[0] - nearest_point_on_front[k][0]);
          UHV_grad_f1[k] = penalty_factor * 2.0 * (sol.mo_test_sols[k]->obj[1] - nearest_point_on_front[k][1]);
          
          //if(UHV_grad_f0[k] != 0) { UHV_grad_f0[k] += penalty_factor * 2.0 * overshoot; }
          //if(UHV_grad_f1[k] != 0) { UHV_grad_f1[k] += penalty_factor * 2.0 * overshoot; }
        }
        
        // normalize the gradients
        double grad_length = sqrt(UHV_grad_f0[k]*UHV_grad_f0[k]  + UHV_grad_f1[k]*UHV_grad_f1[k]);
        // double grad_length = UHV_grad_f0[k]  + UHV_grad_f1[k];
        
        if(grad_length != 0) {
          UHV_grad_f0[k] /= grad_length;
          UHV_grad_f1[k] /= grad_length;
        }
        // std::cout << std::scientific << std::setprecision(3) << " " << sol.mo_test_sols[k]->obj[0] << " " <<sol.mo_test_sols[k]->obj[1] << " " << UHV_grad_f0[k] << " " << UHV_grad_f1[k] << " " << sol.mo_test_sols[k]->param << sol.mo_test_sols[k]->gradients[0] << " " << sol.mo_test_sols[k]->gradients[1];
      }
    
      // Now find gradients for x and y.
      sol.gradient.resize(number_of_parameters);
      sol.gradient.fill(0.0);
      size_t mo_sol_idx;
      size_t so_param_idx;
      for(size_t i = 0; i < number_of_parameters; ++i)
      {
        mo_sol_idx = i / mo_fitness_function->number_of_parameters;
        so_param_idx = i % mo_fitness_function->number_of_parameters;
        sol.gradient[i] = UHV_grad_f0[mo_sol_idx] * sol.mo_test_sols[mo_sol_idx]->gradients[0][so_param_idx] + UHV_grad_f1[mo_sol_idx] * sol.mo_test_sols[mo_sol_idx]->gradients[1][so_param_idx];
      }
      
      // std::cout << sol.gradient << std::endl;
      
    }
  }

  
  void UHV_t::set_mo_solutions(solution_t & sol, size_t number_of_mo_solutions, bool compute_gradients, hicam::fitness_pt mo_fitness_function, double finite_differences_step_size)
  {
    size_t mo_number_of_parameters = mo_fitness_function->number_of_parameters;
    assert(sol.param.size() % mo_number_of_parameters == 0);
    assert(sol.param.size() / mo_number_of_parameters == number_of_mo_solutions);
    
    // clear and reset if required
    if(sol.mo_test_sols.size() != number_of_mo_solutions) {
      sol.mo_test_sols.clear();
      sol.mo_test_sols.resize(number_of_mo_solutions);
    }
    
    for(size_t k = 0; k < number_of_mo_solutions; ++k)
    {
      // allocate mo_sols
      if(sol.mo_test_sols[k] == nullptr) {
        sol.mo_test_sols[k] = std::make_shared<hicam::solution_t>(mo_number_of_parameters);
      }
      
      // set parameters
      for(size_t i = 0; i < mo_number_of_parameters; ++i) {
        sol.mo_test_sols[k]->param[i] = sol.param[i + k * mo_number_of_parameters];
      }
    }
    
    for(size_t k = 0; k < number_of_mo_solutions; ++k)
    {
      if(compute_gradients) {
        if(!use_finite_differences) {
          mo_fitness_function->evaluate_with_gradients(sol.mo_test_sols[k]);
          number_of_mo_evaluations++;
        } else {
          number_of_mo_evaluations += evaluateMOSolutionWithfiniteDifferences(sol.mo_test_sols[k], finite_differences_step_size, use_central_finite_differences);
        }
      } else {
        mo_fitness_function->evaluate(sol.mo_test_sols[k]);
        number_of_mo_evaluations++;
      }
      
      if(collect_all_mo_sols_in_archive) {
        elitist_archive->updateArchive(sol.mo_test_sols[k],true);
      }
      
    }
  }
  
  void UHV_t::update_mo_solutions(solution_t & sol, size_t number_of_mo_solutions, bool compute_gradients, hicam::fitness_pt mo_fitness_function, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol, double finite_differences_step_size)
  {
    //-----------------------------------------------------
    // reconstruct MO touched parameters for reference points
    size_t mo_number_of_parameters = mo_fitness_function->number_of_parameters;
    assert(sol.param.size() % mo_number_of_parameters == 0);
    assert(sol.param.size() / mo_number_of_parameters == number_of_mo_solutions);
    
    std::vector<std::vector<size_t>> mo_touched_parameter_idx;
    get_mo_touched_parameter_idx(mo_touched_parameter_idx, touched_parameter_idx, mo_number_of_parameters);
    
    // if the solution is new, do a full update
    if(sol.mo_test_sols.size() != number_of_mo_solutions) {
      set_mo_solutions(sol, number_of_mo_solutions, compute_gradients, mo_fitness_function, finite_differences_step_size);
      return;
    }
    
    // copy updated parameters to MO solutions
    for(size_t k = 0; k < number_of_mo_solutions; ++k)
    {
      size_t var;
      for(size_t i = 0; i < mo_touched_parameter_idx[k].size(); ++i)
      {
        var = mo_touched_parameter_idx[k][i];
        sol.mo_test_sols[k]->param[var] = sol.param[var + k * mo_number_of_parameters];
      }
    }
    
    // evaluate MO solutions
    for(size_t k = 0; k < number_of_mo_solutions; ++k)
    {
      // but only if it has changed parameters
      if(mo_touched_parameter_idx[k].size() > 0)
      {
        // We can do the MO-feval either partially/full and with/without gradients
        if(mo_fitness_function->partial_evaluations_available)
        {
          if(compute_gradients) {
            if(!use_finite_differences)
            {
              mo_fitness_function->partial_evaluate_with_gradients(sol.mo_test_sols[k], mo_touched_parameter_idx[k], old_sol.mo_test_sols[k]);
              number_of_mo_evaluations += mo_touched_parameter_idx[k].size() / (double) sol.mo_test_sols[k]->number_of_parameters();
            } else {
              number_of_mo_evaluations += partialEvaluateMOSolutionWithfiniteDifferences(sol.mo_test_sols[k], finite_differences_step_size, use_central_finite_differences, mo_touched_parameter_idx[k], old_sol.mo_test_sols[k]);
            }
          } else {
            mo_fitness_function->partial_evaluate(sol.mo_test_sols[k], mo_touched_parameter_idx[k], old_sol.mo_test_sols[k]);
            number_of_mo_evaluations += mo_touched_parameter_idx[k].size() / (double) sol.mo_test_sols[k]->number_of_parameters();
          }
        } else {
          if(compute_gradients)
          {
            if(!use_finite_differences) {
              mo_fitness_function->evaluate_with_gradients(sol.mo_test_sols[k]);
              number_of_mo_evaluations++;
            } else {
              number_of_mo_evaluations += evaluateMOSolutionWithfiniteDifferences(sol.mo_test_sols[k], finite_differences_step_size, use_central_finite_differences);
            }
          }
          else
          {
            mo_fitness_function->evaluate(sol.mo_test_sols[k]);
            number_of_mo_evaluations++;
          }
        }
        
        // if we want to collect all solutions in the archive, this is a right moment to do so.
        if(collect_all_mo_sols_in_archive) {
          elitist_archive->updateArchive(sol.mo_test_sols[k],true);
        }
        
        continue;
      }
    }
    
  }
  
  void UHV_t::get_mo_touched_parameter_idx(std::vector<std::vector<size_t>> & mo_touched_parameter_idx, const std::vector<size_t> & touched_parameter_idx, size_t mo_number_of_parameters)
  {
    //-----------------------------------------
    // reconstruct MO_params
    std::vector<bool> is_mo_parameter_touched(mo_number_of_parameters, false);
    
    mo_touched_parameter_idx.clear();
    mo_touched_parameter_idx.resize(number_of_mo_solutions);
    
    for(size_t k = 0; k < number_of_mo_solutions; ++k) {
      mo_touched_parameter_idx[k].reserve(mo_number_of_parameters);
    };
    
    for(size_t i = 0; i < touched_parameter_idx.size(); ++i) {
      mo_touched_parameter_idx[touched_parameter_idx[i] / mo_fitness_function->number_of_parameters].push_back(touched_parameter_idx[i] % mo_fitness_function->number_of_parameters);
    }
  }
  
  void UHV_t::init_solutions_randomly(population_t & population, size_t sample_size, const vec_t & lower_init_ranges, const vec_t & upper_init_ranges, size_t number_of_elites, rng_pt rng)
  {
    // initialize solutions
    population.sols.resize(sample_size);
    for(size_t i = number_of_elites; i < population.sols.size(); ++i)
    {
      if (population.sols[i] == nullptr) {
        solution_pt sol = std::make_shared<solution_t>(number_of_parameters);
        population.sols[i] = sol;
      }
      population.sols[i]->mo_test_sols.clear();
      population.sols[i]->mo_test_sols.resize(number_of_mo_solutions, nullptr);
    }
    
    if(elitist_archive != nullptr) {
      elitist_archive->removeSolutionNullptrs();
    }
    
    // if there is an elitist archive, we initialize from it.
    if(elitist_archive != nullptr && elitist_archive->sols.size() != 0)
    {
      
      // allocate clusters for each reference sol
      size_t N = elitist_archive->size(); // number of solutions
      // size_t m = mo_fitness_function->number_of_objectives;
      size_t n = mo_fitness_function->number_of_parameters;
      
      // initialize means from leaders
      size_t number_of_means = std::min(number_of_mo_solutions, elitist_archive->sols.size());
      std::vector<hicam::solution_pt> leaders, nonselected_sols;
      hicam::vec_t objective_ranges;
      elitist_archive->objectiveRanges(objective_ranges);
      elitist_archive->gHSS(number_of_means, mo_fitness_function->hypervolume_max_f0, mo_fitness_function->hypervolume_max_f1, leaders, nonselected_sols);
      
      // Do leader-based distance assignment
      std::vector<hicam::population_pt> clusters(leaders.size());
      vec_t distances_to_cluster(N);
      std::vector<size_t> distance_ranks(N);
      
      for (size_t i = 0; i < leaders.size(); i++)
      {
        clusters[i] = (std::make_shared<hicam::population_t>());
        
        for (size_t j = 0; j < N; j++) {
          distances_to_cluster[j] = elitist_archive->sols[j]->param_distance(*leaders[i]);
        }
        
        compute_ranks_asc(distances_to_cluster, distance_ranks);
        
        for (size_t j = 0; j < std::min(sample_size-number_of_elites, N/number_of_mo_solutions*2); ++j) {
          clusters[i]->addSolution(elitist_archive->sols[distance_ranks[j]]);
        }
      }
      
      for(size_t k = 0; k < clusters.size(); ++k)
      {
        for(size_t i = 0; i < clusters[k]->sols.size(); ++i)
        {
          size_t ki = k * n;
          
          for(size_t j = 0; j < n; ++j) {
            population.sols[number_of_elites + i]->param[ki] = clusters[k]->sols[i]->param[j];
            ki++;
          }
          
          assert(population.sols[number_of_elites + i]->mo_test_sols[k] == nullptr);
          population.sols[number_of_elites + i]->mo_test_sols[k] = std::make_shared<hicam::solution_t>(*clusters[k]->sols[i]); // make a copy to be sure.
        }
      }
      
    } // end of elite-based initialization
    
    hicam::population_pt mo_population = std::make_shared<hicam::population_t>();
    // size_t mo_sample_size = number_of_mo_solutions * (sample_size - number_of_elites);
    size_t mo_number_of_parameters = mo_fitness_function->number_of_parameters;
    hicam::vec_t mo_lower_init_ranges(mo_number_of_parameters);
    hicam::vec_t mo_upper_init_ranges(mo_number_of_parameters);
    
    // ugly implementation
    for(size_t i = 0; i < mo_number_of_parameters; ++i) {
      mo_lower_init_ranges[i] = lower_init_ranges[i];
      mo_upper_init_ranges[i] = upper_init_ranges[i];
    }
    
    // sample solutions
    for(size_t i = number_of_elites; i < population.sols.size(); ++i)
    {
    
      size_t mo_sample_size = number_of_mo_solutions;
      if(mo_fitness_function->redefine_random_initialization) {
        mo_fitness_function->init_solutions_randomly(mo_population, mo_sample_size, mo_lower_init_ranges, mo_upper_init_ranges, 0, rng);
      } else {
        mo_population->fill_uniform(mo_sample_size, mo_number_of_parameters, mo_lower_init_ranges, mo_upper_init_ranges, 0, rng);
      }
    
      size_t ki = 0;
      for(size_t k = 0; k < mo_population->size(); ++k)
      {
        for(size_t j = 0; j < mo_fitness_function->number_of_parameters; ++j)
        {
          // skips the parameters that are already set from the elitist archive
          if(population.sols[i]->mo_test_sols[k] == nullptr) {
            population.sols[i]->param[ki] = mo_population->sols[k]->param[j];
          }
          ki++;
        }
      }
      assert(ki == population.sols[i]->param.size());
    }
  }
    
  // the distance matrix determines which solutions get merged in
  // the static linkage tree setup, which is in our case simply
  // the
  void UHV_t::linkage_learning_distance_matrix(matrix_t & M)
  {
    
    rng_t rng(149322);
    std::uniform_real_distribution<double> unif(0, 1);
    
    // get MO distance matrix, or create a random one
    int mo_nop = (int) mo_fitness_function->number_of_parameters;
    hicam::matrix_t mo_M;
    
    if(mo_fitness_function->linkage_learning_distance_matrix_available)
    {
      // mo matrix
      mo_fitness_function->linkage_learning_distance_matrix(mo_M);
    }
    else
    {
      // random matrix
      mo_M.resize(mo_nop, mo_nop);
      for(int i = 0; i < mo_nop; ++i)
      {
        mo_M[i][i] = 0;
        for(int j = 0; j < i; ++j) {
          mo_M[i][j] = unif(rng);
          mo_M[j][i] = mo_M[i][j];
        }
      }
    }
    
    // get min/max entries
    double min_entry = 1e308;
    double max_entry = -1e308;
    for(int i = 0; i < mo_nop; ++i)
    {
      for(int j = i+1; j < mo_nop; ++j)
      {
        if(mo_M[i][j] < min_entry) {
          min_entry = mo_M[i][j];
        }
        if(mo_M[i][j] > max_entry) {
          max_entry = mo_M[i][j];
        }
      }
    }
    
    // SO distance matrix
    M.resize(number_of_parameters, number_of_parameters);

    // merge internal solutions first and only later merge different solutions.
    // so we set all off-block diagonal entries 'large'
    for(int i = 0; i < (int) number_of_parameters; ++i)
    {
      for(int j = 0; j < (int) number_of_parameters; ++j) {
        if((i/mo_nop) != (j/mo_nop)) {
          M[i][j] = fabs((i/mo_nop)-(j/mo_nop)) * ceil(max_entry) * 10;
        }
        else
        {
          M[i][j] = mo_M[i%mo_nop][j%mo_nop];
        }
      }
    }
  }
  
  // the distance matrix determines which solutions get merged in
  // the static linkage tree setup, which is in our case simply
  // the
  void UHV_t::dynamic_linkage_learning_distance_matrix(matrix_t & M, const population_t & pop)
  {
    
    if(pop.size() == 0 || pop.sols[0]->mo_test_sols.size() == 0) {
      linkage_learning_distance_matrix(M);
      return;
    }
    
    size_t N = pop.sols.size();
    size_t K = pop.sols[0]->mo_test_sols.size();
    
    rng_t rng(149322);
    std::uniform_real_distribution<double> unif(0, 1);
    
    // get MO distance matrix, or create a random one
    int mo_nop = (int) mo_fitness_function->number_of_parameters;
    hicam::matrix_t mo_M;
    
    if(mo_fitness_function->linkage_learning_distance_matrix_available)
    {
      // mo matrix
      mo_fitness_function->linkage_learning_distance_matrix(mo_M);
    }
    else
    {
      // random matrix
      mo_M.resize(mo_nop, mo_nop);
      for(int i = 0; i < mo_nop; ++i)
      {
        mo_M[i][i] = 0;
        
        for(int j = 0; j < i; ++j) {
          mo_M[i][j] = unif(rng);
          mo_M[j][i] = mo_M[i][j];
        }
      }
    }
    
    // get min/max entries
    double min_entry = 1e308;
    double max_entry = -1e308;
    for(int i = 0; i < mo_nop; ++i)
    {
      for(int j = i+1; j < mo_nop; ++j)
      {
        if(mo_M[i][j] < min_entry) {
          min_entry = mo_M[i][j];
        }
        if(mo_M[i][j] > max_entry) {
          max_entry = mo_M[i][j];
        }
      }
    }
    
    //------------------------------------
    // compute all means
    
    std::vector<hicam::solution_pt> means(K);
    
    for(size_t k = 0; k < K; ++k) {
      means[k] = std::make_shared<hicam::solution_t>();
      means[k]->obj.reset(mo_fitness_function->number_of_objectives,0.0);
    }
    
    for(size_t i = 0; i < N; ++i) {
      for(size_t k = 0; k < K; ++k) {
        means[k]->obj += pop.sols[i]->mo_test_sols[k]->obj / ((double) N);
      }
    }
    
    
    // SO distance matrix
    M.resize(number_of_parameters, number_of_parameters);
    hicam::vec_t obj_ranges(mo_fitness_function->number_of_objectives, 1.0);
    
    matrix_t mean_dist(means.size(), means.size());
    for(size_t i = 0; i < means.size(); ++i) {
      for(size_t j = i+1; j < means.size(); ++j) {
        mean_dist[i][j] = means[i]->transformed_objective_distance(means[j]->obj,obj_ranges);
        mean_dist[j][i] = mean_dist[i][j];
      }
    }
    
    // merge internal solutions first and only later merge different solutions.
    // so we set all off-block diagonal entries 'large'
    for(int i = 0; i < (int) number_of_parameters; ++i)
    {
      for(int j = 0; j < (int) number_of_parameters; ++j) {
        if((i/mo_nop) != (j/mo_nop)) {
          M[i][j] = mean_dist[(size_t) (i/mo_nop)][(size_t) (j/mo_nop)] + ceil(max_entry) * 10;
        }
        else
        {
          M[i][j] = mo_M[i%mo_nop][j%mo_nop];
        }
      }
    }
  }
  
  
  std::string UHV_t::write_solution_info_header(bool niching_enabled)
  {
    std::ostringstream ss;
    ss << "                Best-HV               Best-IGD                Best-GD";
    
    if(niching_enabled) {
      ss << "              Best-IGDX   Best-MR";
    }
    
    if(write_smoothness) {
      ss << "        Best-Smoothness";
    }
    ss << " size";
    
    ss << "             Archive-HV            Archive-IGD             Archive-GD";
    
    if(niching_enabled) {
      ss << "           Archive-IGDX Archive-MR";
    }
    
    if(write_smoothness) {
      ss << "     Archive-Smoothness";
    }
    
    ss << " size   MO-evals";
    
    return ss.str();
  }
  std::string UHV_t::write_additional_solution_info(const solution_t & best,  const std::vector<solution_pt> & niching_archive, bool niching_enabled)
  {
    
    //------------------------------------------
    // create approximation set
    hicam::rng_pt rng = std::make_shared<hicam::rng_t>(1000);
    hicam::elitist_archive_t approximation_set(1000, rng);
    
    if(!niching_enabled)
    {
      // collect all mo_solutions from all optima
      for(size_t i = 0; i < best.mo_test_sols.size(); ++i) {
        approximation_set.updateArchive(best.mo_test_sols[i]);
      }
    }
    else
    {
      for(size_t i = 0; i < niching_archive.size(); ++i) {
        for(size_t j = 0; j < niching_archive[i]->mo_test_sols.size(); ++j) {
          approximation_set.updateArchive(niching_archive[i]->mo_test_sols[j]);
        }
      }
    }
    approximation_set.removeSolutionNullptrs();
    std::sort(approximation_set.sols.begin(), approximation_set.sols.end(), hicam::solution_t::strictly_better_solution_via_pointers_obj0_unconstraint);
    
    // Allocate default values for parameters
    
    double HV = 0;
    double IGD = 0;
    double GD = 0;
    double IGDX = 0; // if niching
    double MR = 0; // if niching
    double smoothness = 0; // if smoothness
    double size = 0;
    
    double archive_HV = 0;
    double archive_IGD = 0;
    double archive_GD = 0;
    double archive_IGDX = 0; // if niching
    double archive_MR = 0; // if niching
    double archive_smoothness = 0; // if smoothness
    double archive_size = 0;
    
    // IGD
    if (mo_fitness_function->igd_available) {
      IGD = approximation_set.computeIGD(mo_fitness_function->pareto_set);
    }
    
    if(niching_enabled)
    {
      // IGDX
      if (mo_fitness_function->igdx_available) {
        IGDX = approximation_set.computeIGDX(mo_fitness_function->pareto_set);
      }
      
      // MR (= SR)
      if (mo_fitness_function->sr_available)
      {
        hicam::vec_t ones(mo_fitness_function->pareto_sets_max_igdx.size(), 1.0);
        double threshold = (mo_fitness_function->number_of_objectives == 2) ? 5e-2 : 1e-1;
        threshold = 0.15; //TEMP
        MR = approximation_set.computeSR(mo_fitness_function->pareto_sets, threshold, ones);
      }
    }
    
    // HV (2D Hypervolume)
    if(mo_fitness_function->number_of_objectives == 2) {
      HV = approximation_set.compute2DHyperVolume(mo_fitness_function->hypervolume_max_f0, mo_fitness_function->hypervolume_max_f1);
    }
    
    // GD
    if(mo_fitness_function->igd_available) {
      if(mo_fitness_function->analytical_gd_avialable) {
        GD = approximation_set.computeAnalyticGD(*mo_fitness_function);
      } else {
        GD = approximation_set.computeGD(mo_fitness_function->pareto_set);
      }
    }
    
    // Smoothness
    if(write_smoothness) {
      smoothness = approximation_set.computeSmoothness();
    }
    
    // Size
    size = approximation_set.actualSize();
    
    // Elitist Archive (=large_approximation_set)
    //-------------------------------------------
    if(collect_all_mo_sols_in_archive && elitist_archive != nullptr)
    {
      
      hicam::elitist_archive_t large_approximation_set = hicam::elitist_archive_t(elitist_archive_size, rng);
      large_approximation_set.computeApproximationSet(elitist_archive_size, elitist_archive, false);
      
      
      // Compute IGD
      if (mo_fitness_function->igd_available) {
        archive_IGD = large_approximation_set.computeIGD(mo_fitness_function->pareto_set);
      }
      
      if(niching_enabled)
      {
        // IGDX
        if (mo_fitness_function->igdx_available) {
          archive_IGDX = large_approximation_set.computeIGDX(mo_fitness_function->pareto_set);
        }
        
        // MR (= SR)
        if (mo_fitness_function->sr_available)
        {
          hicam::vec_t ones(mo_fitness_function->pareto_sets_max_igdx.size(), 1.0);
          double threshold = (mo_fitness_function->number_of_objectives == 2) ? 5e-2 : 1e-1;
          archive_MR = large_approximation_set.computeSR(mo_fitness_function->pareto_sets, threshold, ones);
        }
      }
      
      // HV
      if(mo_fitness_function->number_of_objectives == 2) {
        archive_HV = large_approximation_set.compute2DHyperVolume(mo_fitness_function->hypervolume_max_f0, mo_fitness_function->hypervolume_max_f1);
      }
      
      // GD
      if (mo_fitness_function->igd_available) {
        
        if(mo_fitness_function->analytical_gd_avialable) {
          archive_GD = large_approximation_set.computeAnalyticGD(*mo_fitness_function);
        } else {
          archive_GD = large_approximation_set.computeGD(mo_fitness_function->pareto_set);
        }
      }
      
      // Smoothness
      if(write_smoothness) {
        archive_smoothness = large_approximation_set.computeSmoothness();
      }
      
      // Size
      archive_size = large_approximation_set.actualSize();
    }
    
    // Start writing stuff
    //-----------------------------------------------------------------------------------------
    
    std::ostringstream ss;
    
    ss
    << " " << std::setw(14) << std::scientific << std::setprecision(16) << HV
    << " " << std::setw(14) << std::scientific << std::setprecision(16) << IGD
    << " " << std::setw(14) << std::scientific << std::setprecision(16) << GD;
    
    if(niching_enabled) {
      ss
      << " " << std::setw(14) << std::scientific << std::setprecision(16) << IGDX
      << " " << std::setw(9) << std::scientific << std::setprecision(3) << MR;
    }
    
    if(write_smoothness) {
      ss << " " << std::setw(14) << std::scientific << std::setprecision(16) << smoothness;
    }
    
    ss
    << " " << std::setw(4) << std::fixed << (int) size
    << " " << std::setw(24) << std::scientific << std::setprecision(16) << archive_HV
    << " " << std::setw(24) << std::scientific << std::setprecision(16) << archive_IGD
    << " " << std::setw(24) << std::scientific << std::setprecision(16) << archive_GD;
    
    if(niching_enabled) {
      ss
      << " " << std::setw(24) << std::scientific << std::setprecision(16) << archive_IGDX
      << " " << std::setw(14) << std::scientific << std::setprecision(3) << archive_MR;
    }
    
    if(write_smoothness) {
      ss << " " << std::setw(14) << std::scientific << std::setprecision(16) << archive_smoothness;
    }
    ss
    << " " << std::setw(4) << std::fixed << (int) archive_size
    << " " << std::setw(4) << std::fixed << (int) number_of_mo_evaluations;
    // << " " << std::setw(4) << std::fixed << (int) mo_fitness_function->number_of_evaluations // alternative way to count MO-evals
    // << " " << std::setw(4) << std::fixed << (int) number_of_evaluations; // alternative way to count SO-evals
    return ss.str();
  }

  std::string UHV_t::name() const
  {
    std::ostringstream ss;
    ss << "Hypervolume_search_linear_" << mo_fitness_function->name();
    return ss.str();
  }
  
  void UHV_t::write_solution(const solution_t & sol, const std::string & filename)
  {
    std::ofstream file;
    file.open(filename, std::ofstream::out | std::ofstream::trunc);
  
    for(size_t k = 0; k < sol.mo_test_sols.size(); ++k) {
      file << sol.mo_test_sols[k]->obj << " " << sol.mo_test_sols[k]->dvis << " " << sol.mo_test_sols[k]->param << std::endl;
    }
    
    file.close();
  }
  
  void UHV_t::sort_population_parameters(population_t & pop, FOS_t & FOS)
  {
    // this needs to be done once in a while, so this is a good moment
    // but it could be elsewhere as well.
    if(collect_all_mo_sols_in_archive && elitist_archive != nullptr) {
      elitist_archive->adaptArchiveSize();
    }
    
    if(pop.size() == 0) {
      return;
    }
    
    if(pop.sols[0]->mo_test_sols.size() == 0) {
      return;
    }
    
    // allocate clusters for each reference sol
    size_t N = pop.size(); // number of solutions
    size_t K = pop.sols[0]->mo_test_sols.size(); // number of reference sols
    size_t m = pop.sols[0]->mo_test_sols[0]->obj.size(); // number of objectives
    
    
    // assign each solution to the nearest mean
    std::vector<std::vector<size_t>> cluster_index(N);
    hicam::vec_t obj_ranges(m,1.0);
    
    for(size_t i = 0; i < N; ++i)
    {
      cluster_index[i].resize(K);
      for(size_t k = 0; k < K; ++k) {
        cluster_index[i][k] = k;
      }
    }
  
    // do a k-means style mean-update
    // we only perform 1 generation to minimize the difference between
    // generations, or it might screw up the hyperparameters of use used optimization method.
    for(size_t itt = 0; itt < 1; ++itt)
    {
      
      //------------------------------------
      // compute all means
      std::vector<hicam::solution_pt> means(K);
      
      for(size_t k = 0; k < K; ++k) {
        means[k] = std::make_shared<hicam::solution_t>();
        means[k]->obj.reset(m,0.0);
      }
      
      for(size_t i = 0; i < N; ++i) {
        for(size_t k = 0; k < K; ++k) {
          means[cluster_index[i][k]]->obj += pop.sols[i]->mo_test_sols[k]->obj / ((double) N);
        }
      }
      
      bool nothing_changed = true;
      for(size_t i = 0; i < N; ++i)
      {
        // distance from solution r to mean k;
        matrix_t dist(K,K);
        for(size_t k = 0; k < K; ++k) {
          for(size_t r = 0; r < K; ++r) {
            dist[r][k] = pop.sols[i]->mo_test_sols[r]->transformed_objective_distance(means[k]->obj, obj_ranges);
          }
        }
        
        // find min_element in dist[r][k]
        double min_dist = 1e300;
        size_t min_r = 0; // reference sol r
        size_t min_k = 0; // mean k
        std::vector<bool> done_r(K,false);
        std::vector<bool> done_k(K,false);
        
        // we find K times the minimum
        for(size_t l = 0; l < K; ++l)
        {
          // find min in distance matrix
          min_dist = 1e300;
          for(size_t r = 0; r < K; ++r)
          {
            if(done_r[r]) { continue; }
            
            for(size_t k = 0; k < K; ++k)
            {
              
              if(done_k[k]) { continue; }
              
              if(dist[r][k] < min_dist) {
                min_dist = dist[r][k];
                min_r = r;
                min_k = k;
              }
            }
          }
          
          assert(!done_k[min_k]);
          assert(!done_r[min_r]);
          done_k[min_k] = true;
          done_r[min_r] = true;
        
          cluster_index[i][min_r] = min_k;
          
          if(nothing_changed && min_r != min_k) {
            nothing_changed = false;
          }
        }
        
        // re-arrange clusters based on cluster_indices
        std::vector<hicam::solution_pt> backup_reference_sols = pop.sols[i]->mo_test_sols;
        pop.sols[i]->mo_test_sols.clear(); // this is a sanity check.
        pop.sols[i]->mo_test_sols.resize(K);
        
        // re-arrange reference sols
        for(size_t d = 0; d < backup_reference_sols.size(); ++d ) {
          assert(backup_reference_sols[d] != nullptr);
          pop.sols[i]->mo_test_sols[cluster_index[i][d]] = backup_reference_sols[d];
          // std::cout << d << "->" << cluster_index[i][d] << std::endl;
          backup_reference_sols[d] = nullptr;
        }
        
        size_t ki = 0;
        for(size_t k = 0; k < pop.sols[i]->mo_test_sols.size(); ++k)
        {
          for(size_t j = 0; j < mo_fitness_function->number_of_parameters; ++j) {
            pop.sols[i]->param[ki] = pop.sols[i]->mo_test_sols[k]->param[j];
            ki++;
          }
        }
        assert(ki == pop.sols[i]->param.size());

        
        // reorder gradients & stuff as well
        vec_t backup_adam_mt = pop.sols[i]->adam_mt;
        vec_t backup_adam_vt = pop.sols[i]->adam_vt;
        vec_t backup_gradient = pop.sols[i]->gradient;

        size_t old_sol_idx, new_param;
        for(size_t j = 0; j < backup_adam_mt.size(); ++j)
        {
          old_sol_idx = j /  mo_fitness_function->number_of_parameters;
          new_param = mo_fitness_function->number_of_parameters * cluster_index[i][old_sol_idx] + j % mo_fitness_function->number_of_parameters;
          pop.sols[i]->adam_mt[new_param] = backup_adam_mt[j];
          // std::cout << j << "->" << new_param << std::endl;
        }
        
        for(size_t j = 0; j < backup_adam_vt.size(); ++j)
        {
          old_sol_idx = j /  mo_fitness_function->number_of_parameters;
          new_param =  mo_fitness_function->number_of_parameters * cluster_index[i][old_sol_idx] + j % mo_fitness_function->number_of_parameters;
          pop.sols[i]->adam_vt[new_param] = backup_adam_vt[j];
        }
        
        for(size_t j = 0; j < backup_gradient.size(); ++j)
        {
          old_sol_idx = j /  mo_fitness_function->number_of_parameters;
          new_param =  mo_fitness_function->number_of_parameters * cluster_index[i][old_sol_idx] + j % mo_fitness_function->number_of_parameters;
          pop.sols[i]->gradient[new_param] = backup_gradient[j];
        }
      } // deze
      
      if(nothing_changed) {
        break;
      }
    }
  }
  
  bool UHV_t::vtr_reached(solution_t & sol, double vtr)
  {
    if(!redefine_vtr) {
      return false;
    }
    
    if(collect_all_mo_sols_in_archive && elitist_archive != nullptr)
    {
      double IGD = elitist_archive->computeIGD(mo_fitness_function->pareto_set);
      return (IGD <= vtr);
    }
    
    return false;
  }
  
  double UHV_t::evaluateMOSolutionWithfiniteDifferences(hicam::solution_pt & sol, double finite_differences_step_size, bool use_central_difference) const
  {
    double FD_number_of_evaluations = 0;
    
    mo_fitness_function->evaluate(sol);
    // mo_fitness_function->evaluate_with_gradients(sol);
    FD_number_of_evaluations++;
    
    hicam::solution_pt sol_plus_h = std::make_shared<hicam::solution_t>(*sol);
    hicam::solution_pt sol_min_h = std::make_shared<hicam::solution_t>(*sol);
    
    sol->gradients.resize(mo_fitness_function->number_of_objectives);
    for(size_t i = 0; i < mo_fitness_function->number_of_objectives; ++i) {
      sol->gradients[i].resize(sol->param.size());
    }
    
    for(size_t i = 0; i < sol->param.size(); ++i)
    {
      double effective_h = finite_differences_step_size;
      double old_param = sol_plus_h->param[i];
      // set the params for the step.
      if(use_central_difference) {
        sol_min_h->param[i] -= 0.5 * effective_h;
        sol_plus_h->param[i] += 0.5 * effective_h;
        while (old_param == sol_min_h->param[i] || old_param == sol_plus_h->param[i]) {
          effective_h *= 100;
          sol_min_h->param[i] = old_param - 0.5 * effective_h;
          sol_plus_h->param[i] = old_param + 0.5 * effective_h;
        }
         assert(old_param != sol_min_h->param[i] && old_param != sol_plus_h->param[i]);
      } else {
        sol_plus_h->param[i] += effective_h;
        while (old_param == sol_plus_h->param[i]) { // increase h if its too small and results in steps smaller than the machine precision.
          effective_h *= 100; // if the previous h did not add anything, make it sufficiently larger to move away from round-off errors.
          sol_plus_h->param[i] = old_param + effective_h;
        }
        assert(old_param != sol_plus_h->param[i]);
      }
      
      // check if sol is in range, else, flip h.
      if(sol_plus_h->param[i] < lower_param_bounds[i] || sol_plus_h->param[i] > upper_param_bounds[i]) {
        effective_h *= -1;
        sol_plus_h->param[i] = old_param + effective_h;
        assert(old_param != sol_plus_h->param[i]);
      }
      
      mo_fitness_function->evaluate(sol_plus_h);
      FD_number_of_evaluations++;
      
      if(use_central_difference) {
        mo_fitness_function->evaluate(sol_min_h);
        FD_number_of_evaluations++;
      }
      
      for(size_t j = 0; j < mo_fitness_function->number_of_objectives; ++j)
      {
        if(use_central_difference) {
          sol->gradients[j][i] = (sol_plus_h->obj[j] - sol_min_h->obj[j]) / effective_h;
        } else {
          sol->gradients[j][i] = (sol_plus_h->obj[j] - sol->obj[j]) / effective_h;
        }
      }
      
      // reset the param we changed in this iteration
      sol_plus_h->param[i] = sol->param[i];
      sol_min_h->param[i] = sol->param[i];
      sol_plus_h->obj = sol->obj;
      sol_min_h->obj = sol->obj;
    }
    
    
    return FD_number_of_evaluations;
  }
  
  
  double UHV_t::partialEvaluateMOSolutionWithfiniteDifferences(hicam::solution_pt & sol, double finite_differences_step_size, bool use_central_difference, const std::vector<size_t> & touched_parameter_idx, const hicam::solution_pt & old_sol) const
  {
    double FD_weighted_number_of_evaluations = 0;
    
    mo_fitness_function->partial_evaluate(sol, touched_parameter_idx, old_sol);
    // mo_fitness_function->partial_evaluate_with_gradients(sol, touched_parameter_idx, old_sol);
    
    double n =  (double) sol->param.size();
    FD_weighted_number_of_evaluations += (touched_parameter_idx.size() / n);
    
    // get next step solutions (only 1 is needed for forward. but hey.. who cares?
    hicam::solution_pt sol_plus_h = std::make_shared<hicam::solution_t>(*sol);
    hicam::solution_pt sol_min_h = std::make_shared<hicam::solution_t>(*sol);
    
    // resize gradients if needed
    if(sol->gradients.size() != mo_fitness_function->number_of_objectives)
    {
      sol->gradients.resize(mo_fitness_function->number_of_objectives);
      for(size_t i = 0; i < mo_fitness_function->number_of_objectives; ++i) {
        sol->gradients[i].resize(sol->param.size());
      }
    }
    
    // loop over the touched parameters, update 1 at a time.
    std::vector<size_t> touched_idx(1,0);
    for(size_t ki = 0; ki < touched_parameter_idx.size(); ++ki)
    {
      size_t i = touched_parameter_idx[ki];
      touched_idx[0] = i;
      

      double effective_h = finite_differences_step_size;
      
      // set the params for the step.
      if(use_central_difference) {
        double old_param = sol_plus_h->param[i];
        sol_min_h->param[i] -= 0.5 * effective_h;
        sol_plus_h->param[i] += 0.5 * effective_h;
        while (old_param == sol_min_h->param[i] || old_param == sol_plus_h->param[i]) {
          effective_h *= 100;
          sol_min_h->param[i] = old_param - 0.5 * effective_h;
          sol_plus_h->param[i] = old_param + 0.5 * effective_h;
        }
        assert(old_param != sol_min_h->param[i] && old_param != sol_plus_h->param[i]);
      } else {
        double old_param = sol_plus_h->param[i];
        sol_plus_h->param[i] += effective_h;
        while (old_param == sol_plus_h->param[i]) { // increase h if its too small and results in steps smaller than the machine precision.
          effective_h *= 100; // if the previous h did not add anything, make it sufficiently larger to move away from round-off errors.
          sol_plus_h->param[i] = old_param + effective_h;
        }
        assert(old_param != sol_plus_h->param[i]);
      }
      
      mo_fitness_function->partial_evaluate(sol_plus_h, touched_idx, old_sol);
      FD_weighted_number_of_evaluations += 1.0 / (double) n;
      
      if(use_central_difference) {
        mo_fitness_function->partial_evaluate(sol_min_h, touched_idx, old_sol);
        FD_weighted_number_of_evaluations += 1.0 / (double) n;
      }
      
      for(size_t j = 0; j < mo_fitness_function->number_of_objectives; ++j)
      {
        if(use_central_difference) {
          sol->gradients[j][i] = (sol_plus_h->obj[j] - sol_min_h->obj[j]) / effective_h;
        } else {
          sol->gradients[j][i] = (sol_plus_h->obj[j] - sol->obj[j]) / effective_h;
        }
      }
      
      // reset the param we changed in this iteration
      sol_plus_h->param[i] = sol->param[i];
      sol_min_h->param[i] = sol->param[i];
      sol_plus_h->obj = sol->obj;
      sol_min_h->obj = sol->obj;
    }
    
    
    return FD_weighted_number_of_evaluations;
  }
}

