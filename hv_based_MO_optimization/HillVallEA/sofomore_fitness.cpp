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
#include "sofomore_fitness.hpp"
#include "fitness.h"
#include "mathfunctions.hpp"

// for MO problems
#include "../../domination_based_MO_optimization/mohillvallea/hicam_external.h"

namespace hillvallea
{
  
  sofomore_fitness_t::sofomore_fitness_t(hicam::fitness_pt mo_fitness_function, std::vector<hicam::solution_pt> * mo_population, bool collect_all_mo_sols_in_archive, hicam::elitist_archive_pt elitist_archive)
  {
    
    this->mo_fitness_function = mo_fitness_function;
    this->mo_population = mo_population;
    this->collect_all_mo_sols_in_archive = collect_all_mo_sols_in_archive;
    this->elitist_archive = elitist_archive;
    
    
    this->use_boundary_repair = mo_fitness_function->use_boundary_repair;
    
    // validate MO Objective Function
    if(this->mo_fitness_function->get_number_of_objectives() != 2) {
      std::cout << "Error: method not implemented for more than 2 objectives" << std::endl;
    }
    
    this->number_of_parameters = this->mo_fitness_function->number_of_parameters;
    maximum_number_of_evaluations = 0;// quite sure this is unused.
    dynamic_objective = true;

    assert(!collect_all_mo_sols_in_archive || this->elitist_archive != nullptr);

  }
  
  sofomore_fitness_t::~sofomore_fitness_t() {}
  
  void sofomore_fitness_t::get_param_bounds(vec_t & lower, vec_t & upper) const
  {
    // a bit of a lame implementation but
    // i use two different libraries for vectors, hicam::vec_t and hillvallea::vec_t
    // they're the same but i'm simply too lazy too merge them.
    hicam::vec_t mo_lower, mo_upper;
    mo_fitness_function->get_param_bounds(mo_lower, mo_upper);
    
    lower.resize(number_of_parameters);
    upper.resize(number_of_parameters);
  
    for(size_t i = 0; i < mo_lower.size(); ++i) {
      lower[i] = mo_lower[i];
      upper[i] = mo_upper[i];
    }
  }
  
  // gets all nondominated solutions that are in the feasible domain, defined by (r_x,r_y)
  void sofomore_fitness_t::get_front(const std::vector<hicam::solution_pt> & mo_sols, std::vector<bool> & is_part_of_front, vec_t & front_x, vec_t & front_y, double r_x, double r_y)
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
  
  
  void sofomore_fitness_t::define_problem_evaluation(solution_t & sol)
  {

    // reset solution.
    sol.f = 0.0;
    sol.constraint = 0.0;
    
    //-----------------------------------------
    // reconstruct & evaluate MO_params in sol.mo_reference_sols[0]

    sol.mo_reference_sols.clear();
    sol.mo_reference_sols.push_back(std::make_shared<hicam::solution_t>());
    
    // copy params
    sol.mo_reference_sols[0]->param.resize(sol.param.size());
    for(size_t i = 0; i < sol.param.size(); ++i) {
      sol.mo_reference_sols[0]->param[i] = sol.param[i];
    }
    
    // evaluate
    mo_fitness_function->evaluate(sol.mo_reference_sols[0]);
    
    if(collect_all_mo_sols_in_archive) {
      elitist_archive->updateArchive(sol.mo_reference_sols[0],true);
    }
    
    // collect all solutions
    std::vector<hicam::solution_pt> all_sols;
    
    for(size_t i = 0 ; i < mo_population->size(); ++i) {
      if(mo_population->at(i) != nullptr) {
        all_sols.push_back(mo_population->at(i));
      }
    }
    
    all_sols.push_back(sol.mo_reference_sols[0]);
    size_t new_sol_idx = all_sols.size()-1;
    
    // get front of the test_sols+reference_sols[0];
    double r0 = mo_fitness_function->hypervolume_max_f0;
    double r1 = mo_fitness_function->hypervolume_max_f1;
    std::vector<bool> is_part_of_front;
    vec_t front_x;
    vec_t front_y;
    get_front(all_sols, is_part_of_front, front_x, front_y, r0, r1);
    
    // if the solution is not part of the front, its fitness is the distance to the front
    if(!is_part_of_front[new_sol_idx])
    {
      std::vector<size_t> sorted_front_order;
      sol.f = distance_to_front(sol.mo_reference_sols[0]->obj[0], sol.mo_reference_sols[0]->obj[1], front_x, front_y, sorted_front_order, r0, r1);
      assert(sol.f >= 0); // distance to the front can be zero
      return;
    }
    
    // the solution is part of the front, compute its hypervolume contribution.
    // idk, should this be the contribution within the front, or within the population?
    // i go for front now
    double new_hv, old_hv;
    {
      std::vector<size_t> sorted_front_order;
      new_hv = compute2DHyperVolume(front_x, front_y, sorted_front_order, r0, r1);
    }
    
    vec_t old_front_x;
    vec_t old_front_y;

    // get front, but now without the new sol
    {
      std::vector<size_t> sorted_front_order;
      
      std::vector<hicam::solution_pt> all_front_sols;
      for(size_t i = 0; i < is_part_of_front.size(); ++i) {
        if(is_part_of_front[i] && i != new_sol_idx) {
          all_front_sols.push_back(all_sols[i]);
        }
      }
      
      // all_sols.resize(all_sols.size()-1); // trim last sol out;
      // get_front(all_sols, is_part_of_front, old_front_x, old_front_y, r0, r1);
      get_front(all_front_sols, is_part_of_front, old_front_x, old_front_y, r0, r1);
      old_hv = compute2DHyperVolume(old_front_x, old_front_y, sorted_front_order, r0, r1);
    }
    
    // we perform minimization, so the distance to the front is positive, and the hypervolume improvement, i.e.
    // new - old, is negative. Resulting in this:
    sol.f = old_hv - new_hv;
    
    // this could not be possible, but round-off errors are a bitch. Therefore, recompute the HV and see if that works better
    if(sol.f > 0)
    {
      std::vector<size_t> sorted_front_order;
      new_hv = compute2DHyperVolume(front_y, front_x, sorted_front_order, r1, r0); // note, y & x are flipped!
      sorted_front_order.clear();
      old_hv = compute2DHyperVolume(old_front_y, old_front_x, sorted_front_order, r1, r0);
      
      double new_f = old_hv - new_hv;
      
      if(new_f < 0) {
        sol.f = new_f;
      }
    }
    
    if(sol.f >= 1e-15) // again roundoffs?
    {
      std::cout << "Sofomore, fitness value corrected to 0 from " << sol.f << "\n";
      sol.f = 1e-15;
    }
    
    
    assert(!isnan(sol.f));
  }
  
  void sofomore_fitness_t::init_solutions_randomly(population_t & population, size_t sample_size, const vec_t & lower_init_ranges, const vec_t & upper_init_ranges, size_t number_of_elites, rng_pt rng)
  {
    
    // not used yet
    /*
    // initialize solutions
    population.sols.resize(sample_size);
    for(size_t i = number_of_elites; i < population.sols.size(); ++i)
    {
      if (population.sols[i] == nullptr) {
        solution_pt sol = std::make_shared<solution_t>(number_of_parameters);
        population.sols[i] = sol;
      }
    }
    
    hicam::population_pt mo_population = std::make_shared<hicam::population_t>();
    size_t mo_sample_size = (sample_size - number_of_elites);
    size_t mo_number_of_parameters = mo_fitness_function->number_of_parameters;
    
    hicam::vec_t mo_lower_init_ranges(mo_number_of_parameters);
    hicam::vec_t mo_upper_init_ranges(mo_number_of_parameters);
    
    for(size_t i = 0; i < mo_number_of_parameters; ++i) {
      mo_lower_init_ranges[i] = lower_init_ranges[i];
      mo_upper_init_ranges[i] = upper_init_ranges[i];
    }
    
    mo_fitness_function->init_solutions_randomly(mo_population, mo_sample_size, mo_lower_init_ranges, mo_upper_init_ranges, 0, rng);
    
    // sample solutions
    for(size_t i = number_of_elites; i < population.sols.size(); ++i)
    {

      population.sols[i]->mo_reference_sols.push_back(mo_population->sols[i-number_of_elites]);
      
      for(size_t j = 0; j < number_of_parameters; ++j) {
        population.sols[i]->param[j] = population.sols[i]->mo_reference_sols[0]->param[j];
      }
    }
     */
  }
  
  std::string sofomore_fitness_t::name() const
  {
    std::ostringstream ss;
    ss << "Hypervolume_search_linear_" << mo_fitness_function->name();
    return ss.str();
  }
  
  // distance to a box defined by [-infty, ref_x, -infty, ref_y]
  double sofomore_fitness_t::distance_to_box(double ref_x, double ref_y, double p_x, double p_y)
  {
    double dx = max(0.0, p_x - ref_x );
    double dy = max(0.0, p_y - ref_y );
    
    return sqrt(dx*dx + dy*dy);
    
  }
  
  // Based on the Uncrowded Hypervolume improvement by the Inria group,
  // but we extened the definition to points that are outside of the reference frame
  // we compute the distance to the non-dominated area, within the reference window (r_x,r_y)
  // define the area points a(P^(i)_x, P^(i-1)_y), for i = 0...n (n =  number of points in front, i.e., obj0.size())
  // and let P^(-1)_y = r_y,   and    P^(n)_x = r_x
  double sofomore_fitness_t::distance_to_front(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y)
  {
    // if the front is empty, use the reference point for the distance measure
    if(obj_x.size() == 0) {
      return distance_to_box(r_x, r_y, p_x, p_y);
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
    double min_dist = min( distance_to_box(obj_x[sorted_obj[0]], r_y, p_x, p_y), distance_to_box(r_x, obj_y[sorted_obj[n-1]], p_x, p_y) );
    
    // distance to 'inner' boxes
    for(size_t k = 1; k < n; ++k)
    {
      dist = distance_to_box(obj_x[sorted_obj[k]], obj_y[sorted_obj[k-1]], p_x, p_y);
      
      if(dist < min_dist) {
        min_dist = dist;
      }
    }
    
    assert(min_dist >= 0);
    return min_dist;
  }
  
  double sofomore_fitness_t::compute2DHyperVolume(const vec_t & obj0, const vec_t & obj1, std::vector<size_t> & sorted_obj, double max_0, double max_1)
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
  
  void sofomore_fitness_t::sort_population_parameters(population_t & pop, FOS_t & FOS){ };
  
}
