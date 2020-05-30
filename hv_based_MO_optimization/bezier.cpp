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
#include "bezier.hpp"
#include "HillVallEA/gomea.hpp"

// for MO problems
#include "../domination_based_MO_optimization/mohillvallea/hicam_external.h"

namespace hillvallea
{
  
  bezierUHV_t::bezierUHV_t
  (
   hicam::fitness_pt mo_fitness_function,
   size_t bezier_degree,
   size_t number_of_test_points,
   bool collect_all_mo_sols_in_archive,
   size_t elitist_archive_size,
   hicam::elitist_archive_pt initial_archive
   )
  {
    
    this->mo_fitness_function = mo_fitness_function;
    this->number_of_test_points = number_of_test_points;
    this->bezier_degree = bezier_degree;
    this->collect_all_mo_sols_in_archive = collect_all_mo_sols_in_archive;
    this->elitist_archive_size = elitist_archive_size;
    
    // validate MO Objective Function
    if(this->mo_fitness_function->get_number_of_objectives() != 2) {
      std::cout << "Error: method not implemented for more than 2 objectives" << std::endl;
    }
    
    // Validate number of reference points
    if(this->bezier_degree <= 1) {
      this->bezier_degree = 2;
    }
    
    // Validate number of test points
    if(this->number_of_test_points < this->bezier_degree) {
      this->number_of_test_points = this->bezier_degree;
    }
    
    this->number_of_parameters = this->bezier_degree * this->mo_fitness_function->number_of_parameters;
    maximum_number_of_evaluations = 0;// quite sure this is unused.
    
    // redefine initialization
    redefine_random_initialization = true; // this->mo_fitness_function->redefine_random_initialization;
    partial_evaluations_available = true; // this->mo_fitness_function->partial_evaluations_available;
    linkage_learning_distance_matrix_available = true; // this->mo_fitness_function->linkage_learning_custom_distance_matrix_available;
    dynamic_linkage_learning_distance_matrix_available = true;
    
    // the upper bound is always all solutions
    fos_element_size_upper_bound = this->bezier_degree * this->mo_fitness_function->fos_element_size_upper_bound;
    
    // if the mo function has partial evaluations, we set the lower bound to the one of the
    // mo function. If not, the lower bound is one mo_sol.
    if(this->mo_fitness_function->partial_evaluations_available) {
      fos_element_size_lower_bound = this->bezier_degree * this->mo_fitness_function->fos_element_size_lower_bound; // this->bezier_degree; //  *
    } else {
      fos_element_size_lower_bound = this->mo_fitness_function->number_of_parameters;
      // fos_element_size_lower_bound = this->mo_fitness_function->number_of_parameters;
      has_round_off_errors_in_partial_evaluations = false; // if the mo-sols have no partial evaluations, we do not have to re-evaluate the HV every once in a while
    }
    
    
    covariance_block_size = number_of_parameters;
    
    // fos_element_size_upper_bound = fos_element_size_lower_bound;
    if(this->elitist_archive_size == 0) {
      this->elitist_archive_size = maximum_number_of_evaluations;
    }
    
    this->elitist_archive = initial_archive;
    
    // allocate archive
    if(this->collect_all_mo_sols_in_archive) {
      rng_pt rng = std::make_shared<rng_t>(142391);
      if(initial_archive == nullptr) {
        elitist_archive = std::make_shared<hicam::elitist_archive_t>(this->elitist_archive_size, rng);
      }
      
      hicam::vec_t r(2);
      r[0] = mo_fitness_function->hypervolume_max_f0;
      r[1] = mo_fitness_function->hypervolume_max_f1;
      elitist_archive->set_use_hypervolume_for_size_control(false, r);
      
    }
    
    number_of_mo_evaluations = 0;
    
  }
    
  bezierUHV_t::~bezierUHV_t() {}
    
  void bezierUHV_t::get_param_bounds(vec_t & lower, vec_t & upper) const
  {
    hicam::vec_t mo_lower, mo_upper;
    mo_fitness_function->get_param_bounds(mo_lower, mo_upper);
    size_t mo_nop = mo_fitness_function->number_of_parameters;
    
    lower.resize(number_of_parameters);
    upper.resize(number_of_parameters);
    
    size_t ki = 0;
    for(size_t k = 0; k < bezier_degree; ++k)
    {
      for(size_t i = 0; i < mo_nop; ++i) {
        lower[ki] = mo_lower[i];
        upper[ki] = mo_upper[i];
        ki++;
      }
    }
    assert(ki == number_of_parameters);
  }
    
  // makes sure that sol[0]->obj[0] < sol[end]->obj[0]
  // if not, flip the order of reference solutions.
  void bezierUHV_t::flip_line_direction(solution_t & sol)
  {
    size_t R = sol.mo_reference_sols.size();
    
    if(sol.mo_reference_sols[R-1]->better_than_unconstraint_per_objective(*sol.mo_reference_sols[0], 0))
    {
      std::vector<hicam::solution_pt> mo_sols_temp = sol.mo_reference_sols;
      
      size_t ki = 0;
      for(size_t k = 0; k < sol.mo_reference_sols.size(); ++k)
      {
        sol.mo_reference_sols[k] = mo_sols_temp[R-1-k];
      
        for(size_t i = 0; i < mo_fitness_function->number_of_parameters; ++i) {
          sol.param[ki] = sol.mo_reference_sols[k]->param[i];
          ki++;
        }
      }
      assert(ki == sol.param.size());
      
      // if there are reference sols,
      // flip them too.
      size_t K = sol.mo_test_sols.size();
      
      if(K > 0)
      {
        mo_sols_temp = sol.mo_test_sols;
        for(size_t k = 0; k < K; ++k) {
          sol.mo_test_sols[k] = mo_sols_temp[K-1-k];
        }
      }
    }
  }
  
  void bezierUHV_t::set_and_evaluate_mo_solutions(solution_t & sol, size_t number_of_test_points, hicam::fitness_pt mo_fitness_function)
  {
    
    // get reference sols
    set_reference_points(sol, mo_fitness_function->number_of_parameters);
    
    for(size_t k = 0; k < bezier_degree; ++k)
    {
      
      if(k == 0 || k == bezier_degree-1)
      {
        mo_fitness_function->evaluate(sol.mo_reference_sols[k]);
        number_of_mo_evaluations++;
        
        if(collect_all_mo_sols_in_archive) {
          elitist_archive->updateArchive(sol.mo_reference_sols[k],true);
        }
        continue;
      }
      
      if(mo_fitness_function->do_evaluate_bezier_controlpoints) {
        mo_fitness_function->evaluate_bezier_controlpoint(*sol.mo_reference_sols[k]);
      }
    }
    
    flip_line_direction(sol);
    
    // Set test points
    set_test_points(sol, number_of_test_points, mo_fitness_function->number_of_parameters);

    for(size_t k = 1; k < number_of_test_points-1; ++k)
    {
      mo_fitness_function->evaluate(sol.mo_test_sols[k]);
      number_of_mo_evaluations++;
      
      if(collect_all_mo_sols_in_archive) {
        elitist_archive->updateArchive(sol.mo_test_sols[k],true);
      }
    }
  }
    
  void bezierUHV_t::update_and_evaluate_mo_solutions(solution_t & sol, size_t number_of_test_points, hicam::fitness_pt mo_fitness_function, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
  {
    //-----------------------------------------------------
    // reconstruct MO touched parameters for reference points
    std::vector<std::vector<size_t>> mo_touched_parameter_idx;
    get_mo_touched_parameter_idx(mo_touched_parameter_idx, touched_parameter_idx, mo_fitness_function->number_of_parameters);
    
    update_reference_points_partial(sol, mo_fitness_function->number_of_parameters, mo_touched_parameter_idx);
    
    for(size_t k = 0; k < bezier_degree; ++k)
    {
      if(mo_touched_parameter_idx[k].size() > 0)
      {
        if(k == 0 || k == bezier_degree - 1)
        {
          if(mo_fitness_function->partial_evaluations_available) {
            mo_fitness_function->partial_evaluate(sol.mo_reference_sols[k], mo_touched_parameter_idx[k], old_sol.mo_reference_sols[k]);
            number_of_mo_evaluations += mo_touched_parameter_idx[k].size() / (double) sol.mo_test_sols[k]->number_of_parameters();
          } else {
            mo_fitness_function->evaluate(sol.mo_reference_sols[k]);
            number_of_mo_evaluations++;
          }
          
          if(collect_all_mo_sols_in_archive) {
            elitist_archive->updateArchive(sol.mo_reference_sols[k],true);
          }
          
          continue;
        }
        
        if(mo_fitness_function->do_evaluate_bezier_controlpoints)
        {
          if(mo_fitness_function->partial_evaluations_available) {
            mo_fitness_function->partial_evaluate_bezier_controlpoint(*sol.mo_reference_sols[k], mo_touched_parameter_idx[k], *old_sol.mo_reference_sols[k]);
          } else {
            mo_fitness_function->evaluate_bezier_controlpoint(*sol.mo_reference_sols[k]);
          }
        }
      }
    }
    flip_line_direction(sol);
    
    //-----------------------------------------------------
    // update and evaluate test points
    
    std::vector<size_t> mo_touched_parameter_idx_bezier;
    get_mo_touched_parameter_idx_bezier(mo_touched_parameter_idx_bezier, mo_touched_parameter_idx, mo_fitness_function->number_of_parameters);
    update_test_points_partial(sol, number_of_test_points, mo_fitness_function->number_of_parameters, mo_touched_parameter_idx_bezier);
    
    for(size_t k = 1; k < number_of_test_points-1; ++k)
    {
      if(mo_fitness_function->partial_evaluations_available) {
        mo_fitness_function->partial_evaluate(sol.mo_test_sols[k], mo_touched_parameter_idx_bezier, old_sol.mo_test_sols[k]);
        number_of_mo_evaluations += mo_touched_parameter_idx_bezier.size() / (double) sol.mo_test_sols[k]->number_of_parameters();
      } else {
        mo_fitness_function->evaluate(sol.mo_test_sols[k]);
        number_of_mo_evaluations++;
      }
      
      if(collect_all_mo_sols_in_archive) {
        elitist_archive->updateArchive(sol.mo_test_sols[k],true);
      }
    }
  }
  
  // gets all nondominated solutions that are in the feasible domain, defined by (r_x,r_y)
  void bezierUHV_t::get_front(const std::vector<hicam::solution_pt> & mo_sols, std::vector<bool> & is_part_of_front, vec_t & front_x, vec_t & front_y, double r_x, double r_y)
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
  
  
  void bezierUHV_t::get_mo_touched_parameter_idx(std::vector<std::vector<size_t>> & mo_touched_parameter_idx, const std::vector<size_t> & touched_parameter_idx, size_t mo_number_of_parameters)
  {
    //-----------------------------------------
    // reconstruct MO_params
    std::vector<bool> is_mo_parameter_touched(mo_number_of_parameters, false);
    
    mo_touched_parameter_idx.clear();
    mo_touched_parameter_idx.resize(bezier_degree);

    for(size_t k = 0; k < bezier_degree; ++k) {
      mo_touched_parameter_idx[k].reserve(mo_number_of_parameters);
    };
    
    for(size_t i = 0; i < touched_parameter_idx.size(); ++i) {
      mo_touched_parameter_idx[touched_parameter_idx[i] / mo_fitness_function->number_of_parameters].push_back(touched_parameter_idx[i] % mo_fitness_function->number_of_parameters);
    }
  }
  
  void bezierUHV_t::get_mo_touched_parameter_idx_bezier(std::vector<size_t> & mo_touched_parameter_idx_bezier, const std::vector<std::vector<size_t>> & mo_touched_parameter_idx, size_t mo_number_of_parameters)
  {
    //-----------------------------------------
    // collect changed variables
    std::vector<bool> is_mo_parameter_touched(mo_number_of_parameters, false);
    
    // if one param changed for any reference sol,
    // we need to update that parameter for all test sols.
    for(size_t k = 0; k < mo_touched_parameter_idx.size(); ++k)
    {
      for(size_t i = 0; i < mo_touched_parameter_idx[k].size(); ++i) {
        is_mo_parameter_touched[mo_touched_parameter_idx[k][i]] = true;
      }
    }
    
    mo_touched_parameter_idx_bezier.clear();
    mo_touched_parameter_idx_bezier.reserve(mo_number_of_parameters);
    
    for(size_t i = 0; i < is_mo_parameter_touched.size(); ++i) {
      if(is_mo_parameter_touched[i]) {
        mo_touched_parameter_idx_bezier.push_back(i);
      }
    }
    
  }
  
  void bezierUHV_t::define_problem_evaluation(solution_t & sol){
    std::vector<size_t> dummy_touched_parameter_idx;
    solution_t dummy_old_sol;
    bool partial_evaluation = false;
    compute_fitness(partial_evaluation, sol, dummy_touched_parameter_idx, dummy_old_sol);
  }
  
  void bezierUHV_t::define_partial_problem_evaluation(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol) {
    bool partial_evaluation = true;
    compute_fitness(partial_evaluation, sol, touched_parameter_idx, old_sol);
  }
  
  void bezierUHV_t::compute_fitness(bool partial_evaluation, solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
  {
    
    // reset solution.
    sol.f = 0.0;
    sol.constraint = 0.0;
    
    //-----------------------------------------
    // reconstruct MO_params
    size_t K = number_of_test_points;
    
    if(partial_evaluation) {
      update_and_evaluate_mo_solutions(sol, number_of_test_points, mo_fitness_function, touched_parameter_idx, old_sol);
    } else {
      set_and_evaluate_mo_solutions(sol, number_of_test_points, mo_fitness_function);
    }
    
    std::vector<bool> is_part_of_front;
    vec_t front_x;
    vec_t front_y;
    double r0 = mo_fitness_function->hypervolume_max_f0;
    double r1 = mo_fitness_function->hypervolume_max_f1;
    get_front(sol.mo_test_sols, is_part_of_front, front_x, front_y, r0, r1);
    
    // get front indices
    std::vector<size_t> front_indices(front_x.size());
    size_t ki = 0;
    for(size_t k = 0; k < is_part_of_front.size(); ++k) {
      if(is_part_of_front[k]) {
        front_indices[ki] = k;
        ki++;
      }
    }
    assert(ki == front_x.size());
    
    // get sorted front (i.e., remove all points that are 'wrongly sorted')
    //-----------------------------------------------
    vec_t filtered_front_x;
    vec_t filtered_front_y;
    std::vector<size_t> is_part_of_sorted_front(sol.mo_test_sols.size(), false);
    
    if(front_indices.size() > 0)
    {
      std::vector<size_t> sorted_front_order(front_indices.size());
      for (size_t i = 0; i < sorted_front_order.size(); ++i) {
        sorted_front_order[i] = i;
      }
      std::sort(std::begin(sorted_front_order), std::end(sorted_front_order), [&front_x](double idx, double idy) { return front_x[(size_t)idx] < front_x[(size_t)idy]; });
      
      filtered_front_x.push_back(front_x[0]);
      filtered_front_y.push_back(front_y[0]);
      is_part_of_sorted_front[front_indices[0]] = true;
      
      size_t largest = 0;
      
      for(size_t i = 1; i < front_x.size(); ++i) {
        if(sorted_front_order[i] > largest) {
          largest = sorted_front_order[i];
          filtered_front_x.push_back(front_x[sorted_front_order[i]]);
          filtered_front_y.push_back(front_y[sorted_front_order[i]]);
          is_part_of_sorted_front[front_indices[sorted_front_order[i]]] = true;
        }
      }
    }
    
    //-----------------------------------------------
    // Set objective (Hyper Volume)
    std::vector<size_t> sorted_front_order; // this could be skipped cuz its already sorted
    double HV = compute2DHyperVolume(filtered_front_x, filtered_front_y, sorted_front_order, r0, r1);
    sol.f = - HV;

    // uncrowded distance
    double penalty = 0.0;
    double penalty_factor = 1.0 /((double) (sol.mo_test_sols.size()));
    std::vector<vec_t> nearest_point_on_front(sol.mo_test_sols.size());
    
    for(size_t k = 0; k < K; ++k)
    {
      if(!is_part_of_sorted_front[k])
      {
        // uncrowded distance
        double dist = distance_to_front(sol.mo_test_sols[k]->obj[0], sol.mo_test_sols[k]->obj[1], filtered_front_x, filtered_front_y, sorted_front_order, r0, r1);
        assert(dist >= 0);
        penalty += penalty_factor * pow(dist,mo_fitness_function->number_of_objectives);
      }
    }

    // the UHV is now complete.
    sol.constraint += penalty;
    
    //-----------------------------------------------
    // Constraint / Penalty : end control points must model endpoints of PF.
    
    for(size_t k = 0; k < K-1; ++k)
    {
      if(!is_part_of_sorted_front[k+1] || !is_part_of_sorted_front[k]) {
        sol.constraint += (sol.mo_test_sols[k+1]->obj - sol.mo_test_sols[k]->obj).norm(); // Euclidean distance between the two points.
      }
    }
    
  }

  void bezierUHV_t::init_solutions_randomly(population_t & population, size_t sample_size, const vec_t & lower_init_ranges, const vec_t & upper_init_ranges, size_t number_of_elites, rng_pt rng)
  {
    // initialize solutions
    population.sols.resize(sample_size);
    for(size_t i = number_of_elites; i < population.sols.size(); ++i)
    {
      if (population.sols[i] == nullptr) {
        solution_pt sol = std::make_shared<solution_t>(number_of_parameters);
        population.sols[i] = sol;
      }
      population.sols[i]->mo_reference_sols.clear();
      population.sols[i]->mo_reference_sols.resize(bezier_degree, nullptr);
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
      size_t number_of_means = std::min(bezier_degree, elitist_archive->sols.size());
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
        
        for (size_t j = 0; j < std::min(sample_size-number_of_elites, N/bezier_degree*2); ++j) {
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
          
          assert(population.sols[number_of_elites + i]->mo_reference_sols[k] == nullptr);
          population.sols[number_of_elites + i]->mo_reference_sols[k] = std::make_shared<hicam::solution_t>(*clusters[k]->sols[i]); // make a copy to be sure.
        }
      }
    } // end of elite-based initialization
    
    
    hicam::population_pt mo_population = std::make_shared<hicam::population_t>();
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
    
      size_t mo_sample_size = bezier_degree;
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
          if(population.sols[i]->mo_reference_sols[k] == nullptr) {
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
  void bezierUHV_t::linkage_learning_distance_matrix(matrix_t & M)
  {
    /*
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
    //*/

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
      for(int j = i; j < mo_nop; ++j)
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

    // merge the same parameter of different control points first,
    // and later merge the different parameters.
    // so we set all block diagonal entries 'large'
    for(int i = 0; i < (int) number_of_parameters; ++i)
    {
      for(int j = 0; j < (int) number_of_parameters; ++j) {
        if((i/mo_nop) != (j/mo_nop)) {
          M[i][j] = fabs((i/mo_nop)-(j/mo_nop)) * ceil(max_entry-min_entry) * 10 + fabs((i%mo_nop)-(j%mo_nop)) * ceil(max_entry-min_entry) + (mo_M[i%mo_nop][j%mo_nop]/(max_entry - min_entry) - min_entry);
        }
        else
        {
          M[i][j] = (1 + number_of_parameters / mo_nop) * ceil(max_entry-min_entry) * 10 + (mo_M[i%mo_nop][j%mo_nop]/(max_entry - min_entry) - min_entry);
        }
      }
    }
    
    // std::cout << M << std::endl;
    //*/
  }
  
  // the distance matrix determines which solutions get merged in
  // the static linkage tree setup, which is in our case simply
  // the
  void bezierUHV_t::dynamic_linkage_learning_distance_matrix(matrix_t & M, const population_t & pop)
  {
    
    if(pop.size() == 0 || pop.sols[0]->mo_reference_sols.size() == 0) {
      linkage_learning_distance_matrix(M);
      return;
    }
    
    size_t N = pop.sols.size();
    size_t K = pop.sols[0]->mo_reference_sols.size();
    
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
        means[k]->obj += pop.sols[i]->mo_reference_sols[k]->obj / ((double) N);
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
  
  std::string bezierUHV_t::write_solution_info_header(bool niching_enabled)
  {
    bool write_smoothness = true;
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
  std::string bezierUHV_t::write_additional_solution_info(const solution_t & best,  const std::vector<solution_pt> & niching_archive, bool niching_enabled)
  {
    bool write_smoothness = true;
    
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
    // << " " << std::setw(4) << std::fixed << (int) mo_fitness_function->number_of_evaluations; // alternative way to count MO-evals
    
    return ss.str();
  }
    
  std::string bezierUHV_t::name() const
  {
    std::ostringstream ss;
    ss << "Hypervolume_search_linear_" << mo_fitness_function->name();
    return ss.str();
  }
  
  void bezierUHV_t::write_solution(const solution_t & sol, const std::string & filename)
  {
    std::ofstream file;
    file.open(filename, std::ofstream::out | std::ofstream::trunc);
  
    for(size_t k = 0; k < sol.mo_test_sols.size(); ++k) {
      file << sol.mo_test_sols[k]->obj << " " << sol.mo_test_sols[k]->dvis << " " << sol.mo_test_sols[k]->param << std::endl;
    }

    file.close();
  }
  
  // distance to a box defined by [-infty, ref_x, -infty, ref_y]
  double bezierUHV_t::distance_to_box(double ref_x, double ref_y, double p_x, double p_y)
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
  double bezierUHV_t::distance_to_front(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y)
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
  
  double bezierUHV_t::distance_to_front_without_corner_boxes(double p_x, double p_y, const vec_t & obj_x, const vec_t & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y)
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
    
    // distance to the 'end' boxes
    vec_t new_nearest_point_on_front(2,0.0);
    double dist;
    double min_dist = 1e300;
    if(obj_x.size() == 1) {
      min_dist = distanceEuclidean2D(obj_x[0], obj_y[0], p_x, p_y);
    }
    else
    {
      // distance to 'inner' boxes
      for(size_t k = 1; k < n; ++k)
      {
        dist = distance_to_box(obj_x[sorted_obj[k]], obj_y[sorted_obj[k-1]], p_x, p_y);
        
        if(dist < min_dist) {
          min_dist = dist;
        }
      }
    }
    
    assert(min_dist >= 0); // can be 0 if its at the front!
    return min_dist;
  }
  
  // creates mo_sols from the parameters of sol.
  void  bezierUHV_t::set_reference_points(solution_t & sol, size_t mo_number_of_parameters)
  {
    size_t bezier_degree = sol.param.size() / mo_number_of_parameters;
    assert(sol.param.size() % mo_number_of_parameters == 0);
    
    // clear and reset if required
    if(sol.mo_reference_sols.size() != bezier_degree) {
      sol.mo_reference_sols.clear();
      sol.mo_reference_sols.resize(bezier_degree);
    }
    
    for(size_t k = 0; k < bezier_degree; ++k)
    {
      // allocate mo_sols
      if(sol.mo_reference_sols[k] == nullptr) {
        sol.mo_reference_sols[k] = std::make_shared<hicam::solution_t>(mo_number_of_parameters);
      }
      
      // set parameters
      // i once defined this reverse, i.e., K-k-1 instead of simply k, but to preserve similarity i kept it
      // todo: replace at some point.
      for(size_t i = 0; i < mo_number_of_parameters; ++i) {
        sol.mo_reference_sols[k]->param[i] = sol.param[i + k * mo_number_of_parameters];
      }
    }
  }
  
  // update the reference points of sol. Partial-update version of 'set_reference_points'
  void  bezierUHV_t::update_reference_points_partial(solution_t & sol, size_t mo_number_of_parameters, const std::vector<std::vector<size_t>> & mo_touched_parameter_idx)
  {
    size_t number_of_reference_points = sol.param.size() / mo_number_of_parameters;
    assert(sol.param.size() % mo_number_of_parameters == 0);
    
    if(sol.mo_reference_sols.size() != number_of_reference_points) {
      set_reference_points(sol, mo_number_of_parameters);
      return;
    }
    
    for(size_t k = 0; k < number_of_reference_points; ++k)
    {
      size_t var;
      for(size_t i = 0; i < mo_touched_parameter_idx[k].size(); ++i)
      {
        var = mo_touched_parameter_idx[k][i];
        sol.mo_reference_sols[k]->param[var] = sol.param[var + k * mo_number_of_parameters];
      }
    }
  }
  
  double bezierUHV_t::bezier_curve(double d, const vec_t & p, int p_start, int p_end)
  {
    if(p_start == p_end) {
      return p[p_start];
    }
    
    double b_left = bezier_curve(d, p, p_start, p_end-1);
    double b_right = bezier_curve(d, p, p_start+1, p_end);
    
    return (1.0 - d) * b_left + d * b_right;
  }
  
  // set the test points based on the reference points of sol.
  void  bezierUHV_t::set_test_points(solution_t & sol, size_t number_of_test_points, size_t mo_number_of_parameters)
  {
    size_t number_of_reference_sols = sol.mo_reference_sols.size();
    
    // clear and reset if required
    if(sol.mo_test_sols.size() != number_of_test_points)
    {
      sol.mo_test_sols.clear();
      sol.mo_test_sols.resize(number_of_test_points);
    }
    
    //-----------------------------------------------------------------------
    // using Bezier Curve interpolation
    // set endpoints as first and last reference sol
    vec_t bezier_points(number_of_reference_sols);
    sol.mo_test_sols[0] = sol.mo_reference_sols[0];
    sol.mo_test_sols[number_of_test_points-1] = sol.mo_reference_sols[number_of_reference_sols-1];
    
    for(size_t k = 1; k < number_of_test_points-1; ++k)
    {
      // Bezier curve interpolation set
      double d = k / ((double) number_of_test_points - 1.0);
      
      // allocate if required
      if(sol.mo_test_sols[k] == nullptr) {
        sol.mo_test_sols[k] = std::make_shared<hicam::solution_t>(mo_number_of_parameters);
      }
      
      for(size_t i = 0; i < mo_number_of_parameters; ++i)
      {
        for(size_t j = 0; j < number_of_reference_sols; ++j) {
          bezier_points[j] = sol.mo_reference_sols[j]->param[i];
        }
        
        sol.mo_test_sols[k]->param[i] = bezier_curve(d, bezier_points, 0, (int) number_of_reference_sols-1);
      }
      
      // brachy specific
      sol.mo_test_sols[k]->doselist_update_required = false;
      sol.mo_test_sols[k]->doselists.resize(sol.mo_reference_sols[0]->doselists.size());
      
      for(size_t i = 0; i < sol.mo_test_sols[k]->doselists.size(); ++i)
      {
        sol.mo_test_sols[k]->doselists[i].resize(sol.mo_reference_sols[0]->doselists[i].size());
        
        for(size_t j = 0; j < sol.mo_test_sols[k]->doselists[i].size(); ++j)
        {
          for(size_t l = 0; l < number_of_reference_sols; ++l) {
            bezier_points[l] = sol.mo_reference_sols[l]->doselists[i][j];
          }
          
          sol.mo_test_sols[k]->doselists[i][j] = bezier_curve(d, bezier_points, 0, (int) number_of_reference_sols-1);
        }
      }
      // end brachy specific
    }
    // end Bezier Curve interpolation
  }
  
  // set the test points based on the reference points of sol.
  void bezierUHV_t::update_test_points_partial(solution_t & sol, size_t number_of_test_points, size_t mo_number_of_parameters, const std::vector<size_t> & mo_touched_parameter_idx_bezier)
  {
    // set if required & return
    assert(sol.mo_test_sols.size() == number_of_test_points);
    
    // only implemented for two reference points so far.
    size_t number_of_reference_sols = sol.mo_reference_sols.size();
    
    //-----------------------------------------------------------------------
    // using Bezier Curve interpolation
    
    // set endpoints as first and last reference sol
    sol.mo_test_sols[0] = sol.mo_reference_sols[0];
    sol.mo_test_sols[number_of_test_points-1] = sol.mo_reference_sols[number_of_reference_sols-1];
    vec_t bezier_points(number_of_reference_sols);
      
    for(size_t k = 1; k < number_of_test_points-1; ++k)
    {
      // set parameters & dose list
      double d = k / ((double) number_of_test_points - 1.0);
      
      size_t var;
      for(size_t i = 0; i < mo_touched_parameter_idx_bezier.size(); ++i)
      {
        var = mo_touched_parameter_idx_bezier[i];
        
        for(size_t j = 0; j < number_of_reference_sols; ++j) {
          bezier_points[j] = sol.mo_reference_sols[j]->param[var];
        }
        sol.mo_test_sols[k]->param[var] = bezier_curve(d, bezier_points, 0, (int) number_of_reference_sols-1);
      }
      
      // brachy specific
      sol.mo_test_sols[k]->doselist_update_required = false;
      sol.mo_test_sols[k]->doselists.resize(sol.mo_reference_sols[0]->doselists.size());
      
      for(size_t i = 0; i < sol.mo_test_sols[k]->doselists.size(); ++i)
      {
        sol.mo_test_sols[k]->doselists[i].resize(sol.mo_reference_sols[0]->doselists[i].size());
        
        for(size_t j = 0; j < sol.mo_test_sols[k]->doselists[i].size(); ++j)
        {
          for(size_t l = 0; l < number_of_reference_sols; ++l) {
            bezier_points[l] = sol.mo_reference_sols[l]->doselists[i][j];
          }
          sol.mo_test_sols[k]->doselists[i][j] = bezier_curve(d, bezier_points, 0, (int) number_of_reference_sols-1);
        }
      }
      // end brachy specific
    }
    
  }
  
  double bezierUHV_t::compute2DHyperVolume(const vec_t & obj0, const vec_t & obj1, std::vector<size_t> & sorted_obj, double max_0, double max_1)
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
      
      assert(obj0[sorted_obj[i+1]] >= obj0[sorted_obj[i]] );
      double d = (min(max_0, obj0[sorted_obj[i + 1]]) - min(max_0, obj0[sorted_obj[i]])) * (max_1 - min(max_1, obj1[sorted_obj[i]]));
      assert(d>=0);
      area += d;
    }
    
    return area;
  }
  
  
  void bezierUHV_t::sort_population_parameters(population_t & pop, FOS_t & FOS)
  {
    if(collect_all_mo_sols_in_archive && elitist_archive != nullptr) {
      elitist_archive->adaptArchiveSize();
    }
  }
  
  bool bezierUHV_t::vtr_reached(solution_t & sol, double vtr)
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

  
  
}
