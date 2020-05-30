/*

By P.A.N. Bosman, S.C. Maree, 2016-2017
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "hillvalleyclustering.h"
#include "mathfunctions.h"
#include "cluster.h"

// c++ rule of three (two)
//----------------------------------------------------------------------------------
hicam::hvc_t::hvc_t(fitness_pt fitness_function)
{
  this->fitness_function = fitness_function;
}

hicam::hvc_t::~hvc_t() {}


double hicam::hvc_t::get_average_edge_length(const population_t & pop, size_t number_of_parameters)
{
  // Hill-Valley Parameters
  vec_t param_ranges;
  pop.computeParameterRanges(param_ranges);
  
  // fewer roundoff errors
  double average_edge_length = pow(pop.size(), -1.0/number_of_parameters);
  for(size_t i = 0; i < param_ranges.size(); ++i) {
    average_edge_length *= pow(param_ranges[i],1.0/number_of_parameters);
  }
  
  if(isinf(average_edge_length) || average_edge_length > 1e300)
  {
    average_edge_length = 0.01;
    
    if(pop.previous != nullptr && pop.previous->clusters.size() > 0) {
      average_edge_length = pop.previous->clusters[0]->init_bandwidth;
    }
  }
  
  return average_edge_length;
}

double hicam::hvc_t::get_max_number_of_neighbours(size_t number_of_parameters)
{
  size_t max_number_of_neighbours = 1;
  if( number_of_parameters <= 2 ) {
    max_number_of_neighbours = number_of_parameters + 1;
  }
  else {
    max_number_of_neighbours = 2 + log(number_of_parameters);
  }
  
  return max_number_of_neighbours;
}

// returns true if the solutions belong to the same niche
bool hicam::hvc_t::MO_HillValleyTest(const hicam::solution_t &sol1, const hicam::solution_t &sol2, const unsigned int max_trials, unsigned int & number_of_evaluations, std::vector<solution_pt> & test_solutions, bool consider_constraints, bool & rejected_because_out_of_domain)
{
  
  test_solutions.clear();

  // if the solution belongs to a different valley in any objective,
  // the solutions do not belong to the same niche
  for (size_t objective_number = 0; objective_number < sol1.number_of_objectives(); ++objective_number)
  {
    if (SO_HillValleyTest(sol1, sol2, objective_number, max_trials, number_of_evaluations, test_solutions, consider_constraints, rejected_because_out_of_domain) == false) {
      return false;
    }
  }

  return true;

}

// hill-valley test in a single objective dimension
// returns true if two solutions belong to the same basin
bool hicam::hvc_t::SO_HillValleyTest(const solution_t &sol1, const solution_t &sol2, size_t objective_number, unsigned int max_trials, unsigned int & number_of_evaluations, std::vector<solution_pt> & test_solutions, bool consider_constraints, bool & rejected_because_out_of_domain)
{
  
  // if 1 is the better sol, then 2 is the worst, so take that value
  bool print = false;
  
  // sample all solutions
  // sample max_trials test points along the line segment from worst and best
  if(test_solutions.size() == 0)
  {
    test_solutions.resize(max_trials);
    
    for (size_t k = 0; k < max_trials; k++)
    {
      test_solutions[k] = std::make_shared<solution_t>();
      test_solutions[k]->param = sol1.param + ((k + 1.0) / (max_trials + 1.0)) * (sol2.param - sol1.param);
    }
  }
  
  // try the points, and evaluate only if the objective is not set yet.
  if(print) {
    std::cout << sol1.param_distance(sol2) << std::endl << std::fixed << std::setw(10) << std::setprecision(3) <<  sol1.dvis << "\t" << sol1.obj << "\t" << sol1.constraint << std::endl;
  }
  
  for (size_t k = 0; k < max_trials; k++)
  {
    if(test_solutions[k]->number_of_objectives() == 0) {
      number_of_evaluations++;
      fitness_function->evaluate(test_solutions[k]);
    }
    
    if(print) {
      std::cout << std::fixed << std::setw(10) << std::setprecision(3) << test_solutions[k]->dvis << "\t" << test_solutions[k]->obj << "\t" << test_solutions[k]->constraint << std::endl;
    }
    
    // if the solution belongs to a different valley in the given objective_number,
    // the solutions do not belong to the same niche
    if (SO_HillValleyTest_point(sol1, sol2, *test_solutions[k],objective_number, consider_constraints, rejected_because_out_of_domain) == false)
    {
      if(print) {
        std::cout << std::fixed << std::setw(10) << std::setprecision(3) <<  sol2.dvis << "\t" << sol2.obj << "\t" << sol2.constraint << " rejected (" << objective_number << ")" << std::endl << std::endl;
      }
      
      return false;
    }
  }
  
  if(print) {
    std::cout << std::fixed << std::setw(10) << std::setprecision(3) <<  sol2.dvis << "\t" << sol2.obj << "\t" << sol2.constraint << " accepted (" << objective_number << ")" << std::endl << std::endl;
  }
  
  return true;
  
}

// returns true if the test solu
bool hicam::hvc_t::SO_HillValleyTest_point(const solution_t & sol1, const solution_t & sol2, const solution_t &test, size_t objective_number, bool consider_constraints, bool & rejected_because_out_of_domain)
{
  const solution_t * worst;
  rejected_because_out_of_domain = false;
  
  // if 'test' is better than the worst, accept the edge,
  // if 'test' is worse, reject the edge, there is a hill in between two valleys (minimization)
  if(consider_constraints)
  {
    
    if(sol1.better_than_per_objective(sol2, objective_number)) {
      worst = &sol2;
    } else {
      worst = &sol1;
    }
    
    if (test.better_than_per_objective(*worst, objective_number)) {
      return true;
    }
    else
    {
      // the solution is false.
      if(worst->constraint == 0 &&  test.constraint > 0) { // note, if worst->constraint == 0, then also best->constraint == 0.
        rejected_because_out_of_domain = true;
      }
    }
  }
  else
  {
    
    if(sol1.better_than_unconstraint_per_objective(sol2, objective_number)) {
      worst = &sol2;
    } else {
      worst = &sol1;
    }
    
    if (test.better_than_unconstraint_per_objective(*worst, objective_number)) {
      return true;
    }
    
  }
  
  return false;
}


// this is the SO code, which assumes that the pop is sorted on fitness.
void hicam::hvc_t::cluster(const population_t & pop, std::vector<population_pt> & subpopulations, unsigned int  & number_of_evaluations, double & average_edge_length, bool add_test_solutions, bool recheck_elites, int optimizer_number, rng_pt & rng)
{

  // nothing to cluster
  if (pop.size() == 0) {
    return;
  }
  
  // set problem parameters
  size_t initial_popsize = pop.size();
  size_t number_of_parameters = pop.problem_size();
  size_t number_of_objectives = pop.sols[0]->number_of_objectives();
  
  // Hill-Valley Parameters
  average_edge_length = get_average_edge_length(pop, number_of_parameters);
  size_t max_number_of_neighbours = get_max_number_of_neighbours(number_of_parameters);
  
  vec_t dist(initial_popsize,0.0);
  
  bool edge_added = false;
  unsigned int max_number_of_trial_solutions;
  
  // number clusters
  vec_t number_of_clusters(number_of_objectives, 1);
  std::vector<vec_t> cluster_index(initial_popsize);
  
  for(size_t i =0 ; i < initial_popsize; ++i) {
    cluster_index[i].resize(number_of_objectives);
  }
  
  // adding test solutions to the clusters
  // we store all test solutions for sol i and sol j in test_solutions[i][j] with i < j.
  size_t test_solution_hash, mini, maxi; // i * initial_popsize + j for (i > j)
  std::map<size_t,std::vector<solution_pt>> test_solution_hashes;

  bool consider_constraints = true;
  
  for(size_t objective_number = 0; objective_number < number_of_objectives; ++objective_number)
  {
    
    // compute population ranks for current objective_number
    std::vector<size_t> sols_sorted_on_fitness(initial_popsize);
    vec_t collected_fitness(initial_popsize);
    
    if(consider_constraints)
    {
      for(size_t i = 0; i < initial_popsize; ++i) {
        collected_fitness[i] = pop.sols[i]->obj[objective_number] + pop.sols[i]->constraint * 1000;
      }
    }
    else
    {
      for(size_t i = 0; i < initial_popsize; ++i) {
        collected_fitness[i] = pop.sols[i]->obj[objective_number];
      }
    }

    compute_ranks_asc(collected_fitness, sols_sorted_on_fitness); // oops, i hardcoded that we do minimization. assumed
    
    // The best is the first cluster.
    cluster_index[sols_sorted_on_fitness[0]][objective_number] = 0;
    
    for (size_t i = 1; i < initial_popsize; i++)
    {
      
      // compute the distance to all better solutions.
      dist[i] = 0.0;
      size_t nearest_better_index = 0;
      size_t furthest_better_index = 0;
      size_t old_nearest_better_index = 0;
      
      for (size_t j = 0; j < i; j++) {
        dist[j] = pop.sols[sols_sorted_on_fitness[i]]->param_distance(*pop.sols[sols_sorted_on_fitness[j]]);
        
        if (dist[j] < dist[nearest_better_index]) {
          nearest_better_index = j;
        }
        
        if (dist[j] > dist[furthest_better_index]) {
          furthest_better_index = j;
        }
      }
      
      edge_added = false;
      
      // check the few nearest solutions with better fitness
      for (size_t j = 0; j < std::min(i, max_number_of_neighbours); j++)
      {
        
        // find the next-to nearest index
        if (j > 0)
        {
          old_nearest_better_index = nearest_better_index;
          nearest_better_index = furthest_better_index;
          
          for (size_t k = 0; k < i; k++)
          {
            if (dist[k] > dist[old_nearest_better_index] && dist[k] < dist[nearest_better_index]) {
              nearest_better_index = k;
            }
          }
          
        }
        
        if (!recheck_elites)
        {
          // if both are elites that belong to the same archive, accept edge.
          // if both are elites that belong to different archives, reject edge.
          
          if( pop.sols[sols_sorted_on_fitness[i]]->elite_origin != nullptr && pop.sols[sols_sorted_on_fitness[nearest_better_index]]->elite_origin != nullptr)
          {
            if(pop.sols[sols_sorted_on_fitness[i]]->elite_origin ==  pop.sols[sols_sorted_on_fitness[nearest_better_index]]->elite_origin)
            {
              // accept edge
              cluster_index[sols_sorted_on_fitness[i]][objective_number] = cluster_index[sols_sorted_on_fitness[nearest_better_index]][objective_number];
              edge_added = true;
              break;
            }
          }
        }
        
        // set budget
        max_number_of_trial_solutions = 1 + ((unsigned int)(dist[nearest_better_index] / average_edge_length));
        
        // such that mini < maxi
        mini = std::min(sols_sorted_on_fitness[nearest_better_index], sols_sorted_on_fitness[i]);
        maxi = std::max(sols_sorted_on_fitness[nearest_better_index], sols_sorted_on_fitness[i]);
        test_solution_hash = maxi * initial_popsize + mini;
        
        // there are no pre-saved test_solutions yet
        if(test_solution_hashes.find(test_solution_hash) == test_solution_hashes.end()) {
          test_solution_hashes[test_solution_hash] = {};
        }
        
        bool rejected_because_out_of_domain = false;
        if (SO_HillValleyTest(*pop.sols[mini], *pop.sols[maxi], objective_number, max_number_of_trial_solutions, number_of_evaluations, test_solution_hashes[test_solution_hash], consider_constraints, rejected_because_out_of_domain))
        {
          cluster_index[sols_sorted_on_fitness[i]][objective_number] = cluster_index[sols_sorted_on_fitness[nearest_better_index]][objective_number];
          edge_added = true;
          break;
        }
        
        if (rejected_because_out_of_domain)
        {
          // its out-of-domain, its not a neighbour. consider the next.
          max_number_of_neighbours++;
        }
      }
      
      
      // its a new clusters, label it like that.
      if (!edge_added)
      {
        cluster_index[sols_sorted_on_fitness[i]][objective_number] = number_of_clusters[objective_number];
        number_of_clusters[objective_number]++;
      }
    }
  }
  
  // create & fill the clusters
  //---------------------------------------------------------------------------
  
  vec_t terms(number_of_objectives,1.0);
  
  for(size_t j = 0; j < number_of_objectives; ++j)
  {
    for(size_t subj = 0; subj < j; ++subj) {
      terms[j] *= number_of_clusters[subj];
    }
    
  }
  
  std::vector<population_pt> new_clusters;
  std::map<int, size_t> cluster_hashes;
  
  int current_hash;
  for (size_t i = 0; i < initial_popsize; ++i)
  {
    current_hash = (int) terms.dot(cluster_index[i]);
    
    // if the hash is not found yet, add it and create corresponding new cluster.
    if(cluster_hashes.find(current_hash) == cluster_hashes.end())
    {
      cluster_hashes[current_hash] = new_clusters.size();
      new_clusters.push_back(std::make_shared<population_t>());
    }
    
    new_clusters[cluster_hashes[current_hash]]->sols.push_back(pop.sols[i]);
  }
  
  // now i want to add the test solutions to the clusters.
  if( add_test_solutions )
  {
    int clusteri, clusterj;
    for(size_t i = 0; i < initial_popsize; ++i)
    {
      clusteri = terms.dot(cluster_index[i]);
      
      for(size_t j = i + 1; j < initial_popsize; ++j)
      {
        clusterj = terms.dot(cluster_index[j]);
        
        if(clusteri == clusterj) // only if the two solutions belong to the same cluster
        {
          test_solution_hash = j * initial_popsize + i;
          for(size_t k = 0; k < test_solution_hashes[test_solution_hash].size(); ++k)
          {
            if(test_solution_hashes[test_solution_hash][k]->number_of_objectives() > 0) // only if it is actually evaluated
            {
              new_clusters[cluster_hashes[clusteri]]->sols.push_back(test_solution_hashes[test_solution_hash][k]);
              new_clusters[cluster_hashes[clusteri]]->sols.back()->population_number = optimizer_number;
            }
          }
        }
      }
    }
  }
  
  
  // add only the clusters with sols
  for(size_t i = 0; i < new_clusters.size(); ++i)
  {
    subpopulations.push_back(new_clusters[i]);
    int cluster_number = (int) (subpopulations.size() - 1);
    
    for(size_t j = 0; j < subpopulations[cluster_number]->sols.size(); ++j) {
      subpopulations.back()->sols[j]->cluster_number = cluster_number;
    }
  }
  
  
}

void hicam::hvc_t::nearest_solution(const population_t & pop, const size_t current_index, const std::vector<size_t> & candidate_indices, const std::vector<vec_t> & pairwise_distance_matrix, size_t number_of_neighbours, std::vector<size_t> & neighbours)
{
  
  vec_t rank_score(candidate_indices.size());
  
  // find the solution j that maximizes Rank(d_param[j])/rank(d_obj[j])
  for (size_t j = 0; j < candidate_indices.size(); ++j)
  {
    if (candidate_indices[j] == current_index) {
      continue;
    }
    
    rank_score[j] = pairwise_distance_matrix[current_index][candidate_indices[j]]; // make ranks start at 1.
  }
  
  size_t Nn = std::min(number_of_neighbours,candidate_indices.size());
  neighbours.resize(Nn);
  for(size_t i = 0; i < Nn; ++i)
  {
    rank_score.min(neighbours[i]);
    rank_score[neighbours[i]] = 1 + (candidate_indices.size() + 1.0) * (candidate_indices.size() + 1.0); // nothing larger than this.
    neighbours[i] = candidate_indices[neighbours[i]]; // back to indices of pop.
  }
}


void hicam::hvc_t::nearest_solution(const population_t & pop, const size_t current_index, const std::vector<size_t> & candidate_indices, const std::vector<vec_t> & d_param, const std::vector<vec_t> & d_obj, size_t number_of_neighbours, std::vector<size_t> & neighbours)
{
  
  size_t N = candidate_indices.size();
  
  vec_t d_param_candidates(N);
  vec_t d_obj_candidates(N);
  
  for (size_t i = 0; i < N; ++i)
  {
    d_param_candidates[i] = d_param[current_index][candidate_indices[i]];
    d_obj_candidates[i] = d_obj[current_index][candidate_indices[i]];
  }
  
  vec_t param_ranks(N), obj_ranks(N), rank_score(N);
  vec_t param_order(N);
  vec_t obj_order(N);
  
  compute_ranks_asc(d_param_candidates,param_order);
  compute_ranks_asc(d_obj_candidates,obj_order);
  
  for(size_t j = 0; j < N; ++j)
  {
    param_ranks[param_order[j]] = j + 1.0;
    obj_ranks[obj_order[j]] = j + 1.0;
  }
  
  for (size_t i = 0; i< N; ++i) {
    rank_score[i] = (param_ranks[i] + 1.0) * (obj_ranks[i] + 1.0); // make ranks start at 1.
  }
  
  size_t Nn = std::min(number_of_neighbours,N);
  neighbours.resize(Nn);
  for(size_t i = 0; i < Nn; ++i)
  {
    rank_score.min(neighbours[i]);
    rank_score[neighbours[i]] = 1 + (N + 1.0) * (N + 1.0); // nothing larger than this.
    neighbours[i] = candidate_indices[neighbours[i]]; // back to indices of pop.
  }
}


void hicam::hvc_t::get_candidates(const population_t & pop, const size_t current_index, std::vector<size_t> & candidates)
{
  
  bool walk_left = true;
  
  candidates.clear();
  candidates.reserve(pop.size());
  
  for(size_t i = 0; i < pop.size(); ++i)
  {
    if(walk_left) {
      // walk left
      if(pop.sols[current_index]->obj[0] > pop.sols[i]->obj[0]) {
        candidates.push_back(i);
      }
      
    } else {
      // walk right
      if(pop.sols[current_index]->obj[0] < pop.sols[i]->obj[0]) {
        candidates.push_back(i);
      }
    }
  }
}

void hicam::hvc_t::get_pairwise_dist_matrix(const population_t & pop, std::vector<vec_t> & pairwise_distance_matrix, std::vector<size_t> & cluster_start_order, std::vector<vec_t> & d_param)
{
  size_t N = pop.size();
  
  d_param.resize(N, vec_t(N));
  std::vector<vec_t> d_obj(N, vec_t(N));
  vec_t d_param_order(N), d_obj_order(N);
  std::vector<vec_t> d_param_ranks(N, vec_t(N));
  std::vector<vec_t> d_obj_ranks(N, vec_t(N));
  vec_t rank_score(N), furthest_distance(N,0.0);
  
  for(size_t i = 0; i < N; ++i)
  {
    d_param[i][i] = 1000;
    d_obj[i][i] = 1000;
    
    for (size_t j = (i+1); j < N; ++j)
    {
      d_param[i][j] = (pop.sols[i]->param - pop.sols[j]->param).norm();
      d_param[j][i] = d_param[i][j];
      
      d_obj[i][j] = (pop.sols[i]->obj - pop.sols[j]->obj).norm();
      d_obj[j][i] = d_obj[i][j];
      
    }
    compute_ranks_asc(d_param[i],d_param_order);
    compute_ranks_asc(d_obj[i],d_obj_order);
    
    for(size_t j = 0; j < N; ++j)
    {
      d_param_ranks[i][d_param_order[j]] = j + 1.0;
      d_obj_ranks[i][d_obj_order[j]] = j + 1.0;
    }
    
    // find the solution j that maximizes Rank(d_param[j])/rank(d_obj[j])
    for (size_t j = 0; j < N; ++j)
    {
      if (j == i) {
        continue;
      }
      
      rank_score[j] = d_param_ranks[i][j] / d_obj_ranks[i][j];
      
      if(rank_score[j] > furthest_distance[i]) {
        furthest_distance[i] = rank_score[j];
        // furthest[i] = j;
      }
    }
  }
  
  pairwise_distance_matrix.clear();
  pairwise_distance_matrix.resize(N, vec_t(N));
  
  for(size_t i = 0; i < N; ++i)
  {
    for(size_t j = 0; j < N; ++j) {
      pairwise_distance_matrix[i][j] = (1.0 + d_param_ranks[i][j]) * (1.0 + d_obj_ranks[i][j]);
    }
    pairwise_distance_matrix[i][i] = 1.0 + (1.0 + N) * (1.0 + N); // to make it the worst for sure.
  }
  
  compute_ranks_asc(furthest_distance, cluster_start_order);
}

// this is the SO code, which assumes that the pop is sorted on fitness.
void hicam::hvc_t::cluster_ObjParamDistanceRanks(const population_t & pop, std::vector<population_pt> & subpopulations, unsigned int  & number_of_evaluations, double & average_edge_length, bool add_test_solutions, bool recheck_elites, int optimizer_number, rng_pt & rng)
{
  
  // nothing to cluster
  if (pop.size() == 0) {
    return;
  }
  
  bool use_hill_valley_test = false;
  bool keep_only_longest_path = true;
  size_t longest_path_tolerance = 1; // allows side-paths that are of at most this length to the final elitist archive.
  
  // set problem parameters
  size_t N = pop.size();
  size_t number_of_parameters = pop.problem_size();
  // size_t number_of_objectives = pop.sols[0]->number_of_objectives();
  size_t max_number_of_neighbours = get_max_number_of_neighbours(number_of_parameters);
  if(!use_hill_valley_test) { max_number_of_neighbours = 1; }
  average_edge_length = get_average_edge_length(pop, number_of_parameters);
  
  // get distance measures
  std::vector<vec_t> pairwise_distance_matrix;
  std::vector<size_t> cluster_start_order;
  std::vector<vec_t> d_param;
  get_pairwise_dist_matrix(pop, pairwise_distance_matrix, cluster_start_order, d_param);
  size_t p_start = cluster_start_order[0];
  
  // loop variables
  std::vector<bool> visited(N, false);
  size_t number_of_visited_solutions = 0;
  size_t p;
  std::vector<int> edges(N,-1);
  std::vector<size_t> edge_from, edge_to; // not used in the algorithm but only for analysis/plotting
  std::vector<size_t> temp_sols;
  subpopulations.clear();
  
  
  vec_t path_lengths(N,0);
  
  // loop untill all points all clustered
  while (number_of_visited_solutions < N)
  {
    // start point p
    p = p_start;
    visited[p] = true;
    number_of_visited_solutions++;
    
    // create a list of temp_points that are added later on to the main tree or a new cluster.
    temp_sols.clear();
    temp_sols.push_back(p);
    
    // start walking from point p, the max-path length is N.
    bool connected_to_main_tree = false;
    
    for(size_t i = 0; i < N; ++i)
    {
      bool edge_added = false;
      
      // create candidates vector
      std::vector<size_t> candidates;
      get_candidates(pop, p, candidates);
      
      // only if there are candidates, try to find a matching one
      if (candidates.size() > 0)
      {
      
        // find nearest solutions
        std::vector<size_t> nearest_list;
        nearest_solution(pop, p, candidates, pairwise_distance_matrix, max_number_of_neighbours, nearest_list);
        // nearest_solution(pop, p, candidates, d_param, d_obj, max_number_of_neighbours, nearest_list);

        for (size_t j = 0; j < nearest_list.size(); ++j)
        {
          ///////////////////////////
          /// hill-valley test
          int max_number_of_trial_solutions = 1 + ((unsigned int)(d_param[p][nearest_list[j]] / average_edge_length));
          std::vector<solution_pt> test_solutions;
          bool consider_constraints = false, rejected_because_out_of_domain = false;
          
          if (!use_hill_valley_test || MO_HillValleyTest(*pop.sols[p], *pop.sols[nearest_list[j]], max_number_of_trial_solutions, number_of_evaluations, test_solutions, consider_constraints, rejected_because_out_of_domain))
          {
            // accept edge
            edge_added = true;
            edge_from.push_back(p);
            edge_to.push_back(nearest_list[j]);
            edges[p] = (int) nearest_list[j];
            
            if(add_test_solutions) {
              std::cout << "not implemented?";
              // temp_sols.insert(temp_sols.end(), test_solutions.begin(), test_solutions.end());
            }
            
            if(visited[nearest_list[j]] == false)
            {
              temp_sols.push_back(nearest_list[j]);
              visited[nearest_list[j]] = true;
              number_of_visited_solutions++;
              p = nearest_list[j];
            }
            else
            {
              for(size_t k = 0; k < temp_sols.size(); ++k) {
                subpopulations[pop.sols[nearest_list[j]]->cluster_number]->addSolution(pop.sols[temp_sols[k]]);
              }
              subpopulations[pop.sols[nearest_list[j]]->cluster_number]->setClusterNumber(pop.sols[nearest_list[j]]->cluster_number);
              
              // backtrace path lengths
              path_lengths[temp_sols[temp_sols.size()-1]] = path_lengths[nearest_list[j]]+1;
              
              for(int k = (int) temp_sols.size()-2; k >= 0; --k) {
                path_lengths[temp_sols[k]] = path_lengths[temp_sols[k+1]] + 1;
              }
              
              temp_sols.clear();
              connected_to_main_tree = true; // breaks the double loop.
              break;
            }
            
            break;
          }
        }
        
        if (connected_to_main_tree) {
          break;
        }
      }
        
      // if no edge is added, create a new cluster.
      if (!edge_added)
      {
        edges[p] = (int) p; // self edge?
        
        subpopulations.push_back(std::make_shared<population_t>());
        
        for(size_t j = 0; j < temp_sols.size(); ++j) {
          subpopulations.back()->addSolution(pop.sols[temp_sols[j]]);
        }
        
        subpopulations.back()->setClusterNumber((int) (subpopulations.size()-1));
        
        // backtrace path lengths
        path_lengths[temp_sols[temp_sols.size()-1]] = 0;
        
        for(int k = (int) temp_sols.size()-2; k >= 0; --k) {
          path_lengths[temp_sols[k]] = path_lengths[temp_sols[k+1]] + 1;
        }
        
        temp_sols.clear();
        break;
      }
      
    } // end walking in a direction
    
    // find the next solution to visit.
    for(size_t i = 0; i < N; ++i)
    {
      if( !visited[cluster_start_order[i]])
      {
        p_start = cluster_start_order[i];
        break;
      }
    }
    
  }
  
  if(keep_only_longest_path)
  {
    population_pt subpop = std::make_shared<population_t>();
  
    size_t path_idx = 0;
    double path_length = path_lengths.max(path_idx);
    
    std::vector<bool> in_tree(N, false);
    
    for(size_t i = 0; i <= path_length; ++i)
    {
      subpop->sols.push_back(pop.sols[path_idx]);
      in_tree[path_idx] = true;
      // std::cout << path_idx+1 << ",";
      path_idx = edges[path_idx];
    }
    std::vector<bool> in_next_tree = in_tree;
    
    for(size_t i = 0; i < longest_path_tolerance; ++i)
    {
      
      for(size_t j = 0; j < N; ++j)
      {
        if(!in_tree[j] && in_tree[edges[j]])
        {
          subpop->sols.push_back(pop.sols[j]);
          in_next_tree[j] = true;
        }
      }
      
      in_tree = in_next_tree;
    }
    
    subpopulations.clear();
    subpopulations.push_back(subpop);
    
  }
  else
  {
    
    vec_t subpop_sizes(subpopulations.size(),0);
    
    for(size_t i = 0; i < subpopulations.size(); ++i) {
      subpop_sizes[i] = subpopulations[i]->size();
    }
    
    vec_t subpop_sort_order(subpopulations.size());
    std::vector<population_pt> subpop_copy = subpopulations;
    
    compute_ranks_desc(subpop_sizes, subpop_sort_order);
    
    for(size_t i = 0; i < subpop_sort_order.size(); ++i)
    {
      subpopulations[i] = subpop_copy[subpop_sort_order[i]];
    }
  }
/*
  std::cout << " sols = [" << std::endl;
  for (size_t i = 0; i < pop.sols.size(); ++i) {
    std::cout << pop.sols[i]->param[0] << " " << pop.sols[i]->param[1] << " " << pop.sols[i]->obj[0] << " " << pop.sols[i]->obj[1] << std::endl;
  }
  std::cout << "];" << std::endl;
  
  std::cout << " edges = [" << std::endl;
  for (size_t i = 0; i < edge_to.size(); ++i) {
    std::cout << edge_from[i] << " " << edge_to[i] << std::endl;
  }
  
  std::cout << "];" << std::endl;
*/

}


void hicam::hvc_t::HL_filter(const population_t & pop, population_t & new_population, double tolerance)
{

  
  new_population.sols = pop.sols;
  new_population.removeSolutionNullptrs();
  bool removed_solution = true;
  // bool first_run = true;
  
  while (removed_solution)
  {
    removed_solution = false;
    size_t N = new_population.size();
    
    // nothing to do here
    // for N <= 1, the HVI formula fails (but could be fixed)
    // but for N <= 2, removing solutions based on path length does not make sense
    if (N <= 2) {
      return;
    }
    
    std::sort(new_population.sols.begin(),new_population.sols.end(),solution_t::strictly_better_solution_via_pointers_obj0_unconstraint);

    // hypervolume
    double r0 = fitness_function->hypervolume_max_f0;
    double r1 = fitness_function->hypervolume_max_f1;
    // double HV = new_population.compute2DHyperVolumeAlreadySortedOnObj0(r0,r1);
	double HV_box = (new_population.sols[0]->obj[0] - r0) *  (new_population.sols[new_population.sols.size()-1]->obj[1] - r1);
    
    vec_t HVI(N,0); // hypervolume contribution
    vec_t dist(N-1,0); // dist[i] = distance from solution i to i+1
    vec_t L(N,0); // detour contribution
    
    HVI[0] = (fmin(r0, new_population.sols[1]->obj[0]) - new_population.sols[0]->obj[0]) * (r1 - fmin(r1, new_population.sols[0]->obj[1]));
    dist[0] = (new_population.sols[0]->param - new_population.sols[1]->param).norm();
    L[0] = dist[0];
    
    for(size_t i = 1; i < N-1; ++i)
    {
      HVI[i] = (fmin(r0, new_population.sols[i+1]->obj[0]) - new_population.sols[i]->obj[0]) * (new_population.sols[i-1]->obj[1] - fmin(r1, new_population.sols[i]->obj[1]));
      dist[i] = (new_population.sols[i]->param - new_population.sols[i+1]->param).norm();
      L[i] = dist[i-1] + dist[i] - (new_population.sols[i-1]->param - new_population.sols[i+1]->param).norm();
    }
    
    HVI[N-1] =  (r0 - new_population.sols[N-1]->obj[0]) * (new_population.sols[N-2]->obj[1] - fmin(r1, new_population.sols[N-1]->obj[1]));
    dist[N-2] = (new_population.sols[N-2]->param - new_population.sols[N-1]->param).norm();
    L[N-1] = dist[N-2];
    
    // compute ratio and eliminate points.
    // vec_t HL_ratio(N,0);
    // for(size_t i = 0; i < N; ++i) {
    //   HL_ratio[i] = HVI[i] / L[i];
    // }
    
    // print stuff
    /* if(first_run)
    {
      std::cout << " sols0 = [" << std::endl;
      for (size_t i = 0; i < N; ++i) {
        std::cout << new_population.sols[i]->param[0] << " " << new_population.sols[i]->param[1] << " " << new_population.sols[i]->obj[0] << " " << new_population.sols[i]->obj[1] << " " << HVI[i] << " " << L[i] << std::endl;
      }
      std::cout << "]; HV = " << HV << ";" << std::endl;
      first_run = false;
    } */
    
    // end print stuff
    
    ////////////////////////
    // remove stuff
    /*
    double tol = 0.0 * (HVI.mean() / L.mean());
    for (size_t i = 1; i < N-1; ++i) // do not remove the endpoints.
    {
      if(HL_ratio[i] < tol)
      {
        new_population.sols[i] = nullptr;
        removed_solution = true;
        break;
      }
    }*/
    
    double max_L = 0.0;
    int idx_max_L = -1;
    
    for(size_t i = 1; i < N-1; ++i)
    {
      if(HVI[i]/HV_box < tolerance && L[i] > max_L) {
        max_L = L[i];
        idx_max_L = (int) i;
      }
    }
    
    if( idx_max_L >= 0)
    {
      new_population.sols[idx_max_L] = nullptr;
      removed_solution = true;
    }
    
    new_population.removeSolutionNullptrs();

  }
/*
  std::cout << " sols = [" << std::endl;
  for (size_t i = 0; i < new_population.size(); ++i) {
    std::cout << new_population.sols[i]->param[0] << " " << new_population.sols[i]->param[1] << " " << new_population.sols[i]->obj[0] << " " << new_population.sols[i]->obj[1] << std::endl;
  }
  std::cout << "];" << std::endl;
*/
  
  // double pathlength = 0.0;
  
  // for(size_t i = 0; i < new_population.size()-1; ++i)
  // {
  //   pathlength += (new_population.sols[i]->param - new_population.sols[i+1]->param).norm();
  // }
  
  // std::cout << std::fixed << std::setw(6) << std::setprecision(0) << pathlength << " ";
}


void hicam::hvc_t::computeDVIRanges(const population_t & pop, vec_t & dvi_ranges) const
{
  
  if (pop.size() == 0)
    return;
  
  size_t start_index = 0;
  for (start_index = 0; start_index < pop.sols.size(); ++start_index)
  {
    if (pop.sols[start_index] != nullptr) {
      break;
    }
  }
  
  size_t number_of_dvis = pop.sols[start_index]->dvis.size();
  
  vec_t largest_dvi(number_of_dvis, -1e+308);
  vec_t smallest_dvi(number_of_dvis, 1e+308);
  
  for (size_t i = start_index; i < pop.size(); i++)
  {
    
    if (pop.sols[i] == nullptr) {
      continue;
    }
    
    for (size_t j = 0; j < number_of_dvis; j++)
    {
      
      if (pop.sols[i]->dvis[j] < smallest_dvi[j]) {
        smallest_dvi[j] = pop.sols[i]->dvis[j];
      }
      
      if (pop.sols[i]->dvis[j] > largest_dvi[j]) {
        largest_dvi[j] = pop.sols[i]->dvis[j];
      }
    }
  }
  
  dvi_ranges = largest_dvi - smallest_dvi;
  
}

// this is the SO code, which assumes that the pop is sorted on fitness.
void hicam::hvc_t::cluster_on_dvis(const population_t & pop, std::vector<population_pt> & subpopulations, unsigned int  & number_of_evaluations, double & average_edge_length, bool add_test_solutions, bool recheck_elites, int optimizer_number, rng_pt & rng)
{
  
  // nothing to cluster
  if (pop.size() == 0) {
    return;
  }
  
  // set problem parameters
  size_t initial_popsize = pop.size();
  size_t number_of_dvis = pop.sols[0]->dvis.size();
  size_t number_of_objectives = pop.sols[0]->number_of_objectives();
  
  // Hill-Valley Parameters
  vec_t dvi_ranges;
  computeDVIRanges(pop, dvi_ranges);
  
  // fewer roundoff errors
  average_edge_length = pow(initial_popsize, -1.0/number_of_dvis);
  for(size_t i = 0; i < dvi_ranges.size(); ++i) {
    average_edge_length *= pow(dvi_ranges[i],1.0/number_of_dvis);
  }
  
  if(isinf(average_edge_length) || average_edge_length > 1e300)
  {
    average_edge_length = 0.01;
    
    if(pop.previous != nullptr && pop.previous->clusters.size() > 0) {
      average_edge_length = pop.previous->clusters[0]->init_bandwidth;
    }
  }
  
  // such that for d={1,2} still Nn = d+1
  size_t max_number_of_neighbours;
  
  //if( number_of_dvis <= 2 ) {
    max_number_of_neighbours = number_of_dvis + 1;
  //}
  //else {
  //  max_number_of_neighbours = 2 + log(number_of_dvis);
  //}
  
  
  vec_t dist(initial_popsize,0.0);
  
  bool edge_added = false;
  unsigned int max_number_of_trial_solutions;
  
  // number clusters
  vec_t number_of_clusters(number_of_objectives, 1);
  std::vector<vec_t> cluster_index(initial_popsize);
  
  for(size_t i =0 ; i < initial_popsize; ++i) {
    cluster_index[i].resize(number_of_objectives);
  }
  
  // adding test solutions to the clusters
  // we store all test solutions for sol i and sol j in test_solutions[i][j] with i < j.
  size_t test_solution_hash, mini, maxi; // i * initial_popsize + j for (i > j)
  std::map<size_t,std::vector<solution_pt>> test_solution_hashes;
  /* std::vector<std::vector<std::vector<solution_pt>>> test_solutions(initial_popsize);
   for(size_t i = 0 ; i < initial_popsize; ++i) {
   test_solutions[i].resize(initial_popsize);
   } */
  
  bool consider_constraints = true;
  
  for(size_t objective_number = 0; objective_number < number_of_objectives; ++objective_number)
  {
    
    // compute population ranks for current objective_number
    std::vector<size_t> sols_sorted_on_fitness(initial_popsize);
    vec_t collected_fitness(initial_popsize);
    
    
    if(consider_constraints)
    {
      for(size_t i = 0; i < initial_popsize; ++i) {
        collected_fitness[i] = pop.sols[i]->obj[objective_number] + pop.sols[i]->constraint * 1000;
      }
    }
    else
    {
      for(size_t i = 0; i < initial_popsize; ++i) {
        collected_fitness[i] = pop.sols[i]->obj[objective_number];
      }
    }
    compute_ranks_asc(collected_fitness, sols_sorted_on_fitness); // oops, i hardcoded that we do minimization. assumed
    
    // pop.getSingleObjectiveRanks(fitness_ranks);
    
    // The best is the first cluster.
    cluster_index[sols_sorted_on_fitness[0]][objective_number] = 0;
    
    for (size_t i = 1; i < initial_popsize; i++)
    {
      
      // compute the distance to all better solutions.
      dist[i] = 0.0;
      size_t nearest_better_index = 0;
      size_t furthest_better_index = 0;
      size_t old_nearest_better_index = 0;
      
      for (size_t j = 0; j < i; j++) {
        // dist[j] = pop.sols[sols_sorted_on_fitness[i]]->param_distance(*pop.sols[sols_sorted_on_fitness[j]]);
        dist[j] = (pop.sols[sols_sorted_on_fitness[i]]->dvis - pop.sols[sols_sorted_on_fitness[j]]->dvis).norm();
        
        if (dist[j] < dist[nearest_better_index]) {
          nearest_better_index = j;
        }
        
        if (dist[j] > dist[furthest_better_index]) {
          furthest_better_index = j;
        }
      }
      
      edge_added = false;
      
      // check the few nearest solutions with better fitness
      for (size_t j = 0; j < std::min(i, max_number_of_neighbours); j++)
      {
        
        // find the next-to nearest index
        if (j > 0)
        {
          old_nearest_better_index = nearest_better_index;
          nearest_better_index = furthest_better_index;
          
          for (size_t k = 0; k < i; k++)
          {
            if (dist[k] > dist[old_nearest_better_index] && dist[k] < dist[nearest_better_index]) {
              nearest_better_index = k;
            }
          }
          
        }
        
        if (!recheck_elites)
        {
          // if both are elites that belong to the same archive, accept edge.
          // if both are elites that belong to different archives, reject edge.
          
          if( pop.sols[sols_sorted_on_fitness[i]]->elite_origin != nullptr && pop.sols[sols_sorted_on_fitness[nearest_better_index]]->elite_origin != nullptr)
          {
            
            if(pop.sols[sols_sorted_on_fitness[i]]->elite_origin ==  pop.sols[sols_sorted_on_fitness[nearest_better_index]]->elite_origin)
            {
              // accept edge
              cluster_index[sols_sorted_on_fitness[i]][objective_number] = cluster_index[sols_sorted_on_fitness[nearest_better_index]][objective_number];
              edge_added = true;
              break;
            }
            else
            {
              // reject edge
              // double check them, so that we combine more clusters.
              // continue;
            }
          }
        }
        
        // set budget
        max_number_of_trial_solutions = 1 + ((unsigned int)(dist[nearest_better_index] / average_edge_length));
        
        // such that mini < maxi
        mini = std::min(sols_sorted_on_fitness[nearest_better_index], sols_sorted_on_fitness[i]);
        maxi = std::max(sols_sorted_on_fitness[nearest_better_index], sols_sorted_on_fitness[i]);
        test_solution_hash = maxi * initial_popsize + mini;
        
        // there are no pre-saved test_solutions yet
        if(test_solution_hashes.find(test_solution_hash) == test_solution_hashes.end()) {
          test_solution_hashes[test_solution_hash] = {};
        }
        
        bool rejected_because_out_of_domain = false;
        
        if (SO_HillValleyTest(*pop.sols[mini], *pop.sols[maxi], objective_number, max_number_of_trial_solutions, number_of_evaluations, test_solution_hashes[test_solution_hash], consider_constraints, rejected_because_out_of_domain))
        {
          cluster_index[sols_sorted_on_fitness[i]][objective_number] = cluster_index[sols_sorted_on_fitness[nearest_better_index]][objective_number];
          edge_added = true;
          break;
        }
        
        if (rejected_because_out_of_domain)
        {
          // its out-of-domain, its not a neighbour. consider the next.
          max_number_of_neighbours++;
        }
      }
      
      
      // its a new clusters, label it like that.
      if (!edge_added)
      {
        cluster_index[sols_sorted_on_fitness[i]][objective_number] = number_of_clusters[objective_number];
        number_of_clusters[objective_number]++;
      }
    }
  }
  
  // create & fill the clusters
  //---------------------------------------------------------------------------
  // map cluster indices:
  // n0 = 0,...,N0 - 1
  // n1 = 0,...,N1 - 1
  // n2 = 0,...,N2 - 1
  // current_cluster_index = N0*N1*n2 + N0*n1 + n0
  
  vec_t terms(number_of_objectives,1.0);
  
  for(size_t j = 0; j < number_of_objectives; ++j)
  {
    for(size_t subj = 0; subj < j; ++subj) {
      terms[j] *= number_of_clusters[subj];
    }
    
  }
  
  std::vector<population_pt> new_clusters;
  std::map<int, size_t> cluster_hashes;
  
  int current_hash;
  for (size_t i = 0; i < initial_popsize; ++i)
  {
    current_hash = (int) terms.dot(cluster_index[i]);
    
    // if the hash is not found yet, add it and create corresponding new cluster.
    if(cluster_hashes.find(current_hash) == cluster_hashes.end())
    {
      cluster_hashes[current_hash] = new_clusters.size();
      new_clusters.push_back(std::make_shared<population_t>());
    }
    
    new_clusters[cluster_hashes[current_hash]]->sols.push_back(pop.sols[i]);
  }
  
  // now i want to add the test solutions to the clusters.
  if( add_test_solutions )
  {
    int clusteri, clusterj;
    for(size_t i = 0; i < initial_popsize; ++i)
    {
      clusteri = terms.dot(cluster_index[i]);
      
      for(size_t j = i + 1; j < initial_popsize; ++j)
      {
        clusterj = terms.dot(cluster_index[j]);
        
        if(clusteri == clusterj) // only if the two solutions belong to the same cluster
        {
          test_solution_hash = j * initial_popsize + i;
          for(size_t k = 0; k < test_solution_hashes[test_solution_hash].size(); ++k)
          {
            if(test_solution_hashes[test_solution_hash][k]->number_of_objectives() > 0) // only if it is actually evaluated
            {
              new_clusters[cluster_hashes[clusteri]]->sols.push_back(test_solution_hashes[test_solution_hash][k]);
              new_clusters[cluster_hashes[clusteri]]->sols.back()->population_number = optimizer_number;
            }
          }
        }
      }
    }
  }
  
  
  // add only the clusters with sols
  for(size_t i = 0; i < new_clusters.size(); ++i)
  {
    subpopulations.push_back(new_clusters[i]);
    int cluster_number = (int) (subpopulations.size() - 1);
    
    for(size_t j = 0; j < subpopulations[cluster_number]->sols.size(); ++j) {
      subpopulations.back()->sols[j]->cluster_number = cluster_number;
    }
  }
  
  
}
