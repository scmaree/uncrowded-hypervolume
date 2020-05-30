/*

HICAM Multi-objective

By S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "elitist_archive.h"
#include "optimizer.h"
#include "mathfunctions.h"
#include "hillvalleyclustering.h"

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <math.h>



hicam::elitist_archive_t::elitist_archive_t(size_t target_size, rng_pt rng) : population_t()
{
  this->target_size = target_size;
  this->use_parameter_distances = false;
  this->use_greedy_selection = false;
  this->objective_discretization_in_effect = false;
  this->parameter_discretization_in_effect = false;
  this->use_hypervolume_for_size_control = false;
  this->rng = rng;
  this->use_niching = false;
  
  this->sols.reserve(1.5*target_size);
}

hicam::elitist_archive_t::~elitist_archive_t() {}



hicam::elitist_archive_t::elitist_archive_t(const elitist_archive_t & other) : population_t(other)
{
  this->target_size = other.target_size;
  this->use_parameter_distances = other.use_parameter_distances;
  this->use_greedy_selection = other.use_greedy_selection;
  this->use_niching = other.use_niching;
  
  this->objective_discretization_in_effect = other.objective_discretization_in_effect;
  this->parameter_discretization_in_effect = other.parameter_discretization_in_effect;
  this->objective_discretization = other.objective_discretization;
  this->parameter_discretization = other.parameter_discretization;
  // this->best_objective_values_in_elitist_archive = other.best_objective_values_in_elitist_archive;
  this->rng = other.rng;
  
  // Copy each of hte clustered archives.
  this->clustered_archive.resize(other.clustered_archive.size());
  
  for(size_t i = 0; i < other.clustered_archive.size(); ++i) {
    this->clustered_archive[i] = std::make_shared<elitist_archive_t>(*other.clustered_archive[i]);
  }

}


/**
* Adds a solution to the elitist archive.
*/
void hicam::elitist_archive_t::addToArchive(const solution_pt sol, size_t & insert_index, bool make_copy_of_sol)
{

  assert(sol != nullptr);
  
  // to prevent gaps please.
  if (insert_index > sols.size()) {
    insert_index = sols.size();
  }

  if (insert_index == sols.size())
  {
    
    if(make_copy_of_sol) {
      sols.push_back(std::make_shared<solution_t>(*sol));
      sols.back()->elite_origin = nullptr;
    }
    else {
      sols.push_back(sol);
    }
    
  }
  else
  {
    
    if(make_copy_of_sol) {
      sols[insert_index] = std::make_shared<solution_t>(*sol);
      sols[insert_index]->elite_origin = nullptr;
    }
    else {
      sols[insert_index] = sol;
    }
    
  }
  

  // note, if all solutions have a constraint value, the archive consists of the solution with the lowest constraint value,
  // and is always size 1.
  if (sols.size() == 1)
  {
    best = solution_t(*sol);
  } 
  else
  {
    
    // in this case, there are solutions with zero constraint.
    for (size_t j = 0; j < sol->number_of_objectives(); j++)
    {
      if (sol->strictly_better_than_unconstraint_per_objective(best, j)) {
        best.obj[j] = sol->obj[j];
        if(sol->use_lex) {
          best.lex_obj[j] = sol->lex_obj[j];
        }
      }
    }
  }
  
}

void hicam::elitist_archive_t::removeFromArchive(size_t sol_index)
{
  sols[sol_index] = nullptr;
}

// Update the Archive
// Updates the elitist archive by offering a new solution to possibly be added to the archive.
// If there are no solutions in the archive yet, the solution is added.
// Otherwise, the number of times the solution is dominated is computed.
//  Solution A is always dominated by solution B that is in the same domination-box if B dominates A or A and B do not dominate each other.

int hicam::elitist_archive_t::updateArchive(const solution_pt sol)
{
  return updateArchive(sol, false);
}

int hicam::elitist_archive_t::updateArchive(const solution_pt sol, bool make_copy_of_sol)
{
  
  int added_index = -1;
  
  if(sol == nullptr) {
    return added_index;
  }
  
  bool is_extreme_compared_to_archive = false;
  
  size_t insert_index = sols.size();

  if (sol->constraint == 0)
  {
    if (sols.size() == 0) {
      is_extreme_compared_to_archive = true;
    }
    else
    {
      for (size_t i = 0; i < sol->number_of_objectives(); ++i)
      {
        if (sol->strictly_better_than_unconstraint_per_objective(best, i)) {
          is_extreme_compared_to_archive = true;
          break;
        }
      }
    }
  }

  // if the archive is empty, just insert the solution.
  if (sols.size() == 0)
  {
    addToArchive(sol, insert_index, make_copy_of_sol);
    added_index = (int) insert_index;
  }
  else
  {

    bool is_dominated_itself = false;
    bool all_to_be_removed = true;

    for (size_t i = 0; i < sols.size(); i++)
    {
      // we replace the first 'empty' spot in the archive.
      if (sols[i] == nullptr)
      {
        if (i < insert_index) {
          insert_index = i;
        }

        continue;
      }

      all_to_be_removed = false;
      if (sols[i]->better_than(*sol))
      {
        is_dominated_itself = true;
      }
      else
      {
        if (!sol->better_than(*sols[i]))
        {
          // we want only one solution per objective box
          // so if it is not better than the solution in the box,
          // we still discard it.
          if (!is_extreme_compared_to_archive) {
            if (sameObjectiveBox(*sols[i], *sol))
            {
              is_dominated_itself = true;
            }
          }
        }
      }

      if (is_dominated_itself) {
        break;
      }
    }

    // if all in the elitist archive are nullpointers, insert_index is set to 0 above, and
    // we can add the solution to the archive.
    if (all_to_be_removed) {
      addToArchive(sol, insert_index, make_copy_of_sol);
      added_index = (int) insert_index;
    }
    else
    {
      // don't add dominated solutions.
      if (!is_dominated_itself)
      {
        for (size_t i = 0; i < sols.size(); i++)
        {
          if (sols[i] == nullptr) {
            continue;
          }
          if (sol->better_than(*sols[i]) || (sameObjectiveBox(*sols[i], *sol)))
          {
            // if the to-be-removed solution is the best in the archive, update the best. 
            for (size_t j = 0; j < sol->number_of_objectives(); j++)
            {
              if (sols[i]->obj[j] == best.obj[j]) {
                best.obj[j] = sol->obj[j];
                
                if(sol->use_lex) {
                  best.lex_obj[j] = sol->lex_obj[j];
                }
              }
            }

            removeFromArchive(i);

          }
        }

        addToArchive(sol, insert_index, make_copy_of_sol);
        added_index = (int) insert_index;
      }

    }
  }
  
  return added_index;
  
}

// returns true if two solutions are in the same discretization box
bool hicam::elitist_archive_t::sameObjectiveBox(const solution_t & sol1, const solution_t & sol2) const
{

  // when using parameter distances, the objective discretization is ignored
  if(!use_parameter_distances)
  {
    if (!objective_discretization_in_effect)
    {
      // If the solutions are identical, they are still in the (infinitely small) same objective box.
      for (size_t i = 0; i < sol1.number_of_objectives(); i++)
      {
        if (sol1.obj[i] != sol2.obj[i])
          return false;
      }

      return true;
    }

    for (size_t i = 0; i < sol1.number_of_objectives(); i++)
    {

      int box1 = (int)(sol1.obj[i] / objective_discretization[i]);
      int box2 = (int)(sol2.obj[i] / objective_discretization[i]);

      if (box1 != box2) {
        return false;
      }
    }

    return true;
  }
  else
  {
    return false;
  }
  
}

void hicam::elitist_archive_t::clear()
{
  if(use_niching)
  {
    for(size_t c = 0; c < clustered_archive.size(); ++c) {
      if(clustered_archive[c] != nullptr) {
        clustered_archive[c]->clear();
      }
    }
    clustered_archive.clear();
  }
  else
  {
    sols.clear();
  }
}


/**
 * Discard similar solutions when the archive  archive needs to be filtered for
 *
 */
void hicam::elitist_archive_t::adaptArchiveSize()
{
  
  if(use_niching)
  {
    // niching
    if(use_greedy_selection) {
      adaptSizeBygreedyScatteredSubsetSelection_mm();
    }
    else {
      adaptObjectiveDiscretization_mm();
    }
  }
  else
  {
    // not niching
    if(use_hypervolume_for_size_control)
    {
      reduceArchiveSizeByHSS(target_size, hypervolume_reference_point[0], hypervolume_reference_point[1]);
    }
    else
    {
      if(use_greedy_selection) {
        adaptSizeBygreedyScatteredSubsetSelection();
      }
      else {
        adaptObjectiveDiscretization();
      }
    }
  }
  
}


/**
* Adapts the objective box discretization. If the numbre of solutions in the elitist archive is too high or too low
* compared to the population size, the objective box discretization is adjusted accordingly. In doing so, the
* entire elitist archive is first emptied and then refilled.
*/
void hicam::elitist_archive_t::adaptObjectiveDiscretization()
{
  size_t elitist_archive_size_target_lower_bound = (size_t)(0.75*target_size);
  size_t elitist_archive_size_target_upper_bound = (size_t)(1.25*target_size); // must be at least 1

  // disable the discretization if the archive is too small
  if (sols.size() < elitist_archive_size_target_lower_bound)
  {
    objective_discretization_in_effect = false;
    
    for(size_t c = 0; c < clustered_archive.size(); ++c) {
      clustered_archive[c]->objective_discretization_in_effect = false;
    }
  }

  if(sols.size() > elitist_archive_size_target_upper_bound){
    removeSolutionNullptrs();
  }
  
  // if the archive size crosses the upperbound, adapt the discretization and dump solutions
  if (sols.size() > elitist_archive_size_target_upper_bound)
  {

    // std::cout << "Achive size reduced by objective space discretization." << std::endl;
    vec_t objective_ranges;
    objectiveRanges(objective_ranges);
    
    objective_discretization_in_effect = true;

    int na = 1;
    int nb = (int)pow(2.0, 25.0);
    
    std::vector<solution_pt> archive_copy = sols;
    
    for (size_t k = 0; k < 25; k++)
    {
      int nc = (na + nb) / 2;

      objective_discretization = objective_ranges / ((double)nc);

      // Clear the elitist archive
      sols.clear();

      // Rebuild the elitist archive 
      for (size_t i = 0; i < archive_copy.size(); i++) {
        updateArchive(archive_copy[i]);
      }

      if (sols.size() <= elitist_archive_size_target_lower_bound) {
        na = nc;
      }
      else {
        nb = nc;
      }

    }

  }
}


bool hicam::elitist_archive_t::solutionHasImproved(const hicam::solution_t & sol) const
{
  
  if (sols.size() == 0)
    return true;
  
  bool result = false;

  // if it is better than the best in a single objective
  if (sol.constraint == 0)
  {
    for (size_t j = 0; j < sol.number_of_objectives(); j++)
    {
      if (sol.strictly_better_than_unconstraint_per_objective(best, j))
      {
        result = true;
        break;
      }
    }
  }

  // if not, it must dominate a solution
  if (result != true)
  {
    result = true;
    // for each sol in the archive,
    for (size_t i = 0; i < sols.size(); i++)
    {

      if (sols[i] == nullptr) {
        continue;
      }

      // if the solution in the archive is better than the solution at hand, it has not improved the archive.
      if (sols[i]->better_than(sol))
      {
        result = false;
        break;
      }
      else if (!sol.better_than(*sols[i]))
      {
        if (sameObjectiveBox(*sols[i],sol))
        {
          result = false;
          break;
        }
      }
    }
  }

  return result;
}


void hicam::elitist_archive_t::adaptSizeBygreedyScatteredSubsetSelection()
{
  size_t target_upper_bound = (size_t)(target_size); // must be at least 1
  
  if(size() > target_upper_bound) {
    removeSolutionNullptrs();
  }
  
  if(size() > target_upper_bound)
  {
    
    std::vector<solution_pt> new_sols;
    
    for(size_t i =0; i < sols.size(); ++i)
    {
      if(sols[i] != nullptr) {
        new_sols.push_back(sols[i]);
      }
    }
    
    sols = new_sols; // removes all nullptrs
    
    std::vector<solution_pt> selected_sols;
  
    if(use_parameter_distances) {
      
      // std::cout << "Achive size reduced by greedy parameter subset selection" << std::endl;
      selectSolutionsBasedOnParameterDiversity(sols, target_size, selected_sols, rng);
    }
    else {
      
      // std::cout << "Achive size reduced by greedy objective subset selection" << std::endl;
      vec_t objective_ranges;
      objectiveRanges(objective_ranges);
      
      selectSolutionsBasedOnObjectiveDiversity(sols, target_size, selected_sols, objective_ranges, rng);
    }
    
    sols = selected_sols;

  }
  
}

double hicam::elitist_archive_t::get_best_objective_values_in_elitist_archive(size_t obj_index)
{
  
  if(this->size() == 0) {
    std::cout << "Size of empty archive requested" << std::endl;
    return NAN;
  }
  
  // compute best objectives.
  
  if(use_niching)
  {
    best = clustered_archive[0]->best;
    
    // niching, collect best objectives
    for(size_t i = 0; i < clustered_archive.size(); ++i)
    {
      
      for (size_t j = 0; j < best.number_of_objectives(); j++)
      {
        // I do !strictly better because best_obj is a vector, not a solution, so i cannot call the comparator the other way around
        if (!best.strictly_better_than_unconstraint_per_objective(clustered_archive[i]->best, j)) {
          best.obj[j] = clustered_archive[i]->best.obj[j];
          
          if(clustered_archive[i]->best.use_lex) {
            best.lex_obj[j] = clustered_archive[i]->best.lex_obj[j];
          }
          
        }
      }
    }
  }
  else
  {
    best.obj.resize(0);
    
    for(size_t i = 0; i < sols.size(); ++i)
    {
      if(sols[i] != nullptr)
      {
        if(best.number_of_objectives() == 0)
        {
          best = solution_t(*sols[i]);
        }
        
        for (size_t j = 0; j < best.number_of_objectives(); j++)
        {
          if (sols[i]->strictly_better_than_unconstraint_per_objective(best, j)) {
            best.obj[j] = sols[i]->obj[j];
            if(sols[i]->use_lex) {
              best.lex_obj[j] = sols[i]->lex_obj[j];
            }
          }
        }
      }
      
    }
  }

  return best.obj[obj_index];;

}

hicam::elitist_archive_pt hicam::elitist_archive_t::initNewArchive() const
{
  elitist_archive_pt new_archive = std::make_shared<elitist_archive_t>(target_size, rng);
  new_archive->target_size = target_size;
  new_archive->use_parameter_distances = use_parameter_distances;
  new_archive->use_greedy_selection = use_greedy_selection;
  new_archive->parameter_discretization_in_effect = parameter_discretization_in_effect;
  new_archive->objective_discretization_in_effect = objective_discretization_in_effect;
  new_archive->parameter_discretization = parameter_discretization;
  new_archive->objective_discretization = objective_discretization;
  new_archive->use_niching = false;
  new_archive->best = best;
  
  return new_archive;
}



void hicam::elitist_archive_t::addArchive(elitist_archive_pt elitist_archive)
{
  clustered_archive.push_back(elitist_archive);
}

void hicam::elitist_archive_t::removeEmptyClusters()
{
  
  std::vector<elitist_archive_pt> temp;
  
  for(size_t i =0 ; i < clustered_archive.size(); ++i)
  {
    if(clustered_archive[i] != nullptr && clustered_archive[i]->sols.size() != 0)
    {
      temp.push_back(clustered_archive[i]);
    }
  }
  
  clustered_archive = temp;
  
}


size_t hicam::elitist_archive_t::size() const {
  
  if(!use_niching)
  {
    return sols.size();
  }
  else
  {
    size_t total_size = 0;
    
    for(size_t i =0 ; i < clustered_archive.size(); ++i)
      total_size += clustered_archive[i]->size();
    
    return total_size;
  }
  
  
}

size_t hicam::elitist_archive_t::actualSize() const {
  
  size_t n = 0;
  if(!use_niching)
  {
    for(size_t i = 0 ; i < sols.size(); ++i) {
      if(sols[i] != nullptr) {
        n++;
      }
    }
  }
  else
  {

    for(size_t i =0 ; i < clustered_archive.size(); ++i) {
      for(size_t j = 0; j < clustered_archive[i]->size(); ++j) {
        if(clustered_archive[i]->sols[j] != nullptr) {
          n++;
        }
      }
    }
  }
  
  return n;
  
  
}

size_t hicam::elitist_archive_t::number_of_clusters() const
{
  return clustered_archive.size();
}





// multi-modal version of the objectivespace discretization
// we use the same discretization for all archives
void hicam::elitist_archive_t::adaptObjectiveDiscretization_mm()
{
  size_t elitist_archive_size_target_lower_bound = (size_t)(0.75*target_size);
  size_t elitist_archive_size_target_upper_bound = (size_t)(1.25*target_size); // must be at least 1
  
  // Get All Sols
  size_t archive_size = actualSize();
  
  // disable the discretization if the archive is too small
  if (archive_size < elitist_archive_size_target_lower_bound)
  {
    objective_discretization_in_effect = false;
    
    for(size_t i = 0; i < clustered_archive.size(); ++i) {
      clustered_archive[i]->objective_discretization_in_effect = false;
    }
  }
  
  // if the archive size crosses the upperbound, adapt the discretization and dump solutions
  if (archive_size > elitist_archive_size_target_upper_bound)
  {

    vec_t objective_ranges;
    objectiveRanges(objective_ranges);

    // per cluster, set the discretization
    for(size_t i = 0; i < clustered_archive.size(); ++i) {
      clustered_archive[i]->objective_discretization_in_effect = true;
    }
    
    objective_discretization_in_effect = true;
    
    int na = 1;
    int nb = (int)pow(2.0, 25.0);
    
    std::vector<std::vector<solution_pt>> archive_copy(clustered_archive.size());
    
    for(size_t c = 0; c < clustered_archive.size(); ++c) {
      archive_copy[c] = clustered_archive[c]->sols;
    };
    
    for (size_t k = 0; k < 25; k++)
    {
      int nc = (na + nb) / 2;
      
      objective_discretization = objective_ranges / ((double)nc);
      
      for(size_t c = 0; c < clustered_archive.size(); ++c)
      {

        // set discretization
        clustered_archive[c]->objective_discretization = objective_discretization;
        clustered_archive[c]->sols.clear();
        
        // Rebuild the elitist archive
        for (size_t i = 0; i < archive_copy[c].size(); i++) {
          clustered_archive[c]->updateArchive(archive_copy[c][i]);
        }
      }
      
      archive_size = actualSize();
      
      // std::cout << archive_size << ",";
      
      if (archive_size <= elitist_archive_size_target_lower_bound) {
        na = nc;
      }
      else {
        nb = nc;
      }
      
      
    }
    
    
    // std::cout << std::endl;

    
  }
  
}

void hicam::elitist_archive_t::adaptSizeBygreedyScatteredSubsetSelection_mm()
{
  size_t target_upper_bound = (size_t)(1.25*target_size);
  
  if(size() > target_upper_bound){
    removeSolutionNullptrs();
  }
  
  if(size() > target_upper_bound)
  {
    
    population_t new_sols;
    std::vector<elitist_archive_pt> sol_origin;
    getAllSols(new_sols.sols, sol_origin);
    
    if(new_sols.sols.size() > target_upper_bound)
    {
      
      std::vector<size_t> selected_indices;
    
      if(use_parameter_distances)
      {
        // convert to the right format
        std::vector<vec_t> parameters(new_sols.sols.size());
        
        for (size_t i = 0; i < new_sols.sols.size(); ++i) {
          parameters[i] = new_sols.sols[i]->param;
        }
        
        greedyScatteredSubsetSelection(parameters, (int)(0.75*target_size), selected_indices, rng);

      }
      else
      {
        vec_t objective_ranges;
        objectiveRanges(objective_ranges);
        
        // we scale the objectives to the objective ranges
        // before performing subset selection
        std::vector<vec_t> scaled_objectives(new_sols.sols.size());

        for (size_t i = 0; i < new_sols.sols.size(); ++i)
        {
          scaled_objectives[i].resize(new_sols.sols[i]->number_of_objectives());
          
          for(size_t j =0; j < new_sols.sols[i]->number_of_objectives(); ++j) {
            scaled_objectives[i][j] = new_sols.sols[i]->obj[j] / objective_ranges[j];
          }
          
        }
        
        greedyScatteredSubsetSelection(scaled_objectives, (int)target_size, selected_indices, rng);

      }
      

      for(size_t j = 0; j < clustered_archive.size(); ++j)
      {
        clustered_archive[j]->sols.clear(); // withing all nullptr's.
      }
        
      
      for(size_t i = 0; i < selected_indices.size(); ++i) {
        sol_origin[selected_indices[i]]->sols.push_back(new_sols.sols[selected_indices[i]]);
      }
    }
  }
  
}



/**
 * Elitism: copies at most 1/k*tau*n solutions per cluster
 * from the elitist archive.
 */
void hicam::elitist_archive_t::copyElitesToClusters(std::vector<cluster_pt> & clusters, size_t max_number_of_elites, const vec_t & objective_ranges, rng_pt rng) const
{
  
  for (size_t i = 0; i < clusters.size(); i++) {
    clusters[i]->average_fitness(clusters[i]->objective_mean);
    clusters[i]->elites.clear();
    clusters[i]->elites.reserve(2* max_number_of_elites);
  }
  
  double distance, distance_smallest;
  
  // divide elites over the clusters
  for (size_t i = 0; i < sols.size(); ++i)
  {
    
    if (sols[i] == nullptr) {
      continue;
    }
    
    distance_smallest = -1;
    size_t j_min = 0;
    for (size_t j = 0; j < clusters.size(); ++j)
    {
      
      distance = sols[i]->transformed_objective_distance(clusters[j]->objective_mean, objective_ranges);
      if ((distance_smallest < 0) || (distance < distance_smallest))
      {
        j_min = j;
        distance_smallest = distance;
      }
      
    }
    
    clusters[j_min]->elites.push_back(sols[i]);
    
    
  }
  
  // if there are more than 'max' elites, do a diversity based selection.
  for (size_t i = 0; i < clusters.size(); i++)
  {
    if (clusters[i]->elites.size() > max_number_of_elites)
    {
      std::vector<solution_pt> selected_elites;
      selectSolutionsBasedOnObjectiveDiversity(clusters[i]->elites, max_number_of_elites, selected_elites,  objective_ranges, rng);
      
      clusters[i]->elites = selected_elites;
      
    }
  }
  
}



/**
 * Elitism: copies at most 1/k*tau*n solutions per cluster
 * from the elitist archive.
 */
size_t hicam::elitist_archive_t::addElitesToPopulation(population_t & population, int max_number_of_elites)
{
  
  size_t popsize = population.size();
  
  if(max_number_of_elites < 0)
  {
    if(use_niching)
    {
      for (size_t i = 0; i < clustered_archive.size(); i++)
      {
        size_t old_popsize = population.size();
        population.addCopyOfSolutions(clustered_archive[i]->sols);
        
        for(size_t j = old_popsize; j < population.size(); ++j) {
          population.sols[j]->elite_origin = clustered_archive[i];
        }
      }
    }
    else
    {
      population.addCopyOfSolutions(sols);
    }
  }
  else
  {
    if(use_niching)
    {
      // if there are more than 'max' elites, do a diversity based selection.
      size_t max_number_of_elites_per_archive = std::max((size_t) 1, max_number_of_elites / clustered_archive.size());
      
      // Add elites to the population
      vec_t objective_ranges;
      objectiveRanges(objective_ranges);
      
      for (size_t i = 0; i < clustered_archive.size(); i++)
      {
        std::vector<solution_pt> selected_elites;
        selectSolutionsBasedOnObjectiveDiversity(clustered_archive[i]->sols, max_number_of_elites_per_archive, selected_elites, objective_ranges, rng);
        
        size_t init_popsize = population.size();
        population.addCopyOfSolutions(selected_elites);
        
        for(size_t j = init_popsize; j < init_popsize + selected_elites.size(); ++j) {
          population.sols[j]->elite_origin = clustered_archive[i];
        }
        
      }
    } // end use_niching
    else
    {
      std::vector<solution_pt> selected_elites;
      // computeObjectiveRanges();
      // selectSolutionsBasedParameterDiversity(sols, max_number_of_elites, selected_elites, rng);
      selectSolutionsBasedOnParameterDiversity(sols, max_number_of_elites, selected_elites, rng);
      
      population.addCopyOfSolutions(selected_elites);
    }
  }
  
  return (population.size() - popsize);
  
}

void hicam::elitist_archive_t::getAllSols(std::vector<solution_pt> & all_sols)
{
  
  all_sols.clear();
  
  if(use_niching)
  {
  
  for(size_t j = 0; j < clustered_archive.size(); ++j)
  {
    all_sols.reserve(all_sols.size() + clustered_archive.size());
    
    std::vector<solution_pt> temp_new_sols;
    temp_new_sols.reserve(clustered_archive[j]->sols.size());
    
    for(size_t i =0; i < clustered_archive[j]->sols.size(); ++i)
    {
      if(clustered_archive[j]->sols[i] != nullptr)
      {
        temp_new_sols.push_back(clustered_archive[j]->sols[i]);
        clustered_archive[j]->sols[i]->cluster_number = (int) j;
      }
    }
    
    clustered_archive[j]->sols = temp_new_sols; // removing all nullptr's.
    
    for(size_t i = 0; i < temp_new_sols.size(); ++i)  {
      all_sols.push_back(temp_new_sols[i]);
    }
  }
  }
  else
  {
    all_sols.reserve(sols.size());
    
    for(size_t i =0; i < sols.size(); ++i)
    {
      if(sols[i] != nullptr)
      {
        all_sols.push_back(sols[i]);
        sols[i]->cluster_number = 0;
      }
    }
    
    sols = all_sols; // removing all nullptr's.

  }

}


void hicam::elitist_archive_t::getAllSols(std::vector<solution_pt> & all_sols, std::vector<elitist_archive_pt> & origin)
{
  
  all_sols.clear();
  origin.clear();
  
  if(use_niching)
  {
    
    for(size_t j = 0; j < clustered_archive.size(); ++j)
    {
      all_sols.reserve(all_sols.size() + clustered_archive.size());
      origin.reserve(origin.size() + clustered_archive.size());
      
      std::vector<solution_pt> temp_new_sols;
      temp_new_sols.reserve(clustered_archive[j]->sols.size());
      
      for(size_t i =0; i < clustered_archive[j]->sols.size(); ++i)
      {
        if(clustered_archive[j]->sols[i] != nullptr)
        {
          temp_new_sols.push_back(clustered_archive[j]->sols[i]);
          origin.push_back(clustered_archive[j]);
        }
      }
      
      clustered_archive[j]->sols = temp_new_sols; // removing all nullptr's.
      
      for(size_t i = 0; i < temp_new_sols.size(); ++i)  {
        all_sols.push_back(temp_new_sols[i]);
      }
    }
  }
  else
  {
    all_sols.reserve(sols.size());
    origin.reserve(sols.size());
    
    for(size_t i =0; i < sols.size(); ++i)
    {
      if(sols[i] != nullptr)
      {
        all_sols.push_back(sols[i]);
        //origin.push_back(*this); // todo
        assert(false);
      }
    }
    
    sols = all_sols; // removing all nullptr's.
    
  }
  
}

void hicam::elitist_archive_t::set_use_parameter_distances(bool value)
{
  use_parameter_distances = value;
  
  if(use_niching)
  {
    for(size_t i =0 ; i < clustered_archive.size(); ++i) {
      clustered_archive[i]->set_use_parameter_distances(value);
    }
  }
}

void hicam::elitist_archive_t::set_use_greedy_selection(bool value)
{
  use_greedy_selection = value;
  // use_hypervolume_for_size_control = !value;
  
  if(use_niching)
  {
    for(size_t i =0 ; i < clustered_archive.size(); ++i) {
      clustered_archive[i]->set_use_greedy_selection(value);
    }
  }
}

void hicam::elitist_archive_t::set_use_hypervolume_for_size_control(bool value, const vec_t & hypervolume_reference_point)
{
  use_hypervolume_for_size_control = value;
  // use_greedy_selection = !value;
  
  if(use_hypervolume_for_size_control)
  {
    this->hypervolume_reference_point = hypervolume_reference_point;
  }
  
  if(use_niching)
  {
    std::cout << "Use hypervolume not implemented yet for niched archive.\n";
  }
}


void hicam::elitist_archive_t::computeApproximationSet(size_t approximation_set_size, const elitist_archive_pt & elitist_archive, bool use_parameter_space_diversity)
{
  if (elitist_archive->use_niching == false || elitist_archive->target_size <= approximation_set_size)
  {
    this->sols.clear();
    
    elitist_archive->getAllSols(this->sols);
    
    // see if adaptation is required.
    this->target_size = approximation_set_size;
    this->set_use_greedy_selection(true);
    this->set_use_parameter_distances(use_parameter_space_diversity);
    this->adaptArchiveSize();
    
  }
  else
  {
    // use niching and approximation set is smaller than elitist archive.
    // first, we get rid of the clusters that have no rank-0 solution.
    this->sols.clear();

    for(size_t i = 0; i < elitist_archive->clustered_archive.size(); ++i)
    {
      for(size_t j = 0; j < elitist_archive->clustered_archive[i]->sols.size(); ++j)
      {
        if(elitist_archive->clustered_archive[i]->sols[j] != nullptr)
        {
          elitist_archive->clustered_archive[i]->sols[j]->cluster_number = (int) i;
          this->updateArchive(elitist_archive->clustered_archive[i]->sols[j]);
        }
      }
    }
    
    std::vector<bool> subpop_with_rank0(elitist_archive->clustered_archive.size(),false);
    
    for(size_t i = 0; i < this->sols.size(); ++i)
    {
      if(this->sols[i] != nullptr) {
        subpop_with_rank0[this->sols[i]->cluster_number] = true; // not sure if cluster number is set properly..
      }
    }
    
    this->sols.clear();
    
    for(size_t i = 0; i < elitist_archive->clustered_archive.size(); ++i) {
      
      if(subpop_with_rank0[i]) {
        this->addSolutions(*elitist_archive->clustered_archive[i]);
      }
    }
    
    // see if adaptation is required.
    this->target_size = approximation_set_size;
    this->set_use_greedy_selection(true);
    this->set_use_parameter_distances(use_parameter_space_diversity);
    this->adaptArchiveSize();
    
  }
}



void hicam::elitist_archive_t::computeApproximationSet(size_t approximation_set_size, std::vector<optimizer_pt> & optimizers, const elitist_archive_pt & elitist_archive, bool use_parameter_space_diversity, bool terminate_pops)
{

  if(optimizers.size() == 0) {
    return;
  }
  
  if (elitist_archive->use_niching == false || elitist_archive->target_size <= approximation_set_size)
  {
    this->sols.clear();
    
    elitist_archive->getAllSols(this->sols);

    // see if adaptation is required.
    this->target_size = approximation_set_size;
    this->set_use_greedy_selection(true);
    this->set_use_parameter_distances(use_parameter_space_diversity);
    this->adaptArchiveSize();
    
  }
  else
  {
    // use niching and approximation set is smaller than elitist archive.
    // first, we get rid of the clusters that have no rank-0 solution.
    this->sols.clear();
    
    //std::vector<solution_pt> all_sols;
    // elitist_archive->getAllSols(all_sols);
    //for(size_t i =0; i < all_sols.size(); ++i) {
    //  this->updateArchive(all_sols[i]);
    //}
    
    for(size_t i = 0; i < elitist_archive->clustered_archive.size(); ++i)
    {
      for(size_t j = 0; j < elitist_archive->clustered_archive[i]->sols.size(); ++j)
      {
        if(elitist_archive->clustered_archive[i]->sols[j] != nullptr)
        {
          elitist_archive->clustered_archive[i]->sols[j]->cluster_number = (int) i;
          this->updateArchive(elitist_archive->clustered_archive[i]->sols[j]);
        }
      }
    }
    
    std::vector<bool> subpop_with_rank0(elitist_archive->clustered_archive.size(),false);
    
    for(size_t i = 0; i < this->sols.size(); ++i)
    {
      if(this->sols[i] != nullptr) {
      subpop_with_rank0[this->sols[i]->cluster_number] = true; // not sure if cluster number is set properly..
      }
    }
    
    this->sols.clear();
    
    for(size_t i = 0; i < elitist_archive->clustered_archive.size(); ++i) {
      
      if(subpop_with_rank0[i]) {
        this->addSolutions(*elitist_archive->clustered_archive[i]);
      }
    }
    
    // see if adaptation is required.
    this->target_size = approximation_set_size;
    this->set_use_greedy_selection(true);
    this->set_use_parameter_distances(use_parameter_space_diversity);
    this->adaptArchiveSize();
    
  }
  
  
  
  // Terminate populations!
  // size_t archive_size = this->actualSize();
  
  if(terminate_pops && optimizers.size() > 1)
  {
    
    size_t elites_added = 0;
    
    for(size_t i =0; i < optimizers.size(); ++i)
    {
      if(!optimizers[i]->terminated) {
        elites_added += optimizers[i]->new_elites_added;
      }
    }
    
    // never terminate latest optimizer
    for(size_t i =0; i < optimizers.size() -1; ++i)
    {
      // if(optimizers[i]->new_elites_added < 0.5 * optimizers[i+1]->new_elites_added) {
      if(!optimizers[i]->terminated && optimizers[i]->new_elites_added < 0.25 * elites_added) {
        optimizers[i]->terminated = true; std::cout << "-";
      }
    }
  }
    
}

size_t hicam::elitist_archive_t::collectSubarchives(std::vector<population_pt> & subpopulations)
{
 
  initArchiveForEachSubpop(subpopulations);
  
  // Create an Elitist archive from all subarchives
  //---------------------------------------------------------------------
  size_t number_of_solutions_added = 0;
  clear();

  for(size_t i = 0; i < subpopulations.size(); ++i)
  {
    addArchive(subpopulations[i]->elitist_archive);
    number_of_solutions_added += subpopulations[i]->new_elites_added;
  }

  return number_of_solutions_added;
}

// creates an archive per subpop
void hicam::elitist_archive_t::initArchiveForEachSubpop(std::vector<population_pt> & subpopulations) const
{

  for(size_t i = 0; i < subpopulations.size(); ++i)
  {
    // create new archive
    subpopulations[i]->elitist_archive = initNewArchive();
    subpopulations[i]->new_elites_added = 0;
    
    // first, add all previously sampled elites (to reconstruct the archive).
    for(size_t j = 0; j < subpopulations[i]->size(); ++j)
    {
      // add the solution to the archive, and count how many non-elites are added (to measure performance of this optimizer)
      if(subpopulations[i]->sols[j]->elite_origin != nullptr) {
        subpopulations[i]->elitist_archive->updateArchive(subpopulations[i]->sols[j], true);
      }
    }
    
    // then, add all novel solutions.
    for(size_t j = 0; j < subpopulations[i]->size(); ++j)
    {
      // add the solution to the archive, and count how many non-elites are added (to measure performance of this optimizer)
      if(subpopulations[i]->sols[j]->elite_origin == nullptr) {
        if (subpopulations[i]->elitist_archive->updateArchive(subpopulations[i]->sols[j], true) >= 0) {
          subpopulations[i]->new_elites_added++;
        }
      }
    }
  }
}

void hicam::elitist_archive_t::objectiveRanges(vec_t & objective_ranges)
{
  
  if(use_niching)
  {
    std::vector<solution_pt> backup_sols = this->sols;
    
    getAllSols(this->sols);
    population_t::objectiveRanges(objective_ranges);
    
    this->sols = backup_sols;
  }
  else
  {
    population_t::objectiveRanges(objective_ranges);
  }
  
}


/*************************************************************************
 
 gHSS - (incremental) greedy hypervolume subset selection in 2D and 3D
 
 ---------------------------------------------------------------------
 
 Copyright (c) 2015-2017
 Andreia P. Guerreiro <apg@dei.uc.pt>
 
 
 This program is free software (software libre); you can redistribute
 it and/or modify it under the terms of the GNU General Public License,
 version 3, as published by the Free Software Foundation.
 
 This program is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program; if not, you can obtain a copy of the GNU
 General Public License at:
 http://www.gnu.org/copyleft/gpl.html
 or by writing to:
 Free Software Foundation, Inc., 59 Temple Place,
 Suite 330, Boston, MA 02111-1307 USA
 
 ----------------------------------------------------------------------
 
 Relevant literature:
 
 [1] A. P. Guerreiro, C. M. Fonseca, and L. Paquete, “Greedy hypervolume subset selection in low dimensions,” Evolutionary Computation, vol. 24, pp. 521-544, Fall 2016.
 [2] A. P. Guerreiro, C. M. Fonseca, and L. Paquete, “Greedy hypervolume subset selection in the three-objective case,” in Proceedings of the 2015 on Genetic and Evolutionary Computation Conference, GECCO '15, (Madrid, Spain), pp. 671-678, ACM, 2015.
 
 *************************************************************************/

namespace hicam
{

  
  void elitist_archive_t::gHSS(size_t target_size, double r0, double r1, std::vector<solution_pt> & selected_sols, std::vector<solution_pt> & nonselected_sols)
  {
    removeSolutionNullptrs();
    int n = (int) sols.size();
    
    if(n <= target_size) {
      selected_sols = sols;
      nonselected_sols.clear();
      return;
    }
    
    double *ref = (double *) malloc(2 * sizeof(double));
    ref[0] = r0;
    ref[1] = r1;
    
    int k = (int) target_size;
    int d = (int) sols[0]->number_of_objectives();
    
    double *volumes = (double *) malloc(n * sizeof(double));
    int *selected = (int *) malloc(n * sizeof(int));
    
    double *data = (double *) malloc(d * n * sizeof(double));
    
    size_t di = 0;
    for(size_t i = 0; i < n; ++i) {
      for(size_t j = 0; j < d; ++j) {
        data[di] = sols[i]->obj[j];
        di++;
      }
    }
    
    greedyhss(data, d, n, k, ref, volumes, selected);
    
    std::vector<bool> is_sol_selected(n,false);
    
    for(size_t i = 0; i < k; ++i) {
      is_sol_selected[selected[i]] = true;
    }
    
    selected_sols.clear();
    selected_sols.reserve(k);
    nonselected_sols.clear();
    nonselected_sols.reserve(n - k);
    

    for(size_t i = 0; i < n; ++i) {
      if(is_sol_selected[i]) {
        selected_sols.push_back(sols[i]);
      } else {
        nonselected_sols.push_back(sols[i]);
      }
    }
    
    free(ref);
    free(selected);
    free(volumes);
    free(data);
  }
  
  void elitist_archive_t::reduceArchiveSizeByHSS(size_t target_size, double r0, double r1)
  {
    removeSolutionNullptrs();
    int n = (int) sols.size();
    
    if(n <= target_size) {
      return;
    }
    
    double *ref = (double *) malloc(2 * sizeof(double));
    ref[0] = r0;
    ref[1] = r1;
    
    int k = (int) target_size;
    int d = (int) sols[0]->number_of_objectives();
    
    double *volumes = (double *) malloc(n * sizeof(double));
    int *selected = (int *) malloc(n * sizeof(int));
    
    double *data = (double *) malloc(d * n * sizeof(double));
    
    size_t di = 0;
    for(size_t i = 0; i < n; ++i) {
      for(size_t j = 0; j < d; ++j) {
        data[di] = sols[i]->obj[j];
        di++;
      }
    }
    
    greedyhss(data, d, n, k, ref, volumes, selected);
    
    std::vector<solution_pt> new_sols;
    new_sols.reserve(k);
    for(size_t i = 0; i < k; ++i) {
      new_sols.push_back(sols[selected[i]]);
    }
    sols = new_sols;
    
    free(ref);
    free(selected);
    free(volumes);
    free(data);
  }
  
  /* ------------------------------------ Data structure ------------------------------------------*/
  
  typedef struct dlnode {
    double x[3];          // Point
    int in;               //True or False - indicates whether the points has been selected (True) or if still left out (False)
    int updated;          //if in == False, then 'updated' indicates whether the contribution of this points was already updated
    
    //current next (for a list of 'in' points that is modified along the execution)
    struct dlnode * cnext[2];
    
    //current next and prev (list with some of the 'out' points)
    struct dlnode * cnextout[2];
    struct dlnode * cprevout[2];
    
    //global circular doubly linked list - keeps the points sorted according to coordinates 1 to 3 (all points, ie, both 'in' points and 'out' points)
    struct dlnode * next[3];
    struct dlnode *prev[3];
    
    //aditional info for computing contributions
    double area;          // area of 2D projections at z = lastSlicez
    double contrib;       // contribution
    double oldcontrib;    //temporary (save last contribution)
    double lastSlicez;    // up to which value of z the contribution is computed
    struct dlnode * replaced; //the point this one replaced (prev in the paper)
    int dom; //is this a dominated point?
    int id;
    
  } dlnode_t;
  
  
  
  
  /* -------------------------------------- Setup Data ---------------------------------------------*/
  
  
  static void copyPoint(double * source, double * dest, int d){
    int i;
    for(i = 0; i < d; i++)
      dest[i] = source[i];
  }
  
  
  
  static int compare_node(const void *p1, const void* p2)
  {
    const double x1 = *((*(const dlnode_t **)p1)->x);
    const double x2 = *((*(const dlnode_t **)p2)->x);
    
    return (x1 < x2) ? -1 : (x1 > x2) ? 1 : 0;
  }
  
  static int compare_node2d(const void *p1, const void* p2)
  {
    const double x1 = *((*(const dlnode_t **)p1)->x+1);
    const double x2 = *((*(const dlnode_t **)p2)->x+1);
    
    return (x1 < x2) ? -1 : (x1 > x2) ? 1 : 0;
  }
  
  
  static int compare_node3d(const void *p1, const void* p2)
  {
    const double x1 = *((*(const dlnode_t **)p1)->x+2);
    const double x2 = *((*(const dlnode_t **)p2)->x+2);
    
    return (x1 < x2) ? -1 : (x1 > x2) ? 1 : 0;
  }
  
  /*
   * Setup circular double-linked list in each dimension (with two sentinels).
   * Initialize data.
   */
  
  static dlnode_t *
  setup_cdllist(double *data, int d, int n)
  {
    dlnode_t *head;
    dlnode_t **scratch;
    int i, j;
    
    head  = (dlnode_t *) malloc ((n+2) * sizeof(dlnode_t));
    head[0].id = -1;
    head[0].in = 1;
    head[0].area = 0; head[0].contrib = 0; head[0].oldcontrib = 0; head[0].lastSlicez = 0;
    head[0].updated = 0; head[0].dom = 0;
    
    for(i = 0; i < d; i++){
      head[0].x[i] = -1;
      head[n+1].x[i] = -1;
    }
    head[n+1].id = -2;
    head[n+1].in = 1;
    head[n+1].area = 0; head[n+1].contrib = 0; head[n+1].oldcontrib = 0; head[n+1].lastSlicez = 0;
    head[n+1].updated = 0; head[n+1].dom = 0;
    
    for (i = 1; i <= n; i++) {
      //         head[i].x = head[i-1].x + d ;/* this will be fixed a few lines below... */
      copyPoint(&(data[(i-1)*d]), head[i].x, d);
      head[i].area = 0;
      head[i].contrib = 0;
      head[i].oldcontrib = 0;
      head[i].lastSlicez = 0;
      head[i].id = i-1;
      head[i].in = 0;
      head[i].updated = 0;
      head[i].dom = 0;
    }
    
    scratch = (dlnode_t **) malloc(n * sizeof(dlnode_t*));
    for (i = 0; i < n; i++)
      scratch[i] = head + i + 1;
    
    for (j = d-1; j >= 0; j--) {
      if(j == 2) qsort(scratch, n, sizeof(dlnode_t*), compare_node3d);
      else if(j == 1) qsort(scratch, n, sizeof(dlnode_t*), compare_node2d);
      else qsort(scratch, n, sizeof(dlnode_t*), compare_node);
      head->next[j] = scratch[0];
      scratch[0]->prev[j] = head;
      for (i = 1; i < n; i++) {
        scratch[i-1]->next[j] = scratch[i];
        scratch[i]->prev[j] = scratch[i-1];
      }
      //         scratch[n-1]->next[j] = head;
      //         head->prev[j] = scratch[n-1];
      scratch[n-1]->next[j] = head+n+1;
      (head+n+1)->prev[j] = scratch[n-1];
      (head+n+1)->next[j] = head;
      head->prev[j] = (head+n+1);
    }
    
    free(scratch);
    
    return head;
  }
  
  /* -------------------------------------- Misc ----------------------------------------------*/
  
  static inline double max(double a, double b){
    return (a > b) ? a : b;
  }
  
  /* -------------------------------------- Algorithms ----------------------------------------------*/
  
  
  
  static void updateVolume(dlnode_t * p, double z){
    
    p->contrib += p->area * (z - p->lastSlicez);
    p->lastSlicez = z;
    
  }
  
  
  
  
  
  /*
   * Find maximum contributor and, since all points are visited, also set the
   * 'updated' flag of all points to false as they may have to be updated
   */
  static dlnode_t * maximumOutContributor(dlnode_t * list){
    
    double c = -DBL_MAX;
    dlnode_t * maxp = list;
    dlnode_t * p = list->next[0];
    dlnode_t * stop = list->prev[0];
    while(p != stop){
      p->updated = 0;
      if(!p->in && ((p->contrib > c) || (p->contrib == c && p->id < maxp->id))){
        c = p->contrib;
        maxp = p;
      }
      p = p->next[0];
    }
    return maxp;
  }
  
  
  
  /*
   * p - the newly selected point
   * xi, yi, zi - indexes of the coordinates
   * ref - reference point
   *
   *
   * (Step 1 of Algorithms 3 and 4 in the paper)
   * Set up the list that will be used to maintain the points already selected that are not dominated in
   * the (xi,yi)-projection at a given value of coordinate zi (here is assumed to be p->x[zi], i.e.,
   * this function sets up the list of delimiters of p at z = p->x[zi]).
   *
   * Note: The first and last elements of such list will be stored in p->cnext[0] (the rightmost
   * delimiter and below) and in p->cnext[1] (delimiter above and to the left)
   *
   */
  static void createFloor(dlnode_t * list, dlnode_t * p, int xi, int yi, int zi, const double * ref){
    
    dlnode_t * q = list->prev[yi];
    
    //set up sentinels
    list->x[xi] = ref[xi];
    list->x[yi] = -DBL_MAX;
    list->x[zi] = -DBL_MAX;
    
    q->x[xi] = -DBL_MAX;
    q->x[yi] = ref[yi];
    q->x[zi] = -DBL_MAX;
    
    
    dlnode_t * xrightbelow = list;
    q = list->next[yi];
    
    
    //find the closest point to p according to the x-coordinate that has lower or equal yi- and zi- coordinates (xrightbelow)
    while(q->x[yi] <= p->x[yi]){
      if(q->in && q->x[zi] <= p->x[zi] && q->x[xi] <= xrightbelow->x[xi] && q->x[xi] > p->x[xi])
        xrightbelow = q;
      
      q = q->next[yi];
    }
    
    //the rightmost delimiter of p area to the right
    p->cnext[0] = xrightbelow;
    
    dlnode_t * last = xrightbelow;
    
    q = p->next[yi];
    
    //set up the list (using cnext)
    while(!q->in || q->x[xi] > p->x[xi] || q->x[zi] > p->x[zi]){
      
      if(q->in && q->x[zi] <= p->x[zi] && q->x[xi] < last->x[xi] && q->x[xi] > p->x[xi]){
        
        if(q->x[yi] == last->x[yi]){
          last = last->cnext[0];
        }
        q->cnext[0] = last;
        last->cnext[1] = q;
        last = q;
        
      }
      q = q->next[yi];
    }
    
    //the delimiter of p area above and to the left
    q->cnext[0] = last;
    last->cnext[1] = q;
    p->cnext[1] = q;
    
    
  }
  
  
  
  
  /* Compute the area exclusive dominated by p
   * (the area is divided in horizontal bars and their areas are summed up)
   */
  static double computeArea(dlnode_t * p, int xi, int yi){
    
    dlnode_t * q = p->cnext[0];
    dlnode_t * qnext = q->cnext[1];
    
    double area = (q->x[xi] - p->x[xi]) * (qnext->x[yi] - p->x[yi]);
    
    q = qnext;
    while(q != p->cnext[1]){
      qnext = q->cnext[1];
      area += (q->x[xi] - p->x[xi]) * (qnext->x[yi] - q->x[yi]);
      q = qnext;
    }
    
    return area;
    
  }
  
  
  
  /*
   * p - is the newly selected point
   * xi, yi, zi - indexes of the coordinates
   *
   * (steps 2 and 3 of Algorithm 4)
   * cnextout and cprevout will be used to maintain the list of 'out' points that are being updated.
   * The base area for the points in that list is computed.
   *
   * Note: The first and last elements of the list of 'out' will be stored in p->cnextout[0] and in p->cnextout[1]
   *
   */
  static void createAndInitializeBases(dlnode_t * list, dlnode_t * p, int xi, int yi, int zi){
    
    dlnode_t * q;
    dlnode_t * stop = p->cnext[1];
    double parea = p->area;
    
    
    //q is set to the first point dominated by p in the list of all points
    // (care must be taken with points with equal yi-coordinate to that of p)
    if(p->prev[yi]->x[yi] == p->x[yi]){
      q = p;
      while(q->x[yi] == p->x[yi]) q = q->prev[yi];    //deal with points with equal yi-coordinate to p but that are before p in the list
      q = q->next[yi];
    }else{
      q = p->next[yi];
    }
    
    dlnode_t * in = p->cnext[0];                        //'in' keeps track of the last 'in' point visited
    double area = 0;
    dlnode_t * out = list;                              // list is used as sentinel
    
    
    p->cnextout[0] = list;
    
    
    //setup the list of 'out' points that have to be updated
    //and do the first part of the computation of their base area
    while(q != stop){
      if(q != p){                                     // if p->prev[yi]->x[yi] == p->x[yi], then p will be visited in this while loop and has to be skipped
        
        if(q->dom){
          q->updated = 1;
          
        }else if(q->in == 0){                       // q is out
          
          if(q->updated == 0){                    // q has to be updated
            if(p->x[xi] <= q->x[xi] && p->x[yi] <= q->x[yi] && p->x[zi] <= q->x[zi]){
              // q is dominated by p then, its contribution is reduced to 0
              
              q->area = 0;
              q->contrib = 0;
              q->oldcontrib = 0;
              q->updated = 1;
              q->dom = 1;
              
            }else if(p->x[xi] <= q->x[xi] && p->x[yi] <= q->x[yi]
                     && (in->x[xi] > q->x[xi] || in->x[yi] > q->x[yi])
                     && (in->cnext[1]->x[xi] > q->x[xi] || in->cnext[1]->x[yi] > q->x[yi])){//check if the contribution of q will be reduced because of p
              
              q->oldcontrib = q->contrib;
              q->contrib = 0;
              q->area = parea - (in->x[xi] - q->x[xi])*(q->x[yi] - p->x[yi]) - area;
              q->lastSlicez = p->x[zi];
              out->cnextout[1] = q;
              q->cprevout[1] = out;
              out = q;
            }
          }
          
        }else{ // q is in
          
          if(q == in->cnext[1]){                  //if q is a delimiter of p
            area += (in->x[xi] - q->x[xi]) * (q->x[yi] - p->x[yi]);
            in = q;
          }
        }
      }
      q = q->next[yi];
    }
    
    q = list->prev[yi];
    out->cnextout[1] = q;
    q->cprevout[1] = out;
    p->cnextout[1] = q;
    out = q;
    
    
    
    area = 0;
    in = p->cnext[1];
    stop = p->cnext[0];
    
    if(p->prev[xi]->x[xi] == p->x[xi]){
      q = p;
      while(q->x[xi] == p->x[xi]) q = q->prev[xi];
      q = q->next[xi];
    }else{
      q = p->next[xi];
    }
    
    
    //do the second part of the computation of the base area of the 'out' points to be updated
    while(q != stop){
      if(q != p){
        
        if(q->dom){
          
          q->updated = 1;
          
        }else if(q->in == 0){           //q is out
          if(q->updated == 0){        //q has to be updated (its contribution has to be reduced)
            if(p->x[xi] <= q->x[xi] && p->x[yi] <= q->x[yi]
               && (in->x[xi] > q->x[xi] || in->x[yi] > q->x[yi])
               && (in->cnext[0]->x[xi] > q->x[xi] || in->cnext[0]->x[yi] > q->x[yi])){
              
              q->area -= (q->x[xi] - p->x[xi])*(in->x[yi] - p->x[yi]) + area;
              out->cnextout[0] = q;
              q->cprevout[0] = out;
              out = q;
            }
          }
        }else{                          // q is in
          
          if(q == in->cnext[0]){
            area += (q->x[xi] - p->x[xi]) * (in->x[yi] - q->x[yi]);
            in = q;
          }
        }
      }
      q = q->next[xi];
    }
    
    q = list;
    out->cnextout[0] = q;
    q->cprevout[0] = out;
    
    
  }
  
  /*
   * p - the newly selected point
   * cutter - a delimiter of p. The area dominated by p and the 'cutter' point is removed from p.
   * xi, yi, zi - indexes of the coordinates
   * xic - indicate the coordinate used for visiting points. Points will be visited in ascending
   *       order of coordinate xi if xic == 0 and of coordinate yi if xic == 1.
   *
   * The area dominated by p is updated and so is the list of points that delimit the area of p at z = cutter->x[zi].
   * Moreover, the volume and areas of some of the 'out' points below p in zi are updated.
   */
  static double cutOffPartial(dlnode_t * p, dlnode_t * cutter, int xi, int yi, int zi, int xic){
    
    int yic = 1 - xic;
    
    dlnode_t * in = p->cnext[yic];
    dlnode_t * out = p->cnextout[yic];
    dlnode_t * stop;
    
    double area = 0;
    
    while(cutter->x[yi] <= in->x[yi]){
      in = in->cnext[xic];
    }
    dlnode_t * upperLeft = in;
    
    while(out->x[xi] < in->x[xi]){
      out = out->cnextout[xic];
    }
    out = out->cprevout[xic];
    
    stop = p->cnext[yic];
    
    
    while(in != stop){
      
      if(in->cnext[yic]->x[xi] > out->x[xi] || out->x[xi] < p->x[xi]){
        
        in = in->cnext[yic];
        area += (in->cnext[xic]->x[xi] - max(in->x[xi], p->x[xi])) * (in->x[yi] - cutter->x[yi]);
        
      }else{
        
        updateVolume(out, cutter->x[zi]);
        
        if(out->x[yi] >= cutter->x[yi]){        // 'out' has no more contribution above z = in->x[zi]
          out->area = 0;
          out->contrib = out->oldcontrib - out->contrib;
          
          //remove points completely updated
          out->updated = 1;
          out->cprevout[xic]->cnextout[xic] = out->cnextout[xic];
          out->cnextout[xic]->cprevout[xic] = out->cprevout[xic];
          out->cprevout[yic]->cnextout[yic] = out->cnextout[yic];
          out->cnextout[yic]->cprevout[yic] = out->cprevout[yic];
          
        }else{
          
          out->area -= area + (in->x[xi] - out->x[xi]) * (in->cnext[yic]->x[yi] - cutter->x[yi]);
        }
        out = out->cprevout[xic];
      }
      
    }
    
    //insert point 'cutter' as the head of the list (the in points dominated by 'cutter' in the (xi, yi)-plane are implicitly removed)
    p->cnext[yic] = cutter;
    cutter->cnext[xic] = upperLeft;
    upperLeft->cnext[yic] = cutter;
    
    p->area -= area;
    return area;
    
    
  }
  
  
  
  /*
   * p - the new point that will be added to the set of selected points
   * zi - indicates which is the third coordinate (the remaining ones are deduced in this function)
   *
   * Note: This function corresponds to Algorithms 3 and 4 of the paper which are done together in this
   * function so as to avoid sweeping all points a second time according to the z coordinate and to
   * avoid repeating some computations.
   *
   */
  static void updateOut(dlnode_t * list, dlnode_t * p, int zi, const double * ref){
    
    int d = 3;
    int xi = (zi + 1) % d;  //first coordinate
    int yi = 3 - (zi + xi); //second coordinate
    dlnode_t * q = list;
    
    createFloor(list, p, xi, yi, zi, ref);
    p->area = computeArea(p, xi, yi);
    createAndInitializeBases(list, p, xi, yi, zi);
    
    dlnode_t * stop = list->prev[zi];
    stop->x[zi] = ref[zi];
    dlnode_t * domr = list;
    
    q = p->next[zi];
    
    while(q != stop){
      
      if(q->in){
        // update the area of p, update domr volume and area (Alg. 3) and do Algorithm 4 (lines 6 to 22)
        
        if(q->x[xi] <= p->x[xi] && q->x[yi] <= p->x[yi]){ //q is the last delimiter of p (p has no contribution above q->x[zi])
          break;
          
        }else if(q->x[xi] <= p->x[xi] && q->x[yi] > p->x[yi] && q->x[yi] < p->cnext[1]->x[yi]){ //q is to the left of p
          updateVolume(domr, q->x[zi]);                                                       // (Alg. 3, line 16)
          domr->area -= cutOffPartial(p, q, xi, yi, zi, 0);                             // (Alg. 3, line 10 and 17) (Alg. 4, line 12 - 14)
          
        }else if(q->x[xi] > p->x[xi] && q->x[yi] <= p->x[yi] && q->x[xi] < p->cnext[0]->x[xi]){ //q is below p
          updateVolume(domr, q->x[zi]);                                                       // (Alg. 3, line 16)
          domr->area -= cutOffPartial(p, q, yi, xi, zi, 1);                             // (Alg. 3, line 14 and 17) (Alg. 4, line 20 - 22)
          
        }
        
        
      }else{                                                                       //q is an 'out' point
        if(q->dom == 0 && q->updated == 0 && q->x[xi] <= p->x[xi] && q->x[yi] <= p->x[yi]){     //q* < p* (Alg. 3, lines 19 - 24)
          
          q->oldcontrib = q->contrib;
          q->contrib = 0;
          q->replaced = domr;
          updateVolume(domr, q->x[zi]);
          q->area = p->area;
          q->lastSlicez = q->x[zi];
          domr = q;
          
        }
        
      }
      q = q->next[zi];
      
    }
    
    //(Alg. 3, lines 25 - 30)
    updateVolume(domr, q->x[zi]);
    double vol = 0;
    while(domr != list){
      vol += domr->contrib;
      domr->contrib = domr->oldcontrib - vol;
      domr->updated = 1;
      domr = domr->replaced;
    }
    
    dlnode_t * q2;
    q2 = p->cnextout[0]->cnextout[1];
    
    while(q2 != p->cnextout[1]){
      updateVolume(q2, q->x[zi]);
      q2->contrib = q2->oldcontrib - q2->contrib;
      q2->updated = 1;
      q2 = q2->cnextout[1];
    }
    
    
    
  }
  
  
  static void gHSS3D(dlnode_t * list, const int k, int * selected, const double * ref){
    
    int i;
    dlnode_t * maxp = NULL;
    dlnode_t * p = list->next[0];
    dlnode_t *stop = list->prev[0];
    while(p != stop){
      if(p->dom)
        p->contrib = 0; //if p does not strongly dominate the reference point
      else
        p->contrib = (ref[0] - p->x[0]) * (ref[1] - p->x[1]) * (ref[2] - p->x[2]);
      p = p->next[0];
    }
    
    
    for(i = 0; i < k-1; i++){
      
      maxp = maximumOutContributor(list);
      if(maxp->dom == 0){
        //update contribution of the points not yet selected (out points)
        updateOut(list, maxp, 2, ref); // order (x,y,z)
        updateOut(list, maxp, 1, ref); // order (z,x,y)
        updateOut(list, maxp, 0, ref); // order (y,z,x)
      }
      
      selected[i] = maxp->id;
      maxp->in = 1;   // point 'maxp' is now part of the set of selected points
      
      
    }
    
    maxp = maximumOutContributor(list);
    selected[i] = maxp->id;
    maxp->in = 1;
    
  }
  
  
  
  
  static void gHSS2D(dlnode_t * list, const int k, int * selected, const double * ref){
    
    int i;
    dlnode_t * maxp = NULL;
    dlnode_t * p = list->next[0];
    dlnode_t * q = list;
    dlnode_t *stop = list->prev[0];
    dlnode_t * rightin, * upin;
    
    // set sentinels
    list->x[0] = -DBL_MAX;
    list->x[1] = ref[1];
    
    stop->x[0] = ref[0];
    stop->x[1] = -DBL_MAX;
    
    // setup list with cnext, excluding dominated points
    while(p != stop){
      // q is dominated
      if(p->x[0] == q->x[0] && q->x[1] >= p->x[1]){
        q->dom = 1;
        q->contrib = 0;
        q->cnext[1]->cnext[0] = p;
        p->cnext[1] = q->cnext[1];
        p->contrib = (ref[0] - p->x[0]) * (ref[1] - p->x[1]);
        q = p;
        //p is dominated
      }else if(p->x[1] >= q->x[1] || p->dom){
        p->dom = 1;
        p->contrib = 0;
      }else{
        p->contrib = (ref[0] - p->x[0]) * (ref[1] - p->x[1]);
        q->cnext[0] = p;
        p->cnext[1] = q;
        q = p;
      }
      p = p->next[0];
    }
    q->cnext[0] = stop;
    stop->cnext[1] = q;
    
    // greedy subset selection in 2D
    for(i = 0; i < k-1; i++){
      maxp = maximumOutContributor(list); //find the point that contributes the most to the already selected points
      if(maxp->dom == 0){
        upin = maxp->cnext[1];
        
        while(!upin->in) upin = upin->cnext[1];
        
        rightin = maxp->cnext[0];
        while(!rightin->in) rightin = rightin->cnext[0];
        
        p = maxp->cnext[0];
        while(p != rightin){
          p->contrib -= (rightin->x[0]-p->x[0]) * (upin->x[1]-maxp->x[1]);
          p = p->cnext[0];
        }
        
        p = maxp->cnext[1];
        while(p != upin){
          p->contrib -= (rightin->x[0]-maxp->x[0]) * (upin->x[1]-p->x[1]);
          p = p->cnext[1];
        }
        
      }else{
        maxp->contrib = 0;
      }
      
      selected[i] = maxp->id;
      maxp->in = 1;           // point 'maxp' is in now part of the set of selected points
    }
    
    // no need to update the data structure after selecting the k-th point
    maxp = maximumOutContributor(list);
    selected[i] = maxp->id;
    maxp->in = 1;
    
  }
  
  
  /*
   * mark and initialize the points that do not strongly dominate the reference point and return
   * how many of such points exist
   */
  int markInvalidPoints(dlnode_t * list, int d, const double * ref){
    
    int di;
    dlnode_t * p;
    dlnode_t * stop;
    
    int nmarked = 0;
    
    for(di = 0; di < d; di++){
      
      p = list->prev[di]->prev[di];
      stop = list;
      while(p != stop && p->x[di] >= ref[di]){
        if(p->dom == 0){
          p->dom = 1;
          p->contrib = 0;
          p->oldcontrib = 0;
          p->area = 0;
          nmarked++;
        }
        p = p->prev[di];
      }
    }
    
    return nmarked;
    
  }
  
  /* Input:
   * data - array containing all 3D points
   * n - number of points
   * k - subset size (select the k most promising points, one at a time)
   * ref - reference point
   *
   * Output:
   * the total volume of the subset selected is returned
   * 'contribs' - the contribution of the selected points at the time their
   *              were selected (ex.: contribs[i] holds the contribution of
   *              the i-th selected point)
   * 'selected' - the index of the selected points regarding their order
   *              in 'data' (ex.: selected[i] holds the index of the i-th
   *              selected point. selected[i] holds a value in the range [0,...,n-1])
   */
  double elitist_archive_t::greedyhss(double *data, int d, int n, const int k, const double *ref, double * contribs, int * selected)
  {
    
    double totalhv = 0;
    
    dlnode_t *list;
    
    list = setup_cdllist(data, d, n);
    
    int nmarked = markInvalidPoints(list, d, ref);
    if(nmarked == n){
      int i;
      for(i = 0; i < n; i++){
        selected[i] = i;
        contribs[i] = 0;
      }
      free(list);
      return 0;
    }
    
    
    if (d == 2){
      gHSS2D(list, k, selected, ref);
    }else if(d == 3){
      gHSS3D(list, k, selected, ref);
    }else{
      free(list);
      return -1;
    }
    int * sel2idx = (int *) malloc(n * sizeof(int));
    dlnode_t * p = list->next[0];
    dlnode_t * stop = list->prev[0];
    int i = 0;
    
    for(i = 0; i < n; i++){
      sel2idx[i] = n;
    }
    for(i = 0; i < k; i++){
      sel2idx[selected[i]] = i;
    }
    
    
    while(p != stop){
      if(sel2idx[p->id] < k){
        contribs[sel2idx[p->id]] = p->contrib;
        totalhv += p->contrib;
      }
      p = p->next[0];
      i++;
    }
    
    
    free(sel2idx);
    free(list);
    
    return totalhv;
  }
  
}











