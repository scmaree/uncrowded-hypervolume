 /*

HICAM Multi-objective

By S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "optimizer.h"
#include "population.h"
#include "mathfunctions.h"
#include "elitist_archive.h"
#include "hillvalleyclustering.h"

hicam::optimizer_t::optimizer_t(
  fitness_pt fitness_function,
  const vec_t & lower_init_ranges,
  const vec_t & upper_init_ranges,
  int local_optimizer_index,
  double HL_tol,
  size_t population_size,
  size_t number_of_mixing_components,
  int optimizer_number,
  unsigned int maximum_number_of_evaluations,
  unsigned int maximum_number_of_seconds,
  double vtr,
  bool use_vtr,
  rng_pt rng
  )
{

  // get problem details
  //---------------------------------------------------------
  this->number_of_parameters = fitness_function->number_of_parameters;
  this->number_of_objectives = fitness_function->number_of_objectives;
  fitness_function->get_param_bounds(lower_param_bounds, upper_param_bounds);
  this->fitness_function = fitness_function;
  this->local_optimizer_index = local_optimizer_index;
  this->HL_tol = HL_tol;
  this->lower_init_ranges = lower_init_ranges;
  this->upper_init_ranges = upper_init_ranges;
  this->population_size = population_size;
  this->number_of_mixing_components = number_of_mixing_components;
  this->optimizer_number = optimizer_number;
  this->maximum_number_of_evaluations = maximum_number_of_evaluations;
  this->maximum_number_of_seconds = maximum_number_of_seconds;
  this->vtr = vtr;
  this->use_vtr = use_vtr;
  this->rng = rng;

  // guideline parameters
  //---------------------------------------------------------
  tau = 0.35;
  maximum_no_improvement_stretch = 10000; // (int)(2.0 + ((double)(25 + number_of_parameters)) / ((double)number_of_mixing_components));
  
  
  if (this->number_of_mixing_components < number_of_objectives + 1) {
    this->number_of_mixing_components = number_of_objectives + 1;
  }
  
  cluster_size = (2 * (size_t)(tau*population_size)) / number_of_mixing_components; // = 2*selection_size / number_of_mixing_components

  terminated = false;
  number_of_generations = 0;
  number_of_evaluations = 0;
  no_improvement_stretch = 0;
  average_edge_length = 1.0;
  
  new_elites_added = 0;
  
  global_selection = false;
}

hicam::optimizer_t::~optimizer_t() {};


// returns true if a termination condition was hit. 
bool hicam::optimizer_t::checkTerminationCondition()
{
  return false;
}


void hicam::optimizer_t::initialize(elitist_archive_t & elitist_archive, unsigned int & number_of_evaluations)
{
  // creat a new population
  population = std::make_shared<population_t>();
  if (fitness_function->redefine_random_initialization) {
    fitness_function->init_solutions_randomly(population, population_size, lower_init_ranges, upper_init_ranges, 0, rng);
  }
  else
  {
    population->fill_uniform(population_size, number_of_parameters, lower_init_ranges, upper_init_ranges, rng);
  }
  
  
  population->evaluate(fitness_function, 0, number_of_evaluations);
  population->setPopulationNumber(optimizer_number); // used in the restart scheme

  population->objectiveRanges(objective_ranges);
  population->compute_fitness_ranks();
  // population->elitist_archive = elitist_archive.initNewArchive();
  // elitist_archive.addArchive(population->elitist_archive);
  
  // update the elitist archive
  new_elites_added = 0;
  for (size_t i = 0; i < population->size(); i++) {
    new_elites_added += (elitist_archive.updateArchive(population->sols[i], true) >= 0);
  }
  
  // update the elitist archive
  population_pt subpopulation = std::make_shared<population_t>(*population);
  subpopulations.push_back(subpopulation);
  
  int cluster_number = (int) (subpopulations.size() - 1);
  
  for(size_t j = 0; j < subpopulations[cluster_number]->sols.size(); ++j) {
    subpopulations[cluster_number]->sols[j]->cluster_number = cluster_number;
  }
  
  elitist_archive.adaptArchiveSize();
  
  number_of_generations++;
  
}

void hicam::optimizer_t::generation(elitist_archive_t & elitist_archive, unsigned int & number_of_evaluations)
{

  // makeSelection
  // SORTS the population based on ranks.
  // population is still the same size.
  //----------------------------------------------------------------------
  std::vector<population_pt> previous_subpopulations;
  if(number_of_generations > 0) {
    previous_subpopulations = subpopulations;
  }
  
  // Cluster the population into subpopulations
  //----------------------------------------------------------------------
  subpopulations.clear();
  population_pt subpopulation = std::make_shared<population_t>(*population);
  subpopulation->elitist_archive = previous_subpopulations[0]->elitist_archive;
  subpopulations.push_back(subpopulation);
  population->sols.clear();

  for(size_t i = 0; i < subpopulations.size(); ++i)
  {

    size_t selection_size = (size_t)(tau * subpopulations[i]->size());
    subpopulations[i]->makeSelection(selection_size, objective_ranges, rng);
 
    // Cluster the population
    //----------------------------------------------------------------------
    size_t minimum_cluster_size;
    
    if( number_of_parameters <= 2 ) {
      minimum_cluster_size = number_of_parameters + 1;
    }
    else {
      minimum_cluster_size = 2 + log(number_of_parameters);
    }
    size_t number_of_clusters = std::min(number_of_mixing_components / subpopulations.size(), selection_size / minimum_cluster_size);
    number_of_clusters = std::max((size_t) 1, number_of_clusters);
    size_t cluster_size = std::max(minimum_cluster_size, 2*selection_size / number_of_clusters);
    
    if(number_of_clusters == 1) { cluster_size = selection_size; }
    if(number_of_clusters == 2) { cluster_size = std::max(minimum_cluster_size, (size_t) (1.5*selection_size / number_of_clusters)); }

    mamalgam_clustering(*subpopulations[i], subpopulations[i]->clusters, number_of_clusters, cluster_size, selection_size, rng);
    
    // Register clusters to clusters of the previous generation
    //----------------------------------------------------------------------
    // todo: handle merged elitist archives.
    std::vector<cluster_pt> previous_clusters;
    for(size_t j = 0; j < previous_subpopulations.size(); ++j)
    {
      if(previous_subpopulations[j]->elitist_archive == subpopulations[i]->elitist_archive)
      {
        for(size_t k = 0; k < previous_subpopulations[j]->clusters.size(); ++k)
        {
          previous_clusters.push_back(previous_subpopulations[j]->clusters[k]);
        }
      }
    }
    
    if (previous_clusters.size() > 0) {
      mamalgam_cluster_registration(subpopulations[i]->clusters, previous_clusters, objective_ranges);
      // direct_cluster_registration_on_mean(subpopulations[i]->clusters, previous_clusters, objective_ranges);
    }
    
    // Copy elites back to the clusters
    //----------------------------------------------------------------------
    double subpop_number_of_elites = (tau*population_size) / ((double) subpopulations.size());
    size_t max_number_of_elites = (size_t) subpop_number_of_elites / subpopulations[i]->clusters.size();
    elitist_archive.copyElitesToClusters(subpopulations[i]->clusters, max_number_of_elites, objective_ranges, rng);
    
    // Estimate sample parameters for each cluster
    //----------------------------------------------------------------------
    for (size_t j = 0; j < subpopulations[i]->clusters.size(); ++j) {
      subpopulations[i]->clusters[j]->estimateParameters();
      subpopulations[i]->clusters[j]->computeParametersForSampling();
    }
    
    // Sample new solutions for each cluster and update the elitist archive
    //----------------------------------------------------------------------
    subpopulations[i]->sols.clear();
    size_t subpopulation_size = (population_size / subpopulations.size()) + ((population_size % subpopulations.size()) > i);
    size_t number_of_elites = 0;
    generateAndEvaluateNewSolutionsToFillPopulation(*subpopulations[i], subpopulation_size, subpopulations[i]->clusters, number_of_elites, number_of_evaluations, rng);
    subpopulations[i]->compute_fitness_ranks();
 
    // Update the strategy parameters of the clusters based on the sampled solutions in the population
    // this is a simple clustering of the current population to the previous cluster means.
    //----------------------------------------------------------------------
    updateStrategyParameters(*subpopulations[i], subpopulations[i]->clusters, elitist_archive, no_improvement_stretch, maximum_no_improvement_stretch, objective_ranges);
    population->addSolutions(*subpopulations[i]);
    population->objectiveRanges(objective_ranges);
    
    // Update Elitist Archive
    //----------------------------------------------------------------------
    new_elites_added = 0;
    for(size_t j = 0; j < subpopulations[i]->size(); ++j) {
      new_elites_added += (elitist_archive.updateArchive(subpopulations[i]->sols[j], true) >= 0);
    }
    
  }
  
  // Clean up the elitist_archive if its too big
  //----------------------------------------------------------------------
  // hvc_t hvc(fitness_function);
  // bool add_test_sols = false, recheck_elites = false;
  // std::vector<population_pt> elitist_subarchives;
  elitist_archive.removeSolutionNullptrs();
  // hvc.cluster_ObjParamDistanceRanks(elitist_archive, elitist_subarchives, number_of_evaluations, average_edge_length, add_test_sols, recheck_elites, optimizer_number, rng);
  
  // hvc.HL_filter(elitist_archive, elitist_archive, HL_tol);
  
  //if(elitist_subarchives.size() > 1)
  //{
    // std::cout << elitist_subarchives.size() << " clusters found. reducing!";
   // elitist_archive.clear();
   // elitist_archive.addSolutions(elitist_subarchives[0]->sols);
  //}
  
  elitist_archive.adaptArchiveSize();

  // finish generation
  number_of_generations++;
  
}

void hicam::optimizer_t::initialize_mm(elitist_archive_t & elitist_archive, unsigned int & number_of_evaluations)
{
  // Create a new population, uniformly sampled
  //-------------------------------------------------------------
  population_pt population = std::make_shared<population_t>();
  elitist_archive.addElitesToPopulation(*population, -1); // -1 == all
  size_t number_of_elites_added = population->size();
  
  // population->fill_maximin(number_of_elites_added + population_size, number_of_parameters, lower_init_ranges, upper_init_ranges, number_of_elites_added, rng);
  if (fitness_function->redefine_random_initialization) {
    fitness_function->init_solutions_randomly(population, number_of_elites_added + population_size, lower_init_ranges, upper_init_ranges, number_of_elites_added, rng);
  }
  else
  {
    population->fill_uniform(number_of_elites_added + population_size, number_of_parameters, lower_init_ranges, upper_init_ranges, number_of_elites_added, rng);
  }
  population->evaluate(fitness_function, number_of_elites_added, number_of_evaluations);
  population->setPopulationNumber(optimizer_number, number_of_elites_added); // used in the restart scheme
  population->objectiveRanges(objective_ranges);
  
  // Cluster the new population
  //-------------------------------------------------------------
  hvc_t hvc(fitness_function);
  subpopulations.clear();
  bool add_test_sols = true, recheck_elites = false;
  
  if(global_selection) {
    // global selection, cluster the selection. But, how what is the selection?
    // MAMaLGaM doesn't have a selection, it has multiple selections, 35% in each objective, 35% in domination rank.
  }
  else
  {
    // local selection, cluster the entire population
    hvc.cluster(*population, subpopulations, number_of_evaluations, average_edge_length, add_test_sols, recheck_elites, optimizer_number, rng);
  }
  // update the elitist archive
  //-------------------------------------------------------------
  new_elites_added = elitist_archive.collectSubarchives(subpopulations);
  elitist_archive.adaptArchiveSize(); // Clean up the elitist_archive if its too big
  
  // remove elites from the subpopulations
  //-------------------------------------------------------------
  size_t max_elites_per_subpop = 0; // none added in the first generation, as in MaMaLGaM
  removeElitesFromSubpopulations(max_elites_per_subpop, subpopulations, rng);
  
  // finish generation
  number_of_generations++;

}


void hicam::optimizer_t::generation_mm(elitist_archive_t & elitist_archive, unsigned int & number_of_evaluations, bool largest_population)
{
  // compute subpopulation means
  std::vector<vec_t> previous_means(subpopulations.size());
  for(size_t j = 0; j < subpopulations.size(); ++j)
  {
    size_t selection_size = std::max((size_t) 1, (size_t)(tau * subpopulations[j]->size()));
    subpopulations[j]->compute_mean_of_selection(previous_means[j], selection_size);
  }
  
  // generate offspring for each subpopulation
  //-------------------------------------------------------------
  size_t to_be_sampled_solutions = (size_t) ((1.0-tau) * population_size);
  for(size_t i = 0; i < subpopulations.size(); ++i) {
    size_t number_of_solutions = to_be_sampled_solutions / subpopulations.size() + (( to_be_sampled_solutions % subpopulations.size()) > i);
    generateOffspring(*subpopulations[i], number_of_solutions); // performs local ranking of solutions.
  }
  
  // Evaluate new population + add elites
  //-------------------------------------------------------------
  population_pt population = std::make_shared<population_t>();
  population->collectSolutions(subpopulations);
  population->evaluate(fitness_function, 0, number_of_evaluations);
  population->setPopulationNumber(optimizer_number); // used in the restart scheme
  population->objectiveRanges(objective_ranges);
  
  // update strategy parameters (before adding elites?)
  for(size_t i = 0 ; i < subpopulations.size(); ++i) {
    updateStrategyParameters(*subpopulations[i], subpopulations[i]->clusters, *subpopulations[i]->elitist_archive, no_improvement_stretch, maximum_no_improvement_stretch, objective_ranges);
  }
  
  elitist_archive.addElitesToPopulation(*population, -1);
  population->objectiveRanges(objective_ranges); // recompute after elites have been added.
  
  // Cluster the new population
  //-------------------------------------------------------------
  std::vector<population_pt> previous_subpopulations = subpopulations; // for linking (transfer of hyper parameters)
  subpopulations.clear();
  
  hvc_t hvc(fitness_function);
  bool add_test_sols = true, recheck_elites = false;
  hvc.cluster(*population, subpopulations, number_of_evaluations, average_edge_length, add_test_sols, recheck_elites, optimizer_number, rng);
  // hvc.cluster_ObjParamDistanceRanks(*population, subpopulations, number_of_evaluations, average_edge_length, add_test_sols, recheck_elites, optimizer_number, rng);
  
  // update the elitist archive
  //-------------------------------------------------------------
  new_elites_added = elitist_archive.collectSubarchives(subpopulations);
  elitist_archive.adaptArchiveSize(); // Clean up the elitist_archive if its too big
  
  // remove elites from the subpopulations
  //-------------------------------------------------------------
  size_t max_elites_per_subpop = (size_t) ((tau * population_size) / subpopulations.size());
  removeElitesFromSubpopulations(max_elites_per_subpop, subpopulations, rng);
  linkSubpopulations(subpopulations, previous_subpopulations, previous_means); // link after removing elites.

  // finish generation
  number_of_generations++;

  //--------------------------
  
}

void hicam::optimizer_t::linkSubpopulations(std::vector<population_pt> & subpopulations, std::vector<population_pt> & previous_subpopulations, const std::vector<vec_t> & previous_means ) const
{
  // match subpopulation to previous subpopulations.
  size_t selection_size;
  vec_t current_mean;

  for(size_t i = 0; i < subpopulations.size(); ++i)
  {
    selection_size = std::max((size_t) 1, (size_t)(tau * subpopulations[i]->size()));
    
    subpopulations[i]->compute_mean_of_selection(current_mean, selection_size);
    int nearest_previous = current_mean.nearest_other(previous_means);
    
    if(nearest_previous >= 0)
    {
      subpopulations[i]->previous = previous_subpopulations[nearest_previous];
      subpopulations[i]->previous->previous = nullptr;
    }
  }
  
}

void hicam::optimizer_t::generateOffspring(population_t & pop, size_t number_of_solutions)
{
  
  // sort the population into a selection.
  //----------------------------------------------------
  size_t selection_size = std::max((size_t) 1, (size_t)(tau * pop.size()));
  pop.compute_fitness_ranks();
  vec_t local_objective_ranges;
  pop.objectiveRanges(local_objective_ranges);
  pop.makeSelection(selection_size, objective_ranges, rng);
  
  // Cluster the population
  //----------------------------------------------------------------------
  size_t minimum_cluster_size;
  
  if( number_of_parameters <= 2 ) {
    minimum_cluster_size = number_of_parameters + 1;
  }
  else {
    minimum_cluster_size = 2 + log(number_of_parameters);
  }
  size_t number_of_clusters = std::min(number_of_mixing_components / subpopulations.size(), selection_size / minimum_cluster_size);
  number_of_clusters = std::max((size_t) 1, number_of_clusters);
  size_t cluster_size = std::max(minimum_cluster_size, 2*selection_size / number_of_clusters);
  
  if(number_of_clusters == 1) { cluster_size = selection_size; }
  if(number_of_clusters == 2) { cluster_size = std::max(minimum_cluster_size, (size_t) (1.5*selection_size / number_of_clusters)); }

  
  mamalgam_bugfix_clustering(pop, selection_size, pop.clusters, number_of_clusters, cluster_size, average_edge_length, rng);
  
  // clusters need to be linked to previous clusters, because we need it in the sampling for the AMS.
  //----------------------------------------------------------------------
  if( pop.previous != nullptr && pop.previous->clusters.size() > 0) {
    direct_cluster_registration_on_mean(pop.clusters, pop.previous->clusters, objective_ranges);
  }
  
  // Sample new solutions for each cluster and collect them in the population
  //----------------------------------------------------------------------
  pop.sols.clear();
  pop.sols.reserve(number_of_solutions);
  
  for (size_t i = 0; i < pop.clusters.size(); ++i)
  {
    int cluster_sample_size = (int) ((number_of_solutions / pop.clusters.size()) + ((number_of_solutions % pop.clusters.size()) > i));
    assert(cluster_sample_size >= 0);
    
    // having AMS here is not really cluster-type-independent -> i should move it inside amalgam_t at some point.
    size_t number_of_ams_solutions = 0;
    if (number_of_generations > 1 && pop.clusters[i]->previous != nullptr) {
      number_of_ams_solutions = (size_t)(0.5*tau*((double)number_of_solutions / pop.clusters.size()));
    }
    
    
    pop.clusters[i]->estimateParameters();
    pop.clusters[i]->computeParametersForSampling();
    
    std::vector<solution_pt> new_solutions(cluster_sample_size);
    pop.clusters[i]->generateNewSolutions(new_solutions, cluster_sample_size, number_of_ams_solutions, rng);
    
    pop.addSolutions(new_solutions);
  }
  
  assert(pop.size() == number_of_solutions);
  
}


void hicam::optimizer_t::removeElitesFromSubpopulations(size_t max_elites_per_subpop, std::vector<population_pt> & subpopulations, rng_pt & rng) const
{
 
  for(size_t i = 0; i < subpopulations.size(); ++i)
  {
    
    assert(subpopulations[i]->size() > 0);
    
    // keep at least number_of_parameters+1 solutions in the cluster
    if(subpopulations[i]->size() > 1)
    {
      std::vector<solution_pt> backup_sols = subpopulations[i]->sols;
      subpopulations[i]->sols.clear();
      std::vector<solution_pt> backup_elites;
      
      for(size_t j = 0 ; j < backup_sols.size(); ++j)
      {
        if(backup_sols[j]->elite_origin == nullptr) {
          subpopulations[i]->sols.push_back(backup_sols[j]);
        }
        else {
          backup_elites.push_back(backup_sols[j]);
        }
      }
      
      if(backup_elites.size() > max_elites_per_subpop)
      {
        // keep that many elites such that the resulting pop is at least of size 'number_of_parameters+1'
        double elite_lowerbound = std::max(1.0, number_of_parameters + 1.0 - subpopulations[i]->size());
        size_t number_of_elites_to_keep = (size_t) (std::max(elite_lowerbound, (double) max_elites_per_subpop));
        selectSolutionsBasedOnParameterDiversity(backup_elites, number_of_elites_to_keep, subpopulations[i]->sols, rng);
      }
      else
      {
        subpopulations[i]->addSolutions(backup_elites);
      }
      
    }
  }
}

void hicam::optimizer_t::updateStrategyParameters(const population_t & population, std::vector<cluster_pt> & clusters, const elitist_archive_t & previous_elitist_archive, size_t & no_improvement_stretch, const size_t maximum_no_improvement_stretch, const vec_t & objective_ranges) const
{

  // redistribution_of_generated_solutions.
  // backup the current solutions in the clusters,
  // replace them temporarily with the newly sampled solutions.
  //-----------------------------------------------------------------------------------

  std::vector<std::vector<solution_pt>> backup_solutions(clusters.size());

  for (size_t i = 0; i < clusters.size(); ++i)
  {
    clusters[i]->average_fitness(clusters[i]->objective_mean); // already computed?
    backup_solutions[i] = clusters[i]->sols;
    clusters[i]->sols.clear();
    clusters[i]->sols.reserve(backup_solutions[i].size());
  }

  // Match population with cluster means
  //-----------------------------------------------------------------------------------
  double distance_smallest, distance;
  size_t index_smallest;
  for (size_t i = 0; i < population.size(); ++i)
  {

    distance_smallest = -1.0;
    index_smallest = 0;
    for (size_t j = 0; j < clusters.size(); ++j)
    {
      distance = population.sols[i]->transformed_objective_distance(clusters[j]->objective_mean, objective_ranges);
      if ((distance_smallest < 0.0) || (distance < distance_smallest))
      {
        index_smallest = j;
        distance_smallest = distance;
      }
    }
    clusters[index_smallest]->sols.push_back(population.sols[i]);
  }
  
  // update strategy parameters
  //-----------------------------------------------------------------------------------
  size_t number_of_cluster_failures = 0;

  for (size_t i = 0; i < clusters.size(); i++)
  {
    if(previous_elitist_archive.use_niching)
    {
      if (clusters[i]->updateStrategyParameters(*clusters[i]->elitist_archive, no_improvement_stretch, maximum_no_improvement_stretch)) {
        number_of_cluster_failures++;
      }
    }
    else
    {
      if (clusters[i]->updateStrategyParameters(previous_elitist_archive, no_improvement_stretch, maximum_no_improvement_stretch)) {
        number_of_cluster_failures++;
      }
    }
  }

  if (number_of_cluster_failures == clusters.size()) {
    no_improvement_stretch++;
  }

  // recover original solutions
  //-----------------------------------------------------------------------------------
  // i think that this doesn't actually do anything at the moment,
  // but its better to keep stuff consistent.
  for (size_t i = 0; i < clusters.size(); ++i) {
    clusters[i]->sols = backup_solutions[i];
  }

}



void hicam::optimizer_t::generateAndEvaluateNewSolutionsToFillPopulationNoElites(population_t & population, size_t number_of_solutions_to_generate, const std::vector<cluster_pt> & clusters, unsigned int & number_of_evaluations, rng_pt & rng) const
{
  
  // solutions (selection of it) is copied to the clusters.
  size_t initial_population_size = population.size();
  population.sols.reserve(initial_population_size + number_of_solutions_to_generate);
  
  
  // sample new solutions
  //---------------------------------------------------------
  int cluster_sample_size;
  
  for (size_t i = 0; i < clusters.size(); i++)
  {
    
    // sample new solutions
    cluster_sample_size = (int) ((number_of_solutions_to_generate / clusters.size()) + ((number_of_solutions_to_generate % clusters.size()) > i));
    assert(cluster_sample_size >= 0);
    
    std::vector<solution_pt> new_solutions(cluster_sample_size);
    
    // having AMS here is not really cluster-type-independent -> i should move it inside amalgam_t at some point.
    size_t number_of_ams_solutions = 0;
    if (number_of_generations > 1 && clusters[i]->previous != nullptr) {
      number_of_ams_solutions = (size_t)(0.5*tau*((double)number_of_solutions_to_generate / clusters.size()));
    }
    clusters[i]->generateNewSolutions(new_solutions, cluster_sample_size, number_of_ams_solutions, rng);
    
    
    population.addSolutions(new_solutions);
    
  }
  
  assert(population.size() == initial_population_size + number_of_solutions_to_generate);
  
  // evaluate new solutions
  //---------------------------------------------------------
  population.evaluate(fitness_function, initial_population_size, number_of_evaluations);
  
}


void hicam::optimizer_t::generateAndEvaluateNewSolutionsToFillPopulation(population_t & population, size_t number_of_solutions_to_generate, const std::vector<cluster_pt> & clusters, size_t & number_of_elites, unsigned int & number_of_evaluations, rng_pt & rng) const
{
  
  // solutions (selection of it) is copied to the clusters.
  size_t initial_population_size = population.size();
  population.sols.reserve(initial_population_size + number_of_solutions_to_generate);
  
  // add elites
  //----------------------------------------------------------
  for (size_t i = 0; i < clusters.size(); i++) {
    
    for (size_t j = 0; j < clusters[i]->elites.size(); ++j) {
      clusters[i]->elites[j]->elite_origin = clusters[i]->elitist_archive;
    }
    
    population.addCopyOfSolutions(clusters[i]->elites);
    
    
    for (size_t j = 0; j < clusters[i]->elites.size(); ++j) {
      clusters[i]->elites[j]->elite_origin = nullptr;
    }
    
    
  }
  
  number_of_elites = population.size();
  
  
  // sample new solutions
  //---------------------------------------------------------
  int cluster_sample_size;
  
  for (size_t i = 0; i < clusters.size(); i++)
  {
    
    // sample new solutions
    cluster_sample_size = (int) ((number_of_solutions_to_generate / clusters.size()) + ((number_of_solutions_to_generate % clusters.size()) > i) - clusters[i]->elites.size());
    assert(cluster_sample_size >= 0);
    
    std::vector<solution_pt> new_solutions(cluster_sample_size);
    
    // having AMS here is not really cluster-type-independent -> i should move it inside amalgam_t at some point.
    size_t number_of_ams_solutions = 0;
    if (number_of_generations > 1 && clusters[i]->previous != nullptr) {
      number_of_ams_solutions = (size_t)(0.5*tau*((double)number_of_solutions_to_generate / clusters.size()));
    }
    clusters[i]->generateNewSolutions(new_solutions, cluster_sample_size, number_of_ams_solutions, rng);

    
    population.addSolutions(new_solutions);
    
  }
  
  assert(population.size() == initial_population_size + number_of_solutions_to_generate);
  
  // evaluate new solutions
  //---------------------------------------------------------
  population.evaluate(fitness_function, number_of_elites, number_of_evaluations);

}

void hicam::optimizer_t::mamalgam_bugfix_clustering(const population_t & pop, size_t selection_size, std::vector<cluster_pt> & clusters, size_t number_of_clusters, size_t cluster_size, double average_edge_length, rng_pt & rng) const
{
  // add MO clusters
  std::vector<cluster_pt> new_clusters;
  mamalgam_MO_clustering(pop, new_clusters, number_of_clusters, cluster_size, selection_size, rng);
  
  // std::cout << number_of_clusters << ",";
  
  if(number_of_clusters <= number_of_objectives) {
    clusters = new_clusters;
  }
  else
  {
    clusters.reserve(number_of_clusters);
    
    for(size_t i = number_of_objectives; i < number_of_clusters; ++i)
    {
      clusters.push_back(new_clusters[i]);
    }
    
    vec_t objective_ranges;
    mamalgam_SO_clustering(pop, clusters, cluster_size);
    // std::cout << "so?";
  }
  
  for(size_t i =0 ; i < clusters.size(); ++i) {
    clusters[i]->init_bandwidth = average_edge_length;
  }
  
}




void hicam::optimizer_t::mamalgam_clustering(const population_t & pop, std::vector<cluster_pt> & clusters, size_t number_of_clusters, size_t cluster_size, size_t selection_size, rng_pt & rng) const
{
  clusters.clear();
  
  // add MO clusters
  
  size_t number_of_mo_clusters = number_of_clusters - number_of_objectives;
  
  if(number_of_clusters <= number_of_objectives) {
    number_of_mo_clusters = number_of_clusters;
  }
  
  mamalgam_MO_clustering(pop, clusters, number_of_mo_clusters, cluster_size, selection_size, rng);


  
  if(number_of_clusters > number_of_objectives) {
    mamalgam_SO_clustering(pop, clusters, cluster_size);
  }
}


void hicam::optimizer_t::mamalgam_MO_clustering(const population_t & pop, std::vector<cluster_pt> & clusters, size_t number_of_clusters, size_t cluster_size, size_t selection_size, rng_pt & rng) const
{
  
  // Determine the leaders from the selection
  //-------------------------------------------------------------------------
  std::vector<solution_pt> leaders;
  population_t selection;
  pop.truncation_size(selection, selection_size);
  
  selectSolutionsBasedOnObjectiveDiversity(selection.sols, number_of_clusters, leaders, objective_ranges, rng);

  // Do leader-based distance assignment
  vec_t distances_to_cluster(selection.size());
  std::vector<size_t> distance_ranks(selection.size());

  for (size_t i = 0; i < number_of_clusters; i++)
  {

    clusters.push_back(init_cluster(local_optimizer_index, fitness_function, rng));
    clusters.back()->elitist_archive = pop.elitist_archive;

    for (size_t j = 0; j < selection.size(); j++) {
      distances_to_cluster[j] = selection.sols[j]->transformed_objective_distance(*leaders[i], objective_ranges);
    }

    compute_ranks_asc(distances_to_cluster, distance_ranks);

    for (size_t j = 0; j < std::min(cluster_size, selection.size()); ++j) {
      clusters.back()->addSolution(selection.sols[distance_ranks[j]]);
    }
  }

}


void hicam::optimizer_t::mamalgam_SO_clustering(const population_t & pop, std::vector<cluster_pt> & clusters, size_t cluster_size) const
{

  // for each objective, create a cluster with the 
  // best solutions in that objective (Single-Objective)

  for (size_t i = 0; i < number_of_objectives; i++)
  {

    clusters.push_back(init_cluster(local_optimizer_index, fitness_function, rng));
    clusters.back()->elitist_archive = pop.elitist_archive;
    clusters.back()->objective_number = number_of_objectives + 1; // 0 == undefined., so use + 1.

    // SO selection

    vec_t individual_objectives(pop.size());
    std::vector<size_t> individual_objectives_ranks(pop.size());

    for (size_t j = 0; j < pop.size(); j++)
    {
      if (pop.sols[j]->constraint == 0) {
        individual_objectives[j] = pop.sols[j]->obj[i];
      }
      else
      {
        individual_objectives[i] = pop.worst.obj[i] + pop.sols[j]->constraint;
      }
    }

    compute_ranks_asc(individual_objectives, individual_objectives_ranks);

    // add solutions to cluster
    cluster_size = std::min(cluster_size, pop.size());
    
    for (size_t j = 0; j < cluster_size; ++j) {
      clusters.back()->addSolution(pop.sols[individual_objectives_ranks[j]]);
    }

  }

}

// cluster registration,
// subfunction of estimateParameters

/* Re-assign cluster indices to achieve cluster registration,
* i.e. make cluster i in this generation to be the cluster that is
* closest to cluster i of the previous generation. The
* algorithm first computes all distances between clusters in
* the current generation and the previous generation. It also
* computes all distances between the clusters in the current
* generation and all distances between the clusters in the
* previous generation. Then it determines the two clusters
* that are the farthest apart. It randomly takes one of
* these two far-apart clusters and its r nearest neighbours.
* It also finds the closest cluster among those of the previous
* generation and its r nearest neighbours. All permutations
* are then considered to register these two sets. Subset
* registration continues in this fashion until all clusters
* are registered. */

void hicam::optimizer_t::mamalgam_cluster_registration(std::vector<cluster_pt> & clusters, const std::vector<cluster_pt> & previous_clusters, const vec_t & obj_ranges)
{

  size_t number_of_nearest_neighbours_in_registration = 7;

  // Compute distances between clusters[i] to previous_clusters[j].
  //----------------------------------------------------------------------------
  matrix_t distance_cluster_i_now_to_cluster_j_previous(clusters.size(), previous_clusters.size());

  for (size_t i = 0; i < clusters.size(); i++) {
    for (size_t j = 0; j < previous_clusters.size(); j++) {
      distance_cluster_i_now_to_cluster_j_previous[i][j] = clusters[i]->objective_distance(*previous_clusters[j], obj_ranges);
    }
  }

  // Compute distances between clusters[i] to clusters[j].
  //----------------------------------------------------------------------------
  matrix_t distance_cluster_i_now_to_cluster_j_now(clusters.size(), previous_clusters.size());

  for (size_t i = 0; i < clusters.size(); i++)
  {
    for (size_t j = 0; j < clusters.size(); j++)
    {
      if (i != j) {
        distance_cluster_i_now_to_cluster_j_now[i][j] = clusters[i]->objective_distance(*clusters[j], obj_ranges);
      }
      else {
        distance_cluster_i_now_to_cluster_j_now[i][j] = 0;
      }
    }
  }

  // Compute distances between previous_clusters[i] to previous_clusters[j].
  //----------------------------------------------------------------------------
  matrix_t distance_cluster_i_previous_to_cluster_j_previous(clusters.size(), previous_clusters.size());

  for (size_t i = 0; i < previous_clusters.size(); i++)
  {
    for (size_t j = 0; j < previous_clusters.size(); j++)
    {
      if (i != j) {
        distance_cluster_i_previous_to_cluster_j_previous[i][j] = previous_clusters[i]->objective_distance(*previous_clusters[j], obj_ranges);
      }
      else {
        distance_cluster_i_previous_to_cluster_j_previous[i][j] = 0;
      }
    }
  }

  std::vector<bool> clusters_now_already_registered(clusters.size(), false);
  std::vector<bool> clusters_previous_already_registered(previous_clusters.size(), false);

  std::vector<size_t> r_nearest_neighbours_now(number_of_nearest_neighbours_in_registration);
  std::vector<size_t> r_nearest_neighbours_previous(number_of_nearest_neighbours_in_registration);
  std::vector<size_t> nearest_neighbour_choice_best(number_of_nearest_neighbours_in_registration);

  size_t number_of_clusters_left_to_register = clusters.size();
  size_t number_of_clusters_to_register_by_permutation;

  int * sorted; // todo: make this a c++ version, this uses mergeSort

  while (number_of_clusters_left_to_register > 0)
  {
    
    if (number_of_clusters_left_to_register == number_of_nearest_neighbours_in_registration + 1) {
      number_of_nearest_neighbours_in_registration++;
    }
    
    // Find the two clusters in the current generation that are farthest apart and haven't been registered yet 
    int i_min = -1;
    int j_min = -1;
    double distance_largest = -1.0;

    for (size_t i = 0; i < clusters.size(); i++)
    {
      if (!clusters_now_already_registered[i])
      {
        for (size_t j = 0; j < clusters.size(); j++)
        {
          if ((i != j) && (!clusters_now_already_registered[j]))
          {
            if ((distance_largest < 0) || (distance_cluster_i_now_to_cluster_j_now[i][j] > distance_largest))
            {
              distance_largest = distance_cluster_i_now_to_cluster_j_now[i][j];
              i_min = (int) i;
              j_min = (int) j;
            }
          }
        }
      }
    }

    // if there is only one cluster remaining
    if (i_min == -1)
    {
      for (size_t i = 0; i < clusters.size(); i++) {
        if (!clusters_now_already_registered[i]) {
          i_min = (int) i;
          break;
        }
      }
    }

    // Find the r nearest clusters of one of the two far-apart clusters that haven't been registered yet 
    sorted = mergeSort(distance_cluster_i_now_to_cluster_j_now[i_min], (int)clusters.size());

    size_t j = 0;
    for (size_t i = 0; i < clusters.size(); i++)
    {
      if (!clusters_now_already_registered[sorted[i]])
      {
        r_nearest_neighbours_now[j] = sorted[i];
        clusters_now_already_registered[sorted[i]] = true;
        j++;
      }

      if (j == number_of_nearest_neighbours_in_registration && number_of_clusters_left_to_register - j != 1) {
        break;
      }

    }
    number_of_clusters_to_register_by_permutation = j;
    free(sorted);

    // Find the closest cluster from the previous generation 
    j_min = -1;
    double distance_smallest = -1.0;
    for (size_t j = 0; j < previous_clusters.size(); j++)
    {
      if (!clusters_previous_already_registered[j])
      {
        if ((distance_smallest < 0) || (distance_cluster_i_now_to_cluster_j_previous[i_min][j] < distance_smallest))
        {
          distance_smallest = distance_cluster_i_now_to_cluster_j_previous[i_min][j];
          j_min = (int) j;
        }
      }
    }

    // Find the r nearest clusters of one of the the closest cluster from the previous generation 
    sorted = mergeSort(distance_cluster_i_previous_to_cluster_j_previous[j_min], (int)clusters.size());
    j = 0;
    for (size_t i = 0; i < clusters.size(); i++)
    {
      if (clusters_previous_already_registered[sorted[i]] == 0)
      {
        r_nearest_neighbours_previous[j] = sorted[i];
        clusters_previous_already_registered[sorted[i]] = 1;
        j++;
      }

      if (j == number_of_nearest_neighbours_in_registration) {
        break;
      }

    }
    free(sorted);

    // Register the r selected clusters from the current and the previous generation 
    int number_of_cluster_permutations;
    int ** all_cluster_permutations = allPermutations((int)number_of_clusters_to_register_by_permutation, &number_of_cluster_permutations); // todo make this a c++ implementation

    distance_smallest = -1;
    for (size_t i = 0; i < (size_t)number_of_cluster_permutations; i++)
    {
      double distance = 0;
      for (j = 0; j < number_of_clusters_to_register_by_permutation; j++) {
        distance += distance_cluster_i_now_to_cluster_j_previous[r_nearest_neighbours_now[j]][r_nearest_neighbours_previous[all_cluster_permutations[i][j]]];
      }

      if ((distance_smallest < 0) || (distance < distance_smallest))
      {
        distance_smallest = distance;
        for (size_t j = 0; j < number_of_clusters_to_register_by_permutation; j++) {
          nearest_neighbour_choice_best[j] = r_nearest_neighbours_previous[all_cluster_permutations[i][j]];
        }
      }
    }

    for (size_t i = 0; i < (size_t)number_of_cluster_permutations; i++) {
      free(all_cluster_permutations[i]);
    }
    free(all_cluster_permutations);

    for (size_t i = 0; i < number_of_clusters_to_register_by_permutation; i++) {
      // selection_indices_of_cluster_members[nearest_neighbour_choice_best[i]] = selection_indices_of_cluster_members_before_registration[r_nearest_neighbours_now[i]];
      clusters[r_nearest_neighbours_now[i]]->previous = previous_clusters[nearest_neighbour_choice_best[i]];
      clusters[r_nearest_neighbours_now[i]]->previous->previous = nullptr; // otherwise we store the entire history of clusters.
      clusters[r_nearest_neighbours_now[i]]->number = clusters[r_nearest_neighbours_now[i]]->previous->number;
    }

    number_of_clusters_left_to_register -= number_of_clusters_to_register_by_permutation;
  }

  // permute clusters
  // std::vector<cluster_pt> clusters_copy(clusters.size());

  // for (size_t i = 0; i < clusters.size(); ++i) {
  //   clusters_copy[clusters[i]->previous->number] = clusters[i];
  // }

  // clusters = clusters_copy;
}

void hicam::optimizer_t::direct_cluster_registration_on_mean(std::vector<cluster_pt> & clusters, const std::vector<cluster_pt> & previous_clusters, const vec_t & obj_ranges) const
{

  // filter out the so_clusters
  std::vector<cluster_pt> remaining_clusters;

  bool so_clusters_matched = false;
  
  for (size_t i = 0; i < clusters.size(); ++i)
  {
    if(clusters[i]->objective_number > 0)
    {
      for(size_t j = 0; j < previous_clusters.size(); ++j)
      {
        if(clusters[i]->objective_number == previous_clusters[j]->objective_number)
        {
          
          clusters[i]->previous = previous_clusters[j];
          clusters[i]->previous->previous = nullptr; // make sure memory doesn't explode
          
          so_clusters_matched = true;
          break;
        }
      }
    }
  }
    
  // compute the distance of cluster[i] to the previous clusters
  for (size_t i = 0; i <  clusters.size(); ++i)
  {

    if(so_clusters_matched && clusters[i]->objective_number > 0) {
      continue;
    }
    
    clusters[i]->average_fitness(clusters[i]->objective_mean);
    // clusters[i]->compute_mean(clusters[i]->parameter_mean);

    double nearest_distance = 1e300;
    double current_distance;

    for (size_t j = 0; j < previous_clusters.size(); ++j)
    {
      if(so_clusters_matched && previous_clusters[j]->objective_number > 0) {
        continue;
      }
      
      current_distance = clusters[i]->objective_mean.scaled_euclidean_distance(previous_clusters[j]->objective_mean, obj_ranges);
      // previous_clusters[j]->compute_mean(previous_clusters[j]->parameter_mean);
      // current_distance = (clusters[i]->parameter_mean - previous_clusters[j]->parameter_mean).norm();

      if (current_distance < nearest_distance)
      {
        nearest_distance = current_distance;
        clusters[i]->previous = previous_clusters[j];
        clusters[i]->previous->previous = nullptr; // make sure memory doesn't explode
      }
    }

  }
}

