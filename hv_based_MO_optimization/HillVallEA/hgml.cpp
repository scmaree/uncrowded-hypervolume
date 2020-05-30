/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA


*/

#include "hillvallea.hpp"
#include "population.hpp"
#include "mathfunctions.hpp"
#include "hgml.hpp"

//---------------------------------------------------------------------
//
// Edges
//
//---------------------------------------------------------------------

hillvallea::edge_t::edge_t() {};
hillvallea::edge_t::~edge_t() {};

hillvallea::edge_t::edge_t(const hgml_cluster_pt from, const hgml_cluster_pt to, const double edge_length)
{
  
  this->from = from;
  this->to = to;
  this->edge_length = edge_length;
  
}

// defined as static!
bool hillvallea::edge_t::shorter_edge(const edge_pt edge1, const edge_pt edge2)
{
  
  if (edge1->edge_length < edge2->edge_length)
    return true;
  
  return false;
  
}

void hillvallea::edge_t::update_edge_length()
{
  
  edge_length = sqrt(from->weight * to->weight) * (from->mean_vector - to->mean_vector).norm();
}


hillvallea::hgml_cluster_t::hgml_cluster_t() : population_t() { }
hillvallea::hgml_cluster_t::~hgml_cluster_t() { }

// computes the spearman rank correlation between the fitness and the probability.
double hillvallea::hgml_cluster_t::compute_fitness_correlation()
{
  
  // set the probability rank of the individuals
  this->set_probability_rank();
  
  // set the fitness rank of the individuals
  // important! Set the fitness rank as last, such that the population is still sorted on fitness!!!!!!!!!!!!!!!!
  this->set_fitness_rank();
  
  fitness_correlation = 0.0;
  double N = (double)size();
  double rankdifference;
  
  if (N <= 1)
    return fitness_correlation;
  
  for (auto sol : sols) {
    rankdifference = (double)(sol->fitness_rank - sol->probability_rank);
    fitness_correlation += rankdifference*rankdifference;
  }
  
  // compute the predictive value
  // https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
  fitness_correlation = 1 - fitness_correlation / (N*(N*N - 1) / 6.0);
  
  return fitness_correlation;
  
}



// set probability rank
void hillvallea::hgml_cluster_t::set_probability_rank()
{
  
  for (auto sol : sols) {
    sol->probability = normpdf(mean_vector, cholesky_matrix, inverse_cholesky_matrix, sol->param);
  }
  
  // sort the population on probability
  sort_on_probability();
  
  // set the fitness rank
  for (size_t i = 0; i < sols.size(); ++i) {
    sols[i]->probability_rank = (int) i;
  }
  
}

// blegh, vergeten dit te implementeren..
void hillvallea::hgml_cluster_t::update_cholesky()
{
  
  // apply the multiplier
  choleskyDecomposition(covariance_matrix, cholesky_matrix);
  
  int n = (int)covariance_matrix.rows();
  inverse_cholesky_matrix.setRaw(matrixLowerTriangularInverse(cholesky_matrix.toArray(), n),n,n);
  
}

// blegh, vergeten dit te implementeren..
void hillvallea::hgml_cluster_t::update_cholesky_univariate()
{
  
  // apply the multiplier
  choleskyDecomposition_univariate(covariance_matrix, cholesky_matrix);
  
  int n = (int)covariance_matrix.rows();
  inverse_cholesky_matrix.reset(n, n, 0.0);
  for (int i = 0; i < n; ++i) {
    inverse_cholesky_matrix[i][i] = 1.0 / cholesky_matrix[i][i];
  }
  
}


hillvallea::hgml_t::hgml_t() {};
hillvallea::hgml_t::~hgml_t() {};



// Hierarchical Clustering
// based on pop, (is non-const because we sort stuff as well)
// result is clusters
void hillvallea::hgml_t::hierarchical_clustering(population_t & pop, std::vector<population_pt> & clusters)
{
  
  clusters.clear();
  // if the pop is empty, do nothing.
  if (pop.size() == 0) {
    return;
  }
  
  // if there should be only one cluster, we do not cluster.
  if (pop.size() == 1)
  {
    population_pt cluster = std::make_shared<population_t>();
    cluster->addSolutions(pop);
    clusters.push_back(cluster);
    
    return;
  }
  
  // allocation
  std::list<edge_pt> edges;
  std::vector<hgml_cluster_pt> temp_clusters;
  std::vector<hgml_cluster_pt> kernels;
  std::vector<hgml_cluster_pt> cluster_tree;

  // nearest better tree
  //---------------------------------------------------------------------------------
  // first, number the individuals
  // pop.set_fitness_rank(); // also sorts the population
  pop.sort_on_fitness(); // todo: maybe i need ranks?
  
  generate_nearest_better_tree(pop, kernels, edges);
  size_t N = kernels.size(); // can be different from popsize if clusters are terminated.
  
  // Merging clusters
  //-------------------------------------------------------------------------------
  // merge relations (parent1 + parent2) = child.
  std::vector<hgml_cluster_pt> parent1;
  std::vector<hgml_cluster_pt> parent2;
  std::vector<hgml_cluster_pt> child;
  
  merge_edges(edges, cluster_tree, parent1, parent2, child);
  
  
  // FDC calculation & select best one
  //--------------------------------------------------------
  vec_t fdc(N, -1.0); // -1 is the worst fdc there is
  size_t best_fdc_index;
  compute_dfcs(fdc, best_fdc_index, temp_clusters, parent1, parent2, child);
  
  // convert internal clusters to clusters
  for(size_t i = 0; i < temp_clusters.size(); ++i)
  {
    population_pt cluster = std::make_shared<population_t>();
    cluster->sols = temp_clusters[i]->sols;
    clusters.push_back(cluster);
  }
  
}


//------------------------------------------------------------------------
void hillvallea::hgml_t::generate_nearest_better_tree(population_t & pop, std::vector<hgml_cluster_pt> & kernels, std::list<edge_pt> & edges) const
{
  
  // if the pop is empty, return;
  if(pop.size() == 0) {
    return;
  }
  
  kernels.clear();
  size_t d = pop.problem_size();
  
  // default covariance matrix
  matrix_t kernel_covariance;
  kernel_covariance.setIdentity(d,d);
 
  for (auto sol : pop.sols)
  {
    hgml_cluster_pt kernel = std::make_shared<hgml_cluster_t>();
    kernel->mean_vector = sol->param;
    kernel->covariance_matrix = kernel_covariance;
    kernel->sols.push_back(sol);
    kernel->weight = (double) kernel->size();
    kernels.push_back(kernel);
  }

  // Create an tree with kernels.size() nodes
  // Assumes the pop is sorted! thus all j < i are better  :  O(N^2d)
  //-------------------------------------------------
  edges.clear();
  size_t nearest_better = 0;
  double nearest_distance = 1e308;
  double current_distance_to_j = 0.0;
  
  nearest_distance = 0; //[0] is the best-so-far, it has no nearest_better
  for (size_t i = 1; i < kernels.size(); ++i)
  {
    nearest_distance = 1e308;
    for (size_t j = 0; j < i; ++j)
    {
      // compute the distance to from i to j.
      current_distance_to_j = sqrt(kernels[i]->weight * kernels[j]->weight) *  kernels[i]->sols[0]->param_distance(*kernels[j]->sols[0]);
      
      if (current_distance_to_j < nearest_distance)
      {
        nearest_distance = current_distance_to_j;
        nearest_better = j;
      }
    }
    
    edge_pt edge = std::make_shared<edge_t>(kernels[i], kernels[nearest_better], nearest_distance);
    edges.push_back(edge);
  }
  
}

void hillvallea::hgml_t::merge_edges(std::list<edge_pt> & edges, std::vector<hgml_cluster_pt> & cluster_tree, std::vector<hgml_cluster_pt> &parent1, std::vector<hgml_cluster_pt> & parent2, std::vector<hgml_cluster_pt> & child) const
{
  
  size_t N = edges.size() + 1;
  parent1.clear();
  parent2.clear();
  child.clear();
  cluster_tree.clear();
  
  // start the merging
  // for N individuals, we can do N-1 merges of two into a single new.
  for (size_t i = 0; i < N - 1; ++i)
  {
    
    // find the minimum edge in the remaining edges
    edge_pt min_edge;
    double min_edge_length = 1e308;
    
    for (auto edge : edges) {
      if (edge->edge_length < min_edge_length)
      {
        min_edge_length = edge->edge_length;
        min_edge = edge;
      }
    }
    
    // 2. the first edge, merge it.
    hgml_cluster_pt cluster = std::make_shared<hgml_cluster_t>();
    cluster_tree.push_back(cluster);
    child.push_back(cluster);
    parent1.push_back(min_edge->from);
    parent2.push_back(min_edge->to);
    
    // 3. MOM update of the cluster parameters
    double w_from = min_edge->from->weight;
    double w_to = min_edge->to->weight;
    double w = w_from + w_to;
    
    cluster->weight = w;
    // cluster->mean_vector = (w_from / w)*min_edge->from->mean_vector + (w_to / w)*min_edge->to->mean_vector;
    
    // add the sols to the cluster
    cluster->sols.clear();
    cluster->addSolutions(*min_edge->from);
    cluster->addSolutions(*min_edge->to);
    
    cluster->mean(cluster->mean_vector);
    
    
    // delete the minimum edge, and update the remaining.
    // replace all previously occuring min_edge->from, min_edge->to
    edges.remove(min_edge);
    
    for (auto other_edge : edges)
    {
      if (other_edge->from == min_edge->to) {
        other_edge->from = cluster;
        other_edge->edge_length = (w_from * (other_edge->to->mean_vector - min_edge->from->mean_vector).norm() + w_to * (other_edge->to->mean_vector - min_edge->to->mean_vector).norm()) / w;
      }
      
      if (other_edge->to == min_edge->from || other_edge->to == min_edge->to) {
        other_edge->to = cluster;
        other_edge->edge_length = (w_from * (other_edge->from->mean_vector - min_edge->from->mean_vector).norm() + w_to * (other_edge->from->mean_vector - min_edge->to->mean_vector).norm()) / w;
      }
    }
    
  }
  
}

void hillvallea::hgml_t::compute_dfcs(vec_t & dfc, size_t & best_dfc_index, std::vector<hgml_cluster_pt> & clusters, std::vector<hgml_cluster_pt> & parent1, std::vector<hgml_cluster_pt> & parent2, std::vector<hgml_cluster_pt> & child)
{
  
  bool use_univariate = false;
  double roundfactor = 0.05;
  
  
  // only possible if the popsize == 1
  if (child.size() == 0)
  {
    dfc.resize(1);
    dfc[0] = 1.0;
    best_dfc_index = 0;
    
    return;
  }
  
  size_t N = child.size();
  size_t min_clustersize = 1 + child[0]->problem_size();
  
  if(use_univariate) {
    min_clustersize = 2;
  }
  
  dfc.fill(-1.0);
  dfc.resize(N+1, -1.0); // -1 is the worst fdc there is
  
  // initial fdc for 1 cluster
  if (use_univariate)
  {
    child[N - 1]->covariance_univariate(child[N - 1]->mean_vector, child[N - 1]->covariance_matrix);
    child[N - 1]->update_cholesky_univariate();
  }
  else
  {
    child[N - 1]->covariance(child[N - 1]->mean_vector, child[N - 1]->covariance_matrix);
    child[N - 1]->update_cholesky();
  }
  
  child[N - 1]->compute_fitness_correlation();
  dfc[N] = child[N - 1]->fitness_correlation * (child[N - 1]->sols.size() / (double)N);
  double best_fdc = dfc[N];
  best_dfc_index = N;
  
  // after each merge, evaluate the fdc recursively
  for (int i = (int)N - 1; i > 0; --i)
  {
    
    // break if any of the two parents is smaller than d+1
    if (parent1[i]->size() < min_clustersize || parent2[i]->size() < min_clustersize) {
      break;
    }
    
    // if the parents are large enough, compute the covariance & FDC
    if (use_univariate)
    {
      parent1[i]->covariance_univariate(parent1[i]->mean_vector, parent1[i]->covariance_matrix);
      parent1[i]->update_cholesky_univariate();
      
      parent2[i]->covariance_univariate(parent2[i]->mean_vector, parent2[i]->covariance_matrix);
      parent2[i]->update_cholesky_univariate();
    }
    else
    {
      parent1[i]->covariance(parent1[i]->mean_vector, parent1[i]->covariance_matrix);
      parent1[i]->update_cholesky();
      
      parent2[i]->covariance(parent2[i]->mean_vector, parent2[i]->covariance_matrix);
      parent2[i]->update_cholesky();
    }
    
    parent1[i]->compute_fitness_correlation();
    parent2[i]->compute_fitness_correlation();
    
    dfc[i] = dfc[i + 1] - child[i]->fitness_correlation * (child[i]->size() / (double)N)
    + parent1[i]->fitness_correlation * (parent1[i]->size() / (double)N)
    + parent2[i]->fitness_correlation * (parent2[i]->size() / (double)N);
    
    // remember the best fdc
    if (dfc[i] > 0.0 && round(dfc[i] / roundfactor)*roundfactor > best_fdc)
    {
      best_fdc = round(dfc[i]/ roundfactor)*roundfactor;
      best_dfc_index = i;
    }
    
    //if (maximum_number_of_clusters > 0 && (int)(N) + 1 - i >= maximum_number_of_clusters) {
    //  break;
    //}
    
  }
  /*
  for (size_t i = 0; i < dfc.size(); ++i)
  {
    if(i == best_dfc_index) {
      std::cout << "*";
    }
    
    if (dfc[i] >= 0) {
      std::cout << (N-i+1) << ":" << dfc[i] << "\n";
    }
    
  }
  */
  // recover the cluster set
  // create a cluster list to cheaply add/remove.
  //--------------------------------------------------------
  std::list<hgml_cluster_pt> cluster_list;
  cluster_list.push_back(child[N - 1]);
  
  for (int i = (int)N - 1; i >= (int)best_dfc_index; --i) {
    cluster_list.remove(child[i]);
    cluster_list.push_back(parent1[i]);
    cluster_list.push_back(parent2[i]);
  }
  
  // copy the list to the vector
  clusters.clear();
  for (auto cluster : cluster_list) {
    clusters.push_back(cluster);
  }
  
}



void hillvallea::hgml_t::nearest_better_clustering(population_t & pop, std::vector<population_pt> & clusters)
{
  
  clusters.clear();
  double mean_edge_length_cutoff_multiplier = 2.0;
  
  // if the pop is empty, do nothing.
  if (pop.size() == 0) {
    return;
  }
  
  // if there should be only one cluster, we do not cluster.
  if (pop.size() == 1)
  {
    population_pt cluster = std::make_shared<population_t>();
    cluster->addSolutions(pop);
    clusters.push_back(cluster);
    
    return;
  }
  
  // allocation
  std::list<edge_pt> edges;
  std::vector<hgml_cluster_pt> temp_clusters;
  std::vector<hgml_cluster_pt> kernels;
  std::vector<hgml_cluster_pt> cluster_tree;
  
  // nearest better tree
  //---------------------------------------------------------------------------------
  // first, number the individuals
  // pop.set_fitness_rank(); // also sorts the population
  pop.sort_on_fitness(); // todo: maybe i need ranks?
  
  generate_nearest_better_tree(pop, kernels, edges);
  
  double mean_edge_length = 0.0;
  
  for(edge_pt edge : edges) {
    mean_edge_length += edge->edge_length;
  }
  
  mean_edge_length /= edges.size();
  

  population_pt cluster = std::make_shared<population_t>();
  cluster->sols.push_back(edges.front()->to->sols[0]); // TO!
  clusters.push_back(cluster);
  cluster->sols.back()->cluster_number = (int) clusters.size() - 1;
  
  for(edge_pt edge : edges)
  {
    if (edge->edge_length > mean_edge_length * mean_edge_length_cutoff_multiplier)
    {
      // cut the edge! New cluster
      population_pt cluster = std::make_shared<population_t>();
      cluster->sols.push_back(edge->from->sols[0]);
      clusters.push_back(cluster);
      cluster->sols.back()->cluster_number = (int) clusters.size() - 1;
    }
    else
    {
      // add edge->from to same cluster as edge->to
      clusters[edge->to->sols[0]->cluster_number]->sols.push_back(edge->from->sols[0]);
      clusters[edge->to->sols[0]->cluster_number]->sols.back()->cluster_number = edge->to->sols[0]->cluster_number;
    }
  }
  
}
