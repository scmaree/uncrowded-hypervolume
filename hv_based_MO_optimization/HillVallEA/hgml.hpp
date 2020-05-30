#pragma once

/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

*/

#include "hillvallea_internal.hpp"
#include "optimizer.hpp"
#include <list>

namespace hillvallea
{
  
  class hgml_cluster_t : public population_t {
    
  public:
    
    hgml_cluster_t();
    ~hgml_cluster_t();
    
    // sample parameters
    vec_t mean_vector;                       // sample mean
    matrix_t covariance_matrix;              // sample covariance matrix C
    matrix_t cholesky_matrix;                // decomposed covariance matrix C = LL^T
    matrix_t inverse_cholesky_matrix;        // inverse of the cholesky decomposition

    // cluster parameters
    double weight;                    // weight of the population (in clustering)
    double fitness_correlation;       // Fitness-density rank correlation
    
    // Fitness-Density Correlation
    //----------------------------------------------
    double compute_fitness_correlation();
    void set_probability_rank();
    
    
    void update_cholesky();
    void update_cholesky_univariate();
    
    
  };
  
  typedef std::shared_ptr<hgml_cluster_t> hgml_cluster_pt;
  
  
  class edge_t {
    
  public:
    
    // constructor & destructor
    edge_t();
    edge_t(const hgml_cluster_pt from, const hgml_cluster_pt to, double edge_length);
    ~edge_t();
    
    // directed edge x->y
    hgml_cluster_pt from;
    hgml_cluster_pt to;
    double edge_length;
    
    // compare edge_length
    static bool shorter_edge(const edge_pt edge1, const edge_pt edge2);
    
    // update edge length
    void update_edge_length();
    
  };
  
  typedef std::shared_ptr<edge_t> edge_pt;
  
  
  class hgml_t {
    
  public:
    
    hgml_t();
    ~hgml_t();
    
    // Hierarchical Clustering
    void hierarchical_clustering(population_t & pop, std::vector<population_pt> & clusters);
    void generate_nearest_better_tree(population_t & pop, std::vector<hgml_cluster_pt> & kernels, std::list<edge_pt> & edges) const;
    void merge_edges(std::list<edge_pt> & edges, std::vector<hgml_cluster_pt> & cluster_tree, std::vector<hgml_cluster_pt> &parent1, std::vector<hgml_cluster_pt> & parent2, std::vector<hgml_cluster_pt> & child) const;
    
    void compute_dfcs(vec_t & fdc,size_t & best_fdc_index,std::vector<hgml_cluster_pt> & clusters, std::vector<hgml_cluster_pt> & parent1, std::vector<hgml_cluster_pt> & parent2, std::vector<hgml_cluster_pt> & child );
    
    void nearest_better_clustering(population_t & pop, std::vector<population_pt> & clusters);
    
    
  };
  
  
}
