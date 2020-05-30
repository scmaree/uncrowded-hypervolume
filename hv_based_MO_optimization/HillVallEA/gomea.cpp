/* GOMEA.CPP
 
 AMaLGaM as part of HillVallEA
 
 Implementation by S.C. Maree
 s.c.maree[at]amc.uva.nl
 github.com/SCMaree/HillVallEA
 
 
 */

#include "population.hpp"
#include "mathfunctions.hpp"
#include "gomea.hpp"
#include "fitness.h"
#include <sys/time.h>
#include "adam.hpp"


// init amalgam default parameters
hillvallea::gomea_t::gomea_t(const size_t number_of_parameters, const vec_t & lower_param_bounds, const vec_t & upper_param_bounds, double init_univariate_bandwidth, int version, fitness_pt fitness_function, rng_pt rng) : optimizer_t(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, fitness_function, rng)
{
  // init settings
  
  population_initialized = false;
  sample_conditionally = false;
  sample_bayesian_factorization = false;
  gradient_step = false;
  number_of_generations = 0;
  weighted_number_of_evaluations = 0;
  active = true;
  no_improvement_stretch = 0;
  
  // default parameters
  selection_fraction = 0.35; // copy to optimizer_t external
  distribution_multiplier_decrease = 0.9;
  st_dev_ratio_threshold           = 1.0;
  // maximum_no_improvement_stretch   = (int) (25 + number_of_parameters);
  
  // linkage learning settings
  static_linkage = 0;
  random_linkage = 0;
  learn_linkage_tree_from_distance_matrix = 0;
  dynamic_filter_large_FOS_elements = false;
  always_keep_largest_FOS_element = false;
  eta_ams = 1.0;
  eta_cov = 1.0;
  use_boundary_repair = fitness_function->use_boundary_repair; // BOUNDARY REPAIR!
  
  FOS_element_lb = (int) fitness_function->fos_element_size_lower_bound;
  FOS_element_ub = (int) fitness_function->fos_element_size_upper_bound; //
  
  if(FOS_element_ub <= 0) {
    FOS_element_ub = (int) number_of_parameters;
  }

  this->version = version;
  switch (version) {
    case 50: // full linkage model
      static_linkage = true;
      random_linkage = true;
      FOS_element_lb = (int) number_of_parameters;
      FOS_element_ub = (int) number_of_parameters;
      break;
    case 59: // bayesian linkage model from marginal (64)
      static_linkage = true;
      FOS_element_ub = FOS_element_lb;
      learn_linkage_tree_from_distance_matrix = true;
      sample_bayesian_factorization = true;
      break;
    case 61: // univariate linkage model (thus static)
      static_linkage = true;
      FOS_element_lb = 1;
      FOS_element_ub = 1;
      break;
    case 62: // non-static non-filtered linkage tree
      static_linkage = false;
      learn_linkage_tree_from_distance_matrix = true;
      break;
    case 63: // static non-filtered linkage tree
      static_linkage = true;
      FOS_element_ub = (int) fitness_function->covariance_block_size;
      learn_linkage_tree_from_distance_matrix = true;
      // dynamic_filter_large_FOS_elements = true;
      // always_keep_largest_FOS_element = true;
      break;
    case 64: // static marginal product linkage model from distance matrix (if available)
      static_linkage = true;
      FOS_element_lb = (int) fitness_function->covariance_block_size;
      FOS_element_ub = (int) fitness_function->covariance_block_size;
      learn_linkage_tree_from_distance_matrix = true;
      break;
    case 65: // non-static filtered random LT
      static_linkage = false;
      dynamic_filter_large_FOS_elements = true;
      learn_linkage_tree_from_distance_matrix = true;
      break;
    case 66: // non-static filtered LT from distance matrix (if available)
      static_linkage = false;
      dynamic_filter_large_FOS_elements = true;
      learn_linkage_tree_from_distance_matrix = true;
      break;
    case 74: // static marginal product conditional linkage model from distance matrix (if available)
      static_linkage = true;
      FOS_element_ub = FOS_element_lb;
      learn_linkage_tree_from_distance_matrix = true;
      sample_conditionally = true;
      break;
    case 76: // non-static filtered Conditional LT from distance matrix (if available)
      static_linkage = false;
      dynamic_filter_large_FOS_elements = true;
      learn_linkage_tree_from_distance_matrix = true;
      sample_conditionally = true;
      break;
    case 84: // hybrid of 64 with gradient step (ADAM)
      static_linkage = true;
      FOS_element_ub = FOS_element_lb;
      learn_linkage_tree_from_distance_matrix = true;
      gradient_step = true;
      break;
    default:
      std::cout << "warning GOMEA version not found.";
      break;
  }
  
  // sanity check: we can only learn from a distance matrix if its there.
  if(!fitness_function->linkage_learning_distance_matrix_available) {
    learn_linkage_tree_from_distance_matrix = false;
    random_linkage = true;
  }
  

  maximum_no_improvement_stretch = (int) (25 + FOS_element_lb);
  distribution_multiplier_increase = 1.0/distribution_multiplier_decrease;
}

hillvallea::gomea_t::~gomea_t()
{

};



hillvallea::optimizer_pt hillvallea::gomea_t::clone() const
{
  
  gomea_pt opt = std::make_shared<gomea_t>(number_of_parameters, lower_param_bounds, upper_param_bounds, init_univariate_bandwidth, version, fitness_function, rng);
  
  // Optimizer data members
  //-------------------------------------------
  opt->active = active;
  opt->number_of_parameters = number_of_parameters;
  opt->lower_param_bounds = lower_param_bounds;
  opt->upper_param_bounds = upper_param_bounds;
  opt->fitness_function = fitness_function;
  opt->number_of_generations = number_of_generations;
  opt->rng = rng;
  opt->pop = std::make_shared<population_t>(); //!! A copy of the contents, not the pointer
  opt->pop->addSolutions(*pop);
  opt->best = best;
  opt->average_fitness_history = average_fitness_history;
  opt->selection_fraction= selection_fraction;
  opt->init_univariate_bandwidth= init_univariate_bandwidth;
  
  // Stopping criteria
  //----------------------------------------------------------------------------------
  opt->maximum_no_improvement_stretch = maximum_no_improvement_stretch;
  opt->param_std_tolerance = param_std_tolerance;
  opt->fitness_std_tolerance = fitness_std_tolerance;
  
  std::cout << "Warning, clone not implemented for GOMEA" << std::endl;
  return opt;
}



// Algorithm Name
std::string hillvallea::gomea_t::name() const { return "GOMEA"; }

// Initial initialization of the algorithm
// Population should be sorted on fitness (fittest first)
void hillvallea::gomea_t::initialize_from_population(population_pt pop, size_t target_popsize)
{
  
  // set population sizes
  population_initialized = true;
  population_size = (int) target_popsize;
  initial_population_size = std::min(population_size, (int) pop->size());
  
  no_improvement_stretch = 0;
  number_of_generations = 0;
  
  // copy population
  this->pop = pop;
  pop->sort_on_fitness();
  
  // set individual NO improvement Stretch
  for(size_t j = 0; j < pop->size(); ++j) {
    pop->sols[j]->NIS = 0;
  }
  
  // set best
  best = solution_t(*pop->sols[0]);
  
  // fill population now with copies
  pop->sols.resize(target_popsize);
  
  for(size_t j = initial_population_size; j < population_size; j++ ) {
    pop->sols[j] = copySolution(pop->sols[j%initial_population_size]);
  }
  
  // allocate stuff
  mean_vector.resize(number_of_parameters,0.0);
  old_mean_vector.resize(number_of_parameters,0.0);
  
  if(gradient_step)
  {
    gradient_methods.resize(pop->size());
    
    vec_t lower_param_range(number_of_parameters, 1e300);
    vec_t upper_param_range(number_of_parameters, -1e300);
    
    for(size_t i = 0; i < pop->sols.size(); ++i) {
      for(size_t j = 0; j < number_of_parameters; ++j) {
        if(pop->sols[i]->param[j] < lower_param_range[j]) { lower_param_range[j] = pop->sols[i]->param[j]; }
        if(pop->sols[i]->param[j] > upper_param_range[j]) { upper_param_range[j] = pop->sols[i]->param[j]; }
      }
    }
    for(size_t i = 0; i < gradient_methods.size(); ++i) {
      gradient_methods[i] = std::make_shared<adam_t>(fitness_function, 50, lower_param_range, upper_param_range, -1, -1, false, 0.0, 1234, false, false, "","",0.01,1e-8);
      gradient_methods[i]->accept_only_improvements = false;
    }
  }
}

void hillvallea::gomea_t::generation(size_t sample_size, int & external_number_of_evaluations)
{
  
  //if(sample_size != population_size) {
  //  std::cout << "Warning, cannot run GOMEA with other popsize than initialized" << std::endl;
  //}
  
  double current_number_of_evaluations = weighted_number_of_evaluations;
  
  // if its the first generation, we have to do some other stuff since
  // this core search algorithm can be initialized with a different population size.
  fitness_function->sort_population_parameters(*pop, *linkage_model);
  population_t selection;
  
  if(number_of_generations == 0) {
    makeSelection(*pop, selection_fraction, selection, (number_of_generations == 0), (int) initial_population_size);
  } else {
    makeSelection(*pop, selection_fraction, selection);
  }
  
  // Maintain the best
  pop->sols[0] = copySolution(selection.sols[0]);
  
  // Estimate Sample mean (and save old for AMS)
  estimateMeanVectorML(selection, mean_vector); //, old_mean_vector, mean_shift_vector);
  mean_shift_vector = mean_vector - old_mean_vector;
  old_mean_vector = mean_vector;
  
  // learn a linkage tree (if needed)
  if( !static_linkage || number_of_generations == 0)
  {
    matrix_t matrix;
    
    if(learn_linkage_tree_from_distance_matrix)
    {
      if( !static_linkage && fitness_function->dynamic_linkage_learning_distance_matrix_available) {
        fitness_function->dynamic_linkage_learning_distance_matrix(matrix, *pop);
      } else {
        fitness_function->linkage_learning_distance_matrix(matrix);
      }
    } else {
      estimateFullCovarianceMatrixML(matrix, selection, mean_vector, init_univariate_bandwidth);
    }
    
    
    // set an upper bound based on the population size.
    int temp_FOS_element_ub = FOS_element_ub;
    int new_temp_FOS_element_ub;
    if(dynamic_filter_large_FOS_elements)
    {
      int selection_size = population_size * selection_fraction;
      temp_FOS_element_ub = std::min(selection_size - 1,temp_FOS_element_ub);
      temp_FOS_element_ub = std::max(FOS_element_lb,temp_FOS_element_ub); // sanity check
    }
    new_temp_FOS_element_ub = temp_FOS_element_ub;
    
    linkage_model = updateLinkageTree (matrix, linkage_model, learn_linkage_tree_from_distance_matrix, random_linkage, static_linkage,
         number_of_generations, FOS_element_lb, new_temp_FOS_element_ub, always_keep_largest_FOS_element,  number_of_parameters, sample_bayesian_factorization, rng );

    if(new_temp_FOS_element_ub != temp_FOS_element_ub) {
      // with Bayesian sampling, we first make a marginal linkage model and later change it to a full (with bayesian factorized covariance matrix).
      // updateLinkageTree can update the upper bound, so we have to correspond that back into the FOS_element_ub.
      FOS_element_ub = new_temp_FOS_element_ub;
    }
    
    if(sample_conditionally) {
      fitness_function->set_conditional_dependencies(*linkage_model, *pop);
    }
    
    initializeCovarianceMatrices(linkage_model);
    
    // std::cout << distribution_multipliers << std::endl << std::endl;
  }
  
  // re-evaluate everything once in a while to get rid of roundoff errors (which occur surprisingly often)
  if( fitness_function->partial_evaluations_available && fitness_function->has_round_off_errors_in_partial_evaluations && (number_of_generations+1) % 50 == 0 ) {
    evaluateCompletePopulation( );
  }
  solution_t current_elite(*selection.sols[0]);
  generateAndEvaluateNewSolutionsToFillPopulation( current_elite );
  
  external_number_of_evaluations += round(weighted_number_of_evaluations - current_number_of_evaluations); // not the most elegant way
  number_of_generations++;
  average_fitness_history.push_back(pop->average_fitness());
  
}

size_t hillvallea::gomea_t::recommended_popsize(const size_t problem_dimension) const
{
  if(FOS_element_ub == 1) {
    return (size_t)std::max((double)((size_t)((2.0 / selection_fraction) + 1)), 10.0*pow((double)problem_dimension, 0.5));
  } else {
    return (size_t)std::max((double)((size_t)((2.0 / selection_fraction) + 1)), 17.0 + 3.0*pow((double)problem_dimension, 1.5));
  }
}


// Fos.c
namespace hillvallea
{
  void printFOS( FOS_pt fos )
  {
    int i,j;
    printf("{");
    for( i = 0; i < fos->length(); i++ )
    {
      printf("[");
      for( j = 0; j < fos->sets[i]->length(); j++ )
      {
        printf("%lu", fos->sets[i]->params[j]);
        if( j != fos->sets[i]->length()-1)
          printf(",");
      }
      printf("]");
      printf("\n");
    }
    printf("}\n");
  }
  
  // construct a similarity matrix from a distance matrix (invert_input_matrix = true) or a correlation matrix (else)
  void getSimilarityMatrix(const matrix_t & similarity_matrix, bool invert_input_matrix, matrix_t & S_matrix, int * index_order)
  {

    size_t number_of_parameters = similarity_matrix.rows();
    
    S_matrix.resize(number_of_parameters, number_of_parameters);
    
    if(invert_input_matrix)
    {
      for(size_t i = 0; i < number_of_parameters-1; i++ )
      {
        for(size_t j = i+1; j < number_of_parameters; j++ )
        {
          S_matrix[i][j] = 1.0 / similarity_matrix[index_order[i]][index_order[j]];
          S_matrix[j][i] = S_matrix[i][j];
        }
        S_matrix[i][i] = 0.0;
      }
    }
    else
    {
      // Compute Mutual Information matrix from Similarity matrix
      matrix_t MI_matrix;
      computeMIMatrix(MI_matrix, similarity_matrix);
      
      for(size_t i = 0; i < number_of_parameters; i++ ) {
        for(size_t j = 0; j < number_of_parameters; j++ ) {
          S_matrix[i][j] = MI_matrix[index_order[i]][index_order[j]];
        }
      }
      
      for(size_t i = 0; i < number_of_parameters; i++ ) {
        S_matrix[i][i] = 0;
      }
    }
  }
  
  FOS_pt learnLinkageTree(
     const matrix_t & similarity_matrix,
     const bool invert_input_matrix, // the input matrix should be a similarty matrix, i.e.,
     const bool random_linkage_tree, // ignores the input matrix
     const size_t number_of_parameters,
     rng_pt rng,
     const int FOS_element_ub // for efficiency
     )
  {
    char     done;
    int      i, j, *order, FOS_index;
    double   mul0, mul1;
    FOS_pt new_FOS = std::make_shared<FOS_t>();
    size_t r0, r1, rswap;
    
    std::vector<size_t> indices;
    
    std::vector<std::vector<size_t>> mpm;
    std::vector<size_t> mpm_number_of_indices;
    int mpm_length;
    
    std::vector<std::vector<size_t>> mpm_new;
    std::vector<size_t> mpm_new_number_of_indices;
    int mpm_new_length;
    
    std::vector<size_t> NN_chain;
    int NN_chain_length;
    
    /* Initialize MPM to the univariate factorization */
    order                 = randomPermutation( (int) number_of_parameters, *rng );
    mpm.resize(number_of_parameters);
    mpm_number_of_indices.resize(number_of_parameters);
    mpm_length            = (int) number_of_parameters;

    for( i = 0; i < number_of_parameters; i++ )
    {
      indices.resize(1);
      indices[0]               = order[i];
      mpm[i]                   = indices;
      mpm_number_of_indices[i] = 1;
    }
    
    // Initialize LT to the initial MPM
    //-------------------------------------------
    new_FOS = std::make_shared<FOS_t>();
    new_FOS->sets.resize(number_of_parameters + number_of_parameters - 1);
    FOS_index = 0;
    
    for(size_t i = 0; i < new_FOS->sets.size(); ++i) {
      new_FOS->sets[i] = std::make_shared<FOS_element_t>();
    }
    
    for( i = 0; i < mpm_length; i++ )
    {
      new_FOS->sets[FOS_index]->params  = mpm[i];
      // new_FOS->set_length[FOS_index] = mpm_number_of_indices[i];
      FOS_index++;
    }
    
    // Initialize similarity matrix
    //-------------------------------------------
    matrix_t S_matrix;
    vec_t S_vector;
    
    if(random_linkage_tree)
    {
      S_vector.resize(number_of_parameters);
      for(size_t i = 0; i < number_of_parameters; i++ ) {
        S_vector[i] = randomRealUniform01(*rng);
      }
    } else {
      getSimilarityMatrix(similarity_matrix, invert_input_matrix, S_matrix, order);
    }
    free( order );
    
    NN_chain.resize(number_of_parameters + 2);
    NN_chain_length = 0;
    done            = 0;
    while( !done )
    {
      if( NN_chain_length == 0 )
      {
        NN_chain[NN_chain_length] = randomInt( mpm_length, *rng );
        NN_chain_length++;
      }
      
      if( NN_chain[NN_chain_length-1] >= mpm_length ) NN_chain[NN_chain_length-1] = mpm_length-1;
      
      while( NN_chain_length < 3 )
      {
        NN_chain[NN_chain_length] = determineNearestNeighbour( NN_chain[NN_chain_length-1], S_matrix, mpm_number_of_indices, mpm_length, FOS_element_ub, number_of_parameters, random_linkage_tree, S_vector );
        NN_chain_length++;
      }
      
      while( NN_chain[NN_chain_length-3] != NN_chain[NN_chain_length-1] )
      {
        NN_chain[NN_chain_length] = determineNearestNeighbour(NN_chain[NN_chain_length-1], S_matrix, mpm_number_of_indices, mpm_length, FOS_element_ub, number_of_parameters, random_linkage_tree, S_vector);
        if( ((getSimilarity(NN_chain[NN_chain_length-1], NN_chain[NN_chain_length], FOS_element_ub, number_of_parameters, mpm_number_of_indices, random_linkage_tree, S_matrix, S_vector) == getSimilarity(NN_chain[NN_chain_length-1],NN_chain[NN_chain_length-2], FOS_element_ub, number_of_parameters, mpm_number_of_indices, random_linkage_tree, S_matrix, S_vector)))
           && (NN_chain[NN_chain_length] != NN_chain[NN_chain_length-2]) )
          NN_chain[NN_chain_length] = NN_chain[NN_chain_length-2];
        NN_chain_length++;
        if( NN_chain_length > number_of_parameters )
          break;
      }
      r0 = NN_chain[NN_chain_length-2];
      r1 = NN_chain[NN_chain_length-1];
      
      if( r1 >= mpm_length || r0 >= mpm_length || mpm_number_of_indices[r0]+mpm_number_of_indices[r1] > FOS_element_ub )
      {
        NN_chain_length = 1;
        NN_chain[0] = 0;
        if( FOS_element_ub < number_of_parameters )
        {
          done = 1;
          for( i = 1; i < mpm_length; i++ )
          {
            if( mpm_number_of_indices[i] + mpm_number_of_indices[NN_chain[0]] <= FOS_element_ub ) done = 0;
            if( mpm_number_of_indices[i] < mpm_number_of_indices[NN_chain[0]] ) NN_chain[0] = i;
          }
          if( done ) break;
        }
        continue;
      }
      
      if( r0 > r1 )
      {
        rswap = r0;
        r0    = r1;
        r1    = rswap;
      }
      NN_chain_length -= 3;
      
      // This test is required for exceptional cases in which the nearest-neighbor
      // ordering has changed within the chain while merging within that chain
      if( r1 < mpm_length && r1 != r0 )
      {
        indices.resize(mpm_number_of_indices[r0]+mpm_number_of_indices[r1]);
        
        i = 0;
        for( j = 0; j < mpm_number_of_indices[r0]; j++ )
        {
          indices[i] = mpm[r0][j];
          i++;
        }
        for( j = 0; j < mpm_number_of_indices[r1]; j++ )
        {
          indices[i] = mpm[r1][j];
          i++;
        }
        
        std::sort(indices.begin(),indices.end());
        new_FOS->sets[FOS_index]->params = indices;
        
        mul0 = ((double) mpm_number_of_indices[r0])/((double) mpm_number_of_indices[r0]+mpm_number_of_indices[r1]);
        mul1 = ((double) mpm_number_of_indices[r1])/((double) mpm_number_of_indices[r0]+mpm_number_of_indices[r1]);
        if( random_linkage_tree )
        {
          S_vector[r0] = mul0*S_vector[r0]+mul1*S_vector[r1];
        }
        else
        {
          for( i = 0; i < mpm_length; i++ )
          {
            if( (i != r0) && (i != r1) )
            {
              S_matrix[i][r0] = mul0*S_matrix[i][r0] + mul1*S_matrix[i][r1];
              S_matrix[r0][i] = S_matrix[i][r0];
            }
          }
        }
        
        mpm_new.resize(mpm_length-1);
        mpm_new_number_of_indices.resize(mpm_length-1);
        mpm_new_length            = mpm_length-1;
        for( i = 0; i < mpm_new_length; i++ )
        {
          mpm_new[i]                   = mpm[i];
          mpm_new_number_of_indices[i] = mpm_number_of_indices[i];
        }
        
        mpm_new[r0]                   = new_FOS->sets[FOS_index]->params;
        mpm_new_number_of_indices[r0] = mpm_number_of_indices[r0]+mpm_number_of_indices[r1];
        if( r1 < mpm_length-1 )
        {
          mpm_new[r1]                   = mpm[mpm_length-1];
          mpm_new_number_of_indices[r1] = mpm_number_of_indices[mpm_length-1];
          
          if( random_linkage_tree )
          {
            S_vector[r1] = S_vector[mpm_length-1];
          }
          else
          {
            for(size_t i = 0; i < r1; i++ )
            {
              S_matrix[i][r1] = S_matrix[i][mpm_length-1];
              S_matrix[r1][i] = S_matrix[i][r1];
            }
            
            for(size_t j = r1+1; j < mpm_new_length; j++ )
            {
              S_matrix[r1][j] = S_matrix[j][mpm_length-1];
              S_matrix[j][r1] = S_matrix[r1][j];
            }
          }
        }
        
        for( i = 0; i < NN_chain_length; i++ )
        {
          if( NN_chain[i] == mpm_length-1 )
          {
            NN_chain[i] = r1;
            break;
          }
        }
        
        mpm                   = mpm_new;
        mpm_number_of_indices = mpm_new_number_of_indices;
        mpm_length            = mpm_new_length;
        
        if( mpm_length == 1 )
          done = 1;
        
        FOS_index++;
      }
    }
    
    return( new_FOS );
  }
  
  void filterFOS( FOS_pt input_FOS, int lb, int ub, bool always_keep_largest_FOS_element )
  {
    
    std::vector<FOS_element_pt> new_sets;
    new_sets.reserve(input_FOS->length());

    size_t largest_FOS_element = 0;
    if(always_keep_largest_FOS_element)
    {
      for(size_t i = 0; i < input_FOS->length(); ++i) {
        if(input_FOS->sets[i]->length() > largest_FOS_element ) {
          largest_FOS_element = input_FOS->sets[i]->length();
        }
      }
    }
    
    for(size_t i = 0; i < input_FOS->length(); ++i) {
      if(input_FOS->sets[i]->length() == largest_FOS_element || ( input_FOS->sets[i]->length() >= lb && input_FOS->sets[i]->length() <= ub )) {
        new_sets.push_back(input_FOS->sets[i]);
      }
    }
    
    input_FOS->sets = new_sets;
  }
  
  double getSimilarity( size_t a, size_t b, int FOS_element_ub, size_t number_of_parameters, const std::vector<size_t> & mpm_number_of_indices, bool random_linkage_tree, const matrix_t & S_matrix, const vec_t & S_vector )
  {
    if( FOS_element_ub < number_of_parameters && mpm_number_of_indices[a] + mpm_number_of_indices[b] > FOS_element_ub ) return( 0 );
    if( random_linkage_tree ) return( 1.0-fabs(S_vector[a]-S_vector[b]) );
    return( S_matrix[a][b] );
  }
  
  size_t determineNearestNeighbour( size_t index, const matrix_t & S_matrix, const std::vector<size_t> & mpm_number_of_indices, int mpm_length, int FOS_element_ub, size_t number_of_parameters, bool random_linkage_tree, const vec_t & S_vector )
  {
    int i, result;
    
    result = 0;
    if( result == index )
      result++;
    for( i = 1; i < mpm_length; i++ )
    {
      if( ((getSimilarity(index,i,FOS_element_ub, number_of_parameters, mpm_number_of_indices, random_linkage_tree, S_matrix, S_vector) > getSimilarity(index,result,FOS_element_ub, number_of_parameters, mpm_number_of_indices, random_linkage_tree, S_matrix, S_vector))
           || ((getSimilarity(index,i,FOS_element_ub, number_of_parameters, mpm_number_of_indices, random_linkage_tree, S_matrix, S_vector) == getSimilarity(index,result,FOS_element_ub, number_of_parameters, mpm_number_of_indices, random_linkage_tree, S_matrix, S_vector)) && (mpm_number_of_indices[i] < mpm_number_of_indices[result]))) && (i != index) )
        result = i;
    }
    
    return( result );
  }
  
  void computeMIMatrix( matrix_t & MI_matrix, const matrix_t & covariance_matrix )
  {
    double si, sj, r;
    
    size_t n = covariance_matrix.cols();
    
    MI_matrix.resize(n,n);
    
    for(size_t i = 0; i < n; i++ )
    {
      MI_matrix[i][i] = 1e20;
      for(size_t j = 0; j < i; j++ )
      {
        si = sqrt(covariance_matrix[i][i]);
        sj = sqrt(covariance_matrix[j][j]);
        r = covariance_matrix[i][j]/(si*sj);
        
        MI_matrix[i][j] = log(sqrt(1/(1-r*r)));
        MI_matrix[j][i] = MI_matrix[i][j];
      }
    }
    
  }
  
  // scm: whats going on here?
  // i think it assumes that the univariate elements do not need matching.
  // i removed all of that now.
  // it also assumes that both are of the same size.
  int *matchFOSElements( FOS_pt new_FOS, const FOS_pt prev_FOS )
  {
    int      i, j, a, b, matches, *permutation, *hungarian_permutation,
    **FOS_element_similarity_matrix;
    
    permutation = (int *) Malloc( new_FOS->length()*sizeof(int));
    FOS_element_similarity_matrix = (int**) Malloc((new_FOS->length())*sizeof(int*));
    for( i = 0; i < new_FOS->length(); i++ )
      FOS_element_similarity_matrix[i] = (int*) Malloc((new_FOS->length())*sizeof(int));
    
    for( i = (int) 0; i < prev_FOS->length(); i++ )
    {
      for( j = (int) 0; j < new_FOS->length(); j++ )
      {
        a = 0; b = 0;
        matches = 0;
        while( a < prev_FOS->sets[i]->length() && b < new_FOS->sets[j]->length() )
        {
          if( prev_FOS->sets[i]->params[a] < new_FOS->sets[j]->params[b] )
          {
            a++;
          }
          else if( prev_FOS->sets[i]->params[a] > new_FOS->sets[j]->params[b] )
          {
            b++;
          }
          else
          {
            a++;
            b++;
            matches++;
          }
        }
        FOS_element_similarity_matrix[i][j] = (int) 10000*(2.0*matches/(prev_FOS->sets[i]->length()+new_FOS->sets[j]->length()));
      }
    }
    
    hungarian_permutation = hungarianAlgorithm(FOS_element_similarity_matrix, (int) (new_FOS->length()));
    for( i = 0; i < new_FOS->length(); i++ ) {
      permutation[i] = (int) (hungarian_permutation[i]);
    }
    for( i = 0; i < new_FOS->length(); i++ )
      free( FOS_element_similarity_matrix[i] );
    free( FOS_element_similarity_matrix );
    free( hungarian_permutation );
    
    return( permutation );
  }
  
  int *hungarianAlgorithm( int **similarity_matrix, int dim )
  {
    int i, j, x, y, root, *q, wr, rd, cx, cy, ty, max_match,
    *lx, *ly, *xy, *yx, *slack, *slackx, *prev, delta;
    short *S, *T, terminated;
    
    lx = (int*) Malloc(dim*sizeof(int));
    ly = (int*) Malloc(dim*sizeof(int));
    xy = (int*) Malloc(dim*sizeof(int));
    yx = (int*) Malloc(dim*sizeof(int));
    slack = (int*) Malloc(dim*sizeof(int));
    slackx = (int*) Malloc(dim*sizeof(int));
    prev = (int*) Malloc(dim*sizeof(int));
    S = (short*) Malloc(dim*sizeof(short));
    T = (short*) Malloc(dim*sizeof(short));
    
    root = -1;
    max_match = 0;
    for( i = 0; i < dim; i++ )
    {
      lx[i] = 0;
      ly[i] = 0;
      xy[i] = -1;
      yx[i] = -1;
    }
    for(i = 0; i < dim; i++)
      for(j = 0; j < dim; j++)
        if(similarity_matrix[i][j] > lx[i])
          lx[i] = similarity_matrix[i][j];
    
    terminated = 0;
    while(!terminated)
    {
      if (max_match == dim) break;
      
      wr = 0;
      rd = 0;
      q = (int*) Malloc(dim*sizeof(int));
      for( i = 0; i < dim; i++ )
      {
        S[i] = 0;
        T[i] = 0;
        prev[i] = -1;
      }
      
      for (x = 0; x < dim; x++)
      {
        if (xy[x] == -1)
        {
          q[wr++] = root = x;
          prev[x] = -2;
          S[x] = 1;
          break;
        }
      }
      
      for (y = 0; y < dim; y++)
      {
        slack[y] = lx[root] + ly[y] - similarity_matrix[root][y];
        slackx[y] = root;
      }
      
      while ( 1 )
      {
        while (rd < wr)
        {
          x = q[rd++];
          for (y = 0; y < dim; y++)
          {
            if (similarity_matrix[x][y] == lx[x] + ly[y] && !T[y])
            {
              if (yx[y] == -1) break;
              T[y] = 1;
              q[wr++] = yx[y];
              hungarianAlgorithmAddToTree(yx[y], x, S, prev, slack, slackx, lx, ly, similarity_matrix, dim);
            }
          }
          if (y < dim) break;
        }
        if (y < dim) break;
        
        delta = 100000000;
        for(y = 0; y < dim; y++)
          if(T[y] == 0 && slack[y] < delta)
            delta = slack[y];
        for(x = 0; x < dim; x++)
          if(S[x] == 1)
            lx[x] -= delta;
        for(y = 0; y < dim; y++)
          if(T[y] == 1)
            ly[y] += delta;
        for(y = 0; y < dim; y++)
          if(T[y] == 0)
            slack[y] -= delta;
        
        wr = 0;
        rd = 0;
        for (y = 0; y < dim; y++)
        {
          if (T[y] == 0 && slack[y] == 0)
          {
            if (yx[y] == -1)
            {
              x = slackx[y];
              break;
            }
            else
            {
              T[y] = 1;
              if (S[yx[y]] == 0)
              {
                q[wr++] = yx[y];
                hungarianAlgorithmAddToTree(yx[y], slackx[y], S, prev, slack, slackx, lx, ly, similarity_matrix, dim);
              }
            }
          }
        }
        if (y < dim) break;
      }
      
      if (y < dim)
      {
        max_match++;
        for (cx = x, cy = y; cx != -2; cx = prev[cx], cy = ty)
        {
          ty = xy[cx];
          yx[cy] = cx;
          xy[cx] = cy;
        }
      }
      else terminated = 1;
      
      free( q );
    }
    
    free( lx );
    free( ly );
    free( yx );
    free( slack );
    free( slackx );
    free( prev );
    free( S );
    free( T );
    
    return xy;
  }
  
  void hungarianAlgorithmAddToTree(int x, int prevx, short *S, int *prev, int *slack, int *slackx, int* lx, int *ly, int** similarity_matrix, int dim)
  {
    int y;
    
    S[x] = 1;
    prev[x] = prevx;
    for (y = 0; y < dim; y++)
    {
      if (lx[x] + ly[y] - similarity_matrix[x][y] < slack[y])
      {
        slack[y] = lx[x] + ly[y] - similarity_matrix[x][y];
        slackx[y] = x;
      }
    }
  }

}

// so_optimization.c

namespace hillvallea
{
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Problems -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Returns whether a parameter is inside the range bound of
   * every problem.
   */
  short gomea_t::isParameterInRangeBounds( double & parameter, int dimension ) const
  {
    
    if( parameter < lower_param_bounds[dimension] ||
       parameter > upper_param_bounds[dimension] ||
       isnan( parameter ) )
    {
      
      if(use_boundary_repair)
      {
        if( parameter < lower_param_bounds[dimension] ) {
          parameter = lower_param_bounds[dimension];
        }
        
        if( parameter > upper_param_bounds[dimension] ) {
          parameter = upper_param_bounds[dimension];
        }
        return( 1 );
        
      }
      return( 0 );
    }
    return( 1 );
  }
  
}

// end of so_optimization.c

namespace hillvallea
{
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  
  FOS_pt updateLinkageTree
  (
   const matrix_t & input_matrix,
   FOS_pt old_FOS,
   const bool learn_from_distance_matrix,
   const bool random_linkage,
   const bool static_linkage,
   const int number_of_generations,
   int & FOS_element_lb,
   int & FOS_element_ub,
   bool always_keep_largest_FOS_element,
   const size_t number_of_parameters,
   bool & sample_bayesian_factorization,
   rng_pt rng
   )
  {
    FOS_pt new_FOS = std::make_shared<FOS_t>();
    
    if(static_linkage && number_of_generations > 0) {
      return old_FOS;
    }
    
    if(number_of_generations == 0 && static_linkage && !learn_from_distance_matrix) {
      assert(random_linkage == true);
    }
    
    // if we learn from a distnace matrix, we have to invert the input matrix
    // to obtain a similariy matrix
    bool invert_similarity_matrix = learn_from_distance_matrix;
    
    int temp_FOS_element_ub = FOS_element_ub;
    if(always_keep_largest_FOS_element) {
      temp_FOS_element_ub = (int) number_of_parameters; // temp disable.
    }
    
    new_FOS = learnLinkageTree(input_matrix, invert_similarity_matrix, random_linkage, number_of_parameters, rng, temp_FOS_element_ub);
    
    // we cannot do filterFOS for non-static linkage yet
    // because the distribution multiplier transfer is based on the hungarian algorithm
    // that requires linkage trees of the same size.
    if(static_linkage)
    {
      if(FOS_element_lb > 1 || FOS_element_ub < number_of_parameters ) {
        filterFOS(new_FOS, FOS_element_lb, FOS_element_ub, always_keep_largest_FOS_element);
      }
    }
    // printFOS(new_FOS);
    
    
    // aggregate all fos elements into a single full fos element with a bayesian factorization
    if(sample_bayesian_factorization) {
      // first, assert if the linkage model is a marginal model, else this cannot be done (yet).
      std::vector<int> variable_occurences(number_of_parameters,0);
      
      for(size_t i = 0; i < new_FOS->sets.size(); ++i) {
        for(size_t j = 0; j < new_FOS->sets[i]->params.size(); ++j) {
          variable_occurences[new_FOS->sets[i]->params[j]]++;
        }
      }
      
      for(size_t i = 0; i < number_of_parameters; ++i) {
        if(variable_occurences[i] != 1) {
          std::cout << "Linkage model is not marginal (variable " << i << " occurs " << variable_occurences[i] << " times), so bayesian factorization is disabled\n";
          sample_bayesian_factorization = false;
          break;
        }
      }
      
      if(sample_bayesian_factorization)
      {
        // then, aggregate all linkage subsets into a single fos element.
        FOS_element_pt full_bayesian_fos = std::make_shared<FOS_element_t>();
        full_bayesian_fos->sample_bayesian_factorization = true;
        full_bayesian_fos->params.resize(number_of_parameters);
        for(size_t i = 0; i < number_of_parameters; ++i) {
          full_bayesian_fos->params[i] = i;
        }
        full_bayesian_fos->bayesian_factorization_indicator_matrix.resize(number_of_parameters, number_of_parameters);
        full_bayesian_fos->bayesian_factorization_indicator_matrix.fill(0);
        
        // for each fos element
        for(size_t i = 0; i < new_FOS->sets.size(); ++i)
        {
          
          for(size_t j = 0; j < new_FOS->sets[i]->params.size(); ++j) {
            size_t vara = new_FOS->sets[i]->params[j];
            for(size_t k = j; k < new_FOS->sets[i]->params.size(); ++k) {
              size_t varb = new_FOS->sets[i]->params[k];
              full_bayesian_fos->bayesian_factorization_indicator_matrix[vara][varb] = 1;
              full_bayesian_fos->bayesian_factorization_indicator_matrix[varb][vara] = 1;
            }
          }
        }
        
        new_FOS->sets.clear();
        new_FOS->sets.push_back(full_bayesian_fos);
      }
      
      FOS_element_lb = (int) number_of_parameters;
      FOS_element_ub = (int) number_of_parameters;
    }
    // end bayesian stuff
    
    
    // Dynamic linkage tree, de-init the old one.
    if( !static_linkage && number_of_generations > 0 )
    {
      inheritSettings( new_FOS, old_FOS );
    }
    return new_FOS ;
  }
  
  void inheritSettings( FOS_pt new_FOS, const FOS_pt prev_FOS )
  {
    int * permutation = matchFOSElements( new_FOS, prev_FOS );
    
    for(size_t i = 0; i < new_FOS->length(); i++ ) {
      new_FOS->sets[permutation[i]]->copySettings(*prev_FOS->sets[i]);
    }
    
    free( permutation );
  }
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=- Section Termination -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Returns 1 if termination should be enforced, 0 otherwise.
   */
  bool gomea_t::checkTerminationCondition( void ) // todo: check if this matches which AMaLGaMs.
  {
    if( number_of_generations == 0 ) {
      active = true;
      return !active;
    }
    
    if (checkFitnessVarianceTerminationSinglePopulation())
    {
      active = false;
      return !active;
    }
    
    if (checkDistributionMultiplierTerminationCondition())
    {
      active = false;
      return !active;
    }
    
    return false;
  }
  
  /**
   * Returns 1 if the fitness variance in a specific population
   * has become too small (user-defined tolerance).
   */
  short gomea_t::checkFitnessVarianceTerminationSinglePopulation( )
  {
    
    if (number_of_generations == 0) {
      active = true;
      return !active;
    }
    
    // 1. if the cluster is empty, deactivate it.
    if (pop->size() == 0) {
      active = false;
      return !active;
    }
    
    vec_t mean;
    pop->mean(mean);
    
    // compute maximum_parameter_variance
    // for some reason, the partial covariances matrices are not available here anymore
    // when we do linkage learning. TODO: FIX
    // double max_param_variance = 1e300;
    /* for(int i = 0; i < linkage_model->length; i++ ) {
      for(int j = 0; j < linkage_model->set_length[i]; j++ ) {
        max_param_variance = std::max(max_param_variance, std::fabs(partial_covariance_matrices[i][j][j]));
      }
    } */
    // end Max_param-variance
    
    
    
    // if the mean equals zero, we can't didivide by it, so terminate it when it is kinda small
    bool terminate_for_param_std_mean_zero =  false; //(mean.infinitynorm() <= 0 && sqrt(max_param_variance) < param_std_tolerance);
    bool terminate_on_parameter_std = false; // sqrt(max_param_variance) / mean.infinitynorm() < param_std_tolerance;
    bool terminate_on_fitness_std = (pop->average_constraint() == 0) && (pop->size() > 1) && (pop->relative_fitness_std() < fitness_std_tolerance);
    bool terminate_on_penalty_std = (pop->average_constraint() > 0) && (pop->size() > 1) && (pop->relative_constraint_std() < penalty_std_tolerance);
    
    if(active)
    {
      if(terminate_for_param_std_mean_zero) std::cout << "terminate_for_param_std_mean_zero \n";
      if(terminate_on_parameter_std) std::cout << "terminate_on_parameter_std \n";
      if(terminate_on_fitness_std) std::cout << "terminate_on_fitness_std \n";
      if(terminate_on_penalty_std) std::cout << "terminate_on_fitness_std \n";
    }
    
    if (terminate_for_param_std_mean_zero || terminate_on_parameter_std || terminate_on_fitness_std || terminate_on_penalty_std)
    {
      active = false;
      return !active;
    }
    
    return false;
  }
  
  /**
   * Checks whether the distribution multiplier in any population
   * has become too small (1e-10).
   */
  short gomea_t::checkDistributionMultiplierTerminationCondition( )
  {
    int j;
    short converged;
    
    if( active )
    {
      converged = 1;
      for( j = 0; j < linkage_model->length(); j++ )
      {
        if( linkage_model->sets[j]->distribution_multiplier > 1e-10 )
        {
          converged = 0;
          break;
        }
      }
      
      if( converged ) {
        return true;
      }
    }
    
    return false;
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  
  
  
  
  
  
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Variation -==-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  
  
  void gomea_t::evaluateCompletePopulation( )
  {
    for(size_t i = 0; i < pop->sols.size(); ++i) {
      if(gradient_step) {
        fitness_function->evaluate_with_gradients(pop->sols[i]);
      } else {
        fitness_function->evaluate(pop->sols[i]);
      }
      weighted_number_of_evaluations++;
    }
    
    // the 'best' is in the population at index_of_best = 0;
    int index_of_best;
    getBestInPopulation(*pop, &index_of_best);
    best = solution_t(*pop->sols[index_of_best]);
    
  }
  
  void applyDistributionMultiplier(FOS_element_pt FOS_element)
  {
    for(int i = 0; i < FOS_element->covariance_matrix.rows(); i++ )
    {
      for(int j = 0; j < FOS_element->covariance_matrix.cols(); j++ ) {
        FOS_element->covariance_matrix[i][j] *= FOS_element->distribution_multiplier;
      }
    }
  }
  
  /**
   * Generates new solutions for each
   * of the populations in turn.
   */
  void gomea_t::generateAndEvaluateNewSolutionsToFillPopulation( const solution_t & elite )
  {
    short   generationalImprovement, all_multipliers_leq_one, apply_AMS;
    int     i, k, number_of_AMS_solutions, best_individual_index;
    double  alpha_AMS;
    
    if( !active ) {
      return;
    }
    
    std::vector<short> FOS_element_caused_improvement(linkage_model->length());
    std::vector<short> individual_improved(population_size, 0);
    
    alpha_AMS = 0.5*selection_fraction*(((double) population_size)/((double) (population_size-1)));
    number_of_AMS_solutions = (int) (alpha_AMS*(population_size-1));
    int *fos_order = randomPermutation((int) linkage_model->length(), *rng);
    int *solution_order = randomPermutation(population_size, *rng);
    
    // it seems to be better to permute solutions for rosenbrock, but worse for sphere.
    // TODO: Properly test solution ordering, but i think it is what we want.
    //for(int i = 0; i < population_size; ++i) {
    //  solution_order[i] = i;
    //}
    
    // set current elite, which is updated during each run.
    solution_t current_elite(elite);
    vec_t sdr(linkage_model->length(),0.0);
    
    // set an upper bound based on the population size.
    int temp_FOS_element_ub = FOS_element_ub;
    if(!static_linkage && dynamic_filter_large_FOS_elements)
    {
      int selection_size = population_size * selection_fraction;
      temp_FOS_element_ub = std::min(selection_size - 1,temp_FOS_element_ub);
      temp_FOS_element_ub = std::max(FOS_element_lb,temp_FOS_element_ub); // sanity check
    }
    
    // find largest for element (if that is the one to keep)
    size_t largest_FOS_element_size = 0;
    if(always_keep_largest_FOS_element) {
      for(int oj = 0; oj < linkage_model->length(); oj++ ) {
        if(linkage_model->sets[oj]->length() > largest_FOS_element_size) {
          largest_FOS_element_size = linkage_model->sets[oj]->length();
        }
      }
    }
    
    // for testing.
    std::vector<int> parameter_update_counter(number_of_parameters, 0);
    
    bool full_fos_element_included = false;
    for(int oj = 0; oj < linkage_model->length(); oj++ )
    {
      int FOS_index = fos_order[oj];
      
      // skip fos elements that are too small or too large.
      if (linkage_model->sets[FOS_index]->length() < FOS_element_lb || linkage_model->sets[FOS_index]->length() > temp_FOS_element_ub) {
        if(!always_keep_largest_FOS_element || linkage_model->sets[FOS_index]->length() != largest_FOS_element_size) {
          continue;
        }
      }
      
      for(size_t j = 0; j < linkage_model->sets[FOS_index]->params.size(); ++j) {
        parameter_update_counter[linkage_model->sets[FOS_index]->params[j]]++;
      }
      
      if(linkage_model->sets[FOS_index]->length() == number_of_parameters) {
        full_fos_element_included = true;
      }
      
      vec_t individual_improved_this_FOS_element(population_size, 0);
      
      // intermediate covariance update
      population_t selection;
      makeSelection(*pop, selection_fraction, selection); // TODO: this is double because we also do it just before this function call in generation(); fix later.
      pop->sols[0] = copySolution(selection.sols[0]);
      estimateMeanVectorML_partial(mean_vector, linkage_model->sets[FOS_index], selection);
      
      
      estimateCovarianceMatrix(linkage_model->sets[FOS_index], selection, mean_vector, init_univariate_bandwidth);
      applyDistributionMultiplier(linkage_model->sets[FOS_index]);
      current_elite = solution_t(*selection.sols[0]);
      
      // compute cholesky decomposition
      bool success = false;
      if(!linkage_model->sets[FOS_index]->enable_regularization) {
        choleskyDecomposition(  linkage_model->sets[FOS_index]->covariance_matrix, linkage_model->sets[FOS_index]->cholesky_factor_lower_triangle, success );
      }
      
      if(!success || linkage_model->sets[FOS_index]->enable_regularization)
      {
        linkage_model->sets[FOS_index]->enable_regularization = true;
        regularizeCovarianceMatrix(linkage_model->sets[FOS_index]->covariance_matrix, selection.sols, mean_vector, linkage_model->sets[FOS_index]->params);
        choleskyDecomposition(  linkage_model->sets[FOS_index]->covariance_matrix, linkage_model->sets[FOS_index]->cholesky_factor_lower_triangle, success );
      }
      
      
      linkage_model->sets[FOS_index]->samples_drawn_from_normal = 0;
      linkage_model->sets[FOS_index]->out_of_bounds_draws       = 0;
      FOS_element_caused_improvement[FOS_index] = 0;
      
      // apply_AMS = 1;
      int best_hit = 0;
      for(size_t k = 0; k < population_size; k++ ) // skips best in population
      {
        if(solution_order[k] == 0) { // solution_order[k] points to the solution we want to update, and sol[0] = best, so this skips the best.
          best_hit = 1;
          continue; // skips best in population
        }
        
        if( k >= (number_of_AMS_solutions + best_hit) ) { // keeps the number_of_ams_solutions the same if the best is hit.
          apply_AMS = 0; // apply ams to some solutions
        } else {
          apply_AMS = 1;
        }
        
        bool force_accept_new_solutions = false; // (number_of_generations == 0 && oj == 0);
        individual_improved_this_FOS_element[solution_order[k]] = generateNewSolutionFromFOSElement( linkage_model->sets[FOS_index], solution_order[k], apply_AMS, force_accept_new_solutions );
        individual_improved[solution_order[k]] |= (bool) individual_improved_this_FOS_element[solution_order[k]];
      }
      
      // Update Hyperparameters
      size_t number_of_improvements = generationalImprovementForFOSElement(linkage_model->sets[FOS_index], &sdr[FOS_index], current_elite );
      adaptDistributionMultiplier( linkage_model->sets[FOS_index]->distribution_multiplier, (number_of_improvements > 0), sdr[FOS_index], linkage_model->sets[FOS_index]->out_of_bounds_draws, linkage_model->sets[FOS_index]->samples_drawn_from_normal );
      
    }
    free( fos_order );
    // end of fos-element-wise solution sampling
    
    // Shouldn't this be skipped for full?? It seems slightly better to just add it.
    // if(!full_fos_element_included && number_of_generations > 0 )
    if(number_of_generations > 0 )
    {
      int best_hit = 0;
      for(size_t k = 0; k < population_size; ++k)
      {
        if(solution_order[k] == 0) {
          best_hit = 1;
          continue;
          // skip elite
        }
        if(k < number_of_AMS_solutions + best_hit) {
          individual_improved[solution_order[k]] |= applyAMS(pop->sols[solution_order[k]], weighted_number_of_evaluations, best);
        }
      }
    }
    
    free(solution_order);
    // std::cout << std::endl << "parameter_update_counter = [";
    // for(size_t i = 0; i < parameter_update_counter.size(); ++i) {
    //   std::cout << parameter_update_counter[i] << " ";
    // }
    // std::cout << "]" << std::endl;
    for(size_t i = 0; i < parameter_update_counter.size(); ++i) {
      if(parameter_update_counter[i] == 0) {
        std::cout << "Warning: parameter " << i << " not updated in generation " << number_of_generations << "." << std::endl;
      }
    }
    
    // adam step
    if(gradient_step)
    {
      // banaan
      
      // full steps
      std::vector<std::vector<size_t> > touched_parameter_idx(1);
      touched_parameter_idx[0].resize(number_of_parameters);
      for(size_t i = 0; i < number_of_parameters; ++i) {
        touched_parameter_idx[0][i] = i;
      }

      
      for(size_t i = 0;  i < population_size; ++i) {
        vec_t gammas(1, gradient_methods[i]->gamma);
        weighted_number_of_evaluations += gradient_methods[i]->gradientOffspring(pop->sols[i], touched_parameter_idx, gammas);
        gradient_methods[i]->gamma = gammas[0];
        
        if(solution_t::better_solution(*pop->sols[i], best)) {
          best = solution_t(*pop->sols[i]);
        }
        individual_improved[i] = true;
      }
    }
    
    
    for( i = 1; i < population_size; i++ )
    {
      if( !individual_improved[i] ) {
        pop->sols[i]->NIS++;
      }
      else {
        pop->sols[i]->NIS = 0;
      }
    }
    
    getBestInPopulation(*pop, &best_individual_index );
    best = solution_t(*pop->sols[best_individual_index]);
    
    for( k = 1; k < population_size; k++ )
    {
      if( pop->sols[k]->NIS > maximum_no_improvement_stretch) {
        applyForcedImprovements( k, best );
      }
    }
    
    generationalImprovement = 0;
    for(int FOS_index = 0; FOS_index < linkage_model->length(); FOS_index++ ) {
      if( FOS_element_caused_improvement[FOS_index] ) {
        generationalImprovement = 1;
        break;
      }
    }
    
    if( generationalImprovement ) {
      no_improvement_stretch = 0;
    }
    else
    {
      all_multipliers_leq_one = 1;
      
      for(int FOS_index = 0; FOS_index < linkage_model->length(); FOS_index++ )
      {
        if( linkage_model->sets[FOS_index]->distribution_multiplier > 1.0 )
        {
          all_multipliers_leq_one = 0;
          break;
        }
      }
      
      if( all_multipliers_leq_one ) {
        (no_improvement_stretch)++;
      }
    }
  }
  
  /**
   * Generates and returns a single new solution by drawing
   * a sample for the variables in the selected FOS element
   * and inserting this into the population.
   */
  double *gomea_t::generateNewPartialSolutionFromFOSElement( FOS_element_pt FOS_element )
  {
    
    short   ready;
    int     i, times_not_in_bounds;
    double *result, *z;
    
    size_t num_indices = FOS_element->length();
    std::vector<size_t> indices = FOS_element->params;
    
    times_not_in_bounds = -1;
    FOS_element->out_of_bounds_draws--;
    
    std::normal_distribution<double> std_normal(0.0, 1.0);
    
    ready = 0;
    do
    {
      times_not_in_bounds++;
      FOS_element->samples_drawn_from_normal++;
      FOS_element->out_of_bounds_draws++;
      if( times_not_in_bounds >= 100 )
      {
        result = (double *) Malloc( num_indices*sizeof( double ) );
        for( i = 0; i < num_indices; i++ )
          result[i] = lower_param_bounds[indices[i]] + (upper_param_bounds[indices[i]] - lower_param_bounds[indices[i]])*randomRealUniform01(*rng);
      }
      else
      {
        z = (double *) Malloc( num_indices*sizeof( double ) );
        
        for( i = 0; i < num_indices; i++ )
          z[i] = std_normal(*rng);
        
        if( FOS_element->length() == 1 )
        {
          result = (double*) Malloc(1*sizeof(double));
          result[0] = z[0]*sqrt(FOS_element->covariance_matrix[0][0]) + mean_vector[indices[0]];
        }
        else
        {
          result = matrixVectorMultiplication( FOS_element->cholesky_factor_lower_triangle.toArray(), z, (int) num_indices, (int) num_indices );
          for( i = 0; i < num_indices; i++ )
            result[i] += mean_vector[indices[i]];
        }
        
        free( z );
      }
      
      ready = 1;
      for( i = 0; i < num_indices; i++ )
      {
        if( !isParameterInRangeBounds( result[i], (int) indices[i] ) )
        {
          ready = 0;
          break;
        }
      }
      if( !ready )
        free( result );
    }
    while( !ready );
    
    return( result );
  }
  
  // sample a new solution based on conditional sampling of its neigbhours
  double *gomea_t::generateNewConditionalPartialSolutionFromFOSElement( FOS_element_pt FOS_element, const solution_t & conditional_sol  )
  {
    
    short   ready;
    int     i, times_not_in_bounds;
    double *result, *z;
    
    size_t num_indices = FOS_element->length();
    std::vector<size_t> indices = FOS_element->params;
    
    times_not_in_bounds = -1;
    FOS_element->out_of_bounds_draws--;
    
    // todo: sample from conditional distribution.
    vec_t delta_contidional_params(FOS_element->params_to_condition_on.size());
    
    for(size_t i = 0; i < FOS_element->params_to_condition_on.size(); ++i) {
      delta_contidional_params[i] = conditional_sol.param[FOS_element->params_to_condition_on[i]] - mean_vector[FOS_element->params_to_condition_on[i]];
    }
    
    vec_t conditional_mean_vector(FOS_element->params.size(),0.0);
    
    for(size_t i = 0; i < FOS_element->params.size(); ++i) {
      conditional_mean_vector[i] = mean_vector[FOS_element->params[i]];
    }
    conditional_mean_vector += FOS_element->mean_shift_matrix_to_condition_on * delta_contidional_params;
    
    // end conditional sampling
    
    std::normal_distribution<double> std_normal(0.0, 1.0);
    
    ready = 0;
    do
    {
      times_not_in_bounds++;
      FOS_element->samples_drawn_from_normal++;
      FOS_element->out_of_bounds_draws++;
      if( times_not_in_bounds >= 100 )
      {
        result = (double *) Malloc( num_indices*sizeof( double ) );
        for( i = 0; i < num_indices; i++ )
          result[i] = lower_param_bounds[indices[i]] + (upper_param_bounds[indices[i]] - lower_param_bounds[indices[i]])*randomRealUniform01(*rng);
      }
      else
      {
        z = (double *) Malloc( num_indices*sizeof( double ) );
        
        for( i = 0; i < num_indices; i++ )
          z[i] = std_normal(*rng);
        
        if( FOS_element->length() == 1 )
        {
          result = (double*) Malloc(1*sizeof(double));
          // result[0] = z[0]*sqrt(FOS_element->covariance_matrix[0][0]) + mean_vector[indices[0]]; // updated
          result[0] = z[0]*sqrt(FOS_element->covariance_matrix[0][0]) + conditional_mean_vector[0]; // updated
        }
        else
        {
          result = matrixVectorMultiplication( FOS_element->cholesky_factor_lower_triangle.toArray(), z, (int) num_indices, (int) num_indices );
          for( i = 0; i < num_indices; i++ )
            // result[i] += mean_vector[indices[i]]; // updated
            result[i] += conditional_mean_vector[i]; // updated
        }
        
        free( z );
      }
      
      ready = 1;
      for( i = 0; i < num_indices; i++ )
      {
        if( !isParameterInRangeBounds( result[i], (int) indices[i] ) )
        {
          ready = 0;
          break;
        }
      }
      if( !ready )
        free( result );
    }
    while( !ready );
    
    return( result );
  }
  
  
  
  /**
   * Generates and returns a single new solution by drawing
   * a single sample from a specified model.
   */
  short gomea_t::generateNewSolutionFromFOSElement( FOS_element_pt FOS_element, int individual_index, short apply_AMS, bool force_accept_new_solutions) //, bool new_solution_from_gradient_update )
  {
    int j, m;
    size_t im;
    double *result, delta_AMS, shrink_factor;
    short improvement, any_improvement, out_of_range;
    
    delta_AMS = 2.0;
    improvement = 0;
    any_improvement = 0;
    size_t num_indices = FOS_element->length();
    std::vector<size_t> indices = FOS_element->params;
    size_t num_touched_indices = num_indices;
    
    std::vector<size_t> touched_indices(num_touched_indices);
    for(size_t i = 0; i < num_touched_indices; ++i) {
      touched_indices[i] = indices[i]; // different datatype
    }
    
    solution_pt backup_sol = copySolution(pop->sols[individual_index]);

    if(FOS_element->sample_conditionally && FOS_element->params_to_condition_on.size() > 0) {
      result = generateNewConditionalPartialSolutionFromFOSElement(FOS_element, *backup_sol);
    } else  {
      result = generateNewPartialSolutionFromFOSElement(FOS_element);
    }
    
    for( j = 0; j < num_indices; j++ ) {
      pop->sols[individual_index]->param[indices[j]] = result[j];
    }
    
    // std::cout << pop->sols[individual_index]->param << std::endl;
    
    for(int m = 0; m < number_of_parameters; m++ ) {
      assert(isParameterInRangeBounds(pop->sols[individual_index]->param[m], m ));
    }
    
    if( apply_AMS && (number_of_generations > 0) )
    {
      out_of_range  = 1;
      shrink_factor = 2;
      while( (out_of_range == 1) && (shrink_factor > 1e-10) )
      {
        shrink_factor *= 0.5;
        out_of_range   = 0;
        for( m = 0; m < num_indices; m++ )
        {
          im = indices[m];
          result[m] = pop->sols[individual_index]->param[im]+shrink_factor*delta_AMS*FOS_element->distribution_multiplier*(mean_shift_vector[im]);
          if( !isParameterInRangeBounds( result[m], (int) im ) )
          {
            out_of_range = 1;
            break;
          }
        }
      }
      if( !out_of_range )
      {
        for( m = 0; m < num_indices; m++ )
        {
          pop->sols[individual_index]->param[indices[m]] = result[m];
        }
      }
    }
    free( result );
    
    for(int m = 0; m < number_of_parameters; m++ ) {
      assert(isParameterInRangeBounds(pop->sols[individual_index]->param[m], m ));
    }
    
    if(!fitness_function->partial_evaluations_available)
    {
      if(gradient_step) {
        fitness_function->evaluate_with_gradients(pop->sols[individual_index]);
      } else {
        fitness_function->evaluate(pop->sols[individual_index]);
      }
      weighted_number_of_evaluations++;
    } else
    {
      if(gradient_step) {
        fitness_function->partial_evaluate_with_gradients(pop->sols[individual_index], touched_indices, backup_sol);
      } else {
        fitness_function->partial_evaluate(pop->sols[individual_index], touched_indices, backup_sol);
      }
      weighted_number_of_evaluations += touched_indices.size()/(double)number_of_parameters;
    }
    
    if(solution_t::better_solution(*pop->sols[individual_index], best)) {
      best = solution_t(*pop->sols[individual_index]);
    }
    
    bool only_accept_improvement = (randomRealUniform01(*rng) >= 0.05);
    
    improvement = solution_t::better_solution_via_pointers(pop->sols[individual_index], backup_sol);
    if( improvement ) {
      return true; // improvement!
    }
    
    if(force_accept_new_solutions) {
      only_accept_improvement = false;
    }
    
    /* scm: if you disable this, i.e., >= 2.0, GOM performs better on OmniTest */
    if( !any_improvement && only_accept_improvement ) {
      pop->sols[individual_index] = backup_sol;
    }
    
    return false;
  }
  
  short gomea_t::applyAMS(solution_pt & sol, double & weighted_number_of_evaluations, solution_t & best) const
  {
    short out_of_range, improvement;
    double shrink_factor, delta_AMS; // , *solution_AMS, obj_val, cons_val;
    int m;
    
    delta_AMS     = 2;
    out_of_range  = 1;
    shrink_factor = 2;
    improvement   = 0;
    
    solution_pt backup_sol = copySolution(sol);
    
    while( (out_of_range == 1) && (shrink_factor > 1e-10) )
    {
      shrink_factor *= 0.5;
      out_of_range   = 0;
      for( m = 0; m < number_of_parameters; m++ )
      {
        // sol->param[m] = backup_sol->param[m] + shrink_factor*delta_AMS*(mean_vector[m]-old_mean_vector[m]);
        sol->param[m] = backup_sol->param[m] + shrink_factor*delta_AMS*(mean_shift_vector[m]); //stef
        if( !isParameterInRangeBounds( sol->param[m], m ) )
        {
          out_of_range = 1;
          break;
        }
      }
    }
    
    if( !out_of_range )
    {
      if(gradient_step) {
        fitness_function->evaluate_with_gradients(sol);
      } else {
        fitness_function->evaluate(sol);
      }
      weighted_number_of_evaluations++;
      
      if(solution_t::better_solution(*sol, best)) {
        best = solution_t(*sol);
      }
      
      if( solution_t::better_solution(*sol, *backup_sol) || randomRealUniform01(*rng) < 0.05 ) {
        improvement = 1;
      }
      else {
        sol = backup_sol;
      }
    } else {
      sol = backup_sol;
    }
    
    // free( solution_AMS );
    return( improvement );
  }
  
  void gomea_t::applyForcedImprovements( int individual_index, const solution_t & donor_sol )
  {
    
    // std::cout << "Applying forced improvements to individual " << individual_index << std::endl;
    int i, io, j, *order, num_touched_indices;
    double alpha;
    short improvement;
    
    improvement = 0;
    alpha = 1.0;
    
    while( alpha >= 0.01 )
    {
      alpha *= 0.5;
      order = randomPermutation( (int) linkage_model->length(), *rng );
      for( io = 0; io < linkage_model->length(); io++ )
      {
        i = order[io];
        
        num_touched_indices = (int) linkage_model->sets[i]->length();
        std::vector<size_t> touched_indices(num_touched_indices);
        for(size_t p = 0; p < touched_indices.size(); ++p) {
          touched_indices[p] = linkage_model->sets[i]->params[p];
        }
        
        solution_pt backup_sol = copySolution(pop->sols[individual_index]);
        
        for( j = 0; j < num_touched_indices; j++ ) {
          pop->sols[individual_index]->param[touched_indices[j]] = alpha*pop->sols[individual_index]->param[touched_indices[j]] + (1-alpha)*donor_sol.param[touched_indices[j]];
        }
        
        if(!fitness_function->partial_evaluations_available)
        {
          if(gradient_step) {
            fitness_function->evaluate_with_gradients(pop->sols[individual_index]);
          } else {
            fitness_function->evaluate(pop->sols[individual_index]);
          }
          weighted_number_of_evaluations++;
        } else
        {
          if(gradient_step) {
            fitness_function->partial_evaluate_with_gradients(pop->sols[individual_index], touched_indices, backup_sol);
          } else {
            fitness_function->partial_evaluate(pop->sols[individual_index], touched_indices, backup_sol);
          }
          weighted_number_of_evaluations += touched_indices.size() /(double)number_of_parameters;
        }
        
        if( solution_t::better_solution(*pop->sols[individual_index], best) ) {
          best = solution_t(*pop->sols[individual_index]);
        }
        // end installedProblemEvaluation
        
        improvement = solution_t::better_solution(*pop->sols[individual_index], *backup_sol);
        
        if( !improvement ) {
          pop->sols[individual_index] = backup_sol;
        }
        
        if( improvement ) {
          break;
        }
      }
      
      free( order );
      
      if( improvement ) {
        break;
      }
    }
    
    if( !improvement ) {
      pop->sols[individual_index] = copySolution(donor_sol);
    }
  }
  
  /**
   * Adapts distribution multipliers according to SDR-AVS mechanism.
   * Returns whether the FOS element with index FOS_index has caused
   * an improvement
   */
  void gomea_t::adaptDistributionMultiplier( double & multiplier, bool improvement, double st_dev_ratio, size_t out_of_bounds_draws, size_t samples_drawn_from_normal  ) const
  {
    if( active )
    {
      if( (((double) out_of_bounds_draws)/((double) samples_drawn_from_normal)) > 0.9 ) {
        multiplier *= 0.5;
      }
      
      if( improvement )
      {
        if( multiplier < 1.0 )
          multiplier = 1.0;
        
        if( st_dev_ratio > st_dev_ratio_threshold )
          multiplier *= distribution_multiplier_increase;
      }
      else
      {
        if( (multiplier > 1.0) || (no_improvement_stretch >= maximum_no_improvement_stretch) )
          multiplier *= distribution_multiplier_decrease;
        
        if( no_improvement_stretch < maximum_no_improvement_stretch && multiplier < 1.0)
          multiplier = 1.0;
      }
    }
    
  }
  
  /**
   * Determines whether an improvement is found for a specified
   * population. Returns 1 in case of an improvement, 0 otherwise.
   * The standard-deviation ratio required by the SDR-AVS
   * mechanism is computed and returned in the pointer variable.
   */
  size_t gomea_t::generationalImprovementForFOSElement( FOS_element_pt FOS_element, double *st_dev_ratio, const solution_t & elite ) const
  {
    
    int    number_of_improvements = 0;
    vec_t average_parameters_of_improvements(FOS_element->length(), 0.0);
    bool  generationalImprovement = 0;
    
    for(size_t i = 0; i < population_size; i++ )
    {
      if( solution_t::better_solution(*pop->sols[i], elite) )
      {
        number_of_improvements++;
        for(size_t j = 0; j < FOS_element->length(); j++ ) {
          average_parameters_of_improvements[j] += pop->sols[i]->param[FOS_element->params[j]];
        }
      }
    }
    
    // Determine st.dev. ratio
    *st_dev_ratio = 0.0;
    if( number_of_improvements > 0 )
    {
      for(size_t i = 0; i < FOS_element->length(); i++ ) {
        average_parameters_of_improvements[i] /= (double) number_of_improvements;
      }
      
      *st_dev_ratio = getStDevRatioForFOSElement( average_parameters_of_improvements, FOS_element );
      generationalImprovement = 1;
    }
    
    return number_of_improvements;
  }
  
  /**
   * Computes and returns the standard-deviation-ratio
   * of a given point for a given model.
   */
  double gomea_t::getStDevRatioForFOSElement( const vec_t & parameters, FOS_element_pt FOS_element ) const
  {
    double **inverse, result, *x_min_mu, *z;
    
    x_min_mu = (double *) Malloc( FOS_element->length()*sizeof( double ) );
    
    for(size_t i = 0; i < FOS_element->length(); i++ ) {
      x_min_mu[i] = parameters[i]-mean_vector[FOS_element->params[i]];
    }
    result = 0.0;
    
    if( FOS_element->length() == 1 ) {
      result = fabs( x_min_mu[0]/sqrt(FOS_element->covariance_matrix[0][0]) );
    }
    else
    {
      inverse = matrixLowerTriangularInverse( FOS_element->cholesky_factor_lower_triangle.toArray(), (int) FOS_element->length() );
      z = matrixVectorMultiplication( inverse, x_min_mu, (int) FOS_element->length(), (int) FOS_element->length() );
      
      for(size_t i = 0; i < FOS_element->length(); i++ )
      {
        if( fabs( z[i] ) > result )
          result = fabs( z[i] );
      }
      
      free( z );
      for(size_t i = 0; i < FOS_element->length(); i++ )
        free( inverse[i] );
      free( inverse );
    }
    
    free( x_min_mu );
    
    return( result );
  }

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
}


// Assisting functions
namespace hillvallea
{
  
  /**
   * Sorts an array of objectives and constraints
   * using constraint domination and returns the
   * sort-order (small to large).
   */
  int *mergeSortFitness( double *objectives, double *constraints, int number_of_solutions )
  {
    int i, *sorted, *tosort;
    
    sorted = (int *) Malloc( number_of_solutions * sizeof( int ) );
    tosort = (int *) Malloc( number_of_solutions * sizeof( int ) );
    for( i = 0; i < number_of_solutions; i++ )
      tosort[i] = i;
    
    if( number_of_solutions == 1 )
      sorted[0] = 0;
    else
      mergeSortFitnessWithinBounds( objectives, constraints, sorted, tosort, 0, number_of_solutions-1 );
    
    free( tosort );
    
    return( sorted );
  }
  
  /**
   * Subroutine of merge sort, sorts the part of the objectives and
   * constraints arrays between p and q.
   */
  void mergeSortFitnessWithinBounds( double *objectives, double *constraints, int *sorted, int *tosort, int p, int q )
  {
    int r;
    
    if( p < q )
    {
      r = (p + q) / 2;
      mergeSortFitnessWithinBounds( objectives, constraints, sorted, tosort, p, r );
      mergeSortFitnessWithinBounds( objectives, constraints, sorted, tosort, r+1, q );
      mergeSortFitnessMerge( objectives, constraints, sorted, tosort, p, r+1, q );
    }
  }
  
  /**
   * Subroutine of merge sort, merges the results of two sorted parts.
   */
  void mergeSortFitnessMerge( double *objectives, double *constraints, int *sorted, int *tosort, int p, int r, int q )
  {
    int i, j, k, first;
    
    i = p;
    j = r;
    for( k = p; k <= q; k++ )
    {
      first = 0;
      if( j <= q )
      {
        if( i < r )
        {
          if( betterFitness( objectives[tosort[i]], constraints[tosort[i]],
                            objectives[tosort[j]], constraints[tosort[j]] ) )
            first = 1;
        }
      }
      else
        first = 1;
      
      if( first )
      {
        sorted[k] = tosort[i];
        i++;
      }
      else
      {
        sorted[k] = tosort[j];
        j++;
      }
    }
    
    for( k = p; k <= q; k++ )
      tosort[k] = sorted[k];
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  
  /**
   * Returns 1 if x is better than y, 0 otherwise.
   * x is not better than y unless:
   * - x and y are both infeasible and x has a smaller sum of constraint violations, or
   * - x is feasible and y is not, or
   * - x and y are both feasible and x has a smaller objective value than y
   */
  
  short betterFitness( double objective_value_x, double constraint_value_x, double objective_value_y, double constraint_value_y )
  {
    short result;
    
    result = 0;
    
    if( constraint_value_x > 0 ) /* x is infeasible */
    {
      if( constraint_value_y > 0 ) /* Both are infeasible */
      {
        if( constraint_value_x < constraint_value_y )
          result = 1;
      }
    }
    else /* x is feasible */
    {
      if( constraint_value_y > 0 ) /* x is feasible and y is not */
        result = 1;
      else /* Both are feasible */
      {
        if( objective_value_x < objective_value_y )
          result = 1;
      }
    }
    
    return( result );
  }
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Ranking -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Computes the ranks of the solutions in one population.
   */
  void computeRanks(const population_t & pop, vec_t & ranks )
  {
    ranks.resize(pop.size());
    vec_t objective_values(pop.size());
    vec_t constraint_values(pop.size());
    
    for(size_t i = 0; i < pop.size(); ++i) {
      objective_values[i] = pop.sols[i]->f;
      constraint_values[i] = pop.sols[i]->constraint;
    }
    
    int * sorted = mergeSortFitness( objective_values.toArray(), constraint_values.toArray(), (int) pop.size() );
    
    int rank = 0;
    ranks[sorted[0]] = rank;
    
    for(size_t i = 1; i < pop.size(); i++ )
    {
      // scm: added constraints here as well
      if( objective_values[sorted[i]] != objective_values[sorted[i-1]] || constraint_values[sorted[i]] != constraint_values[sorted[i-1]] )
        rank++;
      
      ranks[sorted[i]] = rank;
    }
    
    free( sorted );
    
  }
  
  void computeRanksForInitialPopulation(const population_t & pop, int initial_population_size, vec_t & ranks )
  {
    
    ranks.resize(pop.size());
    vec_t objective_values(pop.size());
    vec_t constraint_values(pop.size());
    
    for(int i = 0; i < initial_population_size; ++i) {
      objective_values[i] = pop.sols[i]->f;
      constraint_values[i] = pop.sols[i]->constraint;
    }
    
    // rank only the initial solutions
    int *sorted = mergeSortFitness( objective_values.toArray(), constraint_values.toArray(), initial_population_size );
    
    int rank = 0;
    ranks[sorted[0]] = rank;
    for(int i = 1; i < initial_population_size; i++ )
    {
      // scm: added constraints here as well
      if( objective_values[sorted[i]] != objective_values[sorted[i-1]] || constraint_values[sorted[i]] != constraint_values[sorted[i-1]] )
        rank++;
      
      ranks[sorted[i]] = rank;
    }
    
    // rank all other solutions as less important.
    // assert(population_size == pop->size());
    for(size_t i = initial_population_size; i < pop.size(); i++ ) {
      ranks[i] = rank + 1;
    }
    
    free( sorted );
  }
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  
  
  /**
   * Copies the single very best of the selected solutions
   * to their respective populations.
   */
  solution_pt copySolution(const solution_t & sol)
  {
    solution_pt sol_copy = std::make_shared<solution_t>(sol);
    return sol_copy;
  }
  
  solution_pt copySolution(const solution_pt sol)
  {
    return copySolution(*sol);
  }
  
  void getBestInPopulation(const population_t & pop, int *individual_index )
  {
    *individual_index = 0;
    for(int i = 0; i < pop.size(); i++ ) {
      if( solution_t::better_solution_via_pointers(pop.sols[i], pop.sols[*individual_index])) {
        *individual_index = i;
      }
    }
  }
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Selection =-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  
  /**
   * Performs truncation selection on a single population.
   */
  void makeSelection(const population_t & pop, double selection_fraction, population_t & selection) {
    bool initial_population = false;
    int initial_popsize = (int) pop.size();
    makeSelection(pop, selection_fraction, selection, initial_population, initial_popsize);
  }
  
  void makeSelection(const population_t & pop, double selection_fraction, population_t & selection, bool initial_population, int initial_popsize )
  {
    vec_t ranks;
    if(initial_population) {
      computeRanksForInitialPopulation(pop, initial_popsize, ranks);
    } else {
      computeRanks(pop, ranks );
    }
    int *sorted = mergeSort( ranks.toArray(), (int) ranks.size() );
    if( ranks.size() > 1 && ranks[sorted[ranks.size()-1]] == 0 ) {
      makeSelectionsUsingDiversityOnRank0(pop, selection_fraction, ranks, selection );
    }
    else
    {
      size_t selection_size = std::min((int) ranks.size(), (int) (selection_fraction * pop.size()));
      selection.sols.resize(selection_size);
      for(int i = 0; i < selection_size; i++ ) {
        selection.sols[i] = copySolution(pop.sols[sorted[i]]);
      }
    }
    
    free( sorted );
  }
  
  /**
   * Performs selection from all solutions that have rank 0
   * based on diversity.
   */
  void makeSelectionsUsingDiversityOnRank0(const population_t & pop, double selection_fraction, vec_t & ranks, population_t & selection )
  {
    int     i, j, number_of_rank0_solutions, *preselection_indices,
    index_of_farthest, number_selected_so_far;
    double *nn_distances, distance_of_farthest, value;
    
    size_t selection_size = std::min((int) ranks.size(), (int) (selection_fraction * pop.size()));
    
    number_of_rank0_solutions = 0;
    for( i = 0; i < ranks.size(); i++ )
    {
      if( ranks[i] == 0 )
        number_of_rank0_solutions++;
    }
    
    preselection_indices = (int *) Malloc( number_of_rank0_solutions*sizeof( int ) );
    j                    = 0;
    for( i = 0; i < pop.size(); i++ )
    {
      if( ranks[i] == 0 )
      {
        preselection_indices[j] = i;
        j++;
      }
    }
    
    index_of_farthest    = 0;
    distance_of_farthest = pop.sols[preselection_indices[0]]->f;
    for( i = 1; i < number_of_rank0_solutions; i++ )
    {
      if( pop.sols[preselection_indices[i]]->f > distance_of_farthest )
      {
        index_of_farthest    = i;
        distance_of_farthest = pop.sols[preselection_indices[i]]->f;
      }
    }
    
    number_selected_so_far                    = 0;
    std::vector<int> selection_indices(selection_size);
    selection_indices[number_selected_so_far] = preselection_indices[index_of_farthest];
    preselection_indices[index_of_farthest]   = preselection_indices[number_of_rank0_solutions-1];
    number_of_rank0_solutions--;
    number_selected_so_far++;
    
    nn_distances = (double *) Malloc( number_of_rank0_solutions*sizeof( double ) );
    for( i = 0; i < number_of_rank0_solutions; i++ ) {
      // nn_distances[i] = distanceEuclidean( populations[preselection_indices[i]], populations[selection_indices[number_selected_so_far-1]], (int) number_of_parameters );
      nn_distances[i] = pop.sols[preselection_indices[i]]->param_distance(pop.sols[selection_indices[number_selected_so_far-1]]->param);
    }
    while( number_selected_so_far < selection_size )
    {
      index_of_farthest    = 0;
      distance_of_farthest = nn_distances[0];
      for( i = 1; i < number_of_rank0_solutions; i++ )
      {
        if( nn_distances[i] > distance_of_farthest )
        {
          index_of_farthest    = i;
          distance_of_farthest = nn_distances[i];
        }
      }
      
      selection_indices[number_selected_so_far] = preselection_indices[index_of_farthest];
      preselection_indices[index_of_farthest]   = preselection_indices[number_of_rank0_solutions-1];
      nn_distances[index_of_farthest]           = nn_distances[number_of_rank0_solutions-1];
      number_of_rank0_solutions--;
      number_selected_so_far++;
      
      for( i = 0; i < number_of_rank0_solutions; i++ )
      {
        // value = distanceEuclidean( populations[preselection_indices[i]], populations[selection_indices[number_selected_so_far-1]], (int) number_of_parameters );
        value = pop.sols[preselection_indices[i]]->param_distance(pop.sols[selection_indices[number_selected_so_far-1]]->param);
        
        if( value < nn_distances[i] )
          nn_distances[i] = value;
      }
    }
    
    selection.sols.resize(selection_size);
    
    for( i = 0; i < selection_size; i++ ) {
      // std::cout << i << " " << selection_indices[i] << "\n";
      selection.sols[i] = copySolution(pop.sols[selection_indices[i]]);
    }
    
    free( nn_distances );
    // free( selection_indices );
    free( preselection_indices );
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  
  
  // input: n x n covariance matrix
  //        sols from which the covariance matrix was estimated
  //        mean of size sol[0]->Param.size()
  //        list of parameters of size n // i.e.,
  //        n
  bool regularizeCovarianceMatrix(matrix_t & covariance, const std::vector<solution_pt> & sols, const std::vector<double> & mean, const std::vector<size_t> & parameters)
  {
    // regularization for small populations
    double number_of_samples = (double) sols.size();
    size_t n = parameters.size();

    double phi = 0.0;
    
    // y = x.^2
    // phiMat = y'*y/t-sample.^2
    // phi = sum(sum(phiMat))
    double squared_cov = 0.0;
    
    // matrix_t squared_cov(n,n,0.0);
    double temp;
    for(size_t i = 0; i < n; ++i)
    {
      for(size_t j = 0; j < n; ++j)
      {
        squared_cov = 0.0;
        
        for(size_t k = 0; k < sols.size(); ++k)
        {
          temp = (sols[k]->param[parameters[i]]-mean[parameters[i]])*(sols[k]->param[parameters[j]]-mean[parameters[j]]);
          squared_cov += (temp - covariance[i][j]) * (temp - covariance[i][j]);
        }
        squared_cov /= number_of_samples;
        
        phi += squared_cov;
      }
    }
    
    // Frobenius norm, i.e.,
    // gamma = norm(sample - prior,'fro')^2;
    double gamma = 0.0;
    
    for(size_t i = 0; i < n; ++i)
    {
      for(size_t j = 0; j < n; ++j)
      {
        if(i != j) {
          temp = fabs(covariance[i][j]);
        } else {
          temp = 0.0;
        }
        gamma += temp*temp;
      }
    }
    
    double kappa = phi/gamma;
    double shrinkage = std::max(0.0,std::min(1.0,kappa/number_of_samples));
    // std::cout << "Shrinkage with factor " << std::setprecision(3) << shrinkage << ", Kappa = " << std::setprecision(3) << kappa / number_of_samples << " phi = " << phi << std::endl;
    
    
    if(shrinkage == 0.0) {
      return false;
    }
    
    // shrinking
    for(size_t i = 0; i < n; ++i)
    {
      for(size_t j = 0; j < n; ++j)
      {
        if (i != j) { // i == j remains the same, only off-diagonals are shrunken
          covariance[i][j] = (1.0 - shrinkage) * covariance[i][j];
        }
      }
    }
    
    return true;
  }
  
  void estimateFullCovarianceMatrixML( matrix_t & full_covariance_matrix, const population_t & selection, const vec_t & mean_vector, double init_univariate_bandwidth )
  {
    if(selection.size() == 0) {
      return;
    }
    
    size_t number_of_parameters = selection.sols[0]->param.size();
    full_covariance_matrix.resize(number_of_parameters, number_of_parameters);
    
    double cov;
    for(size_t i = 0; i < number_of_parameters; i++ )
    {
      for(size_t j = 0; j < number_of_parameters; j++ )
      {
        cov = estimateCovariance(i, j, selection.sols, mean_vector, init_univariate_bandwidth);
        
        full_covariance_matrix[i][j] = cov;
        full_covariance_matrix[j][i] = cov;
      }
    }
  }
  
  double estimateCovariance(size_t vara, size_t varb, const std::vector<solution_pt> & sols, const vec_t & mean_vector, double init_univariate_bandwidth)
  {
    // copute the covariance estimate
    double cov = 0.0;
    if(sols.size() == 1)
    {
      if(vara == varb) {
        cov = init_univariate_bandwidth * 0.01;
      }
    }
    else
    {
      for(size_t m = 0; m < sols.size(); m++ ) {
        cov += (sols[m]->param[vara]-mean_vector[vara])*(sols[m]->param[varb]-mean_vector[varb]);
      }
      cov /= (double) sols.size();
    }
    
    return cov;
  }
  
  
  void estimateCovarianceMatrix( FOS_element_pt FOS_element, const population_t & selection, const vec_t & mean_vector, double init_univariate_bandwidth )
  {
    // compute covariance matrix + distribution multiplier.
    size_t n = FOS_element->length();
    FOS_element->covariance_matrix.resize(n, n);
    
    // If there are fewer than n + 1 solutions, estimate a diagonal covariance matrix
    if(!FOS_element->sample_bayesian_factorization && selection.sols.size() < n + 1)
    {
      for(size_t k = 0; k < n; k++ )
      {
        size_t vara = FOS_element->params[k];
        for(size_t m = k; m < n; m++ )
        {
          size_t varb = FOS_element->params[m];
          if(vara == varb) {
            FOS_element->covariance_matrix[k][m] = estimateCovariance(vara, varb, selection.sols, mean_vector, init_univariate_bandwidth);
          } else {
            FOS_element->covariance_matrix[k][m] = 0.0;
            FOS_element->covariance_matrix[m][k] = 0.0;
          }
        }
      }
    }
    else
    {
      // Estimate a full covariance matrix.
      if(!FOS_element->sample_bayesian_factorization)
      {
        for(size_t k = 0; k < n; k++ )
        {
          size_t vara = FOS_element->params[k];
          for(size_t m = k; m < n; m++ )
          {
            size_t varb = FOS_element->params[m];
            FOS_element->covariance_matrix[k][m] = estimateCovariance(vara, varb, selection.sols, mean_vector, init_univariate_bandwidth);
            FOS_element->covariance_matrix[m][k] = FOS_element->covariance_matrix[k][m];
          }
        }
      }
      else
      {
        // estimate only the covariances that correspond to a 1 in the bayesian factorization indicator matrix
        for(size_t k = 0; k < n; k++ )
        {
          size_t vara = FOS_element->params[k];
          for(size_t m = k; m < n; m++ )
          {
            size_t varb = FOS_element->params[m];
            if(FOS_element->bayesian_factorization_indicator_matrix[vara][varb] > 0) {
              FOS_element->covariance_matrix[k][m] = estimateCovariance(vara, varb, selection.sols, mean_vector, init_univariate_bandwidth);
            } else {
              FOS_element->covariance_matrix[k][m] = 0.0;
            }
            FOS_element->covariance_matrix[m][k] = FOS_element->covariance_matrix[k][m];
          }
        }
      }
    }
    
    // estimate conditional sample matrix

    if(FOS_element->sample_conditionally && FOS_element->params_to_condition_on.size() > 0)
    {
      
      // if the selection size is too small, we sample univariate, and we can thus stop
      size_t m = FOS_element->params_to_condition_on.size();
      if(selection.sols.size() < n + 1 || selection.sols.size() < m + 1) {
        FOS_element->mean_shift_matrix_to_condition_on.reset(n, m, 0.0);
        return;
      }

      // setup matrices
      // notation largely as in https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions.
      // matrix_t C11(n,n); = FOS_element->covariance_matrix;
      matrix_t C22(m,m);
      matrix_t C12(n,m);
      // matrix_t C21(m,n) = C12.transpose();
      

      // Estimate C22
      for(size_t k = 0; k < m; k++ )
      {
        size_t vara = FOS_element->params_to_condition_on[k];
        for(size_t l = k; l < m; l++ )
        {
          size_t varb = FOS_element->params_to_condition_on[l];
          C22[k][l] = estimateCovariance(vara, varb, selection.sols, mean_vector, init_univariate_bandwidth);
          C22[l][k] = C22[k][l];
        }
      }
      
      // Estimate C12
      for(size_t k = 0; k < n; k++ )
      {
        size_t vara = FOS_element->params[k];
        for(size_t l = 0; l < m; l++ )
        {
          size_t varb = FOS_element->params_to_condition_on[l];
          C12[k][l] = estimateCovariance(vara, varb, selection.sols, mean_vector, init_univariate_bandwidth);
        }
      }
      
      // Cholesky decompose C22.
      bool success = true;
      matrix_t L22;
      choleskyDecomposition(C22, L22, success);
      if(!success) {
        FOS_element->mean_shift_matrix_to_condition_on.reset(n, m, 0.0);
        return;
      }
      
      // std::cout << FOS_element->covariance_matrix << std::endl << std::endl << C12 << std::endl << std::endl << C22;
      
      // Set mean shift matrix
      matrix_t L22inv;
      L22inv.setRaw(matrixLowerTriangularInverse(L22.toArray(), (int) m), m, m);
      FOS_element->mean_shift_matrix_to_condition_on = C12 * (L22inv.transpose() * L22inv);
      FOS_element->covariance_matrix -= FOS_element->mean_shift_matrix_to_condition_on * (C12.transpose());
      
      //std::cout << FOS_element->covariance_matrix;
      
    }
    // end estimate conditional sample matrix
  }
  
  
  void initializeCovarianceMatrices( FOS_pt linkage_model )
  {
    for(size_t i = 0; i < linkage_model->sets.size(); ++i) {
      linkage_model->sets[i]->covariance_matrix.setIdentity(linkage_model->sets[i]->length(), linkage_model->sets[i]->length());
    }
  }
  
  void estimateMeanVectorML( const population_t & selection, vec_t & mean_vector )
  {
    size_t number_of_parameters = selection.sols[0]->param.size();
    
    // compute new mean vector
    mean_vector.resize(number_of_parameters);
    
    for(size_t i = 0; i < number_of_parameters; i++ )
    {
      mean_vector[i] = 0.0;
      
      for(size_t j = 0; j < selection.size(); j++ ) {
        mean_vector[i] += selection.sols[j]->param[i];
      }
      
      mean_vector[i] /= (double) selection.size();
    }
  }
  
  void estimateMeanVectorML( const population_t & selection, vec_t & mean_vector, vec_t & old_mean_vector, vec_t & mean_shift_vector )
  {
    // backup old mean vector
    old_mean_vector = mean_vector;
    
    estimateMeanVectorML(selection, mean_vector);
    
    // compute mean shift
    mean_shift_vector = mean_vector - old_mean_vector;
  }
  
  void estimateMeanVectorML_partial(vec_t & mean_vector, FOS_element_pt FOS_element, const population_t & selection )
  {
    size_t vara;
    
    for(int k = 0; k < FOS_element->length(); k++ )
    {
      vara =  FOS_element->params[k];
      
      double new_mean = 0.0;
      for(int j = 0; j < selection.size(); j++ ) {
        new_mean += selection.sols[j]->param[vara];
      }
      new_mean /= (double) selection.size();
      
      mean_vector[vara] = new_mean;
    }
  }
  
}
