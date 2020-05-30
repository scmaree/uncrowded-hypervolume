/**
 *
 * MO-RV-GOMEA
 *
 * If you use this software for any purpose, please cite the most recent publication:
 * A. Bouter, N.H. Luong, C. Witteveen, T. Alderliesten, P.A.N. Bosman. 2017.
 * The Multi-Objective Real-Valued Gene-pool Optimal Mixing Evolutionary Algorithm.
 * In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2017).
 * DOI: 10.1145/3071178.3071274
 *
 * Copyright (c) 1998-2017 Peter A.N. Bosman
 *
 * The software in this file is the proprietary information of
 * Peter A.N. Bosman.
 *
 * IN NO EVENT WILL THE AUTHOR OF THIS SOFTWARE BE LIABLE TO YOU FOR ANY
 * DAMAGES, INCLUDING BUT NOT LIMITED TO LOST PROFITS, LOST SAVINGS, OR OTHER
 * INCIDENTIAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR THE INABILITY
 * TO USE SUCH PROGRAM, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGES, OR FOR ANY CLAIM BY ANY OTHER PARTY. THE AUTHOR MAKES NO
 * REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. THE
 * AUTHOR SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY ANYONE AS A RESULT OF
 * USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
 *
 * The software in this file is the result of (ongoing) scientific research.
 * The following people have been actively involved in this research over
 * the years:
 * - Peter A.N. Bosman
 * - Dirk Thierens
 * - Jörn Grahl
 * - Anton Bouter
 * 
 */

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "MO-RV-GOMEA.h"
#include "../mohillvallea/hillvalleyclustering.h"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace gomea
{
  

  /*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
  hicam::vec_t lowerRangeBound, upperRangeBound;
  
  double HL_tol;
  int           number_of_objectives,
  current_population_index,
  approximation_set_size;                        /* Number of solutions in the final answer (the approximation set). */
  double        sum_of_ellipsoids_normalization_factor;
  long          number_of_full_evaluations;
  short         approximation_set_reaches_vtr,
  statistics_file_existed;
  short         objective_discretization_in_effect,            /* Whether the objective space is currently being discretized for the elitist archive. */
  *elitist_archive_indices_inactive;              /* Elitist archive solutions flagged for removal. */
  int           elitist_archive_size,
                previous_elitist_archive_size,                          /* Number of solutions in the elitist archive. */
  elitist_archive_size_target,                   /* The lower bound of the targeted size of the elitist archive. */
  approximation_set_size_target,                   /* The lower bound of the targeted size of the elitist archive. */
  elitist_archive_capacity;                      /* Current memory allocation to elitist archive. */
  double       *best_objective_values_in_elitist_archive,      /* The best objective values in the archive in the individual objectives. */
  *objective_discretization,                      /* The length of the objective discretization in each dimension (for the elitist archive). */
  **ranks;                                         /* Ranks of all solutions in all populations. */
  individual ***populations,                                   /* The population containing the solutions. */
  ***selection,                                     /* Selected solutions, one for each population. */
  **elitist_archive,                               /* Archive of elitist solutions. */
  **approximation_set;                             /* Set of non-dominated solutions from all populations and the elitist archive. */
  int number_of_elites_added_to_archive_this_generation;
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  /*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
  short      print_verbose_overview,                        /* Whether to print a overview of settings (0 = no). */
             use_guidelines;                                /* Whether to override parameters with guidelines (for those that exist). */
  int        base_population_size,
             maximum_number_of_populations,
           **cluster_index_for_population,
            *selection_sizes,                                /* The size of the selection. */
            *cluster_sizes,                                  /* The size of the clusters. */
             number_of_cluster_failures,
           **selection_indices,                             /* Indices of corresponding individuals in population for all selected individuals. */
          ***selection_indices_of_cluster_members,          /* The indices pertaining to the selection of cluster members. */
          ***selection_indices_of_cluster_members_previous, /* The indices pertaining to the selection of cluster members in the previous generation. */
           **pop_indices_selected,
           **single_objective_clusters,
           **num_individuals_in_cluster,
             maximum_number_of_evaluations,                 /* The maximum number of evaluations. */
             number_of_subgenerations_per_population_factor,
             base_number_of_mixing_components,
            *number_of_mixing_components,                   /* The number of components in the mixture distribution. */
             number_of_nearest_neighbours_in_registration,  /* The number of nearest neighbours to consider in cluster registration */
          ***samples_drawn_from_normal,                     /* The number of samples drawn from the i-th normal in the last generation. */
             samples_current_cluster,
          ***out_of_bounds_draws,                           /* The number of draws that resulted in an out-of-bounds sample. */
            *no_improvement_stretch,                        /* The number of subsequent generations without an improvement while the distribution multiplier is <= 1.0. */
             maximum_no_improvement_stretch,                /* The maximum number of subsequent generations without an improvement while the distribution multiplier is <= 1.0. */
           **number_of_elitist_solutions_copied,            /* The number of solutions from the elitist archive copied to the population. */
           **sorted_ranks;
  double     tau,                                           /* The selection truncation percentile (in [1/population_size,1]). */
             delta_AMS,                                     /* The adaptation length for AMS (anticipated mean shift). */
           **objective_ranges,                              /* Ranges of objectives observed in the current population. */
          ***objective_values_selection_previous,           /* Objective values of selected solutions in the previous generation, required for cluster registration. */
           **ranks_selection,                               /* Ranks of the selected solutions. */
            ***distribution_multipliers,                      /* Distribution multipliers (AVS mechanism) */
            ***enable_regularization,                      /* Distribution multipliers (AVS mechanism) */
             distribution_multiplier_increase,              /* The multiplicative distribution multiplier increase. */
             distribution_multiplier_decrease,              /* The multiplicative distribution multiplier decrease. */
             st_dev_ratio_threshold,                        /* The maximum ratio of the distance of the average improvement to the mean compared to the distance of one standard deviation before triggering AVS (SDR mechanism). */
          ***mean_vectors,                                  /* The mean vectors, one for each population. */
          ***mean_vectors_previous,                         /* The mean vectors of the previous generation, one for each population. */
          ***objective_means_scaled,                        /* The means of the clusters in the objective space, linearly scaled according to the observed ranges. */
        *****decomposed_covariance_matrices,                /* The covariance matrices to be used for the sampling. */
        *****decomposed_cholesky_factors_lower_triangle,    /* The unique lower triangular matrix of the Cholesky factorization for every FOS element. */
         ****full_covariance_matrix,
             maximum_number_of_seconds;
  clock_t    start, end;
  FOS       ***linkage_model;
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  short        use_boundary_repair,                          /* Repair out of bound parameter value to its nearest boundary value */
               use_forced_improvement;
  std::string write_directory, file_appendix;
  /**
   * Checks whether the selected options are feasible.
   */
  void checkOptions( void )
  {
    if( number_of_parameters < 1 )
    {
      printf("\n");
      printf("Error: number of parameters < 1 (read: %d). Require number of parameters >= 1.", number_of_parameters);
      printf("\n\n");

      exit( 0 );
    }

    if( ((int) (tau*base_population_size)) <= 0 || tau >= 1 )
    {
      printf("\n");
      printf("Error: tau not in range (read: %e). Require tau in [1/pop,1] (read: [%e,%e]).", tau, 1.0/((double) base_population_size), 1.0);
      printf("\n\n");

      exit( 0 );
    }

    if( base_population_size < 1 )
    {
      printf("\n");
      printf("Error: population size < 1 (read: %d). Require population size >= 1.", base_population_size);
      printf("\n\n");

      exit( 0 );
    }

    if( base_number_of_mixing_components < 1 )
    {
      printf("\n");
      printf("Error: number of mixing components < 1 (read: %d). Require number of mixture components >= 1.", base_number_of_mixing_components);
      printf("\n\n");

      exit( 0 );
    }

    if( elitist_archive_size_target < 1 )
    {
      printf("\n");
      printf("Error: elitist archive size target < 1 (read: %d).", elitist_archive_size_target);
      printf("\n\n");

      exit( 0 );
    }

    if( rotation_angle > 0 && ( !learn_linkage_tree && FOS_element_size >= 0 && FOS_element_size != block_size && FOS_element_size != number_of_parameters) )
    {
        printf("\n");
        printf("Error: invalid FOS element size (read %d). Must be %d, %d or %d.", FOS_element_size, 1, block_size, number_of_parameters );
        printf("\n\n");

        exit( 0 );
    }
  }

  /**
   * Prints the settings as read from the command line.
   */
  void printVerboseOverview( void )
  {
    int i;

    printf("### Settings ######################################\n");
    printf("#\n");
    printf("# Statistics writing every generation: %s\n", write_generational_statistics ? "enabled" : "disabled");
    printf("# Population file writing            : %s\n", write_generational_solutions ? "enabled" : "disabled");
    printf("# Use of value-to-reach (vtr)        : %s\n", (use_vtr != 0) ? "enabled" : "disabled");
    printf("#\n");
    printf("###################################################\n");
    printf("#\n");
    printf("# Problem                  = %s\n", fitness_function->name().c_str());
    printf("# Number of parameters     = %d\n", number_of_parameters);
    printf("# Initialization ranges    = ");
    for( i = 0; i < number_of_parameters; i++ )
    {
      printf("x_%d: [%e;%e]", i, lower_init_ranges[i], upper_init_ranges[i]);
      if( i < number_of_parameters-1 )
        printf("\n#                            ");
    }
    printf("\n");
    printf("# Boundary ranges          = ");
    for( i = 0; i < number_of_parameters; i++ )
    {
      printf("x_%d: [%e;%e]", i, lower_range_bounds[i], upper_range_bounds[i]);
      if( i < number_of_parameters-1 )
        printf("\n#                            ");
    }
    printf("\n");
    printf("# Rotation angle           = %e\n", rotation_angle);
    printf("# Tau                      = %e\n", tau);
    printf("# Population size          = %d\n", base_population_size);
    printf("# Number of populations    = %d\n", maximum_number_of_populations);
    printf("# FOS element size         = %d\n", FOS_element_size);
    printf("# Number of mix. com. (k)  = %d\n", base_number_of_mixing_components);
    printf("# Dis. mult. decreaser     = %e\n", distribution_multiplier_decrease);
    printf("# St. dev. rat. threshold  = %e\n", st_dev_ratio_threshold);
    printf("# Elitist ar. size target  = %d\n", elitist_archive_size_target);
    printf("# Maximum numb. of eval.   = %d\n", maximum_number_of_evaluations);
    printf("# Value to reach (vtr)     = %e\n", vtr);
    printf("# Time limit (s)           = %e\n", maximum_number_of_seconds);
    printf("# Random seed              = %ld\n", (long) random_seed);
    printf("#\n");
    printf("###################################################\n");
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


  /*-=-=-=-=-=-=-=-=-=-=-=-=- Section Initialize -=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
  /**
   * Performs initializations that are required before starting a run.
   */
  void initialize( void )
  {
    int i, j;

    number_of_populations = 0;
    total_number_of_generations  = 0;
    number_of_evaluations  = 0;
    number_of_full_evaluations = 0;
    approximation_set_reaches_vtr = 0;
    distribution_multiplier_increase = 1.0/distribution_multiplier_decrease;

    initializeProblem();

    for( i = 1; i < number_of_parameters; i++ )
    {
        j = (i-1) % block_size;
        sum_of_ellipsoids_normalization_factor += pow( 10.0, 6.0*(((double) (j))/((double) (block_size-1))) );
    }
    //
    delta_AMS      = 2.0;
    if( static_linkage_tree ) {
      random_linkage_tree = 1;
    }
    
    initializeMemory();

    initializeParameterRangeBounds();

    initializeObjectiveRotationMatrix();
  }

  void initializeNewPopulation( void )
  {
      current_population_index = number_of_populations;

      initializeNewPopulationMemory( number_of_populations );

      initializePopulationAndFitnessValues( number_of_populations );

      if( !learn_linkage_tree )
      {
          initializeCovarianceMatrices( number_of_populations );

          initializeDistributionMultipliers( number_of_populations );
      }

      computeObjectiveRanges( number_of_populations );

      // computeUHVIRanks( number_of_populations );
      computeRanks( number_of_populations );
      number_of_populations++;
  }

  void initializeMemory()
  {
      int i;

      elitist_archive_size     = 0;
      elitist_archive_capacity = 10;
      populations                                   = (individual ***) Malloc( maximum_number_of_populations*sizeof( individual ** ) );
      selection                                     = (individual ***) Malloc( maximum_number_of_populations*sizeof( individual ** ) );
      population_sizes                              = (int*) Malloc(maximum_number_of_populations*sizeof(int));
      populations_terminated                        = (short*) Malloc(maximum_number_of_populations*sizeof(short));
      selection_sizes                               = (int*) Malloc(maximum_number_of_populations*sizeof(int));
      cluster_sizes                                 = (int*) Malloc(maximum_number_of_populations*sizeof(int));
      cluster_index_for_population                  = (int **) Malloc( maximum_number_of_populations*sizeof( int * ) );
      ranks                                         = (double **) Malloc( maximum_number_of_populations*sizeof( double * ) );
      sorted_ranks                                  = (int **) Malloc( maximum_number_of_populations*sizeof( int * ) );
      objective_ranges                              = (double **) Malloc( maximum_number_of_populations*sizeof( double * ) );
      selection_indices                             = (int **) Malloc( maximum_number_of_populations*sizeof( int * ) );
      objective_values_selection_previous           = (double ***) Malloc( maximum_number_of_populations*sizeof( double ** ) );
      ranks_selection                               = (double **) Malloc( maximum_number_of_populations*sizeof( double * ) );
      pop_indices_selected                          = (int **) Malloc( maximum_number_of_populations*sizeof( int * ) );
      number_of_elitist_solutions_copied            = (int **) Malloc( maximum_number_of_populations*sizeof( int * ) );
      objective_means_scaled                        = (double ***) Malloc( maximum_number_of_populations*sizeof( double ** ) );
      mean_vectors                                  = (double ***) Malloc( maximum_number_of_populations*sizeof( double ** ) );
      mean_vectors_previous                         = (double ***) Malloc( maximum_number_of_populations*sizeof( double ** ) );
      decomposed_cholesky_factors_lower_triangle    = (double *****) Malloc( maximum_number_of_populations*sizeof( double **** ) );
      selection_indices_of_cluster_members          = (int ***) Malloc( maximum_number_of_populations*sizeof( int ** ) );
      selection_indices_of_cluster_members_previous = (int ***) Malloc( maximum_number_of_populations*sizeof( int ** ) );
      single_objective_clusters                     = (int **) Malloc( maximum_number_of_populations*sizeof( int * ) );
      num_individuals_in_cluster                    = (int **) Malloc( maximum_number_of_populations*sizeof( int * ) );
      number_of_mixing_components                   = (int *) Malloc( maximum_number_of_populations*sizeof( int ) );
    distribution_multipliers                      = (double ***) Malloc( maximum_number_of_populations*sizeof( double ** ) );
    enable_regularization                      = (double ***) Malloc( maximum_number_of_populations*sizeof( double ** ) );
      decomposed_covariance_matrices                = (double *****) Malloc( maximum_number_of_populations * sizeof( double ****) );
      full_covariance_matrix                        = (double ****) Malloc( maximum_number_of_populations * sizeof( double ***) );
      samples_drawn_from_normal                     = (int ***) Malloc( maximum_number_of_populations*sizeof( int ** ) );
      out_of_bounds_draws                           = (int ***) Malloc( maximum_number_of_populations*sizeof( int ** ) );
      number_of_generations                         = (int *) Malloc( maximum_number_of_populations*sizeof( int ) );
      no_improvement_stretch                        = (int *) Malloc( maximum_number_of_populations*sizeof( int ) );
      linkage_model                                 = (FOS ***) Malloc( maximum_number_of_populations*sizeof( FOS ** ) );

      objective_discretization                      = (double *) Malloc( number_of_objectives*sizeof( double ) );
      elitist_archive                               = (individual **) Malloc( elitist_archive_capacity*sizeof( individual * ) );
      best_objective_values_in_elitist_archive      = (double *) Malloc( number_of_objectives*sizeof( double ) );
      elitist_archive_indices_inactive         = (short *) Malloc( elitist_archive_capacity*sizeof( short ) );

      for( i = 0; i < elitist_archive_capacity; i++ )
      {
        elitist_archive[i] = initializeIndividual();
        elitist_archive_indices_inactive[i] = 0;
      }

      for( i = 0; i < number_of_objectives; i++ )
      {
        best_objective_values_in_elitist_archive[i] = 1e+308;
      }

      for(i = 0; i < maximum_number_of_populations; i++ )
      {
        distribution_multipliers[i] = NULL;
        enable_regularization[i] = NULL;
        samples_drawn_from_normal[i] = NULL;
        out_of_bounds_draws[i] = NULL;
      }
  }


  /**
   * Initializes the memory.
   */
  void initializeNewPopulationMemory( int population_index )
  {
    int i;

    if( population_index == 0 )
    {
        population_sizes[population_index] = base_population_size;
        number_of_mixing_components[population_index] = base_number_of_mixing_components;
    }
    else
    {
        population_sizes[population_index] = 2*population_sizes[population_index-1];
        number_of_mixing_components[population_index] = number_of_mixing_components[population_index-1] + 1;
    }
    selection_sizes[population_index] = population_sizes[population_index];//HACK(int) (tau*(population_sizes[population_index]));
    cluster_sizes[population_index]   = (2*((int) (tau*(population_sizes[population_index]))))/number_of_mixing_components[population_index];//HACK(2*selection_size)/number_of_mixing_components;
    number_of_generations[population_index]  = 0;
    populations_terminated[population_index] = 0;
    no_improvement_stretch[population_index] = 0;

    populations[population_index]                                   = (individual **) Malloc( population_sizes[population_index]*sizeof( individual * ) );
    selection[population_index]                                     = (individual **) Malloc( selection_sizes[population_index]*sizeof( individual * ) );
    cluster_index_for_population[population_index]                  = (int *) Malloc( population_sizes[population_index]*sizeof( int ) );
    ranks[population_index]                                         = (double *) Malloc( population_sizes[population_index]*sizeof( double ) );
    sorted_ranks[population_index]                                  = (int *) Malloc( population_sizes[population_index]*sizeof( int ) );
    objective_ranges[population_index]                              = (double *) Malloc( population_sizes[population_index]*sizeof( double ) );
    selection_indices[population_index]                             = (int *) Malloc( selection_sizes[population_index]*sizeof( int ) );
    objective_values_selection_previous[population_index]           = (double **) Malloc( selection_sizes[population_index]*sizeof( double * ) );
    ranks_selection[population_index]                               = (double *) Malloc( selection_sizes[population_index]*sizeof( double ) );
    pop_indices_selected[population_index]                          = (int *) Malloc( population_sizes[population_index]*sizeof( int ) );
    number_of_elitist_solutions_copied[population_index]            = (int *) Malloc( number_of_mixing_components[population_index]*sizeof( int ) );
    objective_means_scaled[population_index]                        = (double **) Malloc( number_of_mixing_components[population_index]*sizeof( double * ) );
    mean_vectors[population_index]                                  = (double **) Malloc( number_of_mixing_components[population_index]*sizeof( double * ) );
    mean_vectors_previous[population_index]                         = (double **) Malloc( number_of_mixing_components[population_index]*sizeof( double * ) );
    decomposed_cholesky_factors_lower_triangle[population_index]    = (double ****) Malloc( number_of_mixing_components[population_index]*sizeof( double *** ) );
    selection_indices_of_cluster_members[population_index]          = (int **) Malloc( number_of_mixing_components[population_index]*sizeof( int * ) );
    selection_indices_of_cluster_members_previous[population_index] = (int **) Malloc( number_of_mixing_components[population_index]*sizeof( int * ) );
    single_objective_clusters[population_index]                     = (int *) Malloc( number_of_mixing_components[population_index]*sizeof( int ) );
    num_individuals_in_cluster[population_index]                    = (int *) Malloc( number_of_mixing_components[population_index]*sizeof( int ) );
    linkage_model[population_index]                                 = (FOS **) Malloc( number_of_mixing_components[population_index]*sizeof( FOS* ) );

    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      linkage_model[population_index][i] = NULL;

    for( i = 0; i < population_sizes[population_index]; i++ )
      populations[population_index][i] = initializeIndividual();

    for( i = 0; i < selection_sizes[population_index]; i++ )
    {
      selection[population_index][i] = initializeIndividual();

      objective_values_selection_previous[population_index][i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
    }

    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
    {
      mean_vectors[population_index][i] = (double *) Malloc( number_of_parameters*sizeof( double ) );

      mean_vectors_previous[population_index][i] = (double *) Malloc( number_of_parameters*sizeof( double ) );

      selection_indices_of_cluster_members[population_index][i] = NULL;

      selection_indices_of_cluster_members_previous[population_index][i] = NULL;

      objective_means_scaled[population_index][i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
    }

    if( learn_linkage_tree )
    {
        full_covariance_matrix[population_index] = (double ***) Malloc( number_of_mixing_components[population_index]*sizeof( double ** ) );
    }
    else
    {
        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
          initializeFOS( population_index, i );
    }
  }

  void initializeCovarianceMatrices( int population_index )
  {
      int i, j, k, m;

      decomposed_covariance_matrices[population_index]  = (double ****) Malloc( number_of_mixing_components[population_index] * sizeof( double ***) );
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      {
          decomposed_covariance_matrices[population_index][i] = (double ***) Malloc( linkage_model[population_index][i]->length * sizeof( double **) );
          for( j = 0; j < linkage_model[population_index][i]->length; j++ )
          {
              decomposed_covariance_matrices[population_index][i][j] = (double **) Malloc( linkage_model[population_index][i]->set_length[j]*sizeof( double * ) );
              for( k = 0; k < linkage_model[population_index][i]->set_length[j]; k++)
              {
                  decomposed_covariance_matrices[population_index][i][j][k] = (double *) Malloc( linkage_model[population_index][i]->set_length[j]*sizeof( double ) );
                  for( m = 0; m < linkage_model[population_index][i]->set_length[j]; m++)
                  {
                      decomposed_covariance_matrices[population_index][i][j][k][m] = 1;
                  }
              }
          }
      }
  }

  /**
   * Initializes the distribution multipliers.
   */
  void initializeDistributionMultipliers( int population_index )
  {
      int i, j;

      distribution_multipliers[population_index] = (double **) Malloc( number_of_mixing_components[population_index]*sizeof( double * ) );
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      {
          distribution_multipliers[population_index][i] = (double *) Malloc( linkage_model[population_index][i]->length*sizeof( double ) );
          for( j = 0; j < linkage_model[population_index][i]->length; j++ )
              distribution_multipliers[population_index][i][j] = 1.0;
      }
    
    enable_regularization[population_index] = (double **) Malloc( number_of_mixing_components[population_index]*sizeof( double * ) );
    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
    {
      enable_regularization[population_index][i] = (double *) Malloc( linkage_model[population_index][i]->length*sizeof( double ) );
      for( j = 0; j < linkage_model[population_index][i]->length; j++ )
        enable_regularization[population_index][i][j] = 0.0;
    }

      samples_drawn_from_normal[population_index] = (int **) Malloc( number_of_mixing_components[population_index]*sizeof( int * ) );
      out_of_bounds_draws[population_index]       = (int **) Malloc( number_of_mixing_components[population_index]*sizeof( int * ) );
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      {
          samples_drawn_from_normal[population_index][i] = (int *) Malloc( linkage_model[population_index][i]->length*sizeof( int ) );
          out_of_bounds_draws[population_index][i]       = (int *) Malloc( linkage_model[population_index][i]->length*sizeof( int ) );
      }
  }

  short initializePopulationProblemSpecific( int population_index )
  {
    return( 0 );
  }

  /**
   * Initializes the population and the fitness values.
   */
  void initializePopulationAndFitnessValues( int population_index )
  {
    int i, j;
    short problem_specific_initialize;

    problem_specific_initialize = initializePopulationProblemSpecific( population_index );
    if( !problem_specific_initialize )
    {
        for( i = 0; i < population_sizes[population_index]; i++ )
            for( j = 0; j < number_of_parameters; j++ )
                populations[population_index][i]->parameters[j] = lower_init_ranges[j] + (upper_init_ranges[j] - lower_init_ranges[j])*randomRealUniform01();
    }

    for( i = 0; i < population_sizes[population_index]; i++ )
    {
      installedProblemEvaluation( populations[population_index][i], number_of_parameters, NULL, NULL, NULL, 0 );

      updateElitistArchive( populations[population_index][i] );
    }
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Ranking -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Computes the ranks of the solutions in all populations.
   */
  void computeRanks( int population_index )
  {
    short **domination_matrix, is_illegal;
    int     i, j, k, *being_dominated_count, rank,
            number_of_solutions_ranked, *indices_in_this_rank;

    for( i = 0; i < population_sizes[population_index]; i++ )
    {
      is_illegal = 0;
      for( j = 0; j < number_of_objectives; j++ )
      {
        if( isnan( populations[population_index][i]->objective_values[j] ) )
        {
          is_illegal = 1;
          break;
        }
      }
      if( isnan( populations[population_index][i]->constraint_value ) )
        is_illegal = 1;

      if( is_illegal )
      {
        for( j = 0; j < number_of_objectives; j++ )
          populations[population_index][i]->objective_values[j] = 1e+308;
        populations[population_index][i]->constraint_value = 1e+308;
      }
    }

    /* The domination matrix stores for each solution i
     * whether it dominates solution j, i.e. domination[i][j] = 1. */
    domination_matrix = (short **) Malloc( population_sizes[population_index]*sizeof( short * ) );
    for( i = 0; i < population_sizes[population_index]; i++ )
      domination_matrix[i] = (short *) Malloc( population_sizes[population_index]*sizeof( short ) );

    being_dominated_count = (int *) Malloc( population_sizes[population_index]*sizeof( int ) );

    for( i = 0; i < population_sizes[population_index]; i++ )
    {
      being_dominated_count[i] = 0;
      for( j = 0; j < population_sizes[population_index]; j++ )
        domination_matrix[i][j] = 0;
    }

    for( i = 0; i < population_sizes[population_index]; i++ )
    {
      for( j = 0; j < population_sizes[population_index]; j++ )
      {
        if( i != j )
        {
          if( constraintParetoDominates( populations[population_index][i]->objective_values, populations[population_index][i]->constraint_value, populations[population_index][j]->objective_values, populations[population_index][j]->constraint_value ) )
          {
            domination_matrix[i][j] = 1;
            being_dominated_count[j]++;
          }
        }
      }
    }

    /* Compute ranks from the domination matrix */
    rank                       = 0;
    number_of_solutions_ranked = 0;
    indices_in_this_rank       = (int *) Malloc( population_sizes[population_index]*sizeof( int ) );
    while( number_of_solutions_ranked < population_sizes[population_index] )
    {
      k = 0;
      for( i = 0; i < population_sizes[population_index]; i++ )
      {
        if( being_dominated_count[i] == 0 )
        {
          ranks[population_index][i]  = rank;
          indices_in_this_rank[k]     = i;
          k++;
          being_dominated_count[i]--;
          number_of_solutions_ranked++;
        }
      }

      for( i = 0; i < k; i++ )
      {
        for( j = 0; j < population_sizes[population_index]; j++ )
        {
          if( domination_matrix[indices_in_this_rank[i]][j] == 1 )
            being_dominated_count[j]--;
        }
      }

      rank++;
    }

    free( indices_in_this_rank );

    free( being_dominated_count );

    for( i = 0; i < population_sizes[population_index]; i++ )
      free( domination_matrix[i] );
    free( domination_matrix );
  }
  
  // distance to a box defined by [-infty, ref_x, -infty, ref_y]
  double distance_to_box(double ref_x, double ref_y, double p_x, double p_y)
  {
    double dx = max(0.0, p_x - ref_x );
    double dy = max(0.0, p_y - ref_y );
    
    return sqrt(dx*dx + dy*dy);
    
  }
  
  double distance_to_front(double p_x, double p_y, const std::vector<double> & obj_x, const std::vector<double> & obj_y, std::vector<size_t> & sorted_obj, double r_x, double r_y)
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
    
    assert(min_dist >= 0); // can be 0 if its at the front!
    return min_dist;
  }
  
  void computeRandomRanks( int population_index)
  {
    
    int *random_ranks = randomPermutation(population_sizes[population_index]);
    
    for(size_t i = 0; i < population_sizes[population_index]; ++i) {
      ranks[population_index][i] = (double) random_ranks[i];
    }
    
    free(random_ranks);
  }
  
  
  void computeUHVIRanks( int population_index )
  {

    for(size_t i = 0; i < population_sizes[population_index]; i++ )
    {
      bool is_illegal = 0;
      for(size_t j = 0; j < number_of_objectives; j++ )
      {
        if( isnan( populations[population_index][i]->objective_values[j] ) )
        {
          is_illegal = 1;
          break;
        }
      }
      if( isnan( populations[population_index][i]->constraint_value ) )
        is_illegal = 1;
      
      if( is_illegal )
      {
        for(size_t j = 0; j < number_of_objectives; j++ ) {
          populations[population_index][i]->objective_values[j] = 1e+308;
        }
        populations[population_index][i]->constraint_value = 1e+308;
      }
    }
    
    // compute UHVIs
    
    hicam::rng_pt rng = std::make_shared<hicam::rng_t>(1104913 + total_number_of_generations);
    hicam::elitist_archive_pt temp_elitist_archive = std::make_shared<hicam::elitist_archive_t>(elitist_archive_size_target, rng);
    
    temp_elitist_archive->sols.reserve(elitist_archive_size_target);
    
    // copy the gomea archive to a HICAM data structure
    for(int i = 0; i < elitist_archive_size; i++ ) {
      if(!elitist_archive_indices_inactive[i]) {
        temp_elitist_archive->updateArchive(IndividualToSol(elitist_archive[i]));
      }
    }

    double HV_archive = temp_elitist_archive->compute2DHyperVolume(fitness_function->hypervolume_max_f0, fitness_function->hypervolume_max_f1);
    temp_elitist_archive->removeSolutionNullptrs();
    std::vector<size_t> sorted_obj;
    std::vector<double> obj_x(temp_elitist_archive->size(), 0.0);
    std::vector<double> obj_y(temp_elitist_archive->size(), 0.0);
    
    for(size_t i = 0; i < temp_elitist_archive->size(); ++i) {
      obj_x[i] = temp_elitist_archive->sols[i]->obj[0];
      obj_y[i] = temp_elitist_archive->sols[i]->obj[1];
    }
    
    std::vector<double> uhvi(population_sizes[population_index], 0.0); // negative, lower = better.
    
    for(size_t i = 0; i < population_sizes[population_index]; i++ )
    {
      bool hit = false;
      
      for(size_t j = 0; j < temp_elitist_archive->size(); j++ )
      {
        // if the solution dominates the elitist archive, add it and compute the NEW HV,
        // which should be larger, therefore UHVI < 0
        // this does not happen when the archive is up-to-date.
        if( constraintParetoDominates( populations[population_index][i]->objective_values, populations[population_index][i]->constraint_value, &temp_elitist_archive->sols[j]->obj[0], temp_elitist_archive->sols[j]->constraint ) )
        {
          hicam::elitist_archive_pt temp_elitist_archive2 = std::make_shared<hicam::elitist_archive_t>(*temp_elitist_archive);
          temp_elitist_archive2->updateArchive(IndividualToSol(populations[population_index][i]));
          
          double new_HV_archive = temp_elitist_archive2->compute2DHyperVolume(fitness_function->hypervolume_max_f0, fitness_function->hypervolume_max_f1);
          uhvi[i] = HV_archive - new_HV_archive;
          // assert(uhvi[i] < 0);
          hit = true;
          break;
        }
        else
        {
          // if the solution is the elitist archive, remove it and compute the NEW HV,
          // which should be smaller (as we removed a sol)
          // we minimize, therefore, this should be < 0
          if(fabs(populations[population_index][i]->objective_values[0] - temp_elitist_archive->sols[j]->obj[0]) < 1e-16 && fabs(populations[population_index][i]->objective_values[1] - temp_elitist_archive->sols[j]->obj[1]) < 1e-16 && fabs(populations[population_index][i]->constraint_value - temp_elitist_archive->sols[j]->constraint) < 1e-16)
          {
            hicam::elitist_archive_pt temp_elitist_archive2 = std::make_shared<hicam::elitist_archive_t>(*temp_elitist_archive);
            temp_elitist_archive2->sols[j] = nullptr;
            temp_elitist_archive2->removeSolutionNullptrs();
            
            double new_HV_archive = temp_elitist_archive2->compute2DHyperVolume(fitness_function->hypervolume_max_f0, fitness_function->hypervolume_max_f1);
            uhvi[i] = new_HV_archive - HV_archive; // i'm not sure this is how it should be, because we compute the HVI with respect to the archive, but there are more sols in the pop, so this UHVI is somehow an overestimate.
            // assert(uhvi[i] < 0);
            hit = true;
            break;
          }
        }
      }
      
      if(!hit)
      {
        // this sol is dominated, set uhvi[i] > 0!
        uhvi[i] = distance_to_front(populations[population_index][i]->objective_values[0], populations[population_index][i]->objective_values[1], obj_x, obj_y, sorted_obj, fitness_function->hypervolume_max_f0, fitness_function->hypervolume_max_f1);
        // assert(uhvi[i] > 0);
      }
    }
    
    
    int *sorted_uhvi;
    
    // sort something
    sorted_uhvi = mergeSort( &uhvi[0], (int) uhvi.size());
    
    
    // todo: update this line
    for(size_t i = 0; i < population_sizes[population_index]; i++ ) {
      ranks[population_index][sorted_uhvi[i]]  = i;
    }
    
    
    
    free( sorted_uhvi );

  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





  /*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Output =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Computes the ranges of all fitness values
   * of all solutions currently in the populations.
   */
  void computeObjectiveRanges( int population_index )
  {
    int    i, j;
    double low, high;

    for( j = 0; j < number_of_objectives; j++ )
    {
      low  = 1e+308;
      high = -1e+308;

      for( i = 0; i < population_sizes[population_index]; i++ )
      {
        if( populations[population_index][i]->objective_values[j] < low )
          low = populations[population_index][i]->objective_values[j];
        if( populations[population_index][i]->objective_values[j] > high && (populations[population_index][i]->objective_values[j] <= 1e+300) )
          high = populations[population_index][i]->objective_values[j];
      }

      objective_ranges[population_index][j] = high - low;
    }
  }

  /**
   * Returns whether a solution is inside the range bound of
   * every objective function.
   */
  short isSolutionInRangeBoundsForFOSElement( double *solution, int population_index, int cluster_index, int FOS_index )
  {
    int i;

    for( i = 0; i < linkage_model[population_index][cluster_index]->set_length[FOS_index]; i++ )
      if( !isParameterInRangeBounds( solution[linkage_model[population_index][cluster_index]->sets[FOS_index][i]], linkage_model[population_index][cluster_index]->sets[FOS_index][i] ) )
        return( 0 );

    return( 1 );
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





  /*-=-=-=-=-=-=-=-=-=-=-=-=- Section Termination -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Returns 1 if termination should be enforced, 0 otherwise.
   */
  short checkTerminationConditionAllPopulations( void )
  {
    int i;

    if( number_of_populations == 0 )
        return( 0 );

    if( checkNumberOfEvaluationsTerminationCondition() ) {
      std::cout << "Terminated because of fevals." << std::endl;
      return( 1 );
    }
    if( checkVTRTerminationCondition() ) {
      std::cout << "Terminated because of VTR. Yay!" << std::endl;
        return( 1 );
    }
      
    if( checkTimeLimitTerminationCondition() ) {
      std::cout << "Terminated because of time." << std::endl;
        return( 1 );
    }
    for( i = 0; i < number_of_populations; i++ )
        if( checkDistributionMultiplierTerminationCondition( i ) )
          populations_terminated[i] = 1;

    // check if all populations are terminated
    if(number_of_populations == maximum_number_of_populations)
    {
      short all_populations_terminated = 1;
      for( i = 0; i < number_of_populations; i++ )
      {
        if( !populations_terminated[i] ) {
          all_populations_terminated = 0;
          break;
        }
      }
      
      if( all_populations_terminated ) {
        std::cout << "Terminated because of internal reasons." << std::endl;
        return ( 1 );
      }
    }
      
    return( 0 );
  }

  short checkTerminationConditionOnePopulation( int population_index )
  {
      if( number_of_populations == 0 )
          return( 0 );

      if( checkNumberOfEvaluationsTerminationCondition() )
        return( 1 );

      if( checkVTRTerminationCondition() )
          return( 1 );

      if( checkTimeLimitTerminationCondition() )
          return( 1 );

      if( checkDistributionMultiplierTerminationCondition( population_index ) )
        populations_terminated[population_index] = 1;

    
    // hack : terminate if more than HL_tol of the pop is rank0.
    if(HL_tol > 0)
    {
      size_t number_of_rank_0_sols = 0;
      size_t worst_rank = 0;
      for(size_t i = 0; i < population_sizes[population_index]; ++i) {
        if(ranks[population_index][i] > worst_rank) {
          worst_rank = ranks[population_index][i];
        }
        if(ranks[population_index][i] == 0) {
          number_of_rank_0_sols++;
        }
      }
      
      if(number_of_rank_0_sols > HL_tol * ((double) population_sizes[population_index])) {
        return( 1 );
      }
      
      //if( number_of_elites_added_to_archive_this_generation == 0 ) {
      //  std::cout << "no elites added to archive\n";
      //  return ( 1 );
      //}
      
      if (objective_discretization_in_effect) {
        return ( 1 );
      }
      
      
    }
    
      return( 0 );
  }

  /**
   * Returns 1 if the maximum number of evaluations
   * has been reached, 0 otherwise.
   */
  short checkNumberOfEvaluationsTerminationCondition( void )
  {
    if( number_of_evaluations >= maximum_number_of_evaluations && maximum_number_of_evaluations > 0 )
      return( 1 );

    return( 0 );
  }

  /**
   * Returns 1 if the value-to-reach has been reached
   * for the multi-objective case. This means that
   * the D_Pf->S metric has reached the value-to-reach.
   * If no D_Pf->S can be computed, 0 is returned.
   */
  short checkVTRTerminationCondition( void )
  {
    int      default_front_size;
    double **default_front, metric_elitist_archive;

    if( use_vtr != 0)
    {
      // IGD based
      if(use_vtr == 1 && number_of_objectives == 2)
      {
        std::cout << "Warning, hypervolume-based termination not yet implemented for MO-GOMEA.\n";
        use_vtr = 0;
      }
      
      // IGD based
      if( use_vtr == 2 && haveDPFSMetric() )
      {
        if( approximation_set_reaches_vtr )
          return ( 1 );

        default_front          = getDefaultFront( &default_front_size );
        metric_elitist_archive = computeDPFSMetric( default_front, default_front_size, elitist_archive, elitist_archive_size, elitist_archive_indices_inactive );

        if( default_front )
        {
          for(int i = 0; i < default_front_size; i++ )
            free( default_front[i] );
          free( default_front );
        }
        
        if( metric_elitist_archive <= vtr )
          return( 1 );
      }
    }
    
    // nothing hit
    return( 0 );
  }

  /**
   * Checks whether the distribution multiplier for any mixture component
   * has become too small (0.5).
   */
  short checkDistributionMultiplierTerminationCondition( int population_index )
  {
    int i, j;

    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
    {
        for( j = 0; j < linkage_model[population_index][i]->length; j++ )
          if( distribution_multipliers[population_index][i][j] > 1e-10 )
              return( 0 );
    }

    std::cout << "Terminated because of multiplier." << std::endl;
    return( 1 );
  }

  short checkTimeLimitTerminationCondition( void )
  {
      if( maximum_number_of_seconds > 0 && getTimer() > maximum_number_of_seconds )
          return( 1 );
      return( 0 );
  }

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





  /*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Selection =-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Makes a set of selected solutions by taking the solutions from all
   * ranks up to the rank of the solution at the selection-size boundary.
   * The selection is filled using a diverse selection from the final rank.
   */
  void makeSelection( int population_index )
  {
    
    number_of_elites_added_to_archive_this_generation = 0;
    
    int i, j, k, individuals_selected, individuals_to_select, last_selected_rank, elitist_solutions_copied;

    for( i = 0; i < selection_sizes[population_index]; i++ )
      for( j = 0; j < number_of_objectives; j++ )
        objective_values_selection_previous[population_index][i][j] = selection[population_index][i]->objective_values[j];

    for( i = 0; i < population_sizes[population_index]; i++ )
      pop_indices_selected[population_index][i] = -1;

    free( sorted_ranks[population_index] );
    sorted_ranks[population_index] = mergeSort( ranks[population_index], population_sizes[population_index] );

    // Copy elitist archive to selection
    elitist_solutions_copied = 0;
    individuals_selected = elitist_solutions_copied;
    individuals_to_select = ((int) (tau*population_sizes[population_index]))-elitist_solutions_copied;
    last_selected_rank = (int) ranks[population_index][sorted_ranks[population_index][individuals_to_select-1]];

    i = 0;
    while( ((int) ranks[population_index][sorted_ranks[population_index][i]]) != last_selected_rank )
    {
      for( j = 0; j < number_of_parameters; j++ )
        selection[population_index][individuals_selected]->parameters[j] = populations[population_index][sorted_ranks[population_index][i]]->parameters[j];
      for( j = 0; j < number_of_objectives; j++ )
        selection[population_index][individuals_selected]->objective_values[j]  = populations[population_index][sorted_ranks[population_index][i]]->objective_values[j];
      selection[population_index][individuals_selected]->constraint_value       = populations[population_index][sorted_ranks[population_index][i]]->constraint_value;
      ranks_selection[population_index][individuals_selected]                   = ranks[population_index][sorted_ranks[population_index][i]];
      selection_indices[population_index][individuals_selected]                 = sorted_ranks[population_index][i];
      pop_indices_selected[population_index][sorted_ranks[population_index][i]] = individuals_selected;

      i++;
      individuals_selected++;
    }

    int *selected_indices, start_index;
    selected_indices = NULL;
    if( individuals_selected < individuals_to_select )
      selected_indices = completeSelectionBasedOnDiversityInLastSelectedRank( population_index, i, individuals_to_select-individuals_selected, sorted_ranks[population_index] );

    if( selected_indices )
    {
      start_index = i;
      for( j = 0; individuals_selected < individuals_to_select; individuals_selected++, j++ )
        pop_indices_selected[population_index][sorted_ranks[population_index][selected_indices[j]+start_index]] = individuals_selected;
    }

    j = individuals_to_select;
    for( i = 0; i < population_sizes[population_index]; i++ )
    {
      if( pop_indices_selected[population_index][i] == -1 )
      {
        for( k = 0; k < number_of_parameters; k++ )
          selection[population_index][j]->parameters[k] = populations[population_index][i]->parameters[k];
        for( k = 0; k < number_of_objectives; k++ )
          selection[population_index][j]->objective_values[k] = populations[population_index][i]->objective_values[k];
        selection[population_index][j]->constraint_value = populations[population_index][i]->constraint_value;
        ranks_selection[population_index][j] = ranks[population_index][i];
        selection_indices[population_index][j] = i;
        pop_indices_selected[population_index][i] = j;
        j++;
      }
    }

    if( selected_indices )
      free( selected_indices );
  }

  /**
   * Fills up the selection by using greedy diversity selection
   * in the last selected rank.
   */
  int *completeSelectionBasedOnDiversityInLastSelectedRank( int population_index, int start_index, int number_to_select, int *sorted )
  {
    int      i, j, *selected_indices, number_of_points, number_of_dimensions;
    double **points;

    /* Determine the solutions to select from */
    number_of_points = 0;
    while( ranks[population_index][sorted[start_index+number_of_points]] == ranks[population_index][sorted[start_index]] )
    {
      number_of_points++;
      if( (start_index+number_of_points) == population_sizes[population_index] )
        break;
    }

    points = (double **) Malloc( number_of_points*sizeof( double * ) );
    for( i = 0; i < number_of_points; i++ )
      points[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
    for( i = 0; i < number_of_points; i++ )
      for( j = 0; j < number_of_objectives; j++ )
        points[i][j] = populations[population_index][sorted[start_index+i]]->objective_values[j]/objective_ranges[population_index][j];

    /* Select */
    number_of_dimensions = number_of_objectives;
    selected_indices     = greedyScatteredSubsetSelection( points, number_of_points, number_of_dimensions, number_to_select );

    /* Copy to selection */
    for( i = 0; i < number_to_select; i++ )
    {
      for( j = 0; j < number_of_parameters; j++ )
        selection[population_index][i+start_index]->parameters[j] = populations[population_index][sorted[selected_indices[i]+start_index]]->parameters[j];
      for( j = 0; j < number_of_objectives; j++ )
        selection[population_index][i+start_index]->objective_values[j] = populations[population_index][sorted[selected_indices[i]+start_index]]->objective_values[j];
      selection[population_index][i+start_index]->constraint_value = populations[population_index][sorted[selected_indices[i]+start_index]]->constraint_value;
      ranks_selection[population_index][i+start_index] = ranks[population_index][sorted[selected_indices[i]+start_index]];
      selection_indices[population_index][i+start_index] = sorted[selected_indices[i]+start_index];
    }

    for( i = 0; i < number_of_points; i++ )
      free( points[i] );
    free( points );

    return( selected_indices );
  }

  /**
   * Selects n points from a set of points. A
   * greedy heuristic is used to find a good
   * scattering of the selected points. First,
   * a point is selected with a maximum value
   * in a randomly selected dimension. The
   * remaining points are selected iteratively.
   * In each iteration, the point selected is
   * the one that maximizes the minimal distance
   * to the points selected so far.
   */
  int *greedyScatteredSubsetSelection( double **points, int number_of_points, int number_of_dimensions, int number_to_select )
  {
    int     i, index_of_farthest, random_dimension_index, number_selected_so_far,
           *indices_left, *result;
    double *nn_distances, distance_of_farthest, value;

    if( number_to_select > number_of_points )
    {
      printf("\n");
      printf("Error: greedyScatteredSubsetSelection asked to select %d solutions from set of size %d.", number_to_select, number_of_points);
      printf("\n\n");

      exit( 0 );
    }

    result = (int *) Malloc( number_to_select*sizeof( int ) );

    indices_left = (int *) Malloc( number_of_points*sizeof( int ) );
    for( i = 0; i < number_of_points; i++ )
      indices_left[i] = i;

    /* Find the first point: maximum value in a randomly chosen dimension */
    random_dimension_index = randomInt( number_of_dimensions );

    index_of_farthest    = 0;
    distance_of_farthest = points[indices_left[index_of_farthest]][random_dimension_index];
    for( i = 1; i < number_of_points; i++ )
    {
      if( points[indices_left[i]][random_dimension_index] > distance_of_farthest )
      {
        index_of_farthest    = i;
        distance_of_farthest = points[indices_left[i]][random_dimension_index];
      }
    }

    number_selected_so_far          = 0;
    result[number_selected_so_far]  = indices_left[index_of_farthest];
    indices_left[index_of_farthest] = indices_left[number_of_points-number_selected_so_far-1];
    number_selected_so_far++;

    /* Then select the rest of the solutions: maximum minimum
     * (i.e. nearest-neighbour) distance to so-far selected points */
    nn_distances = (double *) Malloc( number_of_points*sizeof( double ) );
    for( i = 0; i < number_of_points-number_selected_so_far; i++ )
      nn_distances[i] = distanceEuclidean( points[indices_left[i]], points[result[number_selected_so_far-1]], number_of_dimensions );

    while( number_selected_so_far < number_to_select )
    {
      index_of_farthest    = 0;
      distance_of_farthest = nn_distances[0];
      for( i = 1; i < number_of_points-number_selected_so_far; i++ )
      {
        if( nn_distances[i] > distance_of_farthest )
        {
          index_of_farthest    = i;
          distance_of_farthest = nn_distances[i];
        }
      }

      result[number_selected_so_far]  = indices_left[index_of_farthest];
      indices_left[index_of_farthest] = indices_left[number_of_points-number_selected_so_far-1];
      nn_distances[index_of_farthest] = nn_distances[number_of_points-number_selected_so_far-1];
      number_selected_so_far++;

      for( i = 0; i < number_of_points-number_selected_so_far; i++ )
      {
        value = distanceEuclidean( points[indices_left[i]], points[result[number_selected_so_far-1]], number_of_dimensions );
        if( value < nn_distances[i] )
          nn_distances[i] = value;
      }
    }

    free( nn_distances );
    free( indices_left );

    return( result );
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


  /*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Variation -==-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * First estimates the parameters of a normal mixture distribution
   * in the parameter space from the selected solutions. Then copies
   * the best selected solutions. Finally fills up the population,
   * after the variances of the mixture components have been scaled,
   * by drawing new samples from normal mixture distribution and applying
   * AMS to several of these new solutions. Then, the fitness ranks
   * are recomputed. Finally, the distribution multipliers for the
   * mixture components are adapted according to the SDR-AVS mechanism.
   */
  void makePopulation( int population_index )
  {
    current_population_index = population_index;

    estimateParameters( population_index );

    applyDistributionMultipliers( population_index );

    generateAndEvaluateNewSolutionsToFillPopulationAndUpdateElitistArchive( population_index );

    // computeUHVIRanks( population_index );
    computeRanks( population_index );
    computeObjectiveRanges( population_index );

    // HL-filter!
    //----------------------------------------------------------------------
    /* if (false) // technically not required
    // if(HL_tol)
    {
      hicam::hvc_t hvc(fitness_function);
      hicam::population_t pop;
      pop.sols.reserve(elitist_archive_size);
      
      for(int i = 0; i < elitist_archive_size; ++i )
      {
        if(!elitist_archive_indices_inactive[i])
        {
          pop.sols.push_back(IndividualToSol(elitist_archive[i]));
          pop.sols.back()->population_number = i; // use this to track which solutions remain.
        }
      }
      
      int pop_size_before = (int) pop.size();
      hvc.HL_filter(pop, pop, HL_tol);
      
      int number_of_removed_indices = pop_size_before - (int) pop.size();
      // remove indices from elitist archive;
      if (number_of_removed_indices > 0)
      {
        int *indices = (int *) Malloc( number_of_removed_indices *sizeof( int ) );
        
        size_t insert_index = 0;
        for(int elite_idx = 0; elite_idx < elitist_archive_size; ++elite_idx)
        {
          
          if(elitist_archive_indices_inactive[elite_idx]) {
            continue;
          }
          
          bool found = false;
          for(size_t i = 0; i < pop.size(); ++i)
          {
            if(pop.sols[i]->population_number == elite_idx) {
              found = true;
              break;
            }
          }
          
          if(!found) {
            indices[insert_index] = elite_idx;
            insert_index++;
          }
        }
        
        assert(insert_index == number_of_removed_indices);
        
        removeFromElitistArchive(indices, number_of_removed_indices);
        
        free(indices);
      }
    }
    // END HL-filter
    */
    adaptObjectiveDiscretization();

    ezilaitiniParametersForSampling( population_index );
  }

  /**
   * Estimates the parameters of the multivariate normal
   * mixture distribution.
   */
  void estimateParameters( int population_index )
  {
    short   *clusters_now_already_registered, *clusters_previous_already_registered;
    int      i, j, k, m, q, i_min, j_min, *selection_indices_of_leaders,
             number_of_dimensions, number_to_select,
           **selection_indices_of_cluster_members_before_registration,
            *k_means_cluster_sizes, **selection_indices_of_cluster_members_k_means,
            *nearest_neighbour_choice_best, number_of_clusters_left_to_register,
            *sorted, *r_nearest_neighbours_now, *r_nearest_neighbours_previous,
             number_of_clusters_to_register_by_permutation,
             number_of_cluster_permutations, **all_cluster_permutations;
    double **objective_values_selection_scaled, **objective_values_selection_previous_scaled,
             distance, distance_smallest, distance_largest, **objective_means_scaled_new, **objective_means_scaled_previous,
            *distances_to_cluster, **distance_cluster_i_now_to_cluster_j_previous,
           **distance_cluster_i_now_to_cluster_j_now, **distance_cluster_i_previous_to_cluster_j_previous,
             epsilon;

    /* Determine the leaders */
    objective_values_selection_scaled = (double **) Malloc( selection_sizes[population_index]*sizeof( double * ) );
    for( i = 0; i < selection_sizes[population_index]; i++ )
      objective_values_selection_scaled[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
    for( i = 0; i < selection_sizes[population_index]; i++ )
      for( j = 0; j < number_of_objectives; j++ )
        objective_values_selection_scaled[i][j] = selection[population_index][i]->objective_values[j]/objective_ranges[population_index][j];

    /* Heuristically find k far-apart leaders, taken from an artificial selection */
    int leader_selection_size;

    leader_selection_size = tau*population_sizes[population_index];

    number_of_dimensions         = number_of_objectives;
    number_to_select             = number_of_mixing_components[population_index];
    selection_indices_of_leaders = greedyScatteredSubsetSelection( objective_values_selection_scaled, leader_selection_size, number_of_dimensions, number_to_select );

    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      for( j = 0; j < number_of_objectives; j++ )
        objective_means_scaled[population_index][i][j] = selection[population_index][selection_indices_of_leaders[i]]->objective_values[j]/objective_ranges[population_index][j];

    /* Perform k-means clustering with leaders as initial mean guesses */
    objective_means_scaled_new = (double **) Malloc( number_of_mixing_components[population_index]*sizeof( double * ) );
    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      objective_means_scaled_new[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );

    objective_means_scaled_previous = (double **) Malloc( number_of_mixing_components[population_index]*sizeof( double * ) );
    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      objective_means_scaled_previous[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );

    selection_indices_of_cluster_members_k_means = (int **) Malloc( number_of_mixing_components[population_index]*sizeof( int * ) );
    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      selection_indices_of_cluster_members_k_means[i] = (int *) Malloc( selection_sizes[population_index]*sizeof( int ) );

    k_means_cluster_sizes = (int *) Malloc( number_of_mixing_components[population_index]*sizeof( int ) );

    for ( j = 0; j < number_of_mixing_components[population_index] - number_of_objectives; j++ )
        single_objective_clusters[population_index][j] = -1;
    for( j = 0; j < number_of_objectives; j++ )
        single_objective_clusters[population_index][number_of_mixing_components[population_index]-number_of_objectives+j] = j;

    epsilon = 1e+308;
  //BEGIN HACK: This essentially causes the code to skip k-means clustering
    epsilon = 0;
    for( j = 0; j < number_of_mixing_components[population_index]; j++ )
      k_means_cluster_sizes[j] = 0;
  //END HACK
    while( epsilon > 1e-10 )
    {
      for( j = 0; j < number_of_mixing_components[population_index]-number_of_objectives; j++ )
      {
        k_means_cluster_sizes[j] = 0;
        for( k = 0; k < number_of_objectives; k++ )
          objective_means_scaled_new[j][k] = 0.0;
      }

      for( i = 0; i < selection_sizes[population_index]; i++ )
      {
        j_min             = -1;
        distance_smallest = -1;
        for( j = 0; j < number_of_mixing_components[population_index]-number_of_objectives; j++ )
        {
          distance = distanceEuclidean( objective_values_selection_scaled[i], objective_means_scaled[population_index][j], number_of_objectives );
          if( (distance_smallest < 0) || (distance < distance_smallest) )
          {
            j_min             = j;
            distance_smallest = distance;
          }
        }
        selection_indices_of_cluster_members_k_means[j_min][k_means_cluster_sizes[j_min]] = i;
        for( k = 0; k < number_of_objectives; k++ )
          objective_means_scaled_new[j_min][k] += objective_values_selection_scaled[i][k];
        k_means_cluster_sizes[j_min]++;
      }

      for( j = 0; j < number_of_mixing_components[population_index]-number_of_objectives; j++ )
        for( k = 0; k < number_of_objectives; k++ )
          objective_means_scaled_new[j][k] /= (double) k_means_cluster_sizes[j];

      epsilon = 0;
      for( j = 0; j < number_of_mixing_components[population_index]-number_of_objectives; j++ )
      {
        epsilon += distanceEuclidean( objective_means_scaled[population_index][j], objective_means_scaled_new[j], number_of_objectives );
        for( k = 0; k < number_of_objectives; k++ )
          objective_means_scaled[population_index][j][k] = objective_means_scaled_new[j][k];
      }
    }

    /* Do leader-based distance assignment */
    distances_to_cluster = (double *) Malloc( selection_sizes[population_index]*sizeof( double ) );
    for( i = 0; i < number_of_mixing_components[population_index]-number_of_objectives; i++ )
    {
      for( j = 0; j < selection_sizes[population_index]; j++ )
        distances_to_cluster[j] = distanceEuclidean( objective_values_selection_scaled[j], objective_means_scaled[population_index][i], number_of_objectives );
      for( j = leader_selection_size; j < selection_sizes[population_index]; j++ )
        distances_to_cluster[j] = 1e+308;//HACK

      if( selection_indices_of_cluster_members_previous[population_index][i] != NULL )
        free( selection_indices_of_cluster_members_previous[population_index][i] );
      selection_indices_of_cluster_members_previous[population_index][i] = selection_indices_of_cluster_members[population_index][i];
      selection_indices_of_cluster_members[population_index][i]          = mergeSort( distances_to_cluster, selection_sizes[population_index] );
    }

    // For k-th objective, create a cluster consisting of only the best solutions in that objective (from the overall selection)
    for( j = number_of_mixing_components[population_index]-number_of_objectives; j < number_of_mixing_components[population_index]; j++ )
    {
      double *individual_objectives, worst;

      individual_objectives = (double *) Malloc( selection_sizes[population_index]*sizeof( double ) );

      worst = -1e+308;
      for( i = 0; i < selection_sizes[population_index]; i++ )
      {
        individual_objectives[i] = selection[population_index][i]->objective_values[j-(number_of_mixing_components[population_index]-number_of_objectives)];
        if( individual_objectives[i] > worst )
          worst = individual_objectives[i];
      }
      for( i = 0; i < selection_sizes[population_index]; i++ )
      {
        if( selection[population_index][i]->constraint_value != 0 )
          individual_objectives[i] = worst + selection[population_index][i]->constraint_value;
      }

      if( selection_indices_of_cluster_members_previous[population_index][j] != NULL )
        free( selection_indices_of_cluster_members_previous[population_index][j] );
      selection_indices_of_cluster_members_previous[population_index][j] = selection_indices_of_cluster_members[population_index][j];
      selection_indices_of_cluster_members[population_index][j] = mergeSort( individual_objectives, selection_sizes[population_index] );
      free( individual_objectives );
    }

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
    if( number_of_generations[population_index] > 0 )
    {
      number_of_nearest_neighbours_in_registration = 7;

      objective_values_selection_previous_scaled = (double **) Malloc( selection_sizes[population_index]*sizeof( double * ) );
      for( i = 0; i < selection_sizes[population_index]; i++ )
        objective_values_selection_previous_scaled[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );

      for( i = 0; i < selection_sizes[population_index]; i++ )
        for( j = 0; j < number_of_objectives; j++ )
          objective_values_selection_previous_scaled[i][j] = objective_values_selection_previous[population_index][i][j]/objective_ranges[population_index][j];

      selection_indices_of_cluster_members_before_registration = (int **) Malloc( number_of_mixing_components[population_index]*sizeof( int * ) );
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        selection_indices_of_cluster_members_before_registration[i] = selection_indices_of_cluster_members[population_index][i];

      distance_cluster_i_now_to_cluster_j_previous = (double **) Malloc( number_of_mixing_components[population_index]*sizeof( double * ) );
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        distance_cluster_i_now_to_cluster_j_previous[i] = (double *) Malloc( number_of_mixing_components[population_index]*sizeof( double ) );

      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      {
        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
        {
          distance_cluster_i_now_to_cluster_j_previous[i][j] = 0;
          for( k = 0; k < cluster_sizes[population_index]; k++ )
          {
            distance_smallest = -1;
            for( q = 0; q < cluster_sizes[population_index]; q++ )
            {
              distance = distanceEuclidean( objective_values_selection_scaled[selection_indices_of_cluster_members_before_registration[i][k]],objective_values_selection_previous_scaled[selection_indices_of_cluster_members_previous[population_index][j][q]], number_of_objectives );
              if( (distance_smallest < 0) || (distance < distance_smallest) )
                distance_smallest = distance;
            }
            distance_cluster_i_now_to_cluster_j_previous[i][j] += distance_smallest;
          }
        }
      }

      distance_cluster_i_now_to_cluster_j_now = (double **) Malloc( number_of_mixing_components[population_index]*sizeof( double * ) );
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        distance_cluster_i_now_to_cluster_j_now[i] = (double *) Malloc( number_of_mixing_components[population_index]*sizeof( double ) );

      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      {
        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
        {
          distance_cluster_i_now_to_cluster_j_now[i][j] = 0;
          if( i != j )
          {
            for( k = 0; k < cluster_sizes[population_index]; k++ )
            {
              distance_smallest = -1;
              for( q = 0; q < cluster_sizes[population_index]; q++ )
              {
                distance = distanceEuclidean( objective_values_selection_scaled[selection_indices_of_cluster_members_before_registration[i][k]], objective_values_selection_scaled[selection_indices_of_cluster_members_before_registration[j][q]], number_of_objectives );
                if( (distance_smallest < 0) || (distance < distance_smallest) )
                  distance_smallest = distance;
              }
              distance_cluster_i_now_to_cluster_j_now[i][j] += distance_smallest;
            }
          }
        }
      }

      distance_cluster_i_previous_to_cluster_j_previous = (double **) Malloc( number_of_mixing_components[population_index]*sizeof( double * ) );
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        distance_cluster_i_previous_to_cluster_j_previous[i] = (double *) Malloc( number_of_mixing_components[population_index]*sizeof( double ) );

      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      {
        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
        {
          distance_cluster_i_previous_to_cluster_j_previous[i][j] = 0;
          if( i != j )
          {
            for( k = 0; k < cluster_sizes[population_index]; k++ )
            {
              distance_smallest = -1;
              for( q = 0; q < cluster_sizes[population_index]; q++ )
              {
                distance = distanceEuclidean( objective_values_selection_previous_scaled[selection_indices_of_cluster_members_previous[population_index][i][k]], objective_values_selection_previous_scaled[selection_indices_of_cluster_members_previous[population_index][j][q]], number_of_objectives );
                if( (distance_smallest < 0) || (distance < distance_smallest) )
                  distance_smallest = distance;
              }
              distance_cluster_i_previous_to_cluster_j_previous[i][j] += distance_smallest;
            }
          }
        }
      }

      clusters_now_already_registered      = (short *) Malloc( number_of_mixing_components[population_index]*sizeof( short ) );
      clusters_previous_already_registered = (short *) Malloc( number_of_mixing_components[population_index]*sizeof( short ) );
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      {
        clusters_now_already_registered[i]      = 0;
        clusters_previous_already_registered[i] = 0;
      }

      r_nearest_neighbours_now      = (int *) Malloc( (number_of_nearest_neighbours_in_registration+1)*sizeof( int ) );
      r_nearest_neighbours_previous = (int *) Malloc( (number_of_nearest_neighbours_in_registration+1)*sizeof( int ) );
      nearest_neighbour_choice_best = (int *) Malloc( (number_of_nearest_neighbours_in_registration+1)*sizeof( int ) );

      number_of_clusters_left_to_register = number_of_mixing_components[population_index];
      while( number_of_clusters_left_to_register > 0 )
      {
        /* Find the two clusters in the current generation that are farthest apart and haven't been registered yet */
        i_min            = -1;
        j_min            = -1;
        distance_largest = -1;
        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          if( clusters_now_already_registered[i] == 0 )
          {
            for( j = 0; j < number_of_mixing_components[population_index]; j++ )
            {
              if( (i != j) && (clusters_now_already_registered[j] == 0) )
              {
                distance = distance_cluster_i_now_to_cluster_j_now[i][j];
                if( (distance_largest < 0) || (distance > distance_largest) )
                {
                  distance_largest = distance;
                  i_min            = i;
                  j_min            = j;
                }
              }
            }
          }
        }

        if( i_min == -1 )
        {
          for(i = 0; i < number_of_mixing_components[population_index]; i++ )
          if( clusters_now_already_registered[i] == 0 )
          {
             i_min = i;
             break;
          }
        }

        /* Find the r nearest clusters of one of the two far-apart clusters that haven't been registered yet */
        sorted = mergeSort( distance_cluster_i_now_to_cluster_j_now[i_min], number_of_mixing_components[population_index] );
        j = 0;
        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          if( clusters_now_already_registered[sorted[i]] == 0 )
          {
            r_nearest_neighbours_now[j]                = sorted[i];
            clusters_now_already_registered[sorted[i]] = 1;
            j++;
          }
          if( j == number_of_nearest_neighbours_in_registration && number_of_clusters_left_to_register-j != 1 )
            break;
        }
        number_of_clusters_to_register_by_permutation = j;
        free( sorted );

        /* Find the closest cluster from the previous generation */
        j_min             = -1;
        distance_smallest = -1;
        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
        {
          if( clusters_previous_already_registered[j] == 0 )
          {
            distance = distance_cluster_i_now_to_cluster_j_previous[i_min][j];
            if( (distance_smallest < 0) || (distance < distance_smallest) )
            {
              distance_smallest = distance;
              j_min             = j;
            }
          }
        }

        /* Find the r nearest clusters of one of the the closest cluster from the previous generation */
        sorted = mergeSort( distance_cluster_i_previous_to_cluster_j_previous[j_min], number_of_mixing_components[population_index] );
        j = 0;
        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          if( clusters_previous_already_registered[sorted[i]] == 0 )
          {
            r_nearest_neighbours_previous[j]                = sorted[i];
            clusters_previous_already_registered[sorted[i]] = 1;
            j++;
          }
          if( j == number_of_clusters_to_register_by_permutation )
            break;
        }
        free( sorted );

        /* Register the r selected clusters from the current and the previous generation */
        all_cluster_permutations = allPermutations( number_of_clusters_to_register_by_permutation, &number_of_cluster_permutations );
        distance_smallest = -1;
        for( i = 0; i < number_of_cluster_permutations; i++ )
        {
          distance = 0;
          for( j = 0; j < number_of_clusters_to_register_by_permutation; j++ )
            distance += distance_cluster_i_now_to_cluster_j_previous[r_nearest_neighbours_now[j]][r_nearest_neighbours_previous[all_cluster_permutations[i][j]]];
          if( (distance_smallest < 0) || (distance < distance_smallest) )
          {
            distance_smallest = distance;
            for( j = 0; j < number_of_clusters_to_register_by_permutation; j++ )
              nearest_neighbour_choice_best[j] = r_nearest_neighbours_previous[all_cluster_permutations[i][j]];
          }
        }
        for( i = 0; i < number_of_cluster_permutations; i++ )
          free( all_cluster_permutations[i] );
        free( all_cluster_permutations );

        for( i = 0; i < number_of_clusters_to_register_by_permutation; i++ )
        {
          selection_indices_of_cluster_members[population_index][nearest_neighbour_choice_best[i]] = selection_indices_of_cluster_members_before_registration[r_nearest_neighbours_now[i]];
          if( r_nearest_neighbours_now[i] >= number_of_mixing_components[population_index] - number_of_objectives )
          {
              single_objective_clusters[population_index][nearest_neighbour_choice_best[i]] = r_nearest_neighbours_now[i] - (number_of_mixing_components[population_index] - number_of_objectives);
              single_objective_clusters[population_index][r_nearest_neighbours_now[i]] = -1;
          }
        }

        number_of_clusters_left_to_register -= number_of_clusters_to_register_by_permutation;
      }

      free( nearest_neighbour_choice_best );
      free( r_nearest_neighbours_previous );
      free( r_nearest_neighbours_now );
      free( clusters_now_already_registered );
      free( clusters_previous_already_registered );
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        free( distance_cluster_i_previous_to_cluster_j_previous[i] );
      free( distance_cluster_i_previous_to_cluster_j_previous );
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        free( distance_cluster_i_now_to_cluster_j_now[i] );
      free( distance_cluster_i_now_to_cluster_j_now );
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        free( distance_cluster_i_now_to_cluster_j_previous[i] );
      free( distance_cluster_i_now_to_cluster_j_previous );
      free( selection_indices_of_cluster_members_before_registration );
      for( i = 0; i < selection_sizes[population_index]; i++ )
        free( objective_values_selection_previous_scaled[i] );
      free( objective_values_selection_previous_scaled );
    }

    // Compute objective means
    for( j = 0; j < number_of_mixing_components[population_index]; j++ )
      for( k = 0; k < number_of_objectives; k++ )
        objective_means_scaled[population_index][j][k] = 0.0;

    for( j = 0; j < number_of_mixing_components[population_index]; j++ )
      for( k = 0; k < number_of_objectives; k++ )
        for( q = 0; q < cluster_sizes[population_index]; q++ )
          objective_means_scaled[population_index][j][k] += objective_values_selection_scaled[selection_indices_of_cluster_members[population_index][j][q]][k];

    for( j = 0; j < number_of_mixing_components[population_index]; j++ )
      for( k = 0; k < number_of_objectives; k++ )
        objective_means_scaled[population_index][j][k] /= (double) cluster_sizes[population_index];

    int **full_rankings = (int**) Malloc(number_of_mixing_components[population_index]*sizeof(int*));
    double *distances = (double*) Malloc(selection_sizes[population_index]*sizeof(double));
    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
    {
      for( j = 0; j < selection_sizes[population_index]; j++ )
        distances[j] = distanceEuclidean( objective_values_selection_scaled[j], objective_means_scaled[population_index][i], number_of_objectives );
      full_rankings[i] = mergeSort( distances, selection_sizes[population_index] );
    }

    // Assign exactly 'cluster_size' individuals of the population to each cluster
    for( i = 0; i < number_of_mixing_components[population_index]; i++ ) num_individuals_in_cluster[population_index][i] = 0;
    for( i = 0; i < population_sizes[population_index]; i++ ) cluster_index_for_population[population_index][i] = -1;

    for( j = 0; j < cluster_sizes[population_index]; j++ )
    {
        for( i = number_of_mixing_components[population_index]-1; i >= 0; i-- )
        {
            int inc = 0;
            int individual_index = selection_indices[population_index][selection_indices_of_cluster_members[population_index][i][j]];
            while( cluster_index_for_population[population_index][individual_index] != -1 )
            {
                individual_index = selection_indices[population_index][full_rankings[i][j+inc]];
                inc++;
            }
            cluster_index_for_population[population_index][individual_index] = i;
            num_individuals_in_cluster[population_index][i]++;
        }
    }
    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        free( full_rankings[i] );
    free( full_rankings );
    free( distances );

    double *objective_values_scaled = (double *) Malloc( number_of_objectives*sizeof( double ) );
    int index_smallest;
    for( i = 0; i < population_sizes[population_index]; i++ )
    {
        if( cluster_index_for_population[population_index][i] != -1 )
            continue;

        for( j = 0; j < number_of_objectives; j++ )
          objective_values_scaled[j] = populations[population_index][i]->objective_values[j]/objective_ranges[population_index][j];

        distance_smallest = -1;
        index_smallest = -1;
        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
        {
          distance = distanceEuclidean( objective_values_scaled, objective_means_scaled[population_index][j], number_of_objectives );
          if( (distance_smallest < 0) || (distance < distance_smallest) )
          {
            index_smallest      = j;
            distance_smallest   = distance;
          }
        }
        cluster_index_for_population[population_index][i] = index_smallest;
        num_individuals_in_cluster[population_index][index_smallest]++;
    }
    free( objective_values_scaled );

    /* Elitism, must be done here, before possibly changing the focus of each cluster to an elitist solution */
    copyBestSolutionsToPopulation( population_index, objective_values_selection_scaled );

    /* Estimate the parameters */
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      {
        /* Means */
        if( number_of_generations[population_index] > 0 )
        {
          for( j = 0; j < number_of_parameters; j++ )
            mean_vectors_previous[population_index][i][j] = mean_vectors[population_index][i][j];
        }

        for( j = 0; j < number_of_parameters; j++ )
        {
          mean_vectors[population_index][i][j] = 0.0;

          for( k = 0; k < cluster_sizes[population_index]; k++ )
            mean_vectors[population_index][i][j] += selection[population_index][selection_indices_of_cluster_members[population_index][i][k]]->parameters[j];

          mean_vectors[population_index][i][j] /= (double) cluster_sizes[population_index];
        }
      }

      if( learn_linkage_tree )
      {
          for( i = 0; i < number_of_mixing_components[population_index]; i++ )
          {
              estimateFullCovarianceMatrixML( population_index, i );

              linkage_model[population_index][i] = learnLinkageTreeRVGOMEA( population_index, i );

              for( j = 0; j < number_of_parameters; j++ )
                free( full_covariance_matrix[population_index][i][j] );
              free( full_covariance_matrix[population_index][i] );
          }

          initializeCovarianceMatrices( population_index );

          if( number_of_generations[population_index] == 0 )
              initializeDistributionMultipliers( population_index );
      }

      int vara, varb, cluster_index, fos_length;
      double cov;
      for ( cluster_index = 0; cluster_index < number_of_mixing_components[population_index]; cluster_index++ )
      {
        /* First do the maximum-likelihood estimate from data */
        for( i = 0; i < linkage_model[population_index][cluster_index]->length; i++)
        {
          
          if (cluster_sizes[population_index] < linkage_model[population_index][cluster_index]->length + 1)
          {
            // univariate covariance estimate
            fos_length = linkage_model[population_index][cluster_index]->set_length[i];
            for( j = 0; j < fos_length; j++)
            {
              vara = linkage_model[population_index][cluster_index]->sets[i][j];
              for( k = j; k < fos_length; k++ )
              {
                varb = linkage_model[population_index][cluster_index]->sets[i][k];
                cov = 0.0;
                
                if (vara == varb) {
                  for( m = 0; m < cluster_sizes[population_index]; m++ ) {
                    cov += (selection[population_index][selection_indices_of_cluster_members[population_index][cluster_index][m]]->parameters[vara]-mean_vectors[population_index][cluster_index][vara])
                            *(selection[population_index][selection_indices_of_cluster_members[population_index][cluster_index][m]]->parameters[varb]-mean_vectors[population_index][cluster_index][varb]);
                  }
                }
                
                cov /= (double) cluster_sizes[population_index];
                decomposed_covariance_matrices[population_index][cluster_index][i][j][k] = cov;
                decomposed_covariance_matrices[population_index][cluster_index][i][k][j] = cov;
              }
            }
            
          }
          else
          {
          
            // full estimate
            fos_length = linkage_model[population_index][cluster_index]->set_length[i];
            for( j = 0; j < fos_length; j++)
            {
              vara = linkage_model[population_index][cluster_index]->sets[i][j];
              for( k = j; k < fos_length; k++ )
              {
                  varb = linkage_model[population_index][cluster_index]->sets[i][k];
                  cov = 0.0;

                  for( m = 0; m < cluster_sizes[population_index]; m++ )
                      cov += (selection[population_index][selection_indices_of_cluster_members[population_index][cluster_index][m]]->parameters[vara]-mean_vectors[population_index][cluster_index][vara])
                              *(selection[population_index][selection_indices_of_cluster_members[population_index][cluster_index][m]]->parameters[varb]-mean_vectors[population_index][cluster_index][varb]);

                  cov /= (double) cluster_sizes[population_index];
                  decomposed_covariance_matrices[population_index][cluster_index][i][j][k] = cov;
                  decomposed_covariance_matrices[population_index][cluster_index][i][k][j] = cov;
              }
            }
          }
        }
      }
    

    free( distances_to_cluster );
    free( k_means_cluster_sizes );
    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      free( selection_indices_of_cluster_members_k_means[i] );
    free( selection_indices_of_cluster_members_k_means );
    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      free( objective_means_scaled_new[i] );
    free( objective_means_scaled_new );
    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      free( objective_means_scaled_previous[i] );
    free( objective_means_scaled_previous );
    for( i = 0; i < selection_sizes[population_index]; i++ )
      free( objective_values_selection_scaled[i] );
    free( objective_values_selection_scaled );
    free( selection_indices_of_leaders );

  }

  /**
   * Elitism: copies at most 1/k*tau*n solutions per cluster
   * from the elitist archive.
   */
  void copyBestSolutionsToPopulation( int population_index, double **objective_values_selection_scaled )
  {
    int     i, j, j_min, k, index, **elitist_archive_indices_per_cluster, so_index,
           *number_of_elitist_archive_indices_per_cluster, max, *diverse_indices, skipped;
    double  distance, distance_smallest, *objective_values_scaled, **points;

    number_of_elitist_archive_indices_per_cluster = (int *) Malloc( number_of_mixing_components[population_index]*sizeof( int ) );
    elitist_archive_indices_per_cluster = (int **) Malloc( number_of_mixing_components[population_index]*sizeof( int * ) );
    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
    {
      number_of_elitist_archive_indices_per_cluster[i] = 0;
      number_of_elitist_solutions_copied[population_index][i] = 0;
      elitist_archive_indices_per_cluster[i]           = (int *) Malloc( elitist_archive_size*sizeof( int ) );
    }
    objective_values_scaled = (double *) Malloc( number_of_objectives*sizeof( double ) );

    for( i = 0; i < elitist_archive_size; i++ )
    {
      if( elitist_archive_indices_inactive[i] )
          continue;
      for( j = 0; j < number_of_objectives; j++ )
        objective_values_scaled[j] = elitist_archive[i]->objective_values[j]/objective_ranges[population_index][j];
      j_min             = -1;
      distance_smallest = -1;
      for( j = 0; j < number_of_mixing_components[population_index]; j++ )
      {
        distance = distanceEuclidean( objective_values_scaled, objective_means_scaled[population_index][j], number_of_objectives );
        if( (distance_smallest < 0) || (distance < distance_smallest) )
        {
          j_min             = j;
          distance_smallest = distance;
        }
      }

      elitist_archive_indices_per_cluster[j_min][number_of_elitist_archive_indices_per_cluster[j_min]] = i;
      number_of_elitist_archive_indices_per_cluster[j_min]++;
    }

    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
    {
      max = (int) (tau*num_individuals_in_cluster[population_index][i]);
      skipped = 0;
      if( number_of_elitist_archive_indices_per_cluster[i] <= max )
      {
        for( j = 0; j < number_of_elitist_archive_indices_per_cluster[i]; j++ )
        {
          index = sorted_ranks[population_index][population_sizes[population_index]-1-skipped]; //BLA
          while( cluster_index_for_population[population_index][index] != i && (population_sizes[population_index]-1-skipped) > 0 )
              index = sorted_ranks[population_index][population_sizes[population_index]-1-(++skipped)];
          if( cluster_index_for_population[population_index][index] != i )
              break;
          so_index = single_objective_clusters[population_index][i];
          if( so_index != -1 && populations[population_index][index]->objective_values[so_index] < elitist_archive[elitist_archive_indices_per_cluster[i][j]]->objective_values[so_index] )
              continue;
          copyIndividual(elitist_archive[elitist_archive_indices_per_cluster[i][j]], populations[population_index][index]);
          populations[population_index][index]->NIS = 0;
          skipped++;
        }
        number_of_elitist_solutions_copied[population_index][i] = j;
      }
      else
      {
        points = (double **) Malloc( number_of_elitist_archive_indices_per_cluster[i]*sizeof( double * ) );
        for( j = 0; j < number_of_elitist_archive_indices_per_cluster[i]; j++ )
          points[j] = (double *) Malloc( number_of_objectives*sizeof( double ) );
        for( j = 0; j < number_of_elitist_archive_indices_per_cluster[i]; j++ )
        {
          for( k = 0; k < number_of_objectives; k++ )
            points[j][k] = elitist_archive[elitist_archive_indices_per_cluster[i][j]]->objective_values[k]/objective_ranges[population_index][k];
        }
        diverse_indices = greedyScatteredSubsetSelection( points, number_of_elitist_archive_indices_per_cluster[i], number_of_objectives, max );
        for( j = 0; j < max; j++ )
        {
          index = sorted_ranks[population_index][population_sizes[population_index]-1-skipped]; //BLA
          while( cluster_index_for_population[population_index][index] != i && (population_sizes[population_index]-1-skipped) > 0 )
              index = sorted_ranks[population_index][population_sizes[population_index]-1-(++skipped)];
          if( cluster_index_for_population[population_index][index] != i )
              break;
          so_index = single_objective_clusters[population_index][i];
          if( so_index != -1 && populations[population_index][index]->objective_values[so_index] < elitist_archive[elitist_archive_indices_per_cluster[i][j]]->objective_values[so_index] )
              continue;
          copyIndividual(elitist_archive[elitist_archive_indices_per_cluster[i][diverse_indices[j]]],populations[population_index][index]);
          populations[population_index][index]->NIS = 0;
          skipped++;
        }
        number_of_elitist_solutions_copied[population_index][i] = j;
        free( diverse_indices );
        for( j = 0; j < number_of_elitist_archive_indices_per_cluster[i]; j++ )
          free( points[j] );
        free( points );
      }
    }

    free( objective_values_scaled );
    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      free( elitist_archive_indices_per_cluster[i] );
    free( elitist_archive_indices_per_cluster );
    free( number_of_elitist_archive_indices_per_cluster );
  }

  /**
   * Initializes the FOS
   */
  void initializeFOS( int population_index, int cluster_index )
  {
      int      i;
      FILE    *file;
      FOS     *new_FOS;

      fflush( stdout );
      file = fopen( "FOS.in", "r" );
      if( file != NULL )
      {
        if( population_index == 0 && cluster_index == 0 )
          new_FOS = readFOSFromFile( file );
        else
          new_FOS = copyFOS( linkage_model[0][0] );
      }
      else
      {
          if( static_linkage_tree )
          {
            if( population_index == 0 && cluster_index == 0 )
            {
                new_FOS = learnLinkageTree( NULL );
            }
            else
              new_FOS = copyFOS( linkage_model[0][0] );
          }
          else
          {
            new_FOS = (FOS*) Malloc(sizeof(FOS));
            new_FOS->length             = (int) ((number_of_parameters + FOS_element_size - 1) / FOS_element_size);
            new_FOS->sets       = (int **) Malloc( new_FOS->length*sizeof( int * ) );
            new_FOS->set_length = (int *) Malloc( new_FOS->length*sizeof( int ) );
            for( i = 0; i < new_FOS->length; i++ )
            {
              new_FOS->sets[i] = (int *) Malloc( FOS_element_size*sizeof( int ) );
              new_FOS->set_length[i] = 0;
            }

            for( i = 0; i < number_of_parameters; i++ )
            {
              new_FOS->sets[i/FOS_element_size][i%FOS_element_size] = i;
              new_FOS->set_length[i/FOS_element_size]++;
            }
          }
      }
      linkage_model[population_index][cluster_index] = new_FOS;
  }

  FOS *learnLinkageTreeRVGOMEA( int population_index, int cluster_index )
  {
      int i;
      FOS *new_FOS;

      new_FOS = learnLinkageTree( full_covariance_matrix[population_index][cluster_index] );
      if( learn_linkage_tree && number_of_generations[population_index] > 0 )
          inheritDistributionMultipliers( new_FOS, linkage_model[population_index][cluster_index], distribution_multipliers[population_index][cluster_index] );

      if( learn_linkage_tree && number_of_generations[population_index] > 0 )
      {
          for( i = 0; i < linkage_model[population_index][cluster_index]->length; i++ )
              free( linkage_model[population_index][cluster_index]->sets[i] );
          free( linkage_model[population_index][cluster_index]->sets );
          free( linkage_model[population_index][cluster_index]->set_length );
          free( linkage_model[population_index][cluster_index]);
      }
      return( new_FOS );
  }

  void inheritDistributionMultipliers( FOS *new_FOS, FOS *prev_FOS, double *multipliers )
  {
      int      i, *permutation;
      double   *multipliers_copy;

      multipliers_copy = (double*) Malloc(new_FOS->length*sizeof(double));
      for( i = 0; i < new_FOS->length; i++ )
          multipliers_copy[i] = multipliers[i];

      permutation = matchFOSElements( new_FOS, prev_FOS );

      for( i = 0; i < new_FOS->length; i++ )
          multipliers[permutation[i]] = multipliers_copy[i];

      free( multipliers_copy );
      free( permutation );
  }

  void estimateFullCovarianceMatrixML( int population_index, int cluster_index )
  {
      int i, j, k, q;
      double cov;

      i = cluster_index;
      full_covariance_matrix[population_index][i] = (double **) Malloc( number_of_parameters*sizeof( double * ) );
      for( k = 0; k < number_of_parameters; k++ )
        full_covariance_matrix[population_index][i][k] = (double *) Malloc( number_of_parameters*sizeof( double ) );

      /* Covariance matrices */
      for( j = 0; j < number_of_parameters; j++ )
      {
        for( q = j; q < number_of_parameters; q++ )
        {
          cov = 0.0;
          for( k = 0; k < cluster_sizes[population_index]; k++ )
            cov += (selection[population_index][selection_indices_of_cluster_members[population_index][i][k]]->parameters[j]-mean_vectors[population_index][i][j])
                  *(selection[population_index][selection_indices_of_cluster_members[population_index][i][k]]->parameters[q]-mean_vectors[population_index][i][q]);
          cov /= (double) cluster_sizes[population_index];

          full_covariance_matrix[population_index][i][j][q] = cov;
          full_covariance_matrix[population_index][i][q][j] = cov;
        }
      }
  }

  void evaluateCompletePopulation( int population_index )
  {
      int i;
      for( i = 0; i < population_sizes[population_index]; i++ )
          installedProblemEvaluation( populations[population_index][i], number_of_parameters, NULL, NULL, 0, 0 );
  }

  /**
   * Applies the distribution multipliers.
   */
  void applyDistributionMultipliers( int population_index )
  {
      int i, j, k, m;

      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
          for( j = 0; j < linkage_model[population_index][i]->length; j++ )
              for( k = 0; k < linkage_model[population_index][i]->set_length[j]; k++ )
                  for( m = 0; m < linkage_model[population_index][i]->set_length[j]; m++ )
                      decomposed_covariance_matrices[population_index][i][j][k][m] *= distribution_multipliers[population_index][i][j];
  }


  /**
   * Generates new solutions by sampling the mixture distribution.
   */
  void generateAndEvaluateNewSolutionsToFillPopulationAndUpdateElitistArchive( int population_index )
  {
    short   cluster_failure, all_multipliers_leq_one, *generational_improvement, any_improvement, *is_improved_by_AMS;
    int     i, j, k, m, oj, c, *order;

    if( !black_box_evaluations && (number_of_generations[population_index]+1) % 50 == 0 )
      evaluateCompletePopulation( population_index );

    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      computeParametersForSampling( population_index, i );

    generational_improvement = (short *) Malloc( population_sizes[population_index]*sizeof( short ) );

    for( i = 0; i < population_sizes[population_index]; i++ )
        generational_improvement[i] = 0;

    for( k = 0; k < number_of_mixing_components[population_index]; k++ )
    {
        order = randomPermutation(linkage_model[population_index][k]->length);
        for( m = 0; m < linkage_model[population_index][k]->length; m++ )
        {
            samples_current_cluster = 0;
            oj = order[m];
            samples_drawn_from_normal[population_index][k][oj] = 0;
            out_of_bounds_draws[population_index][k][oj]       = 0;

            for( i = 0; i < population_sizes[population_index]; i++ )
            {
                if( cluster_index_for_population[population_index][i] != k )
                    continue;
                if( generateNewSolutionFromFOSElement( population_index, k, oj, i ) )
                    generational_improvement[i] = 1;
                samples_current_cluster++;
            }

            adaptDistributionMultipliers( population_index, k, oj );
            for( i = 0; i < population_sizes[population_index]; i++ )
              if( cluster_index_for_population[population_index][i] == k && generational_improvement[i] )
                  updateElitistArchive( populations[population_index][i] );
        }
        free( order );

        c = 0;
        if( number_of_generations[population_index] > 0 )
        {
            is_improved_by_AMS = (short*)Malloc( population_sizes[population_index]*sizeof(short));
            for( i = 0; i < population_sizes[population_index]; i++ )
              is_improved_by_AMS[i] = 0;
            for( i = 0; i < population_sizes[population_index]; i++ )
            {
              if( cluster_index_for_population[population_index][i] != k )
                continue;
              is_improved_by_AMS[i] = applyAMS(population_index, i, k);
              generational_improvement[i] |= is_improved_by_AMS[i];

              c++;
              if( c >= 0.5*tau*num_individuals_in_cluster[population_index][k] )
                break;
            }
            c = 0;
            for( i = 0; i < population_sizes[population_index]; i++ )
            {
              if( cluster_index_for_population[population_index][i] != k )
                continue;
              if( is_improved_by_AMS[i] )
                updateElitistArchive( populations[population_index][i] );
              c++;
              if( c >= 0.5*tau*num_individuals_in_cluster[population_index][k] )
                break;
            }
            free( is_improved_by_AMS );
        }
    }

    for( i = 0; i < population_sizes[population_index]; i++ )
    {
        if( generational_improvement[i] )
          populations[population_index][i]->NIS = 0;
        else
          populations[population_index][i]->NIS++;
    }

    // Forced Improvements
    if( use_forced_improvement )
    {
        for( i = 0; i < population_sizes[population_index]; i++ )
        {
            if( populations[population_index][i]->NIS > maximum_no_improvement_stretch )
                applyForcedImprovements(population_index, i, &(generational_improvement[i]));
        }
    }

    cluster_failure = 1;
    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        for( j = 0; j < linkage_model[population_index][i]->length; j++ )
            if( distribution_multipliers[population_index][i][j] > 1.0 )
                { cluster_failure = 0; break;}

    if( cluster_failure )
        no_improvement_stretch[population_index]++;

    any_improvement = 0;
    for( i = 0; i < population_sizes[population_index]; i++ )
    {
        if( generational_improvement[i] )
        {
            any_improvement = 1;
            break;
        }
    }

    if( any_improvement )
        no_improvement_stretch[population_index] = 0;
    else
    {
        all_multipliers_leq_one = 1;
        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
          for( m = 0; m < linkage_model[population_index][i]->length; m++ )
            if( distribution_multipliers[population_index][i][m] > 1.0 )
              {all_multipliers_leq_one = 0; break;}

        if( all_multipliers_leq_one )
            no_improvement_stretch[population_index]++;
    }

    free( generational_improvement );
  }

  short applyAMS( int population_index, int individual_index, int cluster_index )
  {
      short out_of_range, improvement;
      double shrink_factor, delta_AMS, *solution_backup;
      int m;

      individual *ind_backup;
      ind_backup = initializeIndividual();

      delta_AMS     = 2.0;
      out_of_range  = 1;
      shrink_factor = 2;
      improvement   = 0;
      solution_backup = (double *) Malloc( number_of_parameters*sizeof( double ) );

      copyIndividual(populations[population_index][individual_index], ind_backup);
      while( (out_of_range == 1) && (shrink_factor > 1e-10) )
      {
          shrink_factor *= 0.5;
          out_of_range   = 0;
          for( m = 0; m < number_of_parameters; m++ )
          {
              populations[population_index][individual_index]->parameters[m] += shrink_factor*delta_AMS*(mean_vectors[population_index][cluster_index][m]-mean_vectors_previous[population_index][cluster_index][m]);
              // CHEAT
              if( use_boundary_repair )
              {
                  if( populations[population_index][individual_index]->parameters[m] < lower_range_bounds[m] )
                    populations[population_index][individual_index]->parameters[m] = lower_range_bounds[m];
                  else if( populations[population_index][individual_index]->parameters[m] > upper_range_bounds[m] )
                    populations[population_index][individual_index]->parameters[m] = upper_range_bounds[m];
              }
              // END-CHEAT
              if( !isParameterInRangeBounds( populations[population_index][individual_index]->parameters[m], m ) )
              {
                  out_of_range = 1;
                  break;
              }
          }
      }
      if( !out_of_range )
      {
          installedProblemEvaluation( populations[population_index][individual_index], number_of_parameters, NULL, NULL, 0, 0 );
          if( solutionWasImprovedByFOSElement( population_index, cluster_index, -1, individual_index ) || constraintParetoDominates( populations[population_index][individual_index]->objective_values, populations[population_index][individual_index]->constraint_value, ind_backup->objective_values, ind_backup->constraint_value ) )
              improvement = 1;
      }
      if( out_of_range || !improvement )
      {
          copyIndividual( ind_backup, populations[population_index][individual_index]);
      }
      free( solution_backup );
      ezilaitiniIndividual( ind_backup );
      return( improvement );
  }

  void applyForcedImprovements( int population_index, int individual_index, short *improved )
  {
      int i, j, k, m, cluster_index, donor_index, objective_index, *order, num_indices, *indices;
      double distance, distance_smallest, *objective_values_scaled, alpha, *FI_backup;
      individual *ind_backup;

      i = individual_index;
      populations[population_index][i]->NIS = 0;
      cluster_index = cluster_index_for_population[population_index][i];
      donor_index = 0;
      ind_backup = initializeIndividual();

      objective_values_scaled = (double *) Malloc( number_of_objectives*sizeof( double ) );
      for( j = 0; j < number_of_objectives; j++ )
          objective_values_scaled[j] = populations[population_index][i]->objective_values[j]/objective_ranges[population_index][j];
      distance_smallest = 1e308;
      for( j = 0; j < elitist_archive_size; j++ )
      {
          if( elitist_archive_indices_inactive[j] )
              continue;
          for( k = 0; k < number_of_objectives; k++ )
              objective_values_scaled[k] = elitist_archive[j]->objective_values[k]/objective_ranges[population_index][k];
          distance = distanceEuclidean( objective_values_scaled, objective_means_scaled[population_index][cluster_index], number_of_objectives );
          if( distance < distance_smallest )
          {
              donor_index = j;
              distance_smallest = distance;
          }
      }

      alpha = 0.5;
      while( alpha >= 0.05 )
      {
        order = randomPermutation( linkage_model[population_index][cluster_index]->length );
        for( m = 0; m < linkage_model[population_index][cluster_index]->length; m++ )
        {
          num_indices = linkage_model[population_index][cluster_index]->set_length[order[m]];
          indices = linkage_model[population_index][cluster_index]->sets[order[m]];

          FI_backup = (double*) Malloc( num_indices*sizeof( double ) );

          copyIndividualWithoutParameters(populations[population_index][i], ind_backup );

          for( j = 0; j < num_indices; j++ )
          {
              FI_backup[j] = populations[population_index][i]->parameters[indices[j]];
              populations[population_index][i]->parameters[indices[j]] = alpha*populations[population_index][i]->parameters[indices[j]] + (1.0-alpha)*elitist_archive[donor_index]->parameters[indices[j]];
          }
          installedProblemEvaluation( populations[population_index][i], num_indices, indices, FI_backup, populations[population_index][i]->objective_values, populations[population_index][i]->constraint_value );

          if( single_objective_clusters[population_index][cluster_index] != -1 )
          {
              objective_index = single_objective_clusters[population_index][cluster_index];
              if( populations[population_index][i]->objective_values[objective_index] < ind_backup->objective_values[objective_index] )
                  *improved = 1;
          }
          else if( constraintParetoDominates( populations[population_index][i]->objective_values, populations[population_index][i]->constraint_value, ind_backup->objective_values, ind_backup->constraint_value ) )
              *improved = 1;

          if( !(*improved) )
          {
              for( j = 0; j < num_indices; j++ )
                  populations[population_index][i]->parameters[indices[j]] = FI_backup[j];
              copyIndividualWithoutParameters( ind_backup, populations[population_index][i]);
              free( FI_backup );
          }
          else{
              free( FI_backup );
              break;
          }
        }
        alpha *= 0.5;

        free( order );
        if( *improved )
          break;
      }
      if( !(*improved) )
      {
          copyIndividual(elitist_archive[donor_index], populations[population_index][i] );
      }
      updateElitistArchive( populations[population_index][i] );
      ezilaitiniIndividual( ind_backup );

      free( objective_values_scaled );
  }

  /**
   * Computes the Cholesky-factor matrices required for sampling
   * the multivariate normal distributions in the mixture distribution.
   */
  void computeParametersForSampling( int population_index, int cluster_index )
  {
    int i;

    if( !use_univariate_FOS )
    {
        decomposed_cholesky_factors_lower_triangle[population_index][cluster_index] = (double ***) Malloc(linkage_model[population_index][cluster_index]->length * sizeof(double**));
        for( i = 0; i < linkage_model[population_index][cluster_index]->length; i++ )
        {
          bool success = false;
          if(!enable_regularization[population_index][cluster_index][i]) {
            decomposed_cholesky_factors_lower_triangle[population_index][cluster_index][i] = choleskyDecomposition( decomposed_covariance_matrices[population_index][cluster_index][i], linkage_model[population_index][cluster_index]->set_length[i], success );
          }
          
          if(!success || enable_regularization[population_index][cluster_index][i])
          {
            // i hate mallocs
            if(!enable_regularization[population_index][cluster_index][i]) {
              enable_regularization[population_index][cluster_index][i] = true;
              for(size_t j = 0; j < linkage_model[population_index][cluster_index]->set_length[i]; j++ ) {
                free( decomposed_cholesky_factors_lower_triangle[population_index][cluster_index][i][j] );
              }
              free( decomposed_cholesky_factors_lower_triangle[population_index][cluster_index][i] );
            }
            // std::cout << "Regularization. ";
            
            // regularizeCovarianceMatrix(decomposed_covariance_matrices[population_index][cluster_index][i], selection.sols, mean_vector, linkage_model->sets[FOS_index], linkage_model->set_length[FOS_index]);
            decomposed_cholesky_factors_lower_triangle[population_index][cluster_index][i] = choleskyDecomposition( decomposed_covariance_matrices[population_index][cluster_index][i], linkage_model[population_index][cluster_index]->set_length[i] );
            // std::cout << "Cholesky " << (success ? "complete\n" : "failed\n");
          }
          
        }
    }
  }
  
  //  regularizeCovarianceMatrix(partial_covariance_matrices[FOS_index], selection.sols, mean_vector, linkage_model->sets[FOS_index], linkage_model->set_length[FOS_index]);
  // input: n x n covariance matrix
  //        sols from which the covariance matrix was estimated
  //        mean of size sol[0]->Param.size()
  //        list of parameters of size n // i.e.,
  //        n
  
  // selection[population_index][selection_indices_of_cluster_members[population_index][cluster_index][k]]->parameters[vara]-mean_vectors[population_index][cluster_index][vara]
  
  
  bool regularizeCovarianceMatrix(int population_index, int cluster_index, int FOS_index)
  {
    // regularization for small populations
    double number_of_samples = (double) cluster_sizes[population_index];
    size_t n = (size_t) linkage_model[population_index][cluster_index]->set_length[FOS_index];
    
    // either use the univariate matrix as a prior,
    // or a diagonal matrix with the mean variance on all diagonal entries
    bool use_univariate_as_prior = true;
    
    double meanvar = 0.0;
    if(!use_univariate_as_prior)
    {
      for(size_t i = 0; i < n; ++i) {
        meanvar += decomposed_covariance_matrices[population_index][cluster_index][FOS_index][i][i];
      }
      meanvar /= (double) n;
    }
    double phi = 0.0;
    
    // y = x.^2
    // phiMat = y'*y/t-sample.^2
    // phi = sum(sum(phiMat))
    double squared_cov = 0.0;
    
    // matrix_t squared_cov(n,n,0.0);
    double temp;
    int vara, varb;
    for(size_t i = 0; i < n; ++i)
    {
      vara = linkage_model[population_index][cluster_index]->sets[FOS_index][i];
      
      for(size_t j = 0; j < n; ++j)
      {
        varb = linkage_model[population_index][cluster_index]->sets[FOS_index][j];
        
        squared_cov = 0.0;
        
        for(size_t k = 0; k < cluster_sizes[population_index]; ++k)
        {
          temp = (selection[population_index][selection_indices_of_cluster_members[population_index][cluster_index][k]]->parameters[vara]-mean_vectors[population_index][cluster_index][vara])
                  *(selection[population_index][selection_indices_of_cluster_members[population_index][cluster_index][k]]->parameters[varb]-mean_vectors[population_index][cluster_index][varb]);
          squared_cov += (temp - decomposed_covariance_matrices[population_index][cluster_index][FOS_index][i][j]) * (temp - decomposed_covariance_matrices[population_index][cluster_index][FOS_index][i][j]);
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
        if(use_univariate_as_prior) {
          if(i != j) {
            temp = fabs(decomposed_covariance_matrices[population_index][cluster_index][FOS_index][i][j]);
          } else {
            temp = 0.0;
          }
        } else {
          temp = fabs(decomposed_covariance_matrices[population_index][cluster_index][FOS_index][i][j] - (( i == j ) ? meanvar : 0.0));
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
        if(use_univariate_as_prior) {
          if (i != j) { // i == j remains the same, only off-diagonals are shrunken
            decomposed_covariance_matrices[population_index][cluster_index][FOS_index][i][j] = (1.0 - shrinkage) * decomposed_covariance_matrices[population_index][cluster_index][FOS_index][i][j];
          }
        } else {
          decomposed_covariance_matrices[population_index][cluster_index][FOS_index][i][j] = (1.0 - shrinkage) * decomposed_covariance_matrices[population_index][cluster_index][FOS_index][i][j] + ((i==j) ? shrinkage*meanvar : 0.0);
        }
      }
    }
    
    return true;
  }

  /**
   * Generates and returns a single new solution by drawing
   * a sample for the variables in the selected FOS elementmax_clus
   * and inserting this into the population.
   */
  double *generateNewPartialSolutionFromFOSElement( int population_index, int cluster_index, int FOS_index )
  {
      short   ready;
      int     i, times_not_in_bounds, num_indices, *indices;
      double *result, *z;

      num_indices = linkage_model[population_index][cluster_index]->set_length[FOS_index];
      indices = linkage_model[population_index][cluster_index]->sets[FOS_index];
      times_not_in_bounds = -1;
      out_of_bounds_draws[population_index][cluster_index][FOS_index]--;

      ready = 0;
      do
      {
          times_not_in_bounds++;
          samples_drawn_from_normal[population_index][cluster_index][FOS_index]++;
          out_of_bounds_draws[population_index][cluster_index][FOS_index]++;
          if( times_not_in_bounds >= 100 )
          {
              result = (double *) Malloc( num_indices*sizeof( double ) );
              for( i = 0; i < num_indices; i++ )
                  result[i] = lower_init_ranges[indices[i]] + (upper_init_ranges[indices[i]] - lower_init_ranges[indices[i]])*randomRealUniform01();
          }
          else
          {
              z = (double *) Malloc( num_indices*sizeof( double ) );

              for( i = 0; i < num_indices; i++ )
                  z[i] = random1DNormalUnit();

              if( use_univariate_FOS )
              {
                  result = (double*) Malloc(1*sizeof(double));
                  result[0] = z[0]*sqrt(decomposed_covariance_matrices[population_index][cluster_index][FOS_index][0][0]) + mean_vectors[population_index][cluster_index][indices[0]];
              }
              else
              {
                  result = matrixVectorMultiplication( decomposed_cholesky_factors_lower_triangle[population_index][cluster_index][FOS_index], z, num_indices, num_indices );

                  for( i = 0; i < num_indices; i++ )
                      result[i] += mean_vectors[population_index][cluster_index][indices[i]];
              }

              free( z );
          }

          ready = 1;
          for( i = 0; i < num_indices; i++ )
          {
              // CHEAT
              if( use_boundary_repair )
              {
                  if( result[i] < lower_range_bounds[indices[i]] )
                    result[i] = lower_range_bounds[indices[i]];
                  else if( result[i] > upper_range_bounds[indices[i]] )
                    result[i] = upper_range_bounds[indices[i]];
              }
              // END-CHEAT

              if( !isParameterInRangeBounds( result[i], indices[i] ) )
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
  short generateNewSolutionFromFOSElement( int population_index, int cluster_index, int FOS_index, int individual_index )
  {
      int j, m, *indices, num_indices, *touched_indices, num_touched_indices, out_of_range;
      double *result, *solution_AMS, *individual_backup, shrink_factor;
      short improvement;
      individual *ind_backup;
      ind_backup = initializeIndividual();

      num_indices = linkage_model[population_index][cluster_index]->set_length[FOS_index];
      indices = linkage_model[population_index][cluster_index]->sets[FOS_index];
      num_touched_indices = num_indices;
      touched_indices = indices;
      improvement = 0;

      solution_AMS = (double *) Malloc( num_indices*sizeof( double ) );
      individual_backup = (double*) Malloc( num_touched_indices * sizeof( double ) );

      result = generateNewPartialSolutionFromFOSElement( population_index, cluster_index, FOS_index );

      for( j = 0; j < num_touched_indices; j++ )
          individual_backup[j] = populations[population_index][individual_index]->parameters[touched_indices[j]];
      for( j = 0; j < num_indices; j++ )
          populations[population_index][individual_index]->parameters[indices[j]] = result[j];

      copyIndividualWithoutParameters(populations[population_index][individual_index], ind_backup );

      if( (number_of_generations[population_index] > 0) && (samples_current_cluster <= 0.5*tau*num_individuals_in_cluster[population_index][cluster_index]) )
      {
        out_of_range  = 1;
        shrink_factor = 2;
        while( (out_of_range == 1) && (shrink_factor > 1e-10) )
        {
          shrink_factor *= 0.5;
          out_of_range = 0;
          for( m = 0; m < num_indices; m++ )
          {
            j = indices[m];
            solution_AMS[m] = result[m] + shrink_factor*delta_AMS*distribution_multipliers[population_index][cluster_index][FOS_index]*(mean_vectors[population_index][cluster_index][j]-mean_vectors_previous[population_index][cluster_index][j]);
            // CHEAT
            if( use_boundary_repair )
            {
                if( solution_AMS[m] < lower_range_bounds[indices[m]] )
                  solution_AMS[m] = lower_range_bounds[indices[m]];
                else if ( solution_AMS[m] > upper_range_bounds[indices[m]] )
                  solution_AMS[m] = upper_range_bounds[indices[m]];
            }
            // END-CHEAT
            if( !isParameterInRangeBounds( solution_AMS[m], j ) )
            {
              out_of_range = 1;
              break;
            }
          }
        }
        if( !out_of_range )
        {
          for( j = 0; j < num_indices; j++ )
            populations[population_index][individual_index]->parameters[indices[j]] = solution_AMS[j];
        }
      }
      installedProblemEvaluation( populations[population_index][individual_index], num_touched_indices, touched_indices, individual_backup, ind_backup->objective_values, ind_backup->constraint_value );

      if( solutionWasImprovedByFOSElement( population_index, cluster_index, FOS_index, individual_index ) || constraintParetoDominates( populations[population_index][individual_index]->objective_values, populations[population_index][individual_index]->constraint_value, ind_backup->objective_values, ind_backup->constraint_value ) )
      {
          improvement = 1;
      }

      if( !improvement ){
          for( j = 0; j < num_touched_indices; j++ )
              populations[population_index][individual_index]->parameters[touched_indices[j]] = individual_backup[j];
          copyIndividualWithoutParameters( ind_backup,populations[population_index][individual_index]);
      }

      free( solution_AMS );
      free( individual_backup );
      free( result );

      ezilaitiniIndividual( ind_backup );
      return( improvement );
  }

  /**
   * Adapts the distribution multipliers according to
   * the SDR-AVS mechanism.
   */
  void adaptDistributionMultipliers( int population_index, int cluster_index, int FOS_index )
  {
    short  improvementForFOSElement;
    double st_dev_ratio;

    if( (((double) out_of_bounds_draws[population_index][cluster_index][FOS_index])/((double) samples_drawn_from_normal[population_index][cluster_index][FOS_index])) > 0.9 )
    {
        distribution_multipliers[population_index][cluster_index][FOS_index] *= 0.5;
    }

    improvementForFOSElement = generationalImprovementForOneClusterForFOSElement( population_index, cluster_index, FOS_index, &st_dev_ratio );

    if( improvementForFOSElement )
    {
      no_improvement_stretch[population_index] = 0;

      if( distribution_multipliers[population_index][cluster_index][FOS_index] < 1.0 ) distribution_multipliers[population_index][cluster_index][FOS_index] = 1.0;

      if( st_dev_ratio > st_dev_ratio_threshold )
          distribution_multipliers[population_index][cluster_index][FOS_index] *= distribution_multiplier_increase;
    }
    else
    {
      if( (distribution_multipliers[population_index][cluster_index][FOS_index] > 1.0) || (no_improvement_stretch[population_index] >= maximum_no_improvement_stretch) )
          distribution_multipliers[population_index][cluster_index][FOS_index] *= distribution_multiplier_decrease;

      if( no_improvement_stretch[population_index] < maximum_no_improvement_stretch )
      {
          if(distribution_multipliers[population_index][cluster_index][FOS_index] < 1.0)
              distribution_multipliers[population_index][cluster_index][FOS_index] = 1.0;
      }
    }
  }

  /**
   * Determines whether an improvement is found for a specified
   * population. Returns 1 in case of an improvement, 0 otherwise.
   * The standard-deviation ratio required by the SDR-AVS
   * mechanism is computed and returned in the pointer variable.
   */
  short generationalImprovementForOneClusterForFOSElement( int population_index, int cluster_index, int FOS_index, double *st_dev_ratio )
  {
      int     i, number_of_improvements;

      number_of_improvements = 0;

      /* Determine st.dev. ratio */
      *st_dev_ratio = 0.0;
      for( i = 0; i < population_sizes[population_index]; i++ )
      {
        if( cluster_index_for_population[population_index][i] == cluster_index )
        {
          if( solutionWasImprovedByFOSElement( population_index, cluster_index, FOS_index, i ) )
          {
              number_of_improvements++;
              (*st_dev_ratio) += getStDevRatioForOneClusterForFOSElement( population_index, cluster_index, FOS_index, populations[population_index][i]->parameters );
          }
        }
      }

      if( number_of_improvements > 0 )
        (*st_dev_ratio) = (*st_dev_ratio) / number_of_improvements;

      if( number_of_improvements > 0 )
        return( 1 );

      return( 0 );
  }

  /**
   * Computes and returns the standard-deviation-ratio
   * of a given point for a given model.
   */
  double getStDevRatioForOneClusterForFOSElement( int population_index, int cluster_index, int FOS_index, double *parameters )
  {
    int      i, *indices, num_indices;
    double **inverse, result, *x_min_mu, *z;

    result = 0.0;
    indices = linkage_model[population_index][cluster_index]->sets[FOS_index];
    num_indices = linkage_model[population_index][cluster_index]->set_length[FOS_index];

    x_min_mu = (double *) Malloc( num_indices*sizeof( double ) );

    for( i = 0; i < num_indices; i++ )
      x_min_mu[i] = parameters[indices[i]]-mean_vectors[population_index][cluster_index][indices[i]];

    if( use_univariate_FOS )
    {
      result = fabs( x_min_mu[0]/sqrt(decomposed_covariance_matrices[population_index][cluster_index][FOS_index][0][0]) );
    }
    else
    {
      inverse = matrixLowerTriangularInverse( decomposed_cholesky_factors_lower_triangle[population_index][cluster_index][FOS_index], num_indices );
      z = matrixVectorMultiplication( inverse, x_min_mu, num_indices, num_indices );

      for( i = 0; i < num_indices; i++ )
      {
        if( fabs( z[i] ) > result )
          result = fabs( z[i] );
      }

      free( z );
      for( i = 0; i < num_indices; i++ )
        free( inverse[i] );
      free( inverse );
    }

    free( x_min_mu );

    return( result );
  }

  /**
   * Returns whether a solution has the
   * hallmark of an improvement (1 for yes, 0 for no).
   */
  short solutionWasImprovedByFOSElement( int population_index, int cluster_index, int FOS_index, int individual_index )
  {
    short result, in_range;
    int   i, j;

    result = 0;
    in_range = 1;
    if( FOS_index == -1 )
    {
      for( i = 0; i < number_of_parameters; i++ )
          if(!isParameterInRangeBounds( populations[population_index][individual_index]->parameters[i], i ))
              return( result );
    }
    else in_range = isSolutionInRangeBoundsForFOSElement( populations[population_index][individual_index]->parameters, population_index, cluster_index, FOS_index );
    if( in_range )
    {
      if( populations[population_index][individual_index]->constraint_value == 0 )
      {
        for( j = 0; j < number_of_objectives; j++ )
        {
          if( populations[population_index][individual_index]->objective_values[j] < best_objective_values_in_elitist_archive[j] )
          {
            result = 1;
            break;
          }
        }
      }

      if( single_objective_clusters[population_index][cluster_index] != -1 )
      {
          return( result );
      }

      if( result != 1 )
      {
        result = 1;
        for( i = 0; i < elitist_archive_size; i++ )
        {
          if( elitist_archive_indices_inactive[i] ) continue;
          if( constraintParetoDominates( elitist_archive[i]->objective_values, elitist_archive[i]->constraint_value, populations[population_index][individual_index]->objective_values, populations[population_index][individual_index]->constraint_value ) )
          {
            result = 0;
            break;
          }
          else if( !constraintParetoDominates( populations[population_index][individual_index]->objective_values, populations[population_index][individual_index]->constraint_value, elitist_archive[i]->objective_values, elitist_archive[i]->constraint_value ) )
          {
            if( sameObjectiveBox( elitist_archive[i]->objective_values, populations[population_index][individual_index]->objective_values ) )
            {
              result = 0;
              break;
            }
          }
        }
      }
    }

    return( result );
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





  /*-=-=-=-=-=-=-=-=-=-=-=-=- Section Ezilaitini -=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
  /**
   * Undoes initialization procedure by freeing up memory.
   */
  void ezilaitini( void )
  {
    int i;

    ezilaitiniObjectiveRotationMatrix();

    for( i = 0; i < number_of_populations; i++ )
    {
        ezilaitiniDistributionMultipliers( i );

        ezilaitiniMemoryOnePopulation( i );
    }

    ezilaitiniMemory();

    ezilaitiniProblem();
  }

  void ezilaitiniMemory( void )
  {
    int i; // default_front_size;
      // double **default_front;

      for( i = 0; i < elitist_archive_capacity; i++ )
          ezilaitiniIndividual( elitist_archive[i] );
      free( elitist_archive );

      free( full_covariance_matrix );
      free( population_sizes );
      free( selection_sizes );
      free( cluster_sizes );
      free( populations );
      free( ranks );
      free( sorted_ranks );
      free( objective_ranges );
      free( selection );
      free( objective_values_selection_previous );
      free( ranks_selection );
      free( number_of_mixing_components );
      free( decomposed_covariance_matrices );
      free( distribution_multipliers );
      free( enable_regularization );
      free( decomposed_cholesky_factors_lower_triangle );
      free( mean_vectors );
      free( mean_vectors_previous );
      free( objective_means_scaled );
      free( selection_indices );
      free( selection_indices_of_cluster_members );
      free( selection_indices_of_cluster_members_previous );
      free( pop_indices_selected );
      free( samples_drawn_from_normal );
      free( out_of_bounds_draws );
      free( single_objective_clusters );
      free( cluster_index_for_population );
      free( num_individuals_in_cluster );
      free( number_of_generations );
      free( populations_terminated );
      free( no_improvement_stretch );

      free( lower_range_bounds );
      free( upper_range_bounds );
      free( lower_init_ranges );
      free( upper_init_ranges );

      free( number_of_elitist_solutions_copied );
      free( best_objective_values_in_elitist_archive );
      free( elitist_archive_indices_inactive );
      free( objective_discretization );

      free( linkage_model );
  }

  /**
   * Undoes initialization procedure by freeing up memory.
   */
  void ezilaitiniMemoryOnePopulation( int population_index )
  {
    int      i, j;

    for( i = 0; i < population_sizes[population_index]; i++ )
    {
      ezilaitiniIndividual( populations[population_index][i] );
    }
    free( populations[population_index] );

    for( i = 0; i < selection_sizes[population_index]; i++ )
    {
      ezilaitiniIndividual( selection[population_index][i] );
      free( objective_values_selection_previous[population_index][i] );
    }
    free( selection[population_index] );
    free( objective_values_selection_previous[population_index] );

    if( !learn_linkage_tree )
    {
        ezilaitiniCovarianceMatrices(population_index);
    }

    for( i = 0; i < number_of_mixing_components[population_index]; i++ )
    {
      free( mean_vectors[population_index][i] );
      free( mean_vectors_previous[population_index][i] );
      free( objective_means_scaled[population_index][i] );

      if( selection_indices_of_cluster_members[population_index][i] != NULL )
        free( selection_indices_of_cluster_members[population_index][i] );
      if( selection_indices_of_cluster_members_previous[population_index][i] != NULL )
        free( selection_indices_of_cluster_members_previous[population_index][i] );

      if( samples_drawn_from_normal[population_index] != NULL)
      {
        free( samples_drawn_from_normal[population_index][i] );
        free( out_of_bounds_draws[population_index][i] );
      }

      if( linkage_model[population_index][i] != NULL )
      {
        for( j = 0; j < linkage_model[population_index][i]->length; j++ )
          free( linkage_model[population_index][i]->sets[j] );
        free( linkage_model[population_index][i]->sets );
        free( linkage_model[population_index][i]->set_length );
        free( linkage_model[population_index][i] );
      }
    }

    if( learn_linkage_tree )
    {
        free( full_covariance_matrix[population_index] );
    }

    free( linkage_model[population_index] );
    free( ranks[population_index] );
    free( sorted_ranks[population_index] );
    free( objective_ranges[population_index] );
    free( ranks_selection[population_index] );
    free( mean_vectors[population_index] );
    free( mean_vectors_previous[population_index] );
    free( objective_means_scaled[population_index] );
    free( selection_indices[population_index] );
    free( selection_indices_of_cluster_members[population_index] );
    free( selection_indices_of_cluster_members_previous[population_index] );
    free( pop_indices_selected[population_index] );
    free( decomposed_cholesky_factors_lower_triangle[population_index] );
    free( samples_drawn_from_normal[population_index] );
    free( out_of_bounds_draws[population_index] );
    free( single_objective_clusters[population_index] );
    free( cluster_index_for_population[population_index] );
    free( num_individuals_in_cluster[population_index] );
    free( number_of_elitist_solutions_copied[population_index] );
  }

  /**
   * Undoes initialization procedure by freeing up memory.
   */
  void ezilaitiniDistributionMultipliers( int population_index )
  {
    int i;
    if( distribution_multipliers[population_index] != NULL )
    {
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      {
          free( distribution_multipliers[population_index][i] );
      }
      free( distribution_multipliers[population_index] );
    }
    
    
    if( enable_regularization[population_index] != NULL )
    {
      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      {
        free( enable_regularization[population_index][i] );
      }
      free( enable_regularization[population_index] );
    }
  }

  void ezilaitiniCovarianceMatrices( int population_index )
  {
      int i,j,k;

      for( i = 0; i < number_of_mixing_components[population_index]; i++ )
      {
        for( j = 0; j < linkage_model[population_index][i]->length; j++ )
        {
            for( k = 0; k < linkage_model[population_index][i]->set_length[j]; k++ )
              free( decomposed_covariance_matrices[population_index][i][j][k] );
            free( decomposed_covariance_matrices[population_index][i][j] );
        }
        free( decomposed_covariance_matrices[population_index][i] );
      }
     free( decomposed_covariance_matrices[population_index] );
  }

  /**
   * Frees memory of the Cholesky decompositions required for sampling.
   */
  void ezilaitiniParametersForSampling( int population_index )
  {
      int i, j, k;

      if( !use_univariate_FOS )
      {
          for( k = 0; k < number_of_mixing_components[population_index]; k++ )
          {
              for( i = 0; i < linkage_model[population_index][k]->length; i++ )
              {
                  for( j = 0; j < linkage_model[population_index][k]->set_length[i]; j++ )
                      free( decomposed_cholesky_factors_lower_triangle[population_index][k][i][j] );
                  free( decomposed_cholesky_factors_lower_triangle[population_index][k][i] );
              }
              free( decomposed_cholesky_factors_lower_triangle[population_index][k] );
          }
      }
      if( learn_linkage_tree )
      {
          ezilaitiniCovarianceMatrices(population_index);
      }
  }

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Run -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  void generationalStepAllPopulationsRecursiveFold( int population_index_smallest, int population_index_biggest );
  void generationalStepAllPopulations()
  {
    int population_index_smallest, population_index_biggest;

    population_index_biggest  = number_of_populations-1;
    population_index_smallest = 0;
    while( population_index_smallest <= population_index_biggest )
    {
      if( !populations_terminated[population_index_smallest] )
        break;

      population_index_smallest++;
    }

    generationalStepAllPopulationsRecursiveFold( population_index_smallest, population_index_biggest );
  }

  void generationalStepAllPopulationsRecursiveFold( int population_index_smallest, int population_index_biggest )
  {
    int i, j, population_index;

    for( i = 0; i < number_of_subgenerations_per_population_factor-1; i++ )
    {
      for( population_index = population_index_smallest; population_index <= population_index_biggest; population_index++ )
      {
        if( !populations_terminated[population_index] )
        {
            makeSelection( population_index );

            makePopulation( population_index );

            (number_of_generations[population_index])++;

            if( checkTerminationConditionOnePopulation( population_index ) )
            {
                for( j = 0; j < number_of_populations; j++ )
                  populations_terminated[j] = 1;
                return;
            }
        }
      }

      for( population_index = population_index_smallest; population_index < population_index_biggest; population_index++ )
        generationalStepAllPopulationsRecursiveFold( population_index_smallest, population_index );
    }
  }

  void runAllPopulations( void )
  {
      while( !checkTerminationConditionAllPopulations() )
      {
        if( number_of_populations < maximum_number_of_populations )
        {
          initializeNewPopulation();
        }

        computeApproximationSet();

        if( write_generational_statistics) {
		  if(learn_linkage_tree || use_univariate_FOS || total_number_of_generations < 50 || total_number_of_generations % 50 == 0)
			writeGenerationalStatisticsForOnePopulation( number_of_populations-1 );
		}
		
        if( write_generational_solutions )
          writeGenerationalSolutions( 0 );

        freeApproximationSet();

        generationalStepAllPopulations();

        total_number_of_generations++;
      }
  }

  /**
   * Runs the MIDEA.
   */
  void run( void )
  {
    initialize();

    if( print_verbose_overview )
      printVerboseOverview();

    runAllPopulations();

    computeApproximationSet();

    writeGenerationalStatisticsForOnePopulation( number_of_populations-1 );

    writeGenerationalSolutions( 1 );

  }

  /**
   * The main function:
   * - interpret parameters on the command line
   * - run the algorithm with the interpreted parameters
   */
  /* int mainGomea( int argc, char **argv )
  {
    initializeRandomNumberGenerator();

    interpretCommandLine( argc, argv );

    run();

    return( 0 );
  } */
  
  void initGomea(
                 hicam::fitness_pt fitness_function,
                 const hicam::vec_t & lower_init_ranges,
                 const hicam::vec_t & upper_init_ranges,
                 double vtr,
                 int use_vtr,
                 int version,
                 int local_optimizer_index,
                 double HL_tol,
                 size_t elitist_archive_size_target,
                 size_t approximation_set_size_target,
                 size_t maximum_number_of_populations,
                 int base_population_size,
                 int base_number_of_mixing_components,
                 unsigned int number_of_subgenerations_per_population_factor,
                 unsigned int maximum_number_of_evaluations,
                 unsigned int maximum_number_of_seconds,
                 int random_seed,
                 bool write_generational_solutions,
                 bool write_generational_statistics,
                 bool print_generational_statistics,
                 const std::string & write_directory,
                 const std::string & file_appendix,
                 bool print_verbose_overview)
  {
    
    // fixes the random_seed of gomea.
    // note that a different rng is used than within hicam
    if(random_seed > 0) {
      gomea::random_seed_changing = random_seed;
    } else {
      gomea::random_seed_changing = 0;
    }
    
    gomea::lower_user_range = lower_init_ranges[0];
    gomea::upper_user_range = upper_init_ranges[0];
    
    initializeRandomNumberGenerator();
    
    gomea::fitness_function = fitness_function;
    fitness_function->get_param_bounds(lowerRangeBound,upperRangeBound);
    
    // Set parameters
    //------------------------------------------------------------------------------------------
    startTimer(); // this should be in run?
    // begin parseCommandLine( argc, argv ); // TODO
    gomea::write_generational_statistics = write_generational_statistics;
    gomea::write_generational_solutions  = write_generational_solutions;
    gomea::print_verbose_overview        = print_verbose_overview;
    gomea::use_vtr                       = use_vtr;
    gomea::black_box_evaluations         = 1; // not implemented.
    gomea::use_boundary_repair = 0;
    gomea::use_forced_improvement = 1;
    gomea::static_linkage_tree = 0;
    gomea::random_linkage_tree = 0;
    //end parseCommandLine
    gomea::number_of_objectives = (int) fitness_function->number_of_objectives; // installedProblemNumberOfObjectives( problem_index );
    gomea::tau = 0.35;
    gomea::maximum_number_of_populations = (int) maximum_number_of_populations;
    gomea::base_population_size = base_population_size;
    gomea::base_number_of_mixing_components = 0;
    gomea::maximum_number_of_evaluations = maximum_number_of_evaluations;
    gomea::maximum_number_of_seconds = maximum_number_of_seconds;
    gomea::vtr = vtr;
    gomea::rotation_angle = 0.0;
    gomea::number_of_subgenerations_per_population_factor = number_of_subgenerations_per_population_factor;

    if( gomea::base_number_of_mixing_components <= 0) {
      gomea::base_number_of_mixing_components = 5; // default.
    }
  
    if( gomea::maximum_number_of_populations == 1 ) {
      if( gomea::base_number_of_mixing_components <= 0) {
        gomea::base_number_of_mixing_components = 5; // default.
      }
      
      if (gomea::base_population_size <= 0) {
        gomea::base_population_size  = (int) ((0.5*gomea::base_number_of_mixing_components)*(36.1 + 7.58*log2((double) number_of_parameters)));
      }
      else {
        gomea::base_population_size = gomea::base_population_size;
      }
    }
    else {
      gomea::base_number_of_mixing_components = 1 + gomea::number_of_objectives;
      gomea::base_population_size = 10 * gomea::base_number_of_mixing_components;
    }
    gomea::number_of_parameters = (int) fitness_function->number_of_parameters;
    gomea::distribution_multiplier_decrease = 0.9;
    gomea::st_dev_ratio_threshold           = 1.0;
    gomea::maximum_no_improvement_stretch   = (int) (2.0 + ((double) (25 + number_of_parameters))/((double) gomea::base_number_of_mixing_components));
    
    gomea::elitist_archive_size_target = (int) elitist_archive_size_target;
    gomea::approximation_set_size_target = (int) approximation_set_size_target;
    
    // dit moet ik nog checken.
    statistics_file_existed = 0;
    objective_discretization_in_effect = 0;
    block_size = number_of_parameters;
    block_start = 0;
    number_of_blocks = (number_of_parameters - 1 + block_size - 1) / block_size;
    
    FOS_element_ub = number_of_parameters;
    FOS_element_size = -1; // local_optimizer_index
    
    if( FOS_element_size == -1 ) FOS_element_size = number_of_parameters;
    if( FOS_element_size == -2 ) learn_linkage_tree = 1;
    if( FOS_element_size == -3 ) static_linkage_tree = 1;
    if( FOS_element_size == -4 ) {static_linkage_tree = 1; FOS_element_ub = 100;}
    if( FOS_element_size == -5 ) {random_linkage_tree = 1; static_linkage_tree = 1; FOS_element_ub = 100;}
    if( FOS_element_size == 1 ) use_univariate_FOS = 1;
    
    checkOptions();
    
    gomea::write_directory = write_directory;
    gomea::file_appendix = file_appendix;
    gomea::HL_tol = HL_tol;
  }
  
  
  void freeGOMEA()
  {
    freeApproximationSet();
    ezilaitini();
  }
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
}
