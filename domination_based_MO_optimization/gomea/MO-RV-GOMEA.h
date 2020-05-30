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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "Tools.h"
#include "FOS.h"
#include "MO_optimization.h"
#include "../mohillvallea/hicam_internal.h"
#include "../mohillvallea/fitness.h"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace gomea
{
  

  /*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
  extern double HL_tol;
  extern int           number_of_objectives,
  current_population_index,
  approximation_set_size;                        /* Number of solutions in the final answer (the approximation set). */
  extern double        sum_of_ellipsoids_normalization_factor;
  extern long          number_of_full_evaluations;
  extern short         approximation_set_reaches_vtr,
  statistics_file_existed;
  extern short         objective_discretization_in_effect,            /* Whether the objective space is currently being discretized for the elitist archive. */
  *elitist_archive_indices_inactive;              /* Elitist archive solutions flagged for removal. */
  extern int           elitist_archive_size, previous_elitist_archive_size,                          /* Number of solutions in the elitist archive. */
  elitist_archive_size_target,                   /* The lower bound of the targeted size of the elitist archive. */
  elitist_archive_capacity;                      /* Current memory allocation to elitist archive. */
  extern double       *best_objective_values_in_elitist_archive,      /* The best objective values in the archive in the individual objectives. */
  *objective_discretization,                      /* The length of the objective discretization in each dimension (for the elitist archive). */
  **ranks;                                         /* Ranks of all solutions in all populations. */
  extern individual ***populations,                                   /* The population containing the solutions. */
  ***selection,                                     /* Selected solutions, one for each population. */
  **elitist_archive,                               /* Archive of elitist solutions. */
  **approximation_set;                             /* Set of non-dominated solutions from all populations and the elitist archive. */
  extern int number_of_elites_added_to_archive_this_generation;
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  /*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
  void interpretCommandLine( int argc, char **argv );
  void run( void );
  void interpretCommandLine( int argc, char **argv );
  void parseCommandLine( int argc, char **argv );
  void parseOptions( int argc, char **argv, int *index );
  void parseFOSElementSize( int *index, int argc, char** argv );
  void printAllInstalledProblems( void );
  void optionError( char **argv, int index );
  void parseParameters( int argc, char **argv, int *index );
  void printUsage( void );
  void checkOptions( void );
  void printVerboseOverview( void );
  void initialize( void );
  void initializeNewPopulation( void );
  void initializeMemory( void );
  void initializeNewPopulationMemory( int population_index );
  void initializeCovarianceMatrices( int population_index );
  void initializeDistributionMultipliers( int population_index );
  void initializePopulationAndFitnessValues( int population_index );
  short initializePopulationProblemSpecific( int population_index );
  void initializePopulationPFVis( int population_index );
  void computeRanks( int population_index );
  void computeRandomRanks( int population_index );
  void computeUHVIRanks( int population_index );
  void computeObjectiveRanges(int population_index );
  short isSolutionInRangeBoundsForFOSElement( double *solution, int population_index, int cluster_index, int FOS_index );
  short checkTerminationConditionAllPopulations( void );
  short checkTerminationConditionOnePopulation( int population_index );
  short checkNumberOfEvaluationsTerminationCondition( void );
  short checkVTRTerminationCondition( void );
  short checkDistributionMultiplierTerminationCondition(int population_index );
  short checkTimeLimitTerminationCondition( void );
  void makeSelection(int population_index );
  int *completeSelectionBasedOnDiversityInLastSelectedRank(int population_index, int start_index, int number_to_select, int *sorted );
  int *greedyScatteredSubsetSelection( double **points, int number_of_points, int number_of_dimensions, int number_to_select );
  void makePopulation(int population_index );
  void estimateParameters( int population_index );
  void estimateFullCovarianceMatrixML( int population_index, int cluster_index );
  bool regularizeCovarianceMatrix(int population_index, int cluster_index, int FOS_index);
  void initializeFOS( int population_index, int cluster_index );
  FOS *learnLinkageTreeRVGOMEA( int population_index, int cluster_index );
  void inheritDistributionMultipliers( FOS *new_FOS, FOS *prev_FOS, double *multipliers );
  void evaluateCompletePopulation(int population_index );
  void copyBestSolutionsToPopulation(int population_index, double **objective_values_selection_scaled );
  void applyDistributionMultipliers(int population_index );
  void generateAndEvaluateNewSolutionsToFillPopulationAndUpdateElitistArchive(int population_index );
  short applyAMS( int population_index, int individual_index, int cluster_index );
  void applyForcedImprovements( int population_index, int individual_index, short *improved );
  void computeParametersForSampling( int population_index, int cluster_index );
  short generateNewSolutionFromFOSElement(int population_index, int cluster_index, int FOS_index, int individual_index );
  double *generateNewPartialSolutionFromFOSElement( int population_index, int cluster_index, int FOS_index );
  void adaptDistributionMultipliers( int population_index, int cluster_index, int FOS_index );
  short generationalImprovementForOneClusterForFOSElement(int population_index, int cluster_index, int FOS_index, double *st_dev_ratio );
  double getStDevRatioForOneClusterForFOSElement( int population_index, int cluster_index, int FOS_index, double *parameters );
  short solutionWasImprovedByFOSElement(int population_index, int cluster_index, int FOS_index , int individual_index);
  void ezilaitini( void );
  void ezilaitiniMemory( void );
  void ezilaitiniMemoryOnePopulation( int population_index );
  void ezilaitiniDistributionMultipliers( int population_index );
  void ezilaitiniCovarianceMatrices( int population_index );
  void ezilaitiniObjectiveRotationMatrix( void );
  void ezilaitiniParametersForSampling( int population_index );
  void run( void );
  // int mainGomea( int argc, char **argv );
  
  // scm
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
                bool print_verbose_overview);

  void freeGOMEA();
  /*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
  extern short      print_verbose_overview,                        /* Whether to print a overview of settings (0 = no). */
             use_guidelines;                                /* Whether to override parameters with guidelines (for those that exist). */
  extern int        base_population_size,
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
  extern double     tau,                                           /* The selection truncation percentile (in [1/population_size,1]). */
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
  extern clock_t    start, end;
  extern FOS       ***linkage_model;
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  extern short        use_boundary_repair,                          /* Repair out of bound parameter value to its nearest boundary value */
               use_forced_improvement;
  extern std::string write_directory, file_appendix;

}
