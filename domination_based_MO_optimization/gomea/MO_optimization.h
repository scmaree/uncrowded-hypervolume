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

#ifndef MO_OPTIMIZATION_H
#define MO_OPTIMIZATION_H

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "Optimization.h"
#include "FOS.h"
#include "../mohillvallea/hicam_internal.h"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


namespace gomea
{
  
  typedef struct individual{
      double *parameters;
      double *objective_values;
      double constraint_value;
      int NIS;
      double parameter_sum;
      int cluster_number;
  } individual;

  extern hicam::vec_t lowerRangeBound, upperRangeBound;
  
  /*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
  char *installedProblemName( int index );
  int numberOfInstalledProblems( void );
  int installedProblemNumberOfObjectives( int index );
  double installedProblemLowerRangeBound( int index, int dimension );
  double installedProblemUpperRangeBound( int index, int dimension );
  void initializeParameterRangeBounds( void );
  short isParameterInRangeBounds( double parameter, int dimension );
  double repairParameter( double parameter, int dimension );
  double distanceToRangeBounds(double *parameters);
  void installedProblemEvaluation( individual *ind, int number_of_touched_parameters, int *touched_parameters_indices, double *parameters_before, double *objective_values_before, double constraint_value_before );
  void installedProblemEvaluationWithoutRotation( int index, individual *ind, double *parameters, double *objective_value_result, double *constraint_value_result, int number_of_touched_parameters, int *touched_parameters_indices, double *parameters_before, double *objective_values_before, double constraint_value_before, int objective_index );
  void evaluateAdditionalFunctionsFull( individual *ind );
  void evaluateAdditionalFunctionsPartial( individual *ind, int number_of_touched_parameters, double *touched_parameters, double *parameters_before );
  short constraintParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y );
  short paretoDominates( double *objective_values_x, double *objective_values_y );
  void initializeProblem( void );
  void ezilaitiniProblem( void );
  short haveDPFSMetric( void );
  double **getDefaultFront( int *default_front_size );
  void updateElitistArchive( individual *ind );
  void removeFromElitistArchive( int *indices, int number_of_indices );
  void addToElitistArchive( individual *ind, int insert_index );
  void adaptObjectiveDiscretization( void );
  short sameObjectiveBox( double *objective_values_a, double *objective_values_b );
  void writeGenerationalStatisticsForOnePopulation( int population_index );
  void writeGenerationalStatisticsForOnePopulationWithDPFSMetric( int population_index );
  void writeGenerationalStatisticsForOnePopulationWithoutDPFSMetric( int population_index );
  void writeGenerationalSolutions( short final );
  void computeApproximationSet( void );
  void freeApproximationSet( void );
  double computeDPFSMetric( double **default_front, int default_front_size, individual **approximation_front, int approximation_front_size, short *to_be_removed_solution  );
  double compute2DHyperVolume(individual **pareto_front, int population_size );
  individual* initializeIndividual( void );
  individual* initializeIndividual(const hicam::solution_t & sol);
  hicam::solution_pt IndividualToSol(const individual* id);
  void ezilaitiniIndividual( individual *ind );
  void copyIndividual( individual *source, individual *destination );
  void copyIndividualWithoutParameters( individual *source, individual *destination );
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  /*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
  extern int           number_of_objectives,
                current_population_index,
                approximation_set_size;                        /* Number of solutions in the final answer (the approximation set). */
  extern double        sum_of_ellipsoids_normalization_factor;
  extern long          number_of_full_evaluations;
  extern short         approximation_set_reaches_vtr,
                statistics_file_existed;
  extern short         objective_discretization_in_effect,            /* Whether the objective space is currently being discretized for the elitist archive. */
               *elitist_archive_indices_inactive;              /* Elitist archive solutions flagged for removal. */
  extern int           elitist_archive_size,                          /* Number of solutions in the elitist archive. */
                elitist_archive_size_target,                   /* The lower bound of the targeted size of the elitist archive. */
                approximation_set_size_target,                   /* The lower bound of the targeted size of the elitist archive. */
                elitist_archive_capacity;                      /* Current memory allocation to elitist archive. */
  extern double       *best_objective_values_in_elitist_archive,      /* The best objective values in the archive in the individual objectives. */
               *objective_discretization,                      /* The length of the objective discretization in each dimension (for the elitist archive). */
              **ranks;                                         /* Ranks of all solutions in all populations. */
  extern individual ***populations,                                   /* The population containing the solutions. */
             ***selection,                                     /* Selected solutions, one for each population. */
              **elitist_archive,                               /* Archive of elitist solutions. */
              **approximation_set;                             /* Set of non-dominated solutions from all populations and the elitist archive. */
  extern short      print_verbose_overview;                        /* Whether to print a overview of settings (0 = no). */
  extern int number_of_elites_added_to_archive_this_generation;
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  extern std::string write_directory, file_appendix;
}
  
#endif
