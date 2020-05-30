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

#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "Tools.h"
#include "../mohillvallea/hicam_internal.h"
#include "../mohillvallea/fitness.h"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace gomea
{
  extern hicam::fitness_pt fitness_function;

  /*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
  void initializeObjectiveRotationMatrix( void );
  void ezilaitiniObjectiveRotationMatrix( void );
  double *rotateAllParameters( double *parameters );
  double *rotateParametersInRange( double *parameters, int from, int to );
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


  /*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
  extern short  black_box_evaluations;                         /* Whether full (black-box) evaluations must always be performed. */
  extern int       use_vtr;                                       /* Whether to terminate at the value-to-reach (VTR) (0 = no). */
  extern short      vtr_hit_status,                                /* Whether the VTR has been reached. */
        *populations_terminated,                        /* Which populations have been terminated. */
         evaluations_for_statistics_hit,                /* Can be used to write statistics after a certain number of evaluations. */
         write_generational_statistics,                 /* Whether to compute and write statistics every generation (0 = no). */
         write_generational_solutions;                  /* Whether to write the population every generation (0 = no). */
  extern int number_of_parameters,                          /* The number of parameters to be optimized. */
         number_of_populations,                         /* The number of parallel populations that initially partition the search space. */
         block_size,                                    /* The number of variables in one block of the 'sum of rotated ellipsoid blocks' function. */
         number_of_blocks,                              /* The number of blocks the 'sum of rotated ellipsoid blocks' function. */
         block_start,                                   /* The index at which the first block starts of the 'sum of rotated ellipsoid blocks' function. */
        *number_of_generations,                         /* The current generation count of a subgeneration in the interleaved multi-start scheme. */
         total_number_of_generations,                   /* The overarching generation count in the interleaved multi-start scheme. */
        *population_sizes;                              /* The size of the population. */
  extern double number_of_evaluations,                         /* The current number of times a function evaluation was performed. */
         vtr,                                           /* The value-to-reach (function value of best solution that is feasible). */
         rotation_angle,                                /* The angle of rotation to be applied to the problem. */
       **rotation_matrix,                               /* The rotation matrix to be applied before evaluating. */
        *lower_range_bounds,                            /* The respected lower bounds on parameters. */
        *upper_range_bounds,                            /* The respected upper bounds on parameters. */
        *lower_init_ranges,                             /* The initialization range lower bound. */
        *upper_init_ranges,                             /* The initialization range upper bound */
         lower_user_range,                              /* The initial lower range-bound indicated by the user (same for all dimensions). */
         upper_user_range;                              /* The initial upper range-bound indicated by the user (same for all dimensions). */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
}

#endif
