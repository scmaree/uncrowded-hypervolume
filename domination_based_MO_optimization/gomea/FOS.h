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

#ifndef FOS_H
#define FOS_H

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "Tools.h"
#include "Optimization.h"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace gomea
{
  
  typedef struct FOS {
      int length;
      int **sets;
      int *set_length;
  } FOS;

  /*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
  void printFOS( FOS *fos );
  FOS *readFOSFromFile( FILE *file );
  FOS *copyFOS( FOS *f );
  FOS *learnLinkageTree( double **covariance_matrix );
  int determineNearestNeighbour( int index, double **S_matrix, int *mpm_number_of_indices, int mpm_length );
  double getSimilarity( int a, int b );
  double **computeMIMatrix( double **covariance_matrix, int n );
  int *matchFOSElements( FOS *new_FOS, FOS *prev_FOS );
  int *hungarianAlgorithm( int** similarity_matrix, int dim );
  void hungarianAlgorithmAddToTree(int x, int prevx, short *S, int *prev, int *slack, int *slackx, int* lx, int *ly, int** similarity_matrix, int dim);
  int determineNearestNeighbour(int index, double **S_matrix, int *mpm_number_of_indices, int mpm_length );
  void ezilaitiniFOS(FOS *lm );
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  /*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
  extern int      *mpm_number_of_indices,
            FOS_element_ub,                       /* Cut-off value for bounded fixed linkage tree (BFLT). */
            use_univariate_FOS,                   /* Whether a univariate FOS is used. */
            learn_linkage_tree,                   /* Whether the FOS is learned at the start of each generation. */
            static_linkage_tree,                  /* Whether the FOS is fixed throughout optimization. */
            random_linkage_tree,                  /* Whether the fixed linkage tree is learned based on a random distance measure. */
            FOS_element_size;                     /* If positive, the size of blocks of consecutive variables in the FOS. If negative, determines specific kind of linkage tree FOS. */
  extern double ***MI_matrices,
          **S_matrix,
           *S_vector;                             /* Avoids quadratic memory requirements when a linkage tree is learned based on a random distance measure. */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

}
  
#endif
