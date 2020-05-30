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
#include "FOS.h"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace gomea
{
  
  /*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
  int      *mpm_number_of_indices,
  FOS_element_ub,                       /* Cut-off value for bounded fixed linkage tree (BFLT). */
  use_univariate_FOS,                   /* Whether a univariate FOS is used. */
  learn_linkage_tree,                   /* Whether the FOS is learned at the start of each generation. */
  static_linkage_tree,                  /* Whether the FOS is fixed throughout optimization. */
  random_linkage_tree,                  /* Whether the fixed linkage tree is learned based on a random distance measure. */
  FOS_element_size;                     /* If positive, the size of blocks of consecutive variables in the FOS. If negative, determines specific kind of linkage tree FOS. */
  double ***MI_matrices,
  **S_matrix,
  *S_vector;                             /* Avoids quadratic memory requirements when a linkage tree is learned based on a random distance measure. */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  
  
  void printFOS( FOS *fos )
  {
      int i,j;
      printf("{");
      for( i = 0; i < fos->length; i++ )
      {
          printf("[");
          for( j = 0; j < fos->set_length[i]; j++ )
          {
              printf("%d", fos->sets[i][j]);
              if( j != fos->set_length[i]-1)
                  printf(",");
          }
          printf("]");
          printf("\n");
      }
      printf("}\n");
  }

  FOS *readFOSFromFile( FILE *file )
  {
      char    c, string[1000];
      int     i, j, k;
      FOS     *new_FOS;

      new_FOS = (FOS*) Malloc(sizeof(FOS));
      /* Length */
      k = 0;
      new_FOS->length = 0;
      c = fgetc( file );
      while( (c != EOF) )
      {
          while( c != '\n' )
              c = fgetc( file );
          new_FOS->length++;
          c = fgetc( file );
      }
      new_FOS->set_length = (int *) Malloc( new_FOS->length*sizeof( int ) );
      new_FOS->sets       = (int **) Malloc( new_FOS->length*sizeof( int * ) );

      fclose( file );
      fflush( stdout );
      file = fopen( "FOS.in", "r" );
      for( i = 0; i < new_FOS->length; i++ )
      {
          new_FOS->set_length[i] = 0;
          c = fgetc( file );
          k = 0;
          while( (c != '\n') && (c != EOF) )
          {
              while( (c == ' ') || (c == '\n') || (c == '\t') )
                  c = fgetc( file );
              while( (c != ' ') && (c != '\n') && (c != '\t') )
                  c = fgetc( file );
              new_FOS->set_length[i]++;
          }
          printf("|FOS[%d]| = %d\n",i,new_FOS->set_length[i]);
          new_FOS->sets[i] = (int *) Malloc( new_FOS->set_length[i]*sizeof( int ) );
      }
      fclose( file );
      fflush( stdout );
      file = fopen( "FOS.in", "r" );

      for( i = 0; i < new_FOS->length; i++ )
      {
          c = fgetc( file );
          j = 0;
          while( (c != '\n') && (c != EOF) )
          {
              k = 0;
              while( (c == ' ') || (c == '\n') || (c == '\t') )
                  c = fgetc( file );
              while( (c != ' ') && (c != '\n') && (c != '\t') )
              {
                  string[k] = (char) c;
                  c = fgetc( file );
                  k++;
              }
              string[k] = '\0';
              printf("FOS[%d][%d] = %d\n",i,j,(int) atoi( string ));
              new_FOS->sets[i][j] = (int) atoi( string );
              j++;
          }
      }
      fclose( file );

      return( new_FOS );
  }

  FOS *copyFOS( FOS *f )
  {
      int i,j;
      FOS *new_FOS;

      new_FOS = (FOS*) Malloc(sizeof(FOS));
      new_FOS->length = f->length;
      new_FOS->set_length = (int*) Malloc(new_FOS->length*sizeof(int));
      new_FOS->sets = (int**) Malloc(new_FOS->length*sizeof(int*));
      for( i = 0; i < new_FOS->length; i++ )
      {
          new_FOS->set_length[i] = f->set_length[i];
          new_FOS->sets[i] = (int*) Malloc(new_FOS->set_length[i]*sizeof(int));
          for( j = 0; j < new_FOS->set_length[i]; j++ )
              new_FOS->sets[i][j] = f->sets[i][j];
      }
      return( new_FOS );
  }

  FOS *learnLinkageTree( double **covariance_matrix )
  {
      char     done;
      int      i, j, r0, r1, rswap, *indices, *order, *sorted,
              FOS_index, **mpm, mpm_length,
              **mpm_new, *mpm_new_number_of_indices, mpm_new_length,
              *NN_chain, NN_chain_length;
      double   mul0, mul1, **MI_matrix;
      FOS *new_FOS;

      /* Compute Mutual Information matrix */
      MI_matrix = NULL;
      if( learn_linkage_tree )
          MI_matrix = computeMIMatrix( covariance_matrix, number_of_parameters );

      /* Initialize MPM to the univariate factorization */
      order                 = randomPermutation( number_of_parameters );
      mpm                   = (int **) Malloc( number_of_parameters*sizeof( int * ) );
      mpm_number_of_indices = (int *) Malloc( number_of_parameters*sizeof( int ) );
      mpm_length            = number_of_parameters;
      mpm_new               = NULL;
      for( i = 0; i < number_of_parameters; i++ )
      {
          indices                  = (int *) Malloc( 1*sizeof( int ) );
          indices[0]               = order[i];
          mpm[i]                   = indices;
          mpm_number_of_indices[i] = 1;
      }
      free( order );

      /* Initialize LT to the initial MPM */
      new_FOS                     = (FOS*) Malloc(sizeof(FOS));
      new_FOS->length             = number_of_parameters+number_of_parameters-1;
      new_FOS->sets               = (int **) Malloc( new_FOS->length*sizeof( int * ) );
      new_FOS->set_length         = (int *) Malloc( new_FOS->length*sizeof( int ) );
      FOS_index                                   = 0;
      for( i = 0; i < mpm_length; i++ )
      {
          new_FOS->sets[FOS_index]       = mpm[i];
          new_FOS->set_length[FOS_index] = mpm_number_of_indices[i];
          FOS_index++;
      }

      /* Initialize similarity matrix */
      S_matrix = NULL;
      if( !random_linkage_tree ){
          S_matrix = (double **) Malloc( number_of_parameters*sizeof( double * ) );
          for( i = 0; i < number_of_parameters; i++ )
              S_matrix[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );
      }

      if( learn_linkage_tree )
      {
          for( i = 0; i < mpm_length; i++ )
              for( j = 0; j < mpm_length; j++ )
                  S_matrix[i][j] = MI_matrix[mpm[i][0]][mpm[j][0]];
          for( i = 0; i < mpm_length; i++ )
              S_matrix[i][i] = 0;

          for( i = 0; i < number_of_parameters; i++ )
              free( MI_matrix[i] );
          free( MI_matrix );
      }
      else if( random_linkage_tree )
      {
          S_vector = (double *) Malloc( number_of_parameters*sizeof(double));
          for( i = 0; i < number_of_parameters; i++ )
              S_vector[i] = randomRealUniform01();
      }
      else if( static_linkage_tree )
      {
        for( i = 0; i < mpm_length; i++ )
        {
            for( j = 0; j < i; j++ )
            {
                if( mpm[i][0] < block_start || mpm[j][0] < block_start ) S_matrix[i][j] = randomRealUniform01();
                else if( (mpm[i][0]-block_start)/block_size == (mpm[j][0]-block_start)/block_size ) S_matrix[i][j] = randomRealUniform01() + 1e8;
                else S_matrix[i][j] = randomRealUniform01() + 1e3;
                S_matrix[j][i] = S_matrix[i][j];
            }
            S_matrix[i][i] = 0;
        }
      }

      NN_chain        = (int *) Malloc( (number_of_parameters+2)*sizeof( int ) );
      NN_chain_length = 0;
      done            = 0;
      while( !done )
      {
          if( NN_chain_length == 0 )
          {
              NN_chain[NN_chain_length] = randomInt( mpm_length );
              NN_chain_length++;
          }

          if( NN_chain[NN_chain_length-1] >= mpm_length ) NN_chain[NN_chain_length-1] = mpm_length-1;

          while( NN_chain_length < 3 )
          {
              NN_chain[NN_chain_length] = determineNearestNeighbour( NN_chain[NN_chain_length-1], S_matrix, mpm_number_of_indices, mpm_length );
              NN_chain_length++;
          }

          while( NN_chain[NN_chain_length-3] != NN_chain[NN_chain_length-1] )
          {
              NN_chain[NN_chain_length] = determineNearestNeighbour( NN_chain[NN_chain_length-1], S_matrix, mpm_number_of_indices, mpm_length );
              if( ((getSimilarity(NN_chain[NN_chain_length-1],NN_chain[NN_chain_length]) == getSimilarity(NN_chain[NN_chain_length-1],NN_chain[NN_chain_length-2])))
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

          if( r1 < mpm_length && r1 != r0 ) /* This test is required for exceptional cases in which the nearest-neighbor ordering has changed within the chain while merging within that chain */
          {
              indices = (int *) Malloc( (mpm_number_of_indices[r0]+mpm_number_of_indices[r1])*sizeof( int ) );

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

              new_FOS->sets[FOS_index] = (int *) Malloc( (mpm_number_of_indices[r0]+mpm_number_of_indices[r1])*sizeof( int ) );
              new_FOS->set_length[FOS_index] = mpm_number_of_indices[r0]+mpm_number_of_indices[r1];
              sorted = mergeSortInt(indices, mpm_number_of_indices[r0]+mpm_number_of_indices[r1]);
              for( j = 0; j < mpm_number_of_indices[r0]+mpm_number_of_indices[r1]; j++ )
                  new_FOS->sets[FOS_index][j] = indices[sorted[j]];

              free( sorted );
              free( indices );

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

              mpm_new                   = (int **) Malloc( (mpm_length-1)*sizeof( int * ) );
              mpm_new_number_of_indices = (int *) Malloc( (mpm_length-1)*sizeof( int ) );
              mpm_new_length            = mpm_length-1;
              for( i = 0; i < mpm_new_length; i++ )
              {
                  mpm_new[i]                   = mpm[i];
                  mpm_new_number_of_indices[i] = mpm_number_of_indices[i];
              }

              mpm_new[r0]                   = new_FOS->sets[FOS_index];
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
                      for( i = 0; i < r1; i++ )
                      {
                          S_matrix[i][r1] = S_matrix[i][mpm_length-1];
                          S_matrix[r1][i] = S_matrix[i][r1];
                      }

                      for( j = r1+1; j < mpm_new_length; j++ )
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

              free( mpm );
              free( mpm_number_of_indices );
              mpm                   = mpm_new;
              mpm_number_of_indices = mpm_new_number_of_indices;
              mpm_length            = mpm_new_length;

              if( mpm_length == 1 )
                  done = 1;

              FOS_index++;
          }
      }

      new_FOS->length = FOS_index;

      free( NN_chain );

      free( mpm_new );
      free( mpm_number_of_indices );

      if( random_linkage_tree )
          free( S_vector );
      else
      {
          for( i = 0; i < number_of_parameters; i++ )
              free( S_matrix[i] );
          free( S_matrix );
      }

      return( new_FOS );
  }

  double getSimilarity( int a, int b )
  {
      if( FOS_element_ub < number_of_parameters && mpm_number_of_indices[a] + mpm_number_of_indices[b] > FOS_element_ub ) return( 0 );
      if( random_linkage_tree ) return( 1.0-fabs(S_vector[a]-S_vector[b]) );
      return( S_matrix[a][b] );
  }

  int determineNearestNeighbour( int index, double **S_matrix, int *mpm_number_of_indices, int mpm_length )
  {
      int i, result;

      result = 0;
      if( result == index )
          result++;
      for( i = 1; i < mpm_length; i++ )
      {
          if( ((getSimilarity(index,i) > getSimilarity(index,result)) || ((getSimilarity(index,i) == getSimilarity(index,result)) && (mpm_number_of_indices[i] < mpm_number_of_indices[result]))) && (i != index) )
              result = i;
      }

      return( result );
  }

  double **computeMIMatrix( double **covariance_matrix, int n )
  {
      int i, j;
      double si, sj, r, **MI_matrix;

      MI_matrix = (double **) Malloc( n*sizeof( double * ) );
      for( j = 0; j < n; j++ )
          MI_matrix[j] = (double *) Malloc( n*sizeof( double ) );
      for( i = 0; i < n; i++ )
      {
          MI_matrix[i][i] = 1e20;
          for( j = 0; j < i; j++ )
          {
              si = sqrt(covariance_matrix[i][i]);
              sj = sqrt(covariance_matrix[j][j]);
              r = covariance_matrix[i][j]/(si*sj);
              MI_matrix[i][j] = log(sqrt(1/(1-r*r)));
              MI_matrix[j][i] = MI_matrix[i][j];
          }
      }
      return( MI_matrix );
  }

  int *matchFOSElements( FOS *new_FOS, FOS *prev_FOS )
  {
      int      i, j, a, b, matches, *permutation, *hungarian_permutation,
              **FOS_element_similarity_matrix;

      permutation = (int *) Malloc( new_FOS->length*sizeof(int));
      FOS_element_similarity_matrix = (int**) Malloc((new_FOS->length-number_of_parameters)*sizeof(int*));
      for( i = 0; i < new_FOS->length-number_of_parameters; i++ )
          FOS_element_similarity_matrix[i] = (int*) Malloc((new_FOS->length-number_of_parameters)*sizeof(int));
      for( i = 0; i < number_of_parameters; i++ )
      {
          for( j = 0; j < number_of_parameters; j++ )
          {
              if( prev_FOS->sets[i][0] == new_FOS->sets[j][0] )
              {
                  permutation[i] = j;
                  break;
              }
          }
      }
      for( i = number_of_parameters; i < new_FOS->length; i++ )
      {
          for( j = number_of_parameters; j < new_FOS->length; j++ )
          {
              a = 0; b = 0;
              matches = 0;
              while( a < prev_FOS->set_length[i] && b < new_FOS->set_length[j] )
              {
                  if( prev_FOS->sets[i][a] < new_FOS->sets[j][b] )
                  {
                      a++;
                  }
                  else if( prev_FOS->sets[i][a] > new_FOS->sets[j][b] )
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
              FOS_element_similarity_matrix[i-number_of_parameters][j-number_of_parameters] = (int) 10000*(2.0*matches/(prev_FOS->set_length[i]+new_FOS->set_length[j]));
          }
      }

      hungarian_permutation = hungarianAlgorithm(FOS_element_similarity_matrix, new_FOS->length-number_of_parameters);
      for( i = 0; i < new_FOS->length-number_of_parameters; i++ )
          permutation[i+number_of_parameters] = hungarian_permutation[i]+number_of_parameters;

      for( i = 0; i < new_FOS->length-number_of_parameters; i++ )
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


  /*-=-=-=-=-=-=-=-=-=-=-=-=- Section Ezilaitini -=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
  void ezilaitiniFOS( FOS *lm )
  {
      int i;

      for( i = 0; i < lm->length; i++ )
          free( lm->sets[i] );
      free( lm->set_length );
      free( lm->sets );
      free( lm );
  }

}
