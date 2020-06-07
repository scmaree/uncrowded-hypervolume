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
#include "MO_optimization.h"
#include "../mohillvallea/elitist_archive.h"
#include "../mohillvallea/hillvalleyclustering.h"
// #include "../mohillvallea/solution.h"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


namespace gomea
{
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Problems -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Returns the name of an installed problem.
   */
  char *installedProblemName( int index )
  {
      switch( index )
      {
      case  0: return( (char *) "ZDT1" );
      case  1: return( (char *) "ZDT2" );
      case  2: return( (char *) "ZDT3" );
      case  3: return( (char *) "ZDT4" );
      case  4: return( (char *) "ZDT6" );
      case  5: return( (char *) "BD1" );
      case  6: return( (char *) "BD2 (scaled)" );
      case  7: return( (char *) "genMED Convex 2D" );
      case  8: return( (char *) "genMED Concave 2D" );
      case  9: return( (char *) "Sum of ellipsoids" );
      }

      return( NULL );
  }

  /**
   * Returns the number of problems installed.
   */
  int numberOfInstalledProblems( void )
  {
      static int result = -1;

      if( result == -1 )
      {
          result = 0;
          while( installedProblemName( result ) != NULL )
              result++;
      }

      return( result );
  }

  /**
   * Returns the number of objectives of an installed problem.
   */
  int installedProblemNumberOfObjectives( int index )
  {
    return (int) gomea::fitness_function->number_of_objectives;
  }

  /**
   * Returns the lower-range bound of an installed problem.
   */
  double installedProblemLowerRangeBound( int dimension )
  {
    return lowerRangeBound[dimension];
  }

  /**
   * Returns the upper-range bound of an installed problem.
   */
  double installedProblemUpperRangeBound( int dimension )
  {
    return upperRangeBound[dimension];
  }

  /**
   * Returns whether a parameter is inside the range bound of
   * every problem.
   */
  short isParameterInRangeBounds( double parameter, int dimension )
  {
      if( parameter < installedProblemLowerRangeBound(dimension) ||
              parameter > installedProblemUpperRangeBound(dimension ) ||
              isnan( parameter ) )
      {
          return( 0 );
      }

      return( 1 );
  }

  /**
   * Initializes the parameter range bounds.
   */
  void initializeParameterRangeBounds( void )
  {
      int i;

      lower_range_bounds = (double *) Malloc( number_of_parameters*sizeof( double ) );
      upper_range_bounds = (double *) Malloc( number_of_parameters*sizeof( double ) );
      lower_init_ranges  = (double *) Malloc( number_of_parameters*sizeof( double ) );
      upper_init_ranges  = (double *) Malloc( number_of_parameters*sizeof( double ) );

      for( i = 0; i < number_of_parameters; i++ )
      {
          lower_range_bounds[i] = installedProblemLowerRangeBound( i );
          upper_range_bounds[i] = installedProblemUpperRangeBound( i );
      }

      for( i = 0; i < number_of_parameters; i++ )
      {
          lower_init_ranges[i] = lower_user_range;
          if( lower_user_range < lower_range_bounds[i] )
              lower_init_ranges[i] = lower_range_bounds[i];
          if( lower_user_range > upper_range_bounds[i] )
              lower_init_ranges[i] = lower_range_bounds[i];

          upper_init_ranges[i] = upper_user_range;
          if( upper_user_range > upper_range_bounds[i] )
              upper_init_ranges[i] = upper_range_bounds[i];
          if( upper_user_range < lower_range_bounds[i] )
              upper_init_ranges[i] = upper_range_bounds[i];
      }
  }

  double repairParameter( double parameter, int dimension )
  {
      double result;

      result = parameter;
      result = fmax( result, installedProblemLowerRangeBound( dimension ));
      result = fmin( result, installedProblemUpperRangeBound( dimension ));

      return( result );
  }

  double distanceToRangeBounds( double *parameters )
  {
      int i;
      double d, sum;

      sum = 0.0;
      for( i = 0; i < number_of_parameters; i++ )
      {
          if( !isParameterInRangeBounds(parameters[i], i ))
          {
              d = parameters[i] - repairParameter( parameters[i], i );
              sum += d*d;
          }
      }

      return( sqrt(sum) );
  }

  void testEllipsoid( double *parameters, double *objective_value_result, double *constraint_value_result )
  {
      int i, j;
      double result;

      result = 0.0;
      for( i = 1; i < number_of_parameters; i++ )
      {
          j = (i-1) % block_size;
          result += pow( 10.0, 6.0*(((double) (j))/((double) (block_size-1))) )*(parameters[i])*(parameters[i]);
      }

      result /= sum_of_ellipsoids_normalization_factor;

      //
      result = 1.0 - parameters[0] + result;
      //
      //
      *objective_value_result  = result;
      *constraint_value_result = 0;
  }

  /**
   * Compute the value of all objectives
   * and the sum of all constraint violations
   * function after rotating the parameter vector.
   * Both are returned using pointer variables.
   */
  void installedProblemEvaluation( individual *ind, int number_of_touched_parameters, int *touched_parameters_indices, double *parameters_before, double *objective_values_before, double constraint_value_before )
  {
    int     i;
    double *rotated_parameters, *touched_parameters;

    touched_parameters = NULL;
    number_of_full_evaluations++;

    if( touched_parameters_indices != NULL && !black_box_evaluations )
    {
        touched_parameters = (double*) Malloc( number_of_touched_parameters*sizeof( double ) );
        for( i = 0; i < number_of_touched_parameters; i++ )
            touched_parameters[i] = ind->parameters[touched_parameters_indices[i]];
        evaluateAdditionalFunctionsPartial( ind, number_of_touched_parameters, touched_parameters, parameters_before );
        number_of_evaluations += (double)number_of_touched_parameters/(double)number_of_parameters;
    }
    else
    {
       number_of_evaluations++;
       // number_of_evaluations += (double)number_of_touched_parameters/(double)number_of_parameters;
    }

    hicam::solution_t sol((size_t) number_of_parameters, (size_t) number_of_objectives);
    
    if(rotation_angle == 0.0)
    {
      for(int i = 0; i < number_of_parameters; ++i) {
        sol.param[i] = ind->parameters[i];
      }
    }
    else
    {
      rotated_parameters = rotateAllParameters( ind->parameters );
      for(int i = 0; i < number_of_parameters; ++i) {
        sol.param[i] = rotated_parameters[i];
      }
      free( rotated_parameters );
    }
    
    fitness_function->evaluate(sol);
    
    for (int i = 0; i < number_of_objectives; ++i) {
      ind->objective_values[i] = sol.obj[i];
    }
    
    ind->constraint_value = sol.constraint;
    
    if( write_generational_statistics )
    {
      if( (int) (number_of_evaluations+1) % 2000 == 0 ) {
        evaluations_for_statistics_hit = 1;
      }
    }

    if( touched_parameters_indices != NULL )
    {
        free( touched_parameters );
    }
  }

  void evaluateAdditionalFunctionsFull( individual *ind )
  {
    int i;
    ind->parameter_sum = 0;
    for( i = 0; i < number_of_parameters; i++ )
        ind->parameter_sum += ind->parameters[i];
  }

  void evaluateAdditionalFunctionsPartial( individual *ind, int number_of_touched_parameters, double *touched_parameters, double *parameters_before )
  {
    int i;
    for( i = 0; i < number_of_touched_parameters; i++ )
    {
        ind->parameter_sum += touched_parameters[i];
        ind->parameter_sum -= parameters_before[i];
    }
  }
  
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Ranking -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Returns 1 if x constraint-Pareto-dominates y, 0 otherwise.
   * x is not better than y unless:
   * - x and y are both infeasible and x has a smaller sum of constraint violations, or
   * - x is feasible and y is not, or
   * - x and y are both feasible and x Pareto dominates y
   */
  short constraintParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y )
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
              result = paretoDominates( objective_values_x, objective_values_y );
      }

      return( result );
  }

  /**
   * Returns 1 if x Pareto-dominates y, 0 otherwise.
   */
  short paretoDominates( double *objective_values_x, double *objective_values_y )
  {
      short strict;
      int   i, result;

      result = 1;
      strict = 0;
      for( i = 0; i < number_of_objectives; i++ )
      {
        
        
        if( objective_values_x[i] == 0 || fabs( (objective_values_x[i] - objective_values_y[i])/objective_values_x[i] ) >= 1e-14 ) // scm: i removed 0.00001
        {
          if( objective_values_x[i] > objective_values_y[i] )
          {
            result = 0;
            break;
          }
          if( objective_values_x[i] < objective_values_y[i] )
            strict = 1;
        }
      }
      if( strict == 0 && result == 1 )
          result = 0;

      return( result );
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/



  /*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Initialization -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  void initializeProblem( void )
  {
      //switch( problem_index )
      //{
      //default: break;
      //}
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


  /*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Metrics -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /**
   * Returns whether the D_{Pf->S} metric can be computed.
   */
  short haveDPFSMetric( void )
  {
    /*
      int default_front_size;

      get DefaultFront( &default_front_size );
      if( default_front_size > 0 )
          return( 1 );

      return( 0 );
     */
    
    if(fitness_function->get_pareto_set()) {
      return 1;
    }
    return 0;
    
  }

  double **readDefaultFront( char *filename, int *default_front_size )
  {
      int i;
      FILE* f;
      static double **result = NULL;

      *default_front_size = 5000;
      f = fopen(filename, "r");

      if( f == NULL )
      {
          printf("No defaultFront file found.\n");
          fclose( f );
          *default_front_size = 0;
          return( NULL );
      }

      if( result == NULL )
      {
          result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
          for( i = 0; i < (*default_front_size); i++ )
          {
              result[i] = (double *) Malloc( 2*sizeof( double ) );
              fscanf(f, "%lf %lf\n", &result[i][0], &result[i][1]);
          }
      }

      fclose( f );

      return result;
  }

  /**
   * Returns the default front(NULL if there is none).
   * The number of solutions in the default
   * front is returned in the pointer variable.
   */
  double **getDefaultFront( int *default_front_size )
  {
    
    if(fitness_function->get_pareto_set())
    {
      static double **result = NULL;
      *default_front_size = (int) fitness_function->pareto_set.size();
      
      result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
      for( int i = 0; i < (*default_front_size); i++ )
        result[i] = (double *) Malloc( 2*sizeof( double ) );
      
      for( int i = 0; i < (*default_front_size); i++ )
      {
        result[i][0] = fitness_function->pareto_set.sols[i]->obj[0];
        result[i][1] = fitness_function->pareto_set.sols[i]->obj[1]; // TODO: implement this for more than 2 objectives.
      }
      
      return result;
    }
    
    // when front is not found,
    *default_front_size = 0;
    return NULL;
    
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  /*=-=-=-=-=-=-=-=-=-=-=-= Section Elitist Archive -==-=-=-=-=-=-=-=-=-=-=-=-*/
  /**
   * Adapts the objective box discretization. If the numbre
   * of solutions in the elitist archive is too high or too low
   * compared to the population size, the objective box
   * discretization is adjusted accordingly. In doing so, the
   * entire elitist archive is first emptied and then refilled.
   */
  void adaptObjectiveDiscretization( void )
  {
      int    i, j, k, na, nb, nc,
              elitist_archive_size_target_lower_bound,
              elitist_archive_size_target_upper_bound;
      double low, high, *elitist_archive_objective_ranges, elitist_archive_copy_size;
      individual **elitist_archive_copy;

      //printf("===========================================\n");
      //printf("Generation     : %d\n",number_of_generations);
      //printf("No improv. str.: %d\n",no_improvement_stretch);
      //printf("#Elitist target: %d\n",elitist_archive_size_target);
      //printf("#Elitist before: %d\n",elitist_archive_size);
      //printf("In effect      : %s\n",objective_discretization_in_effect?"true":"false");
      //printf("OBD before     : %e %e\n",objective_discretization[0],objective_discretization[1]);

      elitist_archive_size_target_lower_bound = (int) (0.75*elitist_archive_size_target);
      elitist_archive_size_target_upper_bound = (int) (1.25*elitist_archive_size_target);

      if( objective_discretization_in_effect && (elitist_archive_size < elitist_archive_size_target_lower_bound) )
          objective_discretization_in_effect = 0;

      if( elitist_archive_size > elitist_archive_size_target_upper_bound )
      {
          objective_discretization_in_effect = 1;

          elitist_archive_objective_ranges = (double *) Malloc( number_of_objectives*sizeof( double ) );
          for( j = 0; j < number_of_objectives; j++ )
          {
              low  = elitist_archive[0]->objective_values[j];
              high = elitist_archive[0]->objective_values[j];

              for( i = 0; i < elitist_archive_size; i++ )
              {
                  if( elitist_archive[i]->objective_values[j] < low )
                      low = elitist_archive[i]->objective_values[j];
                  if( elitist_archive[i]->objective_values[j] > high )
                      high = elitist_archive[i]->objective_values[j];
              }

              elitist_archive_objective_ranges[j] = high - low;
          }

          na = 1;
          nb = (int) pow(2.0,25.0);
          for( k = 0; k < 25; k++ )
          {
              elitist_archive_copy_size              = elitist_archive_size;
              elitist_archive_copy                   = (individual **) Malloc( elitist_archive_copy_size*sizeof( individual * ) );
              for( i = 0; i < elitist_archive_copy_size; i++ )
                  elitist_archive_copy[i]              = initializeIndividual();
              for( i = 0; i < elitist_archive_copy_size; i++ )
              {
                  copyIndividual( elitist_archive[i], elitist_archive_copy[i]);
              }

              nc = (na + nb) / 2;
              for( i = 0; i < number_of_objectives; i++ )
                  objective_discretization[i] = elitist_archive_objective_ranges[i]/((double) nc);

              /* Restore the original elitist archive after the first cycle in this loop */
              if( k > 0 )
              {
                  elitist_archive_size = 0;
                  for( i = 0; i < elitist_archive_copy_size; i++ )
                      addToElitistArchive( elitist_archive_copy[i], i );
              }

              /* Clear the elitist archive */
              elitist_archive_size = 0;

              /* Rebuild the elitist archive */
              for( i = 0; i < elitist_archive_copy_size; i++ )
                  updateElitistArchive( elitist_archive_copy[i] );

              if( elitist_archive_size <= elitist_archive_size_target_lower_bound )
                  na = nc;
              else
                  nb = nc;

              /* Copy the entire elitist archive */
              if( elitist_archive_copy != NULL )
              {
                  for( i = 0; i < elitist_archive_copy_size; i++ )
                      ezilaitiniIndividual( elitist_archive_copy[i] );
                  free( elitist_archive_copy );
              }
          }

          free( elitist_archive_objective_ranges );
      }
      //printf("In effect      : %s\n",objective_discretization_in_effect?"true":"false");
      //printf("OBD after      : %e %e\n",objective_discretization[0],objective_discretization[1]);
      //printf("#Elitist after : %d\n",elitist_archive_size);
      //printf("===========================================\n");
  }

  /**
   * Returns 1 if two solutions share the same objective box, 0 otherwise.
   */
  short sameObjectiveBox( double *objective_values_a, double *objective_values_b )
  {
      int i;

      if( !objective_discretization_in_effect )
      {
          /* If the solutions are identical, they are still in the (infinitely small) same objective box. */
          for( i = 0; i < number_of_objectives; i++ )
          {
              if( objective_values_a[i] != objective_values_b[i] )
                  return( 0 );
          }

          return( 1 );
      }

      for( i = 0; i < number_of_objectives; i++ )
      {
          if( ((int) (objective_values_a[i] / objective_discretization[i])) != ((int) (objective_values_b[i] / objective_discretization[i])) ){
              return( 0 );
          }
      }

      return( 1 );
  }

  /**
   * Updates the elitist archive by offering a new solution
   * to possibly be added to the archive. If there are no
   * solutions in the archive yet, the solution is added.
   * Otherwise, the number of times the solution is
   * dominated is computed. Solution A is always dominated
   * by solution B that is in the same domination-box if
   * B dominates A or A and B do not dominate each other.
   * If the number of times a solution is dominated, is 0,
   * the solution is added to the archive and all solutions
   * dominated by the new solution, are purged from the archive.
   */
  void updateElitistArchive( individual *ind )
  {
      short is_dominated_itself, is_extreme_compared_to_archive, all_to_be_removed;
      int   i, j, *indices_dominated, number_of_solutions_dominated, insert_index;

      is_extreme_compared_to_archive = 0;
      all_to_be_removed = 1;
      insert_index = elitist_archive_size;
      if( ind->constraint_value == 0 )
      {
          if( elitist_archive_size == 0 )
          {
              is_extreme_compared_to_archive = 1;
          }
          else
          {
              for( j = 0; j < number_of_objectives; j++ )
              {
                  if( ind->objective_values[j] < best_objective_values_in_elitist_archive[j] )
                  {
                      is_extreme_compared_to_archive = 1;
                      break;
                  }
              }
          }
      }

      if( elitist_archive_size == 0 )
          addToElitistArchive( ind, insert_index );
      else
      {
          indices_dominated             = (int *) Malloc( elitist_archive_size*sizeof( int ) );
          number_of_solutions_dominated = 0;
          is_dominated_itself           = 0;
          double *bla = (double*) Malloc(number_of_objectives*sizeof(double));
          bla[0] = 0.5; bla[1] = 0.5;
          for( i = 0; i < elitist_archive_size; i++ )
          {
              if( elitist_archive_indices_inactive[i] )
              {
                  if( i < insert_index )
                      insert_index = i;
                  continue;
              }
              all_to_be_removed = 0;
              if( constraintParetoDominates( elitist_archive[i]->objective_values, elitist_archive[i]->constraint_value, ind->objective_values, ind->constraint_value ) )
                  is_dominated_itself = 1;
              else
              {

                  if( !constraintParetoDominates( ind->objective_values, ind->constraint_value, elitist_archive[i]->objective_values, elitist_archive[i]->constraint_value ) )
                  {
                      if( sameObjectiveBox( elitist_archive[i]->objective_values, ind->objective_values ) && (!is_extreme_compared_to_archive) )
                          is_dominated_itself = 1;
                  }
              }

              if( is_dominated_itself )
                  break;
          }
          free( bla );

          if( all_to_be_removed )
              addToElitistArchive( ind, insert_index );
          else if( !is_dominated_itself )
          {
              for( i = 0; i < elitist_archive_size; i++ )
              {
                  if( elitist_archive_indices_inactive[i] )
                      continue;
                  if( constraintParetoDominates( ind->objective_values, ind->constraint_value, elitist_archive[i]->objective_values, elitist_archive[i]->constraint_value ) || sameObjectiveBox( elitist_archive[i]->objective_values, ind->objective_values ) )
                  {
                      indices_dominated[number_of_solutions_dominated] = i;
                      elitist_archive_indices_inactive[i] = 1;
                      number_of_solutions_dominated++;
                  }
              }

              if( number_of_solutions_dominated > 0 )
              {
                  if( ind->constraint_value == 0 )
                  {
                      for( i = 0; i < number_of_solutions_dominated; i++ )
                      {
                          for( j = 0; j < number_of_objectives; j++ )
                          {
                              if( elitist_archive[indices_dominated[i]]->objective_values[j] == best_objective_values_in_elitist_archive[j] )
                              {
                                  best_objective_values_in_elitist_archive[j] = ind->objective_values[j];
                              }
                          }
                      }
                  }
                  removeFromElitistArchive( indices_dominated, number_of_solutions_dominated );
              }

              addToElitistArchive( ind, insert_index );
          }

          free( indices_dominated );
      }
  }

  void removeFromElitistArchive( int *indices, int number_of_indices )
  {
      int i;

      for( i = 0; i < number_of_indices; i++ )
          elitist_archive_indices_inactive[indices[i]] = 1;
  }

  /**
   * Adds a solution to the elitist archive.
   */
  void addToElitistArchive( individual *ind, int insert_index )
  {
      number_of_elites_added_to_archive_this_generation++;
    
      int      i, j, elitist_archive_capacity_new, elitist_archive_size_new;
      short *elitist_archive_indices_inactive_new;
      individual **elitist_archive_new;

      if( insert_index >= elitist_archive_capacity )
      {
          elitist_archive_size_new              = 0;
          elitist_archive_capacity_new          = elitist_archive_capacity*2+1;
          elitist_archive_new                   = (individual **) Malloc( elitist_archive_capacity_new*sizeof( individual * ) );
          elitist_archive_indices_inactive_new = (short *) Malloc( elitist_archive_capacity_new*sizeof( short ));
          for( i = 0; i < elitist_archive_capacity_new; i++ )
          {
              elitist_archive_new[i]         = initializeIndividual();
              elitist_archive_indices_inactive_new[i] = 0;
          }

          for( i = 0; i < elitist_archive_size; i++ )
          {
              copyIndividual( elitist_archive[i], elitist_archive_new[elitist_archive_size_new] );
              elitist_archive_size_new++;
          }

          for( i = 0; i < elitist_archive_capacity; i++ )
              ezilaitiniIndividual( elitist_archive[i] );
          free( elitist_archive );
          free( elitist_archive_indices_inactive );

          elitist_archive_size              = elitist_archive_size_new;
          elitist_archive_capacity          = elitist_archive_capacity_new;
          elitist_archive                   = elitist_archive_new;
          elitist_archive_indices_inactive = elitist_archive_indices_inactive_new;
          insert_index = elitist_archive_size;
      }

      copyIndividual( ind, elitist_archive[insert_index] );

      if( insert_index == elitist_archive_size )
          elitist_archive_size++;
      elitist_archive_indices_inactive[insert_index] = 0;

      if( ind->constraint_value == 0 )
          for( j = 0; j < number_of_objectives; j++ )
              if( ind->objective_values[j] < best_objective_values_in_elitist_archive[j] )
                  best_objective_values_in_elitist_archive[j] = ind->objective_values[j];
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/




  /*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Output =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  /**
   * Writes (appends) statistics about the current generation to a
   * file named "statistics.dat".
   */
  void writeGenerationalStatisticsForOnePopulation( int population_index )
  {
    
    // begin scm
    
    char    string[1000];
    bool   enable_hyper_volume;
    FILE   *file;
    
    enable_hyper_volume = (fitness_function->get_number_of_objectives() == 2);
    file = NULL;
    if( total_number_of_generations == 0 && statistics_file_existed == 0)
    //if (write_header)
    {
      
      sprintf(string, "%sstatistics%s.dat", write_directory.c_str(), file_appendix.c_str());
      file = fopen(string, "w");
      
      sprintf(string, "# Generation  Evaluations   Time (s)");
      fputs(string, file); if (print_verbose_overview) std::cout << string;
      for (size_t i = 0; i < fitness_function->get_number_of_objectives(); i++)
      {
        sprintf(string, " Best_obj[%zu]", i);
        fputs(string, file); if (print_verbose_overview) std::cout << string;
      }
      
      sprintf(string, " Hypervol. Appr.set     IGD Appr.set           GD Appr.set              IGDX       SR     Smoothness Appr.set.size  Pop.rnk   Subgen.  Pop.size    Rank0.sols Archive_size HV Archive             IGD Archive            GD Archive               IGDX Archive SR Archive\n");
      fputs(string, file); if (print_verbose_overview) std::cout << string;
      
    }
    else {
      sprintf(string, "%sstatistics%s.dat", write_directory.c_str(), file_appendix.c_str());
      file = fopen(string, "a");
    }
    
    sprintf(string, "  %10d %11d %11.3f", total_number_of_generations, (int) number_of_evaluations, getTimer());
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    for (size_t i = 0; i < fitness_function->get_number_of_objectives(); i++)
    {
      sprintf( string, " %11.3e", best_objective_values_in_elitist_archive[i]);
      fputs( string, file ); if (print_verbose_overview) std::cout << string;
    }
    
    
    double HV_appr = 0.0;
    double IGD_appr = 0.0;
    double GD_appr = 0.0;
    double IGDX_appr = 0.0;
    double SR_appr = 0.0;
    
    
    // create some hillvalley data structures for writing stuff
    // approximation set is small,
    // elitist archive contains the entire large archive (which is called the approximation_set by GOMEA, which is confusing, i know).
    
    hicam::rng_pt rng = std::make_shared<hicam::rng_t>(1104913 + total_number_of_generations);
    hicam::elitist_archive_pt elitist_archive = std::make_shared<hicam::elitist_archive_t>(elitist_archive_size_target, rng);
    
    elitist_archive->sols.reserve(elitist_archive_size_target);
    
    // copy the gomea archive to a HICAM data structure
    for(int i = 0; i < approximation_set_size; i++ ) {
      elitist_archive->updateArchive(IndividualToSol(approximation_set[i]));
    }
    
    // construct an approximation set
    hicam::elitist_archive_pt approximation_set = std::make_shared<hicam::elitist_archive_t>(approximation_set_size, rng);
    elitist_archive->getAllSols(approximation_set->sols); // weird line but this returns all sols from the elitits archive and sets them to the approximation set.
    
    
    
    /*
    
    approximation_set->target_size = approximation_set_size_target;
    approximation_set->set_use_greedy_selection(true);
    approximation_set->set_use_parameter_distances(false);
    approximation_set->adaptArchiveSize();
    approximation_set->removeSolutionNullptrs();
    
     */
    
    approximation_set->reduceArchiveSizeByHSS(approximation_set_size_target, fitness_function->hypervolume_max_f0, fitness_function->hypervolume_max_f1);
    
    if(enable_hyper_volume) {
      HV_appr = approximation_set->compute2DHyperVolume(fitness_function->hypervolume_max_f1, fitness_function->hypervolume_max_f1);
    }
    
    if (fitness_function->igd_available) {
      IGD_appr = approximation_set->computeIGD(fitness_function->pareto_set);
      
      if(fitness_function->analytical_gd_avialable) {
        GD_appr = approximation_set->computeAnalyticGD(*fitness_function);
      } else {
        GD_appr = approximation_set->computeGD(fitness_function->pareto_set);
      }
    }
    
    if (fitness_function->igdx_available) {
      IGDX_appr = approximation_set->computeIGDX(fitness_function->pareto_set);
    }
    
    if (fitness_function->sr_available)
    {
      hicam::vec_t ones(fitness_function->pareto_sets_max_igdx.size(), 1.0);
      
      double threshold;
      if(fitness_function->number_of_objectives == 2) {
        threshold = 5e-2;
      }
      else {
        threshold = 1e-1;
      }
      SR_appr = approximation_set->computeSR(fitness_function->pareto_sets, threshold, ones);
    }
    
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
    
    double smoothness = 0.0;
    std::sort(approximation_set->sols.begin(), approximation_set->sols.end(), hicam::solution_t::strictly_better_solution_via_pointers_obj0_unconstraint);
    
    smoothness = approximation_set->computeSmoothness();
    
    size_t pop_ranks = worst_rank;
    
    sprintf(string, " %20.16e", HV_appr);
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    sprintf(string, " %20.16e", IGD_appr);
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    sprintf(string, " %20.16e", GD_appr);
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    sprintf(string, " %11.3e", IGDX_appr);
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    sprintf(string, " %7.4f", SR_appr);
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    sprintf(string, " %11.3e", smoothness);
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    sprintf(string, " %11zu", approximation_set->size());
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    sprintf(string, "   %6zu %9d %9zu %13zu ", pop_ranks, number_of_generations[population_index], (size_t) population_sizes[population_index], number_of_rank_0_sols);
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    sprintf(string, " %11zu", elitist_archive->size());
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    double HV_archive = 0.0;
    double IGD_archive = 0.0;
    double GD_archive = 0.0;
    double IGDX_archive = 0.0;
    double SR_archive = 0.0;
    
    if(enable_hyper_volume) {
      HV_archive = elitist_archive->compute2DHyperVolume(fitness_function->hypervolume_max_f1, fitness_function->hypervolume_max_f1);
    }
    
    if (fitness_function->igd_available) {
      IGD_archive = elitist_archive->computeIGD(fitness_function->pareto_set);
      
      if(fitness_function->analytical_gd_avialable) {
        GD_archive = elitist_archive->computeAnalyticGD(*fitness_function);
      } else {
        GD_archive = elitist_archive->computeGD(fitness_function->pareto_set);
      }
    }
    
    if (fitness_function->igdx_available) {
      IGDX_archive = elitist_archive->computeIGDX(fitness_function->pareto_set);
    }
    
    if (fitness_function->sr_available)
    {
      hicam::vec_t ones(fitness_function->pareto_sets_max_igdx.size(), 1.0);
      
      double threshold;
      if(fitness_function->number_of_objectives == 2) {
        threshold = 5e-2;
      }
      else {
        threshold = 1e-1;
      }
      SR_archive = elitist_archive->computeSR(fitness_function->pareto_sets, threshold, ones);
    }
    
    
    sprintf(string, " %20.16e", HV_archive);
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    sprintf(string, " %20.16e", IGD_archive);
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    sprintf(string, " %20.16e", GD_archive);
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    sprintf(string, " %11.3e", IGDX_archive);
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    sprintf(string, " %7.4f \n", SR_archive);
    fputs(string, file); if (print_verbose_overview) std::cout << string;
    
    fclose(file);
    
  }

  /**
   * Writes the solutions to various files. The filenames
   * contain the generation counter. If the flag final is
   * set (final != 0), the generation number in the filename
   * is replaced with the word "final".
   *
   * approximation_set_generation_xxxxx.dat: the approximation set (actual answer)
   * elitist_archive_generation_xxxxx.dat  : the elitist archive
   * population_generation_xxxxx.dat       : the population
   * selection_generation_xxxxx.dat        : the selected solutions
   * cluster_xxxxx_generation_xxxxx.dat    : the individual clusters
   */
  void writeGenerationalSolutions( short final )
  {
      char  string[1000];
      
      // Approximation set
      if (final) {
        sprintf(string, "%sapproximation_set_final%s.dat", write_directory.c_str(), file_appendix.c_str());
      }
      else {
        sprintf(string, "%sapproximation_set_generation%s_%05d.dat", write_directory.c_str(), file_appendix.c_str(), total_number_of_generations);
      }
    
    hicam::rng_pt rng = std::make_shared<hicam::rng_t>(1104913 + total_number_of_generations);
    hicam::population_t pop;
    pop.sols.resize(approximation_set_size);
	
    // copy the gomea archive to a HICAM data structure
    for(int i = 0; i < approximation_set_size; i++ ) {
      pop.sols[i] = IndividualToSol(approximation_set[i]);
    }
	
    hicam::hvc_pt hvc = std::make_shared<hicam::hvc_t>(fitness_function);
    std::vector<hicam::population_pt> archives;
    // unsigned int temp_fevals = 0;
    // double ael = 0.0;
    // hvc->cluster(pop, archives, temp_fevals, ael, false, true, 0, rng);
    pop.writeToFile(string);
  }
  
  /**
   * Computes the approximation set: the non-dominated solutions
   * in the population and the elitist archive combined.
   */
  void computeApproximationSet( void )
  {
      short    dominated, same_objectives;
      int      i, j, k, *indices_of_rank0, *population_indices_of_rank0, rank0_size, non_dominated_size,
              population_rank0_and_elitist_archive_size,
              *rank0_contribution, tot_rank0_size;
      double **population_rank0_and_elitist_archive,
              **population_rank0_and_elitist_archive_objective_values,
              *population_rank0_and_elitist_archive_constraint_values;

      /* First, join rank0 of the population with the elitist archive */
      indices_of_rank0 = (int *) Malloc( 2*population_sizes[number_of_populations-1]*sizeof( int ) );
      population_indices_of_rank0 = (int *) Malloc( 2*population_sizes[number_of_populations-1]*sizeof( int ) );
      rank0_size       = 0;
      for( i = 0; i < number_of_populations; i++ )
      {
          for( j = 0; j < population_sizes[i]; j++ )
          {
              if( ranks[i][j] == 0 )
              {
                  indices_of_rank0[rank0_size] = j;
                  population_indices_of_rank0[rank0_size] = i;
                  rank0_size++;
              }
          }
      }

      population_rank0_and_elitist_archive_size              = rank0_size + elitist_archive_size;
      population_rank0_and_elitist_archive                   = (double **) Malloc( population_rank0_and_elitist_archive_size*sizeof( double * ) );
      population_rank0_and_elitist_archive_objective_values  = (double **) Malloc( population_rank0_and_elitist_archive_size*sizeof( double * ) );
      population_rank0_and_elitist_archive_constraint_values = (double *) Malloc( population_rank0_and_elitist_archive_size*sizeof( double ) );

      for( i = 0; i < population_rank0_and_elitist_archive_size; i++ )
      {
          population_rank0_and_elitist_archive[i]                  = (double *) Malloc( number_of_parameters*sizeof( double ) );
          population_rank0_and_elitist_archive_objective_values[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
      }

      k = 0;
      for( i = 0; i < rank0_size; i++ )
      {
          for( j = 0; j < number_of_parameters; j++ )
              population_rank0_and_elitist_archive[k][j] = populations[population_indices_of_rank0[i]][indices_of_rank0[i]]->parameters[j];
          for( j = 0; j < number_of_objectives; j++ )
              population_rank0_and_elitist_archive_objective_values[k][j] = populations[population_indices_of_rank0[i]][indices_of_rank0[i]]->objective_values[j];
          population_rank0_and_elitist_archive_constraint_values[k] = populations[population_indices_of_rank0[i]][indices_of_rank0[i]]->constraint_value;

          k++;
      }

      for( i = 0; i < elitist_archive_size; i++ )
      {
          for( j = 0; j < number_of_parameters; j++ )
              population_rank0_and_elitist_archive[k][j] = elitist_archive[i]->parameters[j];
          for( j = 0; j < number_of_objectives; j++ )
              population_rank0_and_elitist_archive_objective_values[k][j] = elitist_archive[i]->objective_values[j];
          population_rank0_and_elitist_archive_constraint_values[k] = elitist_archive[i]->constraint_value;

          k++;
      }
      free( indices_of_rank0 );

      /* Second, compute rank0 solutions amongst all solutions */
      indices_of_rank0 = (int *) Malloc( population_rank0_and_elitist_archive_size*sizeof( int ) );
      rank0_contribution = (int *) Malloc( number_of_populations*sizeof( int ) );
      for( i = 0; i < number_of_populations; i++ ) rank0_contribution[i] = 0;
      non_dominated_size       = 0;
      for( i = 0; i < population_rank0_and_elitist_archive_size; i++ )
      {
          dominated = 0;
          for( j = 0; j < population_rank0_and_elitist_archive_size; j++ )
          {
              if( i != j )
              {
                  if( constraintParetoDominates( population_rank0_and_elitist_archive_objective_values[j], population_rank0_and_elitist_archive_constraint_values[j], population_rank0_and_elitist_archive_objective_values[i], population_rank0_and_elitist_archive_constraint_values[i] ) )
                  {
                      dominated = 1;
                      break;
                  }
                  same_objectives = 1;
                  for( k = 0; k < number_of_objectives; k++ )
                  {
                      if( population_rank0_and_elitist_archive_objective_values[i][k] != population_rank0_and_elitist_archive_objective_values[j][k] )
                      {
                          same_objectives = 0;
                          break;
                      }
                  }
                  if( same_objectives && (population_rank0_and_elitist_archive_constraint_values[i] == population_rank0_and_elitist_archive_constraint_values[j]) && (i > j) )
                  {
                      dominated = 1;
                      if( i < rank0_size && j >= rank0_size ) rank0_contribution[population_indices_of_rank0[i]]++;
                      break;
                  }
              }
          }

          if( !dominated )
          {
              if( i < rank0_size ) rank0_contribution[population_indices_of_rank0[i]]++;
              indices_of_rank0[non_dominated_size] = i;
              non_dominated_size++;
          }
      }

      tot_rank0_size = 0;
      for( i = 0; i < number_of_populations; i++ ) tot_rank0_size += rank0_contribution[i];
      if( tot_rank0_size > 0 )
      {
          for( i = 0; i < number_of_populations-1; i++ )
          {
              if( ((double)rank0_contribution[i])/(double)tot_rank0_size < 0.1 )
                  populations_terminated[i] = 1;
              else
                  break;
          }
      }

      free( rank0_contribution );

      approximation_set_size              = non_dominated_size;
      approximation_set                   = (individual **) Malloc( approximation_set_size*sizeof( individual * ) );
      for( i = 0; i < approximation_set_size; i++ )
          approximation_set[i]                  = initializeIndividual();

      for( i = 0; i < non_dominated_size; i++ )
      {
          for( j = 0; j < number_of_parameters; j++ )
              approximation_set[i]->parameters[j] = population_rank0_and_elitist_archive[indices_of_rank0[i]][j];
          for( j = 0; j < number_of_objectives; j++ )
              approximation_set[i]->objective_values[j] = population_rank0_and_elitist_archive_objective_values[indices_of_rank0[i]][j];
          approximation_set[i]->constraint_value = population_rank0_and_elitist_archive_constraint_values[indices_of_rank0[i]];
      }

      free( indices_of_rank0 );
      free( population_indices_of_rank0 );
      for( i = 0; i < population_rank0_and_elitist_archive_size; i++ )
      {
          free( population_rank0_and_elitist_archive[i] );
          free( population_rank0_and_elitist_archive_objective_values[i] );
      }
      free( population_rank0_and_elitist_archive );
      free( population_rank0_and_elitist_archive_objective_values );
      free( population_rank0_and_elitist_archive_constraint_values );
  }

  /**
   * Frees the memory allocated for the approximation set.
   * The memory is only needed for reporting the current
   * answer (Pareto set), not for the internal workings
   * of the algorithm.
   */
  void freeApproximationSet( void )
  {
      int i;

      for( i = 0; i < approximation_set_size; i++ )
          ezilaitiniIndividual( approximation_set[i] );
      free( approximation_set );
  }

  /**
   * Computes the D_{Pf->S} metric for a given default front,
   * number of solutions in the default front, an approximation
   * front and the number of solutions in the approximation front.
   */
  double computeDPFSMetric(double **default_front, int default_front_size, individual **approximation_front, int approximation_front_size, short *to_be_removed_solution )
  {
      int    i, j;
      double result, distance, smallest_distance;

      if( approximation_front_size == 0 )
          return( 1e+308 );

      result = 0.0;
      for( i = 0; i < default_front_size; i++ )
      {
          smallest_distance = 1e+308;
          for( j = 0; j < approximation_front_size; j++ )
          {
              if( approximation_front[j]->constraint_value == 0 && to_be_removed_solution[j] == 0 )
              {
                  distance = distanceEuclidean( default_front[i], approximation_front[j]->objective_values, number_of_objectives );
                  if( distance < smallest_distance )
                      smallest_distance = distance;
              }
          }
          result += smallest_distance;
      }
      result /= (double) default_front_size;

      return( result );
  }

  double compute2DHyperVolume( individual **pareto_front, int population_size )
  {
    int i, n, *sorted;
    double max_0, max_1, *obj_0, area;

    n = population_size;
    max_0 = 1.1;
    max_1 = 1.1;
    
    max_0 = fitness_function->hypervolume_max_f0;
    max_1 = fitness_function->hypervolume_max_f1;

    if (population_size == 0) {
      return 0.0;
    }
    
    obj_0 = (double *) Malloc( n*sizeof( double ) );
    for( i = 0; i < n; i++ )
        obj_0[i] = pareto_front[i]->objective_values[0];
    sorted = mergeSort( obj_0, n );

    area = (max_0 - fmin(max_0, obj_0[sorted[n-1]])) * (max_1 - fmin(max_1, pareto_front[sorted[n-1]]->objective_values[1]));
    for( i = n-2; i >= 0; i-- )
        area += (fmin(max_0, obj_0[sorted[i+1]]) - fmin(max_0, obj_0[sorted[i]])) * (max_1-fmin(max_1, pareto_front[sorted[i]]->objective_values[1]));

    free( obj_0 );
    free( sorted );

    return area;
  }

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Individuals -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  individual* initializeIndividual( void )
  {
      individual* new_individual;

      new_individual = (individual*) Malloc(sizeof(individual));
      new_individual->parameters = (double*) Malloc( number_of_parameters*sizeof( double ) );
      new_individual->objective_values = (double*) Malloc( number_of_objectives*sizeof( double ) );
      new_individual->constraint_value = 0;
      new_individual->NIS = 0;
      new_individual->parameter_sum = 0;
      new_individual->cluster_number = 0;
      return( new_individual );
  }
  
  individual* initializeIndividual(const hicam::solution_t & sol )
  {
    individual* new_individual;
    
    new_individual = (individual*) Malloc(sizeof(individual));
    new_individual->parameters = (double*) Malloc( number_of_parameters*sizeof( double ) );
    new_individual->objective_values = (double*) Malloc( number_of_objectives*sizeof( double ) );
    new_individual->constraint_value = 0;
    new_individual->NIS = 0;
    new_individual->parameter_sum = 0;
    new_individual->cluster_number = 0;
    
    for(int j = 0; j < number_of_parameters; ++j) {
      new_individual->parameters[j] = sol.param[j];
    }
    
    for(int j = 0; j < number_of_objectives; ++j) {
      new_individual->objective_values[j] = sol.obj[j];
    }
    
    new_individual->constraint_value = sol.constraint;
    new_individual->cluster_number = sol.cluster_number;
    
    return( new_individual );
  }
  
  hicam::solution_pt IndividualToSol(const individual* id)
  {
    
    hicam::solution_pt sol = std::make_shared<hicam::solution_t>();
    
    sol->rank = -1;
    sol->elite_origin = nullptr;
    sol->population_number  = -1;
    sol->dvis.resize(10,0);
    sol->current_batch = 0;
    sol->batch_size = -1;
    
    sol->param.resize(gomea::number_of_parameters);
    for(int j = 0; j < gomea::number_of_parameters; ++j) {
      sol->param[j] = id->parameters[j];
    }
    
    sol->obj.resize(gomea::number_of_objectives);
    for(int j = 0; j < gomea::number_of_objectives; ++j) {
      sol->obj[j] = id->objective_values[j];
    }
    
    sol->constraint = id->constraint_value;
    sol->cluster_number = id->cluster_number;
    
    return sol;
  }
  

  void ezilaitiniIndividual( individual *ind )
  {
      free( ind->objective_values );
      free( ind->parameters );
      free( ind );
  }

  void copyIndividual( individual *source, individual *destination )
  {
      int i;
      for( i = 0; i < number_of_parameters; i++ )
          destination->parameters[i] = source->parameters[i];
      for( i = 0; i < number_of_objectives; i++ )
          destination->objective_values[i] = source->objective_values[i];
      destination->constraint_value = source->constraint_value;
      destination->parameter_sum = source->parameter_sum;
  }

  void copyIndividualWithoutParameters( individual *source, individual *destination )
  {
      int i;
      for( i = 0; i < number_of_objectives; i++ )
          destination->objective_values[i] = source->objective_values[i];
      destination->constraint_value = source->constraint_value;
      destination->parameter_sum = source->parameter_sum;
  }

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Ezilaitini -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  void ezilaitiniProblem( void )
  {
      return;
  }
}
