#define _CRT_SECURE_NO_WARNINGS


#include "mohillvallea/hicam_external.h"
#include "gomea/MO-RV-GOMEA.h"
#include "../benchmark_functions/mo_benchmarks.h"

/**
* Parses only the options from the command line.
*/

bool write_generational_statistics;
bool write_generational_solutions;
bool print_verbose_overview;
bool use_vtr;
int version;
int local_optimizer_index;
std::string write_directory;
std::string patfile;

size_t  problem_index,
        number_of_parameters,
        number_of_objectives;

size_t  elitist_archive_size_target,
        approximation_set_size;
unsigned int number_of_subgenerations_per_population_factor,
        maximum_number_of_populations,
        maximum_number_of_seconds;
double  lower_user_range,
        upper_user_range,
        vtr;
int base_population_size;

unsigned int maximum_number_of_evaluations;
int random_seed;

hicam::fitness_pt fitness_function;

double HL_tol;


void printUsage(void)
{
  printf("Usage: MMMO [-?] [-P] [-s] [-w] [-v] [-r] [-V] pro dim loc low upp pops pop ela appr eva vtr sec subg rnd wrp\n");
  printf(" -?: Prints out this usage information.\n");
  printf(" -P: Prints out a list of all installed optimization problems.\n");
  printf(" -s: Enables computing and writing of statistics every generation.\n");
  printf(" -w: Enable writing of solutions and their fitnesses every generation.\n");
  printf(" -v: Verbose mode. Prints the settings before starting the run + ouput of generational statistics to terminal.\n");
  printf(" -r: Enables use of vtr in termination condition (value-to-reach).\n");
  printf(" -V: Which version to run.\n");
  printf("\n");
  printf("  pro: Index of optimization problem to be solved (minimization).\n");
  printf("  dim: Number of parameters.\n");
  printf("  loc: local optimizer index.\n");
  printf("  low: Overall initialization lower bound.\n");
  printf("  upp: Overall initialization upper bound.\n");
  printf(" pops: Number of populations.\n");
  printf("  pop: Population size (initial in case of pops = 1).\n");
  printf("  ela: Elitist archive target size.\n");
  printf(" appr: Approximation set size.\n");
  printf("  eva: Maximum number of evaluations allowed.\n");
  printf("  vtr: The value to reach. If the objective value of the best feasible solution reaches\n");
  printf("       this value, termination is enforced (if -r is specified).\n");
  printf("  sec: Time limit in seconds.\n");
  printf(" subg: Number of subgenerations in the interleaved multistart scheme (>1).\n");
  printf("  rnd: Random seed.\n");
  printf("HLtol: Tolerance for HL filtering > 0.\n");
  printf("  wrp: write path.\n");
  exit(0);

}
 
/**
* Returns the problems installed.
*/
void printAllInstalledProblems(void)
{
  int i = 0; 

  hicam::fitness_pt objective = getObjectivePointer(i);

  std::cout << "Installed optimization problems:\n";

  while(objective != nullptr)
  {
    std::cout << std::setw(3) << i << ": " << objective->name() << std::endl;

    i++;
    objective = getObjectivePointer(i);
  }

  exit(0);
}

/**
* Informs the user of an illegal option and exits the program.
*/
void optionError(char **argv, int index)
{
  printf("Illegal option: %s\n\n", argv[index]);

  printUsage();
}

void parseVersion(int *index, int argc, char** argv)
{
  short noError = 1;

  (*index)++;
  noError = noError && sscanf(argv[*index], "%d", &version);

  if (!noError)
  {
    printf("Error parsing version parameter.\n\n");

    printUsage();
  }
}

void parsePatfile(int *index, int argc, char** argv)
{
  (*index)++;
  patfile = argv[*index];
}


void parseOptions(int argc, char **argv, int *index)
{
  double dummy;

  write_generational_statistics = 0;
  write_generational_solutions = 0;
  print_verbose_overview = 0;
  use_vtr = 0;
  version = 99; // default, run the latest version default

  for (; (*index) < argc; (*index)++)
  {
    if (argv[*index][0] == '-')
    {
      /* If it is a negative number, the option part is over */
      if (sscanf(argv[*index], "%lf", &dummy) && argv[*index][1] != '\0')
        break;

      if (argv[*index][1] == '\0')
        optionError(argv, *index);
      else if (argv[*index][2] != '\0')
        optionError(argv, *index);
      else
      {
        switch (argv[*index][1])
        {
        case '?': printUsage(); break;
        case 'P': printAllInstalledProblems(); break;
        case 's': write_generational_statistics = 1; break;
        case 'w': write_generational_solutions = 1; break;
        case 'v': print_verbose_overview = 1; break;
        case 'r': use_vtr = 1; break;
        case 'V': parseVersion(index, argc, argv); break;
        case 'p': parsePatfile(index, argc, argv); break;
        default: optionError(argv, *index);
        }
      }
    }
    else /* Argument is not an option, so option part is over */
      break;
  }

}

void parseParameters(int argc, char **argv, int *index)
{
  int noError;

  if ((argc - *index) != 16)
  {
    printf("Number of parameters is incorrect, require 16 parameters (you provided %d).\n\n", (argc - *index));

    printUsage();
  }

  noError = 1;
  noError = noError && sscanf(argv[*index + 0], "%zu", &problem_index);
  noError = noError && sscanf(argv[*index + 1], "%zu", &number_of_parameters);
  noError = noError && sscanf(argv[*index + 2], "%d", &local_optimizer_index);
  noError = noError && sscanf(argv[*index + 3], "%lf", &lower_user_range);
  noError = noError && sscanf(argv[*index + 4], "%lf", &upper_user_range);
  noError = noError && sscanf(argv[*index + 5], "%d", &maximum_number_of_populations);
  noError = noError && sscanf(argv[*index + 6], "%d", &base_population_size);
  noError = noError && sscanf(argv[*index + 7], "%zu", &elitist_archive_size_target);
  noError = noError && sscanf(argv[*index + 8], "%zu", &approximation_set_size);
  noError = noError && sscanf(argv[*index + 9], "%d", &maximum_number_of_evaluations);
  noError = noError && sscanf(argv[*index + 10], "%lf", &vtr);
  noError = noError && sscanf(argv[*index + 11], "%d", &maximum_number_of_seconds);
  noError = noError && sscanf(argv[*index + 12], "%d", &number_of_subgenerations_per_population_factor);
  noError = noError && sscanf(argv[*index + 13], "%d", &random_seed);
  noError = noError && sscanf(argv[*index + 14], "%lf", &HL_tol);
  write_directory = argv[*index + 15];
  
  if (!noError)
  {
    printf("Error parsing parameters.\n\n");
    printUsage();
  }

  fitness_function = getObjectivePointer((int) problem_index);
}

void checkOptions(void)
{
  if (number_of_parameters < 1)
  {
    printf("\n");
    printf("Error: number of parameters < 1 (read: %d). Require number of parameters >= 1.", (int) number_of_parameters);
    printf("\n\n");

    exit(0);
  }

  if (approximation_set_size < 1)
  {
    printf("\n");
    printf("Error: approximation set size < 1 (read: %zu).", approximation_set_size);
    printf("\n\n");

    exit(0);
  }

  
  if (problem_index < 100 && getObjectivePointer((int) problem_index) == nullptr)
  {
    printf("\n");
    printf("Error: unknown index for problem (read index %d).", (int) problem_index);
    printf("\n\n");

    exit(0);
  }

}



void parseCommandLine(int argc, char **argv)
{
  int index;

  index = 1;

  parseOptions(argc, argv, &index);

  parseParameters(argc, argv, &index);
}

void interpretCommandLine(int argc, char **argv)
{

  parseCommandLine(argc, argv);
  number_of_objectives = fitness_function->number_of_objectives;
  checkOptions();
}


int main(int argc, char **argv)
{

  interpretCommandLine(argc, argv);

  // Objective function settings
  //--------------------------------------------------------------------------------------------
  // std::cout << "Problem: " << objective->name() << std::endl;

  fitness_function->set_number_of_parameters(number_of_parameters);

  hicam::vec_t lower_init_ranges(fitness_function->get_number_of_parameters(), lower_user_range);
  hicam::vec_t upper_init_ranges(fitness_function->get_number_of_parameters(), upper_user_range);

  std::string file_appendix = "_" + fitness_function->name() + "_" + std::to_string(random_seed);

  bool print_generational_statistics = print_verbose_overview;
  
  // elitist_archive_size_target = 10*approximation_set_size;
  // approximation_set_size = elitist_archive_size_target;
  
  if(version == 1000)
  {

    int base_number_of_mixing_components = 0;
    gomea::initGomea(
     fitness_function,
     lower_init_ranges,
     upper_init_ranges,
     vtr,
     use_vtr,
     version,
     local_optimizer_index,
     HL_tol,
     elitist_archive_size_target,
     approximation_set_size,
     maximum_number_of_populations,
     base_population_size,
     base_number_of_mixing_components,
     number_of_subgenerations_per_population_factor,
     maximum_number_of_evaluations,
     maximum_number_of_seconds,
     random_seed,
     write_generational_solutions,
     write_generational_statistics,
     print_generational_statistics,
     write_directory,
     file_appendix,
     print_verbose_overview
    );
    
    gomea::run();
    
  }
  else
  {
    hicam::recursion_scheme_t opt(
      fitness_function,
      lower_init_ranges,
      upper_init_ranges,
      vtr,
      use_vtr,
      version,
      local_optimizer_index,
      HL_tol,
      elitist_archive_size_target,
      approximation_set_size,
      maximum_number_of_populations,
      base_population_size,
      number_of_subgenerations_per_population_factor,
      maximum_number_of_evaluations,
      maximum_number_of_seconds,
      random_seed,
      write_generational_solutions,
      write_generational_statistics,
      print_generational_statistics,
      write_directory,
      file_appendix,
      print_verbose_overview
    );
    opt.run();
  
    /* std::cout << "old_front = [\n";
    for(size_t i = 0; i < opt.elitist_archive->sols.size(); ++i)
    {
      if(opt.elitist_archive->sols[i] != nullptr)
      {
        std::cout << lci(opt.elitist_archive->sols[i]->dvis)
        << " " << lsi(opt.elitist_archive->sols[i]->dvis)
        << "  " << opt.elitist_archive->sols[i]->dvis
        << std::endl;
      }
    }
    std::cout << "];\n";
    */
  }
  
  return(0);
}
