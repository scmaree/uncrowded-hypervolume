#pragma once

/*

HICAM Multi-objective

By S.C. Maree, 2018
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include <string>
#include <vector>
#include <ctime>
#include <algorithm>
#include <memory>
#include <map>
#include <math.h>
#include <random>
#include <functional>
#include <iostream>
#include <sstream>
#include <assert.h>
#include <fstream>
#include <iomanip>
#include <ctype.h>
#include <list>
#include <cstdlib>


/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Constants -=-=-=-=-=-=-=-=-=-=-=-=-=-*/
#ifndef PI
#define PI 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798
#endif
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace hicam {


  class recursion_scheme_t;
  typedef std::shared_ptr<recursion_scheme_t> recursion_scheme_pt;

  class elitist_archive_t;
  typedef std::shared_ptr<elitist_archive_t> elitist_archive_pt;

  class optimizer_t;
  typedef std::shared_ptr<optimizer_t> optimizer_pt;

  class population_t;
  typedef std::shared_ptr<population_t> population_pt;
  
  class solution_t;
  typedef std::shared_ptr<solution_t> solution_pt;

  typedef std::mt19937 rng_t;
  typedef std::shared_ptr<std::mt19937> rng_pt;

  class fitness_t;
  typedef std::shared_ptr<fitness_t> fitness_pt;
  
  class edge_t;
  typedef std::shared_ptr<edge_t> edge_pt;
  
  // clusters
  class cluster_t;
  typedef std::shared_ptr<cluster_t> cluster_pt;

  class amalgam_t;
  typedef std::shared_ptr<amalgam_t> amalgam_pt;
  
  class iamalgam_t;
  typedef std::shared_ptr<iamalgam_t> iamalgam_pt;
  
  class hvc_t;
  typedef std::shared_ptr<hvc_t> hvc_pt;
  
}
