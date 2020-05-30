#pragma once

/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

*/

#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <random>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <functional>
#include <memory>

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Constants -=-=-=-=-=-=-=-=-=-=-=-=-=-*/
#ifndef PI
#define PI 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798
#endif
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace hillvallea {

  class solution_t;
  class population_t;
  class edge_t;
  class node_t;
  class hillvallea_t;
  class optimizer_t;
  class amalgam_t;
  class hgml_t;
  class amalgam_univariate_t;
  class iamalgam_t;
  class cmsaes_t;
  class sep_cmaes_t;
  class cmaes_t;
  class iamalgam_univariate_t;
  class gomea_t;
  class vec_t;
  class FOS_t;
  class FOS_element_t;
  class adam_t;

  typedef std::shared_ptr<solution_t> solution_pt;
  typedef std::shared_ptr<population_t> population_pt;
  typedef std::shared_ptr<edge_t> edge_pt;
  typedef std::shared_ptr<node_t> node_pt;
  typedef std::shared_ptr<hillvallea_t> hillvallea_pt;
  typedef std::shared_ptr<optimizer_t> optimizer_pt;
  typedef std::shared_ptr<amalgam_t> amalgam_pt;
  typedef std::shared_ptr<cmsaes_t> cmsaes_pt;
  typedef std::shared_ptr<cmaes_t> cmaes_pt;
  typedef std::shared_ptr<sep_cmaes_t> sep_cmaes_pt;
  typedef std::shared_ptr<hgml_t> hgml_pt;
  typedef std::shared_ptr<FOS_t> FOS_pt;
  typedef std::shared_ptr<FOS_element_t> FOS_element_pt;
  typedef std::shared_ptr<amalgam_univariate_t> amalgam_univariate_pt;
  typedef std::shared_ptr<iamalgam_t> iamalgam_pt;
  typedef std::shared_ptr<gomea_t> gomea_pt;
  typedef std::shared_ptr<adam_t> adam_pt;
  typedef std::shared_ptr<iamalgam_univariate_t> iamalgam_univariate_pt;
  typedef std::mt19937 rng_t;
  typedef std::shared_ptr<std::mt19937> rng_pt;

  class fitness_t;
  typedef std::shared_ptr<fitness_t> fitness_pt;
  
}

