#pragma once

#include "genMED.h"
#include "genMEDcurve.h"
#include "ZDT1.h"
#include "ZDT2.h"
#include "ZDT3.h"
#include "ZDT4.h"
#include "ZDT6.h"
#include "BD1.h"
#include "BD2.h"
#include "genMED_plateau.h"
#include "genMEDmm.h"
#include "TwoOnOne.h"
#include "SSUF1.h"
#include "SSUF3.h"
#include "OmniTest.h"
#include "SYMPART.h"
#include "triangles.h"
#include "circles_in_a_square.h"
#include "eye.h"
#include "elli-sep-1.h"
#include "sphere-sep-1.h"
#include "sphere-sep-1-concave.h"
#include "cigtab-sep-1.h"
#include "sphere-elli.h"
#include "sphere-rosenbrock.h"
#include "sphere-rastrigin-strong.h"
#include "sphere-rastrigin-weak.h"
#include "MinDistMM.h"
#include "wfg.h"


hicam::fitness_pt getObjectivePointer(int index)
{
  switch (index)
  {
    case  0: return(std::make_shared<hicam::ZDT1_t>());
    case  1: return(std::make_shared<hicam::ZDT2_t>());
    case  2: return(std::make_shared<hicam::ZDT3_t>());
    case  3: return(std::make_shared<hicam::ZDT4_t>());
    case  4: return(std::make_shared<hicam::ZDT6_t>());
    case  5: return(std::make_shared<hicam::BD1_t>());
    case  6: return(std::make_shared<hicam::BD2_t>());
    case  7: return(std::make_shared<hicam::genMED_t>(2.0));
    case  8: return(std::make_shared<hicam::genMED_t>(0.5));
    case  9: return(std::make_shared<hicam::genMEDmm_t>());
    case  10: return(std::make_shared<hicam::TwoOnOne_t>());
    case  11: return(std::make_shared<hicam::OmniTest_t>());
    case  12: return(std::make_shared<hicam::SYMPART_t>(1));
    case  13: return(std::make_shared<hicam::SYMPART_t>(2));
    case  14: return(std::make_shared<hicam::SYMPART_t>(3));
    case  15: return(std::make_shared<hicam::SSUF1_t>());
    case  16: return(std::make_shared<hicam::SSUF3_t>());
    case  17: return(std::make_shared<hicam::triangles_t>(4));
    case  18: return(std::make_shared<hicam::triangles_t>(3));
    case  19: return(std::make_shared<hicam::triangles_t>(2));
    case  20: return(std::make_shared<hicam::triangles_t>(1));
    case  21: return(std::make_shared<hicam::circles_t>(1));
    case  22: return(std::make_shared<hicam::eye_t>());
    case  23: return(std::make_shared<hicam::genMED_plateau_t>());
    case  24: return(std::make_shared<hicam::genMEDcurve_t>(0.5));
    case  25: return(std::make_shared<hicam::genMEDcurve_t>(2.0));
    case  26: return(std::make_shared<hicam::sphereSep1_t>());
    case  27: return(std::make_shared<hicam::elliSep1_t>());
    case  28: return(std::make_shared<hicam::cigtabSep1_t>());
    case  29: return(std::make_shared<hicam::sphereElli_t>());
    case  30: return(std::make_shared<hicam::sphereRosenbrock_t>());
    case  31: return(std::make_shared<hicam::sphereSepConcave_t>());
    case  32: return(std::make_shared<hicam::sphereRastriginWeak_t>(10));
    case  33: return(std::make_shared<hicam::sphereRastriginStrong_t>(10));
    case  34: return(std::make_shared<hicam::MinDistmm_t>());
      
    case  51: return(std::make_shared<hicam::wfg_t>(1));
    case  52: return(std::make_shared<hicam::wfg_t>(2));
    case  53: return(std::make_shared<hicam::wfg_t>(3));
    case  54: return(std::make_shared<hicam::wfg_t>(4));
    case  55: return(std::make_shared<hicam::wfg_t>(5));
    case  56: return(std::make_shared<hicam::wfg_t>(6));
    case  57: return(std::make_shared<hicam::wfg_t>(7));
    case  58: return(std::make_shared<hicam::wfg_t>(8));
    case  59: return(std::make_shared<hicam::wfg_t>(9));
  }
  
  return nullptr;
}

