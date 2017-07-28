// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: vramasub / Anyone is free to add their own pair potentials here

#ifndef __GRID_PAIR_POTENTIALS__H__
#define __GRID_PAIR_POTENTIALS__H__

#include "GridPotentialPair.h"
#include "hoomd/md/EvaluatorPairLJ.h"
#include "hoomd/md/EvaluatorPairEwald.h"
#include "hoomd/md/EvaluatorPairSLJ.h"
#include "hoomd/md/EvaluatorPairSLJ.h"
#include "hoomd/md/EvaluatorPairForceShiftedLJ.h"

/*! \file AllGridPairPotentials.h
    \brief Handy list of typedefs for all of the templated pair potentials in hoomd
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Pair potential force compute for lj forces
typedef solvent::GridPotentialPair<EvaluatorPairLJ> GridPotentialPairLJ;
//! Pair potential force compute for slj forces
typedef solvent::GridPotentialPair<EvaluatorPairSLJ> GridPotentialPairSLJ;
//! Pair potential force compute for ewald forces
typedef solvent::GridPotentialPair<EvaluatorPairEwald> GridPotentialPairEwald;
//! Pair potential force compute for force shifted LJ on the GPU
typedef solvent::GridPotentialPair<EvaluatorPairForceShiftedLJ> GridPotentialPairForceShiftedLJ;

#endif // __GRID_PAIR_POTENTIALS_H__
