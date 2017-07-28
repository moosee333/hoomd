// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "GridData.h"
#include "GridPotentialPair.h"
#include "AllGridPairPotentials.h"

//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
    create the hoomd python module and define the exports here.
*/

using namespace solvent;

PYBIND11_PLUGIN(_md)
    {
    pybind11::module m("_solvent");

    export_GridData(m);
    export_GridPotentialPair<GridPotentialPairLJ>(m, "GridPotentialPairLJ");
    export_GridPotentialPair<GridPotentialPairSLJ>(m, "GridPotentialPairSLJ");
    export_GridPotentialPair<GridPotentialPairEwald>(m, "GridPotentialPairEwald");
    export_GridPotentialPair<GridPotentialPairForceShiftedLJ>(m, "GridPotentialPairForceShiftedLJ");
    export_SnapshotGridData<float>(m);
    export_SnapshotGridData<double>(m);

    return m.ptr();
    }
