// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: vramasub

#include "LevelSetSolver.h"

using namespace std;
namespace py = pybind11;

/*! \file LevelSetSolver.cc
    \brief Contains code for the LevelSetSolver class
*/

namespace solvent
{

//! Constructor
LevelSetSolver::LevelSetSolver(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<GridData> grid, bool ignore_zero)
    : ForceCompute(sysdef),
      m_sysdef(sysdef),
      m_pdata(sysdef->getParticleData()),
      m_exec_conf(m_pdata->getExecConf()),
      m_updater(new SparseFieldUpdater(m_sysdef, grid, ignore_zero)),
      m_marcher(new FastMarcher(m_sysdef, m_updater)),
      m_grid(grid)
    { }

//! Destructor
LevelSetSolver::~LevelSetSolver()
    { }

/*! \param gfc GridForceCompute to add
*/
void LevelSetSolver::addGridForceCompute(std::shared_ptr<GridForceCompute> gfc)
    {
    assert(gfc);
    m_grid_forces.push_back(gfc);
    }

void export_LevelSetSolver(py::module& m)
    {
    pybind11::class_<LevelSetSolver, std::shared_ptr<LevelSetSolver> >(m, "LevelSetSolver", pybind11::base<ForceCompute>())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<GridData>, bool>())
        .def("addGridForceCompute", &LevelSetSolver::addGridForceCompute)
    ;
    }

void LevelSetSolver::computeForces(unsigned int timestep)
    {
    // We need to precompute the energy for each of grid forces before performing any level set operations
	for(std::vector<std::shared_ptr<GridForceCompute> >::iterator grid_force = m_grid_forces.begin(); grid_force != m_grid_forces.end(); ++grid_force)
        {
		// (*grid_force)->setGrid(m_grid); // Happens on the python side.
		(*grid_force)->compute(timestep);
        }

    // Utilize the SparseFieldUpdater to keep track of the level sets of interest
    m_updater->clearField();
    m_updater->computeInitialField();

    // Now use the fast marcher to compute the distances
    m_marcher->march();
    }

} // end namespace solvent
