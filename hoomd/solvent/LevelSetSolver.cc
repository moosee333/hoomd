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
LevelSetSolver::LevelSetSolver(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<GridData> grid)
    : ForceCompute(sysdef),
      m_sysdef(sysdef),
      m_pdata(sysdef->getParticleData()),
      m_exec_conf(m_pdata->getExecConf()),
      m_updater(new SparseFieldUpdater(m_sysdef, grid)),
      m_marcher(new FastMarcher(m_sysdef, m_updater)),
      m_grid(grid)
    { }

//! Destructor
LevelSetSolver::~LevelSetSolver()
    { }

void LevelSetSolver::addGridForceCompute(std::shared_ptr<GridForceCompute> gfc)
    {
    assert(gfc);
    m_grid_forces.push_back(gfc);
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

    // Once the initial grid is established we compute the numerical derivatives
    /*
     * The next steps once the distances are computed is to actually compute the move.
     * For that, we need to do a number of things
     * I'm breaking up the computation like the VISM paper, so I need A and B terms separately
     */
    computeA();
    //computeB();
    }

void LevelSetSolver::computeA()
    {
    /*
     * This function needs to compute the curvatures, etc
     */ 
    // Use the gradient to find the direction to the boundary, then multiply by the phi grid's value to compute the vector
    const std::vector<std::vector<uint3> > layers = m_updater->getLayers();
    const std::map<char, char> layer_indexer = m_updater->getIndex();

    const std::vector<uint3> Lz = layers[layer_indexer.find(0)->second];
    Scalar missing_value = 0; // The grid value that indicates that a cell's distance has not yet been finalized

    GPUArray<Scalar> divx(Lz.size(), m_exec_conf), divy(Lz.size(), m_exec_conf), divz(Lz.size(), m_exec_conf); 
    ArrayHandle<Scalar> h_divx(divx, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_divy(divy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_divz(divz, access_location::host, access_mode::readwrite);

    std::vector<Scalar3> boundary_vecs;
    m_grid->grad(divx, divy, divz, Lz);
    m_grid->getMeanCurvature(Lz);
    }

void export_LevelSetSolver(py::module& m)
    {
    pybind11::class_<LevelSetSolver, std::shared_ptr<LevelSetSolver> >(m, "LevelSetSolver", pybind11::base<ForceCompute>())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<GridData> >())
        .def("addGridForceCompute", &LevelSetSolver::addGridForceCompute)
    ;
    }

} // end namespace solvent
