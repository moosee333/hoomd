// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: vramasub

#include "LevelSetSolver.h"
#include "hoomd/extern/num_util.h"

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
      m_grid(grid)
    { }

//! Destructor
LevelSetSolver::~LevelSetSolver()
    { }

void LevelSetSolver::initializeGrid()
    {
    if (m_need_init_grid)
        {
        unsigned int n_elements = m_dim.x*m_dim.y*m_dim.z;
        GPUArray<Scalar> phi(n_elements, m_exec_conf);
        m_phi.swap(phi);

        GPUArray<Scalar> fn(n_elements, m_exec_conf);
        m_fn.swap(fn);
        m_need_init_grid = false;
        }
    }

void export_LevelSetSolver(py::module& m)
    {
    pybind11::class_<LevelSetSolver, std::shared_ptr<LevelSetSolver> >(m, "LevelSetSolver")
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<GridData>>());
    }

} // end namespace solvent
