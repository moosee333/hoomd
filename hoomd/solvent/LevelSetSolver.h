// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: vramasub

/*! \file LevelSetSolver.h
    \brief Defines the LevelSetSolver class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <memory>

#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/SystemDefinition.h"
#include "hoomd/GPUArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#include "GridData.h"
#include "GridForceCompute.h"
#include "hoomd/ForceCompute.h"

#ifndef __LEVEL_SET_SOLVER_H__
#define __LEVEL_SET_SOLVER_H__

namespace solvent
{

/*! This class implements the core of the hoomd solvent package, the level set method used
    to solve for the system energies. The core of the variational implicit solvent model 
    lives here.

    This class stores a GridData instance and uses the SparseFieldUpdater to construct the 
    layers of the sparse field on that grid. It then uses the FastMarcher class to compute 
    the values of the phi grid. The core solver logic encapsulated in this class is the 
    utilization of the resulting phi grid and the fn grid computed using various GridForceComputes
    to determine the location of the zero level set, which corresponds to the interface.

    */
class LevelSetSolver : public ForceCompute
    {
    public:
        //! Constructor
        /* \param sysdef The HOOMD system definition
         * \param grid the grid we'll be solving on
         */
        LevelSetSolver(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<GridData> grid);

        //! Destructor
        virtual ~LevelSetSolver();

        //! Add a GridForceCompute to the list
        virtual void addGridForceCompute(std::shared_ptr<GridForceCompute> gfc);

        //! Actually compute the forces
        /*! In this class, the forces are computed by simply summing the forces due
            to each of the GridForceComputes that are added to the class
         */
        virtual void computeForces(unsigned int timestep);

        //! Return the grid instance associated with this class
        std::shared_ptr<GridData> getGridData() const
            {
            return m_grid;
            }

    protected:
        std::shared_ptr<SystemDefinition> m_sysdef; //!< HOOMD system definition
        std::shared_ptr<ParticleData> m_pdata;               //!< HOOMD particle data
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration

        std::vector< std::shared_ptr<GridForceCompute> > m_grid_forces;    //!< List of all the grid force computes
        std::shared_ptr<GridData> m_grid; //!< The grid data object

    };

//! Export LevelSetSolver to python
void export_LevelSetSolver(pybind11::module& m);

} // end namespace solvent

#endif // __LEVEL_SET_SOLVER_H__
