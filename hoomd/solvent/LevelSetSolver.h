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

        //! Return the grid spacing along every dimension
        Scalar3 getSpacing()
            {
            initializeGrid(); // initialize grid if necessary

            Scalar3 L = m_pdata->getBox().getL();
            return make_scalar3(L.x/m_dim.x, L.y/m_dim.y, L.z/m_dim.z);
            }

        //! Set the maximum grid spacing
        void setSigma(Scalar sigma)
            {
            m_sigma = sigma;
            m_need_init_grid = true;
            }

        //! Get the current sigma
        Scalar getSigma() const
            {
            return m_sigma;
            }

    protected:
        //! Helper function to re-initialize the grid when necessary
        void initializeGrid();

        std::shared_ptr<SystemDefinition> m_sysdef; //!< HOOMD system definition
        std::shared_ptr<ParticleData> m_pdata;               //!< HOOMD particle data
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration

        Scalar m_sigma;     //!< The maximum grid spacing along each axis
        uint3 m_dim;         //!< The current grid dimensions

        bool m_need_init_grid;  //!< True if we need to re-initialize the grid

        GPUArray<Scalar> m_phi; //!< The phi grid, of dimensions m_dim
        GPUArray<Scalar> m_fn;  //!< The velocity grid

        Index3D m_indexer;      //!< The grid indexer

        std::shared_ptr<GridData> m_grid; //!< The grid data object

    };

//! Export LevelSetSolver to python
void export_LevelSetSolver(pybind11::module& m);

} // end namespace solvent

#endif // __LEVEL_SET_SOLVER_H__
