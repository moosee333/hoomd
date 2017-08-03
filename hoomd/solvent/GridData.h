// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file GridData.h
    \brief Defines the GridData class and associated utilities
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

#ifndef __GRID_DATA_H__
#define __GRID_DATA_H__

namespace solvent
{

template<class Real>
struct SnapshotGridData;

/*! This class implements a storage for scalars on a 3D grid. It is used in the computations
    of the variational implicit solvent model.

    In the solvent model, an instantaenous location of the interface is maintained through
    an implicitly defined surface as the zero level set of a function phi: IR^3 -> IR.
    It is iteratively updated using another grid of scalar values, f_n, that define normal
    velocities for the propagation of the interface on every grid point. The function defining
    the implicit surface, phi, and the precomputed velocities, f_n, are variables in the level
    set equation.  Within every HOOMD time step, multiple iterations of the level set methods are
    performed until convergence.

    The class design is as follows: this class, GridData, holds the data structures necessary
    for the grid, phi, and f_n, to update it. Other classes may access those data through
    its public interface.

    In particular, phi will be initialized as a signed distance function to the interface,
    that is, every cell of phi holds the closest distance to its zero set with the sign
    indicating if the point is inside (-) or outside (+) the boundary. The choice of sign
    here is purely by convention. Initializaion is the task of a field updater class (such as,
    SparseFieldUpdater). The important optimization here is that updates are only performed
    on a thin layer around the zero or 'active set'.

    The GridData class is used as a base data structure for iteratively finding the solutions to
    the level set equation, the LevelSetSolver class.

    The internal data is stored as GPUArrays to maintain a consistent public interface that supports
    both CPU and GPU data storage.

    For most purposes, we want to maintain an approximately constant grid spacing along every
    direction. Hence we determine the number of cells along each dimension by rounding
    to the nearest integer muliple of the box dimensions divided by the grid spacing.
    The number of grid cells is automatically updated when the box changes.

    For interfacing with python, the class also a thin wrapper around a copy of the
    arrays, a SnapshotGridData, that can be acccess via numpy.
    */
class GridData
    {
    public:
        //! Constructor
        /* \param sysdef The HOOMD system definition
         * \param sigma the maximum requested grid spacing along every dimension
         */
        GridData(std::shared_ptr<SystemDefinition> sysdef, Scalar sigma);

        //! Destructor
        virtual ~GridData();

        //! Returns the distance function grid
        const GPUArray<Scalar>& getPhiGrid()
            {
            initializeGrid(); // initialize grid if necessary
            return m_phi;
            }

        //! Returns the velocity grid
        const GPUArray<Scalar>& getVelocityGrid()
            {
            initializeGrid(); // initialize grid if necessary
            return m_fn;
            }

        //! Returns a snapshot of the grid arrays
        template<class Real>
        std::shared_ptr<SnapshotGridData<Real> > takeSnapshot();

        //! Function to set grid values, primarily useful for initialization
        /*! \param flags Bits indicating which grid to update (1 for energies, 2 for forces, 3 for both)
            \param value The value to set the grid to
         */
        void setGridValues(unsigned int flags = 0, double value = 0.0);

        //! Return the grid spacing along every dimension
        Scalar3 getSpacing()
            {
            initializeGrid(); // initialize grid if necessary

            Scalar3 L = m_pdata->getBox().getL();
            return make_scalar3(L.x/m_dim.x, L.y/m_dim.y, L.z/m_dim.z);
            }

        //! Return the dimensions of the grid
        uint3 getDimensions()
            {
            initializeGrid();

            return m_dim;
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

        Index3D getIndexer() const
            {
            return m_indexer;
            }

        //! \name Enumerations
        //@{

        //! Simple enumeration of flags to employ for working with forces, energies, or both
        enum flags
            {
            energies = 1,
            forces = 2
            };

    protected:
        //! Helper function to re-initialize the grid when necessary
        void initializeGrid();

        //! Signal slot for box changes
        void setBoxChanged()
            {
            m_need_init_grid = true;
            }

        //! Helper function to compute grid dimensions
        void computeDimensions();

        std::shared_ptr<SystemDefinition> m_sysdef; //!< HOOMD system definition
        //std::shared_ptr<ExecutionConfiguration> m_exec_conf; //!< HOOMD execution configuration
        std::shared_ptr<ParticleData> m_pdata;               //!< HOOMD particle data (required for box)
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration

        Scalar m_sigma;     //!< The maximum grid spacing along each axis
        uint3 m_dim;         //!< The current grid dimensions

        bool m_need_init_grid;  //!< True if we need to re-initialize the grid

        GPUArray<Scalar> m_phi; //!< The phi grid, of dimensions m_dim
        GPUArray<Scalar> m_fn;  //!< The velocity grid

        Index3D m_indexer;      //!< The grid indexer

    };

template<class Real>
struct SnapshotGridData
    {
    SnapshotGridData(unsigned int n_elements, uint3 dim)
        : m_dim(make_uint3(dim.x, dim.y, dim.z))
        {
        phi.resize(n_elements,Scalar(0.0));
        fn.resize(n_elements,Scalar(0.0));
        }

    //! Returns the distance function grid as a Numpy array
    pybind11::object getPhiGridNP() const;

    //! Returns the velocity grid
    pybind11::object getVelocityGridNP() const;
    
    std::vector<Real> phi;
    std::vector<Real> fn;
    uint3 m_dim;         //!< The grid dimensions of the underlying grid
    };

//! Export SnapshotGridData to python
void export_SnapshotGridData(pybind11::module& m);

//! Export GridData to python
void export_GridData(pybind11::module& m);

} // end namespace solvent

#endif // __GRID_DATA_H__
