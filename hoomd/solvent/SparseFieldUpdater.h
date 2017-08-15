// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: vramasub

/*! \file SparseFieldUpdaterUpdater
 *
    \brief Defines the SparseFieldUpdater class and associated utilities
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <memory>

#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/SystemDefinition.h"
#include "hoomd/HOOMDMath.h"

#include "GridData.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __SPARSE_FIELD_UPDATER_H__
#define __SPARSE_FIELD_UPDATER_H__

namespace solvent
{

/*! This class is responsible for constructing the sparse field associated with a level set
    on a grid. It is used in the computations of the variational implicit solvent model (VISM).
    The interface is maintained as the zero level set of a function phi. To track the evolution
    of this function through time, we could apply a simple Euler stepping scheme and maintain
    the value of phi on the full gridded space. However, this is extremely expensive; instead,
    we only maintain the values of phi on a small set of grid points directly adjacent to the
    interface. The size of this set is governed by the set of derivatives that must be
    calculated at any point; in order to compute first derivatives of phi, for example, we
    require at least one layer of cells on either side of the actual grid cells encompassing
    the interface. For the VISM, we need second derivatives to find curvature, so we will
    generally require two layers.

    This class operates on a GridData object owned by a LevelSetSolver; it should never be
    instantiated independently of a LevelSetSolver. The interface position is determined by 
    looking for sign flips in the energies on grid, and additional layers are added by simply 
    adding neighbors of the zero level set. These data structures are then made accessible for
    other classes (e.g. the FastMarcher) to adapt.

    */
class SparseFieldUpdater 
    {
    public:
        //! Constructor
        /*
         * \param sysdef The HOOMD system definition
         * \param num_layers The number of layers to track
         */
        SparseFieldUpdater(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<GridData> grid, char num_layers = 2);

        //! Destructor
        virtual ~SparseFieldUpdater();

        //! Generate the sparse field lists based on the current grid
        void computeInitialField();

		//! Clear the layers; required at each timestep
        void clearField() 
            {
            for (unsigned char i = 0; i < (2*m_num_layers+1); i++)
                m_layers[i].clear();
            }

    protected:

        //! Initialize the zero layer
        void initializeLz();

        //! Initialize Ln1 and Lp1
        // Must be a special case from initializeLayer because the positive and negative first layers
        // are initialized simultaneously with signs determined by the values of the energy.
        void initializeL1();

        //! Initialize the ith layer
        /*
         * \param layer The layer number to initialize
         */
        void initializeLayer(int layer);

        std::shared_ptr<SystemDefinition> m_sysdef; //!< HOOMD system definition
        std::shared_ptr<ParticleData> m_pdata;               //!< HOOMD particle data (required for box)
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
        std::shared_ptr<solvent::GridData> m_grid; //!< The grid that layers are maintained on

        unsigned char m_num_layers; //!< The number of layers to include (e.g. two for Lz, Lp1, Ln1, Lp2, Ln2)
        std::vector<std::vector<uint3> > m_layers; //!< Sparse field layers
        
    private:

		//! Simple helper function to compute the sign
		template <class Real> 
		inline int sgn(Real num) 
            {
			return (num > Real(0)) - (num < Real(0));
            }

		//! Index positive and negative layer numbers into the std::vector of layers
		inline int get_layer_index(char num) 
            {
            return num + (char) m_num_layers;
            }

    };

}
#endif //__SPARSE_FIELD_UPDATER_H__

