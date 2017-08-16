// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: vramasub

/*! \file FastMarcher.h
    \brief Defines the FastMarcher class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <memory>
#include <limits>

#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/SystemDefinition.h"
#include "hoomd/HOOMDMath.h"

#include "SparseFieldUpdater.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __FAST_MARCHER_H__
#define __FAST_MARCHER_H__

namespace solvent
{

/*! This class implements the fast marching method for calculating distances from an 
    interface defined via a level set. It is used in the computations of the variational 
    implicit solvent model. The class operates directly on a sparse field (maintained 
    by the SparseFieldUpdater), computing distances based on the sparse field's grid 
    and marching according to the cells in the layers built by the sparse field.

    The logic encapsulated in this class is based on the fast marching method, which
    generalizes Dijkstra's method for computing shortest paths on DAGs to use a Euclidean
    metric instead of the Manhattan distance used in Dijkstra's algorithm (the lengths of
    the graph edges). In general, the method provides a method to solve the Eikonal
    equation by tracking the motion of a wave on a grid; in our case, we will be using it
    to track the position of various grid points relative to the interface. The best
    approximation for these distances to the interface is provided by starting with some
    known set of grid points that sit on the interface as well as some guess of the distance
    to each of the remaining grid points. The algorithm then iteratively updates the set of
    grid points with finalized distances by accepting the closest remaining uncertain point,
    and then updating its neighbors tentative distances based on this newly finalized point.
    This process concludes when all desired cells have been updated. Rather than operating
    on the entire grid, which is prohibitively expensive, only a minimal set of cells (as
    defined by the SparseFieldUpdater) is maintained.
    */
class FastMarcher
    {
    public:
        //! Constructor
        /* \param field The Sparse Field on which we march
         */
        FastMarcher(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<SparseFieldUpdater> field);

        //! Destructor
        virtual ~FastMarcher();

        //! Compute distances
        void march();

    protected:

        std::shared_ptr<SystemDefinition> m_sysdef; //!< HOOMD system definition
        std::shared_ptr<ParticleData> m_pdata;               //!< HOOMD particle data (required for box)
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
        std::shared_ptr<SparseFieldUpdater> m_field; //!< Sparse field
        std::shared_ptr<solvent::GridData> m_grid; //!< The grid that layers are maintained on

        //! Compute distances
        void estimateL0Distances();

    private:
		//! Simple helper function to compute the sign
		template <class Real> 
		inline int sgn(Real num) 
            {
			return (num > Real(0)) - (num < Real(0));
            }

    };

} // end namespace solvent

#endif // __FAST_MARCHER_H__
