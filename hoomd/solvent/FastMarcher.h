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
#include <queue>

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

        //! Propagate some known field (typically the velocity field) from the boundaries along characteristics
        /* \param velocities A GPUArray of the same shape as the grid that contains interface velocities; the rest will be filled in
         */
        void extend_velocities(GPUArray<Scalar>& velocities);

        //! Perform a trilinear interpolation to find the best estimate of the velocities at the true boundary
        /* \param B_Lz A GPUArray indexed by the grid indexer that contains the value of B on Lz cells (
         */
        GPUArray<Scalar> boundaryInterp(GPUArray<Scalar>& B_Lz);

    protected:

        //! Compute distances for Lz using linear interpolation
        void estimateLzDistances();

        //! Compute tentative distances for cell
        /* \param cell The grid cell to calculate tentative distances for
         * \param positive Whether the grid cell is positive or negative
         */
        Scalar calculateTentativeDistance(uint3 cell, bool positive);

        std::shared_ptr<SystemDefinition> m_sysdef; //!< HOOMD system definition
        std::shared_ptr<ParticleData> m_pdata;               //!< HOOMD particle data (required for box)
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
        std::shared_ptr<SparseFieldUpdater> m_field; //!< Sparse field
        std::shared_ptr<solvent::GridData> m_grid; //!< The grid that layers are maintained on

    private:
		//! Simple helper function to compute the sign
		template <class Real>
		inline int sgn(Real num)
            {
			return (num > Real(0)) - (num < Real(0));
            }

        //! The helper functions to compute the roots of the quadratic Lagrange multiplier solution for the tentative distance calculation
        std::pair<Scalar, Scalar> lagrange1D(Scalar delta_x1, Scalar phi1);
        std::pair<Scalar, Scalar> lagrange2D(Scalar delta_x1, Scalar phi1, Scalar delta_x2, Scalar phi2);
        std::pair<Scalar, Scalar> lagrange3D(Scalar delta_x1, Scalar phi1, Scalar delta_x2, Scalar phi2, Scalar delta_x3, Scalar phi3);

        //! The helper functions used to test validity of specific Lagrange solutions
        Scalar lagrangeP2(Scalar phi, Scalar delta_x1, Scalar phi1, Scalar delta_x2, Scalar phi2);
        Scalar lagrangeP3(Scalar phi, Scalar delta_x1, Scalar phi1, Scalar delta_x2, Scalar phi2, Scalar delta_x3, Scalar phi3);

        //! The greater than comparator to use for the PQ
		class CompareScalarLess
			{
			public:
				bool operator()(std::pair<uint3, Scalar> element1, std::pair<uint3, Scalar> element2)
                    {
					return element1.second < element2.second;
                    }
			};

        //! The greater than comparator to use for the PQ
		class CompareScalarGreater
			{
			public:
				bool operator()(std::pair<uint3, Scalar> element1, std::pair<uint3, Scalar> element2)
                    {
					return element1.second > element2.second;
                    }
			};

        //! The greater than comparator to use for the map
        // Need to provide ordering on int3 for hashing purposes.
        // Order is based on the components in order; first check x,
        // then check y, and finally check z. Note that this is not
        // a meaningful ordering; however it is sufficient to provide
        // a strict weak ordering as required for sorting
		class CompareInt
			{
			public:
				bool operator()(uint3 first, uint3 second)
                    {
                    if (first.x < second.x)
                        return true;
                    else if (first.x > second.x)
                        return false;
                    else
                        {
                        if (first.y < second.y)
                            return true;
                        else if (first.y > second.y)
                            return false;
                        else
                            {
                            if (first.z < second.z)
                                return true;
                            else
                                return false;
                            }
                        }
                    }
			};

    };

} // end namespace solvent

#endif // __FAST_MARCHER_H__
