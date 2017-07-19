// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

/*! \file GridForceCompute.h
    \brief Defines the GridForceCompute abstract class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __GRID_FORCE_COMPUTE_H__
#define __GRID_FORCE_COMPUTE_H__

#include "GridData.h"

namespace solvent
{

/*! A GridForceCompute computes energy terms on the velocity grid that correspond
    to solute-solvent interactions. These include non-polar (vdW) and polar (charged)
    interactions. An instantiation of a GridForceCompute also back-interpolates from the grid
    to compute the actual solute particle forces.

    This class is an abstract interface which defines the type of energy term are
    accepted in the LevelSetSolver class. The LevelSetSolver takes care of summing up
    the individual force contributions and passing the result to the integrator.

    This class is **not** actually used as a ForceCompute inside IntegratorTwoStep, we are merely
    subclassing ForceCompute to not have to re-implement the relevant internal data structures.

 */
template<class evaluator>
class GridForceCompute : public ForceCompute
    {
    public:
        GridForceCompute(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<GridData> grid)
            : Compute(sysdef)
            { }

        //! Abstract method to pre-compute the energy terms
        /*! \param timestep Current time step
            Derived classes will implement this method to calculate their results
        */
        virtual void compute(unsigned int timestep)
            {
            // skip if we shouldn't compute this step
            if (!m_particles_sorted && !shouldCompute(timestep))
                return;

            precomputeEnergyTerms(timestep);
            m_particles_sorted = false;
            }

        //! Abstract method that performs the computation of forces from the grid
        /*! \param timestep Current time step
            Derived classes will implement this method to calculate their results
        */
        virtual void computeGridForces(unsigned int timestep){}

        //! Not implemented
        virtual void computeForces(unsigned int timestep)
            {
            throw std::runtime_error("Not implemented\n");
            }

    protected:
        //! Abstract method that interpolates the energy term onto the grid
        /*! \param timestep Current time step
            Derived classes will implement this method to calculate their results
        */
        virtual void precomputeEnergyTerms(unsigned int timestep){}

    };

} // end namespace

#endif // __GRID_FORCE_COMPUTE_H__
