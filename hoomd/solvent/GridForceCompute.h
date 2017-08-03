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
#include "hoomd/ForceCompute.h"

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
class GridForceCompute : public ForceCompute
    {
    public:
        //! Constructor
        GridForceCompute(std::shared_ptr<SystemDefinition> sysdef)
            : ForceCompute(sysdef)
            { }

        //! Destructor
        ~GridForceCompute()
            { }

        //! Abstract method to pre-compute the energy terms
        /*! \param timestep Current time step
            Derived classes will implement this method to calculate their results
        */
        virtual void compute(unsigned int timestep)
            {
            // fails if no grid set yet
            assert(m_grid);

            // skip if we shouldn't compute this step
            if (!m_particles_sorted && !shouldCompute(timestep))
                return;

            precomputeEnergyTerms(timestep);
            m_particles_sorted = false;
            }

        //! Set the grid to be used
        /*! \param grid Shared pointer to grid object
        */
        void setGrid(std::shared_ptr<GridData> grid)
            {
            m_grid = grid;
            }

        //! Abstract method that performs the computation of forces from the grid
        /*! \param timestep Current time step
            Derived classes will implement this method to calculate their results
        */
        virtual void computeGridForces(unsigned int timestep){}

        //! Abstract method that interpolates the energy term onto the grid
        /*! \param timestep Current time step
            Derived classes will implement this method to calculate their results
        */
        virtual void precomputeEnergyTerms(unsigned int timestep){}

        //! Not implemented
        virtual void computeForces(unsigned int timestep)
            {
            throw std::runtime_error("Not implemented\n");
            }

    protected:
        std::shared_ptr<GridData> m_grid; //!< The grid on which we compute

    };

//! Export the grid force compute to python. This should never be initialized directly, however, 
//! and it exists solely to allow the GridPotentialPair to declare it as a base class.
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated GridPotentialPair class template.
*/
void export_GridForceCompute(pybind11::module& m, const std::string& name);

} // end namespace
#endif // __GRID_FORCE_COMPUTE_H__
