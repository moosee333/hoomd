// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "ForceComposite.h"
#include "NeighborList.h"
#include "hoomd/Autotuner.h"

#include "hoomd/GPUFlags.h"

/*! \file ForceCompositeGPU.h
    \brief Implementation of a rigid body force compute, GPU version
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __ForceCompositeGPU_H__
#define __ForceCompositeGPU_H__

class ForceCompositeGPU : public ForceComposite
    {
    public:
        //! Constructs the compute
        ForceCompositeGPU(std::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~ForceCompositeGPU();

        //! Update the constituent particles of a composite particle
        /*  Using the position, velocity and orientation of the central particle
         * \param remote If true, consider remote bodies, otherwise bodies
         *        with a local central particle
         */
        virtual void updateCompositeParticles(unsigned int timestep);

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs

            Derived classes should override this to set the parameters of their autotuners.
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            ForceComposite::setAutotunerParams(enable, period);

            m_tuner_force->setPeriod(period);
            m_tuner_force->setEnabled(enable);

            m_tuner_virial->setPeriod(period);
            m_tuner_virial->setEnabled(enable);

            m_tuner_update->setPeriod(period);
            m_tuner_update->setEnabled(enable);
            }


    protected:
        //! Compute the forces and torques on the central particle
        virtual void computeForces(unsigned int timestep);

        std::unique_ptr<Autotuner> m_tuner_force;  //!< Autotuner for block size and threads per particle
        std::unique_ptr<Autotuner> m_tuner_virial; //!< Autotuner for block size and threads per particle
        std::unique_ptr<Autotuner> m_tuner_update; //!< Autotuner for block size of update kernel

        GPUFlags<uint2> m_flag;               //!< Flag to read out error condition
    };

//! Exports the ForceCompositeGPU to python
void export_ForceCompositeGPU(pybind11::module& m);

#endif
