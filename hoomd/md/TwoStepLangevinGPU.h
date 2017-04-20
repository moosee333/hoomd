// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepLangevin.h"

#ifndef __TWO_STEP_LANGEVIN_GPU_H__
#define __TWO_STEP_LANGEVIN_GPU_H__

/*! \file TwoStepLangevinGPU.h
    \brief Declares the TwoStepLangevinGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Implements Langevin dynamics on the GPU
/*! GPU accelerated version of TwoStepLangevin

    \ingroup updaters
*/
class TwoStepLangevinGPU : public TwoStepLangevin
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepLangevinGPU(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<ParticleGroup> group,
                           std::shared_ptr<Variant> T,
                           unsigned int seed,
                           bool use_lambda,
                           Scalar lambda,
                           bool noiseless_t,
                           bool noiseless_r,
                           const std::string& suffix = std::string(""));
        virtual ~TwoStepLangevinGPU() {};

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        unsigned int m_block_size;               //!< block size for partial sum memory
        unsigned int m_num_blocks;               //!< number of memory blocks reserved for partial sum memory
        GPUArray<Scalar> m_partial_sum1;         //!< memory space for partial sum over bd energy transfers
        GPUArray<Scalar> m_sum;                  //!< memory space for sum over bd energy transfers
    };

//! Exports the TwoStepLangevinGPU class to python
void export_TwoStepLangevinGPU(pybind11::module& m);

#endif // #ifndef __TWO_STEP_LANGEVIN_GPU_H__
