// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "AllDriverAnisoPotentialPairGPU.cuh"

/*! \file AllDriverAnisoPotentialPairGPU.cu
    \brief Defines the driver functions for computing all types of anisotropic pair forces on the GPU
*/

cudaError_t gpu_compute_pair_aniso_forces_gb(const a_pair_args_t& pair_args,
            const EvaluatorPairGB::param_type* d_param)
    {
    return gpu_compute_pair_aniso_forces<EvaluatorPairGB>(pair_args, d_param);
    }

cudaError_t gpu_compute_pair_aniso_forces_dipole(const a_pair_args_t& pair_args,
            const EvaluatorPairDipole::param_type* d_param)
    {
    return gpu_compute_pair_aniso_forces<EvaluatorPairDipole>(pair_args, d_param);
    }
