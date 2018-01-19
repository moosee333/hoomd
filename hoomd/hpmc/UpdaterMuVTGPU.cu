// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "UpdaterMuVTGPU.cuh"

namespace hpmc
{

namespace detail
{

/*! \file UpdaterMuVTGPU.cu
    \brief Definition of CUDA kernels drivers for UpdaterMuVTGPU
*/

__global__
void gpu_muvt_set_particle_properties_kernel(
    unsigned int n_ptls_inserted,
    const unsigned int *d_rtag,
    const unsigned int *d_tags,
    Scalar4 *d_postype,
    Scalar4 *d_orientation,
    const Scalar4 *d_postype_insert,
    const Scalar4 *d_orientation_insert)
    {
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (idx >= n_ptls_inserted)
        return;

    unsigned int ptl_idx = d_rtag[d_tags[idx]];

    d_postype[ptl_idx] = d_postype_insert[idx];
    d_orientation[ptl_idx] = d_orientation_insert[idx];
    }

cudaError_t gpu_muvt_set_particle_properties(
    unsigned int n_ptls_inserted,
    const unsigned int *d_rtag,
    const unsigned int *d_tags,
    Scalar4 *d_postype,
    Scalar4 *d_orientation,
    const Scalar4 *d_postype_insert,
    const Scalar4 *d_orientation_insert,
    unsigned int block_size
    )
    {
    gpu_muvt_set_particle_properties_kernel<<<n_ptls_inserted/block_size+1, block_size>>>(
        n_ptls_inserted,
        d_rtag,
        d_tags,
        d_postype,
        d_orientation,
        d_postype_insert,
        d_orientation_insert);

    return cudaSuccess;
    }

}; // end namespace detail

} // end namespace hpmc

