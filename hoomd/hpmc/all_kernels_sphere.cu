// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "UpdaterClustersGPU.cuh"
#include "UpdaterMuVTGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitGPU.cuh"
#include "IntegratorHPMCMonoImplicitNewGPU.cuh"
#include "BVHGPU.cuh"

#include "ShapeSphere.h"

namespace hpmc
{

namespace detail
{

//! HPMC kernels for ShapeSphere
template cudaError_t gpu_hpmc_free_volume<ShapeSphere>(const hpmc_free_volume_args_t &args,
                                                       const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_bvh_merge_shapes<ShapeSphere, OBBNodeGPU>(const hpmc_bvh_shapes_args_t& args,
                                                       OBBNodeGPU *d_tree_nodes,
                                                       const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_hpmc_clusters<ShapeSphere, OBBNodeGPU>(const hpmc_clusters_args_t &args,
                                                       const OBBNodeGPU *d_tree_nodes,
                                                       const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_hpmc_clusters_overlaps<ShapeSphere>(const hpmc_clusters_args_t &args,
                                                       const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeSphere>(const hpmc_args_t& args,
                                                  const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_hpmc_update_aabb<ShapeSphere>(const hpmc_args_t& args,
                                                  const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_hpmc_moves<ShapeSphere>(const hpmc_args_t& args,
                                                  const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_hpmc_check_overlaps<ShapeSphere>(const hpmc_args_t& args,
                                                  const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_hpmc_accept<ShapeSphere>(const hpmc_args_t& args,
                                                  const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_count_overlaps<ShapeSphere>(const hpmc_implicit_args_t& args,
                                                  const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject<ShapeSphere>(const hpmc_implicit_args_t& args,
                                                  const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_queue<ShapeSphere>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_dp<ShapeSphere>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject_new<ShapeSphere>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeSphere::param_type *d_params);
template cudaError_t gpu_hpmc_muvt<ShapeSphere>(const hpmc_muvt_args_t &args,
                                                       const typename ShapeSphere::param_type *d_params);
}; // end namespace detail

} // end namespace hpmc
