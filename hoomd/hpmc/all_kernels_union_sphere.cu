// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "UpdaterClustersGPU.cuh"
#include "UpdaterMuVTGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitGPU.cuh"
#include "IntegratorHPMCMonoImplicitNewGPU.cuh"
#include "BVHGPU.cuh"

#include "ShapeUnion.h"

namespace hpmc
{

namespace detail
{

//! HPMC kernels for ShapeUnion<ShapeSphere>
template cudaError_t gpu_hpmc_free_volume<ShapeUnion<ShapeSphere> >(const hpmc_free_volume_args_t &args,
                                                       const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_bvh_merge_shapes<ShapeUnion<ShapeSphere>, OBBNodeGPU>(const hpmc_bvh_shapes_args_t& args,
                                                       OBBNodeGPU *d_tree_nodes,
                                                       const typename ShapeUnion<ShapeSphere>::param_type *d_params);
template cudaError_t gpu_hpmc_clusters<ShapeUnion<ShapeSphere>, OBBNodeGPU >(const hpmc_clusters_args_t &args,
                                                       const OBBNodeGPU *d_tree_nodes,
                                                       const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_hpmc_clusters_overlaps<ShapeUnion<ShapeSphere> >(const hpmc_clusters_args_t &args,
                                                       const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeUnion<ShapeSphere> >(const hpmc_args_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_hpmc_update_aabb<ShapeUnion<ShapeSphere> >(const hpmc_args_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_hpmc_moves<ShapeUnion<ShapeSphere> >(const hpmc_args_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_hpmc_check_overlaps<ShapeUnion<ShapeSphere> >(const hpmc_args_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_hpmc_accept<ShapeUnion<ShapeSphere> >(const hpmc_args_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_count_overlaps<ShapeUnion<ShapeSphere> >(const hpmc_implicit_args_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject<ShapeUnion<ShapeSphere> >(const hpmc_implicit_args_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_queue<ShapeUnion<ShapeSphere> >(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_dp<ShapeUnion<ShapeSphere> >(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject_new<ShapeUnion<ShapeSphere> >(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_hpmc_muvt<ShapeUnion<ShapeSphere> >(const hpmc_muvt_args_t &args,
                                                       const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
}; // end namespace detail

} // end namespace hpmc
