// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "UpdaterClustersGPU.cuh"
#include "UpdaterMuVTGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitGPU.cuh"
#include "IntegratorHPMCMonoImplicitNewGPU.cuh"
#include "BVHGPU.cuh"

#include "ShapeSphinx.h"

namespace hpmc
{

namespace detail
{
#ifdef ENABLE_SPHINX_GPU
//! HPMC kernels for ShapeSphinx
template cudaError_t gpu_hpmc_free_volume<ShapeSphinx>(const hpmc_free_volume_args_t &args,
                                                       const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_bvh_merge_shapes<ShapeSphinx, AABBNodeGPU>(const hpmc_bvh_shapes_args_t& args,
                                                       AABBNodeGPU *d_tree_nodes,
                                                       const typename ShapeSphinx::param_type *d_params)
template cudaError_t gpu_hpmc_clusters<ShapeSphinx, AABBNodeGPU>(const hpmc_clusters_args_t &args,
                                                       const AABBNodeGPU *d_tree_nodes,
                                                       const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_bvh_merge_shapes<ShapeSphinx, OBBNodeGPU>(const hpmc_bvh_shapes_args_t& args,
                                                       OBBNodeGPU *d_tree_nodes,
                                                       const typename ShapeSphinx::param_type *d_params)
template cudaError_t gpu_hpmc_clusters<ShapeSphinx, OBBNodeGPU>(const hpmc_clusters_args_t &args,
                                                       const OBBNodeGPU *d_tree_nodes,
                                                       const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_hpmc_clusters_overlaps<ShapeSphinx>(const hpmc_clusters_args_t &args,
                                                       const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeSphinx>(const hpmc_args_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_hpmc_update_aabb<ShapeSphinx>(const hpmc_args_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_hpmc_moves<ShapeSphinx>(const hpmc_args_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_hpmc_check_overlaps<ShapeSphinx>(const hpmc_args_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_hpmc_accept<ShapeSphinx>(const hpmc_args_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_hpmc_count_overlaps<ShapeSphinx>(const hpmc_implicit_args_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject<ShapeSphinx>(const hpmc_implicit_args_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_queue<ShapeSphinx>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_dp<ShapeSphinx>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject_new<ShapeSphinx>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_hpmc_muvt<ShapeSphinx>(const hpmc_muvt_args_t &args,
                                                       const typename ShapeSphinx::param_type *d_params);
#endif
}; // end namespace detail

} // end namespace hpmc
