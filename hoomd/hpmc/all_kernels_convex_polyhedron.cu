// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "UpdaterClustersGPU.cuh"
#include "UpdaterMuVTGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitGPU.cuh"
#include "IntegratorHPMCMonoImplicitNewGPU.cuh"
#include "BVHGPU.cuh"

#include "ShapeConvexPolyhedron.h"

namespace hpmc
{

namespace detail
{

//! HPMC kernels for ShapeConvexPolyhedron
template cudaError_t gpu_hpmc_free_volume<ShapeConvexPolyhedron >(const hpmc_free_volume_args_t &args,
                                                       const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_bvh_merge_shapes<ShapeConvexPolyhedron, AABBNodeGPU>(const hpmc_bvh_shapes_args_t& args,
                                                       AABBNodeGPU *d_tree_nodes,
                                                       const typename ShapeConvexPolyhedron::param_type *d_params);
template cudaError_t gpu_hpmc_clusters<ShapeConvexPolyhedron, AABBNodeGPU >(const hpmc_clusters_args_t &args,
                                                       const AABBNodeGPU *d_tree_nodes,
                                                       const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_bvh_merge_shapes<ShapeConvexPolyhedron, OBBNodeGPU>(const hpmc_bvh_shapes_args_t& args,
                                                       OBBNodeGPU *d_tree_nodes,
                                                       const typename ShapeConvexPolyhedron::param_type *d_params);
template cudaError_t gpu_hpmc_clusters<ShapeConvexPolyhedron, OBBNodeGPU >(const hpmc_clusters_args_t &args,
                                                       const OBBNodeGPU *d_tree_nodes,
                                                       const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_clusters_overlaps<ShapeConvexPolyhedron >(const hpmc_clusters_args_t &args,
                                                       const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeConvexPolyhedron >(const hpmc_args_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_update_aabb<ShapeConvexPolyhedron >(const hpmc_args_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_moves<ShapeConvexPolyhedron >(const hpmc_args_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_check_overlaps<ShapeConvexPolyhedron >(const hpmc_args_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_accept<ShapeConvexPolyhedron >(const hpmc_args_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_count_overlaps<ShapeConvexPolyhedron >(const hpmc_implicit_args_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject<ShapeConvexPolyhedron >(const hpmc_implicit_args_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_queue<ShapeConvexPolyhedron >(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_dp<ShapeConvexPolyhedron >(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject_new<ShapeConvexPolyhedron >(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeConvexPolyhedron ::param_type *d_params);
template cudaError_t gpu_hpmc_muvt<ShapeConvexPolyhedron >(const hpmc_muvt_args_t &args,
                                                       const typename ShapeConvexPolyhedron ::param_type *d_params);
}; // end namespace detail

} // end namespace hpmc
