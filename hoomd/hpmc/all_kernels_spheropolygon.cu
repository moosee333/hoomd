// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "UpdaterClustersGPU.cuh"
#include "UpdaterMuVTGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitGPU.cuh"
#include "IntegratorHPMCMonoImplicitNewGPU.cuh"
#include "BVHGPU.cuh"

#include "ShapeSpheropolygon.h"

namespace hpmc
{

namespace detail
{

//! HPMC kernels for ShapeSpheropolygon
template cudaError_t gpu_hpmc_free_volume<ShapeSpheropolygon>(const hpmc_free_volume_args_t &args,
                                                       const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_bvh_merge_shapes<ShapeSpheropolygon, AABBNodeGPU>(const hpmc_bvh_shapes_args_t& args,
                                                       AABBNodeGPU *d_tree_nodes,
                                                       const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_clusters<ShapeSpheropolygon, AABBNodeGPU>(const hpmc_clusters_args_t &args,
                                                       const AABBNodeGPU *d_tree_nodes,
                                                       const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_bvh_merge_shapes<ShapeSpheropolygon, OBBNodeGPU>(const hpmc_bvh_shapes_args_t& args,
                                                       OBBNodeGPU *d_tree_nodes,
                                                       const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_clusters<ShapeSpheropolygon, OBBNodeGPU>(const hpmc_clusters_args_t &args,
                                                       const OBBNodeGPU *d_tree_nodes,
                                                       const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_clusters_overlaps<ShapeSpheropolygon>(const hpmc_clusters_args_t &args,
                                                       const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeSpheropolygon>(const hpmc_args_t& args,
                                                  const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_update_aabb<ShapeSpheropolygon>(const hpmc_args_t& args,
                                                  const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_moves<ShapeSpheropolygon>(const hpmc_args_t& args,
                                                  const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_check_overlaps<ShapeSpheropolygon>(const hpmc_args_t& args,
                                                  const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_accept<ShapeSpheropolygon>(const hpmc_args_t& args,
                                                  const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_count_overlaps<ShapeSpheropolygon>(const hpmc_implicit_args_t& args,
                                                  const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject<ShapeSpheropolygon>(const hpmc_implicit_args_t& args,
                                                  const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_queue<ShapeSpheropolygon>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_dp<ShapeSpheropolygon>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject_new<ShapeSpheropolygon>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeSpheropolygon::param_type *d_params);
template cudaError_t gpu_hpmc_muvt<ShapeSpheropolygon>(const hpmc_muvt_args_t &args,
                                                       const typename ShapeSpheropolygon::param_type *d_params);
}; // end namespace detail

} // end namespace hpmc
