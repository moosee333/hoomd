// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "UpdaterClustersGPU.cuh"
#include "UpdaterMuVTGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitGPU.cuh"
#include "IntegratorHPMCMonoImplicitNewGPU.cuh"

#include "ShapeConvexPolygon.h"

namespace hpmc
{

namespace detail
{

//! HPMC kernels for ShapeConvexPolygon
template cudaError_t gpu_hpmc_free_volume<ShapeConvexPolygon>(const hpmc_free_volume_args_t &args,
                                                       const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_hpmc_clusters<ShapeConvexPolygon>(const hpmc_clusters_args_t &args,
                                                       const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_hpmc_clusters_overlaps<ShapeConvexPolygon>(const hpmc_clusters_args_t &args,
                                                       const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeConvexPolygon>(const hpmc_args_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_hpmc_update_aabb<ShapeConvexPolygon>(const hpmc_args_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_hpmc_moves<ShapeConvexPolygon>(const hpmc_args_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_hpmc_check_overlaps<ShapeConvexPolygon>(const hpmc_args_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_hpmc_accept<ShapeConvexPolygon>(const hpmc_args_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_count_overlaps<ShapeConvexPolygon>(const hpmc_implicit_args_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject<ShapeConvexPolygon>(const hpmc_implicit_args_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_queue<ShapeConvexPolygon>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_hpmc_insert_depletants_dp<ShapeConvexPolygon>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_hpmc_implicit_accept_reject_new<ShapeConvexPolygon>(const hpmc_implicit_args_new_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_hpmc_muvt<ShapeConvexPolygon>(const hpmc_muvt_args_t &args,
                                                       const typename ShapeConvexPolygon::param_type *d_params);
}; // end namespace detail

} // end namespace hpmc
