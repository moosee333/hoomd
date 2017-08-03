
#include "hoomd/hoomd_config.h"
#include "hoomd/md/PotentialPairGPU.cuh"
#include "hoomd/VectorMath.h"
#include "GeometryArgs.h"

template<typename Potential>
cudaError_t compute_comp_force_gpu(pair_args_t args, Scalar4 *d_quat, Scalar4 *d_torque,
                                   const GeometryArgs &geom_args, typename Potential::param_type *params);

#ifndef ENABLE_CUDA
template<typename Potential>
void compute_comp_force_gpu(pair_args_t args, Scalar4 *d_quat, Scalar4 *d_torque,
                            const GeometryArgs &geom_args, Potential::param_type *params)
{
}
#else

#ifdef NVCC
#include "COMPForceComputeGPU.cu"
#endif

#endif
