
//#include "hoomd/hoomd_config.h"
#include "../COMPForceComputeGPU.cuh"
#include "../GeometryArgs.h"
#include "hoomd/md/EvaluatorPairGauss.h"

typedef EvaluatorPairGauss Potential;

template
cudaError_t compute_comp_force_gpu<Potential>(pair_args_t args, Scalar4 *d_quat,
                                              Scalar4 *d_torque,
                                              const GeometryArgs &geom_args,
                                              typename Potential::param_type *params);
