// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "IntegratorHPMCMonoImplicitNew.h"
#include "ComputeFreeVolume.h"

#include "ShapeFacetedSphere.h"
#include "AnalyzerSDF.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"
#include "ExternalCallback.h"

#include "UpdaterExternalFieldWall.h"
#include "UpdaterRemoveDrift.h"
#include "UpdaterMuVT.h"
#include "UpdaterMuVTImplicit.h"
#include "UpdaterClusters.h"
#include "UpdaterClustersImplicit.h"

#ifdef ENABLE_CUDA
#include "IntegratorHPMCMonoGPU.h"
#include "IntegratorHPMCMonoImplicitGPU.h"
#include "IntegratorHPMCMonoImplicitNewGPU.h"
#include "ComputeFreeVolumeGPU.h"
#include "UpdaterClustersGPU.h"
#include "UpdaterMuVTGPU.h"
#include "BVHGPU.h"
#endif




namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the base HPMCMono integrators
void export_faceted_sphere(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeFacetedSphere >(m, "IntegratorHPMCMonoFacetedSphere");
    export_IntegratorHPMCMonoImplicit< ShapeFacetedSphere >(m, "IntegratorHPMCMonoImplicitFacetedSphere");
    export_IntegratorHPMCMonoImplicitNew< ShapeFacetedSphere >(m, "IntegratorHPMCMonoImplicitNewFacetedSphere");
    export_ComputeFreeVolume< ShapeFacetedSphere >(m, "ComputeFreeVolumeFacetedSphere");
    export_AnalyzerSDF< ShapeFacetedSphere >(m, "AnalyzerSDFFacetedSphere");
    export_UpdaterMuVT< ShapeFacetedSphere >(m, "UpdaterMuVTFacetedSphere");
    export_UpdaterClusters< ShapeFacetedSphere >(m, "UpdaterClustersFacetedSphere");
    export_UpdaterClustersImplicit< ShapeFacetedSphere, IntegratorHPMCMonoImplicit<ShapeFacetedSphere> >(m, "UpdaterClustersImplicitFacetedSphere");
    export_UpdaterClustersImplicit< ShapeFacetedSphere, IntegratorHPMCMonoImplicitNew<ShapeFacetedSphere> >(m, "UpdaterClustersImplicitNewFacetedSphere");
    export_UpdaterMuVTImplicit< ShapeFacetedSphere, IntegratorHPMCMonoImplicit<ShapeFacetedSphere> >(m, "UpdaterMuVTImplicitFacetedSphere");
    export_UpdaterMuVTImplicit< ShapeFacetedSphere, IntegratorHPMCMonoImplicitNew<ShapeFacetedSphere> >(m, "UpdaterMuVTImplicitNewFacetedSphere");

    export_ExternalFieldInterface<ShapeFacetedSphere>(m, "ExternalFieldFacetedSphere");
    export_LatticeField<ShapeFacetedSphere>(m, "ExternalFieldLatticeFacetedSphere");
    export_ExternalFieldComposite<ShapeFacetedSphere>(m, "ExternalFieldCompositeFacetedSphere");
    export_RemoveDriftUpdater<ShapeFacetedSphere>(m, "RemoveDriftUpdaterFacetedSphere");
    export_ExternalFieldWall<ShapeFacetedSphere>(m, "WallFacetedSphere");
    export_UpdaterExternalFieldWall<ShapeFacetedSphere>(m, "UpdaterExternalFieldWallFacetedSphere");
    export_ExternalCallback<ShapeFacetedSphere>(m, "ExternalCallbackFacetedSphere");

    #ifdef ENABLE_CUDA
    using BVH_GPU_AABB = BVHGPU< AABBNodeGPU, ShapeFacetedSphere, IntegratorHPMCMono< ShapeFacetedSphere > >;
    export_BVHGPU< AABBNodeGPU, ShapeFacetedSphere, IntegratorHPMCMono< ShapeFacetedSphere > >(m, "BVHGPUAABBFacetedSphere");

    using BVH_GPU_OBB = BVHGPU< OBBNodeGPU, ShapeFacetedSphere, IntegratorHPMCMono< ShapeFacetedSphere > >;
    export_BVHGPU< OBBNodeGPU, ShapeFacetedSphere, IntegratorHPMCMono< ShapeFacetedSphere > >(m, "BVHGPUOBBFacetedSphere");

    export_IntegratorHPMCMonoGPU< ShapeFacetedSphere >(m, "IntegratorHPMCMonoGPUFacetedSphere");
    export_IntegratorHPMCMonoImplicitGPU< ShapeFacetedSphere >(m, "IntegratorHPMCMonoImplicitGPUFacetedSphere");
    export_IntegratorHPMCMonoImplicitNewGPU< ShapeFacetedSphere >(m, "IntegratorHPMCMonoImplicitNewGPUFacetedSphere");
    export_ComputeFreeVolumeGPU< ShapeFacetedSphere >(m, "ComputeFreeVolumeGPUFacetedSphere");
    export_UpdaterClustersGPU< ShapeFacetedSphere, BVH_GPU_AABB >(m, "UpdaterClustersGPUFacetedSphereAABB");
    export_UpdaterClustersGPU< ShapeFacetedSphere, BVH_GPU_OBB >(m, "UpdaterClustersGPUFacetedSphereOBB");
    export_UpdaterMuVTGPU< ShapeFacetedSphere >(m, "UpdaterMuVTGPUFacetedSphere");

    #endif
    }

}
