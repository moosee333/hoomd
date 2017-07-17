// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file ComputeThermoGPU.cc
    \brief Contains code for the ComputeThermoGPU class
*/


#include "ComputeThermoGPU.h"
#include "ComputeThermoGPU.cuh"

namespace py = pybind11;

#ifdef ENABLE_MPI
#include "Communicator.h"
#include "HOOMDMPI.h"
#endif

#include <iostream>
using namespace std;

/*! \param sysdef System for which to compute thermodynamic properties
    \param group Subset of the system over which properties are calculated
    \param suffix Suffix to append to all logged quantity names
*/

ComputeThermoGPU::ComputeThermoGPU(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
                                   const std::string& suffix)
    : ComputeThermo(sysdef, group, suffix), m_scratch(m_exec_conf), m_scratch_pressure_tensor(m_exec_conf),
        m_scratch_rot(m_exec_conf)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a ComputeThermoGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing ComputeThermoGPU");
        }

    m_block_size = 512;

    // override base class allocation using mapped memory
    GPUArray< Scalar > properties(thermo_index::num_quantities, m_exec_conf,true);
    m_properties.swap(properties);

    cudaEventCreate(&m_event, cudaEventDisableTiming);
    }

//! Destructor
ComputeThermoGPU::~ComputeThermoGPU()
    {
    cudaEventDestroy(m_event);
    }

/*! Computes all thermodynamic properties of the system in one fell swoop, on the GPU.
 */
void ComputeThermoGPU::computeProperties()
    {
    // just drop out if the group is an empty group
    if (m_group->getNumMembersGlobal() == 0)
        return;

    unsigned int group_size = m_group->getNumMembers();

    if (m_prof) m_prof->push(m_exec_conf,"Thermo");

    assert(m_pdata);
    assert(m_ndof != 0);

    // number of blocks in reduction
    unsigned int num_blocks = m_group->getNumMembers() / m_block_size + 1;

    // resize work space
    m_scratch.resize(num_blocks);
    m_scratch_pressure_tensor.resize(num_blocks*6);
    m_scratch_rot.resize(num_blocks);

    // access the particle data
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getGlobalBox();

    PDataFlags flags = m_pdata->getFlags();

    { // scope these array handles so they are released before the additional terms are added
    // access the net force, pe, and virial
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_scratch(m_scratch, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_scratch_pressure_tensor(m_scratch_pressure_tensor, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_scratch_rot(m_scratch_rot, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_properties(m_properties, access_location::device, access_mode::overwrite);

    // access the group
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    // build up args list
    num_blocks = m_group->getNumMembers() / m_block_size + 1;
    compute_thermo_args args;
    args.d_net_force = d_net_force.data;
    args.d_net_virial = d_net_virial.data;
    args.d_orientation = d_orientation.data;
    args.d_angmom = d_angmom.data;
    args.d_inertia = d_inertia.data;
    args.virial_pitch = net_virial.getPitch();
    args.ndof = m_ndof;
    args.D = m_sysdef->getNDimensions();
    args.d_scratch = d_scratch.data;
    args.d_scratch_pressure_tensor = d_scratch_pressure_tensor.data;
    args.d_scratch_rot = d_scratch_rot.data;
    args.block_size = m_block_size;
    args.n_blocks = num_blocks;
    args.external_virial_xx = m_pdata->getExternalVirial(0);
    args.external_virial_xy = m_pdata->getExternalVirial(1);
    args.external_virial_xz = m_pdata->getExternalVirial(2);
    args.external_virial_yy = m_pdata->getExternalVirial(3);
    args.external_virial_yz = m_pdata->getExternalVirial(4);
    args.external_virial_zz = m_pdata->getExternalVirial(5);
    args.external_energy = m_pdata->getExternalEnergy();

    // perform the computation on the GPU
    gpu_compute_thermo( d_properties.data,
                        d_vel.data,
                        d_index_array.data,
                        group_size,
                        box,
                        args,
                        flags[pdata_flag::pressure_tensor],
                        flags[pdata_flag::rotational_kinetic_energy]);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

    #ifdef ENABLE_MPI
    // in MPI, reduce extensive quantities only when they're needed
    m_properties_reduced = !m_pdata->getDomainDecomposition();

    if (!m_properties_reduced) cudaEventRecord(m_event);
    #endif // ENABLE_MPI

    if (m_prof) m_prof->pop(m_exec_conf);
    }

#ifdef ENABLE_MPI
void ComputeThermoGPU::reduceProperties()
    {
    if (m_properties_reduced) return;

    ArrayHandleAsync<Scalar> h_properties(m_properties, access_location::host, access_mode::readwrite);
    cudaEventSynchronize(m_event);

    // reduce properties
    MPI_Allreduce(MPI_IN_PLACE, h_properties.data, thermo_index::num_quantities, MPI_HOOMD_SCALAR,
            MPI_SUM, m_exec_conf->getMPICommunicator());

    m_properties_reduced = true;
    }
#endif


void export_ComputeThermoGPU(py::module& m)
    {
    py::class_<ComputeThermoGPU, std::shared_ptr<ComputeThermoGPU> >(m,"ComputeThermoGPU",py::base<ComputeThermo>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
         std::shared_ptr<ParticleGroup>,
         const std::string& >())
        ;
    }
