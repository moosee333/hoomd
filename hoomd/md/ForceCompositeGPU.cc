// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "ForceCompositeGPU.h"
#include "hoomd/VectorMath.h"

#include "ForceCompositeGPU.cuh"

namespace py = pybind11;

/*! \file ForceCompositeGPU.cc
    \brief Contains code for the ForceCompositeGPU class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
ForceCompositeGPU::ForceCompositeGPU(std::shared_ptr<SystemDefinition> sysdef)
        : ForceComposite(sysdef)
    {

    // power of two block sizes
    const cudaDeviceProp& dev_prop = m_exec_conf->dev_prop;
    std::vector<unsigned int> valid_params;
    unsigned int bodies_per_block = 1;
    for (unsigned int i = 0; i < 5; ++i)
        {
        bodies_per_block = 1 << i;
        unsigned int cur_block_size = 32;
        while (cur_block_size <= (unsigned int) dev_prop.maxThreadsPerBlock)
            {
            if (cur_block_size >= bodies_per_block)
                {
                valid_params.push_back(cur_block_size + bodies_per_block*10000);
                }
            cur_block_size *=2;
            }
        }

    m_tuner_force.reset(new Autotuner(valid_params, 5, 100000, "force_composite", this->m_exec_conf));
    m_tuner_virial.reset(new Autotuner(valid_params, 5, 100000, "virial_composite", this->m_exec_conf));

    // initialize autotuner
    std::vector<unsigned int> valid_params_update;
    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
        valid_params_update.push_back(block_size);

    m_tuner_update.reset(new Autotuner(valid_params_update, 5, 100000, "update_composite", this->m_exec_conf));

    GPUFlags<uint2> flag(m_exec_conf);
    m_flag.swap(flag);
    m_flag.resetFlags(make_uint2(0,0));
    }

ForceCompositeGPU::~ForceCompositeGPU()
    {}


//! Compute the forces and torques on the central particle
void ForceCompositeGPU::computeForces(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "constrain_rigid");

    if (m_prof)
        m_prof->push(m_exec_conf, "sum force and torque");

    // access local molecule data (need to move this on top because of GPUArray scoping issues)
    const Index2D& molecule_indexer = getMoleculeIndexer();
    unsigned int nmol = molecule_indexer.getW();

    ArrayHandle<unsigned int> d_molecule_length(getMoleculeLengths(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_molecule_list(getMoleculeList(), access_location::device, access_mode::read);

    // access particle data
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

    // access net force and torque acting on constituent particles
    ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_net_virial(m_pdata->getNetVirial(), access_location::device, access_mode::readwrite);

    // access the force and torque array for the central ptl
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_torque(m_torque, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);

    // access rigid body definition
    ArrayHandle<Scalar3> d_body_pos(m_body_pos, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_body_orientation(m_body_orientation, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body_len(m_body_len, access_location::device, access_mode::read);

    m_tuner_force->begin();
    unsigned int param = m_tuner_force->getParam();
    unsigned int block_size = param % 10000;
    unsigned int n_bodies_per_block = param/10000;

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = false;
    if (flags[pdata_flag::isotropic_virial] || flags[pdata_flag::pressure_tensor])
        {
        compute_virial = true;
        }

    // launch GPU kernel
    gpu_rigid_force(d_force.data,
                    d_torque.data,
                    d_molecule_length.data,
                    d_molecule_list.data,
                    molecule_indexer,
                    d_postype.data,
                    d_orientation.data,
                    m_body_idx,
                    d_body_pos.data,
                    d_body_orientation.data,
                    d_body_len.data,
                    d_body.data,
                    d_tag.data,
                    m_flag.getDeviceFlags(),
                    d_net_force.data,
                    d_net_torque.data,
                    nmol,
                    m_pdata->getN(),
                    n_bodies_per_block,
                    block_size,
                    m_exec_conf->dev_prop,
                    !compute_virial);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    uint2 flag = m_flag.readFlags();
    if (flag.x)
        {
        m_exec_conf->msg->error() << "constrain.rigid(): Composite particle with body tag " << flag.x-1 << " incomplete"
            << std::endl << std::endl;
        throw std::runtime_error("Error computing composite particle forces.\n");
        }

    m_tuner_force->end();

    if (compute_virial)
        {
        m_tuner_virial->begin();
        param = m_tuner_virial->getParam();
        block_size = param % 10000;
        n_bodies_per_block = param/10000;

        // launch GPU kernel
        gpu_rigid_virial(d_virial.data,
                        d_molecule_length.data,
                        d_molecule_list.data,
                        molecule_indexer,
                        d_postype.data,
                        d_orientation.data,
                        m_body_idx,
                        d_body_pos.data,
                        d_body_orientation.data,
                        d_net_force.data,
                        d_net_virial.data,
                        d_body.data,
                        d_tag.data,
                        nmol,
                        m_pdata->getN(),
                        n_bodies_per_block,
                        m_pdata->getNetVirial().getPitch(),
                        m_virial_pitch,
                        block_size,
                        m_exec_conf->dev_prop);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_virial->end();
        }


    if (m_prof) m_prof->pop(m_exec_conf);
    if (m_prof) m_prof->pop(m_exec_conf);
    }

void ForceCompositeGPU::updateCompositeParticles(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "constrain_rigid");

    if (m_prof)
        m_prof->push(m_exec_conf, "update");

    // access molecule order
    ArrayHandle<unsigned int> d_molecule_order(getMoleculeOrder(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_molecule_len(getMoleculeLengths(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_molecule_idx(getMoleculeIndex(), access_location::device, access_mode::read);

    // access the particle data arrays
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);

    // access body positions and orientations
    ArrayHandle<Scalar3> d_body_pos(m_body_pos, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_body_orientation(m_body_orientation, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body_len(m_body_len, access_location::device, access_mode::read);

    m_tuner_update->begin();
    unsigned int block_size = m_tuner_update->getParam();

    gpu_update_composite(m_pdata->getN(),
        m_pdata->getNGhosts(),
        d_body.data,
        d_rtag.data,
        d_postype.data,
        d_orientation.data,
        m_body_idx,
        d_body_pos.data,
        d_body_orientation.data,
        d_body_len.data,
        d_molecule_order.data,
        d_molecule_len.data,
        d_molecule_idx.data,
        d_image.data,
        m_pdata->getBox(),
        m_pdata->getGlobalBox(),
        block_size,
        m_flag.getDeviceFlags());

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_update->end();

    uint2 flag = m_flag.readFlags();

    if (flag.x)
        {
        ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

        unsigned int idx = flag.x - 1;
        unsigned int body_id = h_body.data[idx];
        unsigned int tag = h_tag.data[idx];

        m_exec_conf->msg->error() << "constrain.rigid(): Particle " << tag << " part of composite body " << body_id << " is missing central particle"
            << std::endl << std::endl;
        throw std::runtime_error("Error while updating constituent particles");
        }

    if (flag.y)
        {
        ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);

        unsigned int idx = flag.y - 1;
        unsigned int body_id = h_body.data[idx];

        m_exec_conf->msg->error() << "constrain.rigid(): Composite particle with body id " << body_id << " incomplete"
            << std::endl << std::endl;
        throw std::runtime_error("Error while updating constituent particles");
        }

    if (m_prof)
        m_prof->pop(m_exec_conf);

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void export_ForceCompositeGPU(py::module& m)
    {
    py::class_< ForceCompositeGPU, std::shared_ptr<ForceCompositeGPU> >(m, "ForceCompositeGPU", py::base<ForceComposite>())
        .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }
