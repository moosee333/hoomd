// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jwrm

#include "LBFGSEnergyMinimizer.h"

using namespace std;
namespace py = pybind11;

/*! \file LBGFSEnergyMinimizer.cc
    \brief Contains code for the LBFGSEnergyMinimizer class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param dt maximum step size

    \post The method is constructed with the given particle data and a NULL profiler.
*/
LBFGSEnergyMinimizer::LBFGSEnergyMinimizer(std::shared_ptr<SystemDefinition> sysdef, Scalar dt)
    :   IntegratorTwoStep(sysdef, dt),
        m_dguess(0.1),
        m_etol(1e-3),
        m_ftol(1e-1),
        m_max_decrease(10),
        m_max_erise(1e-4),
        m_max_fails(5),
        m_max_step(0.1),
        m_scale(0.1),
        m_updates(4)
    {
    m_exec_conf->msg->notice(5) << "Constructing LBFGSEnergyMinimizer" << endl;

    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    reset();
    }

LBFGSEnergyMinimizer::~LBFGSEnergyMinimizer()
    {
    m_exec_conf->msg->notice(5) << "Destroying LBFGSEnergyMinimizer" << endl;
    }

/*! \param dguess is the new guess to set
*/
void LBFGSEnergyMinimizer::setDguess(Scalar dguess)
    {
    if (!(dguess > 0.0))
        {
        m_exec_conf->msg->error() << "integrate.mode_minimize_lbfgs: initial guess for the diagonal elements of the inverse Hessian should be > 1" << endl;
        throw runtime_error("Error setting parameters for LBFGSEnergyMinimizer");
        }
    m_dguess = dguess;
    }

/*! \param erise is the new maximum energy rise to set
*/
void LBFGSEnergyMinimizer::setMaxErise(Scalar erise)
    {
    if (!(erise >= 0.0))
        {
        m_exec_conf->msg->error() << "integrate.mode_minimize_lbfgs: maximum energy rise should be >= 0" << endl;
        throw runtime_error("Error setting parameters for LBFGSEnergyMinimizer");
        }
        m_max_erise = erise;
    }

/*! \param step is the new maximum step size to set
*/
void LBFGSEnergyMinimizer::setMaxStep(Scalar step)
    {
    if (!(step > 0.0))
        {
        m_exec_conf->msg->error() << "integrate.mode_minimize_lbfgs: maximum step size should be > 0" << endl;
        throw runtime_error("Error setting parameters for LBFGSEnergyMinimizer");
        }
    m_max_step = step;
    }

/*! \param scale is the new scale to set
*/
void LBFGSEnergyMinimizer::setScale(Scalar scale)
    {
    if (!(scale > 0.0 && scale < 1.0))
        {
        m_exec_conf->msg->error() << "integrate.mode_minimize_lbfgs: scaling of the step size should be beteeen 0 and 1" << endl;
        throw runtime_error("Error setting parameters for LBFGSEnergyMinimizer");
        }
    m_scale = scale;
    }

void LBFGSEnergyMinimizer::reset()
    {
    m_converged = false;
    m_energy_total = 0.0;
    m_failed = false;
    m_iter = 0;
    m_no_fails = 0;
    m_was_reset = true;

    unsigned int n = m_pdata->getN();
    GPUArray<Scalar4> pos_history(n*m_updates, m_exec_conf);
    GPUArray<Scalar3> grad_history(n*m_updates, m_exec_conf);
    GPUArray<Scalar> rho_history(m_updates, m_exec_conf);
    m_pos_history.swap(pos_history);
    m_grad_history.swap(grad_hisory);

    }

/*! \param timesteps is the current timestep
*/
void LBFGSEnergyMinimizer::update(unsigned int timesteps)
    {

    if (m_converged || m_failed)
        return;

    // Check whether we've got stuck
    if (m_no_fails >= m_max_fails)
        {
            m_failed = true;
            return;
        }

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // perform all necessary communication steps. This ensures
        // a) that particles have migrated to the correct domains
        // b) that forces are calculated correctly, if ghost atom positions are updated every time step

        // also updates rigid bodies after ghost updating
        m_comm->communicate(timestep+1);
        }
    else
#endif
        {
        updateRigidBodies(timestep+1);
        }

    // compute the net force on all particles
#ifdef ENABLE_CUDA
    if (m_exec_conf->exec_mode == ExecutionConfiguration::GPU)
        computeNetForceGPU(timestep+1);
    else
#endif
        computeNetForce(timestep+1);

    // Total number of particles
    unsigned int no_particles = m_pdata->getN();
    // Total number of particles in integration groups
    unsigned int total_group_size = 0;
    // The step direction
    ArrayHandle<Scalar3> h_step(m_step, access_location::host, access_mode::readwrite);
    // Particle positions
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    // Gradients (forces) on the particles
    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(), access_location::host, access_mode::read);

    // Calculate total energy and RMS force
    Scalar pe_total(0.0);
    Scalar rms_force(0.0)
    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
        unsigned int group_size = current_group->getIndexArray().getNumElements();
        total_group_size += group_size;
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = current_group->getMemberIndex(group_idx);
            pe_total += h_net_force.data[j].w;
            rms_force += h_net_force.data[j].x * h_net_force.data[j].x;
            rms_force += h_net_force.data[j].y * h_net_force.data[j].y;
            rms_force += h_net_force.data[j].z * h_net_force.data[j].z;
            }
        }

    // Check convergence
    if (m_was_reset)
        {
        m_was_reset = false;
        m_total_energy = pe_total + Scalar(100000)*m_etol;
        }
    unsigned int ndof = m_sysdef->getNDimensions()*total_group_size;
    rms_force = sqrt(rms_force / ndof);
    if (rms_force < m_ftol && fabs(pe_total/total_group_size - m_energy_total) < m_etol))
        {
        m_converged = true;
        m_energy_total = pe_total;
        return;
        }

    if (pe_total > m_energy_total + m_max_erise && m_iter != 0)
        {
        // The energy has increased: undo the step
        ArrayHandle<Scalar4> h_pos_history(m_pos_history, access_location::host, access_mode::read);
        Scalar pe_total = 0.0;
        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
            unsigned int group_size = current_group->getIndexArray().getNumElements();
            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);
                h_pos.data[j] = h_pos_history.data[no_particles * (m_iter - 1) + j];
                }
            }
        --m_iter;
        ++m_no_decrease;

        // Check whether we've had enough increases to be a step failure
        if (m_no_decrease >= m_max_decrease)
            {
            // Reset to zero history
            m_iter = 0;
            m_no_decrease = 0;

            ++m_no_fails;
            return;
            }     

        // Reduce the step size
        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
            unsigned int group_size = current_group->getIndexArray().getNumElements();
            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);
                h_step.data[j].x *= m_scale;
                h_step.data[j].y *= m_scale;
                h_step.data[j].z *= m_scale;
                }
            }
        }
    else
        {
        // The previous step was successful, generate a new step direction
        ArrayHandle<Scalar4> h_pos_history(m_pos_history, access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_grad_history(m_grad_history, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_rho_history(m_rho_history, access_location::host, access_mode::read);

        // Update energy
        m_energy_total = pe_total;

        // Extract the current gradients
        GPUArray<Scalar3> q(no_particles, m_exec_conf);
        ArrayHandle<Scalar3> h_q(q, access_location::host, access_mode::readwrite);
        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
            unsigned int group_size = current_group->getIndexArray().getNumElements();
            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);
                h_q.data[j].x = h_net_force.data[j].x;
                h_q.data[j].y = h_net_force.data[j].y;
                h_q.data[j].z = h_net_force.data[j].z;
                }
            }

        // If we're at the full number of updates, we need to shift the history
        // back one.
        if (m_iter == m_updates - 1)
            {
            for (unsigned int i = 0; i < m_iter; ++i)
                {
                for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
                    {
                    std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
                    unsigned int group_size = current_group->getIndexArray().getNumElements();
                    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                        {
                        unsigned int j = current_group->getMemberIndex(group_idx);
                        h_pos_history.data[no_particles * i + j] = h_pos_history.data[no_particles * (i + 1) + j];
                        h_grad_history.data[no_particles * i + j] = h_grad_history.data[no_particles * (i + 1) + j];
                        }
                    }
                }
            }
        // Add the current positions and gradients to the history
        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
            unsigned int group_size = current_group->getIndexArray().getNumElements();
            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);
                h_pos_history.data[m_iter * no_particles + j] = h_pos.data[j];
                h_grad_history.data[m_iter * no_particles + j].x = h_net_force.data[j].x;
                h_grad_history.data[m_iter * no_particles + j].y = h_net_force.data[j].y;
                h_grad_history.data[m_iter * no_particles + j].z = h_net_force.data[j].z;
                }
            }

        // Generate the position and gradient differences
        GPUArray<Scalar3> s(no_particles * m_iter, m_exec_conf);
        GPUArray<Scalar3> y(no_particles * m_iter, m_exec_conf);
        ArrayHandle<Scalar3> h_s(s, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_y(s, access_location::host, access_mode::readwrite);
        for (unsigned int i = 0; i < m_iter; ++i)
            {
            for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
                {
                std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
                unsigned int group_size = current_group->getIndexArray().getNumElements();
                for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                    {
                    unsigned int j = current_group->getMemberIndex(group_idx);
                    h_s.data[no_particles * i + j].x = h_pos_history.data[no_particles * (i + 1) + j].x - h_pos_history.data[no_particles * i + j].x;
                    h_s.data[no_particles * i + j].y = h_pos_history.data[no_particles * (i + 1) + j].y - h_pos_history.data[no_particles * i + j].y;
                    h_s.data[no_particles * i + j].z = h_pos_history.data[no_particles * (i + 1) + j].z - h_pos_history.data[no_particles * i + j].z;
                    h_y.data[no_particles * i + j].x = h_grad_history.data[no_particles * (i + 1) + j].x - h_grad_history.data[no_particles * i + j].x;
                    h_y.data[no_particles * i + j].y = h_grad_history.data[no_particles * (i + 1) + j].y - h_grad_history.data[no_particles * i + j].y;
                    h_y.data[no_particles * i + j].z = h_grad_history.data[no_particles * (i + 1) + j].z - h_grad_history.data[no_particles * i + j].z;
                    }
                }
            }
        
        // Generate the guesses for the diagonal Hessian elements
        Scalar hess(0.0);
        if (m_iter == 0)
            {
            // No history, use initial guess
            hess = m_dguess;
            }
        else
            {
            // Some history, use previous step information
            Scalar yy(0.0);
            Scalar ys(0.0);
            for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
                {
                std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
                unsigned int group_size = current_group->getIndexArray().getNumElements();
                for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                    {
                    unsigned int j = current_group->getMemberIndex(group_idx);
                    yy += h_y.data[no_particles * (m_iter - 1) + j].x * h_y.data[no_particles * (m_iter - 1) + j].x;
                    yy += h_y.data[no_particles * (m_iter - 1) + j].y * h_y.data[no_particles * (m_iter - 1) + j].y;
                    yy += h_y.data[no_particles * (m_iter - 1) + j].z * h_y.data[no_particles * (m_iter - 1) + j].z;
                    ys += h_y.data[no_particles * (m_iter - 1) + j].x * h_s.data[no_particles * (m_iter - 1) + j].x;
                    ys += h_y.data[no_particles * (m_iter - 1) + j].y * h_s.data[no_particles * (m_iter - 1) + j].y;
                    ys += h_y.data[no_particles * (m_iter - 1) + j].z * h_s.data[no_particles * (m_iter - 1) + j].z;
                    }
                }
            hess = ys/yy;
            }

        // First loop of the two loop update scheme
        // Described on the LBFGS Wikipedia page
        ArrayHandle<Scalar> h_rho(m_rho, access_location::host, access_mode::read);
        GPUArray<Scalar3> alpha(m_iter, m_exec_conf);
        ArrayHandle<Scalar3> h_alpha(alpha, access_location::host, access_mode::readwrite);
        for (unsigned int i = m_iter - 1; i >= 0; --i)
            {
            for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
                {
                std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
                unsigned int group_size = current_group->getIndexArray().getNumElements();
                for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                    {
                    unsigned int j = current_group->getMemberIndex(group_idx);
                    h_alpha.data[i] += h_s.data[no_particles * i + j].x * h_q.data[j].x;
                    h_alpha.data[i] += h_s.data[no_particles * i + j].y * h_q.data[j].y;
                    h_alpha.data[i] += h_s.data[no_particles * i + j].z * h_q.data[j].z;
                    }
                }
            h_alpha[i] *= h_rho[i];
            for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
                {
                std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
                unsigned int group_size = current_group->getIndexArray().getNumElements();
                for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                    {
                    unsigned int j = current_group->getMemberIndex(group_idx);
                    h_q.data[j].x -= h_y.data[no_particles * i + j].x * h_alpha.data[i];
                    h_q.data[j].y -= h_y.data[no_particles * i + j].y * h_alpha.data[i];
                    h_q.data[j].z -= h_y.data[no_particles * i + j].z * h_alpha.data[i];
                    }
                }
            }

        // Generate initial value of step
        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
            unsigned int group_size = current_group->getIndexArray().getNumElements();
            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);
                h_step.data[j].x = hess * h_q.data[j].x;
                h_step.data[j].y = hess * h_q.data[j].y;
                h_step.data[j].z = hess * h_q.data[j].z;
                }
            }       

        // Second loop of the two loop update scheme
        for (unsigned int i = 0; i < m_iter; ++i)
            {
            Scalar beta(0.0);
            for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
                {
                std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
                unsigned int group_size = current_group->getIndexArray().getNumElements();
                for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                    {
                    unsigned int j = current_group->getMemberIndex(group_idx);
                    beta += h_y.data[no_particles * i + j].x * h_step.data[j].x;
                    beta += h_y.data[no_particles * i + j].y * h_step.data[j].y;
                    beta += h_y.data[no_particles * i + j].z * h_step.data[j].z;
                    }
                }
            beta *= h_rho[];
            for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
                {
                std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
                unsigned int group_size = current_group->getIndexArray().getNumElements();
                for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                    {
                    unsigned int j = current_group->getMemberIndex(group_idx);
                    h_step.data[j].x += h_s.data[no_particles * i + j].x * (h_alpha.data[i] - beta);
                    h_step.data[j].y += h_s.data[no_particles * i + j].y * (h_alpha.data[i] - beta);
                    h_step.data[j].z += h_s.data[no_particles * i + j].z * (h_alpha.data[i] - beta);
                    }
                }
            }

        // Check whether the step points up or downhill
        Scalar proj(0.0);
        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
            unsigned int group_size = current_group->getIndexArray().getNumElements();
            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);
                proj += h_step.data[j].x * h_net_force.data[j].x;
                proj += h_step.data[j].y * h_net_force.data[j].y;
                proj += h_step.data[j].z * h_net_force.data[j].z;
                }
            } 
        if (proj > 0)
            {
            // Step points uphill, reverse it
            for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
                {
                std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
                unsigned int group_size = current_group->getIndexArray().getNumElements();
                for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                    {
                    unsigned int j = current_group->getMemberIndex(group_idx);
                    h_step.data[j].x *= -1;
                    h_step.data[j].y *= -1;
                    h_step.data[j].z *= -1;
                    }
                }
            }
        }

    // Make sure the step length does not exceed tha maximum
    Scalar step_length(0.0);
    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
        unsigned int group_size = current_group->getIndexArray().getNumElements();
        total_group_size += group_size;
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = current_group->getMemberIndex(group_idx);
            step_length += h_step.data[j].x * h_step.data[j].x;
            step_length += h_step.data[j].y * h_step.data[j].y;
            step_length += h_step.data[j].z * h_step.data[j].z;
            }
        }
    step_length = sqrt(step_length);

    if (step_length > m_max_step)
        {
        // Step is too big, reduce to the maximum
        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
            unsigned int group_size = current_group->getIndexArray().getNumElements();
            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);
                h_step.data[j].x *= m_max_step/step_length;
                h_step.data[j].y *= m_max_step/step_length;
                h_step.data[j].z *= m_max_step/step_length;
                }
            }
        }

    // Apply the step
    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
        unsigned int group_size = current_group->getIndexArray().getNumElements();
        total_group_size += group_size;
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = current_group->getMemberIndex(group_idx);
            h_pos.data[j].x += h_step.data[j].x;
            h_pos.data[j].y += h_step.data[j].y;
            h_pos.data[j].z += h_step.data[j].z;
            }
        }
        if (m_iter < m_updates - 1)
            ++m_iter;
    }

void export_LBFGSEnergyMinimizer(py::module& m)
    {
    py::class_<LBFGSEnergyMinimizer, std::shared_ptr<LBFGSEnergyMinimizer> >(m, "LBFGSEnergyMinimizer", py::base<IntegratorTwoStep>())
        .def(py::init< std::shared_ptr<SystemDefinition>, Scalar>())
        .def("reset", &LBFGSEnergyMinimizer::reset)
        .def("hasConverged", &LBFGSEnergyMinimizer::hasConverged)
        .def("hasFailed", &LBFGSEnergyMinimizer::hasFailed)
        .def("getEnergy", &LBFGSEnergyMinimizer::getEnergy)
        .def("setDguess", &LBFGSEnergyMinimizer::setDguess)
        .def("setEtol", &LBFGSEnergyMinimizer::setEtol)
        .def("setFtol", &LBFGSEnergyMinimizer::setFtol)
        .def("setMaxDecrease", &LBFGSEnergyMinimizer::setMaxDecrease)
        .def("setMaxErise", &LBFGSEnergyMinimizer::setMaxErise)
        .def("setMaxFails", &LBFGSEnergyMinimizer::setMaxFails)
        .def("setMaxStep", &LBFGSEnergyMinimizer::setMaxStep)
        .def("setScale", &LBFGSEnergyMinimizer::setScale)
        .def("setUpdates", &LBFGSEnergyMinimizer::setUpdates)
        .def("setWtol", &LBFGSEnergyMinimizer::setWtol)
        ;
    }
