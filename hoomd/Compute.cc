// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander



#include "Compute.h"

namespace py = pybind11;


#include <iostream>
#include <stdexcept>
using namespace std;

/*! \file Compute.cc
    \brief Contains code for the Compute class
*/

/*! \param sysdef SystemDefinition this compute will act on. Must not be NULL.
    \post The Compute is constructed with the given particle data and a NULL profiler.
*/
Compute::Compute(std::shared_ptr<SystemDefinition> sysdef) : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()),
        exec_conf(m_pdata->getExecConf()), m_force_compute(false), m_last_computed(0), m_first_compute(true)
    {
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    m_exec_conf = exec_conf;
    }

/*! \param num_iters Number of iterations to average for the benchmark
    \returns Milliseconds of execution time per calculation
    Derived classes can optionally implement this method. */
double Compute::benchmark(unsigned int num_iters)
    {
    m_exec_conf->msg->error() << "This compute doesn't support benchmarking" << endl;
    throw runtime_error("Error benchmarking compute");
    return 0.0;
    }

/*! It is useful for the user to know where computation time is spent, so all Computes
    should profile themselves. This method sets the profiler for them to use.
    This method does not need to be called, as Computes will not profile themselves
    on a NULL profiler
    \param prof Pointer to a profiler for the compute to use. Set to NULL
        (std::shared_ptr<Profiler>()) to stop the
        analyzer from profiling itself.
    \note Derived classes MUST check if m_prof is set before calling any profiler methods.
*/
void Compute::setProfiler(std::shared_ptr<Profiler> prof)
    {
    m_prof = prof;
    }

/*! \param timestep Current time step
    \returns true if computations should be performed, false if they have already been done
        at this \a timestep.
    \note This method is designed to only be called once per call to compute() like so:
\code
void SomeClass::compute(unsgned int timestep)
    {
    if (!shouldCompute(timestep))
        return;
    ... do compute tasks
    }
\endcode
*/
bool Compute::shouldCompute(unsigned int timestep)
    {
    // handle case where no computation has been performed yet
    if (m_first_compute)
        {
        m_first_compute = false;
        m_last_computed = timestep;
        return true;
        }

    // Update if computation is enforced, but leave m_last_computed unchanged
    // (such that after a forced compute(), still a regular compute() is possible)
    if (m_force_compute)
        {
        m_force_compute = false;
        return true;
        }

    // otherwise, we update if the last computed timestep is less than the current
    if (m_last_computed != timestep)
        {
        m_last_computed = timestep;
        return true;
        }


    // failing the above, we perform no computation
    return false;
    }

void Compute::forceCompute(unsigned int timestep)
    {
    m_force_compute = true;

    compute(timestep);
    }



void export_Compute(py::module& m)
    {
    py::class_<Compute, std::shared_ptr<Compute> >(m,"Compute")
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    .def("compute", &Compute::compute)
    .def("benchmark", &Compute::benchmark)
    .def("printStats", &Compute::printStats)
    .def("setProfiler", &Compute::setProfiler)
    ;
    }
