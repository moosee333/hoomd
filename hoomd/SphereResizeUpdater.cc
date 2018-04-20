// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

/*! \file SphereResizeUpdater.cc
    \brief Defines the SphereResizeUpdater class
*/


#include "SphereResizeUpdater.h"

#include "SphereDim.h"

#include <math.h>
#include <iostream>
#include <stdexcept>

using namespace std;
namespace py = pybind11;

/*! \param sysdef System definition containing the particle data to set the sphere size on
    \param R radius of the sphere over time

    The default setting is to scale particle positions along with the sphere.
*/
SphereResizeUpdater::SphereResizeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<Variant> R)
    : Updater(sysdef), m_R(R)
    {
    assert(m_pdata);
    assert(m_R);

    m_exec_conf->msg->notice(5) << "Constructing SphereResizeUpdater" << endl;
    }

SphereResizeUpdater::~SphereResizeUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying SphereResizeUpdater" << endl;
    }

/*! Rescales the simulation sphere
    \param timestep Current time step of the simulation
*/
void SphereResizeUpdater::update(unsigned int timestep)
    {
    m_exec_conf->msg->notice(10) << "Sphere resize update" << endl;
    if (m_prof) m_prof->push("SphereResize");

    // first, compute what the current sphere size and tilt factors should be
    Scalar R = m_R->getValue(timestep);

    SphereDim sphere = m_pdata->getSphere();
    sphere.setR(R);

    // set the new sphere
    m_pdata->setSphere(sphere);

    if (m_prof) m_prof->pop();
    }

void export_SphereResizeUpdater(py::module& m)
    {
    py::class_<SphereResizeUpdater, std::shared_ptr<SphereResizeUpdater> >(m,"SphereResizeUpdater",py::base<Updater>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
     std::shared_ptr<Variant> >());
    }
