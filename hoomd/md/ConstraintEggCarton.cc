// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "ConstraintEggCarton.h"
#include "EvaluatorConstraintEggCarton.h"

using namespace std;
namespace py = pybind11;

/*! \file ConstraintEggCarton.cc
    \brief Contains code for the ConstraintEggCarton class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group Group of particles on which to apply this constraint
    \param xFreq Number of cosine waves in the X direction.
    \param xFreq Number of cosine waves in the Y direction.
    \param xHeight Amplitude of cosine wave pattern in X direction.
    \param yHeight Amplitude of cosine wave pattern in Y direction.
*/
ConstraintEggCarton::ConstraintEggCarton(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
                                   int xFreq,
                                   int yFreq,
                                   Scalar xHeight,
                                   Scalar yHeight)
        : Updater(sysdef), m_group(group), m_xFreq(xFreq), m_yFreq(yFreq),
          m_xHeight(xHeight), m_yHeight(yHeight)
    {
    m_exec_conf->msg->notice(5) << "Constructing ConstraintEggCarton" << endl;

    validate();
    }

ConstraintEggCarton::~ConstraintEggCarton()
    {
    m_exec_conf->msg->notice(5) << "Destroying ConstraintEggCarton" << endl;
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
void ConstraintEggCarton::update(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    if (m_prof) m_prof->push("ConstraintEggCarton");

    assert(m_pdata);

    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    EvaluatorConstraintEggCarton EggCarton(m_xFreq, m_yFreq, m_xHeight, m_yHeight);
    // for each of the particles in the group
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        // get the current particle properties
        unsigned int j = m_group->getMemberIndex(group_idx);
        Scalar3 X = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
        BoxDim boxD = m_pdata->getBox();
        Scalar3 hi = boxD.getHi();
        Scalar3 box = make_scalar3(hi.x,hi.y,hi.z);

        // evaluate the constraint position
        Scalar3 C = EggCarton.evalClosest(X, box);

        // apply the constraint
        h_pos.data[j].x = C.x;
        h_pos.data[j].y = C.y;
        h_pos.data[j].z = C.z;
        }

    if (m_prof)
        m_prof->pop();
    }

/*! Print warning messages if the Egg Carton is outside the box.
    Generate an error if any particle in the group is not near the Egg Carton.
*/
void ConstraintEggCarton::validate()
    {
    BoxDim box = m_pdata->getBox();
    Scalar3 hi = box.getHi();

    if (abs(m_xHeight) > abs(hi.z) || abs(m_yHeight) > abs(hi.z))
        {
        m_exec_conf->msg->warning() << "constrain.egg_carton: egg carton constraint is taller than box."
             << endl;
        }

    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);

    EvaluatorConstraintEggCarton EggCarton(m_xFreq, m_yFreq, m_xHeight, m_yHeight);
    // for each of the particles in the group
    bool errors = false;
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        // get the current particle properties
        unsigned int j = m_group->getMemberIndex(group_idx);

        Scalar3 X = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
        BoxDim boxD = m_pdata->getBox();
        Scalar3 hi = boxD.getHi();
        Scalar3 box = make_scalar3(hi.x,hi.y,hi.z);
        Scalar3 C = EggCarton.evalClosest(X,box);
        Scalar3 V;
        V.x = C.x - X.x;
        V.y = C.y - X.y;
        V.z = C.z - X.z;
        Scalar dist = slow::sqrt(V.x*V.x + V.y*V.y + V.z*V.z);

        if (dist > Scalar(1.0))
            {
            m_exec_conf->msg->error() << "constrain.egg_carton: Particle " << h_tag.data[j] << " is more than 1 unit of"
                                      << " distance from the closest point on the egg carton constraint. Position: " 
                                      <<X.x<<", "<<X.y<<", "<<X.z<< ". Closest Point: "<<C.x<<", "<<C.y<< ", "<<C.z<<endl;
            errors = true;
            }

        if (h_body.data[j] != NO_BODY)
            {
            m_exec_conf->msg->error() << "constrain.egg_carton: Particle " << h_tag.data[j] << " belongs to a rigid body"
                                      << " - cannot constrain" << endl;
            errors = true;
            }
        }

    if (errors)
        {
        throw std::runtime_error("Invalid constraint specified");
        }
    }


void export_ConstraintEggCarton(py::module& m)
    {
    py::class_< ConstraintEggCarton, std::shared_ptr<ConstraintEggCarton> >(m, "ConstraintEggCarton", py::base<Updater>())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, int, int, Scalar, Scalar >())
    ;
    }
