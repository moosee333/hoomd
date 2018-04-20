// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file SphereResizeUpdater.h
    \brief Declares an updater that resizes the simulation sphere of the system
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "Updater.h"
#include "Variant.h"

#include <memory>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#pragma once

//! Updates the simulation sphere over time, for hyperspherical simulations
/*! This simple updater gets the sphere radius froe a specified variant and sets the sphere size
    over time. Particles always remain on the sphere.

    \ingroup updaters
*/
class PYBIND11_EXPORT SphereResizeUpdater : public Updater
    {
    public:
        //! Constructor
        SphereResizeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<Variant> R);

        //! Destructor
        virtual ~SphereResizeUpdater();

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

    private:
        std::shared_ptr<Variant> m_R;    //!< Sphere radius vs time
    };

//! Export the SphereResizeUpdater to python
void export_SphereResizeUpdater(pybind11::module& m);

