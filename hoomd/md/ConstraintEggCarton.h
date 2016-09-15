// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/ParticleGroup.h"
#include "hoomd/Updater.h"

/*! \file ConstraintEggCarton.h
    \brief Declares a class for computing egg carton constraint forces
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif
    
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __CONSTRAINT_EggCarton_H__
#define __CONSTRAINT_EggCarton_H__

//! Applys a constraint force to keep a group of particles on an Egg carton
/*! \ingroup computes
*/
class ConstraintEggCarton : public Updater
    {
    public:
        //! Constructs the compute
        ConstraintEggCarton(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,
                         int xFreq,
                         int yFreq,
                         Scalar xHeight,
                         Scalar yHeight);

        //! Destructor
        virtual ~ConstraintEggCarton();

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

    protected:
        std::shared_ptr<ParticleGroup> m_group;   //!< Group of particles on which this constraint is applied
        int m_xFreq;          //!< Number of cosine waves in the X direction.
        int m_yFreq;          //!< Number of cosine waves in the Y direction.
        Scalar m_xHeight;          //!< Amplitude of cosine wave pattern in X direction.
        Scalar m_yHeight;          //!< Amplitude of cosine wave pattern in Y direction.

    private:
        //! Validate that the egg carton is in the box and all particles are very near the constraint
        void validate();
    };

//! Exports the ConstraintEggCarton class to python
void export_ConstraintEggCarton(pybind11::module& m);
#endif
