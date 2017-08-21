// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jwrm

#include "IntegratorTwoStep.h"

#include <memory>

#ifndef __LBFGS_ENERGY_MINIMIZER_H__
#define __LBFGS_ENERGY_MINIMIZER_H__

/*! \file LBFGSEnergyMinimizer.h
    \brief Declares the LBFGS energy minimiser class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Finds the nearest basin in the potential energy landscape
/*! \b Overview

    \ingroup updaters
*/
class LBFGSEnergyMinimizer : public IntegratorTwoStep
    {
    public:
        //! Constructs the minimizer and associates it with the system
        LBFGSEnergyMinimizer(std::shared_ptr<SystemDefinition>,  Scalar);
        virtual ~LBFGSEnergyMinimizer();

        //! Reset the minimization
        virtual void reset();

        //! Perform one minimization iteration
        virtual void update(unsigned int);

        //! Return whether or not the minimization has converged
        bool hasConverged() const {return m_converged;}

        //! Return whether or not the minimiser has got stuck
        bool hasFailed() const {return m_failed;}

        //! Return the potential energy after the last iteration
        Scalar getEnergy() const
            {
            if (m_was_reset)
                {
                m_exec_conf->msg->warning() << "LBFGS has just been initialized. Return energy==0."
                    << std::endl;
                return Scalar(0.0);
                }

            return m_energy_total;
            }

        //! Set the intial guess for the diagonal elements of the inverse Hessian
        void setDguess(Scalar dguess);

        //! Set the stopping criterion based on the change in energy between successive iterations
        /*! \param etol is the new energy tolerance to set
        */
        void setEtol(Scalar etol) {m_etol = etol;}

        //! Set the stopping criterion based on the total force on all particles in the system
        /*! \param ftol is the new force tolerance to set
        */
        void setFtol(Scalar ftol) {m_ftol = ftol;}

        //! Set the number of step size decreases before a step fails
        /*! \param decrease is the new number of decreases to set
        */
        void setMaxDecrease(unsigned int decrease) {m_max_decrease = decrease;}

        //! Set the maximum energy rise permitted in the step direction
        */
        void setMaxErise(Scalar erise);

        //! Set the maximum number of step failures before the minimisation is considered to have failed
        /*! \param fails is the new number of failures
        */
        void setMaxFails(unsigned int fails) {m_max_fails = fails;}

        //! Set the maximum permitted step size
        void setMaxStep(Scalar step);

        //! Set the scale factor for decreasing the step size
        void setScale(Scalar scale);

        //! Set the number of previous steps used when calculating the step direction
        /*! \param updates is the new number of steps
        */
        void setUpdates(unsigned int updates) {m_updates = updates;}

        //! Set the stopping criterion based on the total torque on all particles in the system
        /*! \param wtol is the new torque tolerance to set
        */
        void setWtol(Scalar wtol) {m_wtol = wtol;}

        //! Get needed pdata flags
        /*! LBFGSEnergyMinimzer needs the potential energy, so its flag is set
        */
        virtual PDataFlags getRequestedPDataFlags()
            {
            PDataFlags flags = IntegratorTwoStep::getRequestedPDataFlags();
            flags[pdata_flag::potential_energy] = 1;
            return flags;
            }

    protected:

        // Minimiser parameters
        Scalar m_dguess;                    //!< initial guess for the diagonal elements of the inverse Hessian
        Scalar m_etol;                      //!< stopping tolerance based on the chance in energy
        Scalar m_ftol;                      //!< stopping tolerance based on total force
        unsigned int m_max_decrease;        //!< number of decreases in the step size before the step is considered to have failed
        Scalar m_max_erise;                 //!< largest permitted energy increase in the step direction
        unsigned int m_max_fails;           //!< number of step failures before the minimisation is considered to have failed
        Scalar m_max_step;                  //!< largest permitted step size
        Scalar m_scale;                     //!< factor to decrease the step size by on an energy increase
        unsigned int m_updates;             //!< number of previous steps to consider when calculating the step direction
//      Scalar m_wtol;                      //!< stopping tolerance based on total torque

        // State of the minimiser
        bool m_converged;                   //!< whether the minimisation has converged
        Scalar m_energy_total;              //!< Total energy of all integrator groups
        bool m_failed;                      //!< whether the minimisation has failed
        unsigned int m_iter;                //!< number of previous steps available
        unsigned int m_no_decrease;         //!< current number of step decreases
        unsigned int m_no_fails;            //!< number of step failures experienced
        bool m_was_reset;                   //!< whether or not the minimizer was reset
        GPUArray<Scalar4> m_pos_history;    //!< history of positions
        GPUArray<Scalar3> m_grad_history;   //!< history of gradients
        GPUArray<Scalar> m_rho_history;      //!< rho history, intermediate value in step calculation
        GPUArray<Scalar3> m_step;           //!< calculated step direction

    private:

    };

//! Exports the LBFGSEnergyMinimizer class to python
void export_LBFGSEnergyMinimizer(pybind11::module& m);

#endif // #ifndef __LBFGS_ENERGY_MINIMIZER_H__