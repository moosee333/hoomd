// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer:

#ifndef __PAIR_EVALUATOR_DLVO_H__
#define __PAIR_EVALUATOR_DLVO_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairDLVO.h
    \brief Defines the pair evaluator class for DLVO potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the DLVO pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ.

    <b>DLVO specifics</b>

    EvaluatorPairDLVO evaluates the function:
    \begin{eqnarray*}
    V_{\mathrm{DLVO}}(r) = & \left(\frac{4 \pi \varepsilon_r \varepsilon_0}{z^2 e^2}\right)(2 k_\mathrm{B} T)^2 \left(\frac{2 + \kappa d_i}{1 + \kappa d_i} \right)\left(\frac{2 + \kappa d_j}{1 + \kappa d_j} \right) \tanh{\left(\frac{z e \zeta_i}{4 k_\mathrm{B} T} \right)} \tanh{\left(\frac{z e \zeta_j}{4 k_\mathrm{B} T} \right)} \cdot \frac{d_i d_j}{r} \exp{(-\kappa (r - \Delta_+))} \\
                           & -\frac{A_\mathrm{H}}{12} \left(\frac{d_i d_j}{r^2 - \Delta_-^2} + 2 \ln{\left(\frac{r^2 - \Delta_+^2}{r^2 - \Delta_-^2}\right)}\right) & r < r_{\mathrm{cut}} \\
                         = & 0 & r > r_{\mathrm{cut}} \\
    \end{eqnarray*}

    The DLVO potential does not need charge, but it does need diameter. Three parameters are specified and stored in a Scalar4, for speed. \a prefactorEL is
    placed in \a params.x, \a prefactorVDW is in \a params.y and \a kappa is in \a params.z.
    \a params.w is always set to zero, and is ignored.

    These are related to the DLVO parameters by:
    - \a prefactorEL = (4.0 * kT * kT)/(z * z * elementary_charge * elementary_charge) * math.tanh((z * eV_i)/(4.0 * kT)) math.tanh((z * eV_j)/(4.0 * kT));
    - \a prefactorVDW = A_H/12.0;

*/
class EvaluatorPairDLVO
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar4 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance beteen the particles
            \param _rcutsq Sqauared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairDLVO(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), prefactorEL(_params.x), prefactorVDW(_params.y), kappa(_params.z)
            {
            }

        //! DLVO uses diameter
        DEVICE static bool needsDiameter() { return true; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) {

            deltap = (di+dj) / Scalar(2.0);
            deltam = (di-dj) / Scalar(2.0);
        }

        //! DLVO doesn't use charge
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional diameter values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

        //! Evaluate the force and energy
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed
                  in PotentialPair.

            \return True if they are evaluated or false if they are not because we are beyond the cuttoff
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq && (prefactorEL != 0 && prefactorVDW != 0))
                {
                Scalar rinv = fast::rsqrt(rsq);
                Scalar r = Scalar(1.0) / rinv;
                Scalar r2inv = Scalar(1.0) / rsq;

                Scalar exp_val = fast::exp(-kappa * (r - deltap));
                Scalar curv = ((2.0 + kappa * di) * (2.0 + kappa * dj)) / ((1.0 + kappa * di) * (1.0 + kappa * dj));

                force_divr = prefactorEL * curv * exp_val * di * dj * r2inv * (rinv + kappa);
                force_divr += prefactorVDW * Scalar(512.0) * di * di * di * dj * dj * dj / ((di * di * di * di + (dj * dj - Scalar(4.0) * di * di) * (dj * dj - Scalar(4.0) * di * di) - Scalar(2.0) * di * di * (dj * dj + Scalar(4.0) * r * r)) * (di * di * di * di + (dj * dj - Scalar(4.0) * di * di) * (dj * dj - Scalar(4.0) * di * di) - Scalar(2.0) * di * di * (dj * dj + Scalar(4.0) * r * r)));

                pair_eng = prefactorEL * curv * exp_val * di * dj * rinv * exp_val;
                pair_eng += - prefactorVDW * ((di * dj) / (r * r - deltap * deltap) + (di * dj) / (r * r - deltam * deltam) + Scalar(2.0) * log((r * r - deltap * deltap) / (r * r - deltam * deltam)));

                if (energy_shift)
                    {
                    Scalar rcutinv = fast::rsqrt(rcutsq);
                    Scalar rcut = Scalar(1.0) / rcutinv;

                    pair_eng -= prefactorEL * curv * fast::exp(-kappa * (rcut - deltap)) * di * dj * rcutinv;
                    pair_eng -= - prefactorVDW * ((di * dj) / (rcut * rcut - deltap * deltap) + (di * dj) / (rcut * rcut - deltam * deltam) + Scalar(2.0) * log((rcut * rcut - deltap * deltap) / (rcut * rcut - deltam * deltam)));

                    }
                return true;
                }
            else
                return false;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("DLVO");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar prefactorEL;     //!< prefactorEL parameter extracted from the params passed to the constructor
        Scalar prefactorVDW;     //!< prefactorVDW parameter extracted from the params passed to the constructor
        Scalar deltap;  //!< Deltap parameter extracted from the call to setDiameter
        Scalar deltam; //!< Deltam parameter extracted from the call to setDiameter
        Scalar kappa;   //!< kappa parameter extracted from the params passed to the constructor
        Scalar di;
        Scalar dj;
    };


#endif // __PAIR_EVALUATOR_LJ_H__
