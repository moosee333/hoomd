// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#ifndef __EVALUATOR_PAIR_GB_H__
#define __EVALUATOR_PAIR_GB_H__

#ifndef NVCC
#include <string>
#endif

#define HOOMD_GB_MIN(i,j) ((i > j) ? j : i)
#define HOOMD_GB_MAX(i,j) ((i > j) ? i : j)

#include "hoomd/VectorMath.h"

/*! \file EvaluatorPairGB.h
    \brief Defines a an evaluator class for the Gay-Berne potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

/*!
 * Gay-Berne potential as formulated by Allen and Germano,
 * with shape-independent energy parameter, for identical uniaxial particles.
 */

class EvaluatorPairGB
    {
    public:
        typedef Scalar3 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _dr Displacement vector between particle centres of mass
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _q_i Quaterion of i^th particle
            \param _q_j Quaterion of j^th particle
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairGB(Scalar3& _dr,
                               Scalar4& _qi,
                               Scalar4& _qj,
                               Scalar _rcutsq,
                               param_type& _params)
            : dr(_dr),rcutsq(_rcutsq),qi(_qi),qj(_qj),
              epsilon(_params.x), lperp(_params.y), lpar(_params.z)
            {
            }

        //! uses diameter
        DEVICE static bool needsDiameter()
            {
            return false;
            }

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj){}

        //! whether pair potential requires charges
        DEVICE static bool needsCharge( )
            {
            return false;
            }

        //! Accept the optional diameter values
        //! This function is pure virtual
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj){}

        //! Evaluate the force and energy
        /*! \param force Output parameter to write the computed force.
            \param pair_eng Output parameter to write the computed pair energy.
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff.
            \param torque_i The torque exterted on the i^th particle.
            \param torque_j The torque exterted on the j^th particle.
            \return True if they are evaluated or false if they are not because we are beyond the cutoff.
        */
        DEVICE  bool
        evaluate(Scalar3& force, Scalar& pair_eng, bool energy_shift, Scalar3& torque_i, Scalar3& torque_j)
            {
            Scalar rsq = dot(dr,dr);
            Scalar r = fast::sqrt(rsq);
            vec3<Scalar> unitr = fast::rsqrt(dot(dr,dr))*dr;

            // obtain rotation matrices (space->body)
            rotmat3<Scalar> rotA(conj(qi));
            rotmat3<Scalar> rotB(conj(qj));

            // last row of rotation matrix
            vec3<Scalar> a3 = rotA.row2;
            vec3<Scalar> b3 = rotB.row2;

            Scalar ca = dot(a3,unitr);
            Scalar cb = dot(b3,unitr);
            Scalar cab = dot(a3,b3);
            Scalar lperpsq = lperp*lperp;
            Scalar lparsq = lpar*lpar;
            Scalar chi=(lparsq - lperpsq)/(lparsq+lperpsq);
            Scalar chic = chi*cab;

            Scalar chi_fact = chi/(Scalar(1.0)-chic*chic);
            vec3<Scalar> kappa = Scalar(1.0/2.0)*r/lperpsq
                *(unitr - chi_fact*((ca-chic*cb)*a3+(cb-chic*ca)*b3));

            Scalar phi = Scalar(1.0/2.0)*dot(dr, kappa)/rsq;
            Scalar sigma = fast::rsqrt(phi);

            Scalar sigma_min = Scalar(2.0)*HOOMD_GB_MIN(lperp,lpar);

            Scalar zeta = (r-sigma+sigma_min)/sigma_min;
            Scalar zetasq = zeta*zeta;

            Scalar rcut = fast::sqrt(rcutsq);
            Scalar dUdphi,dUdr;

            // define r_cut to be along the long axis
            Scalar sigma_max = Scalar(2.0)*HOOMD_GB_MAX(lperp,lpar);
            Scalar zetacut = (rcut-sigma_max+sigma_min)/sigma_min;
            Scalar zetacutsq = zetacut*zetacut;

            // compute the force divided by r in force_divr
            if (zetasq < zetacutsq && epsilon != Scalar(0.0))
                {
                Scalar zeta2inv = Scalar(1.0)/zetasq;
                Scalar zeta6inv = zeta2inv * zeta2inv *zeta2inv;

                dUdr  = -Scalar(24.0)*epsilon*(zeta6inv/zeta*(Scalar(2.0)*zeta6inv-Scalar(1.0)))/sigma_min;
                dUdphi = dUdr*Scalar(1.0/2.0)*sigma*sigma*sigma;

                pair_eng = Scalar(4.0)*epsilon*zeta6inv * (zeta6inv - Scalar(1.0));

                if (energy_shift)
                    {
                    Scalar zetacut2inv = Scalar(1.0)/zetacutsq;
                    Scalar zetacut6inv = zetacut2inv * zetacut2inv * zetacut2inv;
                    pair_eng -= Scalar(4.0)*epsilon*zetacut6inv * (zetacut6inv - Scalar(1.0));
                    }
                }
            else
                return false;

            // compute vector force and torque
            Scalar r2inv = Scalar(1.0)/rsq;
            vec3<Scalar> fK = -r2inv*dUdphi*kappa;
            vec3<Scalar> f = -dUdr*unitr + fK + r2inv*dUdphi*unitr*dot(kappa,unitr);
            force = vec_to_scalar3(f);

            vec3<Scalar> rca = Scalar(1.0/2.0)* (-dr - r*chi_fact*((ca-chic*cb)*a3-(cb-chic*ca)*b3));
            vec3<Scalar> rcb = rca + dr;
            torque_i = vec_to_scalar3(cross(rca, fK));
            torque_j = -vec_to_scalar3(cross(rcb, fK));

            return true;
            }

        #ifndef NVCC
        //! Get the name of the potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return "gb";
            }
        #endif

    protected:
        vec3<Scalar> dr;   //!< Stored dr from the constructor
        Scalar rcutsq;     //!< Stored rcutsq from the constructor
        quat<Scalar> qi;   //!< Orientation quaternion for particle i
        quat<Scalar> qj;   //!< Orientation quaternion for particle j
        Scalar epsilon;    //!< Energy parameter
        Scalar lperp;      //!< Short axis length
        Scalar lpar;       //!< Longt axis length
    };


#undef HOOMD_GB_MIN
#undef HOOMD_GB_MAX
#endif // __EVALUATOR_PAIR_GB_H__
