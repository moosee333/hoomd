// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

/*! \file SphereDim.h
    \brief Defines the SphereDim class
*/

#pragma once

#include "HOOMDMath.h"
#include "VectorMath.h"

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

//! Stores the radius and dimensionality of the (hyper-)sphere, on which particle coordinates are defined
/*! This class stores the (hyper-)sphere radius of the spherical coordinate system on which the simulation
    is carried out, in three or four embedding dimensions. It also provides some helper methods
    to project particles back into 2d or 3d.

    On the 3-sphere, coordinates are stored as a set of two unit quaternions (q_l and q_r), which transform a four vector, such as a particle
    director or a particle center position  v, like this:

        q'(v) = q_l*v*q_r,

    where * is the quaternion multiplication.

    On the 2-sphere, we only need a single quaternion, and we take q_l = conj(q_r) = q.

    For the 2-sphere, the standard position is (0,0,0,R), i.e., parallel to the z-axis. This keeps the quaternion real under rotations.
    On the 3-sphere, the standard position is a purely imaginary quaternion, (R,0,0,0).

    On the hypersphere, improper transformations would require storing an extra parity bit and are currently not implemented.

    For more details, see Sinkovits, Barr and Luijten JCP 136, 144111 (2012).
 */

struct __attribute__((visibility("default"))) SphereDim
    {
    public:
        //! Default constructor
        SphereDim()
            : R(1.0), two_sphere(false)
            { }

        /*! Define spherical boundary conditions
            \param R Radius of the (hyper-) sphere
            \param two_sphere True if 2-sphere, false if 3-sphere
         */
        SphereDim(Scalar _R, bool _two_sphere)
            : R(_R), two_sphere(_two_sphere) {}

        //! Get the sphere radius
        Scalar getR() const
            {
            return R;
            }

        //! Set the sphere radius
        void setR(Scalar _R)
            {
            R = _R;
            }

        //! Return the simulation volume
        /* \param two_d True if on a 2-sphere, 3-sphere otherwise
         */
        Scalar getVolume()
            {
            if (two_sphere)
                return Scalar(4.0*M_PI*R*R);
            else
                return Scalar(2.0*M_PI*M_PI*R*R*R);
            }

        /*! Convert a hyperspherical coordinate into a cartesian one

            \param q_l The first unit quaternion specifying particle position and orientation
            \param q_r The second unit quaternion specifying particle position and orientation
            \returns the projection as a 3-vector
         */
        template<class Real>
        quat<Real> sphericalToCartesian(const quat<Real>& q_l, const quat<Real>& q_r) const
            {
            return two_sphere ?
                q_l*quat<Real>(0,vec3<Scalar>(0,0,R))*q_r : q_l*quat<Real>(R,vec3<Real>(0,0,0))*q_r;
            }

    private:
        Scalar R;        //!< Hypersphere radius
        bool two_sphere; //!< True if 2-sphere, false if 3-sphere
    };
#undef HOSTDEVICE
