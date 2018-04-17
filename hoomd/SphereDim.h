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

    Improper transformations (reflections) on the two-sphere are accounted for by a negative sign of the left quaternion.

    On the thre-sphere, improper transformations require an extra conjugation bit and are currently not implemented.

    For more details, see Sinkovits, Barr and Luijten JCP 136, 144111 (2012).
 */

struct __attribute__((visibility("default"))) SphereDim
    {
    public:
        //! Default constructor
        SphereDim()
            : R(1.0)
            { }

        /*! Define spherical boundary conditions
            \param R Radius of the (hyper-) sphere
         */
        SphereDim(Scalar _R)
            : R(_R) {}

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

        /*! Convert a hyperspherical coordinate into a cartesian one

            \param q_l The first unit quaternion specifying particle position and orientation
            \param q_r The second unit quaternion specifying particle position and orientation
            \returns the projection as a 3-vector
         */
        template<class Real>
        quat<Real> sphericalToCartesian(const quat<Real>& q_l, const quat<Real>& q_r) const
            {
            return q_l*quat<Real>(0,vec3<Scalar>(0,0,R))*q_r;
            }

    private:
        Scalar R;        //!< Hypersphere radius
    };
#undef HOSTDEVICE
