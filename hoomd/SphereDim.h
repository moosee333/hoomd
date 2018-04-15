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

        trans_q(v) = q_l*v*q_r,

    where * is the quaternion multiplication.

    On the 2-sphere, we only need a single quaternion, and we take q_l = conj(q_r) = q.

    For more details, see Sinkovits, Barr and Luijten JCP 136, 144111 (2012).
 */

struct __attribute__((visibility("default"))) SphereDim
    {
    public:
        /*! Define spherical boundary conditions
            \param R Radius of the (hyper-) sphere
         */
        SphereDim(Scalar _R, bool two_sphere)
            : R(_R) {}

        /*! Project a hyperspherical particle coordinate back onto an equatatorial hyperplane,
            using the stereographic projection

            \param q_l The first unit quaternion specifying particle position and orientation
            \param q_r The second unit quaternion specifying particle position and orientation
            \param lower if True, project around the lower pole [(-1,0,0,0) in 4d, (-1,0,0) in 3d], use
                         upper pole otherwise
            \returns the projection as a 3-vector
         */
        template<class Real>
        sphereToHyperplane(const quat<Real> q_l, const quat<Real> q_r, bool lower)
            {
            quat<Real> q = q_l*quat<Real>(R,vec3<Scalar>(0,0,0))*q_r;

            return lower ? q.v/(Real(1.0)+q.s) : q.v/(Real(1.0)-q.s);
            }

    private:
        Scalar R;        //!< Hypersphere radius
        bool two_sphere; //!< True if this is a 2-sphere, false for 3-sphere
    };
#undef HOSTDEVICE
