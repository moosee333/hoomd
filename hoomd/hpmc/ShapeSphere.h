// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/SphereDim.h"
#include "HPMCPrecisionSetup.h"
#include "hoomd/VectorMath.h"
#include "Moves.h"
#include "hoomd/AABB.h"

#include <stdexcept>

#ifndef __SHAPE_SPHERE_H__
#define __SHAPE_SPHERE_H__

/*! \file ShapeSphere.h
    \brief Defines the sphere shape
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

#define SMALL 1e-5

namespace hpmc
{

// put a few misc math functions here as they don't have any better home
namespace detail
    {
    // !helper to call CPU or GPU signbit
    template <class T> HOSTDEVICE inline int signbit(const T& a)
        {
        #ifdef __CUDA_ARCH__
        return ::signbit(a);
        #else
        return std::signbit(a);
        #endif
        }

    template <class T> HOSTDEVICE inline T min(const T& a, const T& b)
        {
        #ifdef __CUDA_ARCH__
        return ::min(a,b);
        #else
        return std::min(a,b);
        #endif
        }

    template <class T> HOSTDEVICE inline T max(const T& a, const T& b)
        {
        #ifdef __CUDA_ARCH__
        return ::max(a,b);
        #else
        return std::max(a,b);
        #endif
        }

    template<class T> HOSTDEVICE inline void swap(T& a, T&b)
        {
        T c;
        c = a;
        a = b;
        b = c;
        }
    }

//! Base class for parameter structure data types
struct param_base
    {
    //! Custom new operator
    static void* operator new(std::size_t sz)
        {
        void *ret = 0;
        int retval = posix_memalign(&ret, 32, sz);
        if (retval != 0)
            {
            throw std::runtime_error("Error allocating aligned memory");
            }

        return ret;
        }

    //! Custom new operator for arrays
    static void* operator new[](std::size_t sz)
        {
        void *ret = 0;
        int retval = posix_memalign(&ret, 32, sz);
        if (retval != 0)
            {
            throw std::runtime_error("Error allocating aligned memory");
            }

        return ret;
        }

    //! Custom delete operator
    static void operator delete(void *ptr)
        {
        free(ptr);
        }

    //! Custom delete operator for arrays
    static void operator delete[](void *ptr)
        {
        free(ptr);
        }

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr,unsigned int &available_bytes) const
        {
        // default implementation does nothing
        }
    };


//! Sphere shape template
/*! ShapeSphere implements IntegragorHPMC's shape protocol. It serves at the simplest example of a shape for HPMC

    The parameter defining a sphere is just a single Scalar, the sphere radius.

    \ingroup shape
*/
struct sph_params : param_base
    {
    OverlapReal radius;                 //!< radius of sphere
    unsigned int ignore;                //!< Bitwise ignore flag for stats, overlaps. 1 will ignore, 0 will not ignore
                                        //   First bit is ignore overlaps, Second bit is ignore statistics
    bool isOriented;                    //!< Flag to specify whether a sphere has orientation or not. Intended for
                                        //!  for use with anisotropic/patchy pair potentials.

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const
        {
        // default implementation does nothing
        }
    #endif
    } __attribute__((aligned(32)));

struct ShapeSphere
    {
    //! Define the parameter type
    typedef sph_params param_type;

    //! Initialize a shape with a given orientation
    DEVICE ShapeSphere(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), params(_params) {}

    //! Initialize a shape with a given left and right quaternion (hyperspherical coordinates)
    DEVICE ShapeSphere(const quat<Scalar>& _quat_l, const quat<Scalar>& _quat_r, const param_type& _params)
        : quat_l(_quat_l), quat_r(_quat_r), params(_params) {}

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() const
        {
        return params.isOriented;
        }

    //!Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const { return params.ignore; }

    //! Get the circumsphere diameter
    DEVICE OverlapReal getCircumsphereDiameter() const
        {
        return params.radius*OverlapReal(2.0);
        }

    //! Get the in-sphere radius
    DEVICE OverlapReal getInsphereRadius() const
        {
        return params.radius;
        }

    //! Return the bounding box of the shape in world coordinates
    template<class T>
    DEVICE detail::AABB getAABB(const T& pos) const
        {
        return detail::AABB(pos, params.radius);
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() { return false; }

    quat<Scalar> orientation;    //!< Orientation of the sphere (unused)

    quat<Scalar> quat_l;         //!< Left quaternion (for hyperspherical coordinates)
    quat<Scalar> quat_r;         //!< Left quaternion (for hyperspherical coordinates)

    const sph_params &params;        //!< Sphere and ignore flags
    };

//! Check if circumspheres overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
DEVICE inline bool check_circumsphere_overlap(const vec3<Scalar>& r_ab, const ShapeSphere& a,
    const ShapeSphere &b)
    {
    // for now, always return true
    return true;
    }

//! Define the general overlap function (cartesian version)
/*! This is just a convenient spot to put this to make sure it is defined early
    \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err Incremented if there is an error condition. Left unchanged otherwise.
    \returns true when *a* and *b* overlap, and false when they are disjoint
*/
template <class ShapeA, class ShapeB>
DEVICE inline bool test_overlap(const vec3<Scalar>& r_ab, const ShapeA &a, const ShapeB& b, unsigned int& err)
    {
    // default implementation returns true, will make it obvious if something calls this
    return true;
    }

//! Returns true if the shape overlaps with itself
template<class Shape>
DEVICE inline bool test_overlap_self_sphere(const Shape& shape, const SphereDim& sphere)
    {
    // default implementation returns true, will make it obvious if something calls this
    return true;
    }

//! Define the general overlap function (hyperspherical version)
/*! This is just a convenient spot to put this to make sure it is defined early
    \param a first shape
    \param b second shape
    \param sphere Boundary conditions
    \param err Incremented if there is an error condition. Left unchanged otherwise.
    \returns true when *a* and *b* overlap, and false when they are disjoint
*/
template <class ShapeA, class ShapeB>
DEVICE inline bool test_overlap_sphere(const ShapeA& a, const ShapeB& b, const SphereDim& sphere, unsigned int& err)
    {
    // default implementation returns true, will make it obvious if something calls this
    return true;
    }

//! Sphere-Sphere overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
template <>
DEVICE inline bool test_overlap<ShapeSphere, ShapeSphere>(const vec3<Scalar>& r_ab, const ShapeSphere& a, const ShapeSphere& b, unsigned int& err)
    {
    vec3<OverlapReal> dr(r_ab);

    OverlapReal rsq = dot(dr,dr);

    if (rsq < (a.params.radius + b.params.radius)*(a.params.radius + b.params.radius))
        {
        return true;
        }
    else
        {
        return false;
        }
    }

//! Returns true if the shape overlaps with itself
DEVICE inline bool test_overlap_self_sphere(const ShapeSphere& shape, const SphereDim& sphere)
    {
    return shape.params.radius >= Scalar(M_PI)*sphere.getR();
    }

//! Sphere-Sphere overlap on a hypersphere
/*!  \param a first shape
    \param b second shape
    \param sphere Boundary conditions
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
template <>
DEVICE inline bool test_overlap_sphere<ShapeSphere, ShapeSphere>(const ShapeSphere& a, const ShapeSphere& b, const SphereDim& sphere, unsigned int& err)
    {
    // transform spherical coordinates into 4d-cartesian ones
    quat<OverlapReal> pos_a = sphere.sphericalToCartesian(quat<OverlapReal>(a.quat_l),quat<OverlapReal>(a.quat_r));
    quat<OverlapReal> pos_b = sphere.sphericalToCartesian(quat<OverlapReal>(b.quat_l),quat<OverlapReal>(b.quat_r));

    // normalize
    OverlapReal inv_norm_a = fast::rsqrt(dot(pos_a,pos_a));
    OverlapReal inv_norm_b = fast::rsqrt(dot(pos_b,pos_b));

    pos_a.s *= inv_norm_a;
    pos_a.v *= inv_norm_a;

    pos_b.s *= inv_norm_b;
    pos_b.v *= inv_norm_b;

    // arc-length along a geodesic
    OverlapReal arc_length = sphere.getR()*fast::acos(dot(pos_a,pos_b));

    if (arc_length < (a.params.radius + b.params.radius))
        {
        return true;
        }
    else
        {
        return false;
        }
    }

}; // end namespace hpmc

#endif //__SHAPE_SPHERE_H__
