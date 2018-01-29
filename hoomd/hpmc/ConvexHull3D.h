// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/HOOMDMath.h"

/*! \file ConvexHull.h
    \brief A minimalist implementation of the 3D convex hull for use in device code
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE __attribute__((always_inline))
#endif

namespace hpmc
{

namespace detail
{

// Maintainer: jglaser

/* Adaption from http://tmc.web.engr.illinois.edu/pub.html

   Original comments:

   Timothy Chan    "ch3dquad.cc"    12/02    3-d lower hull (in C++)

   O(nh) gift-wrapping algorithm

   input/output: see "ch3d.cc"
   warning: ignores degeneracies and robustness
*/

template<unsigned int Nvert, class Vector, class VectorIt = const Vector *, class Real = Scalar>
class ConvexHull3D
    {
    public:
        //! Constructor
        /*! \param _points The points in 3D space
            \param _n Number of points
         */
        DEVICE ConvexHull3D(VectorIt points, unsigned int n)
            : m_points(points), m_n(n), m_h(0)
            { }

        //! Return maximum number of triangle vertices needed
        DEVICE static constexpr inline unsigned int getVertexStorageSize()
            {
            return 2*Nvert;
            }

        /*! \param gift-wrap the points and return triangle indices in

            \param I first triangle vertex
            \param J second triangle vertex
            \param K third triangle vertex

            \post The I,J,K indices contain the triangle vertex indices of the convex hull
         */
        DEVICE inline void compute(unsigned int *I, unsigned int *J, unsigned int *K)
            {
            // find initial edge ij
            unsigned int i, j, l;
            for (i = 0, l = 1; l < m_n; l++)
                if (m_points[i].x > m_points[l].x) i = l;
            for (j = i, l = 0; l < m_n; l++)
                if (i != l && turn(m_points[i],m_points[j],m_points[l]) >= 0)
                    j = l;

            m_I = I; m_J = J; m_K = K;
            m_h = 0;
            wrap(i,j);
            }

        //! \returns the number of generated facets
        DEVICE inline unsigned int getNumFacets() const
            {
            return m_h;
            }

    private:
        const VectorIt m_points;
        const unsigned int m_n;

        unsigned int *m_I;
        unsigned int *m_J;
        unsigned int *m_K;
        unsigned int m_h;

        //! Recursive gift wrapping algorithm
        DEVICE void wrap(unsigned int i, unsigned int j)
            {
            unsigned int k, l, m;
            for (m = 0; m < m_h; m++)  // check if facet hasn't been explored
                if ((m_I[m] == i && m_J[m] == j) || (m_J[m] == i && m_K[m] == j) ||
                    (m_K[m] == i && m_I[m] == j))
                    return;
             for (k = i, l = 0; l < m_n; l++)  // wrap from edge ij to find facet ijk
                if (turn(m_points[i],m_points[j],m_points[l]) < 0 && orient(m_points[i],m_points[j],m_points[k],m_points[l]) >= 0)
                    k = l;

            if (turn(m_points[i],m_points[j],m_points[k]) >= 0) return;
            m_I[m_h] = i;  m_J[m_h] = j;  m_K[m_h++] = k;
            wrap(k,j);  // explore adjacent facets
            wrap(i,k);
            }

        DEVICE inline Real turn(Vector p, Vector q, Vector r)
            {  // <0 iff cw
            return (q.x-p.x)*(r.y-p.y) - (r.x-p.x)*(q.y-p.y);
            }

        DEVICE inline Real orient(Vector p, Vector q, Vector r, Vector s)
            {
            // <0 iff s is above pqr, assuming pqr is cw
            return (q.z-p.z)*turn(p,r,s) - (r.z-p.z)*turn(p,q,s) + (s.z-p.z)*turn(p,q,r);
            }
    };

} // end namespace detail
} // end namespace hpmc
#undef DEVICE
