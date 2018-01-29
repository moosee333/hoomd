// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/HOOMDMath.h"

/*! \file ConvexHull3D.h
    \brief A straight-forward (but neither particularly robust, time- nor space efficient) implementation of the 3D convex hull
           using the gift wrapping algorithm, for use in device code.

           This algorithm uses fixed-size device storage in local memory.
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

/* Adaptation from http://tmc.web.engr.illinois.edu/pub.html

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
            : m_points(points), m_n(n), m_nfacet(0), m_lower(1.0)
            { }

        //! Return maximum number of triangles needed
        DEVICE static constexpr inline unsigned int getStorageSize()
            {
            return 4*Nvert; // worst case edges (lower + upper)
            }

        /*! \param gift-wrap the points and return triangle indices in

            \param I first triangle vertex
            \param J second triangle vertex
            \param K third triangle vertex

            \post The I,J,K indices contain the triangle vertex indices of the convex hull
         */
        DEVICE inline void compute(unsigned int *I, unsigned int *J, unsigned int *K)
            {
            m_I = I; m_J = J; m_K = K;
            m_nfacet = 0; // output size

            // lower convex hull
            m_lower = 1.0;

            // find initial edge ij
            unsigned int i, j, l;
            for (i = 0, l = 1; l < m_n; l++)
                if (m_points[i].x > m_points[l].x)
                    i = l;
            for (j = i, l = 0; l < m_n; l++)
                if (i != l && turn(m_points[i],m_points[j],m_points[l]) >= 0)
                    j = l;

            wrap(i,j);

            // upper hull
            m_I = I+m_nfacet; m_J = J+m_nfacet; m_K = K+m_nfacet;
            m_lower = -1;
            for (i = 0, l = 1; l < m_n; l++)
                if (m_points[i].x < m_points[l].x)
                    i = l;
            for (j = i, l = 0; l < m_n; l++)
                if (i != l && turn(m_points[i],m_points[j],m_points[l]) >= 0)
                    j = l;
            wrap(i,j);
            }

        //! \returns the number of generated facets
        DEVICE inline unsigned int getNumFacets() const
            {
            return m_nfacet;
            }

    private:
        const VectorIt m_points;
        const unsigned int m_n;

        unsigned int *m_I;
        unsigned int *m_J;
        unsigned int *m_K;
        unsigned int m_nfacet;
        Scalar m_lower;

        //! Iterative gift wrapping algorithm
        DEVICE void wrap(unsigned int i0, unsigned int j0)
            {
            // a priority queue using a ring buffer
            const unsigned int capacity = Nvert*Nvert+1; // worst case + 1
            unsigned int queue_i[capacity];
            unsigned int queue_j[capacity];
            unsigned int head = 0;
            unsigned int tail = 0;

            unsigned int h = 0;

            queue_i[tail] = i0;
            queue_j[tail] = j0;

            tail = (tail + 1) % capacity;

            do
                {
                unsigned int k, l, m;

                // pop from queue
                unsigned int i = queue_i[head];
                unsigned int j = queue_j[head];
                head = (head + 1) % capacity;

                bool visited = false;
                for (m = 0; m < h; m++)  // check if facet hasn't been explored
                    if ((m_I[m] == i && m_J[m] == j) || (m_J[m] == i && m_K[m] == j) || (m_K[m] == i && m_I[m] == j))
                        {
                        visited = true;
                        break;
                        }
                if (visited)
                    continue;

                for (k = i, l = 0; l < m_n; l++)  // wrap from edge ij to find facet ijk
                    if (turn(m_points[i],m_points[j],m_points[l]) < 0 &&
                        m_lower*orient(m_points[i],m_points[j],m_points[k],m_points[l]) >= 0)
                        k = l;

                if (turn(m_points[i],m_points[j],m_points[k]) >= 0)
                    continue;

                m_I[h] = i;  m_J[h] = j;  m_K[h++] = k;

                queue_i[tail]= k;
                queue_j[tail] = j; // push edge k,j
                tail = (tail + 1) % capacity;

                queue_i[tail] = i;
                queue_j[tail] = k; // push edge i,k
                tail = (tail + 1) % capacity;
                } while (head != tail);

            m_nfacet += h;
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
