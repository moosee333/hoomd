// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __EVALUATOR_CONSTRAINT_EggCarton_H__
#define __EVALUATOR_CONSTRAINT_EggCarton_H__

#include "hoomd/HOOMDMath.h"
using namespace std;

/*! \file EvaluatorConstraintEggCarton.h
    \brief Defines the constraint evaluator class for egg carton surface
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating egg carton constraints
/*! <b>General Overview</b>
    EvaluatorConstraintEggCarton is a low level computation helper class to aid in evaluating particle constraints on
    an egg carton surface. It will find the nearest point on the egg carton to a given position.
*/
class EvaluatorConstraintEggCarton
    {
    public:
        //! Constructs the constraint evaluator
        /*! \param _xFreq     Number of cosine waves in the X direction.
            \param _yFreq     Number of cosine waves in the Y direction.
            \param _xHeight   Number of cosine waves in the X direction.
            \param _yHeight   Number of cosine waves in the Y direction.
        */
        DEVICE EvaluatorConstraintEggCarton(int _xFreq, int _yFreq, Scalar _xHeight, Scalar _yHeight)
            : xFreq(_xFreq), yFreq(_yFreq), xHeight(_xHeight), yHeight(_yHeight)
            {
            }

        //! Evaluate the closest point on the egg carton surface.
        /*! \param U unconstrained point

            \return Nearest point on the egg carton
        */
        DEVICE Scalar3 evalClosest(const Scalar3& U, const Scalar3& box)
            {   
            Scalar hx, hy;
            hx = xHeight * sin(U.x * xFreq * M_PI / box.x) * xFreq * M_PI / box.x;
            hy = yHeight * sin(U.y * yFreq * M_PI / box.y) * yFreq * M_PI / box.y;
            
            Scalar nNorm, nHatX, nHatY, nHatZ;
            nNorm = slow::sqrt(hx*hx + hy*hy + 1);
            nHatX = hx / nNorm;
            nHatY = hy / nNorm;
            nHatZ = 1.0 / nNorm;
            
            Scalar phi, el, d;
            phi = acos(nHatZ);
            el = U.z - (xHeight * cos(U.x * xFreq * M_PI / box.x) + yHeight * cos(U.y * yFreq * M_PI / box.y));
            d = 0.5 * el * sin(2.0 * phi);
            
            Scalar3 C;
            C.x = U.x - nHatX * d;
            C.y = U.y - nHatY * d;
            C.z = (xHeight * cos(C.x * xFreq * M_PI / box.x) + yHeight * cos(C.y * yFreq * M_PI / box.y));

            return C;
            }

        //! Evaluate the normal unit vector for point on the egg carton surface.
        /*! \param U point on egg carton surface
            \return normal unit vector for  point on the egg carton
        */
        DEVICE Scalar3 evalNormal(const Scalar3& U, const Scalar3& box)
            {
            Scalar hx, hy, nNorm;
            hx = xHeight * sin(U.x * xFreq * M_PI / box.x) * xFreq * M_PI / box.x;
            hy = yHeight * sin(U.y * yFreq * M_PI / box.y) * yFreq * M_PI / box.y;
            nNorm = slow::sqrt(hx*hx + hy*hy + 1);
            
            Scalar3 N;
            N.x = hx / nNorm;
            N.y = hy / nNorm;
            N.z = 1.0 / nNorm;

            return N;
            }

    protected:
        int xFreq;          //!< Number of cosine waves in the X direction.
        int yFreq;          //!< Number of cosine waves in the Y direction.
        Scalar xHeight;       //!< Number of cosine waves in the X direction.
        Scalar yHeight;       //!< Number of cosine waves in the Y direction.
    };


#endif // __PAIR_EVALUATOR_LJ_H__
