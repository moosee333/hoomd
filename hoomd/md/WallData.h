// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jproc

/*! \file WallData.h
    \brief Contains declarations for all types (currently Sphere, Cylinder, and
    Plane) of WallData and associated utilities.
 */
#ifndef WALL_DATA_H
#define WALL_DATA_H

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/md/QuaternionMath.h"
#include "hoomd/BoxDim.h"

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

#ifdef SINGLE_PRECISION
#define ALIGN_SCALAR 4
#else
#define ALIGN_SCALAR 8
#endif

//! SphereWall Constructor
/*! \param r Radius of the sphere
    \param origin The x,y,z coordinates of the center of the sphere
    \param inside Determines which half space is evaluated.
*/
struct SphereWall
    {
    SphereWall(Scalar rad = 0.0, Scalar3 orig = make_scalar3(0.0,0.0,0.0), bool ins = true) : origin(orig), r(rad), inside(ins) {}
    Scalar3    origin; // need to order datatype in descending order of type size for Fermi
    Scalar     r;
    bool       inside;
    } __attribute__((aligned(ALIGN_SCALAR))); // align according to first member of Scalar3

//! CylinderWall Constructor
/*! \param r Radius of the sphere
    \param origin The x,y,z coordinates of a point on the cylinder axis
    \param axis A x,y,z vector along the cylinder axis used to define the axis
    \param quatAxisToZRot (Calculated not input) The quaternion which rotates the simulation space such that the axis of the cylinder is parallel to the z' axis
    \param inside Determines which half space is evaluated.
*/
struct CylinderWall
    {
    CylinderWall(Scalar rad = 0.0, Scalar3 orig = make_scalar3(0.0,0.0,0.0), Scalar3 zorient = make_scalar3(0.0,0.0,1.0), bool ins=true) : origin(orig), axis(zorient), r(rad), inside(ins)
        {
        Scalar3 zVec(axis);
        Scalar3 zNorm = make_scalar3(0.0,0.0,1.0);

        // method source: http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final
        // easily simplified due to zNorm being a normalized vector
        Scalar normVec=fast::sqrt(dot(zVec,zVec));
        Scalar realPart=normVec + dot(zNorm,zVec);
        Scalar3 w;
        if (realPart < Scalar(1.0e-6) * normVec)
            {
                realPart=Scalar(0.0);
                w=make_scalar3(0.0, -1.0, 0.0);
            }
        else
            {
                // cross product
                w=make_scalar3(zNorm.y * zVec.z - zNorm.z * zVec.y,
                               zNorm.z * zVec.x - zNorm.x * zVec.z,
                               zNorm.x * zVec.y - zNorm.y * zVec.x);
                realPart=Scalar(realPart);
            }
            quatAxisToZRot=quat<Scalar>(realPart,vec3<Scalar>(w));
            Scalar norm=fast::rsqrt(norm2(quatAxisToZRot));
            quatAxisToZRot=norm*quatAxisToZRot;
        }
    quat<Scalar>    quatAxisToZRot; // need to order datatype in descending order of type size for Fermi
    Scalar3         origin;
    Scalar3         axis;
    Scalar          r;
    bool            inside;
    } __attribute__((aligned(ALIGN_SCALAR))); // align according to first member of quaternion

//! PlaneWall Constructor
/*! \param origin The x,y,z coordinates of a point on the cylinder axis
    \param normal The x,y,z normal vector of the plane (normalized upon input)
    \param inside Determines which half space is evaluated.
*/
struct PlaneWall
    {
    PlaneWall(Scalar3 orig = make_scalar3(0.0,0.0,0.0), Scalar3 norm = make_scalar3(0.0,0.0,1.0), bool ins = true) : normal(norm), origin(orig), inside(ins)
        {
        Scalar3 nVec;
        nVec = normal;
        Scalar invNormLength;
        invNormLength=fast::rsqrt(dot(nVec,nVec));
        normal=nVec*invNormLength;
        }
    Scalar3         normal;
    Scalar3         origin;
    bool            inside;
    } __attribute__((aligned(ALIGN_SCALAR))); // align according to first member of Scalar3

//! Point to wall vector for a sphere wall geometry
/* Returns 0 vector when all normal directions are equal
*/
DEVICE inline Scalar3 vecPtToWall(const SphereWall& wall, const Scalar3& position, bool& inside)
    {
    Scalar3 t = position;
    t-=wall.origin;
    Scalar3 shiftedPos(t);
    Scalar rxyz = fast::sqrt(dot(shiftedPos,shiftedPos));
    if (rxyz > 0.0)
        {
        inside = (((rxyz <= wall.r) && wall.inside) || ((rxyz > wall.r) && !(wall.inside))) ? true : false;
        t *= wall.r/rxyz;
        Scalar3 dx = t - shiftedPos;
        return dx;
        }
    else
        {
        inside = (wall.inside) ? true : false;
        return make_scalar3(0.0,0.0,0.0);
        }
    };

//! Point to wall vector for a cylinder wall geometry
/* Returns 0 vector when all normal directions are equal
*/
DEVICE inline Scalar3 vecPtToWall(const CylinderWall& wall, const Scalar3& position, bool& inside)
    {
    vec3<Scalar> t = vec3<Scalar>(position - wall.origin);
    vec3<Scalar> shiftedPos = rotate(wall.quatAxisToZRot,t);
    shiftedPos.z = 0.0;
    Scalar rxy = sqrt(dot(shiftedPos,shiftedPos));
    if (rxy > 0.0)
        {
        inside = (((rxy <= wall.r) && wall.inside) || ((rxy > wall.r) && !(wall.inside))) ? true : false;
        t = (wall.r / rxy) * shiftedPos;
        vec3<Scalar> dx = t - shiftedPos;
        dx = rotate(conj(wall.quatAxisToZRot),dx);
        return vec_to_scalar3(dx);
        }
    else
        {
        inside = (wall.inside) ? true : false;
        return make_scalar3(0.0,0.0,0.0);
        }
    };

//! Point to wall vector for a plane wall geometry
DEVICE inline Scalar3 vecPtToWall(const PlaneWall& wall, const Scalar3& position, bool& inside)
    {
    Scalar d = dot(wall.normal, position) - dot(wall.normal,wall.origin);
    inside = (((d >= 0.0) && wall.inside) || ((d < 0.0) && !(wall.inside))) ? true : false;
    Scalar3 dx = -d * wall.normal;
    return dx;
    };

//! Distance of point to inside sphere wall geometry, not really distance, +- based on if it's inside or not
DEVICE inline Scalar distWall(const SphereWall& wall, const Scalar3& position)
    {
    Scalar3 shiftedPos = position - wall.origin;
    Scalar rxyz2 = shiftedPos.x*shiftedPos.x + shiftedPos.y*shiftedPos.y + shiftedPos.z*shiftedPos.z;
    Scalar d = wall.r - fast::sqrt(rxyz2);
    d = (wall.inside) ? d : -d;
    return d;
    };

//! Distance of point to inside cylinder wall geometry, not really distance, +- based on if it's inside or not
DEVICE inline Scalar distWall(const CylinderWall& wall, const Scalar3& position)
    {
    vec3<Scalar> shiftedPos = vec3<Scalar>(position - wall.origin);
    shiftedPos=rotate(wall.quatAxisToZRot,shiftedPos);
    Scalar rxy2= shiftedPos.x*shiftedPos.x + shiftedPos.y*shiftedPos.y;
    Scalar d = wall.r - fast::sqrt(rxy2);
    d = (wall.inside) ? d : -d;
    return d;
    };

//! Distance of point to inside plane wall geometry, not really distance, +- based on if it's inside or not
DEVICE inline Scalar distWall(const PlaneWall& wall, const Scalar3& position)
    {
    Scalar d = dot(wall.normal,position) - dot(wall.normal,wall.origin);
    d = (wall.inside) ? d : -d;
    return d;
    };

//! Method for rescaling the plane wall properties iteratively
//Andres:Rescale Plane Walls
inline void getTransMatrix(const BoxDim& old_box, const BoxDim& new_box, Scalar *A )
    {
    //Get the Column Vectors of the old and new box
    //Old Box
    Scalar3 a_old = old_box.getLatticeVector(0);
    Scalar3 b_old = old_box.getLatticeVector(1);
    Scalar3 c_old = old_box.getLatticeVector(2);
    //New Box
    Scalar3 a_new = new_box.getLatticeVector(0);
    Scalar3 b_new = new_box.getLatticeVector(1);
    Scalar3 c_new = new_box.getLatticeVector(2);

    //Calculate the inverse of old matrix
    //Conventional Formula to get Inverse Matrix
    Scalar inv11 = b_old.y * c_old.z - c_old.y * b_old.z;
    Scalar inv12 = c_old.x * b_old.z - b_old.x * c_old.z;
    Scalar inv13 = b_old.x * c_old.y - c_old.x * b_old.y;
    Scalar inv21 = c_old.y * a_old.z - a_old.y * c_old.z;
    Scalar inv22 = a_old.x * c_old.z - c_old.x * a_old.z;
    Scalar inv23 = c_old.x * a_old.y - a_old.x * c_old.y;
    Scalar inv31 = a_old.y * b_old.z - b_old.y * a_old.z;
    Scalar inv32 = b_old.x * a_old.z - a_old.x * b_old.z;
    Scalar inv33 = a_old.x * b_old.y - b_old.x * a_old.y;

    Scalar detinv = a_old.x * inv11 + b_old.x * inv21 + c_old.x * inv31;
    inv11 /= detinv;
    inv12 /= detinv;
    inv13 /= detinv;
    inv21 /= detinv;
    inv22 /= detinv;
    inv23 /= detinv;
    inv31 /= detinv;
    inv32 /= detinv;
    inv33 /= detinv;

    //Calculate transformation matrix elements
    // A = new_box_matrix * inverse(old_box_matrix)
    //First Row of elements
    A[0] = a_new.x*inv11 + b_new.x*inv21 + c_new.x*inv31;
    A[1] = a_new.x*inv12 + b_new.x*inv22 + c_new.x*inv32;
    A[2] = a_new.x*inv13 + b_new.x*inv23 + c_new.x*inv33;


    //Second Row of elements
    A[3] = a_new.y*inv11 + b_new.y*inv21 + c_new.y*inv31;
    A[4] = a_new.y*inv12 + b_new.y*inv22 + c_new.y*inv32;
    A[5] = a_new.y*inv13 + b_new.y*inv23 + c_new.y*inv33;


    //Third Row of elements
    A[6] = a_new.z*inv11 + b_new.z*inv21 + c_new.z*inv31;
    A[7] = a_new.z*inv12 + b_new.z*inv22 + c_new.z*inv32;
    A[8] = a_new.z*inv13 + b_new.z*inv23 + c_new.z*inv33;
    }

// find inverse of Matrix
inline void getInvMatrix(Scalar *A, Scalar *B)
    {
    B[0] = A[4] * A[8] - A[5] * A[7];
    B[1] = A[2] * A[7] - A[1] * A[8];
    B[2] = A[1] * A[5] - A[2] * A[4];
    B[3] = A[5] * A[6] - A[3] * A[8];
    B[4] = A[0] * A[8] - A[2] * A[6];
    B[5] = A[2] * A[3] - A[0] * A[5];
    B[6] = A[3] * A[7] - A[4] * A[6];
    B[7] = A[1] * A[6] - A[0] * A[7];
    B[8] = A[0] * A[4] - A[1] * A[3];

    Scalar detinv = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    B[0] /= detinv;
    B[1] /= detinv;
    B[2] /= detinv;
    B[3] /= detinv;
    B[4] /= detinv;
    B[5] /= detinv;
    B[6] /= detinv;
    B[7] /= detinv;
    B[8] /= detinv;
    }

//!Rescale a Plane Wall using transformation matrix and transpose inverse of the transformation matrix
inline void rescaleWall(PlaneWall& wall, const Scalar *transMatrix, const Scalar *invTransMatrix)
    {
	//rescale origin
    Scalar3 new_origin;
    new_origin.x = wall.origin.x * transMatrix[0] + wall.origin.y * transMatrix[1] +wall.origin.z * transMatrix[2];
    new_origin.y = wall.origin.x * transMatrix[3] + wall.origin.y * transMatrix[4] +wall.origin.z * transMatrix[5];
    new_origin.z = wall.origin.x * transMatrix[6] + wall.origin.y * transMatrix[7] +wall.origin.z * transMatrix[8];
    wall.origin = new_origin;

    //rescale normal using the inverse transpose of the transMatrix
    Scalar3 new_normal;
    new_normal.x = wall.normal.x * invTransMatrix[0] + wall.normal.y * invTransMatrix[3] +wall.normal.z * invTransMatrix[6];
    new_normal.y = wall.normal.x * invTransMatrix[1] + wall.normal.y * invTransMatrix[4] +wall.normal.z * invTransMatrix[7];
    new_normal.z = wall.normal.x * invTransMatrix[2] + wall.normal.y * invTransMatrix[5] +wall.normal.z * invTransMatrix[8];
    Scalar invNormLength = fast::rsqrt(new_normal.x*new_normal.x + new_normal.y*new_normal.y + new_normal.z*new_normal.z);
    wall.normal = new_normal*invNormLength;
    };

//!Rescale a Spherical Wall using transformation matrix
inline void rescaleWall(SphereWall& wall, const Scalar *transMatrix)
    {
    //TODO: NPT_walls, consider if any of this would work with a 2D system
    Scalar k = (transMatrix[0] + transMatrix[4] + transMatrix[8])/3.0;

	//rescale origin
    wall.origin.x *= k;
    wall.origin.y *= k;
    wall.origin.z *= k;

    //rescale r
    wall.r *= k;
    };

//!Rescale a Cylindrical Wall using transformation matrix
inline void rescaleWall(CylinderWall& wall, const Scalar *transMatrix)
    {
    //rescale origin
    Scalar3 new_origin;
    new_origin.x = wall.origin.x * transMatrix[0] + wall.origin.y * transMatrix[1] +wall.origin.z * transMatrix[2];
    new_origin.y = wall.origin.x * transMatrix[3] + wall.origin.y * transMatrix[4] +wall.origin.z * transMatrix[5];
    new_origin.z = wall.origin.x * transMatrix[6] + wall.origin.y * transMatrix[7] +wall.origin.z * transMatrix[8];
    wall.origin = new_origin;

    // TODO: NPT_walls, decide if we can skew or not... seems like not, if confirmed can remove adative 0s in origin calc

    // //rescale axis
    // Scalar3 new_axis;
    // new_axis.x = wall.axis.x * transMatrix[0] + wall.axis.y * transMatrix[1] +wall.axis.z * transMatrix[2];
    // new_axis.y = wall.axis.x * transMatrix[3] + wall.axis.y * transMatrix[4] +wall.axis.z * transMatrix[5];
    // new_axis.z = wall.axis.x * transMatrix[6] + wall.axis.y * transMatrix[7] +wall.axis.z * transMatrix[8];
    // wall.axis = new_axis;
    //
    // //recalculate the rotation quaternion
    // Scalar3 zNorm = make_scalar3(0.0,0.0,1.0);
    // Scalar normVec=fast::sqrt(dot(wall.axis,wall.axis));
    // Scalar realPart=normVec + dot(zNorm,wall.axis);
    // Scalar3 w;
    // if (realPart < Scalar(1.0e-6) * normVec)
    //     {
    //         realPart=Scalar(0.0);
    //         w=make_scalar3(0.0, -1.0, 0.0);
    //     }
    // else
    //     {
    //         w=make_scalar3(zNorm.y * wall.axis.z - zNorm.z * wall.axis.y,
    //                        zNorm.z * wall.axis.x - zNorm.x * wall.axis.z,
    //                        zNorm.x * wall.axis.y - zNorm.y * wall.axis.x);
    //         realPart=Scalar(realPart);
    //     }
    //     wall.quatAxisToZRot=quat<Scalar>(realPart,vec3<Scalar>(w));
    //     Scalar norm=fast::rsqrt(norm2(wall.quatAxisToZRot));
    //     wall.quatAxisToZRot=norm*wall.quatAxisToZRot;

    //figure out which directions are coupled and scale r
    Scalar k;
    if (transMatrix[0]==transMatrix[4])
        {k=transMatrix[0];}
    else if (transMatrix[0]==transMatrix[8])
        {k=transMatrix[0];}
    else if (transMatrix[4]==transMatrix[8])
        {k=transMatrix[4];}
    else
        {//you messed up? make it crash
        // TODO: NPT_walls, think this through fully, remove or change for numerical precision problems
        k=0;}
    wall.r *= k;

    };

#endif
