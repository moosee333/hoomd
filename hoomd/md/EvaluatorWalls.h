// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jproc

/*! \file EvaluatorWalls.h
    \brief Executes an external field potential of several evaluator types for each wall in the system.
 */

#ifndef __EVALUATOR_WALLS_H__
#define __EVALUATOR_WALLS_H__

#ifndef NVCC
#include <string>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

#include "hoomd/BoxDim.h"
#include "WallData.h"


#undef DEVICE
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

// sets the max numbers for each wall geometry type
const unsigned int MAX_N_SWALLS=20;
const unsigned int MAX_N_CWALLS=20;
const unsigned int MAX_N_PWALLS=60;

struct wall_type
    {
    unsigned int     numSpheres; // these data types come first, since the structs are aligned already
    unsigned int     numCylinders;
    unsigned int     numPlanes;
    SphereWall       Spheres[MAX_N_SWALLS];
    CylinderWall     Cylinders[MAX_N_CWALLS];
    PlaneWall        Planes[MAX_N_PWALLS];
    };


//! Applys a wall force from all walls in the field parameter
/*! \ingroup computes
*/
template<class evaluator>
class EvaluatorWalls
    {
    public:
        typedef struct
            {
            typename evaluator::param_type params;
            Scalar rcutsq;
            Scalar rextrap;
            } param_type;

        typedef wall_type field_type;

        //! Constructs the external wall potential evaluator
        DEVICE EvaluatorWalls(Scalar3 pos, const BoxDim& box, const param_type& p, const field_type& f)
            : m_pos(pos),
              m_box(box),
              m_field(f),
              m_params(p)
            {
            }

        //! Test if evaluator needs Diameter
        DEVICE static bool needsDiameter()
            {
            return evaluator::needsDiameter();
            }

        //! Accept the optional diameter value
        /*! \param di Diameter of particle i
        */
        DEVICE void setDiameter(Scalar diameter)
            {
            di = diameter;
            }

        //! Charges not supported by walls evals
        DEVICE static bool needsCharge()
            {
            return evaluator::needsCharge();
            }

        //! Accept the optional charge value
        /*! \param qi Charge of particle i
        Walls charge currently assigns a charge of 0 to the walls. It is however unused by implemented potentials.
        */
        DEVICE void setCharge(Scalar charge)
            {
            qi = charge;
            }

        DEVICE static bool needsFieldRescale()
            {
            return true;
            }

        DEVICE static void rescaleField(field_type& field, const BoxDim& new_box, const BoxDim& old_box  )
            {
			//!Rescale and rotate geometries
			//Transformation Matrix
            Scalar trans_matrix[9];
            //Calculate Transformation Matrix
            getTransMatrix(old_box, new_box, trans_matrix);



            //Rescale through wall planes
            if (field.numPlanes > 0)
                {
                Scalar tranf_inv_trans[9];
                getInvMatrix(trans_matrix, tranf_inv_trans);
                for(unsigned int k = 0; k < field.numPlanes; k++)
                    {
                    rescaleWall(field.Planes[k], trans_matrix, tranf_inv_trans);
                    }
                }

            //Rescale through wall spheres
            for(unsigned int k = 0; k < field.numSpheres; k++)
                {
                rescaleWall(field.Spheres[k], trans_matrix);
                }

            //Rescale through wall cylinders
            for(unsigned int k = 0; k < field.numCylinders; k++)
                {
                rescaleWall(field.Cylinders[k], trans_matrix);
                }
            }


        #ifndef NVCC
        //! Update the python copy of the field
        /*! \updates the python object for a field that has been modified durring a run, i.e. via rescaleField.
        */
        static void updateFieldPy(field_type& field, pybind11::object walls)
            {
            for(unsigned int i = 0; i < field.numSpheres; i++)
                {
                walls.attr("spheres").cast<pybind11::list>()[i].cast<pybind11::object>().attr("_cpp") = pybind11::cast(field.Spheres[i]);
                }
            for(unsigned int i = 0; i < field.numCylinders; i++)
                {
                walls.attr("cylinders").cast<pybind11::list>()[i].cast<pybind11::object>().attr("_cpp") = pybind11::cast(field.Cylinders[i]);
                }
            for(unsigned int i = 0; i < field.numPlanes; i++)
                {
                walls.attr("planes").cast<pybind11::list>()[i].cast<pybind11::object>().attr("_cpp") = pybind11::cast(field.Planes[i]);
                }
            }
        #endif



        DEVICE inline void callEvaluator(Scalar3& F, Scalar& energy, const Scalar3 dr)
            {
            Scalar rsq = dot(dr, dr);

            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);
            evaluator eval(rsq, m_params.rcutsq, m_params.params);
            if (evaluator::needsDiameter())
                eval.setDiameter(di, Scalar(0.0));
            if (evaluator::needsCharge())
                eval.setCharge(qi, Scalar(0.0));

            bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, true);

            if (evaluated)
                {
                // correctly result in a 0 force in this case
                #ifdef NVCC
                if (!isfinite(force_divr))
                #else
                if (!std::isfinite(force_divr))
                #endif
                    {
                        force_divr = Scalar(0.0);
                        pair_eng = Scalar(0.0);
                    }
                // add the force and potential energy to the particle i
                F += -dr*force_divr;
                energy += pair_eng; // removing half since the other "particle" won't be represented * Scalar(0.5);
                }
            }

        DEVICE inline void extrapEvaluator(Scalar3& F, Scalar& energy, const Scalar3 dr, const Scalar rextrapsq, const Scalar r)
            {
            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);

            evaluator eval(rextrapsq, m_params.rcutsq, m_params.params);
            if (evaluator::needsDiameter())
                eval.setDiameter(di, Scalar(0.0));
            if (evaluator::needsCharge())
                eval.setCharge(qi, Scalar(0.0));

            bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, true);

            if (evaluated)
                {
                // add the force and potential energy to the particle i
                pair_eng = pair_eng + force_divr * m_params.rextrap * r; // removing half since the other "particle" won't be represented * Scalar(0.5);
                force_divr *= m_params.rextrap / r;
                // correctly result in a 0 force in this case
                #ifdef NVCC
                if (!isfinite(force_divr))
                #else
                if (!std::isfinite(force_divr))
                #endif
                    {
                        force_divr = Scalar(0.0);
                        pair_eng = Scalar(0.0);
                    }
                // add the force and potential energy to the particle i
                F += -dr*force_divr;
                energy += pair_eng; // removing half since the other "particle" won't be represented * Scalar(0.5);
                }
            }

        //! Generates force and energy from standard evaluators using wall geometry functions
        DEVICE void evalForceEnergyAndVirial(Scalar3& F, Scalar& energy, Scalar* virial)
            {
            F.x = Scalar(0.0);
            F.y = Scalar(0.0);
            F.z = Scalar(0.0);
            energy = Scalar(0.0);
            // initialize virial
            for (unsigned int i = 0; i < 6; i++)
                virial[i] = Scalar(0.0);

            Scalar3 dr;
            bool inside = false; //keeps compiler from complaining
            if (m_params.rextrap>0.0) //extrapolated mode
                {
                Scalar rextrapsq=m_params.rextrap * m_params.rextrap;
                Scalar rsq;
                for (unsigned int k = 0; k < m_field.numSpheres; k++)
                    {
                    dr = vecPtToWall(m_field.Spheres[k], m_pos, inside);
                    rsq = dot(dr, dr);
                    if (inside && rsq>=rextrapsq)
                        {
                        callEvaluator(F, energy, dr);
                        }
                    else
                        {
                        Scalar r = fast::sqrt(rsq);
                        if (rsq == 0.0)
                            {
                            inside = true; //just in case
                            dr = (m_pos - m_field.Spheres[k].origin) / m_field.Spheres[k].r;
                            }
                        else
                            {
                            dr *= 1/r;
                            }
                        r = (inside) ? m_params.rextrap - r : m_params.rextrap + r;
                        dr *= (inside) ? r : -r;
                        extrapEvaluator(F, energy, dr, rextrapsq, r);
                        }
                    }
                for (unsigned int k = 0; k < m_field.numCylinders; k++)
                    {
                    dr = vecPtToWall(m_field.Cylinders[k], m_pos, inside);
                    rsq = dot(dr, dr);
                    if (inside && rsq>=rextrapsq)
                        {
                        callEvaluator(F, energy, dr);
                        }
                    else
                        {
                        Scalar r = fast::sqrt(rsq);
                        if (rsq == 0.0)
                            {
                            inside = true; //just in case
                            vec3<Scalar> drv;
                            drv = rotate(m_field.Cylinders[k].quatAxisToZRot,vec3<Scalar>(m_pos - m_field.Cylinders[k].origin));
                            drv.z = 0.0;
                            drv = rotate(conj(m_field.Cylinders[k].quatAxisToZRot),drv) / m_field.Cylinders[k].r;
                            dr = vec_to_scalar3(drv);
                            }
                        else
                            {
                            dr *= 1/r;
                            }
                        r = (inside) ? m_params.rextrap - r : m_params.rextrap + r;
                        dr *= (inside) ? r : -r;
                        extrapEvaluator(F, energy, dr, rextrapsq, r);
                        }
                    }
                for (unsigned int k = 0; k < m_field.numPlanes; k++)
                    {
                    dr = vecPtToWall(m_field.Planes[k], m_pos, inside);
                    rsq = dot(dr, dr);
                    if (inside && rsq>=rextrapsq)
                        {
                        callEvaluator(F, energy, dr);
                        }
                    else
                        {
                        Scalar r = fast::sqrt(rsq);
                        if (rsq == 0.0)
                            {
                            inside = true; //just in case
                            dr = m_field.Planes[k].normal;
                            }
                        else
                            {
                            dr *= 1/r;
                            }
                        r = (inside) ? m_params.rextrap - r : m_params.rextrap + r;
                        dr *= (inside) ? r : -r;
                        extrapEvaluator(F, energy, dr, rextrapsq, r);
                        }
                    }
                }
            else //normal mode
                {
                for (unsigned int k = 0; k < m_field.numSpheres; k++)
                    {
                    dr = vecPtToWall(m_field.Spheres[k], m_pos, inside);
                    if (inside)
                        {
                        callEvaluator(F, energy, dr);
                        }
                    }
                for (unsigned int k = 0; k < m_field.numCylinders; k++)
                    {
                    dr = vecPtToWall(m_field.Cylinders[k], m_pos, inside);
                    if (inside)
                        {
                        callEvaluator(F, energy, dr);
                        }
                    }
                for (unsigned int k = 0; k < m_field.numPlanes; k++)
                    {
                    dr = vecPtToWall(m_field.Planes[k], m_pos, inside);
                    if (inside)
                        {
                        callEvaluator(F, energy, dr);
                        }
                    }
                }

            // evaluate virial
            virial[0] = F.x*m_pos.x;
            virial[1] = F.x*m_pos.y;
            virial[2] = F.x*m_pos.z;
            virial[3] = F.y*m_pos.y;
            virial[4] = F.y*m_pos.z;
            virial[5] = F.z*m_pos.z;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("wall_") + evaluator::getName();
            }
        #endif

    protected:
        Scalar3               m_pos;      //!< particle position
        const BoxDim          m_box;      //!<contain box information
        const field_type&     m_field;    //!< contains all information about the walls.
        param_type            m_params;
        Scalar                di;
        Scalar                qi;


    };

template < class evaluator >
typename EvaluatorWalls<evaluator>::param_type make_wall_params(typename evaluator::param_type p, Scalar rcutsq, Scalar rextrap)
    {
    typename EvaluatorWalls<evaluator>::param_type params;
    params.params = p;
    params.rcutsq = rcutsq;
    params.rextrap = rextrap;
    return params;
    }

#endif //__EVALUATOR__WALLS_H__
