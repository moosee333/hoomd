// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#ifndef __ANISO_POTENTIAL_PAIR_H__
#define __ANISO_POTENTIAL_PAIR_H__

#include <iostream>
#include <stdexcept>
#include <memory>

#include "NeighborList.h"
#include "hoomd/ForceCompute.h"

/*! \file AnisoPotentialPair.h
    \brief Defines the template class for anisotropic pair potentials
    \details The heart of the code that computes anisotropic pair potentials is in this file.
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Template class for computing pair potentials
/*! <b>Overview:</b>
    AnisoPotentialPair computes standard pair potentials (and forces) between all particle pairs in the simulation. It
    employs the use of a neighbor list to limit the number of computations done to only those particles with the
    cuttoff radius of each other. The computation of the actual V(r) is not performed directly by this class, but
    by an aniso_evaluator class (e.g. EvaluatorPairLJ) which is passed in as a template parameter so the compuations
    are performed as efficiently as possible.

    AnisoPotentialPair handles most of the gory internal details common to all standard pair potentials.
     - A cuttoff radius to be specified per particle type pair
     - The energy can be globally shifted to 0 at the cuttoff
     - Per type pair parameters are stored and a set method is provided
     - Logging methods are provided for the energy
     - And all the details about looping through the particles, computing dr, computing the virial, etc. are handled

    \note XPLOR switching is not supported

    <b>Implementation details</b>

    rcutsq and the params are stored per particle type pair. It wastes a little bit of space, but benchmarks
    show that storing the symmetric type pairs and indexing with Index2D is faster than not storing redudant pairs
    and indexing with Index2DUpperTriangular. All of these values are stored in GPUArray
    for easy access on the GPU by a derived class. The type of the parameters is defined by \a param_type in the
    potential aniso_evaluator class passed in. See the appropriate documentation for the aniso_evaluator for the definition of each
    element of the parameters.

    For profiling and logging, AnisoPotentialPair needs to know the name of the potential. For now, that will be queried from
    the aniso_evaluator. Perhaps in the future we could allow users to change that so multiple pair potentials could be logged
    independantly.

    \sa export_AnisoAnisoPotentialPair()
*/

template <class aniso_evaluator>
class AnisoPotentialPair : public ForceCompute
    {
    public:
        //! Param type from aniso_evaluator
        typedef typename aniso_evaluator::param_type param_type;

        //! Construct the pair potential
        AnisoPotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<NeighborList> nlist,
                      const std::string& log_suffix="");
        //! Destructor
        virtual ~AnisoPotentialPair();

        //! Set the pair parameters for a single type pair
        virtual void setParams(unsigned int typ1, unsigned int typ2, const param_type& param);
        //! Set the rcut for a single type pair
        virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();
        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Shifting modes that can be applied to the energy
        enum energyShiftMode
            {
            no_shift = 0,
            shift,
            xplor
            };

        //! Set the mode to use for shifting the energy
        void setShiftMode(energyShiftMode mode)
            {
            m_shift_mode = mode;
            }

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(unsigned int timestep);
        #endif

        //! Returns true because we compute the torque
        virtual bool isAnisotropic()
            {
            return true;
            }

    protected:
        std::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
        energyShiftMode m_shift_mode;               //!< Store the mode with which to handle the energy shift at r_cut
        Index2D m_typpair_idx;                      //!< Helper class for indexing per type pair arrays
        GPUArray<Scalar> m_rcutsq;                  //!< Cuttoff radius squared per type pair
        GPUArray<param_type> m_params;   //!< Pair parameters per type pair
        std::string m_prof_name;                    //!< Cached profiler name
        std::string m_log_name;                     //!< Cached log name

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

        //! Method to be called when number of types changes
        void slotNumTypesChange()
            {
            // skip the reallocation if the number of types does not change
            // this keeps old potential coefficients when restoring a snapshot
            // it will result in invalid coeficients if the snapshot has a different type id -> name mapping
            if (m_pdata->getNTypes() == m_typpair_idx.getW())
                return;

            // if the number of types is different, build a new indexer and reallocate memory
            m_typpair_idx = Index2D(m_pdata->getNTypes());

            // reallocate parameter arrays
            GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), m_exec_conf);
            m_rcutsq.swap(rcutsq);
            GPUArray<param_type> params(m_typpair_idx.getNumElements(), m_exec_conf);
            m_params.swap(params);
            }
    };

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param log_suffix Name given to this instance of the force
*/
template < class aniso_evaluator >
AnisoPotentialPair< aniso_evaluator >::AnisoPotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                                                std::shared_ptr<NeighborList> nlist,
                                                const std::string& log_suffix)
    : ForceCompute(sysdef), m_nlist(nlist), m_shift_mode(no_shift), m_typpair_idx(m_pdata->getNTypes())
    {
    m_exec_conf->msg->notice(5) << "Constructing AnisoPotentialPair<" << aniso_evaluator::getName() << ">" << std::endl;
    assert(m_pdata);
    assert(m_nlist);

    GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), exec_conf);
    m_rcutsq.swap(rcutsq);
    GPUArray<param_type> params(m_typpair_idx.getNumElements(), exec_conf);
    m_params.swap(params);

    // initialize name
    m_prof_name = std::string("Aniso_Pair ") + aniso_evaluator::getName();
    m_log_name = std::string("aniso_pair_") + aniso_evaluator::getName() + std::string("_energy") + log_suffix;

    // connect to the ParticleData to receive notifications when the maximum number of particles changes
    m_pdata->getNumTypesChangeSignal().template connect<AnisoPotentialPair<aniso_evaluator>,
                                                        &AnisoPotentialPair<aniso_evaluator>::slotNumTypesChange>(this);
    }

template<class aniso_evaluator>
AnisoPotentialPair<aniso_evaluator>::~AnisoPotentialPair()
    {
    m_exec_conf->msg->notice(5) << "Destroying AnisoPotentialPair<" << aniso_evaluator::getName() << ">" << std::endl;

    // disconnect from type change signal
    m_pdata->getNumTypesChangeSignal().template disconnect<AnisoPotentialPair<aniso_evaluator>,
                                                           &AnisoPotentialPair<aniso_evaluator>::slotNumTypesChange>(this);
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param param Parameter to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
template< class aniso_evaluator >
void AnisoPotentialPair< aniso_evaluator >::setParams(unsigned int typ1, unsigned int typ2, const param_type& param)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "pair." << aniso_evaluator::getName() << ": Trying to set pair params for a non existant type! "
                  << typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in AnisoPotentialPair");
        }

    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[m_typpair_idx(typ1, typ2)] = param;
    h_params.data[m_typpair_idx(typ2, typ1)] = param;
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param rcut Cuttoff radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
template< class aniso_evaluator >
void AnisoPotentialPair< aniso_evaluator >::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "pair." << aniso_evaluator::getName() << ": Trying to set rcut for a non existant type! "
                  << typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in AnisoPotentialPair");
        }

    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::readwrite);
    h_rcutsq.data[m_typpair_idx(typ1, typ2)] = rcut * rcut;
    h_rcutsq.data[m_typpair_idx(typ2, typ1)] = rcut * rcut;
    }

/*! AnisoPotentialPair provides:
     - \c pair_"name"_energy
    where "name" is replaced with aniso_evaluator::getName()
*/
template< class aniso_evaluator >
std::vector< std::string > AnisoPotentialPair< aniso_evaluator >::getProvidedLogQuantities()
    {
    std::vector<std::string> list;
    list.push_back(m_log_name);
    return list;
    }
/*! \param quantity Name of the log value to get
    \param timestep Current timestep of the simulation
*/
template< class aniso_evaluator >
Scalar AnisoPotentialPair< aniso_evaluator >::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "pair." << aniso_evaluator::getName() << ": " << quantity << " is not a valid log quantity for AnisoPotentialPair"
                                  << std::endl << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    }

/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is called to ensure
    that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template< class aniso_evaluator >
void AnisoPotentialPair< aniso_evaluator >::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push(m_prof_name);

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host,access_mode::read);

    //force arrays
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);

    const BoxDim& box = m_pdata->getBox();
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

    {
    // need to start from a zero force, energy and virial
    memset(&h_force.data[0] , 0, sizeof(Scalar4)*m_pdata->getN());
    memset(&h_torque.data[0] , 0, sizeof(Scalar4)*m_pdata->getN());
    memset(&h_virial.data[0] , 0, sizeof(Scalar)*m_virial.getNumElements());

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial];

    // for each particle
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        Scalar4 quat_i = h_orientation.data[i];

        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access diameter and charge (if needed)
        Scalar di = Scalar(0.0);
        Scalar qi = Scalar(0.0);
        if (aniso_evaluator::needsDiameter())
            di = h_diameter.data[i];
        if (aniso_evaluator::needsCharge())
            qi = h_charge.data[i];

        // initialize current particle force, torque, potential energy, and virial to 0
        Scalar fxi = Scalar(0.0);
        Scalar fyi = Scalar(0.0);
        Scalar fzi = Scalar(0.0);
        Scalar txi = Scalar(0.0);
        Scalar tyi = Scalar(0.0);
        Scalar tzi = Scalar(0.0);
        Scalar pei = Scalar(0.0);
        Scalar virialxxi = 0.0;
        Scalar virialxyi = 0.0;
        Scalar virialxzi = 0.0;
        Scalar virialyyi = 0.0;
        Scalar virialyzi = 0.0;
        Scalar virialzzi = 0.0;

        // loop over all of the neighbors of this particle
        const unsigned int myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;
            Scalar4 quat_j = h_orientation.data[j];

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());

            // access diameter and charge (if needed)
            Scalar dj = Scalar(0.0);
            Scalar qj = Scalar(0.0);
            if (aniso_evaluator::needsDiameter())
                dj = h_diameter.data[j];
            if (aniso_evaluator::needsCharge())
                qj = h_charge.data[j];

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // get parameters for this type pair
            unsigned int typpair_idx = m_typpair_idx(typei, typej);
            param_type param = h_params.data[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];

            // design specifies that energies are shifted if
            // shift mode is set to shift
            bool energy_shift = false;
            if (m_shift_mode == shift)
                energy_shift = true;

            // compute the force and potential energy
            Scalar3 force = make_scalar3(0.0,0.0,0.0);
            Scalar3 torque_i = make_scalar3(0.0,0.0,0.0);
            Scalar3 torque_j = make_scalar3(0.0,0.0,0.0);

            Scalar pair_eng = Scalar(0.0);

            aniso_evaluator eval(dx, quat_i, quat_j, rcutsq, param);

            if (aniso_evaluator::needsDiameter())
                eval.setDiameter(di, dj);
            if (aniso_evaluator::needsCharge())
                eval.setCharge(qi, qj);

            bool evaluated = eval.evaluate(force, pair_eng, energy_shift,torque_i,torque_j);

            if (evaluated)
                {
                Scalar3 force2 = Scalar(0.5)*force;

                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fxi += force.x;
                fyi += force.y;
                fzi += force.z;
                txi += torque_i.x;
                tyi += torque_i.y;
                tzi += torque_i.z;
                pei += pair_eng * Scalar(0.5);

                if (compute_virial)
                    {
                    virialxxi += dx.x*force2.x;
                    virialxyi += dx.y*force2.x;
                    virialxzi += dx.z*force2.x;
                    virialyyi += dx.y*force2.y;
                    virialyzi += dx.z*force2.y;
                    virialzzi += dx.z*force2.z;
                    }

                // add the force to particle j if we are using the third law (MEM TRANSFER: 10 scalars / FLOPS: 8)
                if (third_law)
                    {
                    h_force.data[j].x -= force.x;
                    h_force.data[j].y -= force.y;
                    h_force.data[j].z -= force.z;
                    h_torque.data[j].x += torque_j.x;
                    h_torque.data[j].y += torque_j.y;
                    h_torque.data[j].z += torque_j.z;
                    h_force.data[j].w += pair_eng * Scalar(0.5);
                    if (compute_virial)
                        {
                        h_virial.data[0*m_virial_pitch+j] += dx.x*force2.x;
                        h_virial.data[1*m_virial_pitch+j] += dx.y*force2.x;
                        h_virial.data[2*m_virial_pitch+j] += dx.z*force2.x;
                        h_virial.data[3*m_virial_pitch+j] += dx.y*force2.y;
                        h_virial.data[4*m_virial_pitch+j] += dx.z*force2.y;
                        h_virial.data[5*m_virial_pitch+j] += dx.z*force2.z;
                        }
                    }
                }
            }

        // finally, increment the force, potential energy and virial for particle i
        h_force.data[i].x += fxi;
        h_force.data[i].y += fyi;
        h_force.data[i].z += fzi;
        h_torque.data[i].x += txi;
        h_torque.data[i].y += tyi;
        h_torque.data[i].z += tzi;
        h_force.data[i].w += pei;
        if (compute_virial)
            {
            h_virial.data[0*m_virial_pitch+i] += virialxxi;
            h_virial.data[1*m_virial_pitch+i] += virialxyi;
            h_virial.data[2*m_virial_pitch+i] += virialxzi;
            h_virial.data[3*m_virial_pitch+i] += virialyyi;
            h_virial.data[4*m_virial_pitch+i] += virialyzi;
            h_virial.data[5*m_virial_pitch+i] += virialzzi;
            }
        }
    }

    if (m_prof) m_prof->pop();
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template < class aniso_evaluator >
CommFlags AnisoPotentialPair< aniso_evaluator >::getRequestedCommFlags(unsigned int timestep)
    {
    CommFlags flags = CommFlags(0);

    // we need orientations for anisotropic ptls
    flags[comm_flag::orientation] = 1;

    if (aniso_evaluator::needsCharge())
        flags[comm_flag::charge] = 1;

    if (aniso_evaluator::needsDiameter())
        flags[comm_flag::diameter] = 1;

    // with rigid bodies, include net torque
    flags[comm_flag::net_torque] = 1;

    flags |= ForceCompute::getRequestedCommFlags(timestep);

    return flags;
    }
#endif

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated AnisoPotentialPair class template.
*/
template < class T > void export_AnisoPotentialPair(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<T, std::shared_ptr<T> > anisopotentialpair(m, name.c_str(), pybind11::base<ForceCompute>());
    anisopotentialpair.def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, const std::string& >())
        .def("setParams", &T::setParams)
        .def("setRcut", &T::setRcut)
        .def("setShiftMode", &T::setShiftMode)
    ;

    pybind11::enum_<typename T::energyShiftMode>(anisopotentialpair,"energyShiftMode")
        .value("no_shift", T::energyShiftMode::no_shift)
        .value("shift", T::energyShiftMode::shift)
        .export_values()
    ;
    }

#endif // __ANISO_POTENTIAL_PAIR_H__
