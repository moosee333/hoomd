
//#include "COMPForceCompute.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace py = pybind11;

template<typename Potential>
COMPForceCompute<Potential>::COMPForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist, const std::string &log_suffix):
    ForceCompute(sysdef), m_nlist(nlist), m_shift_mode(no_shift),
    m_typpair_idx(m_pdata->getNTypes()), m_typeVertices()
{
    m_exec_conf->msg->notice(5) << "Constructing COMPForceCompute" << std::endl;

    assert(m_pdata);
    assert(m_nlist);

    GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), exec_conf);
    m_rcutsq.swap(rcutsq);
    GPUArray<Scalar> ronsq(m_typpair_idx.getNumElements(), exec_conf);
    m_ronsq.swap(ronsq);
    GPUArray<param_type> params(m_typpair_idx.getNumElements(), exec_conf);
    m_params.swap(params);

    // initialize name
    m_prof_name = std::string("comp ") + Potential::getName();
    m_log_name = std::string("comp_") + Potential::getName() +
        std::string("_energy") + log_suffix;
}

/*! setVertices: set the vertices for a numeric particle type from a python list.
  \param type Particle type index
  \param vertices Python list of 2D vertices specifying a polygon
*/
template<typename Potential>
void COMPForceCompute<Potential>::setVertices(
    unsigned int type, const py::list &vertices)
{
    if(type >= m_pdata->getNTypes())
    {
        m_exec_conf->msg->error() <<
            "comp: Trying to set params for a non existent type! " << type << std::endl;
        throw std::runtime_error("Error setting parameters in COMPForceCompute");
    }

    for(int i(type - m_typeVertices.size()); i >= 0; --i)
        m_typeVertices.push_back(std::vector<vec3<Scalar> >(0));

    // build a vector of points
    std::vector<vec3<Scalar> > points;

    for(size_t i(0); i < (size_t) len(vertices); i++)
    {
        const py::tuple pyPoint = py::cast<py::tuple>(vertices[i]);

        if(py::len(pyPoint) != 3)
            throw std::runtime_error("Non-3D vertex given for COMPForceCompute::setVertices");

        const Scalar x = py::cast<Scalar>(pyPoint[0]);
        const Scalar y = py::cast<Scalar>(pyPoint[1]);
        const Scalar z = py::cast<Scalar>(pyPoint[2]);
        const vec3<Scalar> point(x, y, z);
        points.push_back(point);
    }

    m_typeVertices[type] = points;
}

/*! \param typ1 First type index in the pair
  \param typ2 Second type index in the pair
  \param param Parameter to set
  \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
  set.
*/
template<typename Potential>
void COMPForceCompute<Potential>::setParams(unsigned int typ1, unsigned int typ2,
                                             const param_type& param)
{
    if(typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
    {
        m_exec_conf->msg->error() << "comp." << Potential::getName() <<
            ": Trying to set pair params for a non existent type! " <<
            typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in COMPForceCompute");
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
template<typename Potential>
void COMPForceCompute<Potential>::setRcut(unsigned int typ1, unsigned int typ2,
                                           Scalar rcut)
{
    if(typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
    {
        m_exec_conf->msg->error() << "comp." << Potential::getName() << ": Trying to set rcut for a non existent type! "
                                  << typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in COMPForceCompute");
    }

    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::readwrite);
    h_rcutsq.data[m_typpair_idx(typ1, typ2)] = rcut * rcut;
    h_rcutsq.data[m_typpair_idx(typ2, typ1)] = rcut * rcut;
}

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param ron XPLOR r_on radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
template<typename Potential>
void COMPForceCompute<Potential>::setRon(unsigned int typ1, unsigned int typ2, Scalar ron)
{
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
    {
        this->m_exec_conf->msg->error() << "pair." << Potential::getName() <<
            ": Trying to set ron for a non existent type! "
                                        << typ1 << "," << typ2 << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialPair");
    }

    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::readwrite);
    h_ronsq.data[m_typpair_idx(typ1, typ2)] = ron * ron;
    h_ronsq.data[m_typpair_idx(typ2, typ1)] = ron * ron;
}

/*! COMPForceCompute provides:
  - \c pair_"name"_energy
  where "name" is replaced with Potential::getName()
*/
template<typename Potential>
std::vector<std::string> COMPForceCompute<Potential>::getProvidedLogQuantities()
{
    std::vector<std::string> list;
    list.push_back(m_log_name);
    return list;
}

/*! \param quantity Name of the log value to get
  \param timestep Current timestep of the simulation
*/
template<typename Potential>
Scalar COMPForceCompute<Potential>::getLogValue(const std::string& quantity,
                                                 unsigned int timestep)
{
    if(quantity == m_log_name)
    {
        compute(timestep);
        return calcEnergySum();
    }
    else
    {
        m_exec_conf->msg->error() << "comp." << Potential::getName() <<
            ": " << quantity << " is not a valid log quantity for COMPForceCompute" <<
            std::endl << std::endl;
        throw std::runtime_error("Error getting log value");
    }
}

/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is called to ensure
    that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template<typename Potential>
void COMPForceCompute<Potential>::computeForces(unsigned int timestep)
{
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if(m_prof) m_prof->push(m_prof_name);

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

    // need to start from a zero force, energy and virial
    memset(&h_force.data[0] , 0, sizeof(Scalar4)*m_pdata->getN());
    memset(&h_torque.data[0] , 0, sizeof(Scalar4)*m_pdata->getN());
    memset(&h_virial.data[0] , 0, sizeof(Scalar)*m_virial.getNumElements());

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial];

    Shape tempVerticesI, tempVerticesJ;

    // for each particle
    for(int i = 0; i < (int)m_pdata->getN(); i++)
    {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        const vec3<Scalar> ri(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        const unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const quat<Scalar> quati(h_orientation.data[i]);

        tempVerticesI.resize(m_typeVertices[typei].size());
        for(unsigned int v(0); v < m_typeVertices[typei].size(); ++v)
            tempVerticesI[v] = rotate(quati, m_typeVertices[typei][v]);

        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access diameter and charge (if needed)
        Scalar di = Scalar(0.0);
        Scalar qi = Scalar(0.0);
        if(Potential::needsDiameter())
            di = h_diameter.data[i];
        if(Potential::needsCharge())
            qi = h_charge.data[i];

        // initialize current particle force, torque, potential energy, and virial to 0
        vec3<Scalar> forcei(0, 0, 0);
        vec3<Scalar> torquei(0, 0, 0);
        Scalar energyi(0);
        Scalar viriali[6];

        for(unsigned int viridx(0); viridx < 6; ++viridx)
            viriali[viridx] = 0;

        // loop over all of the neighbors of this particle
        const unsigned int myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for(unsigned int k = 0; k < size; k++)
        {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            const vec3<Scalar> rj(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            vec3<Scalar> rij(rj - ri);
            const quat<Scalar> quatj(h_orientation.data[j]);

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            const unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());

            // create rotated vertices for shape j
            tempVerticesJ.resize(m_typeVertices[typej].size());
            for(unsigned int v(0); v < m_typeVertices[typej].size(); ++v)
                tempVerticesJ[v] = rotate(quatj, m_typeVertices[typej][v]);

            // access diameter and charge (if needed)
            Scalar dj = Scalar(0.0);
            Scalar qj = Scalar(0.0);
            if(Potential::needsDiameter())
                dj = h_diameter.data[j];
            if(Potential::needsCharge())
                qj = h_charge.data[j];

            // apply periodic boundary conditions
            const Scalar3 image(box.minImage(vec_to_scalar3(rij)));
            rij = vec3<Scalar>(image.x, image.y, image.z);

            // get parameters for this type pair
            const unsigned int typpair_idx = m_typpair_idx(typei, typej);
            const param_type param = h_params.data[typpair_idx];
            const Scalar rcutsq = h_rcutsq.data[typpair_idx];

            // design specifies that energies are shifted if
            // shift mode is set to shift
            bool energy_shift = false;
            if(m_shift_mode == shift)
                energy_shift = true;

            // compute the force and potential energy
            vec3<Scalar> forceij(0, 0, 0);
            vec3<Scalar> torqueij(0, 0, 0);
            vec3<Scalar> torqueji(0, 0, 0);
            Scalar virialij[6];
            Scalar energyij(0);

            for(unsigned int viridx(0); viridx < 6; ++viridx)
                virialij[viridx] = 0;

            for(unsigned int vi(0); vi < tempVerticesI.size(); ++vi)
            {
                const vec3<Scalar> verti(tempVerticesI[vi]);
                for(unsigned int vj(0); vj < tempVerticesJ.size(); ++vj)
                {
                    const vec3<Scalar> vertj(tempVerticesJ[vj]);
                    const vec3<Scalar> rvivj(rij + vertj - verti);
                    const Scalar rsq(dot(rvivj, rvivj));

                    Potential eval(rsq, rcutsq, param);
                    Scalar force_divr(0);
                    Scalar pairEnergy(0);
                    if(Potential::needsDiameter())
                        eval.setDiameter(di, dj);
                    if(Potential::needsCharge())
                        eval.setCharge(qi, qj);

                    bool evaluated = eval.evalForceAndEnergy(force_divr, pairEnergy, energy_shift);

                    if(evaluated)
                    {
                        const vec3<Scalar> force(-force_divr*rvivj);
                        forceij += force;
                        energyij += pairEnergy;
                        torqueij += cross(verti, force);

                        if(third_law)
                            torqueji += cross(vertj, -force);
                    }
                }
            }

            forcei += forceij;
            energyi += energyij;
            torquei += torqueij;

            if(compute_virial)
            {
                virialij[0] -= forceij.x*rij.x*Scalar(0.5);
                virialij[1] -= forceij.x*rij.y*Scalar(0.5);
                virialij[2] -= forceij.x*rij.z*Scalar(0.5);
                virialij[3] -= forceij.y*rij.y*Scalar(0.5);
                virialij[4] -= forceij.y*rij.z*Scalar(0.5);
                virialij[5] -= forceij.z*rij.z*Scalar(0.5);

                for(unsigned int viridx(0); viridx < 6; ++viridx)
                    viriali[viridx] += virialij[viridx];
            }

            if(third_law)
            {
                h_force.data[j].x -= forceij.x;
                h_force.data[j].y -= forceij.y;
                h_force.data[j].z -= forceij.z;
                h_force.data[j].w += energyij*Scalar(0.5);

                h_torque.data[j].x += torqueji.x;
                h_torque.data[j].y += torqueji.y;
                h_torque.data[j].z += torqueji.z;

                if(compute_virial)
                {
                    for(unsigned int viridx(0); viridx < 6; ++viridx)
                        h_virial.data[viridx*m_virial_pitch + j] += virialij[viridx];
                }
            }
        }

        // finally, increment the force, potential energy and virial for particle i
        h_force.data[i].x += forcei.x;
        h_force.data[i].y += forcei.y;
        h_force.data[i].z += forcei.z;
        h_force.data[i].w += energyi*Scalar(0.5);

        h_torque.data[i].x += torquei.x;
        h_torque.data[i].y += torquei.y;
        h_torque.data[i].z += torquei.z;

        if(compute_virial)
        {
            for(unsigned int viridx(0); viridx < 6; ++viridx)
                h_virial.data[viridx*m_virial_pitch + i] += viriali[viridx];
        }
    }

    if(m_prof) m_prof->pop();
}

template<typename Potential>
void COMPForceCompute<Potential>::slotNumTypesChange()
{
    m_typpair_idx = Index2D(m_pdata->getNTypes());

    // reallocate parameter arrays
    GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), exec_conf);
    m_rcutsq.swap(rcutsq);
    GPUArray<Scalar> ronsq(m_typpair_idx.getNumElements(), exec_conf);
    m_ronsq.swap(ronsq);
    GPUArray<param_type> params(m_typpair_idx.getNumElements(), exec_conf);
    m_params.swap(params);
}

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template<typename Potential>
CommFlags COMPForceCompute<Potential>::getRequestedCommFlags(unsigned int timestep)
{
    CommFlags flags = CommFlags(0);

    // we need orientations for anisotropic ptls
    flags[comm_flag::orientation] = 1;

    if(Potential::needsCharge())
        flags[comm_flag::charge] = 1;

    if(Potential::needsDiameter())
        flags[comm_flag::diameter] = 1;

    flags |= ForceCompute::getRequestedCommFlags(timestep);

    return flags;
}
#endif

// Modified to work with pybind11
//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
  \tparam T Class type to export. \b Must be an instantiated COMPForceCompute class template.
*/
/*
template<typename T> void export_COMPForceCompute(const std::string& name)
{
    boost::python::scope in_comp_pair =
        boost::python::class_<T, std::shared_ptr<T>, boost::python::bases<ForceCompute>, boost::noncopyable >
        (name.c_str(), boost::python::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, const std::string& >())
        .def("setParams", &T::setParams)
        .def("setVertices", &T::setVertices)
        .def("setRcut", &T::setRcut)
        .def("setRon", &T::setRon)
        .def("setShiftMode", &T::setShiftMode)
        ;

    boost::python::enum_<typename T::energyShiftMode>("energyShiftMode")
        .value("no_shift", T::no_shift)
        .value("shift", T::shift)
        ;
}
*/
template<typename T> void export_COMPForceCompute(py::module& m, const std::string& name)
    {
    py::class_<T, std::shared_ptr<T> > comp_pair(m, name.c_str(), py::base<ForceCompute>());

    comp_pair
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, const std::string& >())
        .def("setParams", &T::setParams)
        .def("setVertices", &T::setVertices)
        .def("setRcut", &T::setRcut)
        .def("setRon", &T::setRon)
        .def("setShiftMode", &T::setShiftMode)
        ;

    py::enum_<typename T::energyShiftMode>(comp_pair, "energyShiftMode")
        .value("no_shift", T::no_shift)
        .value("shift", T::shift)
        ;
    }
