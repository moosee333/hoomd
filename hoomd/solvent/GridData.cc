// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

#include "GridData.h"

using namespace std;
namespace py = pybind11;

/*! \file GridData.cc
    \brief Contains code for the GridData class
*/

namespace solvent
{

//! Constructor
GridData(std::shared_ptr<SystemDefinition> sysdef, Scalar sigma)
    : m_sysdef(sysdef), m_exec_conf(sysdef->getExecConf()), m_pdata(sysdef->getParticleData)
      m_sigma(sigma), m_dim(0,0,0), m_need_init_grid(true)
    {
    // connect to box change signal
    m_pdata->getBoxChangeSignal() .connect<GridData, &GridData::setBoxChanged>(this);

    }

//! Destructor
GridData::~GridData()
    {
    m_pdata->getBoxChangeSignal().disconnect<GridData, &GridData::setBoxchanged>(this);
    }

GridData::computeDimensions()
    {
    const BoxDim& box = m_pdata->getBox();
    if (m_sysdef->getNDimensions() != 3)
        {
        throw std::runtime_error("hoomd.solvent only supports 3D boxes."):
        }

    Scalar3 L = box.getL();
    m_dim = make_int3(ceil(L.x/m_sigma),ceil(L.y/m_sigma),ceil(L.x/m_sigma));

    m_exec_conf->msg->notice(6) << "hoomd.solvent: Initializing grid as "
        << m_dim.x << "x" << m_dim.y << "x" << m_dim.z << std::endl;
    }

GridData::initializeGrid()
    {
    if (m_need_init_grid)
        {
        computeDimensions();

        unsigned int n_elements = m_dim.x*m_dim.y*m_dim.z;
        GPUArray<Scalar> phi(m_exec_conf, n_elements);
        m_phi.swap(phi);

        GPUArray<Scalar> fn(m_exec_conf, n_elements);
        m_fn.swap(fn);
        m_need_init_grid = false;
        }
    }

template<class Real>
std::shared_ptr<SnapshotGridData<Real> > takeSnapshot()
    {
    initializeGrid();

    unsigned int n = m_dim.x * m_dim.y * m_dim.z;
    auto snap = std::shared_ptr<typename SnapshotGridData<Real> >(new typename SnapshotGridData<Real>(n));

    // access the GPU arrays
    ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_fn(m_fn, access_location::host, access_mode::read);

    // copy over data
    std::copy(h_phi.data, h_phi.data + n, snap.phi.begin());
    std::copy(h_fn.data, h_fn.data + n, snap.fn.begin());

    return snap;
    }

template<class Real>
SnapshotGridData<Real>::getPhiGridNP()
    {
    std::vector<intp> dims(3);
    dims[0] = m_dim.x;
    dims[1] = m_dim.y;
    dims[3] = m_dim.z;

    //! Return a Numpy array
    return py::object(num_util::makeNumFromData((Real*)&phi[0], dims), false);
    }

template<class Real>
SnapshotGridData<Real>::getVelocityGridNP()
    {
    std::vector<intp> dims(3);
    dims[0] = m_dim.x;
    dims[1] = m_dim.y;
    dims[3] = m_dim.z;

    //! Return a Numpy array
    return py::object(num_util::makeNumFromData((Real*)&fn[0], dims), false);
    }

void export_SnapshotGridData(py::module& m)
    {
    py::class_<SnapshotGridData<float>, std::shared_ptr<SnapshotGridData<float> > >(m,"SnapshotGridData_float")
    .def(py::init<unsigned int>())
    .def_property_readonly("phi", &SnapshotGridData<float>::getPhiNP, py::return_value_policy::take_ownership)
    .def_property_readonly("fn", &SnapshotGridData<float>::getPhiNP, py::return_value_policy::take_ownership)

    py::class_<SnapshotGridData<double>, std::shared_ptr<SnapshotGridData<double> > >(m,"SnapshotGridData_double")
    .def(py::init<unsigned int>())
    .def_property_readonly("phi", &SnapshotGridData<double>::getPhiNP, py::return_value_policy::take_ownership)
    .def_property_readonly("fn", &SnapshotGridData<double>::getPhiNP, py::return_value_policy::take_ownership)
    }

void export_GridData(py::module& m)
    {
    py::class_<GridData, std::shared_ptr<GridData>(m,"GridDAta")
        .def(py::init<std::shared_ptr<SystemDefinition>, Scalar>())
        .def("takeSnapshot",&GridData::takeSnapshot);
    }

} // end namespace solvent
