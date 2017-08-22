// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

#include "GridData.h"
#include "hoomd/extern/num_util.h"

using namespace std;
namespace py = pybind11;

/*! \file GridData.cc
    \brief Contains code for the GridData class
*/

namespace solvent
{

//! Constructor
GridData::GridData(std::shared_ptr<SystemDefinition> sysdef, Scalar sigma)
    : m_sysdef(sysdef), 
      m_pdata(sysdef->getParticleData()), 
      m_exec_conf(m_pdata->getExecConf()), 
      m_sigma(sigma), 
      m_dim(make_uint3(0,0,0)),
      m_need_init_grid(true)
    {
    // connect to box change signal
    m_pdata->getBoxChangeSignal() .connect<GridData, &GridData::setBoxChanged>(this);

    }

//! Destructor
GridData::~GridData()
    {
    m_pdata->getBoxChangeSignal().disconnect<GridData, &GridData::setBoxChanged>(this);
    }

void GridData::computeDimensions()
    {
    const BoxDim& box = m_pdata->getBox();
    if (m_sysdef->getNDimensions() != 3)
        {
        throw std::runtime_error("hoomd.solvent only supports 3D boxes.");
        }

    Scalar3 L = box.getL();
    m_dim = make_uint3(ceil(L.x/m_sigma),ceil(L.y/m_sigma),ceil(L.x/m_sigma));

    m_indexer = Index3D(m_dim.x, m_dim.y, m_dim.z);

    m_exec_conf->msg->notice(6) << "hoomd.solvent: Initializing grid as "
        << m_dim.x << "x" << m_dim.y << "y" << m_dim.z << "z" << std::endl;
    }

void GridData::setGridValues(unsigned int flags, double value)
    {
    // access the GPU arrays
    ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_fn(m_fn, access_location::host, access_mode::read);

    Index3D indexer = this->getIndexer();
    for (unsigned int i = 0; i < m_dim.x; i++)
        for (unsigned int j = 0; j < m_dim.y; j++)
            for (unsigned int k = 0; k < m_dim.z; k++)
                {
                unsigned int idx = indexer(i, j, k);
                if (flags & ENERGIES)
                    h_fn.data[idx] = value;
                if (flags & DISTANCES)
                    h_phi.data[idx] = value;
                }
    }

void GridData::initializeGrid()
    {
    if (m_need_init_grid)
        {
        computeDimensions();

        unsigned int n_elements = m_dim.x*m_dim.y*m_dim.z;
        GPUArray<Scalar> phi(n_elements, m_exec_conf);
        m_phi.swap(phi);

        GPUArray<Scalar> fn(n_elements, m_exec_conf);
        m_fn.swap(fn);
        m_need_init_grid = false;
        }
    }

template<class Real>
std::shared_ptr<SnapshotGridData<Real> > GridData::takeSnapshot()
    {
    initializeGrid();

    unsigned int n = m_dim.x * m_dim.y * m_dim.z;
    auto snap = std::shared_ptr<class SnapshotGridData<Real> >(new SnapshotGridData<Real>(n, m_dim));

    // access the GPU arrays
    ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_fn(m_fn, access_location::host, access_mode::read);

    // copy over data
    std::copy(h_phi.data, h_phi.data + n, snap->phi.begin());
    std::copy(h_fn.data, h_fn.data + n, snap->fn.begin());


    return snap;
    }

GPUArray<Scalar> GridData::heaviside(unsigned int order)
    {
    // Call helper functions for different order heaviside functions
    switch(order)
        {
        case 0:
            return this->heaviside0();
            break;
        default:
            throw std::runtime_error("The heaviside function order you have requested is not available.");
        }
    }

inline GPUArray<Scalar> GridData::heaviside0()
    {
    // Create the GPUArray to return and access it
    unsigned int n_elements = m_dim.x*m_dim.y*m_dim.z;
    GPUArray<Scalar> heavi(n_elements, m_exec_conf);
    ArrayHandle<Scalar> h_heavi(heavi, access_location::host, access_mode::read); //NOTE: Is this the right access location?

    // access the GPU arrays
    ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_fn(m_fn, access_location::host, access_mode::read);

    Index3D indexer = this->getIndexer();
    for (unsigned int i = 0; i < m_dim.x; i++)
        for (unsigned int j = 0; j < m_dim.y; j++)
            for (unsigned int k = 0; k < m_dim.z; k++)
                {
                unsigned int idx = indexer(i, j, k);
                if(h_phi.data[idx] > 0)
                    h_heavi.data[idx] = 1;
                }
    return heavi;
    }

//NOTE: NOT YET WRITTEN, JUST A COPY OF ZEROTH ORDER FOR NOW
inline GPUArray<Scalar> GridData::heaviside2()
    {
    // Create the GPUArray to return and access it
    unsigned int n_elements = m_dim.x*m_dim.y*m_dim.z;
    GPUArray<Scalar> heavi(n_elements, m_exec_conf);
    ArrayHandle<Scalar> h_heavi(heavi, access_location::host, access_mode::read); //NOTE: Is this the right access location?

    // access the GPU arrays
    ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_fn(m_fn, access_location::host, access_mode::read);

    Index3D indexer = this->getIndexer();
    for (unsigned int i = 0; i < m_dim.x; i++)
        for (unsigned int j = 0; j < m_dim.y; j++)
            for (unsigned int k = 0; k < m_dim.z; k++)
                {
                unsigned int idx = indexer(i, j, k);
                if(h_phi.data[idx] > 0)
                    h_heavi.data[idx] = 1;
                }
    return heavi;
    }

inline int GridData::wrap(int index, unsigned int dim)
    {
    // Use periodic flags
    const BoxDim& box = this->m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();

    if (periodic.x && index < 0)
        index += dim;
    if (periodic.x && index >= (int) dim)
        index -= dim;
    
    return index;
    }

std::tuple<GPUArray<Scalar>, GPUArray<Scalar>, GPUArray<Scalar> > GridData::grad(GridData::deriv_direction dir)
	{
    // access the GPU arrays
    ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
    Index3D indexer = this->getIndexer();

    // Create GPUArrays to return and access them
    unsigned int n_elements = m_dim.x*m_dim.y*m_dim.z;
    Scalar3 spacing = this->getSpacing();

    GPUArray<Scalar> divx(n_elements, m_exec_conf);
    GPUArray<Scalar> divy(n_elements, m_exec_conf);
    GPUArray<Scalar> divz(n_elements, m_exec_conf);

    ArrayHandle<Scalar> h_divx(divx, access_location::host, access_mode::read); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_divy(divy, access_location::host, access_mode::read); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_divz(divz, access_location::host, access_mode::read); //NOTE: Is this the right access location?

    for (unsigned int i = 0; i < m_dim.x; i++)
        for (unsigned int j = 0; j < m_dim.y; j++)
            for (unsigned int k = 0; k < m_dim.z; k++)
                {
                unsigned int cur_idx = indexer(i, j, k);
                if(dir == GridData::FORWARD)
                    {
                    int x = this->wrapx(i+1);
                    int y = this->wrapy(j+1);
                    int z = this->wrapz(k+1);

                    unsigned int x_idx = indexer(x, j, k);
                    unsigned int y_idx = indexer(i, y, k);
                    unsigned int z_idx = indexer(i, j, z);
                    h_divx.data[cur_idx] = (h_phi.data[x_idx] - h_phi.data[cur_idx])/spacing.x;
                    h_divy.data[cur_idx] = (h_phi.data[y_idx] - h_phi.data[cur_idx])/spacing.y;
                    h_divz.data[cur_idx] = (h_phi.data[z_idx] - h_phi.data[cur_idx])/spacing.z;
                    }
                else if(dir == GridData::REVERSE)
                    {
                    int x = this->wrapx(i-1);
                    int y = this->wrapy(j-1);
                    int z = this->wrapz(k-1);

                    unsigned int x_idx = indexer(x, j, k);
                    unsigned int y_idx = indexer(i, y, k);
                    unsigned int z_idx = indexer(i, j, z);
                    h_divx.data[cur_idx] = (h_phi.data[cur_idx] - h_phi.data[x_idx])/spacing.x;
                    h_divy.data[cur_idx] = (h_phi.data[cur_idx] - h_phi.data[y_idx])/spacing.y;
                    h_divz.data[cur_idx] = (h_phi.data[cur_idx] - h_phi.data[z_idx])/spacing.z;
                    }
                else if(dir == GridData::CENTRAL)
                    {
                    int x_forward = this->wrapx(i+1);
                    int y_forward = this->wrapy(j+1);
                    int z_forward = this->wrapz(k+1);
                    int x_reverse = this->wrapx(i-1);
                    int y_reverse = this->wrapy(j-1);
                    int z_reverse = this->wrapz(k-1);

                    unsigned int x_forward_idx = indexer(x_forward, j, k);
                    unsigned int y_forward_idx = indexer(i, y_forward, k);
                    unsigned int z_forward_idx = indexer(i, j, z_forward);
                    unsigned int x_reverse_idx = indexer(x_reverse, j, k);
                    unsigned int y_reverse_idx = indexer(i, y_reverse, k);
                    unsigned int z_reverse_idx = indexer(i, j, z_reverse);

                    h_divx.data[cur_idx] = (h_phi.data[x_forward_idx] - h_phi.data[x_reverse_idx])/spacing.x/2;
                    h_divy.data[cur_idx] = (h_phi.data[y_forward_idx] - h_phi.data[y_reverse_idx])/spacing.y/2;
                    h_divz.data[cur_idx] = (h_phi.data[z_forward_idx] - h_phi.data[z_reverse_idx])/spacing.z/2;
                    }
                }
    return std::tuple<GPUArray<Scalar>, GPUArray<Scalar>, GPUArray<Scalar> >(divx, divy, divz);
	}

template<class Real>
pybind11::object SnapshotGridData<Real>::getPhiGridNP() const
    {
    std::vector<intp> dims(3);
    dims[0] = m_dim.x;
    dims[1] = m_dim.y;
    dims[2] = m_dim.z;

    //! Return a Numpy array
    return py::object(num_util::makeNumFromData((Real*)&phi[0], dims), false);
    }

template<class Real>
pybind11::object SnapshotGridData<Real>::getVelocityGridNP() const
    {
    std::vector<intp> dims(3);
    dims[0] = m_dim.x;
    dims[1] = m_dim.y;
    dims[2] = m_dim.z;

    //! Return a Numpy array
    return py::object(num_util::makeNumFromData((Real*)&fn[0], dims), false);
    }

void export_SnapshotGridData(py::module& m)
    {
    py::class_<SnapshotGridData<float>, std::shared_ptr<SnapshotGridData<float> > >(m,"SnapshotGridData_float")
    .def(py::init<unsigned int, uint3>())
    .def_property_readonly("phi", &SnapshotGridData<float>::getPhiGridNP, py::return_value_policy::take_ownership)
    .def_property_readonly("fn", &SnapshotGridData<float>::getVelocityGridNP, py::return_value_policy::take_ownership);

    py::class_<SnapshotGridData<double>, std::shared_ptr<SnapshotGridData<double> > >(m,"SnapshotGridData_double")
    .def(py::init<unsigned int, uint3>())
    .def_property_readonly("phi", &SnapshotGridData<double>::getPhiGridNP, py::return_value_policy::take_ownership)
    .def_property_readonly("fn", &SnapshotGridData<double>::getPhiGridNP, py::return_value_policy::take_ownership);
    }

void export_GridData(py::module& m)
    {
    pybind11::class_<GridData, std::shared_ptr<GridData> >(m, "GridData")
        .def(py::init<std::shared_ptr<SystemDefinition>, Scalar>())
        .def("takeSnapshot_double", &GridData::takeSnapshot<double>)
        .def("takeSnapshot_float", &GridData::takeSnapshot<float>)
        .def("setSigma", &GridData::setSigma);
    }

} // end namespace solvent
