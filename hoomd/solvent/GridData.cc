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
GridData::GridData(std::shared_ptr<SystemDefinition> sysdef, Scalar sigma, bool ignore_zero)
    : m_sysdef(sysdef), 
      m_pdata(sysdef->getParticleData()), 
      m_exec_conf(m_pdata->getExecConf()), 
      m_sigma(sigma), 
      m_dim(make_uint3(0,0,0)),
      m_need_init_grid(true),
      m_ignore_zero(ignore_zero)
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

void GridData::hessian(GPUArray<Scalar>& dx_square, GPUArray<Scalar>& dy_square, GPUArray<Scalar>& dz_square, 
                    GPUArray<Scalar>& dxdy, GPUArray<Scalar>& dxdz, GPUArray<Scalar>& dydz, 
                    GridData::deriv_direction dir)
	{
    // access the GPU arrays
    ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
    Index3D indexer = this->getIndexer();

    // Create and access intermediate GPUArrays
    unsigned int n_elements = m_dim.x*m_dim.y*m_dim.z;
    GPUArray<Scalar> divx(n_elements, m_exec_conf);
    GPUArray<Scalar> divy(n_elements, m_exec_conf);
    GPUArray<Scalar> divz(n_elements, m_exec_conf);
    ArrayHandle<Scalar> h_divx(divx, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_divy(divy, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_divz(divz, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?

    // Access argument GPUArrays
    ArrayHandle<Scalar> h_dx_square(dx_square, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_dy_square(dy_square, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_dz_square(dz_square, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_dxdy(dxdy, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_dxdz(dxdz, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_dydz(dydz, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?

    Scalar3 spacing = this->getSpacing();
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
	}

// NOTE: AVOID THIS MUCH CODE DUPLICATION BETWEEN THE TWO GRADIENT FUNCTIONS
void GridData::grad(GPUArray<Scalar>& divx, GPUArray<Scalar>& divy, GPUArray<Scalar>& divz, std::vector<uint3> points, GridData::deriv_direction dir)
	{
    // access the GPU arrays
    ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
    Index3D indexer = this->getIndexer();

    // Access GPUArrays
    ArrayHandle<Scalar> h_divx(divx, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_divy(divy, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_divz(divz, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?

    Scalar3 spacing = this->getSpacing();
    for (std::vector<uint3>::iterator point = points.begin();
            point != points.end(); point++)
        {
        unsigned int i = point->x, j = point->y, k = point->z;
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
	}

void GridData::grad(GPUArray<Scalar>& divx, GPUArray<Scalar>& divy, GPUArray<Scalar>& divz, GridData::deriv_direction dir)
	{
    // access the GPU arrays
    ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
    Index3D indexer = this->getIndexer();

    // Access GPUArrays
    ArrayHandle<Scalar> h_divx(divx, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_divy(divy, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_divz(divz, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?

    Scalar3 spacing = this->getSpacing();
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
	}

GPUArray<Scalar> GridData::delta(std::vector<uint3> points)
    {
    // Create the GPUArray to return and access it
    unsigned int n_elements = m_dim.x*m_dim.y*m_dim.z;
    GPUArray<Scalar> delta(n_elements, m_exec_conf);

    // access GPUArrays
    ArrayHandle<Scalar> h_delta(delta, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?
    ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
    Scalar3 spacing = this->getSpacing();

    // Loop and check for neighbors
    for (std::vector<uint3>::iterator point = points.begin();
            point != points.end(); point++)
        {
        unsigned int i = point->x, j = point->y, k = point->z;
        unsigned int cur_cell = m_indexer(i, j, k);
        int cur_sign = sgn(h_phi.data[cur_cell]);

        // In very sparse systems where the r_cut is much smaller than
        // the box dimensions, it is possible for there to be cells that
        // appear to overlap with the vdW surface simply because no
        // potential reaches there. Since there is no definite way to
        // detect this, a notice is simply printed in the default case.
        // The ignore_zero option can be set to avoid these cells.
        if (cur_sign == 0)
            {
            if (!m_ignore_zero)
                {
                m_exec_conf->msg->notice(5) << "Your system has grid cells with an exactly zero value of the energy."
                    "This is likely because your system is very sparse relative to the specified r_cut."
                    "These cells will be added to L_0. If you wish to ignore them, please set ignore_zero to True" << std::endl;
                }
            else
                {
                continue;
                }
            }

        bool on_boundary = false;
        int3 neighbor_indices[6] =
            {
            make_int3(i+1, j, k),
            make_int3(i-1, j, k),
            make_int3(i, j+1, k),
            make_int3(i, j-1, k),
            make_int3(i, j, k+1),
            make_int3(i, j, k-1),
            }; // The neighboring cells
        
        // Check all directions for changes in the sign of the energy (h_fn)
        for(unsigned int idx = 0; idx < sizeof(neighbor_indices)/sizeof(int3); idx++)
            {
                int3 neighbor_idx = neighbor_indices[idx];
                this->wrap(neighbor_idx);
                int x = neighbor_idx.x;
                int y = neighbor_idx.y;
                int z = neighbor_idx.z;
                unsigned int neighbor_cell = m_indexer(x, y, z);

                if(cur_sign != sgn(h_phi.data[neighbor_cell]))
                    {
                    if (std::abs(cur_sign - sgn(h_phi.data[neighbor_cell])) == 1)
                        {
                        if (!m_ignore_zero)
                            {
                            m_exec_conf->msg->notice(5) << "Your system has grid cells with an exactly zero value of the energy."
                                "This is likely because your system is very sparse relative to the specified r_cut."
                                "These cells will be added to L_0. If you wish to ignore them, please set ignore_zero to True" << std::endl;
                            on_boundary = true;
                            break;
                            }
                        }
                    else
                        {
                        on_boundary = true;
                        }
                    // Once we've found some direction of change, we are finished with this cell
                    }
            } // End loop over neighbors

        // We only need to worry about higher order accuracy for cells on the boundary
        if (on_boundary)
            {
            int x_forward = this->wrapx(i+1);
            int y_forward = this->wrapy(j+1);
            int z_forward = this->wrapz(k+1);
            int x_reverse = this->wrapx(i-1);
            int y_reverse = this->wrapy(j-1);
            int z_reverse = this->wrapz(k-1);

            unsigned int x_forward_idx = m_indexer(x_forward, j, k);
            unsigned int y_forward_idx = m_indexer(i, y_forward, k);
            unsigned int z_forward_idx = m_indexer(i, j, z_forward);
            unsigned int x_reverse_idx = m_indexer(x_reverse, j, k);
            unsigned int y_reverse_idx = m_indexer(i, y_reverse, k);
            unsigned int z_reverse_idx = m_indexer(i, j, z_reverse);

            // Extracting distance values from grid for convenience only, can remove unnecessary vars later
            Scalar phi0 = h_phi.data[cur_cell];

            Scalar phixp = h_phi.data[x_forward_idx];
            Scalar phiyp = h_phi.data[y_forward_idx];
            Scalar phizp = h_phi.data[z_forward_idx];

            Scalar phixm = h_phi.data[x_reverse_idx];
            Scalar phiym = h_phi.data[y_reverse_idx];
            Scalar phizm = h_phi.data[z_reverse_idx];

            Scalar dxp = (phixp - phi0)/spacing.x;
            Scalar dyp = (phiyp - phi0)/spacing.y;
            Scalar dzp = (phizp - phi0)/spacing.z;

            Scalar dxm = (phi0 - phixm)/spacing.x;
            Scalar dym = (phi0 - phiym)/spacing.y;
            Scalar dzm = (phi0 - phizm)/spacing.z;

            Scalar dx = (dxp + dxm)/2;
            Scalar dy = (dyp + dym)/2;
            Scalar dz = (dzp + dzm)/2;

            Scalar mag_grad_sq = dx*dx*dy*dy*dz*dz;
            if (mag_grad_sq == 0)
                {
                h_delta.data[cur_cell] = 0;
                }
            else
                {
                if (dxp != 0 && phixp*phi0 <= 0)
                    h_delta.data[cur_cell] += abs(phixp*dx)/(spacing.x*spacing.x)/abs(dxp)/mag_grad_sq;
                if (dxm != 0 && phixm*phi0 < 0)
                    h_delta.data[cur_cell] += abs(phixm*dx)/(spacing.x*spacing.x)/abs(dxm)/mag_grad_sq;

                if (dyp != 0 && phiyp*phi0 <= 0)
                    h_delta.data[cur_cell] += abs(phiyp*dy)/(spacing.y*spacing.y)/abs(dyp)/mag_grad_sq;
                if (dym != 0 && phiym*phi0 < 0)
                    h_delta.data[cur_cell] += abs(phiym*dy)/(spacing.y*spacing.y)/abs(dym)/mag_grad_sq;

                if (dzp != 0 && phizp*phi0 <= 0)
                    h_delta.data[cur_cell] += abs(phizp*dz)/(spacing.z*spacing.z)/abs(dzp)/mag_grad_sq;
                if (dzm != 0 && phizm*phi0 <= 0)
                    h_delta.data[cur_cell] += abs(phizm*dz)/(spacing.z*spacing.z)/abs(dzm)/mag_grad_sq;
                }
            }
        else
            {
            h_delta.data[cur_cell] = 0;
            }
        }
    return delta;
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
        .def(py::init<std::shared_ptr<SystemDefinition>, Scalar, bool>())
        .def("takeSnapshot_double", &GridData::takeSnapshot<double>)
        .def("takeSnapshot_float", &GridData::takeSnapshot<float>)
        .def("setSigma", &GridData::setSigma);
    }

} // end namespace solvent
