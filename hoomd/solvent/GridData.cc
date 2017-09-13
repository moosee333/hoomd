// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: vramasub

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
    m_pdata->getBoxChangeSignal().connect<GridData, &GridData::setBoxChanged>(this);
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

        GPUArray<Scalar> tmp(n_elements, m_exec_conf);
        m_tmp.swap(tmp);
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
    ArrayHandle<Scalar> h_tmp(m_tmp, access_location::host, access_mode::read);

    // copy over data
    std::copy(h_phi.data, h_phi.data + n, snap->phi.begin());
    std::copy(h_fn.data, h_fn.data + n, snap->fn.begin());
    std::copy(h_tmp.data, h_tmp.data + n, snap->tmp.begin());


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
        case 1:
            return this->heaviside1();
            break;
        default:
            throw std::runtime_error("The heaviside function order you have requested is not available.");
        }
    }

void GridData::hessian(GPUArray<Scalar>& ddxx, GPUArray<Scalar>& ddxy, GPUArray<Scalar>& ddxz, 
        GPUArray<Scalar>& ddyy, GPUArray<Scalar>& ddyz, GPUArray<Scalar>& ddzz, std::vector<uint3> points)
	{
    Scalar3 spacing = this->getSpacing();

    // Create intermediate GPUArrays
    unsigned int n_points = points.size();
    GPUArray<Scalar> dx(n_points, m_exec_conf), dy(n_points, m_exec_conf), dz(n_points, m_exec_conf);
    GPUArray<Scalar> dxpx(n_points, m_exec_conf), dypx(n_points, m_exec_conf), dzpx(n_points, m_exec_conf);
    GPUArray<Scalar> dxpy(n_points, m_exec_conf), dypy(n_points, m_exec_conf), dzpy(n_points, m_exec_conf);
    GPUArray<Scalar> dxpz(n_points, m_exec_conf), dypz(n_points, m_exec_conf), dzpz(n_points, m_exec_conf);
    GPUArray<Scalar> dxmx(n_points, m_exec_conf), dymx(n_points, m_exec_conf), dzmx(n_points, m_exec_conf);
    GPUArray<Scalar> dxmy(n_points, m_exec_conf), dymy(n_points, m_exec_conf), dzmy(n_points, m_exec_conf);
    GPUArray<Scalar> dxmz(n_points, m_exec_conf), dymz(n_points, m_exec_conf), dzmz(n_points, m_exec_conf);

    // Construct lists of neighboring points
    std::vector<uint3> px, py, pz, mx, my, mz;
    px.reserve(points.size());
    py.reserve(points.size());
    pz.reserve(points.size());
    mx.reserve(points.size());
    my.reserve(points.size());
    mz.reserve(points.size());

    for(unsigned int i = 0; i < points.size(); i++)
        {
        px.push_back(make_uint3(this->wrapx(points[i].x + 1), points[i].y, points[i].z));
        py.push_back(make_uint3(points[i].x, this->wrapy(points[i].y + 1), points[i].z));
        pz.push_back(make_uint3(points[i].x, points[i].y, this->wrapz(points[i].z + 1)));

        mx.push_back(make_uint3(this->wrapx(points[i].x - 1), points[i].y, points[i].z));
        my.push_back(make_uint3(points[i].x, this->wrapy(points[i].y - 1), points[i].z));
        mz.push_back(make_uint3(points[i].x, points[i].y, this->wrapz(points[i].z - 1)));
        }

    // Compute gradient for each of the neighboring points
    this->grad(dx, dy, dz, points);
    this->grad(dxpx, dypx, dzpx, px);
    this->grad(dxpy, dypy, dzpy, py);
    this->grad(dxpz, dypz, dzpz, pz);
    this->grad(dxmx, dymx, dzmx, mx);
    this->grad(dxmy, dymy, dzmy, my);
    this->grad(dxmz, dymz, dzmz, mz);

    // Access intermediate GPUArrays (make sure to do this after the grad calls to avoid multiple simultaneous accesses)
    ArrayHandle<Scalar> h_dx(dx, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dy(dy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dz(dz, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar> h_dxpx(dxpx, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dypx(dypx, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dzpx(dzpx, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar> h_dxpy(dxpy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dypy(dypy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dzpy(dzpy, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar> h_dxpz(dxpz, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dypz(dypz, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dzpz(dzpz, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar> h_dxmx(dxmx, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dymx(dymx, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dzmx(dzmx, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar> h_dxmy(dxmy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dymy(dymy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dzmy(dzmy, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar> h_dxmz(dxmz, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dymz(dymz, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dzmz(dzmz, access_location::host, access_mode::readwrite);

    // Access main GPUArrays
    ArrayHandle<Scalar> h_ddxx(ddxx, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddyy(ddyy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddzz(ddzz, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar> h_ddxy(ddxy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddxz(ddxz, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddyz(ddyz, access_location::host, access_mode::readwrite);

        { // GPU arrays scoping
        ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
        Index3D indexer = this->getIndexer();
        for(unsigned int i = 0; i < points.size(); i++)
            {
            // Determine order of approximation based on whether or not points are on the boundary
            uint3 pxi = px[i];
            uint3 pyi = py[i];
            uint3 pzi = pz[i];
            uint3 mxi = mx[i];
            uint3 myi = my[i];
            uint3 mzi = mz[i];

            unsigned int x_forward_idx = indexer(pxi.x, pxi.y, pxi.z);
            unsigned int y_forward_idx = indexer(pyi.x, pyi.y, pyi.z);
            unsigned int z_forward_idx = indexer(pzi.x, pzi.y, pzi.z);
            unsigned int x_reverse_idx = indexer(mxi.x, mxi.y, mxi.z);
            unsigned int y_reverse_idx = indexer(myi.x, myi.y, myi.z);
            unsigned int z_reverse_idx = indexer(mzi.x, mzi.y, mzi.z);

            Scalar dyx = 0, dzx = 0;
            if (h_phi.data[x_forward_idx] != GridData::MISSING_VALUE && h_phi.data[x_reverse_idx] != GridData::MISSING_VALUE)
                {
                dyx = (h_dypx.data[i] - h_dymx.data[i])/spacing.y/2;
                dzx = (h_dzpx.data[i] - h_dzmx.data[i])/spacing.z/2;
                h_ddxx.data[i] = (h_dxpx.data[i] - h_dxmx.data[i])/spacing.x/2;
                }
            else if (h_phi.data[x_forward_idx] == GridData::MISSING_VALUE && h_phi.data[x_reverse_idx] != GridData::MISSING_VALUE)
                {
                dyx = (h_dy.data[i] - h_dymx.data[i])/spacing.y;
                dzx = (h_dz.data[i] - h_dzmx.data[i])/spacing.z;
                h_ddxx.data[i] = (h_dxmx.data[i] - h_dx.data[i])/spacing.x;
                }
            else if (h_phi.data[x_forward_idx] != GridData::MISSING_VALUE && h_phi.data[x_reverse_idx] == GridData::MISSING_VALUE)
                {
                dyx = (h_dypx.data[i] - h_dy.data[i])/spacing.y;
                dzx = (h_dzpx.data[i] - h_dz.data[i])/spacing.z;
                h_ddxx.data[i] = (h_dxpx.data[i] - h_dx.data[i])/spacing.x;
                }

            Scalar dxy = 0, dzy = 0;
            if (h_phi.data[y_forward_idx] != GridData::MISSING_VALUE && h_phi.data[y_reverse_idx] != GridData::MISSING_VALUE)
                {
                dxy = (h_dxpy.data[i] - h_dxmy.data[i])/spacing.x/2;
                dzy = (h_dzpy.data[i] - h_dzmy.data[i])/spacing.z/2;
                h_ddyy.data[i] = (h_dypy.data[i] - h_dymy.data[i])/spacing.y/2;
                }
            else if (h_phi.data[y_forward_idx] == GridData::MISSING_VALUE && h_phi.data[y_reverse_idx] != GridData::MISSING_VALUE)
                {
                dxy = (h_dx.data[i] - h_dxmy.data[i])/spacing.x;
                dzy = (h_dz.data[i] - h_dzmy.data[i])/spacing.z;
                h_ddyy.data[i] = (h_dymy.data[i] - h_dy.data[i])/spacing.y;
                }
            else if (h_phi.data[y_forward_idx] != GridData::MISSING_VALUE && h_phi.data[y_reverse_idx] == GridData::MISSING_VALUE)
                {
                dxy = (h_dxpy.data[i] - h_dx.data[i])/spacing.x;
                dzy = (h_dzpy.data[i] - h_dz.data[i])/spacing.z;
                h_ddyy.data[i] = (h_dypy.data[i] - h_dy.data[i])/spacing.y;
                }

            Scalar dxz = 0, dyz = 0;
            if (h_phi.data[z_forward_idx] != GridData::MISSING_VALUE && h_phi.data[z_reverse_idx] != GridData::MISSING_VALUE)
                {
                dxz = (h_dxpz.data[i] - h_dxmz.data[i])/spacing.x/2;
                dyz = (h_dypz.data[i] - h_dymz.data[i])/spacing.y/2;
                h_ddzz.data[i] = (h_dzpz.data[i] - h_dzmz.data[i])/spacing.z/2;
                }
            else if (h_phi.data[z_forward_idx] == GridData::MISSING_VALUE && h_phi.data[z_reverse_idx] != GridData::MISSING_VALUE)
                {
                dxz = (h_dx.data[i] - h_dxmz.data[i])/spacing.x;
                dyz = (h_dy.data[i] - h_dymz.data[i])/spacing.y;
                h_ddzz.data[i] = (h_dzmz.data[i] - h_dz.data[i])/spacing.z;
                }
            else if (h_phi.data[z_forward_idx] != GridData::MISSING_VALUE && h_phi.data[z_reverse_idx] == GridData::MISSING_VALUE)
                {
                dxz = (h_dxpz.data[i] - h_dx.data[i])/spacing.x;
                dyz = (h_dypz.data[i] - h_dy.data[i])/spacing.y;
                h_ddzz.data[i] = (h_dzpz.data[i] - h_dz.data[i])/spacing.z;
                }

            h_ddxy.data[i] = (dxy + dyx)/2;
            h_ddxz.data[i] = (dxz + dzx)/2;
            h_ddyz.data[i] = (dyz + dzy)/2;
            }
        }
	}

void GridData::grad(GPUArray<Scalar>& divx, GPUArray<Scalar>& divy, GPUArray<Scalar>& divz, std::vector<uint3> points)
	{
    // access the GPU arrays
    ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
    Index3D indexer = this->getIndexer();
    Scalar3 spacing = this->getSpacing();

    ArrayHandle<Scalar> h_divx(divx, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_divy(divy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_divz(divz, access_location::host, access_mode::readwrite);

    for (unsigned int cur_point = 0; cur_point < points.size(); cur_point++)
        {
        uint3 point = points[cur_point];
        unsigned int i = point.x, j = point.y, k = point.z;

        int x_forward = this->wrapx(i+1);
        int y_forward = this->wrapy(j+1);
        int z_forward = this->wrapz(k+1);
        int x_reverse = this->wrapx(i-1);
        int y_reverse = this->wrapy(j-1);
        int z_reverse = this->wrapz(k-1);

        unsigned int cur_idx = indexer(i, j, k);
        unsigned int x_forward_idx = indexer(x_forward, j, k);
        unsigned int y_forward_idx = indexer(i, y_forward, k);
        unsigned int z_forward_idx = indexer(i, j, z_forward);
        unsigned int x_reverse_idx = indexer(x_reverse, j, k);
        unsigned int y_reverse_idx = indexer(i, y_reverse, k);
        unsigned int z_reverse_idx = indexer(i, j, z_reverse);

        // On the boundaries we will have to use lower order approximations
        if (h_phi.data[x_forward_idx] != GridData::MISSING_VALUE && h_phi.data[x_reverse_idx] != GridData::MISSING_VALUE)
            h_divx.data[cur_point] = (h_phi.data[x_forward_idx] - h_phi.data[x_reverse_idx])/spacing.x/2;
        else if (h_phi.data[x_forward_idx] == GridData::MISSING_VALUE && h_phi.data[x_reverse_idx] != GridData::MISSING_VALUE)
            h_divx.data[cur_point] = (h_phi.data[x_reverse_idx] - h_phi.data[cur_idx])/spacing.x;
        else if (h_phi.data[x_forward_idx] != GridData::MISSING_VALUE && h_phi.data[x_reverse_idx] == GridData::MISSING_VALUE)
            h_divx.data[cur_point] = (h_phi.data[x_forward_idx] - h_phi.data[cur_idx])/spacing.x;
        else
            h_divx.data[cur_point] = 0;
            
        if (h_phi.data[y_forward_idx] != GridData::MISSING_VALUE && h_phi.data[y_reverse_idx] != GridData::MISSING_VALUE)
            h_divy.data[cur_point] = (h_phi.data[y_forward_idx] - h_phi.data[y_reverse_idx])/spacing.y/2;
        else if (h_phi.data[y_forward_idx] == GridData::MISSING_VALUE && h_phi.data[y_reverse_idx] != GridData::MISSING_VALUE)
            h_divy.data[cur_point] = (h_phi.data[y_reverse_idx] - h_phi.data[cur_idx])/spacing.y;
        else if (h_phi.data[y_forward_idx] != GridData::MISSING_VALUE && h_phi.data[y_reverse_idx] == GridData::MISSING_VALUE)
            h_divy.data[cur_point] = (h_phi.data[y_forward_idx] - h_phi.data[cur_idx])/spacing.y;
        else
            h_divy.data[cur_point] = 0;

        if (h_phi.data[z_forward_idx] != GridData::MISSING_VALUE && h_phi.data[z_reverse_idx] != GridData::MISSING_VALUE)
            h_divz.data[cur_point] = (h_phi.data[z_forward_idx] - h_phi.data[z_reverse_idx])/spacing.z/2;
        else if (h_phi.data[z_forward_idx] == GridData::MISSING_VALUE && h_phi.data[z_reverse_idx] != GridData::MISSING_VALUE)
            h_divz.data[cur_point] = (h_phi.data[z_reverse_idx] - h_phi.data[cur_idx])/spacing.z;
        else if (h_phi.data[z_forward_idx] != GridData::MISSING_VALUE && h_phi.data[z_reverse_idx] == GridData::MISSING_VALUE)
            h_divz.data[cur_point] = (h_phi.data[z_forward_idx] - h_phi.data[cur_idx])/spacing.z;
        else
            h_divz.data[cur_point] = 0;
        }
	}

//! Find the value of the mean curvature K on the set of points
void GridData::getMeanCurvature(GPUArray<Scalar> H, const GPUArray<Scalar>& dx, const GPUArray<Scalar>& dy, const GPUArray<Scalar>& dz, 
        const GPUArray<Scalar>& ddxx, const GPUArray<Scalar>& ddxy, const GPUArray<Scalar>& ddxz, 
        const GPUArray<Scalar>& ddyy, const GPUArray<Scalar>& ddyz, const GPUArray<Scalar>& ddzz, std::vector<uint3> points)
    {
    /*
     * The mean curvature is defined as the average of the principal curvatures.
     * It can also be calculated as (1/2) \del \cdot \vec{n}, where the normal
     * vector is computed using the derivatives of an implicit surface. In this
     * function, that formula is expanded out explicitly to give the curvature
     * as a function of the trace of the Hessian and gradients of the phi field.
     * cf. Goldman, R. (2005). "Curvature formulas for implicit curves and surfaces". 
     * Computer Aided Geometric Design. 22 (7): 632.
     */
        //NOTE: THE const-ness may not actually be enforceable since I end up accessing through ArrayHandles
    // The curvatures
    ArrayHandle<Scalar> h_H(H, access_location::host, access_mode::readwrite);

    // Access all derivative GPUArrays
    ArrayHandle<Scalar> h_dx(dx, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dy(dy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dz(dz, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar> h_ddxx(ddxx, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddxy(ddxy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddxz(ddxz, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddyy(ddyy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddyz(ddyz, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddzz(ddzz, access_location::host, access_mode::readwrite);

    for (unsigned int i = 0; i < points.size(); i++)
        {
        // Multiply the requisite terms
        h_H.data[i] += h_dx.data[i]*h_ddxx.data[i]*h_dx.data[i];
        h_H.data[i] += h_dx.data[i]*h_ddxy.data[i]*h_dy.data[i];
        h_H.data[i] += h_dx.data[i]*h_ddxz.data[i]*h_dz.data[i];
        h_H.data[i] += h_dy.data[i]*h_ddxy.data[i]*h_dx.data[i];
        h_H.data[i] += h_dy.data[i]*h_ddyy.data[i]*h_dy.data[i];
        h_H.data[i] += h_dy.data[i]*h_ddyz.data[i]*h_dz.data[i];
        h_H.data[i] += h_dz.data[i]*h_ddxz.data[i]*h_dx.data[i];
        h_H.data[i] += h_dz.data[i]*h_ddyz.data[i]*h_dy.data[i];
        h_H.data[i] += h_dz.data[i]*h_ddzz.data[i]*h_dz.data[i];

        h_H.data[i] -= (h_dx.data[i]*h_dx.data[i] + h_dy.data[i]*h_dy.data[i] + h_dz.data[i]*h_dz.data[i]) * (h_ddxx.data[i] + h_ddyy.data[i] + h_ddzz.data[i]);

        // Normalize the gradients to get the normal vector
        auto norm = sqrt(h_dx.data[i]*h_dx.data[i] + h_dy.data[i]*h_dy.data[i] + h_dz.data[i]*h_dz.data[i]);
        h_H.data[i] /= -2*norm*norm*norm;
        }
    }

void GridData::getGaussianCurvature(GPUArray<Scalar> K, const GPUArray<Scalar>& dx, const GPUArray<Scalar>& dy, const GPUArray<Scalar>& dz, 
        const GPUArray<Scalar>& ddxx, const GPUArray<Scalar>& ddxy, const GPUArray<Scalar>& ddxz, 
        const GPUArray<Scalar>& ddyy, const GPUArray<Scalar>& ddyz, const GPUArray<Scalar>& ddzz, std::vector<uint3> points)
    {
    // The curvatures
    ArrayHandle<Scalar> h_K(K, access_location::host, access_mode::readwrite);

    // Access all derivative GPUArrays
    ArrayHandle<Scalar> h_dx(dx, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dy(dy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dz(dz, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar> h_ddxx(ddxx, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddxy(ddxy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddxz(ddxz, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddyy(ddyy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddyz(ddyz, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ddzz(ddzz, access_location::host, access_mode::readwrite);

    for (unsigned int i = 0; i < points.size(); i++)
        {
        // Construct the adjoint matrix
        auto adj_xx = -h_ddyz.data[i]*h_ddyz.data[i] + h_ddyy.data[i]*h_ddzz.data[i];
        auto adj_xy = h_ddxz.data[i]*h_ddyz.data[i] - h_ddxy.data[i]*h_ddzz.data[i];
        auto adj_xz = -h_ddxz.data[i]*h_ddyy.data[i] + h_ddxy.data[i]*h_ddyz.data[i];
        auto adj_yx = h_ddyz.data[i]*h_ddxz.data[i] - h_ddxy.data[i]*h_ddzz.data[i];
        auto adj_yy = -h_ddxz.data[i]*h_ddxz.data[i] + h_ddxx.data[i]*h_ddzz.data[i];
        auto adj_yz = h_ddxz.data[i]*h_ddxy.data[i] - h_ddxx.data[i]*h_ddyz.data[i];
        auto adj_zx = -h_ddyy.data[i]*h_ddxz.data[i] + h_ddxy.data[i]*h_ddyz.data[i];
        auto adj_zy = h_ddxy.data[i]*h_ddxz.data[i] - h_ddxx.data[i]*h_ddyz.data[i];
        auto adj_zz = -h_ddxy.data[i]*h_ddxy.data[i] + h_ddxx.data[i]*h_ddyy.data[i];

        // Multiply the requisite terms
        h_K.data[i] += h_dx.data[i]*adj_xx*h_dx.data[i];
        h_K.data[i] += h_dx.data[i]*adj_xy*h_dy.data[i];
        h_K.data[i] += h_dx.data[i]*adj_xz*h_dz.data[i];
        h_K.data[i] += h_dy.data[i]*adj_yx*h_dx.data[i];
        h_K.data[i] += h_dy.data[i]*adj_yy*h_dy.data[i];
        h_K.data[i] += h_dy.data[i]*adj_yz*h_dz.data[i];
        h_K.data[i] += h_dz.data[i]*adj_zx*h_dx.data[i];
        h_K.data[i] += h_dz.data[i]*adj_zy*h_dy.data[i];
        h_K.data[i] += h_dz.data[i]*adj_zz*h_dz.data[i];

        // Normalize the gradients to get the normal vector
        auto norm = sqrt(h_dx.data[i]*h_dx.data[i] + h_dy.data[i]*h_dy.data[i] + h_dz.data[i]*h_dz.data[i]);
        h_K.data[i] /= norm*norm*norm*norm;
        }
    }

GPUArray<Scalar> GridData::delta(std::vector<uint3> points)
    {
    // Create the GPUArray to return and access it
    unsigned int n_elements = m_dim.x*m_dim.y*m_dim.z;
    GPUArray<Scalar> delta(n_elements, m_exec_conf);

    // access GPUArrays
    ArrayHandle<Scalar> h_delta(delta, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
    Scalar3 spacing = this->getSpacing();

    // Loop and check for neighbors
    for (std::vector<uint3>::iterator point = points.begin();
            point != points.end(); point++)
        {
        unsigned int i = point->x, j = point->y, k = point->z;
        unsigned int cur_cell = m_indexer(i, j, k);

        //NOTE: ASSUMING GPUARRAY IS INITIALIZED TO ZERO, SO ONLY MODIFYING THE BOUNDARY
        //SHOULD CONFIRM THAT THIS WORKS
        // We only need to worry about higher order accuracy for cells on the boundary
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
    return delta;
    }

GPUArray<Scalar> GridData::delta_old(std::vector<uint3> points)
    {
        //NOTE: I SHOULDN'T HAVE TO DO ANY OF THESE CHECKS; THE WHOLE
        //POINT OF PROVIDING THE INPUT VECTOR IS SO TO HAVE THE SET OF
        //BOUNDARY POINTS PRECOMPUTED
    // Create the GPUArray to return and access it
    unsigned int n_elements = m_dim.x*m_dim.y*m_dim.z;
    GPUArray<Scalar> delta(n_elements, m_exec_conf);

    // access GPUArrays
    ArrayHandle<Scalar> h_delta(delta, access_location::host, access_mode::readwrite);
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

GPUArray<Scalar3> GridData::vecToBoundary(std::vector<uint3> points)
    {
    // Use the gradient to find the direction to the boundary, then multiply by the phi grid's value to compute the vector
    //NOTE: TAKE THE GRAD AS ARGS?
    GPUArray<Scalar> divx(points.size(), m_exec_conf), divy(points.size(), m_exec_conf), divz(points.size(), m_exec_conf); 
    GPUArray<Scalar3> boundary_vecs(points.size(), m_exec_conf);
    ArrayHandle<Scalar3> h_boundary_vecs(boundary_vecs, access_location::host, access_mode::readwrite);
    grad(divx, divy, divz, points);

        {
        ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);

        ArrayHandle<Scalar> h_divx(divx, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_divy(divy, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_divz(divz, access_location::host, access_mode::read);
        
        for (unsigned int i = 0; i < points.size(); i++)
            {
            Scalar norm_grad = sqrt(h_divx.data[i]*h_divx.data[i] + h_divy.data[i]*h_divy.data[i] + h_divz.data[i]*h_divz.data[i]);
            h_divx.data[i] /= norm_grad;
            h_divy.data[i] /= norm_grad;
            h_divz.data[i] /= norm_grad;

            Scalar dist = h_phi.data[m_indexer(points[i].x, points[i].y, points[i].z)];

        h_boundary_vecs.data[i] = make_scalar3(-h_divx.data[i]*dist, -h_divy.data[i]*dist, -h_divz.data[i]*dist); 
            }
        }

    return boundary_vecs;
    }

GPUArray<Scalar> GridData::getNormUpwind(std::vector<uint3> points)
    {
        std::cerr << "Inside get norm upwind" << std::endl;
    //NOTE:For now I'm just going to assume the maximum order interpolation
    ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_fn(m_fn, access_location::host, access_mode::read);
    unsigned int n_points = points.size();
    Scalar3 spacing = this->getSpacing();

    GPUArray<Scalar> phi_upwind(n_points, m_exec_conf);
    ArrayHandle<Scalar> h_phi_upwind(phi_upwind, access_location::host, access_mode::read);

    for (unsigned int pt_idx = 0; pt_idx < n_points; pt_idx++)
        {
        uint3 point = points[pt_idx];
        
        //NOTE: Have to make sure that everything we use is included in the sparse band; 
        //at the moment I'm just trusting the user to specify a sufficently large band.
        //That's why I don't have to use Jens's logic to determine the maximal possible
        //order; with the narrow band he doesn't know ahead of time what exists
        //MAY HAVE TO INCLUDE THE LOWER ORDER OPTION IF MY LOGIC IS WRONG

        // Assign all points
        unsigned int i = point.x;
        unsigned int j = point.y;
        unsigned int k = point.z;

        unsigned int ip = this->wrapx(i+1);
        unsigned int jp = this->wrapx(j+1);
        unsigned int kp = this->wrapx(k+1);
        unsigned int ip2 = this->wrapx(ip+1);
        unsigned int jp2 = this->wrapx(jp+1);
        unsigned int kp2 = this->wrapx(kp+1);
        unsigned int ip3 = this->wrapx(ip2+1);
        unsigned int jp3 = this->wrapx(jp+21);
        unsigned int kp3 = this->wrapx(kp+21);

        unsigned int im = this->wrapx(i-1);
        unsigned int jm = this->wrapx(j-1);
        unsigned int km = this->wrapx(k-1);
        unsigned int im2 = this->wrapx(im-1);
        unsigned int jm2 = this->wrapx(jm-1);
        unsigned int km2 = this->wrapx(km-1);
        unsigned int im3 = this->wrapx(im2-1);
        unsigned int jm3 = this->wrapx(jm2-1);
        unsigned int km3 = this->wrapx(km2-1);

        // x-direction
        Scalar vx1 = (h_phi.data[m_indexer(im2, j, k)] - h_phi.data[m_indexer(im3, j, k)])/spacing.x;
        Scalar vx2 = (h_phi.data[m_indexer(im, j, k)] - h_phi.data[m_indexer(im2, j, k)])/spacing.x;
        Scalar vx3 = (h_phi.data[m_indexer(i, j, k)] - h_phi.data[m_indexer(im, j, k)])/spacing.x;
        Scalar vx4 = (h_phi.data[m_indexer(ip, j, k)] - h_phi.data[m_indexer(i, j, k)])/spacing.x;
        Scalar vx5 = (h_phi.data[m_indexer(ip2, j, k)] - h_phi.data[m_indexer(ip, j, k)])/spacing.x;
        Scalar vx6 = (h_phi.data[m_indexer(ip3, j, k)] - h_phi.data[m_indexer(ip2, j, k)])/spacing.x;

        Scalar phi_dxm = calculateWENO(vx1, vx2, vx3, vx4, vx5);
        Scalar phi_dxp = calculateWENO(vx6, vx5, vx4, vx3, vx2);

        // y-direction
        Scalar vy1 = (h_phi.data[m_indexer(i, jm2, k)] - h_phi.data[m_indexer(i, jm3, k)])/spacing.y;
        Scalar vy2 = (h_phi.data[m_indexer(i, jm, k)] - h_phi.data[m_indexer(i, jm2, k)])/spacing.y;
        Scalar vy3 = (h_phi.data[m_indexer(i, j, k)] - h_phi.data[m_indexer(i, jm, k)])/spacing.y;
        Scalar vy4 = (h_phi.data[m_indexer(i, jp, k)] - h_phi.data[m_indexer(i, j, k)])/spacing.y;
        Scalar vy5 = (h_phi.data[m_indexer(i, jp2, k)] - h_phi.data[m_indexer(i, jp, k)])/spacing.y;
        Scalar vy6 = (h_phi.data[m_indexer(i, jp3, k)] - h_phi.data[m_indexer(i, jp2, k)])/spacing.y;

        Scalar phi_dym = calculateWENO(vy1, vy2, vy3, vy4, vy5);
        Scalar phi_dyp = calculateWENO(vy6, vy5, vy4, vy3, vy2);

        // z-direction
        Scalar vz1 = (h_phi.data[m_indexer(i, j, km2)] - h_phi.data[m_indexer(i, j, km3)])/spacing.z;
        Scalar vz2 = (h_phi.data[m_indexer(i, j, km)] - h_phi.data[m_indexer(i, j, km2)])/spacing.z;
        Scalar vz3 = (h_phi.data[m_indexer(i, j, k)] - h_phi.data[m_indexer(i, j, km)])/spacing.z;
        Scalar vz4 = (h_phi.data[m_indexer(i, j, kp)] - h_phi.data[m_indexer(i, j, k)])/spacing.z;
        Scalar vz5 = (h_phi.data[m_indexer(i, j, kp2)] - h_phi.data[m_indexer(i, j, kp)])/spacing.z;
        Scalar vz6 = (h_phi.data[m_indexer(i, j, kp3)] - h_phi.data[m_indexer(i, j, kp2)])/spacing.z;

        Scalar phi_dzm = calculateWENO(vz1, vz2, vz3, vz4, vz5);
        Scalar phi_dzp = calculateWENO(vz6, vz5, vz4, vz3, vz2);

        Scalar fn_ijk = h_fn.data[m_indexer(point.x, point.y, point.z)];
        Scalar nx2 = 0, ny2 = 0, nz2 = 0;
        if (fn_ijk > 0)
            {
            nx2 = max(max(phi_dxm,0.0)*max(phi_dxm,0.0), min(phi_dxp,0.0)*min(phi_dxp,0.0));
            ny2 = max(max(phi_dym,0.0)*max(phi_dym,0.0), min(phi_dyp,0.0)*min(phi_dyp,0.0));
            nz2 = max(max(phi_dzm,0.0)*max(phi_dzm,0.0), min(phi_dzp,0.0)*min(phi_dzp,0.0));
            }
        else
            {
            nx2 = max(min(phi_dxm,0.0)*min(phi_dxm,0.0), max(phi_dxp,0.0)*max(phi_dxp,0.0));
            ny2 = max(min(phi_dym,0.0)*min(phi_dym,0.0), max(phi_dyp,0.0)*max(phi_dyp,0.0));
            nz2 = max(min(phi_dzm,0.0)*min(phi_dzm,0.0), max(phi_dzp,0.0)*max(phi_dzp,0.0));
            }
        h_phi_upwind.data[i] = sqrt(nx2+ny2+nz2);
        }
    return phi_upwind;
    }

Scalar GridData::calculateWENO(Scalar v1, Scalar v2, Scalar v3, Scalar v4, Scalar v5)
    {
    //NOTE: Currently this is almost entirely just copied from Jens's function;
    //I need to go through the calculation myself to check

    // Double stencil
    Scalar a = v2 - v1;
    Scalar b = v3 - v2;
    Scalar c = v4 - v3;
    Scalar d = v5 - v4;

    // Estimate smoothness
    Scalar S0 = 13*(a - b)*(a - b) + 3*(a - (3*b))*(a-3*b);
    Scalar S1 = 13*(b - c)*(b - c) + 3*(b + c)*(b + c);
    Scalar S2 = 13*(c - d)*(c - d) + 3*((3*c) - d)*((3*c) - d);

    // eps choosen so as to bias towards fifth order flux
    // large values of eps bias towards central differencing, causing oscillations
    // small values bias towards 3rd order ENO (lowering the order)
    // see Fedkiw, Merriman and Osher J Comp Phys 2000
    Scalar eps0 = 1e-6;
    Scalar eps = (eps0 * max(v1*v1, max(v2*v2, max(v3*v3, max(v4*v4, v5*v5))))) + 1e-99; // Prevent eps = 0

    Scalar alpha0 = 1.0/((eps + S0)*(eps + S0));
    Scalar alpha1 = 6.0/((eps + S1)*(eps + S1));
    Scalar alpha2 = 3.0/((eps + S2)*(eps + S2));

    Scalar alphasum = alpha0 + alpha1 + alpha2;
    Scalar w0 = alpha0/alphasum;
    Scalar w2 = alpha2/alphasum;

    Scalar phi_weno = ((1/3.0)*w0*(a - (2.0*b) + c)) + ((1/6.0)*(w2 - (1/2.0))*(b - (2.0*c) + d));

    Scalar phi_1 = (1/12.0)*(-v2 + (7*v3) + (7*v4) - v5);
    return phi_1 - phi_weno;
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

template<class Real>
pybind11::object SnapshotGridData<Real>::getTempGridNP() const
   {
    std::vector<intp> dims(3);
    dims[0] = m_dim.x;
    dims[1] = m_dim.y;
    dims[2] = m_dim.z;

    //! Return a Numpy array
    return py::object(num_util::makeNumFromData((Real*)&tmp[0], dims), false);
    }

void export_SnapshotGridData(py::module& m)
    {
    py::class_<SnapshotGridData<float>, std::shared_ptr<SnapshotGridData<float> > >(m,"SnapshotGridData_float")
    .def(py::init<unsigned int, uint3>())
    .def_property_readonly("phi", &SnapshotGridData<float>::getPhiGridNP, py::return_value_policy::take_ownership)
    .def_property_readonly("fn", &SnapshotGridData<float>::getVelocityGridNP, py::return_value_policy::take_ownership)
    .def_property_readonly("tmp", &SnapshotGridData<float>::getTempGridNP, py::return_value_policy::take_ownership);

    py::class_<SnapshotGridData<double>, std::shared_ptr<SnapshotGridData<double> > >(m,"SnapshotGridData_double")
    .def(py::init<unsigned int, uint3>())
    .def_property_readonly("phi", &SnapshotGridData<double>::getPhiGridNP, py::return_value_policy::take_ownership)
    .def_property_readonly("fn", &SnapshotGridData<double>::getVelocityGridNP, py::return_value_policy::take_ownership)
    .def_property_readonly("tmp", &SnapshotGridData<double>::getTempGridNP, py::return_value_policy::take_ownership);
    }

void export_GridData(py::module& m)
    {
    pybind11::class_<GridData, std::shared_ptr<GridData> >(m, "GridData")
        .def(py::init<std::shared_ptr<SystemDefinition>, Scalar, bool>())
        .def("takeSnapshot_double", &GridData::takeSnapshot<double>)
        .def("takeSnapshot_float", &GridData::takeSnapshot<float>)
        .def("setSigma", &GridData::setSigma)
        .def("getDimensions", &GridData::getDimsPy, py::return_value_policy::take_ownership)
        .def("setGrid", &GridData::setVelocityGrid);
    }

} // end namespace solvent
