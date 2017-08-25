// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file GridData.h
    \brief Defines the GridData class and associated utilities
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <memory>

#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/SystemDefinition.h"
#include "hoomd/GPUArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __GRID_DATA_H__
#define __GRID_DATA_H__

namespace solvent
{

template<class Real>
struct SnapshotGridData;

/*! This class implements a storage for scalars on a 3D grid. It is used in the computations
    of the variational implicit solvent model.

    In the solvent model, an instantaenous location of the interface is maintained through
    an implicitly defined surface as the zero level set of a function phi: IR^3 -> IR.
    It is iteratively updated using another grid of scalar values, f_n, that define normal
    velocities for the propagation of the interface on every grid point. The function defining
    the implicit surface, phi, and the precomputed velocities, f_n, are variables in the level
    set equation.  Within every HOOMD time step, multiple iterations of the level set methods are
    performed until convergence.

    The class design is as follows: this class, GridData, holds the data structures necessary
    for the grid, phi, and f_n, to update it. Other classes may access those data through
    its public interface.

    In particular, phi will be initialized as a signed distance function to the interface,
    that is, every cell of phi holds the closest distance to its zero set with the sign
    indicating if the point is inside (-) or outside (+) the boundary. The choice of sign
    here is purely by convention. Initializaion is the task of a field updater class (such as,
    SparseFieldUpdater). The important optimization here is that updates are only performed
    on a thin layer around the zero or 'active set'.

    The GridData class is used as a base data structure for iteratively finding the solutions to
    the level set equation, the LevelSetSolver class.

    The internal data is stored as GPUArrays to maintain a consistent public interface that supports
    both CPU and GPU data storage.

    For most purposes, we want to maintain an approximately constant grid spacing along every
    direction. Hence we determine the number of cells along each dimension by rounding
    to the nearest integer muliple of the box dimensions divided by the grid spacing.
    The number of grid cells is automatically updated when the box changes.

    For interfacing with python, the class also a thin wrapper around a copy of the
    arrays, a SnapshotGridData, that can be acccess via numpy.
    */
class GridData
    {
    public:
        //! Constructor
        /* \param sysdef The HOOMD system definition
         * \param sigma the maximum requested grid spacing along every dimension
         */
        GridData(std::shared_ptr<SystemDefinition> sysdef, Scalar sigma, bool ignore_zero);

        //! Destructor
        virtual ~GridData();

        //! Whether or not to ignore zero cells when determining the boundary
        bool ignoreZero()
            {
            return m_ignore_zero;
            }

        //! Returns the distance function grid
        const GPUArray<Scalar>& getPhiGrid()
            {
            initializeGrid(); // initialize grid if necessary
            return m_phi;
            }

        //! Returns the velocity grid
        const GPUArray<Scalar>& getVelocityGrid()
            {
            initializeGrid(); // initialize grid if necessary
            return m_fn;
            }

        //! Returns a snapshot of the grid arrays
        template<class Real>
        std::shared_ptr<SnapshotGridData<Real> > takeSnapshot();

        //! Simple enumeration of flags to employ to identify the grids
        typedef enum deriv_direction
            {
            FORWARD = 1,
            REVERSE,
            CENTRAL
            } deriv_direction;

        //! Returns the gradient of the phi grid
        /*! \param dx GPUArray to insert x derivatives into of same dimension as underlying grid
            \param dy GPUArray to insert y derivatives into of same dimension as underlying grid
            \param dz GPUArray to insert z derivatives into of same dimension as underlying grid
            \param points A vector of points for which to compute the derivative; the remainder of the input grids will be untouched
            \param dir The type of finite-differencing to use to compute the derivative
        */
        void grad(GPUArray<Scalar>& divx, GPUArray<Scalar>& divy, GPUArray<Scalar>& divz, std::vector<uint3> points, GridData::deriv_direction dir = FORWARD);

        //! Returns the gradient of the phi grid
        /*! \param dx GPUArray to insert x derivatives into of same dimension as underlying grid
            \param dy GPUArray to insert y derivatives into of same dimension as underlying grid
            \param dz GPUArray to insert z derivatives into of same dimension as underlying grid
            \param dir The type of finite-differencing to use to compute the derivative
        */
        //NOTE: I SHOULD REMOVE THIS
        void grad(GPUArray<Scalar>& divx, GPUArray<Scalar>& divy, GPUArray<Scalar>& divz, GridData::deriv_direction dir = FORWARD);

        //! Returns the hessian of the phi grid
        /*! \param dx_square GPUArray to insert x derivatives into
            \param dy_square GPUArray to insert y derivatives into
            \param dz_square  GPUArray to insert z derivatives into
            \param dxdy GPUArray to insert x derivatives into
            \param dydz GPUArray to insert y derivatives into
            \param dzdz GPUArray to insert z derivatives into
        */
        void hessian(GPUArray<Scalar>& dx_square, GPUArray<Scalar>& dy_square, GPUArray<Scalar>& dz_square, 
                            GPUArray<Scalar>& dxdy, GPUArray<Scalar>& dxdz, GPUArray<Scalar>& dydz, 
                            GridData::deriv_direction dir);

        //! Returns a heaviside function of the desired order on the phi grid
        GPUArray<Scalar> heaviside(unsigned int order = 0);

        //! Second order accurate delta function
        GPUArray<Scalar> delta(std::vector<uint3> points);

        //! Function to set grid values, primarily useful for initialization
        /*! \param flags Bits indicating which grid to update (1 for energies, 2 for distances, 3 for both)
            \param value The value to set the grid to
         */
        void setGridValues(unsigned int flags = 0, double value = 0.0);

        //! Return the grid spacing along every dimension
        Scalar3 getSpacing()
            {
            initializeGrid(); // initialize grid if necessary

            Scalar3 L = m_pdata->getBox().getL();
            return make_scalar3(L.x/m_dim.x, L.y/m_dim.y, L.z/m_dim.z);
            }

        //! Return the dimensions of the grid
        uint3 getDimensions()
            {
            initializeGrid();

            return m_dim;
            }

        //! Set the maximum grid spacing
        void setSigma(Scalar sigma)
            {
            m_sigma = sigma;
            m_need_init_grid = true;
            }

        //! Get the current sigma
        Scalar getSigma() const
            {
            return m_sigma;
            }

        Index3D getIndexer() const
            {
            return m_indexer;
            }

        //! Wrap a vector around a grid
        /*! \param coords Vector to wrap, updated in place
        */
        inline uint3 wrap(int3 coords)
            {
            // Use periodic flags
            const BoxDim& box = this->m_pdata->getBox();
            uchar3 periodic = box.getPeriodic();

            if (periodic.x && coords.x < 0)
                coords.x += m_dim.x;
            if (periodic.y && coords.y < 0)
                coords.y += m_dim.y;
            if (periodic.z && coords.z < 0)
                coords.z += m_dim.z;
            if (periodic.x && coords.x >= (int) m_dim.x)
                coords.x -= m_dim.x;
            if (periodic.y && coords.y >= (int) m_dim.y)
                coords.y -= m_dim.y;
            if (periodic.z && coords.z >= (int) m_dim.z)
                coords.z -= m_dim.z;

            return make_uint3(coords.x, coords.y, coords.z);
           }

        //! Wrap grid indices in the x direction
        /*! \param index Index to wrap in the x direction
        */
        inline int wrapx(int index)
            {
            return this->wrap(index, m_dim.x);
            }

        //! Wrap grid indices in the y direction
        /*! \param index Index to wrap in the y direction
        */
        inline int wrapy(int index)
            {
            return this->wrap(index, m_dim.y);
            }

        //! Wrap grid indices in the z direction
        /*! \param index Index to wrap in the z direction
        */
        inline int wrapz(int index)
            {
            return this->wrap(index, m_dim.z);
            }

        //! \name Enumerations
        //@{

        //! Simple enumeration of flags to employ to identify the grids
        enum flags
            {
            ENERGIES = 1,
            DISTANCES = 2
            };

    protected:
        //! Helper function to re-initialize the grid when necessary
        void initializeGrid();

        //! Signal slot for box changes
        void setBoxChanged()
            {
            m_need_init_grid = true;
            }

        //! Helper function to compute grid dimensions
        void computeDimensions();

        std::shared_ptr<SystemDefinition> m_sysdef; //!< HOOMD system definition
        std::shared_ptr<ParticleData> m_pdata;               //!< HOOMD particle data (required for box)
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration

        Scalar m_sigma;     //!< The maximum grid spacing along each axis
        uint3 m_dim;         //!< The current grid dimensions

        bool m_need_init_grid;  //!< True if we need to re-initialize the grid
        bool m_ignore_zero; //!< Whether to cells with energy values of exactly 0 are considered boundary cells

        GPUArray<Scalar> m_phi; //!< The phi grid, of dimensions m_dim
        GPUArray<Scalar> m_fn;  //!< The velocity grid

        Index3D m_indexer;      //!< The grid indexer

    private:
        //! Core function to wrap grid indices
        /*! \param index Index to wrap
            \param dim Dimension to use for wrapping
        */
        inline int wrap(int index, unsigned int dim)
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

        // The various helper functions to compute mathematical functions at different orders

        //! Zeroth order heaviside function on the phi grid
        inline GPUArray<Scalar> heaviside0()
            {
            // Create the GPUArray to return and access it
            unsigned int n_elements = m_dim.x*m_dim.y*m_dim.z;
            GPUArray<Scalar> heavi(n_elements, m_exec_conf);
            ArrayHandle<Scalar> h_heavi(heavi, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?

            // access the GPU arrays
            ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_fn(m_fn, access_location::host, access_mode::read);

            for (unsigned int i = 0; i < m_dim.x; i++)
                for (unsigned int j = 0; j < m_dim.y; j++)
                    for (unsigned int k = 0; k < m_dim.z; k++)
                        {
                        unsigned int idx = m_indexer(i, j, k);
                        if(h_phi.data[idx] > 0)
                            h_heavi.data[idx] = 1;
                        }
            return heavi;
            }

        //! Second order heaviside function on the phi grid
        inline GPUArray<Scalar> heaviside2()
            {
            // Create the GPUArray to return and access it
            unsigned int n_elements = m_dim.x*m_dim.y*m_dim.z;
            GPUArray<Scalar> heavi(n_elements, m_exec_conf);

            // access GPUArrays
            ArrayHandle<Scalar> h_heavi(heavi, access_location::host, access_mode::readwrite); //NOTE: Is this the right access location?
            ArrayHandle<Scalar> h_phi(m_phi, access_location::host, access_mode::read);
            Scalar3 spacing = this->getSpacing();

            // Loop and check for neighbors
            for (unsigned int i = 0; i < m_dim.x; i++)
                for (unsigned int j = 0; j < m_dim.y; j++)
                    for (unsigned int k = 0; k < m_dim.z; k++)
                        {
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
                        if (on_boundary)
                            {
                            // Compute at second order approximation on the boundary
                            std::vector<uint3> point;
                            point.push_back(make_uint3(i, j, k));
                            GPUArray<Scalar> divx(point.size(), m_exec_conf), divy(point.size(), m_exec_conf), divz(point.size(), m_exec_conf);
                            grad(divx, divy, divz, point, this->CENTRAL);
                            
                            Scalar dx, dy, dz;
                            {
                            ArrayHandle<Scalar> h_divx(divx, access_location::host, access_mode::read);
                            ArrayHandle<Scalar> h_divy(divy, access_location::host, access_mode::read);
                            ArrayHandle<Scalar> h_divz(divz, access_location::host, access_mode::read);
                            dx = h_divx.data[0], dy = h_divy.data[0], dz = h_divz.data[0];
                            }
                            Scalar mag_grad_sq = dx*dx*dy*dy*dz*dz;
                            mag_grad_sq = mag_grad_sq != 0 ? mag_grad_sq : 1;

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

                            Scalar Ipx = h_phi.data[x_forward_idx] >= 0 ? h_phi.data[x_forward_idx] : 0;
                            Scalar Imx = h_phi.data[x_reverse_idx] >= 0 ? h_phi.data[x_reverse_idx] : 0;
                            Scalar Ipy = h_phi.data[y_forward_idx] >= 0 ? h_phi.data[y_forward_idx] : 0;
                            Scalar Imy = h_phi.data[y_reverse_idx] >= 0 ? h_phi.data[y_reverse_idx] : 0;
                            Scalar Ipz = h_phi.data[z_forward_idx] >= 0 ? h_phi.data[z_forward_idx] : 0;
                            Scalar Imz = h_phi.data[z_reverse_idx] >= 0 ? h_phi.data[z_reverse_idx] : 0;

                            Scalar del_I_x = (Ipx - Imx)/spacing.x/2;
                            Scalar del_I_y = (Ipy - Imy)/spacing.y/2;
                            Scalar del_I_z = (Ipz - Imz)/spacing.z/2;

                            h_heavi.data[cur_cell] = (del_I_x*dx + del_I_y*dy + del_I_z*dz)/mag_grad_sq;
                            }
                        else
                            {
                            // Away from the boundary, the 0th order aproximation is enough
                            h_heavi.data[cur_cell] = (cur_sign > 0);
                            }
                        } // End loop over cells
            return heavi;
            }

		template <class Real>
		inline int sgn(Real num)
            {
			return (num > Real(0)) - (num < Real(0));
            }
    };

template<class Real>
struct SnapshotGridData
    {
    SnapshotGridData(unsigned int n_elements, uint3 dim)
        : m_dim(make_uint3(dim.x, dim.y, dim.z))
        {
        phi.resize(n_elements,Scalar(0.0));
        fn.resize(n_elements,Scalar(0.0));
        }

    //! Returns the distance function grid as a Numpy array
    pybind11::object getPhiGridNP() const;

    //! Returns the velocity grid
    pybind11::object getVelocityGridNP() const;
    
    std::vector<Real> phi;
    std::vector<Real> fn;
    uint3 m_dim;         //!< The grid dimensions of the underlying grid
    };

//! Export SnapshotGridData to python
void export_SnapshotGridData(pybind11::module& m);

//! Export GridData to python
void export_GridData(pybind11::module& m);

} // end namespace solvent

#endif // __GRID_DATA_H__
