// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

/*! \file GridPotentialPair.h
    \brief Defines the GridPotentialPair template class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <memory>

#include "GridData.h"
#include "GridForceCompute.h"

#include "hoomd/CellList.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __GRID_POTENTIAL_PAIR_H__
#define __GRID_POTENTIAL_PAIR_H__

namespace solvent
{

template<class evaluator>
class GridPotentialPair : public GridForceCompute
    {
    //! Param type from evaluator
    typedef typename evaluator::param_type param_type;

    public:
        //! Constructor
        GridPotentialPair(std::shared_ptr<SystemDefinition>,
            std::shared_ptr<CellList> cl,
            std::shared_ptr<LevelSetSolver> solver);

        //! Destructor
        virtual ~GridPotentialPair();

        //! Pre-compute the energy terms on the grid
        virtual void precomputeEnergyTerms(unsigned int timestep;

        //! Actually compute the forces
        virtual void computeGridForces(unsigned int timestep) {}

        //! Set the parameters for a single type
        virtual void setParams(unsigned int type, const param_type& param);

        //! Set the rcut for a single type
        virtual void setRcut(unsigned int type, Scalar rcut);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();
        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Shifting modes that can be applied to the energy
        enum energyShiftMode
            {
            no_shift = 0,
            shift
            };

        //! Set the mode to use for shifting the energy
        void setShiftMode(energyShiftMode mode)
            {
            m_shift_mode = mode;
            }

    protected:
        std::shared_ptr<CellList> m_cl; //!< The solute cell list
        std::shared_ptr<LevelSetSolver> m_solver; //!< The level set solver

        energyShiftMode m_shift_mode;               //!< Store the mode with which to handle the energy shift at r_cut

        unsigned int m_radius;          //!< Radius of cells to take into account for interpolation
        bool m_diameter_shift;          //!< True if diameter shifting is enabled
        Scalar m_d;                     //!< Solvent diameter
        Scalar m_q;                     //!< Solvent charge

        GPUArray<param_type> m_params; //!< A list of potential parameters, per type
        GPUArray<Scalar> m_rcutsq;     //!< Cuttoff radius squared per type

        std::string m_prof_name;                    //!< Cached profiler name
        std::string m_log_name;                     //!< Cached log name

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange()
            {
            unsigned int ntypes  = m_pdata->getNTypes();
            // reallocate parameter arrays
            GPUArray<Scalar> rcutsq(ntypes, m_exec_conf);
            m_rcutsq.swap(rcutsq);
            GPUArray<param_type> params(ntypes, m_exec_conf);
            m_params.swap(params);
            }
    };

//! Constructor
template<class evaluator>
GridPotentialPair::GridPotentialPair(std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<CellList> cl,
    std::shared_ptr<LevelSetSolver> solver)
    : GridForceCompute(sysdef), m_cl(cl), m_solver(solver),
      m_shift_mode(no_shift),
      m_radius(1), m_diameter_shift(false), m_d(1.0), m_q(0.0)
    {
    assert(m_sysdef);
    assert(m_solver);

    // create a default cell list if none was specified
    if (!m_cl)
        m_cl = std::shared_ptr<CellList>(new CellList(sysdef));

    // request cell list properties
    m_cl->setRadius(0); // we don't need the adjacency matrix
    m_cl->setComputeTDB(false);
    m_cl->setFlagIndex();

    GPUArray<Scalar> rcutsq(m_pdata->getNTypes(), m_exec_conf);
    m_rcutsq.swap(rcutsq);
    GPUArray<param_type> params(m_pdata->getNTypes(), m_exec_conf);
    m_params.swap(params);

    // initialize name
    m_prof_name = std::string("Solvent Pair ") + evaluator::getName();
    m_log_name = std::string("solvent_pair_") + evaluator::getName() + std::string("_energy") + log_suffix;

    // connect to the ParticleData to receive notifications when the maximum number of particles changes
    m_pdata->getNumTypesChangeSignal().template connect<GridPotentialPair<evaluator>, &GridPotentialPair<evaluator>::slotNumTypesChange>(this);
    }

//! Set the parameters for a single type
template<class evaluator>
void GridPotentialPair<evaluator>::setParams(unsigned int type, const param_type& param)
    {
    if (type >= m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "solvent.pair." << evaluator::getName()
            << ": Trying to set params for a non existant type! " << type<< std::endl;
        throw std::runtime_error("Error setting parameters in GridPotentialPair");
        }

    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type] = param;
    }

//! Set rcut for a single type
template<class evaluator>
void GridPotentialPair<evaluator>::setRcut(unsigned int type, Scalar rcut)
    {
    if (type >= m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "solvent.pair." << evaluator::getName()
            << ": Trying to set rcut for a non existant type! "<< type << std::endl;
        throw std::runtime_error("Error setting parameters in GridPotentialPair");
        }

    ArrayHandle<Scalar> h_rcutsq(m_params, access_location::host, access_mode::readwrite);
    h_rcutsq.data[type] = rcut*rcut;
    }

/*! GridPotentialPair provides:
     - \c pair_"name"_energy
    where "name" is replaced with evaluator::getName()
*/
template< class evaluator >
std::vector< std::string > GridPotentialPair< evaluator >::getProvidedLogQuantities()
    {
    std::vector<std::string> list;
    list.push_back(m_log_name);
    return list;
    }

/*! \param quantity Name of the log value to get
    \param timestep Current timestep of the simulation
*/
template< class evaluator >
Scalar GridPotentialPair< evaluator >::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        this->m_exec_conf->msg->error() << "solvent.pair." << evaluator::getName() << ": " << quantity << " is not a valid log quantity"
                  << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    }

//! Compute particle forces
template< class evaluator >
void GridPotentialPair< evaluator >::precomputeEnergyTerms(unsigned int timestep)
    {
    // skip if we shouldn't compute this step
    if (!m_particles_sorted && !shouldCompute(timestep))
        return;

    m_particles_sorted = false;

    // begin by updating the CellList
    m_cl->compute(timestep);

    std::shared_ptr<GridData> grid_data(m_solver->getGridData());

    // get the velocity grid to precompute the energy on
    ArrayHandle<Scalar> h_fn(grid_data, access_location::host, access_mode::readwrite);

    Index3D gi = grid_data->getIndexer();

    // get the particle data arrays
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharge(), access_location::host, access_mode::read);

    // get the cell list arrays
    ArrayHandle<unsigned int> h_cell_size(m_cl->getCellSizeArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cell_idx(m_cl->getIndexArray(), access_location::host, access_mode::read);

    // arrays for cut-off and potential parameters
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

    int3 dim = grid_data->getDimensions();

    Scalar3 cell_width = m_cl->getCellWidth();
    uint3 cell_dim = m_cl->getDim();

    Index3D ci = m_cl->getCellIndexer();
    Index3D cli = m_cl->getCellListIndexer();

    const BoxDim& box = m_pdata->getBox();

    uchar3 periodic = box.getPeriodic();

    bool energy_shift = false;
    if (m_shift_mode == shift)
        energy_shift = true;

    // now loop over the grid cells, and update them with pair energy values
    for (unsigned int ix = 0; ix < dim.x; ++ix)
        for (unsigned int iy = 0; iy < dim.y; ++iy)
            for (unsigned int iz = 0; iz < dim.z; ++iz)
                {
                unsigned int grid_idx = gi(ix,iy,iz);

                // Lower left and upper right corners of grid cell in fractional coordinates
                vec3<Scalar> grid_lower(ix/dim.x, iy/dim.y, iz/dim.z);
                vec3<Scalar> grid_upper((ix+1)/dim.x, (iy+1)/dim.y, (iz+1)/dim.z);

                vec3<Scalar> center(((Scalar)ix+0.5)/dim.x, ((Scalar)iy+0.5)/dim.y, ((Scalar)iz+0.5)/dim.z);

                // find the intersecting cells
                int3 lower_idx, upper_idx;
                lower_idx.x = f_lower.x*cell_dim.x-m_radius;
                lower_idx.y = f_lower.y*cell_dim.x-m_radius;
                lower_idx.z = f_lower.z*cell_dim.x-m_radius;
                upper_idx.x = ceil(f_upper.x*cell_dim.x)+m_radius;
                upper_idx.y = ceil(f_upper.y*cell_dim.x)+m_radius;
                upper_idx.z = ceil(f_upper.z*cell_dim.x)+m_radius;

                for (unsigned int jx = lower_idx.x; jx < upper_idx.x; ++jx)
                    for (unsigned int jy = lower_idx.y; jy < upper_idx.y; ++jy)
                        for (unsigned int jz = lower_idx.z; jz < upper_idx.z; ++jz)
                            {
                            if (periodic.x && jx < 0)
                                jx += cell_dim.x;
                            if (periodic.y && jy < 0)
                                jy += cell_dim.y;
                            if (periodic.z && jz < 0)
                                jz += cell_dim.z;
                            if (periodic.x && jx >= cell_dim.x)
                                jx -= cell_dim.x;
                            if (periodic.y && jy >= cell_dim.y)
                                jy -= cell_dim.y;
                            if (periodic.z && jz >= cell_dim.z)
                                jz -= cell_dim.z;

                            unsigned int cell_idx = ci(jx,jy,jz);

                            unsigned int cell_size = h_cell_size[cell_idx];

                            // add potential energies from particles in cell
                            for (unsigned int cur_ptl = 0; cur_ptl < cell_size; ++cur_ptl)
                                {
                                unsigned int idx_i = h_cell_idx[cli(cur_ptl, cell_idx)];

                                Scalar4 postype = h_postype.data[idx_i];
                                vec3<Scalar> dr = center-vec3<Scalar>(postype);
                                unsigned int cur_type = __scalar_as_int(postype.w);

                                // shift into minimum image
                                dr = box.minImage(dr);

                                assert(cur_type < m_pdata->getNTypes());
                                Scalar rcutsq = h_rcutsq.data[cur_type];
                                Scalar drsq = dot(dr,dr);

                                Scalar sqshift = Scalar(0.0);
                                if (m_diameter_shift)
                                    {
                                    const Scalar delta = (m_d + h_diameter.data[idx_i])* Scalar(0.5) - Scalar(1.0);
                                    // r^2 < (r_list + delta)^2
                                    // r^2 < r_listsq + delta^2 + 2*r_list*delta
                                    sqshift = (delta + Scalar(2.0) * sqrt(rcutsq) * delta;
                                    }

                                // exclude if skipping is requested via r_cut
                                bool excluded = (r_cut <= Scalar(0.0));

                                // move the squared rlist by the diameter shift if necessary
                                if (drsq <= (rcutsq + sqshift) && !excluded)
                                    {
                                    // access diameter and charge (if needed)
                                    Scalar di = Scalar(0.0);
                                    Scalar qi = Scalar(0.0);
                                    if (evaluator::needsDiameter())
                                        di = h_diameter.data[i];
                                    if (evaluator::needsCharge())
                                        qi = h_charge.data[i];

                                    // compute the force and potential energy
                                    Scalar force_divr = Scalar(0.0);
                                    Scalar pair_eng = Scalar(0.0);
                                    evaluator eval(rsq, rcutsq, param);
                                    if (evaluator::needsDiameter())
                                        eval.setDiameter(di, dj);
                                    if (evaluator::needsCharge())
                                        eval.setCharge(qi, qj);

                                    bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);

                                    // update the energy on the grid cell
                                    h_fn.data[grid_idx] += pair_eng;
                                    }
                                }
                            }
                }
    }
} // end namespace

#endif // __GRID_PAIR_POTENTIAL_H__
