// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __HPMC_MONO_IMPLICIT_NEW_GPU_H__
#define __HPMC_MONO_IMPLICIT_NEW_GPU_H__

#ifdef ENABLE_CUDA

#include "IntegratorHPMCMonoImplicitNew.h"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitNewGPU.cuh"
#include "hoomd/Autotuner.h"

#include "hoomd/GPUVector.h"
#include "hoomd/CellListGPU.h"

#include <cuda_runtime.h>

/*! \file IntegratorHPMCMonoImplicitNewGPU.h
    \brief Defines the template class for HPMC with implicit generated depletant solvent on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace hpmc
{

//! Template class for HPMC update with implicit depletants on the GPU
/*!
    Depletants are generated randomly on the fly according to the semi-grand canonical ensemble.

    The penetrable depletants model is simulated.

    \ingroup hpmc_integrators
*/
template< class Shape >
class IntegratorHPMCMonoImplicitNewGPU : public IntegratorHPMCMonoImplicitNew<Shape>
    {
    public:
        //! Construct the integrator
        IntegratorHPMCMonoImplicitNewGPU(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<CellList> cl,
                              unsigned int seed);
        //! Destructor
        virtual ~IntegratorHPMCMonoImplicitNewGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            // call base class method first
            unsigned int ndim = this->m_sysdef->getNDimensions();
            if (ndim == 3)
                {
                m_tuner_update->setPeriod(period*this->m_nselect*8);
                m_tuner_moves->setPeriod(period*this->m_nselect);
                m_tuner_check_overlaps->setPeriod(period*this->m_nselect);
                m_tuner_accept->setPeriod(period*this->m_nselect*8);
                m_tuner_implicit->setPeriod(period*this->m_nselect*8);
                }
            else
                {
                m_tuner_update->setPeriod(period*this->m_nselect*4);
                m_tuner_moves->setPeriod(period*this->m_nselect);
                m_tuner_check_overlaps->setPeriod(period*this->m_nselect);
                m_tuner_accept->setPeriod(period*this->m_nselect*4);
                m_tuner_implicit->setPeriod(period*this->m_nselect*4);
                }

            m_tuner_update->setEnabled(enable);
            m_tuner_implicit->setEnabled(enable);

            m_tuner_moves->setEnabled(enable);
            m_tuner_check_overlaps->setEnabled(enable);
            m_tuner_accept->setEnabled(enable);

            m_tuner_excell_block_size->setPeriod(period);
            m_tuner_excell_block_size->setEnabled(enable);

            }

    protected:
        std::shared_ptr<CellList> m_cl;           //!< Cell list

        std::vector<std::unique_ptr<CellList> > m_cl_type;        //!< Per-type cell lists
        std::vector<GPUArray<unsigned int> > m_cell_sets;         //!< List of cells active during each subsweep, per type
        std::vector<GPUArray<unsigned int> > m_inverse_cell_set;  //!< Inverse permutation of cell sets per cell and type
        std::vector<Index2D> m_cell_set_indexer;                  //!< Indexer into the cell set array per type
        uint3 m_last_dim;                     //!< Dimensions of the cell list on the last call to update
        unsigned int m_last_ntypes;           //!< Number of types at last cell set initialization
        unsigned int m_last_nmax;             //!< Last cell list NMax value allocated in excell
        detail::UpdateOrder m_cell_set_order; //!< Update order for cell sets

        GPUVector<Scalar4> m_old_postype;                            //!< List of old particle positions
        GPUVector<Scalar4> m_old_orientation;                    //!< List of old particle orientations

        GPUArray<unsigned int> m_excell_idx;  //!< Particle indices in expanded cells
        GPUArray<unsigned int> m_excell_cell_set;  //!< Particle indices in expanded cells
        GPUArray<unsigned int> m_excell_size; //!< Number of particles in each expanded cell
        Index2D m_excell_list_indexer;        //!< Indexer to access elements of the excell_idx list
        GPUArray<unsigned int> m_excell_overlap; //!< Per neighbor flag, == 1 if update in old config, == 2 if update in new config

        std::unique_ptr<Autotuner> m_tuner_update;             //!< Autotuner for the update step group and block sizes
        std::unique_ptr<Autotuner> m_tuner_excell_block_size;  //!< Autotuner for excell block_size
        std::unique_ptr<Autotuner> m_tuner_implicit;           //!< Autotuner for the depletant overlap check

        std::unique_ptr<Autotuner> m_tuner_moves;             //!< Autotuner for proposing moves
        std::unique_ptr<Autotuner> m_tuner_check_overlaps;    //!< Autotuner for checking overlaps
        std::unique_ptr<Autotuner> m_tuner_accept;             //!< Autotuner for the acceptance stage

        GPUArray<curandState_t> m_curand_state_cell;               //!< Array of cuRAND states per active cell
        GPUArray<curandState_t> m_curand_state_cell_new;           //!< Array of cuRAND states per active cell after update
        GPUArray<unsigned int> m_overlap_cell;                   //!< Flag per cell to indicate overlap
        GPUArray<curandDiscreteDistribution_t> m_poisson_dist; //!< Handles for the poisson distribution histogram
        std::vector<bool> m_poisson_dist_created;               //!< Flag to indicate if Poisson distribution has been initialized

        GPUArray<unsigned int> m_active_cell_ptl_idx;  //!< List of update particle indicies per active cell
        GPUArray<unsigned int> m_active_cell_accept;   //!< List of accept/reject flags per active cell
        GPUArray<unsigned int> m_active_cell_move_type_translate;   //!< Type of move proposed in active cell

        GPUVector<Scalar4> m_trial_postype;                   //!< New positions (and type) of particles
        GPUVector<Scalar4> m_trial_orientation;               //!< New orientations
        GPUVector<unsigned int > m_trial_updated;             //!< per-particle flag if trial move has been carried out
        GPUVector<unsigned int> m_trial_move_type_translate;  //!< Flags to indicate which type of move

        cudaStream_t m_stream;                                  //! GPU kernel stream

        Index2D m_queue_indexer;                                //!< Indexer for overlap check queue
        GPUArray<unsigned int> m_queue_active_cell_idx;         //!< Queue of active cell indices
        GPUArray<Scalar4> m_queue_postype;                      //!< Queue of new particle positions and types
        GPUArray<Scalar4> m_queue_orientation;                  //!< Queue of new particle orientations
        GPUArray<unsigned int> m_queue_excell_idx;              //!< Queue of excell indices of neighbors
        GPUArray<unsigned int> m_cell_overlaps;                 //!< Result of queue overlap checks

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Initialize the Poisson distributions
        virtual void initializePoissonDistribution();

        //! Set up cell sets
        virtual void initializeCellSets();

        //! Set up excell_list
        virtual void initializeExcellMem();

        //! Set up queue
        virtual void initializeQueueMem();

        //! Update the cell width
        virtual void updateCellWidth();

        //! Update cell lists
        virtual void slotNumTypesChange()
            {
            // base class method
            IntegratorHPMCMonoImplicitNew<Shape>::slotNumTypesChange();

            initializeCellLists();
            }

        virtual void initializeCellLists()
            {
            m_cl_type.clear();

            for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
                {
                m_cl_type.push_back(std::unique_ptr<CellListGPU>(new CellListGPU(this->m_sysdef)));

                // set standard cell list flags
                m_cl_type[i]->setRadius(1);
                m_cl_type[i]->setComputeTDB(false);
                m_cl_type[i]->setFlagType();
                m_cl_type[i]->setComputeIdx(true);
                m_cl_type[i]->setMultiple(2);

                // specialize to this type
                m_cl_type[i]->setFilterType(true);
                m_cl_type[i]->setType(i);

                // set nominal width
                m_cl_type[i]->setNominalWidth(this->m_nominal_width);
                }
            }
    };

/*! \param sysdef System definition
    \param cl Cell list
    \param seed Random number generator seed
    */

template< class Shape >
IntegratorHPMCMonoImplicitNewGPU< Shape >::IntegratorHPMCMonoImplicitNewGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                                   std::shared_ptr<CellList> cl,
                                                                   unsigned int seed)
    : IntegratorHPMCMonoImplicitNew<Shape>(sysdef, seed), m_cl(cl), m_cell_set_order(this->m_exec_conf, seed+this->m_exec_conf->getRank())
    {
    this->m_exec_conf->msg->notice(5) << "Constructing IntegratorHPMCImplicitGPU" << std::endl;

    this->m_cl->setRadius(1);
    this->m_cl->setComputeTDB(false);
    this->m_cl->setFlagType();
    this->m_cl->setComputeIdx(true);

    // require that cell lists have an even number of cells along each direction
    this->m_cl->setMultiple(2);

    // set last dim to a bogus value so that it will re-init on the first call
    m_last_dim = make_uint3(0xffffffff, 0xffffffff, 0xffffffff);
    m_last_nmax = 0xffffffff;
    m_last_ntypes = UINT_MAX;

    GPUArray<unsigned int> excell_size(0, this->m_exec_conf);
    m_excell_size.swap(excell_size);

    GPUArray<unsigned int> excell_idx(0, this->m_exec_conf);
    m_excell_idx.swap(excell_idx);

    GPUArray<unsigned int>(0, this->m_exec_conf).swap(m_excell_cell_set);
    GPUArray<unsigned int>(0, this->m_exec_conf).swap(m_excell_overlap);

    GPUVector<Scalar4> old_postype(this->m_exec_conf);
    m_old_postype.swap(old_postype);

    GPUVector<Scalar4> old_orientation(this->m_exec_conf);
    m_old_orientation.swap(old_orientation);

    GPUVector<Scalar4>(this->m_exec_conf).swap(m_trial_postype);
    GPUVector<Scalar4>(this->m_exec_conf).swap(m_trial_orientation);
    GPUVector<unsigned int>(this->m_exec_conf).swap(m_trial_updated);
    GPUVector<unsigned int>(this->m_exec_conf).swap(m_trial_move_type_translate);

    // initialize the autotuners
    // the full block size, stride and group size matrix is searched,
    // encoded as block_size*1000000 + stride*100 + group_size.

    // valid params for dynamic parallelism kernel
    std::vector<unsigned int> valid_params_dp;

    // valid params for kernel without dynamic parallelism, but with extra shared memory tuning
    std::vector<unsigned int> valid_params_tune_shared;

    cudaDeviceProp dev_prop = this->m_exec_conf->dev_prop;

    // whether to load extra data into shared mem or fetch directly from global mem
    for (unsigned int load_shared = 0; load_shared < 2; ++load_shared)
        {
        for (unsigned int block_size_overlaps = dev_prop.warpSize; block_size_overlaps <= (unsigned int) dev_prop.maxThreadsPerBlock;
            block_size_overlaps += dev_prop.warpSize)
            {
            for (unsigned int block_size = dev_prop.warpSize; block_size <= (unsigned int) dev_prop.maxThreadsPerBlock; block_size += dev_prop.warpSize)
                {
                for (unsigned int group_size=1; group_size <= (unsigned int)dev_prop.warpSize; group_size++)
                    {
                    if ((block_size % group_size) == 0)
                        valid_params_dp.push_back(block_size*1000000 + block_size_overlaps*100 + group_size + load_shared * dev_prop.warpSize);
                    }
                }
            }
        for (unsigned int block_size = dev_prop.warpSize; block_size <= (unsigned int) dev_prop.maxThreadsPerBlock; block_size += dev_prop.warpSize)
            {
            for (unsigned int group_size=1; group_size <= (unsigned int)dev_prop.warpSize; group_size++)
                {
                if ((block_size % group_size) == 0)
                    valid_params_tune_shared.push_back(block_size*1000000 + group_size + load_shared * dev_prop.warpSize);
                }
            }
        }

    // parameters for other kernels
    std::vector<unsigned int> valid_params;

    for (unsigned int block_size = dev_prop.warpSize; block_size <= (unsigned int) dev_prop.maxThreadsPerBlock; block_size += dev_prop.warpSize)
        {
        for (unsigned int group_size=1; group_size <= (unsigned int)dev_prop.warpSize; group_size++)
            {
            if ((block_size % group_size) == 0)
                valid_params.push_back(block_size*1000000 +  group_size);
            }
        }

    m_tuner_update.reset(new Autotuner(valid_params, 5, 1000000, "hpmc_update", this->m_exec_conf));
    m_tuner_excell_block_size.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 1000000, "hpmc_excell_block_size", this->m_exec_conf));
    m_tuner_implicit.reset(new Autotuner(valid_params_dp, 5, 1000000, "hpmc_insert_depletants", this->m_exec_conf));

    m_tuner_moves.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 1000000, "hpmc_moves", this->m_exec_conf));
    m_tuner_check_overlaps.reset(new Autotuner(Shape::isParallel() ? valid_params_dp : valid_params_tune_shared,
        5, 1000000, "hpmc_check_overlaps", this->m_exec_conf));
    m_tuner_accept.reset(new Autotuner(valid_params, 5, 1000000, "hpmc_accept", this->m_exec_conf));

    GPUArray<hpmc_implicit_counters_t> implicit_count(1,this->m_exec_conf);
    this->m_implicit_count.swap(implicit_count);

    GPUArray<curandDiscreteDistribution_t> poisson_dist(1,this->m_exec_conf);
    m_poisson_dist.swap(poisson_dist);

    m_poisson_dist_created.resize(this->m_pdata->getNTypes(), false);

    // create a CUDA stream for kernel execution
    cudaStreamCreate(&m_stream);
    CHECK_CUDA_ERROR();

    if (Shape::isParallel() && this->m_exec_conf->getComputeCapability() > 300)
        {
        GPUArray<unsigned int>(0, this->m_exec_conf).swap(m_queue_active_cell_idx);
        GPUArray<Scalar4>(0, this->m_exec_conf).swap(m_queue_postype);
        GPUArray<Scalar4>(0, this->m_exec_conf).swap(m_queue_orientation);
        GPUArray<unsigned int>(0, this->m_exec_conf).swap(m_queue_excell_idx);
        GPUArray<unsigned int>(0, this->m_exec_conf).swap(m_cell_overlaps);
        }

    // initialize cell lists
    initializeCellLists();
    }

//! Destructor
template< class Shape >
IntegratorHPMCMonoImplicitNewGPU< Shape >::~IntegratorHPMCMonoImplicitNewGPU()
    {
    // destroy the registered poisson RNG's
    ArrayHandle<curandDiscreteDistribution_t> h_poisson_dist(m_poisson_dist, access_location::host, access_mode::read);
    for (unsigned int type = 0; type < this->m_pdata->getNTypes(); ++type)
        {
        if (m_poisson_dist_created[type])
            {
            curandDestroyDistribution(h_poisson_dist.data[type]);
            }
        }

    cudaStreamDestroy(m_stream);
    CHECK_CUDA_ERROR();
    }

template< class Shape >
void IntegratorHPMCMonoImplicitNewGPU< Shape >::update(unsigned int timestep)
    {
    IntegratorHPMC::update(timestep);

    if (this->m_exec_conf->getComputeCapability() < 350)
        {
        // update poisson distributions
        if (this->m_need_initialize_poisson)
            {
            this->updatePoissonParameters();
            initializePoissonDistribution();
            this->m_need_initialize_poisson = false;
            }
        }

        {
        ArrayHandle<hpmc_implicit_counters_t> h_implicit_counters(this->m_implicit_count, access_location::host, access_mode::readwrite);
        this->m_implicit_count_step_start = h_implicit_counters.data[0];
        }

    // check if we are below a minimum image convention box size
    BoxDim box = this->m_pdata->getBox();
    Scalar3 npd = box.getNearestPlaneDistance();

    if ((box.getPeriodic().x && npd.x <= this->m_nominal_width*2) ||
        (box.getPeriodic().y && npd.y <= this->m_nominal_width*2) ||
        (this->m_sysdef->getNDimensions() == 3 && box.getPeriodic().z && npd.z <= this->m_nominal_width*2))
        {
        this->m_exec_conf->msg->error() << "Simulation box too small for implicit depletant simulations on GPU - increase it so the minimum image convention works" << std::endl;
        throw std::runtime_error("Error performing HPMC update");
        }


    // update the cell list
    this->m_cl->compute(timestep);

    // update per-type cell lists
    for (unsigned int itype = 0; itype < this->m_pdata->getNTypes(); ++itype)
        {
        m_cl_type[itype]->compute(timestep);
        }

    // start the profile
    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "HPMC");

    // rng for shuffle and grid shift
    hoomd::detail::Saru rng(this->m_seed, timestep, 0xf4a3210e);

    // if the cell list is a different size than last time, reinitialize the cell sets list
    uint3 cur_dim = this->m_cl->getDim();

    if (this->m_last_dim.x != cur_dim.x || this->m_last_dim.y != cur_dim.y || this->m_last_dim.z != cur_dim.z || m_last_ntypes != this->m_pdata->getNTypes())
        {
        this->initializeCellSets();
        this->initializeExcellMem();

        if (Shape::isParallel() && this->m_exec_conf->getComputeCapability() > 300)
            {
            this->initializeQueueMem();
            }

        this->m_last_dim = cur_dim;
        this->m_last_nmax = this->m_cl->getNmax();

        this->m_last_ntypes = this->m_pdata->getNTypes();

        // initialize the cell set update order
        assert(this->m_pdata->getNTypes() > 0);
        this->m_cell_set_order.resize(m_cell_set_indexer[0].getH());
        }

    unsigned int last_n_active = this->m_cell_set_indexer[0].getW();

    // initialize RNG states
    // NOTE these arrays rely on all cell sets having the same dimensions
    if (this->m_cell_set_indexer[0].getW() != last_n_active)
        {
        GPUArray<curandState_t> curand_state_cell(this->m_cell_set_indexer[0].getW(), this->m_exec_conf);
        m_curand_state_cell.swap(curand_state_cell);

        GPUArray<curandState_t> curand_state_cell_new(this->m_cell_set_indexer[0].getW(), this->m_exec_conf);
        m_curand_state_cell_new.swap(curand_state_cell_new);

        GPUArray<unsigned int> overlap_cell(this->m_cell_set_indexer[0].getW(), this->m_exec_conf);
        m_overlap_cell.swap(overlap_cell);

        GPUArray<unsigned int> active_cell_ptl_idx(this->m_cell_set_indexer[0].getW(), this->m_exec_conf);
        m_active_cell_ptl_idx.swap(active_cell_ptl_idx);

        GPUArray<unsigned int> active_cell_accept(this->m_cell_set_indexer[0].getW(), this->m_exec_conf);
        m_active_cell_accept.swap(active_cell_accept);

        GPUArray<unsigned int> active_cell_move_type_translate(this->m_cell_set_indexer[0].getW(), this->m_exec_conf);
        m_active_cell_move_type_translate.swap(active_cell_move_type_translate);
        }

    // if only NMax changed, only need to reallocate excell memory
    if (this->m_last_nmax != this->m_cl->getNmax())
        {
        this->initializeExcellMem();

        if (Shape::isParallel() && this->m_exec_conf->getComputeCapability() > 300)
            {
            this->initializeQueueMem();
            }

        this->m_last_nmax = this->m_cl->getNmax();
        }

    // test if we are in domain decomposition mode
    bool domain_decomposition = false;
#ifdef ENABLE_MPI
    if (this->m_comm)
        domain_decomposition = true;
#endif

    bool have_depletants = this->getDepletantDensity() > Scalar(0.0);

    this->m_old_postype.resize(this->m_pdata->getMaxN());
    this->m_old_orientation.resize(this->m_pdata->getMaxN());

    if (this->m_exec_conf->getComputeCapability() < 350)
        {
        // no dynamic parallelism

        // access the particle data
        ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);

        ArrayHandle< unsigned int > d_excell_idx(this->m_excell_idx, access_location::device, access_mode::readwrite);
        ArrayHandle< unsigned int > d_excell_size(this->m_excell_size, access_location::device, access_mode::readwrite);

        // access the parameters and interaction matrix
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = this->getParams();
        ArrayHandle<unsigned int> d_overlaps(this->m_overlaps, access_location::device, access_mode::read);

        Scalar3 ghost_width = this->m_cl->getGhostWidth();
        Scalar3 ghost_fraction = this->m_nominal_width / npd;

            {
            // access the global cell list data
            ArrayHandle<unsigned int> d_cell_size(this->m_cl->getCellSizeArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_cell_idx(this->m_cl->getIndexArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_cell_adj(this->m_cl->getCellAdjArray(), access_location::device, access_mode::read);

            // update the expanded cells
            this->m_tuner_excell_block_size->begin();
            detail::gpu_hpmc_excell(d_excell_idx.data,
                                    d_excell_size.data,
                                    this->m_excell_list_indexer,
                                    d_cell_idx.data,
                                    d_cell_size.data,
                                    d_cell_adj.data,
                                    this->m_cl->getCellIndexer(),
                                    this->m_cl->getCellListIndexer(),
                                    this->m_cl->getCellAdjIndexer(),
                                    this->m_tuner_excell_block_size->getParam());
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            this->m_tuner_excell_block_size->end();
            }

        for (unsigned int itype = 0; itype < this->m_pdata->getNTypes(); ++itype)
            {
                {
                // optimization
                ArrayHandle<Scalar> h_d(this->m_d, access_location::host, access_mode::read);
                ArrayHandle<Scalar> h_a(this->m_a, access_location::host, access_mode::read);

                if (h_d.data[itype]==Scalar(0.0) && (!this->m_hasOrientation || h_a.data[itype] == 0))
                    continue;
                }

            // access the move sizes by type
            ArrayHandle<Scalar> d_d(this->m_d, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_a(this->m_a, access_location::device, access_mode::read);

            // access the per-type cell list data
            ArrayHandle<unsigned int> d_cell_size(this->m_cl_type[itype]->getCellSizeArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_cell_idx(this->m_cl_type[itype]->getIndexArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_cell_adj(this->m_cl_type[itype]->getCellAdjArray(), access_location::device, access_mode::read);

            // compute the ceiling of the average number of particles in each cell, accounting for ghost particles
            uint3 dim = this->m_cl_type[itype]->getDim();
            int ncells = dim.x * dim.y * dim.z;
            int particles_per_cell = int(ceil(double(this->m_cl_type[itype]->getNumParticles()) / double(ncells)));

            // access cell sets
            ArrayHandle< unsigned int > d_cell_sets(this->m_cell_sets[itype], access_location::device, access_mode::read);

            // on first iteration, synchronize GPU execution stream and update shape parameters
            bool first = true;

            for (unsigned int i = 0; i < this->m_nselect*particles_per_cell; i++)
                {
                // loop over cell sets in a shuffled order
                this->m_cell_set_order.shuffle(timestep,i);

                for (unsigned int j = 0; j < this->m_cell_set_indexer[itype].getH(); j++)
                    {
                    unsigned cur_set = this->m_cell_set_order[j];

                    // save old positions
                    ArrayHandle<Scalar4> d_old_postype(this->m_old_postype, access_location::device, access_mode::overwrite);
                    ArrayHandle<Scalar4> d_old_orientation(this->m_old_orientation, access_location::device, access_mode::overwrite);

                    cudaMemcpyAsync(d_old_postype.data, d_postype.data, sizeof(Scalar4)*(this->m_pdata->getN()+this->m_pdata->getNGhosts()), cudaMemcpyDeviceToDevice, m_stream);
                    cudaMemcpyAsync(d_old_orientation.data, d_orientation.data, sizeof(Scalar4)*(this->m_pdata->getN()+this->m_pdata->getNGhosts()), cudaMemcpyDeviceToDevice, m_stream);

                    // flags about updated particles
                    ArrayHandle<unsigned int> d_active_cell_ptl_idx(m_active_cell_ptl_idx, access_location::device, access_mode::overwrite);
                    ArrayHandle<unsigned int> d_active_cell_accept(m_active_cell_accept, access_location::device, access_mode::overwrite);
                    ArrayHandle<unsigned int> d_active_cell_move_type_translate(m_active_cell_move_type_translate, access_location::device, access_mode::overwrite);

                    ArrayHandle<hpmc_counters_t> d_counters(this->m_count_total, access_location::device, access_mode::readwrite);

                    // move particles
                    this->m_tuner_update->begin();

                    auto dev_prop = this->m_exec_conf->dev_prop;

                    unsigned int param = this->m_tuner_update->getParam();
                    unsigned int block_size = param / 1000000;
                    unsigned int block_size_overlaps = (param % 1000000 ) / 100;
                    unsigned int group_size = ((param % 100)-1) % (dev_prop.warpSize) + 1;

                    auto args = detail::hpmc_args_t(d_postype.data,
                            d_orientation.data,
                            d_counters.data,
                            d_cell_idx.data,
                            d_cell_size.data,
                            d_excell_idx.data,
                            0, // excell_cell_set
                            0, // excell_overlap
                            d_excell_size.data,
                            this->m_cl_type[itype]->getCellIndexer(),
                            this->m_cl_type[itype]->getCellListIndexer(),
                            this->m_excell_list_indexer,
                            this->m_cl_type[itype]->getDim(),
                            ghost_width,
                            &d_cell_sets.data[this->m_cell_set_indexer[itype](0,cur_set)],
                            this->m_cell_set_indexer[itype].getW(),
                            this->m_pdata->getN(),
                            this->m_pdata->getNTypes(),
                            this->m_seed + this->m_exec_conf->getRank()*this->m_nselect,
                            d_d.data,
                            d_a.data,
                            d_overlaps.data,
                            this->m_overlap_idx,
                            this->m_move_ratio,
                            timestep,
                            this->m_sysdef->getNDimensions(),
                            box,
                            i+particles_per_cell*this->m_nselect*(3*j),
                            ghost_fraction,
                            domain_decomposition,
                            block_size,
                            1, //stride
                            group_size,
                            this->m_hasOrientation,
                            this->m_pdata->getMaxN(),
                            dev_prop,
                            first,
                            m_stream,
                            have_depletants ? d_active_cell_ptl_idx.data : 0,
                            have_depletants ? d_active_cell_accept.data : 0,
                            have_depletants ? d_active_cell_move_type_translate.data : 0);

                    detail::gpu_hpmc_update<Shape>(args, params.data());

                    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();

                    this->m_tuner_update->end();

                    if (have_depletants)
                        {
                        if (this->m_prof) this->m_prof->push(this->m_exec_conf,"Depletants");

                        // RNG state
                        ArrayHandle<curandState_t> d_curand_state_cell(this->m_curand_state_cell, access_location::device, access_mode::readwrite);
                        ArrayHandle<curandState_t> d_curand_state_cell_new(this->m_curand_state_cell_new, access_location::device, access_mode::readwrite);

                        // overlap flags
                        ArrayHandle<unsigned int> d_overlap_cell(this->m_overlap_cell, access_location::device, access_mode::overwrite);

                        // min/max diameter of insertion sphere
                        ArrayHandle<Scalar> d_d_min(this->m_d_min, access_location::device, access_mode::read);
                        ArrayHandle<Scalar> d_d_max(this->m_d_max, access_location::device, access_mode::read);

                        // Poisson distribution
                        ArrayHandle<curandDiscreteDistribution_t> d_poisson_dist(m_poisson_dist, access_location::device, access_mode::read);

                            {
                            // counters
                            ArrayHandle<hpmc_implicit_counters_t> d_implicit_count(this->m_implicit_count, access_location::device, access_mode::readwrite);

                            // Kernel driver arguments
                            unsigned int param = m_tuner_implicit->getParam();
                            unsigned int block_size = param / 1000000;
                            unsigned int stride = (param % 1000000) / 100;
                            unsigned int group_size = ((param % 100)-1) % (dev_prop.warpSize) + 1;
                            bool load_shared = ((param % 100)-1) / (dev_prop.warpSize);

                            m_tuner_implicit->begin();

                            unsigned int block_size_overlaps = 512; // for now

                            // kernel parameters
                            auto args =   detail::hpmc_implicit_args_new_t(d_postype.data,
                                    d_orientation.data,
                                    d_old_postype.data,
                                    d_old_orientation.data,
                                    d_cell_idx.data,
                                    d_cell_size.data,
                                    d_excell_idx.data,
                                    d_excell_size.data,
                                    this->m_cl_type[itype]->getCellIndexer(),
                                    this->m_cl_type[itype]->getCellListIndexer(),
                                    this->m_excell_list_indexer,
                                    this->m_cl_type[itype]->getDim(),
                                    ghost_width,
                                    &d_cell_sets.data[this->m_cell_set_indexer[itype](0,cur_set)],
                                    this->m_cell_set_indexer[itype].getW(),
                                    this->m_pdata->getN(),
                                    this->m_pdata->getNTypes(),
                                    this->m_seed + this->m_exec_conf->getRank()*this->m_nselect,
                                    d_overlaps.data,
                                    this->m_overlap_idx,
                                    timestep,
                                    this->m_sysdef->getNDimensions(),
                                    box,
                                    i+particles_per_cell*this->m_nselect*(3*j+1),
                                    block_size,
                                    stride,
                                    group_size,
                                    this->m_hasOrientation,
                                    this->m_pdata->getMaxN(),
                                    this->m_exec_conf->dev_prop,
                                    d_curand_state_cell.data,
                                    d_curand_state_cell_new.data,
                                    this->m_type,
                                    d_counters.data,
                                    d_implicit_count.data,
                                    d_poisson_dist.data,
                                    d_overlap_cell.data,
                                    d_active_cell_ptl_idx.data,
                                    d_active_cell_accept.data,
                                    d_active_cell_move_type_translate.data,
                                    d_d_min.data,
                                    d_d_max.data,
                                    first,
                                    this->getDepletantDensity(),
                                    m_stream,
                                    this->m_exec_conf->isCUDAErrorCheckingEnabled(),
                                    block_size_overlaps,
                                    load_shared);

                            detail::gpu_hpmc_insert_depletants_queue<Shape>(args, params.data());

                            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                                CHECK_CUDA_ERROR();

                            m_tuner_implicit->end();
                            }

                            {
                            // counters
                            ArrayHandle<hpmc_implicit_counters_t> d_implicit_count(this->m_implicit_count, access_location::device, access_mode::readwrite);

                            // apply acceptance/rejection criterium
                            detail::gpu_hpmc_implicit_accept_reject_new<Shape>(
                                detail::hpmc_implicit_args_new_t(d_postype.data,
                                    d_orientation.data,
                                    d_old_postype.data,
                                    d_old_orientation.data,
                                    d_cell_idx.data,
                                    d_cell_size.data,
                                    d_excell_idx.data,
                                    d_excell_size.data,
                                    this->m_cl_type[itype]->getCellIndexer(),
                                    this->m_cl_type[itype]->getCellListIndexer(),
                                    this->m_excell_list_indexer,
                                    this->m_cl_type[itype]->getDim(),
                                    ghost_width,
                                    &d_cell_sets.data[this->m_cell_set_indexer[itype](0,cur_set)],
                                    this->m_cell_set_indexer[itype].getW(),
                                    this->m_pdata->getN(),
                                    this->m_pdata->getNTypes(),
                                    this->m_seed + this->m_exec_conf->getRank(),
                                    d_overlaps.data,
                                    this->m_overlap_idx,
                                    timestep,
                                    this->m_sysdef->getNDimensions(),
                                    box,
                                    i+particles_per_cell*this->m_nselect*(3*j+2),
                                    256, // block_size
                                    1, // stride
                                    1, //group_size
                                    this->m_hasOrientation,
                                    this->m_pdata->getMaxN(),
                                    this->m_exec_conf->dev_prop,
                                    d_curand_state_cell.data,
                                    d_curand_state_cell_new.data,
                                    this->m_type,
                                    d_counters.data,
                                    d_implicit_count.data,
                                    d_poisson_dist.data,
                                    d_overlap_cell.data,
                                    d_active_cell_ptl_idx.data,
                                    d_active_cell_accept.data,
                                    d_active_cell_move_type_translate.data,
                                    d_d_min.data,
                                    d_d_max.data,
                                    first,
                                    this->getDepletantDensity(),
                                    m_stream,
                                    this->m_exec_conf->isCUDAErrorCheckingEnabled(),
                                    block_size_overlaps,
                                    false // load_shared
                                    ),
                                params.data());

                            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                                CHECK_CUDA_ERROR();
                            }
                        if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
                        }

                    first = false;
                    } // end loop over cell sets
                } // end loop nselect*particles_per_cell
            } // end loop over types
        }
    else
        { // compute capability >= 350

        // access the particle data
        ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);

        // access the parameters and interaction matrix
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = this->getParams();
        ArrayHandle<unsigned int> d_overlaps(this->m_overlaps, access_location::device, access_mode::read);

        Scalar3 ghost_width = this->m_cl->getGhostWidth();
        Scalar3 ghost_fraction = this->m_nominal_width / npd;

        for (unsigned int itype = 0; itype < this->m_pdata->getNTypes(); ++itype)
            {
                {
                // optimization
                ArrayHandle<Scalar> h_d(this->m_d, access_location::host, access_mode::read);
                ArrayHandle<Scalar> h_a(this->m_a, access_location::host, access_mode::read);

                if (h_d.data[itype]==Scalar(0.0) && (!this->m_hasOrientation || h_a.data[itype] == 0))
                    continue;
                }


            // access the move sizes by type
            ArrayHandle<Scalar> d_d(this->m_d, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_a(this->m_a, access_location::device, access_mode::read);

            // accces cell set
            ArrayHandle< unsigned int > d_cell_sets(this->m_cell_sets[itype], access_location::device, access_mode::read);

            // compute the ceiling of the average number of particles in each cell, accounting for ghost particles
            uint3 dim = this->m_cl_type[itype]->getDim();
            int ncells = dim.x * dim.y * dim.z;
            int particles_per_cell = int(ceil(double(this->m_cl_type[itype]->getNumParticles()) / double(ncells)));

                {
                if (this->m_prof)
                    this->m_prof->push(this->m_exec_conf, "ex_cell");

                // access the global cell list data
                ArrayHandle<unsigned int> d_cell_size(this->m_cl->getCellSizeArray(), access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_cell_idx(this->m_cl->getIndexArray(), access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_cell_adj(this->m_cl->getCellAdjArray(), access_location::device, access_mode::read);

                // cell sets for this type

                // for now, we are writing out an expanded cell list for every type separately
                // this is certainly not efficient, but we need the cell set pointer to be updated

                // NOTE this works only if all cell lists have equal dimensions!
                assert(m_cl_type[itype]->getDim() == m_cl->getDim());
                ArrayHandle<unsigned int> d_inverse_cell_set(this->m_inverse_cell_set[itype], access_location::device, access_mode::read);

                ArrayHandle< unsigned int > d_excell_idx(this->m_excell_idx, access_location::device, access_mode::overwrite);
                ArrayHandle< unsigned int > d_excell_cell_set(this->m_excell_cell_set, access_location::device, access_mode::overwrite);
                ArrayHandle< unsigned int > d_excell_size(this->m_excell_size, access_location::device, access_mode::readwrite);

                // update the expanded cells and update cell set lookup
                this->m_tuner_excell_block_size->begin();
                detail::gpu_hpmc_excell_and_cell_set(d_inverse_cell_set.data,
                                        d_excell_idx.data,
                                        d_excell_cell_set.data,
                                        d_excell_size.data,
                                        this->m_excell_list_indexer,
                                        d_cell_idx.data,
                                        d_cell_size.data,
                                        d_cell_adj.data,
                                        this->m_cl->getCellIndexer(),
                                        this->m_cl->getCellListIndexer(),
                                        this->m_cl->getCellAdjIndexer(),
                                        this->m_tuner_excell_block_size->getParam());
                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();

                this->m_tuner_excell_block_size->end();

                if (this->m_prof)
                    this->m_prof->pop(this->m_exec_conf);
                }

            // access the cell list data
            ArrayHandle<unsigned int> d_cell_size(this->m_cl_type[itype]->getCellSizeArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_cell_idx(this->m_cl_type[itype]->getIndexArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_cell_adj(this->m_cl_type[itype]->getCellAdjArray(), access_location::device, access_mode::read);

            m_trial_postype.resize(this->m_pdata->getMaxN());
            m_trial_orientation.resize(this->m_pdata->getMaxN());
            m_trial_updated.resize(this->m_pdata->getMaxN());
            m_trial_move_type_translate.resize(this->m_cl_type[itype]->getCellIndexer().getNumElements());

            // access data for proposed moves
            ArrayHandle<Scalar4> d_trial_postype(m_trial_postype, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_trial_orientation(m_trial_orientation, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_trial_updated(m_trial_updated, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_trial_move_type_translate(m_trial_move_type_translate, access_location::device, access_mode::overwrite);

            // queue arrays for overlap checks
            ArrayHandle<unsigned int> d_queue_active_cell_idx(m_queue_active_cell_idx, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_queue_postype(m_queue_postype, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_queue_orientation(m_queue_orientation, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_queue_excell_idx(m_queue_excell_idx, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_cell_overlaps(m_cell_overlaps, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_excell_overlap(m_excell_overlap, access_location::device, access_mode::overwrite);

            // on first iteration, synchronize GPU execution stream and update shape parameters
            bool first = true;

            // flags about updated particles
            ArrayHandle<unsigned int> d_active_cell_ptl_idx(m_active_cell_ptl_idx, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_active_cell_accept(m_active_cell_accept, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_active_cell_move_type_translate(m_active_cell_move_type_translate, access_location::device, access_mode::overwrite);

            // backup of particle positions and orientations
            ArrayHandle<Scalar4> d_old_postype(this->m_old_postype, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_old_orientation(this->m_old_orientation, access_location::device, access_mode::overwrite);

            ArrayHandle< unsigned int > d_excell_idx(this->m_excell_idx, access_location::device, access_mode::read);
            ArrayHandle< unsigned int > d_excell_cell_set(this->m_excell_cell_set, access_location::device, access_mode::read);
            ArrayHandle< unsigned int > d_excell_size(this->m_excell_size, access_location::device, access_mode::read);

            for (unsigned int i = 0; i < this->m_nselect*particles_per_cell; i++)
                {
                // loop over cell sets in a shuffled order
                this->m_cell_set_order.shuffle(timestep,i);

                // propose moves
                ArrayHandle<hpmc_counters_t> d_counters(this->m_count_total, access_location::device, access_mode::readwrite);

                // move particles in all active cells
                this->m_tuner_moves->begin();
                unsigned int block_size = m_tuner_moves->getParam();
                auto dev_prop = this->m_exec_conf->dev_prop;

                auto args = detail::hpmc_args_t(d_postype.data,
                        d_orientation.data,
                        d_counters.data,
                        d_cell_idx.data,
                        d_cell_size.data,
                        d_excell_idx.data,
                        d_excell_cell_set.data,
                        d_excell_overlap.data,
                        d_excell_size.data,
                        this->m_cl_type[itype]->getCellIndexer(),
                        this->m_cl_type[itype]->getCellListIndexer(),
                        this->m_excell_list_indexer,
                        this->m_cl_type[itype]->getDim(),
                        ghost_width,
                        d_cell_sets.data,
                        this->m_cell_set_indexer[itype].getW(),
                        this->m_pdata->getN(),
                        this->m_pdata->getNTypes(),
                        this->m_seed + this->m_exec_conf->getRank()*this->m_nselect,
                        d_d.data,
                        d_a.data,
                        d_overlaps.data,
                        this->m_overlap_idx,
                        this->m_move_ratio,
                        timestep,
                        this->m_sysdef->getNDimensions(),
                        box,
                        i+particles_per_cell*this->m_nselect,
                        ghost_fraction,
                        domain_decomposition,
                        block_size,
                        1, //stride
                        0, // group_size
                        this->m_hasOrientation,
                        this->m_pdata->getMaxN(),
                        dev_prop,
                        first,
                        m_stream,
                        have_depletants ? d_active_cell_ptl_idx.data : 0,
                        have_depletants ? d_active_cell_accept.data : 0,
                        have_depletants ? d_active_cell_move_type_translate.data : 0,
                        m_queue_indexer,
                        d_queue_active_cell_idx.data,
                        d_queue_postype.data,
                        d_queue_orientation.data,
                        d_queue_excell_idx.data,
                        d_cell_overlaps.data,
                        m_queue_indexer.getH(),
                        this->m_exec_conf->isCUDAErrorCheckingEnabled(),
                        0, // block_size_overlaps
                        0, // load_shared
                        d_trial_postype.data,
                        d_trial_orientation.data,
                        d_trial_updated.data,
                        d_trial_move_type_translate.data,
                        0, // d_update_order
                        0, // cur_set
                        this->m_cell_set_indexer[itype]
                        );

                    {
                    ArrayHandle<unsigned int> d_update_order(this->m_cell_set_order.getInverse(), access_location::device, access_mode::read);
                    args.d_update_order = d_update_order.data;
                    }

                detail::gpu_hpmc_moves<Shape>(args, params.data());

                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                m_tuner_moves->end();

                // check for overlaps between old and old, and old and new configuration
                m_tuner_check_overlaps->begin();

                unsigned int param = this->m_tuner_check_overlaps->getParam();
                block_size = param / 1000000;
                unsigned int block_size_overlaps = (param % 1000000 ) / 100;
                unsigned int group_size = ((param % 100)-1) % (dev_prop.warpSize) + 1;
                bool load_shared = ((param % 100)-1) / (dev_prop.warpSize);

                args.group_size = group_size;
                args.block_size_overlaps = block_size_overlaps;
                args.block_size = block_size;
                args.load_shared = load_shared;

                detail::gpu_hpmc_check_overlaps<Shape>(args, params.data());

                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();

                m_tuner_check_overlaps->end();

                ArrayHandle<unsigned int> h_update_order(this->m_cell_set_order.get(), access_location::host, access_mode::read);

                // acceptance, this part is serial over the cell sets
                for (unsigned int j = 0; j < this->m_cell_set_indexer[itype].getH(); j++)
                    {
                    unsigned cur_set = h_update_order.data[j];

                    if (have_depletants)
                        {
                        // save old positions
                        cudaMemcpyAsync(d_old_postype.data, d_postype.data, sizeof(Scalar4)*(this->m_pdata->getN()+this->m_pdata->getNGhosts()), cudaMemcpyDeviceToDevice, m_stream);
                        cudaMemcpyAsync(d_old_orientation.data, d_orientation.data, sizeof(Scalar4)*(this->m_pdata->getN()+this->m_pdata->getNGhosts()), cudaMemcpyDeviceToDevice, m_stream);
                        }

                    // accept particle moves from overlap checks
                    this->m_tuner_accept->begin();

                    param = this->m_tuner_accept->getParam();
                    block_size = param / 1000000;
                    group_size = param % 100;

                    args.cur_set = cur_set;
                    args.block_size = block_size;
                    args.group_size = group_size;

                    detail::gpu_hpmc_accept<Shape>(args, params.data());

                    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();

                    this->m_tuner_accept->end();

                    if (have_depletants)
                        {
                        if (this->m_prof) this->m_prof->push(this->m_exec_conf,"Depletants");

                        // overlap flags
                        ArrayHandle<unsigned int> d_overlap_cell(this->m_overlap_cell, access_location::device, access_mode::overwrite);

                            {
                            // insert depletants

                            // counters
                            ArrayHandle<hpmc_implicit_counters_t> d_implicit_count(this->m_implicit_count, access_location::device, access_mode::readwrite);

                            // Kernel driver arguments
                            unsigned int param = m_tuner_implicit->getParam();
                            unsigned int block_size = param / 1000000;
                            unsigned int block_size_overlaps = (param % 1000000) / 100;
                            unsigned int group_size = ((param % 100)-1) % (dev_prop.warpSize) + 1;
                            bool load_shared = ((param % 100)-1) / (dev_prop.warpSize);

                            m_tuner_implicit->begin();

                            // kernel parameters
                            auto args = detail::hpmc_implicit_args_new_t(d_postype.data,
                                    d_orientation.data,
                                    d_old_postype.data,
                                    d_old_orientation.data,
                                    d_cell_idx.data,
                                    d_cell_size.data,
                                    d_excell_idx.data,
                                    d_excell_size.data,
                                    this->m_cl_type[itype]->getCellIndexer(),
                                    this->m_cl_type[itype]->getCellListIndexer(),
                                    this->m_excell_list_indexer,
                                    this->m_cl_type[itype]->getDim(),
                                    ghost_width,
                                    &d_cell_sets.data[this->m_cell_set_indexer[itype](0,cur_set)],
                                    this->m_cell_set_indexer[itype].getW(),
                                    this->m_pdata->getN(),
                                    this->m_pdata->getNTypes(),
                                    this->m_seed + this->m_exec_conf->getRank()*this->m_nselect,
                                    d_overlaps.data,
                                    this->m_overlap_idx,
                                    timestep,
                                    this->m_sysdef->getNDimensions(),
                                    box,
                                    i+particles_per_cell*this->m_nselect*(3*j+1),
                                    block_size,
                                    1, //stride
                                    group_size,
                                    this->m_hasOrientation,
                                    this->m_pdata->getMaxN(),
                                    this->m_exec_conf->dev_prop,
                                    0,// curand_state_cell
                                    0,// curand_state_cell_new
                                    this->m_type,
                                    d_counters.data,
                                    d_implicit_count.data,
                                    0, // d_poisson_dist
                                    d_overlap_cell.data,
                                    d_active_cell_ptl_idx.data,
                                    d_active_cell_accept.data,
                                    d_active_cell_move_type_translate.data,
                                    0, // d_min
                                    0, // dmax
                                    first,
                                    this->getDepletantDensity(),
                                    m_stream,
                                    this->m_exec_conf->isCUDAErrorCheckingEnabled(),
                                    block_size_overlaps,
                                    load_shared);

                            detail::gpu_hpmc_insert_depletants_dp<Shape>(args, params.data());

                            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                                CHECK_CUDA_ERROR();

                            m_tuner_implicit->end();
                            }

                            {
                            // counters
                            ArrayHandle<hpmc_implicit_counters_t> d_implicit_count(this->m_implicit_count, access_location::device, access_mode::readwrite);

                            // apply depletant acceptance/rejection criterium
                            detail::gpu_hpmc_implicit_accept_reject_new<Shape>(
                                detail::hpmc_implicit_args_new_t(d_postype.data,
                                    d_orientation.data,
                                    d_old_postype.data,
                                    d_old_orientation.data,
                                    d_cell_idx.data,
                                    d_cell_size.data,
                                    d_excell_idx.data,
                                    d_excell_size.data,
                                    this->m_cl_type[itype]->getCellIndexer(),
                                    this->m_cl_type[itype]->getCellListIndexer(),
                                    this->m_excell_list_indexer,
                                    this->m_cl_type[itype]->getDim(),
                                    ghost_width,
                                    &d_cell_sets.data[this->m_cell_set_indexer[itype](0,cur_set)],
                                    this->m_cell_set_indexer[itype].getW(),
                                    this->m_pdata->getN(),
                                    this->m_pdata->getNTypes(),
                                    this->m_seed + this->m_exec_conf->getRank(),
                                    d_overlaps.data,
                                    this->m_overlap_idx,
                                    timestep,
                                    this->m_sysdef->getNDimensions(),
                                    box,
                                    i+particles_per_cell*this->m_nselect*(3*j+2),
                                    256, // block_size
                                    1, // stride
                                    1, //group_size
                                    this->m_hasOrientation,
                                    this->m_pdata->getMaxN(),
                                    this->m_exec_conf->dev_prop,
                                    0, // curand_state
                                    0, // curand_state_new
                                    this->m_type,
                                    d_counters.data,
                                    d_implicit_count.data,
                                    0, // poisson_dst
                                    d_overlap_cell.data,
                                    d_active_cell_ptl_idx.data,
                                    d_active_cell_accept.data,
                                    d_active_cell_move_type_translate.data,
                                    0, // d_min
                                    0, // d_max
                                    first,
                                    this->getDepletantDensity(),
                                    m_stream,
                                    this->m_exec_conf->isCUDAErrorCheckingEnabled(),
                                    block_size_overlaps,
                                    false,// load_shared
                                    d_trial_updated.data
                                    ),
                                params.data());

                            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                                CHECK_CUDA_ERROR();
                            }
                        if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
                        }

                    first = false;
                    } // end loop over cell sets
                } // end loop nselect*particles_per_cell
            } // end loop over types
        } // end compute capability >= 350

    // wait for kernels to catch up and release managed memory to host
    cudaDeviceSynchronize();

        {
        // shift particles
        Scalar3 shift = make_scalar3(0,0,0);
        shift.x = rng.s(-this->m_nominal_width/Scalar(2.0),this->m_nominal_width/Scalar(2.0));
        shift.y = rng.s(-this->m_nominal_width/Scalar(2.0),this->m_nominal_width/Scalar(2.0));
        if (this->m_sysdef->getNDimensions() == 3)
            {
            shift.z = rng.s(-this->m_nominal_width/Scalar(2.0),this->m_nominal_width/Scalar(2.0));
            }

        ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<int3> d_image(this->m_pdata->getImages(), access_location::device, access_mode::readwrite);

        detail::gpu_hpmc_shift(d_postype.data,
                               d_image.data,
                               this->m_pdata->getN(),
                               box,
                               shift,
                               128);

        // update the particle data origin
        this->m_pdata->translateOrigin(shift);

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

    this->communicate(true);

    // all particle have been moved, the aabb tree is now invalid
    this->m_aabb_tree_invalid = true;
    }

template<class Shape>
void IntegratorHPMCMonoImplicitNewGPU< Shape >::initializePoissonDistribution()
    {
    // resize GPUArray
    m_poisson_dist.resize(this->m_pdata->getNTypes());
    m_poisson_dist_created.resize(this->m_pdata->getNTypes(), false);

    ArrayHandle<curandDiscreteDistribution_t> h_poisson_dist(m_poisson_dist, access_location::host, access_mode::readwrite);
    for (unsigned int i_type = 0; i_type < this->m_pdata->getNTypes(); ++i_type)
        {
        // parameter for Poisson distribution
        Scalar lambda = this->m_lambda[i_type];

        if (lambda <= Scalar(0.0))
            {
            // guard against invalid parameters
            continue;
            }

        if (m_poisson_dist_created[i_type])
            {
            // release memory for old parameter
            this->m_exec_conf->msg->notice(6) << "Destroying Poisson distribution for type id " << i_type << std::endl;
            curandDestroyDistribution(h_poisson_dist.data[i_type]);

            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // create a Poisson distribution object
        this->m_exec_conf->msg->notice(6) << "Creating Poisson distribution for type id " << i_type << std::endl;
        curandCreatePoissonDistribution(lambda,&h_poisson_dist.data[i_type]);

        // keep track of state
        m_poisson_dist_created[i_type] = true;

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    }

template< class Shape >
void IntegratorHPMCMonoImplicitNewGPU< Shape >::initializeCellSets()
    {
    this->m_exec_conf->msg->notice(4) << "hpmc recomputing active cells" << std::endl;
    // "ghost cells" might contain active particles. So they must be included in the active cell sets
    // we should not run into a multiple issue since the base multiple is 2 and the ghost cells added are 2 in each
    // direction. Check just to be on the safe side

    m_cell_sets.resize(this->m_pdata->getNTypes());
    m_inverse_cell_set.resize(this->m_pdata->getNTypes());
    m_cell_set_indexer.resize(this->m_pdata->getNTypes());

    for (unsigned int itype = 0; itype < this->m_pdata->getNTypes(); ++itype)
        {
        // compute the number of cells in each set
        // every other cell is active along each direction, excluding ghost cells
        uint3 dim = m_cl_type[itype]->getDim();
        const Index3D& cell_indexer = m_cl_type[itype]->getCellIndexer();
        unsigned int n_active = dim.x / 2 * dim.y / 2;
        unsigned int n_sets = 4;

        if (this->m_sysdef->getNDimensions() == 3)
            {
            n_active *= dim.z / 2;
            n_sets = 8;
            }

        GPUArray< unsigned int > cell_sets(n_active, n_sets, this->m_exec_conf);
        m_cell_sets[itype].swap(cell_sets);

        GPUArray< unsigned int > inverse_cell_set(cell_indexer.getNumElements(), this->m_exec_conf);
        m_inverse_cell_set[itype].swap(inverse_cell_set);

        m_cell_set_indexer[itype] = Index2D(n_active, n_sets);

        // build a list of active cells
        ArrayHandle< unsigned int > h_cell_sets(m_cell_sets[itype], access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_inverse_cell_set(m_inverse_cell_set[itype], access_location::host, access_mode::overwrite);

        // offsets for x and y based on the set index
        unsigned int ox[] = {0, 1, 0, 1, 0, 1, 0, 1};
        unsigned int oy[] = {0, 0, 1, 1, 0, 0, 1, 1};
        unsigned int oz[] = {0, 0, 0, 0, 1, 1, 1, 1};

        // set inverse cell set to a defined value for ghost cells
        memset(h_inverse_cell_set.data, UINT_MAX, sizeof(unsigned int)*cell_indexer.getNumElements());

        for (unsigned int cur_set = 0; cur_set < n_sets; cur_set++)
            {
            unsigned int active_idx = 0;
            // loop over all cells in the active region, using information from num_ghost cells to avoid adding ghost cells
            // to the active set
            for (int k = oz[cur_set]; k < int(dim.z); k+=2)
                for (int j = oy[cur_set]; j < int(dim.y); j+=2)
                    for (int i = ox[cur_set]; i < int(dim.x); i+=2)
                        {
                        h_cell_sets.data[m_cell_set_indexer[itype](active_idx, cur_set)] = cell_indexer(i,j,k);
                        h_inverse_cell_set.data[cell_indexer(i,j,k)] = cur_set;
                        active_idx++;
                        }
            }
        } // end loop over types
    }

template< class Shape >
void IntegratorHPMCMonoImplicitNewGPU< Shape >::initializeExcellMem()
    {
    this->m_exec_conf->msg->notice(4) << "hpmc resizing expanded cells" << std::endl;

    // get the current cell dimensions
    unsigned int num_cells = this->m_cl->getCellIndexer().getNumElements();
    unsigned int num_adj = this->m_cl->getCellAdjIndexer().getW();
    unsigned int num_max = this->m_cl->getNmax();

    // make the excell dimensions the same, but with room for Nmax*Nadj in each cell
    m_excell_list_indexer = Index2D(num_max * num_adj, num_cells);

    // reallocate memory
    m_excell_idx.resize(m_excell_list_indexer.getNumElements());
    m_excell_cell_set.resize(m_excell_list_indexer.getNumElements());
    m_excell_overlap.resize(m_excell_list_indexer.getNumElements());
    m_excell_size.resize(num_cells);
    }

template< class Shape >
void IntegratorHPMCMonoImplicitNewGPU< Shape >::initializeQueueMem()
    {
    this->m_exec_conf->msg->notice(4) << "hpmc resizing queue" << std::endl;

    unsigned int num_adj = this->m_cl->getCellAdjIndexer().getW();
    unsigned int num_max = this->m_cl->getNmax();

    // the number of active cells is an upper bound for the number of queues
    m_queue_indexer = Index2D(this->m_cell_set_indexer[0].getNumElements(), num_max*num_adj);

    if (Shape::isParallel())
        {
        m_queue_active_cell_idx.resize(m_queue_indexer.getNumElements());
        m_queue_postype.resize(m_queue_indexer.getNumElements());
        m_queue_orientation.resize(m_queue_indexer.getNumElements());
        m_queue_excell_idx.resize(m_queue_indexer.getNumElements());
        }

    m_cell_overlaps.resize(this->m_cell_set_indexer[0].getW());
    }

template< class Shape >
void IntegratorHPMCMonoImplicitNewGPU< Shape >::updateCellWidth()
    {
    IntegratorHPMCMonoImplicitNew<Shape>::updateCellWidth();

    this->m_cl->setNominalWidth(this->m_nominal_width);

    for (unsigned int itype = 0; itype < this->m_pdata->getNTypes(); ++itype)
        {
        m_cl_type[itype]->setNominalWidth(this->m_nominal_width);
        }

    // attach the parameters to the kernel stream so that they are visible
    // when other kernels are called

    cudaStreamAttachMemAsync(m_stream, this->m_params.data(), 0, cudaMemAttachSingle);
    CHECK_CUDA_ERROR();

    #if (CUDART_VERSION >= 8000)
    cudaMemAdvise(this->m_params.data(), this->m_params.size()*sizeof(typename Shape::param_type), cudaMemAdviseSetReadMostly, 0);
    CHECK_CUDA_ERROR();
    #endif

    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        {
        // attach nested memory regions
        this->m_params[i].attach_to_stream(m_stream);
        }
    }


//! Export this hpmc integrator to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorHPMCMono<Shape> will be exported
*/
template < class Shape > void export_IntegratorHPMCMonoImplicitNewGPU(pybind11::module& m, const std::string& name)
    {
     pybind11::class_<IntegratorHPMCMonoImplicitNewGPU<Shape>, std::shared_ptr< IntegratorHPMCMonoImplicitNewGPU<Shape> > >(m, name.c_str(), pybind11::base< IntegratorHPMCMonoImplicitNew<Shape> >())
              .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<CellList>, unsigned int >())
        ;
    }

} // end namespace hpmc

#endif // ENABLE_CUDA

#endif // __HPMC_MONO_IMPLICIT_NEW_GPU_H__
