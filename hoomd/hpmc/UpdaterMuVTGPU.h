#ifndef __UPDATER_MUVT_GPU_H__
#define __UPDATER_MUVT_GPU_H__

#ifdef ENABLE_CUDA

#include "UpdaterMuVT.h"
#include "hoomd/CellListGPU.h"

#include "UpdaterMuVTGPU.cuh"

#include <cuda_runtime.h>

namespace hpmc
{

/*!
 * Specialization of UpdaterMuVT for parallel insertion of particles on the GPU (Gibbs sampler)
 */
template<class Shape>
class UpdaterMuVTGPU : public UpdaterMuVT<Shape>
    {
    public:
        //! Constructor
        UpdaterMuVTGPU(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
            unsigned int seed,
            unsigned int npartition);
        virtual ~UpdaterMuVTGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            // call base class method first
            m_tuner_insert->setPeriod(period);
            m_tuner_insert->setEnabled(enable);

            m_tuner_set_properties->setPeriod(period);
            m_tuner_set_properties->setEnabled(enable);

            m_tuner_excell_block_size->setPeriod(period);
            m_tuner_excell_block_size->setEnabled(enable);
            }

    protected:
        //! Generate a random configuration for a Gibbs sampler
        virtual void generateGibbsSamplerConfiguration(unsigned int timestep);

        void initializeExcellMem();

        CellListGPU m_cl;                     //!< The cell list

        uint3 m_last_dim;                     //!< Dimensions of the cell list on the last call to update
        unsigned int m_last_nmax;             //!< Last cell list NMax value allocated in excell

        GPUArray<unsigned int> m_excell_idx;  //!< Particle indices in expanded cells
        GPUArray<unsigned int> m_excell_size; //!< Number of particles in each expanded cell
        Index2D m_excell_list_indexer;        //!< Indexer to access elements of the excell_idx list

        cudaStream_t m_stream;                //!< CUDA stream for kernel execution

        std::unique_ptr<Autotuner> m_tuner_insert;     //!< Autotuner for inserting particles
        std::unique_ptr<Autotuner> m_tuner_set_properties; //!< Autotuner for setting particle properties
        std::unique_ptr<Autotuner> m_tuner_excell_block_size;  //!< Autotuner for excell block_size

        GPUVector<unsigned int> m_ptl_overlap;    //!< Overlap flag per inserted particle
        GPUVector<Scalar4> m_postype_insert;  //!< Positions and types of particles inserted
        GPUVector<Scalar4> m_orientation_insert;  //!< Orientations of particles inserted

        GPUVector<unsigned int> m_tags;         //!< Tags of inserted particles
    };

template< class Shape >
UpdaterMuVTGPU<Shape>::UpdaterMuVTGPU(std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
    unsigned int seed,
    unsigned int npartition)
    : UpdaterMuVT<Shape>(sysdef, mc, seed, npartition), m_cl(sysdef)
    {
    GPUVector<unsigned int>(this->m_exec_conf).swap(m_ptl_overlap);
    GPUVector<Scalar4>(this->m_exec_conf).swap(m_postype_insert);
    GPUVector<Scalar4>(this->m_exec_conf).swap(m_orientation_insert);
    GPUVector<unsigned int>(this->m_exec_conf).swap(m_tags);

    this->m_cl.setRadius(1);
    this->m_cl.setComputeTDB(false);
    this->m_cl.setFlagType();
    this->m_cl.setComputeIdx(true);

    // initialize the autotuners
    // the full block size, stride and group size matrix is searched,
    // encoded as block_size*1000000 + stride*100 + group_size.
    std::vector<unsigned int> valid_params;
    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
        {
        unsigned int s=1;
        while (s <= (unsigned int) this->m_exec_conf->dev_prop.warpSize)
            {
            if ((block_size % s) == 0)
                valid_params.push_back(block_size*1000000 + s);
            s = s * 2;
            }
        }
    m_tuner_insert.reset(new Autotuner(valid_params, 5, 1000000, "hpmc_muvt_insert", this->m_exec_conf));

    GPUArray<unsigned int> excell_size(0, this->m_exec_conf);
    m_excell_size.swap(excell_size);

    GPUArray<unsigned int> excell_idx(0, this->m_exec_conf);
    m_excell_idx.swap(excell_idx);

    // set last dim to a bogus value so that it will re-init on the first call
    m_last_dim = make_uint3(0xffffffff, 0xffffffff, 0xffffffff);
    m_last_nmax = 0xffffffff;

    m_tuner_excell_block_size.reset(new Autotuner(32,1024,32, 5, 1000000, "hpmc_muvt_excell_block_size", this->m_exec_conf));
    m_tuner_set_properties.reset(new Autotuner(32,1024,32, 5, 1000000, "hpmc_muvt_set_properties", this->m_exec_conf));

    // create a cuda stream to ensure managed memory coherency
    cudaStreamCreate(&m_stream);
    CHECK_CUDA_ERROR();
    }

template<class Shape>
UpdaterMuVTGPU<Shape>::~UpdaterMuVTGPU()
    {
    cudaStreamDestroy(m_stream);
    CHECK_CUDA_ERROR();
    }

//! Generate a random configuration for a Gibbs sampler
template<class Shape>
void UpdaterMuVTGPU<Shape>::generateGibbsSamplerConfiguration(unsigned int timestep)
    {
    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "Gibbs sampler");

    // set nominal width
    Scalar nominal_width = this->m_mc->getMaxCoreDiameter();

    if (this->m_cl.getNominalWidth() != nominal_width)
        this->m_cl.setNominalWidth(nominal_width);

    const BoxDim &box = this->m_pdata->getBox();
    Scalar3 npd = box.getNearestPlaneDistance();

    if ((box.getPeriodic().x && npd.x <= nominal_width*2) ||
        (box.getPeriodic().y && npd.y <= nominal_width*2) ||
        (this->m_sysdef->getNDimensions() == 3 && box.getPeriodic().z && npd.z <= nominal_width*2))
        {
        this->m_exec_conf->msg->error() << "Simulation box too small for update.muvt() on GPU - increase it so the minimum image convention works" << endl;
        throw runtime_error("Error performing HPMC update");
        }

    // compute cell list
    this->m_cl.compute(timestep);

    // if the cell list is a different size than last time, reinitialize expanded cell list
    uint3 cur_dim = this->m_cl.getDim();
    if (this->m_last_dim.x != cur_dim.x || this->m_last_dim.y != cur_dim.y || this->m_last_dim.z != cur_dim.z ||
        this->m_last_nmax != this->m_cl.getNmax())
        {
        this->initializeExcellMem();
        m_last_dim = cur_dim;
        m_last_nmax = this->m_cl.getNmax();
        }

    // access the cell list data
    ArrayHandle<unsigned int> d_cell_size(this->m_cl.getCellSizeArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_cell_idx(this->m_cl.getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_cell_adj(this->m_cl.getCellAdjArray(), access_location::device, access_mode::read);
    ArrayHandle< unsigned int > d_excell_idx(this->m_excell_idx, access_location::device, access_mode::readwrite);
    ArrayHandle< unsigned int > d_excell_size(this->m_excell_size, access_location::device, access_mode::readwrite);

    // update the expanded cells
    this->m_tuner_excell_block_size->begin();
    detail::gpu_hpmc_excell(d_excell_idx.data,
                            d_excell_size.data,
                            this->m_excell_list_indexer,
                            d_cell_idx.data,
                            d_cell_size.data,
                            d_cell_adj.data,
                            this->m_cl.getCellIndexer(),
                            this->m_cl.getCellListIndexer(),
                            this->m_cl.getCellAdjIndexer(),
                            this->m_tuner_excell_block_size->getParam());
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    this->m_tuner_excell_block_size->end();


    // perform parallel insertion/removal
    for (auto it_type = this->m_parallel_types.begin(); it_type != this->m_parallel_types.end(); it_type++)
        {
        unsigned int type = *it_type;

        // existing particles to be removed
        std::vector<unsigned int> remove_tags = this->m_type_map[type];

        #ifdef ENABLE_MPI
        if (m_comm)
            m_mc->communicate(false);
        #endif

        // combine four seeds
        std::vector<unsigned int> seed_seq(4);
        seed_seq[0] = this->m_seed;
        seed_seq[1] = timestep;
        seed_seq[2] = this->m_exec_conf->getRank();
        seed_seq[3] = 0x374df9a2;
        std::seed_seq seed(seed_seq.begin(), seed_seq.end());

        // RNG for poisson distribution
        std::mt19937 rng_poisson(seed);

        // local box volume
        Scalar V_box = this->m_pdata->getBox().getVolume(this->m_sysdef->getNDimensions()==2);

        // draw a poisson-random number
        Scalar fugacity = this->m_fugacity[type]->getValue(timestep);
        std::poisson_distribution<unsigned int> poisson(fugacity*V_box);

        // generate particles locally
        unsigned int n_insert = poisson(rng_poisson);

        // number of particles actually inserted
        unsigned int n_ptls_inserted = 0;

        // resize scratch space
        m_ptl_overlap.resize(n_insert);
        m_postype_insert.resize(n_insert);
        m_orientation_insert.resize(n_insert);

        // access the particle data
        ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);

        // interaction matrix
        ArrayHandle<unsigned int> d_overlaps(this->m_mc->getInteractionMatrix(), access_location::device, access_mode::read);

        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        // access the parameters
        auto& params = this->m_mc->getParams();

            {
            // access the temporary storage for inserted particles and overlap flags
            ArrayHandle<Scalar4> d_postype_insert(m_postype_insert,access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_orientation_insert(m_orientation_insert, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ptl_overlap(m_ptl_overlap,access_location::device, access_mode::overwrite);

            m_tuner_insert->begin();
            unsigned int param= m_tuner_insert->getParam();
            unsigned int block_size = param / 1000000;
            unsigned int group_size = param % 100;

            detail::hpmc_muvt_args_t muvt_args(n_insert,
                                               type,
                                               d_postype.data,
                                               d_orientation.data,
                                               d_cell_idx.data,
                                               d_cell_size.data,
                                               this->m_cl.getCellIndexer(),
                                               this->m_cl.getCellListIndexer(),
                                               d_excell_idx.data,
                                               d_excell_size.data,
                                               this->m_excell_list_indexer,
                                               this->m_cl.getDim(),
                                               this->m_pdata->getN(),
                                               this->m_pdata->getNTypes(),
                                               this->m_seed+this->m_exec_conf->getRank(),
                                               type,
                                               timestep,
                                               this->m_sysdef->getNDimensions(),
                                               box,
                                               block_size,
                                               1, //stride
                                               group_size,
                                               this->m_pdata->getMaxN(),
                                               this->m_cl.getGhostWidth(),
                                               d_overlaps.data,
                                               overlap_idx,
                                               d_ptl_overlap.data,
                                               d_postype_insert.data,
                                               d_orientation_insert.data,
                                               n_ptls_inserted,
                                               m_stream,
                                               this->m_exec_conf->dev_prop,
                                               this->m_exec_conf->getCachedAllocator());


            // invoke kernel for counting total overlap volume
            detail::gpu_hpmc_muvt<Shape>(muvt_args, params.data());

            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            m_tuner_insert->end();
            }

        if (this->m_prof)
            this->m_prof->push(this->m_exec_conf, "remove particles");

        // remove old particles first *after* checking overlaps (Gibbs sampler)
        this->m_exec_conf->msg->notice(7) << "UpdaterMuVTGPU " << timestep << " removing " << remove_tags.size()
             << " ptls of type " << this->m_pdata->getNameByType(type) << std::endl;

        // remove all particles of the given types
        this->m_pdata->removeParticlesGlobal(remove_tags);

        if (this->m_prof)
            this->m_prof->pop(this->m_exec_conf);

        this->m_exec_conf->msg->notice(7) << "UpdaterMuVTGPU " << timestep << " inserting " << n_ptls_inserted
             << " ptls of type " << this->m_pdata->getNameByType(type) << std::endl;

        if (this->m_prof)
            this->m_prof->push(this->m_exec_conf,"add particles");

        // bulk-insert the particles
        auto inserted_tags = this->m_pdata->addParticlesGlobal(n_ptls_inserted);

        assert(inserted_tags.size() == n_ptls_inserted);

        if (this->m_prof)
            this->m_prof->pop(this->m_exec_conf);

        m_tags.resize(inserted_tags.size());

            {
            // copy tags over to GPU array
            ArrayHandle<unsigned int> h_tags(m_tags, access_location::host, access_mode::overwrite);
            std::copy(inserted_tags.begin(), inserted_tags.end(), h_tags.data);
            }

            {
            // set the particle properties
            ArrayHandle<unsigned int> d_rtag(this->m_pdata->getRTags(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_postype_insert(m_postype_insert,access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_orientation_insert(m_orientation_insert, access_location::device, access_mode::read);

            ArrayHandle<unsigned int> d_tags(m_tags, access_location::device, access_mode::read);

            m_tuner_set_properties->begin();
            unsigned int block_size = m_tuner_set_properties->getParam();

            detail::gpu_muvt_set_particle_properties(
                n_ptls_inserted,
                d_rtag.data,
                d_tags.data,
                d_postype.data,
                d_orientation.data,
                d_postype_insert.data,
                d_orientation_insert.data,
                block_size
                );

            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            m_tuner_set_properties->end();
            }

        // types have changed
        this->m_pdata->notifyParticleSort();
        } // end loop over types that can be inserted in parallel

    if (this->m_prof)
        this->m_prof->pop();
    }

template< class Shape >
void UpdaterMuVTGPU< Shape >::initializeExcellMem()
    {
    this->m_exec_conf->msg->notice(4) << "hpmc resizing expanded cells" << std::endl;

    // get the current cell dimensions
    unsigned int num_cells = this->m_cl.getCellIndexer().getNumElements();
    unsigned int num_adj = this->m_cl.getCellAdjIndexer().getW();
    unsigned int num_max = this->m_cl.getNmax();

    // make the excell dimensions the same, but with room for Nmax*Nadj in each cell
    m_excell_list_indexer = Index2D(num_max * num_adj, num_cells);

    // reallocate memory
    m_excell_idx.resize(m_excell_list_indexer.getNumElements());
    m_excell_size.resize(num_cells);
    }


//! Export the UpdaterMuVT class to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of UpdaterMuVT<Shape> will be exported
*/
template < class Shape > void export_UpdaterMuVTGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterMuVTGPU<Shape>, std::shared_ptr< UpdaterMuVTGPU<Shape> > >(m, name.c_str(), pybind11::base<UpdaterMuVT<Shape> >())
          .def( pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr< IntegratorHPMCMono<Shape> >, unsigned int, unsigned int>())
          ;
    }


} // end namespace

#endif // ENABLE_CUDA
#endif
