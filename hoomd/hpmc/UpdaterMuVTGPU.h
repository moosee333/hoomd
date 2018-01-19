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
            }

    protected:
        //! Generate a random configuration for a Gibbs sampler
        virtual void generateGibbsSamplerConfiguration(unsigned int timestep);

        cudaStream_t m_stream;                //!< CUDA stream for kernel execution

        std::unique_ptr<Autotuner> m_tuner_insert;     //!< Autotuner for inserting particles
        std::unique_ptr<Autotuner> m_tuner_set_properties; //!< Autotuner for setting particle properties

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
    : UpdaterMuVT<Shape>(sysdef, mc, seed, npartition)
    {
    GPUVector<unsigned int>(this->m_exec_conf).swap(m_ptl_overlap);
    GPUVector<Scalar4>(this->m_exec_conf).swap(m_postype_insert);
    GPUVector<Scalar4>(this->m_exec_conf).swap(m_orientation_insert);
    GPUVector<unsigned int>(this->m_exec_conf).swap(m_tags);

    m_tuner_insert.reset(new Autotuner(32,1024,32, 5, 1000000, "hpmc_muvt_insert", this->m_exec_conf));
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

    const BoxDim &box = this->m_pdata->getBox();

    // perform parallel insertion/removal
    for (auto it_type = this->m_parallel_types.begin(); it_type != this->m_parallel_types.end(); it_type++)
        {
        unsigned int type = *it_type;

        // existing particles to be removed
        const std::vector<unsigned int>& remove_tags = this->m_type_map[type];

        if (this->m_prof)
            this->m_prof->push(this->m_exec_conf, "remove particles");

        // remove old particles first, so we save same time on the following AABB tree traversal
        this->m_exec_conf->msg->notice(7) << "UpdaterMuVTGPU " << timestep << " removing " << remove_tags.size()
             << " ptls of type " << this->m_pdata->getNameByType(type) << std::endl;

        // remove all particles of the given types
        this->m_pdata->removeParticlesGlobal(remove_tags);

        if (this->m_prof)
            this->m_prof->pop(this->m_exec_conf);

        #ifdef ENABLE_MPI
        if (this->m_comm)
            this->m_mc->communicate(false);
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
        Scalar V_box = box.getVolume(this->m_sysdef->getNDimensions()==2);

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

        auto& image_list = this->m_mc->updateImageList();
        auto& aabb_tree = this->m_mc->buildAABBTree();

            {
            // access the temporary storage for inserted particles and overlap flags
            ArrayHandle<Scalar4> d_postype_insert(m_postype_insert,access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_orientation_insert(m_orientation_insert, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ptl_overlap(m_ptl_overlap,access_location::device, access_mode::overwrite);

            m_tuner_insert->begin();
            unsigned int param= m_tuner_insert->getParam();
            unsigned int block_size = param;

            detail::hpmc_muvt_args_t muvt_args(n_insert,
                                               type,
                                               d_postype.data,
                                               d_orientation.data,
                                               aabb_tree,
                                               image_list,
                                               this->m_pdata->getN(),
                                               this->m_pdata->getNTypes(),
                                               this->m_seed+this->m_exec_conf->getRank(),
                                               type,
                                               timestep,
                                               this->m_sysdef->getNDimensions(),
                                               box,
                                               block_size,
                                               1, //stride
                                               this->m_pdata->getMaxN(),
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
