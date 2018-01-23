// inclusion guard
#ifndef _UPDATER_HPMC_CLUSTERS_GPU_
#define _UPDATER_HPMC_CLUSTERS_GPU_

/*! \file UpdaterBoxClusters.h
    \brief Declaration of UpdaterBoxClusters
*/

// nvgraph not stable yet
#undef NVGRAPH_AVAILABLE

#ifdef ENABLE_CUDA

#include "UpdaterClusters.h"
#include "UpdaterClustersGPU.cuh"
#include "hoomd/AABBTree.h"

#include <cuda_runtime.h>

namespace hpmc
{

/*!
   Implementation of UpdaterClusters on the GPU
*/

template< class Shape >
class UpdaterClustersGPU : public UpdaterClusters<Shape>
    {
    public:
        //! Constructor
        /*! \param sysdef System definition
            \param mc HPMC integrator
            \param seed PRNG seed
        */
        UpdaterClustersGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
                        unsigned int seed);

        //! Destructor
        virtual ~UpdaterClustersGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tuner_old_new_collisions->setPeriod(period);
            m_tuner_old_new_collisions->setEnabled(enable);

            m_tuner_old_new_overlaps->setPeriod(period);
            m_tuner_old_new_overlaps->setEnabled(enable);

            m_tuner_new_new_collisions->setPeriod(period);
            m_tuner_new_new_collisions->setEnabled(enable);

            m_tuner_new_new_overlaps->setPeriod(period);
            m_tuner_new_new_overlaps->setEnabled(enable);
            }


    protected:
        GPUVector<unsigned int> m_new_tag;   //!< Lookup of snapshot order based on old particle index
        GPUFlags<uint2> m_conditions;        //!< Flags returned from GPU kernel

        GPUArray<unsigned int> m_n_overlaps; //!< Length of list of overlapping pairs
        GPUVector<uint3> m_collisions;       //!< List of collisions between circumspheres
        GPUVector<uint2> m_overlaps;         //!< List of overlaps between old and new configuration
        GPUArray<unsigned int> m_n_reject;   //!< Length of list of rejected particle moves
        GPUVector<unsigned int> m_reject;    //!< List of rejected particle moves

        GPUArray<unsigned int> m_n_interact_new_new; //!< Length of list of particles interacting new-new
        unsigned int m_n_overlaps_old_new;     //!< Number of overlaps between old and new configuration
        unsigned int m_n_overlaps_new_new;     //!< Number of overlaps between new and new configuration

        #ifdef NVGRAPH_AVAILABLE
        GPUVector<unsigned int> m_components;  //!< The connected component labels per particle
        #endif

        std::unique_ptr<Autotuner> m_tuner_old_new_collisions;     //!< Autotuner for checking new against new (broad phase)
        std::unique_ptr<Autotuner> m_tuner_old_new_overlaps;       //!< Autotuner for checking new against new (narrow phase)
        std::unique_ptr<Autotuner> m_tuner_new_new_collisions;     //!< Autotuner for checking new against new (broad phase)
        std::unique_ptr<Autotuner> m_tuner_new_new_overlaps;       //!< Autotuner for checking new against new (narrow phase)

        cudaStream_t m_stream;                //!< CUDA stream for kernel execution

        //! Helper function to initialize expanded cell list
        void initializeExcellMem();

        //! Find interactions between particles due to overlap and depletion interaction
        /*! \param timestep Current time step
            \param pivot The current pivot point
            \param q The current line reflection axis
            \param line True if this is a line reflection
            \param map Map to lookup new tag from old tag
        */
        virtual void findInteractions(unsigned int timestep, vec3<Scalar> pivot, quat<Scalar> q, bool swap,
            bool line, const std::map<unsigned int, unsigned int>& map);

        //! Determine connected components of the interaction graph
        #ifdef ENABLE_TBB
        virtual void findConnectedComponents(unsigned int timestep, unsigned int N, bool line, bool swap, std::vector<tbb::concurrent_vector<unsigned int> >& clusters);
        #else
        virtual void findConnectedComponents(unsigned int timestep, unsigned int N, bool line, bool swap, std::vector<std::vector<unsigned int> >& clusters);
        #endif
    };

template< class Shape >
UpdaterClustersGPU<Shape>::UpdaterClustersGPU(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
                                 unsigned int seed)
        : UpdaterClusters<Shape>(sysdef, mc, seed), m_conditions(this->m_exec_conf)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing UpdaterClustersGPU" << std::endl;

    // allocate memory
    GPUVector<unsigned int>(this->m_exec_conf).swap(m_new_tag);

    GPUVector<uint3>(this->m_exec_conf).swap(m_collisions);
    GPUArray<unsigned int>(1, this->m_exec_conf).swap(m_n_overlaps);
    GPUVector<uint2>(this->m_exec_conf).swap(m_overlaps);
    GPUArray<unsigned int>(1, this->m_exec_conf).swap(m_n_interact_new_new);
    GPUArray<unsigned int>(1, this->m_exec_conf).swap(m_n_reject);
    GPUVector<unsigned int>(this->m_exec_conf).swap(m_reject);

    #ifdef NVGRAPH_AVAILABLE
    GPUVector<unsigned int>(this->m_exec_conf).swap(m_components);
    #endif

    m_n_overlaps_old_new = 0;
    m_n_overlaps_new_new = 0;

    // initialize the autotuners
    cudaDeviceProp dev_prop = this->m_exec_conf->dev_prop;

    // parameters for broad phase kernel
    std::vector<unsigned int> valid_params;

    for (unsigned int block_size = dev_prop.warpSize; block_size <= (unsigned int) dev_prop.maxThreadsPerBlock; block_size += dev_prop.warpSize)
        {
        for (unsigned int group_size=1; group_size <= detail::NODE_CAPACITY; group_size++)
            {
            if ((block_size % group_size) == 0)
                valid_params.push_back(block_size*1000000 +  group_size);
            }
        }

    m_tuner_old_new_collisions.reset(new Autotuner(valid_params, 5, 1000000, "hpmc_clusters_old_broad_phase", this->m_exec_conf));
    m_tuner_old_new_overlaps.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 1000000, "hpmc_clusters_old_narrow_phase", this->m_exec_conf));

    m_tuner_new_new_collisions.reset(new Autotuner(valid_params, 5, 1000000, "hpmc_clusters_new_broad_phase", this->m_exec_conf));
    m_tuner_new_new_overlaps.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 1000000, "hpmc_clusters_new_narrow_phase", this->m_exec_conf));

    // create a cuda stream to ensure managed memory coherency
    cudaStreamCreate(&m_stream);
    CHECK_CUDA_ERROR();
    }

template< class Shape >
UpdaterClustersGPU<Shape>::~UpdaterClustersGPU()
    {
    cudaStreamDestroy(m_stream);
    CHECK_CUDA_ERROR();

    this->m_exec_conf->msg->notice(5) << "Destroying UpdaterClustersGPU" << std::endl;
    }

template< class Shape >
void UpdaterClustersGPU<Shape>::findInteractions(unsigned int timestep, vec3<Scalar> pivot, quat<Scalar> q, bool swap,
    bool line, const std::map<unsigned int, unsigned int>& map)
    {
    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf,"Interactions");

    if (this->m_mc->getPatchInteraction())
        throw std::runtime_error("Patch interactions not supported with update.clusters() on GPU.");

    const BoxDim &box = this->m_pdata->getBox();

    m_new_tag.resize(this->m_n_particles_old);
    assert(m_n_particles_old == map.size());
        {
        // fill in snapshot order
        ArrayHandle<unsigned int> h_new_tag(m_new_tag, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tag_backup(this->m_tag_backup, access_location::host, access_mode::read);
        unsigned int nptl = this->m_n_particles_old;
        for (unsigned int i = 0; i < nptl; ++i)
            {
            auto it = map.find(h_tag_backup.data[i]);
            assert(it != map.end());
            unsigned int new_tag = it->second;
            h_new_tag.data[i] = new_tag;
            }
        }

    auto &params = this->m_mc->getParams();

    auto &aabb_tree = this->m_mc->buildAABBTree();
    auto &image_list = this->m_mc->updateImageList();
    auto &image_hkl = this->m_mc->getImageHKL();

    // old new
    bool reallocate = true;
        {
        // access new particle data
        ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::read);
        ArrayHandle<int3> d_image(this->m_pdata->getImages(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(), access_location::device, access_mode::read);

        // access backed up particle data
        ArrayHandle<Scalar4> d_postype_backup(this->m_postype_backup, access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_orientation_backup(this->m_orientation_backup, access_location::device, access_mode::read);
        ArrayHandle<int3> d_image_backup(this->m_image_backup, access_location::device, access_mode::read);

        ArrayHandle<unsigned int> d_new_tag(m_new_tag, access_location::device, access_mode::read);

        // interaction matrix
        ArrayHandle<unsigned int> d_check_overlaps(this->m_mc->getInteractionMatrix(), access_location::device, access_mode::read);
        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        detail::hpmc_clusters_args_t clusters_args(this->m_n_particles_old,
                                           0, // ncollisions
                                           d_postype.data,
                                           d_orientation.data,
                                           d_image.data,
                                           d_tag.data,
                                           this->m_pdata->getNTypes(),
                                           timestep,
                                           this->m_sysdef->getNDimensions(),
                                           box,
                                           0, //block_size
                                           1, //stride
                                           0, // group_size
                                           this->m_pdata->getMaxN(),
                                           d_check_overlaps.data,
                                           overlap_idx,
                                           line,
                                           d_postype_backup.data,
                                           d_orientation_backup.data,
                                           d_image_backup.data,
                                           d_new_tag.data,
                                           0, //d_overlaps.data,
                                           0, //d_collisions.data,
                                           0, //m_overlaps.getNumElements(),
                                           0, //d_n_overlaps.data,
                                           0, //d_reject.data,
                                           0, //m_reject.getNumElements(),
                                           0, //d_n_reject.data,
                                           0, //m_conditions.getDeviceFlags(),
                                           swap,
                                           swap ? this->m_ab_types[0] : 0,
                                           swap ? this->m_ab_types[1] : 0,
                                           aabb_tree,
                                           image_list,
                                           image_hkl,
                                           m_stream,
                                           this->m_exec_conf->dev_prop);

        do
            {
            // reset flags
            m_conditions.resetFlags(make_uint2(0,0));

                {
                ArrayHandle<uint3> d_collisions(m_collisions, access_location::device, access_mode::overwrite);
                ArrayHandle<unsigned int> d_n_overlaps(m_n_overlaps, access_location::device, access_mode::overwrite);

                m_tuner_old_new_collisions->begin();

                unsigned int param = m_tuner_old_new_collisions->getParam();
                clusters_args.block_size = param / 1000000;
                clusters_args.group_size = param % 1000000;

                clusters_args.d_collisions = d_collisions.data;
                clusters_args.max_n_overlaps = m_overlaps.getNumElements();
                clusters_args.d_n_overlaps = d_n_overlaps.data;
                clusters_args.d_conditions = m_conditions.getDeviceFlags();

                // invoke kernel for checking collisions between circumspheres
                detail::gpu_hpmc_clusters<Shape>(clusters_args, params.data());

                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();

                m_tuner_old_new_collisions->end();
                }

            uint2 flags = m_conditions.readFlags();

            // resize to largest element + 1
            if (flags.x)
                {
                m_overlaps.resize(flags.x);
                m_collisions.resize(flags.x);
                }

            reallocate = flags.x;
            } while(reallocate);

            {
            // get size of collision list
            ArrayHandle<unsigned int> h_n_overlaps(m_n_overlaps, access_location::host, access_mode::read);
            clusters_args.ncollisions = *h_n_overlaps.data;
            }

        do
            {
            // reset flags
            m_conditions.resetFlags(make_uint2(0,0));

                {
                ArrayHandle<uint3> d_collisions(m_collisions, access_location::device, access_mode::read);
                ArrayHandle<uint2> d_overlaps(m_overlaps, access_location::device, access_mode::overwrite);
                ArrayHandle<unsigned int> d_n_overlaps(m_n_overlaps, access_location::device, access_mode::overwrite);
                ArrayHandle<unsigned int> d_reject(m_reject, access_location::device, access_mode::overwrite);
                ArrayHandle<unsigned int> d_n_reject(m_n_reject, access_location::device, access_mode::overwrite);

                m_tuner_old_new_overlaps->begin();

                unsigned int param = m_tuner_old_new_collisions->getParam();
                clusters_args.block_size = param;

                clusters_args.d_overlaps = d_overlaps.data;
                clusters_args.d_collisions = d_collisions.data;
                clusters_args.d_n_overlaps = d_n_overlaps.data;
                clusters_args.d_conditions = m_conditions.getDeviceFlags();
                clusters_args.d_reject = d_reject.data;
                clusters_args.max_n_reject = m_reject.getNumElements();
                clusters_args.d_n_reject = d_n_reject.data;

                // invoke kernel for checking actual overlaps (narrow phase)
                detail::gpu_hpmc_clusters_overlaps<Shape>(clusters_args, params.data());

                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();

                m_tuner_old_new_overlaps->end();
                }

            uint2 flags = m_conditions.readFlags();

            // resize to largest element + 1
            if (flags.y)
                m_reject.resize(flags.y);

            reallocate = flags.y;
            } while(reallocate);

        // resize array to reflect actual size of data
        ArrayHandle<unsigned int> h_n_reject(m_n_reject, access_location::host, access_mode::read);
        m_reject.resize(*h_n_reject.data);

        // extract number of entries in the adjacency matrix
        ArrayHandle<unsigned int> h_n_overlaps(m_n_overlaps, access_location::host, access_mode::read);
        m_n_overlaps_old_new = *h_n_overlaps.data;
        } // end ArrayHandle scope

    m_n_overlaps_new_new = 0;

    if (line && !swap)
        {
        // with line transformations, check new configuration against itself to detect interactions across PBC
        // append to existing list of overlapping pairs

        // access new particle data
        ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::read);
        ArrayHandle<int3> d_image(this->m_pdata->getImages(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(), access_location::device, access_mode::read);

        // interaction matrix
        ArrayHandle<unsigned int> d_check_overlaps(this->m_mc->getInteractionMatrix(), access_location::device, access_mode::read);
        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        detail::hpmc_clusters_args_t clusters_args(this->m_pdata->getN(),
                                           0, // ncollisions
                                           d_postype.data,
                                           d_orientation.data,
                                           d_image.data,
                                           d_tag.data,
                                           this->m_pdata->getNTypes(),
                                           timestep,
                                           this->m_sysdef->getNDimensions(),
                                           box,
                                           0, // block_size
                                           1, //stride
                                           0, // group_size
                                           this->m_pdata->getMaxN(),
                                           d_check_overlaps.data,
                                           overlap_idx,
                                           line,
                                           d_postype.data,
                                           d_orientation.data,
                                           d_image.data,
                                           d_tag.data,
                                           0, // d_overlaps
                                           0, // d_collisions
                                           0, // max_n_overlaps
                                           0, // d_n_overlaps
                                           0, // d_reject
                                           0, // max_n_reject
                                           0, // d_n_reject
                                           0, // d_conditions
                                           swap,
                                           swap ? this->m_ab_types[0] : 0,
                                           swap ? this->m_ab_types[1] : 0,
                                           aabb_tree,
                                           image_list,
                                           image_hkl,
                                           m_stream,
                                           this->m_exec_conf->dev_prop);


        do
            {
            // reset flags
            m_conditions.resetFlags(make_uint2(0,0));

                {
                // access collision list arrays
                ArrayHandle<uint3> d_collisions(m_collisions, access_location::device, access_mode::overwrite);
                ArrayHandle<unsigned int> d_n_interact_new_new(m_n_interact_new_new, access_location::device, access_mode::overwrite);

                clusters_args.d_collisions = d_collisions.data;
                clusters_args.max_n_overlaps = m_overlaps.getNumElements() - m_n_overlaps_old_new;
                clusters_args.d_n_overlaps = d_n_interact_new_new.data;
                clusters_args.d_conditions = m_conditions.getDeviceFlags();

                m_tuner_new_new_collisions->begin();

                unsigned int param = m_tuner_new_new_collisions->getParam();
                clusters_args.block_size = param / 1000000;
                clusters_args.group_size = param % 1000000;

                // invoke kernel for checking circumsphere overlaps (broad phase)
                detail::gpu_hpmc_clusters<Shape>(clusters_args, params.data());

                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();

                m_tuner_new_new_collisions->end();
                }

            uint2 flags = m_conditions.readFlags();

            if (flags.x)
                {
                m_overlaps.resize(m_n_overlaps_old_new+flags.x);
                m_collisions.resize(flags.x);
                }

            reallocate = flags.x;
            } while(reallocate);

            {
            // extract number of collisions
            ArrayHandle<unsigned int> h_n_interact_new_new(m_n_interact_new_new, access_location::host, access_mode::read);
            clusters_args.ncollisions = *h_n_interact_new_new.data;
            }

            {
            // access collision and overlap list
            ArrayHandle<uint3> d_collisions(m_collisions, access_location::device, access_mode::read);
            ArrayHandle<uint2> d_overlaps(m_overlaps, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_n_interact_new_new(m_n_interact_new_new, access_location::device, access_mode::overwrite);

            clusters_args.d_overlaps = d_overlaps.data + m_n_overlaps_old_new;
            clusters_args.d_collisions = d_collisions.data;
            clusters_args.d_n_overlaps = d_n_interact_new_new.data;

            m_tuner_new_new_overlaps->begin();

            unsigned int param = m_tuner_new_new_overlaps->getParam();
            clusters_args.block_size = param;

            // invoke kernel for checking actual overlaps (narrow phase)
            detail::gpu_hpmc_clusters_overlaps<Shape>(clusters_args, params.data());

            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            m_tuner_new_new_overlaps->end();
            }

            {
            // extract number of overlaps
            ArrayHandle<unsigned int> h_n_interact_new_new(m_n_interact_new_new, access_location::host, access_mode::read);
            m_n_overlaps_new_new = *h_n_interact_new_new.data;
            }
        } // end if line transformation

    if (this->m_prof)
        this->m_prof->pop(this->m_exec_conf);
    }

template<class Shape>
#ifdef ENABLE_TBB
void UpdaterClustersGPU<Shape>::findConnectedComponents(unsigned int timestep, unsigned int N, bool line, bool swap, std::vector<tbb::concurrent_vector<unsigned int> >& clusters)
#else
void UpdaterClustersGPU<Shape>::findConnectedComponents(unsigned int timestep, unsigned int N, bool line, bool swap, std::vector<std::vector<unsigned int> >& clusters)
#endif
    {
    // collect interactions on rank 0
    bool master = !this->m_exec_conf->getRank();

    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        // combine lists from different ranks
        std::vector< std::vector<uint2> > all_overlaps;
        std::vector< std::vector<uint2> > all_interact_new_new;
        std::vector< std::vector<unsigned int> > all_local_reject;

        // overlap old new
        std::vector<uint2> overlaps(m_n_overlaps_old_new);
            {
            ArrayHandle<uint2> h_overlaps(m_overlaps, access_location::host, access_mode::read);
            std::copy(h_overlaps.data, h_overlaps.data + m_n_overlaps_old_new, overlaps.begin());
            }
        gather_v(overlaps, all_overlaps, 0, this->m_exec_conf->getMPICommunicator());

        // boundary interactions new new
        std::vector<uint2> interact_new_new(m_n_overlaps_new_new);
            {
            ArrayHandle<uint2> h_overlaps(m_overlaps, access_location::host, access_mode::read);
            std::copy(h_overlaps.data+m_n_overlaps_old_new,
                h_overlaps.data + m_n_overlaps_old_new + m_n_overlaps_new_new,
                interact_new_new.begin());
            }
        gather_v(interact_new_new, all_interact_new_new, 0, this->m_exec_conf->getMPICommunicator());

        // rejected particle moves
        std::vector<unsigned int> local_reject(m_reject.size());
            {
            ArrayHandle<unsigned int> h_reject(m_reject, access_location::host, access_mode::read);
            std::copy(h_reject.data, h_reject.data + m_reject.size(), local_reject.begin());
            }
        gather_v(local_reject, all_local_reject, 0, this->m_exec_conf->getMPICommunicator());

        if (master)
            {
            #ifdef ENABLE_MPI
            // complete the list of rejected particles
            for (auto it_i = all_local_reject.begin(); it_i != all_local_reject.end(); ++it_i)
                for (auto it_j = it_i->begin(); it_j != it_i->end(); ++it_j)
                    this->m_ptl_reject.insert(*it_j);

            // determine new size for overlaps list
            unsigned int n_overlaps = 0;
            for (auto it = all_overlaps.begin(); it != all_overlaps.end(); ++it)
                n_overlaps += it->size();
            for (auto it = all_interact_new_new.begin(); it != all_interact_new_new.end(); ++it)
                n_overlaps += it->size();

            // resize local adjacency list
            m_overlaps.resize(n_overlaps);

                {
                ArrayHandle<uint2> h_overlaps(m_overlaps, access_location::host, access_mode::overwrite);

                // collect adjacency matrix
                unsigned int offs = 0;
                for (auto it = all_overlaps.begin(); it != all_overlaps.end(); ++it)
                    {
                    std::copy(it->begin(), it->end(), h_overlaps.data + offs);
                    offs += it->size();
                    }
                for (auto it = all_interact_new_new.begin(); it != all_interact_new_new.end(); ++it)
                    {
                    std::copy(it->begin(), it->end(), h_overlaps.data + offs);
                    offs += it->size();
                    }
                }
            }
        #endif
        }
    #endif

    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "connected components");

    if (master)
        {
        // fill in the cluster bonds, using bond formation probability defined in Liu and Luijten

        this->m_ptl_reject.clear();
        this->m_local_reject.clear();

        #ifndef NVGRAPH_AVAILABLE
        // resize the number of graph nodes in place
        this->m_G.resize(N);
        #endif

        bool mpi = false;
        #ifdef ENABLE_MPI
        mpi = (bool) this->m_comm;
        #endif

        if (!mpi)
            {
            ArrayHandle<unsigned int> h_reject(m_reject, access_location::host, access_mode::read);
            for (unsigned int i = 0; i < m_reject.size(); ++i)
                this->m_local_reject.insert(h_reject.data[i]);
            }

        if (line && !swap)
            {
            ArrayHandle<uint2> h_overlaps(m_overlaps, access_location::host, access_mode::read);
            unsigned int offs = m_n_overlaps_old_new;

            #ifdef ENABLE_TBB
            tbb::parallel_for((unsigned int) 0, m_n_overlaps_new_new, [&] (unsigned int i)
            #else
            for (unsigned int i = 0; i < m_n_overlaps_new_new; ++i)
            #endif
                {
                #ifndef NVGRAPH_AVAILABLE
                this->m_G.addEdge(h_overlaps.data[offs + i].x, h_overlaps.data[offs + i].y);
                #endif

                // add to list of rejected particles
                this->m_local_reject.insert(h_overlaps.data[offs + i].x);
                this->m_local_reject.insert(h_overlaps.data[offs + i].y);
                }
            #ifdef ENABLE_TBB
                );
            #endif
            }

        #ifndef NVGRAPH_AVAILABLE
            {
            ArrayHandle<uint2> h_overlaps(m_overlaps, access_location::host, access_mode::read);

            #ifdef ENABLE_TBB
            tbb::parallel_for((unsigned int) 0, m_n_overlaps_old_new, [&] (unsigned int i) {
            #else
            for (unsigned int i = 0; i < m_n_overlaps_old_new; ++i)
            #endif
                this->m_G.addEdge(h_overlaps.data[i].x, h_overlaps.data[i].y);
            #ifdef ENABLE_TBB
                });
            #endif
            }
        #endif

        clusters.clear();

        #ifdef NVGRAPH_AVAILABLE
        m_components.resize(N);

        // access edges of adajacency matrix
        ArrayHandle<uint2> d_overlaps(m_overlaps, access_location::device, access_mode::read);

        // this will contain the number of strongly connected components
        unsigned int num_components = 0;

            {
            // access the output array
            ArrayHandle<unsigned int> d_components(m_components, access_location::device, access_mode::overwrite);

            unsigned int max_iterations = 50;
            float tol = 1e-10;
            float jump_tol = 1; // tolerance with which we detect jumps of the eigenvector components

            detail::gpu_connected_components(
                d_overlaps.data,
                N,
                m_n_overlaps_old_new + m_n_overlaps_new_new,
                d_components.data,
                num_components,
                m_stream,
                max_iterations,
                tol,
                jump_tol,
                this->m_exec_conf->getCachedAllocator());

            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // copy to host
        clusters.resize(num_components);
        ArrayHandle<unsigned int> h_components(m_components, access_location::host, access_mode::read);

        for (unsigned int i = 0; i < N; ++i)
            clusters[h_components.data[i]].push_back(i);
        #else
        // compute connected components on CPU
        this->m_G.connectedComponents(clusters);
        #endif
        }

    if (this->m_prof)
        this->m_prof->pop(this->m_exec_conf);
    }


template < class Shape> void export_UpdaterClustersGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterClustersGPU<Shape>, std::shared_ptr< UpdaterClustersGPU<Shape> > >(m, name.c_str(), pybind11::base<UpdaterClusters<Shape> >())
          .def( pybind11::init< std::shared_ptr<SystemDefinition>,
                         std::shared_ptr< IntegratorHPMCMono<Shape> >,
                         unsigned int >())
    ;
    }

} // end namespace hpmc

#endif // ENABLE_CUDA
#endif // _UPDATER_HPMC_CLUSTERS_GPU_
