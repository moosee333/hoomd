// inclusion guard
#ifndef _UPDATER_HPMC_CLUSTERS_GPU_
#define _UPDATER_HPMC_CLUSTERS_GPU_

/*! \file UpdaterBoxClusters.h
    \brief Declaration of UpdaterBoxClusters
*/

#ifdef ENABLE_CUDA

#include "UpdaterClusters.h"
#include "UpdaterClustersGPU.cuh"

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
            m_tuner_old_new->setPeriod(period);
            m_tuner_old_new->setEnabled(enable);

            m_tuner_new_new->setPeriod(period);
            m_tuner_new_new->setEnabled(enable);

            m_tuner_excell_block_size->setPeriod(period);
            m_tuner_excell_block_size->setEnabled(enable);
            }


    protected:
        GPUVector<unsigned int> m_new_tag;   //!< Lookup of snapshot order based on old particle index
        GPUFlags<uint2> m_conditions;        //!< Flags returned from GPU kernel

        GPUArray<unsigned int> m_n_overlaps; //!< Length of list of overlapping pairs
        GPUVector<uint2> m_overlaps;         //!< List of overlaps between old and new configuration
        GPUArray<unsigned int> m_n_reject;   //!< Length of list of rejected particle moves
        GPUVector<unsigned int> m_reject;    //!< List of rejected particle moves

        GPUArray<unsigned int> m_n_interact_new_new; //!< Length of list of particles interacting new-new
        GPUVector<uint2> m_interact_new_new; //!< Interactions between particles in the new configuration (only with line reflections)

        std::unique_ptr<Autotuner> m_tuner_old_new;     //!< Autotuner for checking old against new
        std::unique_ptr<Autotuner> m_tuner_new_new;     //!< Autotuner for checking new against new
        std::unique_ptr<Autotuner> m_tuner_excell_block_size;  //!< Autotuner for excell block_size

        CellListGPU m_cl;                     //!< The cell list

        uint3 m_last_dim;                     //!< Dimensions of the cell list on the last call to update
        unsigned int m_last_nmax;             //!< Last cell list NMax value allocated in excell

        GPUArray<unsigned int> m_excell_idx;  //!< Particle indices in expanded cells
        GPUArray<unsigned int> m_excell_size; //!< Number of particles in each expanded cell
        Index2D m_excell_list_indexer;        //!< Indexer to access elements of the excell_idx list

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
        virtual void findConnectedComponents(unsigned int timestep, unsigned int N, bool line, std::vector<tbb::concurrent_vector<unsigned int> >& clusters);
        #else
        virtual void findConnectedComponents(unsigned int timestep, unsigned int N, bool line, std::vector<std::vector<unsigned int> >& clusters);
        #endif
    };

template< class Shape >
UpdaterClustersGPU<Shape>::UpdaterClustersGPU(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
                                 unsigned int seed)
        : UpdaterClusters<Shape>(sysdef, mc, seed), m_conditions(this->m_exec_conf), m_cl(sysdef)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing UpdaterClustersGPU" << std::endl;

    // allocate memory
    GPUVector<unsigned int>(this->m_exec_conf).swap(m_new_tag);

    GPUArray<unsigned int>(1, this->m_exec_conf).swap(m_n_overlaps);
    GPUVector<uint2>(this->m_exec_conf).swap(m_overlaps);
    GPUArray<unsigned int>(1, this->m_exec_conf).swap(m_n_interact_new_new);
    GPUVector<uint2>(this->m_exec_conf).swap(m_interact_new_new);
    GPUArray<unsigned int>(1, this->m_exec_conf).swap(m_n_reject);
    GPUVector<unsigned int>(this->m_exec_conf).swap(m_reject);

    // initialize the autotuners
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
    m_tuner_old_new.reset(new Autotuner(valid_params, 5, 1000000, "hpmc_clusters_old_new", this->m_exec_conf));
    m_tuner_new_new.reset(new Autotuner(valid_params, 5, 1000000, "hpmc_clusters_new_new", this->m_exec_conf));

    // set up cell list
    this->m_cl.setRadius(1);
    this->m_cl.setComputeTDB(false);
    this->m_cl.setFlagType();
    this->m_cl.setComputeIdx(true);

    // set up expanded cell list
    GPUArray<unsigned int> excell_size(0, this->m_exec_conf);
    m_excell_size.swap(excell_size);

    GPUArray<unsigned int> excell_idx(0, this->m_exec_conf);
    m_excell_idx.swap(excell_idx);

    // set last dim to a bogus value so that it will re-init on the first call
    m_last_dim = make_uint3(0xffffffff, 0xffffffff, 0xffffffff);
    m_last_nmax = 0xffffffff;

    m_tuner_excell_block_size.reset(new Autotuner(32,1024,32, 5, 1000000, "hpmc_clusters_excell_block_size", this->m_exec_conf));

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
        this->m_exec_conf->msg->error() << "Simulation box too small for update.clusters() on GPU - increase it so the minimum image convention works" << endl;
        throw runtime_error("Error performing HPMC clusters");
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

    // old new
    bool reallocate = true;
    do
        {
        // reset flags
        m_conditions.resetFlags(make_uint2(0,0));

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

            ArrayHandle<uint2> d_overlaps(m_overlaps, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_reject(m_reject, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_n_overlaps(m_n_overlaps, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_n_reject(m_n_reject, access_location::device, access_mode::overwrite);

            ArrayHandle<unsigned int> d_check_overlaps(this->m_mc->getInteractionMatrix(), access_location::device, access_mode::read);
            const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

            m_tuner_old_new->begin();
            unsigned int param= m_tuner_old_new->getParam();
            unsigned int block_size = param / 1000000;
            unsigned int group_size = param % 100;

            detail::hpmc_clusters_args_t clusters_args(this->m_n_particles_old,
                                               d_postype.data,
                                               d_orientation.data,
                                               d_image.data,
                                               d_tag.data,
                                               d_cell_idx.data,
                                               d_cell_size.data,
                                               this->m_cl.getCellIndexer(),
                                               this->m_cl.getCellListIndexer(),
                                               d_excell_idx.data,
                                               d_excell_size.data,
                                               this->m_excell_list_indexer,
                                               this->m_cl.getDim(),
                                               this->m_pdata->getNTypes(),
                                               timestep,
                                               this->m_sysdef->getNDimensions(),
                                               box,
                                               block_size,
                                               1, //stride
                                               group_size,
                                               this->m_pdata->getMaxN(),
                                               this->m_cl.getGhostWidth(),
                                               d_check_overlaps.data,
                                               overlap_idx,
                                               line,
                                               d_postype_backup.data,
                                               d_orientation_backup.data,
                                               d_image_backup.data,
                                               d_new_tag.data,
                                               d_overlaps.data,
                                               m_overlaps.getNumElements(),
                                               d_n_overlaps.data,
                                               d_reject.data,
                                               m_reject.getNumElements(),
                                               d_n_reject.data,
                                               m_conditions.getDeviceFlags(),
                                               swap,
                                               swap ? this->m_ab_types[0] : 0,
                                               swap ? this->m_ab_types[1] : 0,
                                               m_stream,
                                               this->m_exec_conf->dev_prop);

            // invoke kernel for checking overlaps
            detail::gpu_hpmc_clusters<Shape>(clusters_args, params.data());

            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            m_tuner_old_new->end();
            }

        uint2 flags = m_conditions.readFlags();

        // resize to largest element + 1
        if (flags.x)
            m_overlaps.resize(flags.x);

        if (flags.y)
            m_reject.resize(flags.y);

        reallocate = flags.x || flags.y;
        } while(reallocate);

        {
        // refit arrays to actual size
        ArrayHandle<unsigned int> h_n_overlaps(m_n_overlaps, access_location::host, access_mode::read);
        m_overlaps.resize(*h_n_overlaps.data);

        ArrayHandle<unsigned int> h_n_reject(m_n_reject, access_location::host, access_mode::read);
        m_reject.resize(*h_n_reject.data);
        }

    if (line)
        {
        // with line transformations, check new configuration against itself to detect interactions across PBC
        do
            {
            // reset flags
            m_conditions.resetFlags(make_uint2(0,0));

                {
                // access new particle data
                ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::read);
                ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::read);
                ArrayHandle<int3> d_image(this->m_pdata->getImages(), access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(), access_location::device, access_mode::read);

                ArrayHandle<uint2> d_interact_new_new(m_interact_new_new, access_location::device, access_mode::overwrite);
                ArrayHandle<unsigned int> d_n_interact_new_new(m_n_interact_new_new, access_location::device, access_mode::overwrite);

                ArrayHandle<unsigned int> d_check_overlaps(this->m_mc->getInteractionMatrix(), access_location::device, access_mode::read);
                const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

                m_tuner_new_new->begin();
                unsigned int param = m_tuner_new_new->getParam();
                unsigned int block_size = param / 1000000;
                unsigned int group_size = param % 100;

                detail::hpmc_clusters_args_t clusters_args(this->m_pdata->getN(),
                                                   d_postype.data,
                                                   d_orientation.data,
                                                   d_image.data,
                                                   d_tag.data,
                                                   d_cell_idx.data,
                                                   d_cell_size.data,
                                                   this->m_cl.getCellIndexer(),
                                                   this->m_cl.getCellListIndexer(),
                                                   d_excell_idx.data,
                                                   d_excell_size.data,
                                                   this->m_excell_list_indexer,
                                                   this->m_cl.getDim(),
                                                   this->m_pdata->getNTypes(),
                                                   timestep,
                                                   this->m_sysdef->getNDimensions(),
                                                   box,
                                                   block_size,
                                                   1, //stride
                                                   group_size,
                                                   this->m_pdata->getMaxN(),
                                                   this->m_cl.getGhostWidth(),
                                                   d_check_overlaps.data,
                                                   overlap_idx,
                                                   line,
                                                   d_postype.data,
                                                   d_orientation.data,
                                                   d_image.data,
                                                   d_tag.data,
                                                   d_interact_new_new.data,
                                                   m_interact_new_new.getNumElements(),
                                                   d_n_interact_new_new.data,
                                                   0, // d_reject
                                                   0, // max_n_reject
                                                   0, // d_n_reject
                                                   m_conditions.getDeviceFlags(),
                                                   swap,
                                                   swap ? this->m_ab_types[0] : 0,
                                                   swap ? this->m_ab_types[1] : 0,
                                                   m_stream,
                                                   this->m_exec_conf->dev_prop);

                // invoke kernel for checking overlaps
                detail::gpu_hpmc_clusters<Shape>(clusters_args, params.data());

                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();

                m_tuner_new_new->end();
                }

            uint2 flags = m_conditions.readFlags();

            if (flags.x)
                m_interact_new_new.resize(flags.x);

            reallocate = flags.x;
            } while(reallocate);

            {
            // refit array to actual size
            ArrayHandle<unsigned int> h_n_interact_new_new(m_n_interact_new_new, access_location::host, access_mode::read);
            m_interact_new_new.resize(*h_n_interact_new_new.data);
            }
        } // end if line transformation

    if (this->m_prof)
        this->m_prof->pop(this->m_exec_conf);
    }

template<class Shape>
#ifdef ENABLE_TBB
void UpdaterClustersGPU<Shape>::findConnectedComponents(unsigned int timestep, unsigned int N, bool line, std::vector<tbb::concurrent_vector<unsigned int> >& clusters)
#else
void UpdaterClustersGPU<Shape>::findConnectedComponents(unsigned int timestep, unsigned int N, bool line, std::vector<std::vector<unsigned int> >& clusters)
#endif
    {
    // collect interactions on rank 0
    std::vector< std::vector<uint2> > all_overlap;
    std::vector< std::vector<uint2> > all_interact_new_new;
    std::vector< std::vector<unsigned int> > all_local_reject;

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // combine lists from different ranks

        // overlap old new
        std::vector<uint2> overlaps(m_overlaps.size());
            {
            ArrayHandle<uint2> h_overlaps(m_overlaps, access_location::host, access_mode::read);
            std::copy(h_overlaps.data, h_overlaps.data + m_overlaps.size(), overlaps.begin());
            }
        gather_v(interact_new_new, all_overlap, 0, m_exec_conf->getMPICommunicator());

        // boundary interactions new new
        std::vector<uint2> interact_new_new(m_interact_new_new.size());
            {
            ArrayHandle<uint2> h_interact_new_new(m_interact_new_new, access_location::host, access_mode::read);
            std::copy(h_interact_new_new.data, h_interact_new_new.data + m_interact_new_new.size(), interact_new_new.begin());
            }
        gather_v(interact_new_new, all_interact_new_new, 0, m_exec_conf->getMPICommunicator());

        // rejected particle moves
        std::vector<unsigned int> local_reject(m_reject.size());
            {
            ArrayHandle<unsigned int> h_reject(m_reject, access_location::host, access_mode::read);
            std::copy(h_reject.data, h_reject.data + m_reject.size(), local_reject.begin());
            }
        gather_v(local_reject, all_local_reject, 0, m_exec_conf->getMPICommunicator());
        }
    #endif

    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "connected components");

    bool master = !this->m_exec_conf->getRank();
    if (master)
        {
        // fill in the cluster bonds, using bond formation probability defined in Liu and Luijten

        // resize the number of graph nodes in place
        this->m_G.resize(N);

        #ifdef ENABLE_MPI
        if (this->m_comm)
            {
            // complete the list of rejected particles
            for (auto it_i = all_local_reject.begin(); it_i != all_local_reject.end(); ++it_i)
                for (auto it_j = it_i->begin(); it_j != it_i->end(); ++it_j)
                    this->m_ptl_reject.insert(*it_j);
            }
        else
        #endif
            {
            ArrayHandle<unsigned int> h_reject(m_reject, access_location::host, access_mode::read);
            for (unsigned int i = 0; i < m_reject.size(); ++i)
                this->m_local_reject.insert(h_reject.data[i]);
            }

        if (line)
            {
            #ifdef ENABLE_MPI
            if (this->m_comm)
                {
                for (auto it_i = all_interact_new_new.begin(); it_i != all_interact_new_new.end(); ++it_i)
                    for (auto it_j = it_i->begin(); it_j != it_i->end(); ++it_j)
                        {
                        this->m_G.addEdge(it_j->x, it_j->y);

                        // add to list of rejected particles
                        this->m_ptl_reject.insert(it_j->x);
                        this->m_ptl_reject.insert(it_j->y);
                        }
                }
            else
            #endif
                {
                ArrayHandle<uint2> h_interact_new_new(m_interact_new_new, access_location::host, access_mode::read);

                for (unsigned int i = 0; i < m_interact_new_new.size(); ++i)
                    {
                    this->m_G.addEdge(h_interact_new_new.data[i].x, h_interact_new_new.data[i].y);

                    // add to list of rejected particles
                    this->m_ptl_reject.insert(h_interact_new_new.data[i].x);
                    this->m_ptl_reject.insert(h_interact_new_new.data[i].y);
                    }
                }
            }

        #ifdef ENABLE_MPI
        if (this->m_comm)
            {
            for (auto it_i = all_overlap.begin(); it_i != all_overlap.end(); ++it_i)
                for (auto it_j = it_i->begin(); it_j != it_i->end(); ++it_j)
                    this->m_G.addEdge(it_j->first,it_j->second);
            }
        else
        #endif
            {
            ArrayHandle<uint2> h_overlaps(m_overlaps, access_location::host, access_mode::read);

            for (unsigned int i = 0; i < m_overlaps.size(); ++i)
                this->m_G.addEdge(h_overlaps.data[i].x, h_overlaps.data[i].y);
            }

        // compute connected components
        clusters.clear();
        this->m_G.connectedComponents(clusters);
        }

    if (this->m_prof)
        this->m_prof->pop(this->m_exec_conf);
    }

template< class Shape >
void UpdaterClustersGPU< Shape >::initializeExcellMem()
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
