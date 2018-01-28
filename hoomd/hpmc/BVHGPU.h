// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser, mphoward

#ifdef ENABLE_CUDA
#include "BVHGPU.cuh"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#include "IntegratorHPMCMono.h"

#include "hoomd/Compute.h"
#include "hoomd/Autotuner.h"
#include "hoomd/md/NeighborListGPUTree.cuh"

#include "OBB.h"

#include <type_traits>
#include <tuple>

#include <cuda_runtime.h>

/*! \file BVHGPU.h
    \brief Declares the BVHGPU template class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __BVHGPU_H__
#define __BVHGPU_H__

namespace hpmc
{

//! Template class for efficient BVH construction on the GPU, supporting general bounding volumes
/*!
 * Templated GPU kernel methods are defined in BVHGPU.cuh and implemented in BVHGPU.cu.
 * This class is modeled after NeighborListGPUTree, for further documentation see there.
 *
 * \ingroup computes
 */
template<class BVNode, class Shape = detail::EmptyShape, class IntHPMC = std::tuple<> >
class BVHGPU : public Compute
    {
    public:
        typedef BVNode node_type;  //!< Export the type of BVH node

        //! Constructs the compute
        /*! \param sysdef The system definition
            \param mc The HPMC integrator

            If mc is not provided, no shape parameters will be loaded.
            Therefore, the class can also be used in an MD setting with a point shape.
        */
        BVHGPU(std::shared_ptr<SystemDefinition> sysdef,
               std::shared_ptr<IntHPMC> mc = std::shared_ptr<IntHPMC>() );

        //! Destructor
        virtual ~BVHGPU();

        //! Update the bounding volume hierarchy
        virtual void compute(unsigned int timestep);

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tuner_morton->setPeriod(period);
            m_tuner_morton->setEnabled(enable);

            m_tuner_merge->setPeriod(period);
            m_tuner_merge->setEnabled(enable);

            m_tuner_hierarchy->setPeriod(period);
            m_tuner_hierarchy->setEnabled(enable);

            m_tuner_bubble->setPeriod(period);
            m_tuner_bubble->setEnabled(enable);

            m_tuner_move->setPeriod(period);
            m_tuner_move->setEnabled(enable);

            m_tuner_map->setPeriod(period);
            m_tuner_map->setEnabled(enable);
            }

        //! Accessor methods

        const GPUArray<unsigned int>& getLeafOffsets()
            {
            return m_leaf_offset;
            }

        const GPUArray<unsigned int>& getTreeRoots()
            {
            return m_tree_roots;
            }

        const GPUArray<BVNode>& getTreeNodes()
            {
            return m_tree_nodes;
            }

        const GPUArray<Scalar4>& getLeafXYZF()
            {
            return m_leaf_xyzf;
            }

        unsigned int getParticlesPerLeaf()
            {
            return m_particles_per_leaf;
            }

    protected:
        std::shared_ptr<IntHPMC> m_mc;               //!< The HPMC integrator

        // some metaprogramming to not load parameters when the template argument doesn't support it

        //! If IntHPMC is HPMC integrator, return its parameters
        /*! \returns a pointer to the parameter data structure per type
         */
        template<typename T = IntHPMC>
        typename std::enable_if<std::is_same<T, IntegratorHPMCMono<Shape> >::value,
            const typename Shape::param_type *>::type getDeviceParams()
            {
            return m_mc->getParams().data();
            }

        //! If no HPMC integrator is provided, just return a nullptr (empty parameter)
        template<typename T = IntHPMC>
        typename std::enable_if<!std::is_same<T, IntegratorHPMCMono<Shape> >::value,
            const typename Shape::param_type *>::type getDeviceParams()
            {
            return nullptr;
            }

    private:
        //! \name Autotuners
        // @{
        std::unique_ptr<Autotuner> m_tuner_morton;    //!< Tuner for kernel to calculate morton codes
        std::unique_ptr<Autotuner> m_tuner_merge;     //!< Tuner for kernel to merge particles into leafs
        std::unique_ptr<Autotuner> m_tuner_hierarchy; //!< Tuner for kernel to generate tree hierarchy
        std::unique_ptr<Autotuner> m_tuner_bubble;    //!< Tuner for kernel to bubble bounding volumes up hierarchy
        std::unique_ptr<Autotuner> m_tuner_move;      //!< Tuner for kernel to move particles to leaf order
        std::unique_ptr<Autotuner> m_tuner_map;       //!< Tuner for kernel to help map particles by type
        // @}

        //! \name Signal updates
        // @{

        //! Notification of a change in the maximum number of particles on any rank
        void slotMaxNumChanged()
            {
            m_max_num_changed = true;
            }

        //! Notification of a change in the number of types
        void slotNumTypesChanged()
            {
            // skip the reallocation if the number of types does not change
            // this keeps old parameters when restoring a snapshot
            // it will result in invalid coeficients if the snapshot has a different type id -> name mapping
            if (m_pdata->getNTypes() == m_prev_ntypes)
                return;

            m_type_changed = true;
            }

        unsigned int m_prev_ntypes;                         //!< Previous number of types
        bool m_type_changed;                                //!< Flag if types changed
        bool m_max_num_changed;                             //!< Flag if max number of particles changed
        // @}

        //! \name Tree building
        // @{
        // mapping and sorting
        GPUArray<unsigned int> m_map_tree_pid;      //!< Map a leaf order id to a particle id
        GPUArray<unsigned int> m_map_tree_pid_alt;  //!< Double buffer for map needed for sorting

        GPUArray<uint64_t> m_morton_types;      //!< 30 bit morton codes + type for particles to sort on z-order curve
        GPUArray<uint64_t> m_morton_types_alt;  //!< Double buffer for morton codes needed for sorting
        GPUFlags<int> m_morton_conditions;      //!< Condition flag to catch out of bounds particles

        GPUArray<unsigned int> m_leaf_offset;   //!< Total offset in particle index for leaf nodes by type
        GPUArray<unsigned int> m_num_per_type;  //!< Number of particles per type
        GPUArray<unsigned int> m_type_head;     //!< Head list to each particle type
        GPUArray<unsigned int> m_tree_roots;    //!< Index for root node of each tree by type

        // hierarchy generation
        unsigned int m_n_leaf;                      //!< Total number of leaves in trees
        unsigned int m_n_internal;                  //!< Total number of internal nodes in trees
        unsigned int m_n_node;                      //!< Total number of leaf + internal nodes in trees
        unsigned int m_particles_per_leaf;          //!< Number of particles per leaf

        GPUVector<uint32_t> m_morton_codes_red;     //!< Reduced capacity 30 bit morton code array (per leaf)
        GPUVector<BVNode> m_tree_nodes;             //!< Nodes and bounding volume of both merged leaf nodes and internal nodes
        GPUVector<unsigned int> m_node_locks;       //!< Node locks for if node has been visited or not
        GPUVector<uint2> m_tree_parent_sib;         //!< Parents and siblings of all nodes

        cudaStream_t m_stream;                //!< CUDA stream for kernel execution

        //! Performs initial allocation of tree internal data structure memory
        void allocateTree();

        //! Performs all tasks needed before tree build and traversal
        void setupTree();

        //! Determines the number and head indexes for particle types and leafs
        void countParticlesAndTrees();

        //! Driver for tree multi-step tree build on the GPU
        void buildTree();

        //! Calculates 30-bit morton codes for particles
        void calcMortonCodes();

        //! Driver to sort particles by type and morton code along a Z order curve
        void sortMortonCodes();

        //! Calculates the number of bits needed to represent the largest particle type
        void calcTypeBits();
        unsigned int m_n_type_bits;     //!< the number of bits it takes to represent all the type ids

        //! Merges sorted particles into leafs based on adjacency
        void mergeLeafParticles();

        //! Generates the edges between nodes based on the sorted morton codes
        void genTreeHierarchy();

        //! Constructs enclosing bounding volums from leaf to roots
        void bubbleBoundingVolumes();

        // @}
        //! \name Tree traversal
        // @{

        GPUArray<Scalar4> m_leaf_xyzf;          //!< Position and id of each particle in a leaf
        GPUArray<Scalar2> m_leaf_db;            //!< Diameter and body of each particle in a leaf

        //! Moves particles from ParticleData order to leaf order for more efficient tree traversal
        void moveLeafParticles();

        //! Traverses the trees on the GPU
        void traverseTree();
        // @}
    };

template<class BVNode, class Shape, class IntHPMC>
BVHGPU<BVNode, Shape, IntHPMC>::BVHGPU(std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<IntHPMC> mc)
    : Compute(sysdef), m_mc(mc), m_type_changed(false),
      m_max_num_changed(false), m_n_leaf(0), m_n_internal(0), m_n_node(0),
      m_particles_per_leaf(4)
    {
    m_exec_conf->msg->notice(5) << "Constructing BVHGPU" << std::endl;

    m_pdata->getNumTypesChangeSignal().connect<BVHGPU, &BVHGPU::slotNumTypesChanged>(this);
    m_pdata->getMaxParticleNumberChangeSignal().connect<BVHGPU, &BVHGPU::slotMaxNumChanged>(this);

    m_tuner_morton.reset(new Autotuner(32, 1024, 32, 5, 100000, "bvh_morton_codes", this->m_exec_conf));
    m_tuner_merge.reset(new Autotuner(32, 1024, 32, 5, 100000, "bvh_merge_shapes", this->m_exec_conf));
    m_tuner_hierarchy.reset(new Autotuner(32, 1024, 32, 5, 100000, "bvh_gen_hierarchy", this->m_exec_conf));
    m_tuner_bubble.reset(new Autotuner(32, 1024, 32, 5, 100000, "bvh_bubble_bounding_volumes", this->m_exec_conf));
    m_tuner_move.reset(new Autotuner(32, 1024, 32, 5, 100000, "bvh_move_particles", this->m_exec_conf));
    m_tuner_map.reset(new Autotuner(32, 1024, 32, 5, 100000, "bvh_map_particles", this->m_exec_conf));

    allocateTree();

    calcTypeBits();

    m_prev_ntypes = m_pdata->getNTypes();

    // create a cuda stream to ensure managed memory coherency
    cudaStreamCreate(&m_stream);
    CHECK_CUDA_ERROR();
    }

template<class BVNode, class Shape, class IntHPMC>
BVHGPU<BVNode, Shape, IntHPMC>::~BVHGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying BVHGPU" << std::endl;
    m_pdata->getNumTypesChangeSignal().disconnect<BVHGPU, &BVHGPU::slotNumTypesChanged>(this);
    m_pdata->getMaxParticleNumberChangeSignal().disconnect<BVHGPU, &BVHGPU::slotMaxNumChanged>(this);

    cudaStreamDestroy(m_stream);
    CHECK_CUDA_ERROR();
    }

template<class BVNode, class Shape, class IntHPMC>
void BVHGPU<BVNode, Shape, IntHPMC>::compute(unsigned int timestep)
    {
    // skip if we shouldn't compute this step
    if (!shouldCompute(timestep))
        return;

    // kernels will crash in strange ways if there are no particles owned by the rank
    // so the build should just be aborted here (there are no neighbors to compute if there are no particles)
    if (!(m_pdata->getN()+m_pdata->getNGhosts()))
        {
        // maybe we should clear the arrays here, but really whoever's using the BVH should
        // just be smart enough to not try to use something that shouldn't exist
        return;
        }

    // allocate the tree memory as needed based on the mapping
    setupTree();

    // build the tree
    buildTree();
    }

template<class BVNode, class Shape, class IntHPMC>
void BVHGPU<BVNode, Shape, IntHPMC>::allocateTree()
    {
    // allocate per particle memory
    GPUArray<uint64_t> morton_types(m_pdata->getMaxN(), m_exec_conf);
    m_morton_types.swap(morton_types);
    GPUArray<uint64_t> morton_types_alt(m_pdata->getMaxN(), m_exec_conf);
    m_morton_types_alt.swap(morton_types_alt);

    GPUArray<unsigned int> map_tree_pid(m_pdata->getMaxN(), m_exec_conf);
    m_map_tree_pid.swap(map_tree_pid);
    GPUArray<unsigned int> map_tree_pid_alt(m_pdata->getMaxN(), m_exec_conf);
    m_map_tree_pid_alt.swap(map_tree_pid_alt);

    GPUArray<Scalar4> leaf_xyzf(m_pdata->getMaxN(), m_exec_conf);
    m_leaf_xyzf.swap(leaf_xyzf);

    GPUArray<Scalar2> leaf_db(m_pdata->getMaxN(), m_exec_conf);
    m_leaf_db.swap(leaf_db);

    // allocate per type memory
    GPUArray<unsigned int> leaf_offset(m_pdata->getNTypes(), m_exec_conf);
    m_leaf_offset.swap(leaf_offset);

    GPUArray<unsigned int> tree_roots(m_pdata->getNTypes(), m_exec_conf);
    m_tree_roots.swap(tree_roots);

    GPUArray<unsigned int> num_per_type(m_pdata->getNTypes(), m_exec_conf);
    m_num_per_type.swap(num_per_type);

    GPUArray<unsigned int> type_head(m_pdata->getNTypes(), m_exec_conf);
    m_type_head.swap(type_head);

    // allocate the tree memory to default lengths of 0 (will be resized later)
    // we use a GPUVector instead of GPUArray for the amortized resizing
    GPUVector<uint2> tree_parent_sib(m_exec_conf);
    m_tree_parent_sib.swap(tree_parent_sib);

    // holds two Scalar4s per node in tree
    GPUVector<BVNode> tree_nodes(m_exec_conf);
    m_tree_nodes.swap(tree_nodes);

    // we really only need as many morton codes as we have leafs
    GPUVector<uint32_t> morton_codes_red(m_exec_conf);
    m_morton_codes_red.swap(morton_codes_red);

    // 1 / 0 locks for traversing up the tree
    GPUVector<unsigned int> node_locks(m_exec_conf);
    m_node_locks.swap(node_locks);

    // conditions
    GPUFlags<int> morton_conditions(m_exec_conf);
    m_morton_conditions.swap(morton_conditions);
    }

/*!
 * \post Tree internal data structures are updated to begin a build.
 */
template<class BVNode, class Shape, class IntHPMC>
void BVHGPU<BVNode, Shape, IntHPMC>::setupTree()
    {
    // increase arrays that depend on the local number of particles
    if (m_max_num_changed)
        {
        m_morton_types.resize(m_pdata->getMaxN());
        m_morton_types_alt.resize(m_pdata->getMaxN());
        m_map_tree_pid.resize(m_pdata->getMaxN());
        m_map_tree_pid_alt.resize(m_pdata->getMaxN());
        m_leaf_xyzf.resize(m_pdata->getMaxN());
        m_leaf_db.resize(m_pdata->getMaxN());

        // all done with the particle data reallocation
        m_max_num_changed = false;
        }

    // allocate memory that depends on type
    if (m_type_changed)
        {
        m_leaf_offset.resize(m_pdata->getNTypes());
        m_tree_roots.resize(m_pdata->getNTypes());
        m_num_per_type.resize(m_pdata->getNTypes());
        m_type_head.resize(m_pdata->getNTypes());

        // get the number of bits needed to represent all the types
        calcTypeBits();

        // all done with the type reallocation
        m_type_changed = false;
        m_prev_ntypes = m_pdata->getNTypes();
        }
    }

/*!
 * Determines the number of bits needed to represent the largest type index for more efficient particle sorting.
 * This is done by taking the ceiling of the log2 of the type index using integers.
 * \sa sortMortonCodes
 */
template<class BVNode, class Shape, class IntHPMC>
inline void BVHGPU<BVNode, Shape, IntHPMC>::calcTypeBits()
    {
    if (m_pdata->getNTypes() > 1)
        {
        unsigned int n_type_bits = 0;

        // start with the maximum type id that there can be
        unsigned int tmp = m_pdata->getNTypes() - 1;

        // see how many times you can bit shift
        while (tmp >>= 1)
            {
            ++n_type_bits;
            }

        // add one to get the number of bits needed (rounding up int logarithm)
        m_n_type_bits = n_type_bits + 1;
        }
    else
        {
        // if there is only one type, you don't need to do any sorting
        m_n_type_bits = 0;
        }
    }

/*!
 * Determines the number of particles per type (and their starting indexes) in the flat leaf particle order. Also
 * determines the leaf offsets and and tree roots. When there is only one type, most operations are skipped since these
 * values are simple to determine.
 */
template<class BVNode, class Shape, class IntHPMC>
void BVHGPU<BVNode, Shape, IntHPMC>::countParticlesAndTrees()
    {
    if (m_prof) m_prof->push(m_exec_conf,"map");

    if (m_pdata->getNTypes() > 1)
        {
        // first do the stuff with the particle data on the GPU to avoid a costly copy
            {
            ArrayHandle<unsigned int> d_type_head(m_type_head, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_map_tree_pid(m_map_tree_pid, access_location::device, access_mode::read);
            m_tuner_map->begin();
            detail::gpu_bvh_init_count(d_type_head.data,
                                 d_pos.data,
                                 d_map_tree_pid.data,
                                 m_pdata->getN() + m_pdata->getNGhosts(),
                                 m_pdata->getNTypes(),
                                 m_tuner_map->getParam());
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            m_tuner_map->end();
            }


        // then do the harder to parallelize stuff on the cpu because the number of types is usually small
        // so what's the point of trying this in parallel to save a copy of a few bytes?
            {
            // the number of leafs is the first tree root
            ArrayHandle<unsigned int> h_type_head(m_type_head, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_num_per_type(m_num_per_type, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_leaf_offset(m_leaf_offset, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_tree_roots(m_tree_roots, access_location::host, access_mode::overwrite);

            // loop through the type heads and figure out how many there are of each
            m_n_leaf = 0;
            unsigned int total_offset = 0;
            unsigned int active_types = 0; // tracks the number of types that currently have particles
            for (unsigned int cur_type = 0; cur_type < m_pdata->getNTypes(); ++cur_type)
                {
                const unsigned int head_plus_1 = h_type_head.data[cur_type];

                unsigned int N_i = 0;

                if (head_plus_1 > 0) // there are particles of this type
                    {
                    // so loop over the types (we are ordered), and try to find a match
                    unsigned int next_head_plus_1 = 0;
                    for (unsigned int next_type = cur_type + 1; !next_head_plus_1 && next_type < m_pdata->getNTypes(); ++next_type)
                        {
                        if (h_type_head.data[next_type]) // this head exists
                            {
                            next_head_plus_1 = h_type_head.data[next_type];
                            }
                        }
                    // if we still haven't found a match, then the end index (+1) should be the end of the list
                    if (!next_head_plus_1)
                        {
                        next_head_plus_1 = m_pdata->getN() + m_pdata->getNGhosts() + 1;
                        }
                    N_i = next_head_plus_1 - head_plus_1;
                    }

                // set the number per type
                h_num_per_type.data[cur_type] = N_i;
                if (N_i > 0) ++active_types;

                // compute the number of leafs for this type, and accumulate it
                // temporarily stash the number of leafs in the tree root array
                unsigned int cur_n_leaf = (N_i + m_particles_per_leaf - 1)/m_particles_per_leaf;
                h_tree_roots.data[cur_type] = cur_n_leaf;
                m_n_leaf += cur_n_leaf;

                // compute the offset that is needed for this type, set and accumulate the total offset required
                const unsigned int remainder = N_i % m_particles_per_leaf;
                const unsigned int cur_offset = (remainder > 0) ? (m_particles_per_leaf - remainder) : 0;
                h_leaf_offset.data[cur_type] = total_offset;
                total_offset += cur_offset;
                }

            // each tree has Nleaf,i - 1 internal nodes
            // so in total we have N_leaf - N_types internal nodes for each type that has at least one particle
            m_n_internal = m_n_leaf - active_types;
            m_n_node = m_n_leaf + m_n_internal;

            // now loop over the roots one more time, and set each of them
            unsigned int leaf_head = 0;
            unsigned int internal_head = m_n_leaf;
            for (unsigned int cur_type = 0; cur_type < m_pdata->getNTypes(); ++cur_type)
                {
                const unsigned int n_leaf_i = h_tree_roots.data[cur_type];
                if (n_leaf_i == 0)
                    {
                    h_tree_roots.data[cur_type] = BVH_GPU_INVALID_NODE;
                    }
                else if (n_leaf_i == 1)
                    {
                    h_tree_roots.data[cur_type] = leaf_head;
                    }
                else
                    {
                    h_tree_roots.data[cur_type] = internal_head;
                    internal_head += n_leaf_i - 1;
                    }
                leaf_head += n_leaf_i;
                }
            }
        }
    else // only one type
        {
        ArrayHandle<unsigned int> h_type_head(m_type_head, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_num_per_type(m_num_per_type, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_leaf_offset(m_leaf_offset, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tree_roots(m_tree_roots, access_location::host, access_mode::overwrite);

        // with one type, we don't need to do anything fancy
        // type head is the first particle
        h_type_head.data[0] = 0;

        // num per type is all the particles in the rank
        h_num_per_type.data[0] = m_pdata->getN() + m_pdata->getNGhosts();

        // there is no leaf offset
        h_leaf_offset.data[0] = 0;

        // number of leafs is for all particles
        m_n_leaf = (m_pdata->getN() + m_pdata->getNGhosts() + m_particles_per_leaf - 1)/m_particles_per_leaf;

        // number of internal nodes is one less than number of leafs
        m_n_internal = m_n_leaf - 1;
        m_n_node = m_n_leaf + m_n_internal;

        // the root is the end of the leaf list if multiple leafs, otherwise the only leaf
        h_tree_roots.data[0] = (m_n_leaf > 1) ? m_n_leaf : 0;
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * Driver to implement the tree build algorithm of Karras,
 * "Maximizing parallelism in the construction of BVHs, octrees, and k-d trees", High Performance Graphics (2012).
 * \post a valid tree is allocated and ready for traversal
 */
template<class BVNode, class Shape, class IntHPMC>
void BVHGPU<BVNode, Shape, IntHPMC>::buildTree()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Build tree");

    // step one: morton code calculation
    calcMortonCodes();

    // step two: particle sorting
    sortMortonCodes();

    // step three: map the particles by type
    countParticlesAndTrees();

    // (re-) allocate memory that depends on tree size
    // GPUVector should only do this as needed
    m_tree_parent_sib.resize(m_n_node);
    m_tree_nodes.resize(m_n_node);
    m_morton_codes_red.resize(m_n_leaf);
    m_node_locks.resize(m_n_internal);

    // step four: merge leaf particles into nodes by morton code
    mergeLeafParticles();

    // step five: hierarchy generation from morton codes
    genTreeHierarchy();

    // step six: bubble up the bounding volumes
    bubbleBoundingVolumes();

    // step seven: move particle information into leaf nodes
    moveLeafParticles();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * \post One morton code-type key is assigned per particle
 * \note Call before sortMortonCodes().
 */
template<class BVNode, class Shape, class IntHPMC>
void BVHGPU<BVNode, Shape, IntHPMC>::calcMortonCodes()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Morton codes");

    // particle data and where to write it
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_map_tree_pid(m_map_tree_pid, access_location::device, access_mode::overwrite);

    ArrayHandle<uint64_t> d_morton_types(m_morton_types, access_location::device, access_mode::overwrite);

    // need a ghost layer width to get the fractional position of particles in the local box
    const BoxDim& box = m_pdata->getBox();

    Scalar ghost_layer_width(0.0);
    #ifdef ENABLE_MPI
    if (m_comm) ghost_layer_width = m_comm->getGhostLayerMaxWidth();
    #endif

    Scalar3 ghost_width = make_scalar3(0.0, 0.0, 0.0);
    if (!box.getPeriodic().x) ghost_width.x = ghost_layer_width;
    if (!box.getPeriodic().y) ghost_width.y = ghost_layer_width;
    if (this->m_sysdef->getNDimensions() == 3 && !box.getPeriodic().z)
        {
        ghost_width.z = ghost_layer_width;
        }


    // reset the flag to zero before calling the compute
    m_morton_conditions.resetFlags(0);

    m_tuner_morton->begin();
    detail::gpu_bvh_morton_types(d_morton_types.data,
                           d_map_tree_pid.data,
                           m_morton_conditions.getDeviceFlags(),
                           d_pos.data,
                           m_pdata->getN(),
                           m_pdata->getNGhosts(),
                           box,
                           ghost_width,
                           m_tuner_morton->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_morton->end();

    // error check that no local particles are out of bounds
    const unsigned int morton_conditions = m_morton_conditions.readFlags();
    if (morton_conditions > 0)
        {
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
        Scalar4 post_i = h_pos.data[morton_conditions-1];
        Scalar3 pos_i = make_scalar3(post_i.x, post_i.y, post_i.z);
        Scalar3 f = box.makeFraction(pos_i);
        m_exec_conf->msg->error() << "BVHGPU: Particle " << h_tag.data[morton_conditions-1] << " is out of bounds "
                                  << "(x: " << post_i.x << ", y: " << post_i.y << ", z: " << post_i.z
                                  << ", fx: "<< f.x <<", fy: "<<f.y<<", fz:"<<f.z<<")"<< std::endl;
        throw std::runtime_error("Error updating neighborlist");
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * Invokes the CUB libraries to sort the morton code-type keys.
 * \pre Morton code-keys are in local ParticleData order
 * \post Morton code-keys are sorted by type then position along the Z order curve.
 * \note Call after calcMortonCodes(), but before mergeLeafParticles().
 */
template<class BVNode, class Shape, class IntHPMC>
void BVHGPU<BVNode, Shape, IntHPMC>::sortMortonCodes()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Sort");

    bool swap_morton = false;
    bool swap_map = false;
        {
        ArrayHandle<uint64_t> d_morton_types(m_morton_types, access_location::device, access_mode::readwrite);
        ArrayHandle<uint64_t> d_morton_types_alt(m_morton_types_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_map_tree_pid(m_map_tree_pid, access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_map_tree_pid_alt(m_map_tree_pid_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> h_num_per_type(m_num_per_type, access_location::host, access_mode::read);

        // size the temporary storage
        void *d_tmp_storage = NULL;
        size_t tmp_storage_bytes = 0;
        detail::gpu_bvh_morton_sort(d_morton_types.data,
                              d_morton_types_alt.data,
                              d_map_tree_pid.data,
                              d_map_tree_pid_alt.data,
                              d_tmp_storage,
                              tmp_storage_bytes,
                              swap_morton,
                              swap_map,
                              m_pdata->getN() + m_pdata->getNGhosts(),
                              m_n_type_bits);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        /*
         * Always allocate at least 4 bytes. In CUB 1.4.1, sorting N < the tile size (which I believe is a thread block)
         * does not require any temporary storage, and tmp_storage_bytes returns 0. But, d_tmp_storage must be not NULL
         * for the sort to occur on the second pass. C++ standards forbid specifying a pointer to memory that
         * isn't properly allocated / doesn't exist (for example, a pointer to an odd address), so we allocate a small
         * bit of memory as temporary storage that isn't used.
         */
        size_t alloc_size = (tmp_storage_bytes > 0) ? tmp_storage_bytes : 4;
        // unsigned char = 1 B
        ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), alloc_size);
        d_tmp_storage = (void *)d_alloc();

        // perform the sort
        detail::gpu_bvh_morton_sort(d_morton_types.data,
                              d_morton_types_alt.data,
                              d_map_tree_pid.data,
                              d_map_tree_pid_alt.data,
                              d_tmp_storage,
                              tmp_storage_bytes,
                              swap_morton,
                              swap_map,
                              m_pdata->getN() + m_pdata->getNGhosts(),
                              m_n_type_bits);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // we want the sorted data in the real data because the alt is just a tmp holder
    if (swap_morton)
        {
        m_morton_types.swap(m_morton_types_alt);
        }

    if (swap_map)
        {
        m_map_tree_pid.swap(m_map_tree_pid_alt);
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * \post Leafs are constructed for adjacent groupings of particles.
 * \note Call after sortMortonCodes(), but before genTreeHierarchy().
 */
template<class BVNode, class Shape, class IntHPMC>
void BVHGPU<BVNode, Shape, IntHPMC>::mergeLeafParticles()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Leaf merge");

    // particle position data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_num_per_type(m_num_per_type, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_type_head(m_type_head, access_location::device, access_mode::read);

    // leaf particle data
    ArrayHandle<uint64_t> d_morton_types(m_morton_types, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_map_tree_pid(m_map_tree_pid, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_leaf_offset(m_leaf_offset, access_location::device, access_mode::read);

    // tree bounding volumes and reduced morton codes to overwrite
    ArrayHandle<BVNode> d_tree_nodes(m_tree_nodes, access_location::device, access_mode::overwrite);
    ArrayHandle<uint32_t> d_morton_codes_red(m_morton_codes_red, access_location::device, access_mode::overwrite);
    ArrayHandle<uint2> d_tree_parent_sib(m_tree_parent_sib, access_location::device, access_mode::overwrite);

    auto d_params = getDeviceParams();

    detail::hpmc_bvh_shapes_args_t args(d_morton_codes_red.data,
                              d_tree_parent_sib.data,
                              d_morton_types.data,
                              d_pos.data,
                              d_num_per_type.data,
                              m_pdata->getNTypes(),
                              d_map_tree_pid.data,
                              d_leaf_offset.data,
                              d_type_head.data,
                              m_pdata->getN() + m_pdata->getNGhosts(),
                              m_n_leaf,
                              m_particles_per_leaf,
                              m_tuner_merge->getParam(),
                              m_stream);

    m_tuner_merge->begin();
    detail::gpu_bvh_merge_shapes<Shape, BVNode>(args,
                                                d_tree_nodes.data,
                                                d_params);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_merge->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * \post Parent-child-sibling relationships are established between nodes.
 * \note This function should always be called alongside bubbleBoundingVolumes to generate a complete hierarchy.
 *       genTreeHierarchy saves only the left children of the nodes for downward traversal because bubbleBoundingVolumes
 *       saves the right child as a rope to complete the edge graph.
 * \note Call after mergeLeafParticles(), but before bubbleBoundingVolumes().
 */
template<class BVNode, class Shape, class IntHPMC>
void BVHGPU<BVNode, Shape, IntHPMC>::genTreeHierarchy()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Hierarchy");

    // don't bother to process if there are no internal nodes
    if (!m_n_internal)
        return;

    ArrayHandle<uint2> d_tree_parent_sib(m_tree_parent_sib, access_location::device, access_mode::overwrite);

    ArrayHandle<uint32_t> d_morton_codes_red(m_morton_codes_red, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_num_per_type(m_num_per_type, access_location::device, access_mode::read);

    m_tuner_hierarchy->begin();
    detail::gpu_bvh_gen_hierarchy(d_tree_parent_sib.data,
                            d_morton_codes_red.data,
                            d_num_per_type.data,
                            m_pdata->getNTypes(),
                            m_n_leaf,
                            m_n_internal,
                            m_particles_per_leaf,
                            m_tuner_hierarchy->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_hierarchy->end();
    if (m_prof) m_prof->pop(m_exec_conf);
    }

//! walk up the tree from the leaves, and assign stackless ropes for traversal, and conservative BoundingVolumes
/*!
 * \post Conservative bounding volumes are assigned to all internal nodes, and stackless "ropes" for downward traversal are
 *       defined between nodes.
 * \note Call after genTreeHierarchy()
 */
template<class BVNode, class Shape, class IntHPMC>
void BVHGPU<BVNode, Shape, IntHPMC>::bubbleBoundingVolumes()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Bubble");
    ArrayHandle<unsigned int> d_node_locks(m_node_locks, access_location::device, access_mode::overwrite);
    ArrayHandle<BVNode> d_tree_nodes(m_tree_nodes, access_location::device, access_mode::readwrite);

    ArrayHandle<uint2> d_tree_parent_sib(m_tree_parent_sib, access_location::device, access_mode::read);

    m_tuner_bubble->begin();
    detail::gpu_bvh_bubble_bounding_volumes<BVNode>(d_node_locks.data,
                           d_tree_nodes.data,
                           d_tree_parent_sib.data,
                           m_pdata->getNTypes(),
                           m_n_leaf,
                           m_n_internal,
                           m_tuner_bubble->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_bubble->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

template<class BVNode, class Shape, class IntHPMC>
void BVHGPU<BVNode, Shape, IntHPMC>::moveLeafParticles()
    {
    if (m_prof) m_prof->push(m_exec_conf,"move");
    ArrayHandle<Scalar4> d_leaf_xyzf(m_leaf_xyzf, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar2> d_leaf_db(m_leaf_db, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_map_tree_pid(m_map_tree_pid, access_location::device, access_mode::read);

    m_tuner_move->begin();
    detail::gpu_bvh_move_particles(d_leaf_xyzf.data,
                             d_leaf_db.data,
                             d_pos.data,
                             d_diameter.data,
                             d_body.data,
                             d_map_tree_pid.data,
                             m_pdata->getN() + m_pdata->getNGhosts(),
                             m_tuner_move->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_move->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

template <class BVNode, class Shape = detail::EmptyShape, class IntHPMC = std::tuple<> >
void export_BVHGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< BVHGPU<BVNode, Shape, IntHPMC>, std::shared_ptr< BVHGPU<BVNode, Shape, IntHPMC> > >(
        m, name.c_str(), pybind11::base<Compute>())
        .def( pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<IntHPMC> >())
        .def( pybind11::init< std::shared_ptr<SystemDefinition> >())
    ;
    }

} // end namespace hpmc

#endif //_BVHGPU_H__
#endif // ENABLE_CUDA
