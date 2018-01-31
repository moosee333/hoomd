// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainers: jglaser, mphoward

#include "BVHGPU.cuh"
#include "hoomd/TextureTools.h"

#include "hoomd/extern/cub/cub/cub.cuh"


/*! \file BVHGPU.cu
    \brief Defines GPU kernel code for BVH generation on the GPU
*/

namespace hpmc
{
namespace detail
{

#define MORTON_CODE_BITS   30       //!< Length of the Morton code in bits (k = 10 bits per direction)
#define MORTON_CODE_N_BINS 1024     //!< Number of bins (2^10) per direction to generate 30 bit Morton codes

//!< Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
/*!
 * \param v unsigned integer with 10 bits set
 * \returns The integer expanded with two zeros interleaved between bits
 * http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
 */
__device__ inline unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

//! Assigns the Morton code-type key for each particle on this processor
/*!
 * \param d_morton_types Morton code-type keys per particle
 * \param d_map_tree_pid List to be overwritten with particle ids in ascending order
 * \param d_morton_conditions Flag if a local particle (not a ghost) is detected out of bounds
 * \param d_pos Particle positions
 * \param N Number of local particles
 * \param nghosts Number of ghost particles
 * \param box Local simulation box
 * \param ghost_width Anticipated size of the ghost layer for nonbonded interactions
 *
 * \b Implementation
 * A sorting key is generated for each particle by determining the 30 bit Morton code for each particle, and then
 * concatenating onto the type. Both the Morton code and the type are 32 bit integers, so the concatenation is stored
 * compactly in a 64 bit integer morton_type = (type << 30) + morton code. In this way, a lexicographic sort will
 * sort first by type, then by morton code. The corresponding particle id (thread index) is stashed into d_map_tree_pid
 * to track particles after sorting.
 */
__global__ void gpu_bvh_morton_types_kernel(uint64_t *d_morton_types,
                                              unsigned int *d_map_tree_pid,
                                              int *d_morton_conditions,
                                              const Scalar4 *d_pos,
                                              const unsigned int N,
                                              const unsigned int nghosts,
                                              const BoxDim box,
                                              const Scalar3 ghost_width)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per particle
    if (idx >= N+nghosts)
        return;

    // acquire particle data
    Scalar4 postype = d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const unsigned int type = __scalar_as_int(postype.w);

    // get position in simulation box
    uchar3 periodic = box.getPeriodic();
    Scalar3 f = box.makeFraction(pos,ghost_width);

    /* check if the particle is inside the unit cell + ghost layer in all dimensions
     * this tolerance is small enough that when we multiply by the morton code bin size, we are still in range
     * we silently ignore ghosts outside of this width, and instead deal with that special case below
     * where extra ghosts are communicated (e.g. for bonded interactions)
     */
    if (((f.x < Scalar(-0.00001) || f.x >= Scalar(1.00001)) ||
        (f.y < Scalar(-0.00001) || f.y >= Scalar(1.00001)) ||
        (f.z < Scalar(-0.00001) || f.z >= Scalar(1.00001))) && idx < N)
        {
        atomicMax(d_morton_conditions,idx+1);
        return;
        }

    // find the bin each particle belongs in
    int ib = (int)(f.x * MORTON_CODE_N_BINS);
    int jb = (int)(f.y * MORTON_CODE_N_BINS);
    int kb = (int)(f.z * MORTON_CODE_N_BINS);

    if (!periodic.x) // ghosts exist and may be past layer width
        {
        // handle special cases where random ghosts are beyond the expected layer
        // by just rounding to the nearest edge
        if (ib < 0)
            {
            ib = 0;
            }
        else if (ib >= MORTON_CODE_N_BINS)
            {
            ib = MORTON_CODE_N_BINS - 1;
            }
        }
    else if (ib == MORTON_CODE_N_BINS) // some particles lie exactly on the edge, floor them to zero
        {
        ib = 0;
        }

    // do as for x in y
    if (!periodic.y)
        {
        if (jb < 0)
            {
            jb = 0;
            }
        else if (jb >= MORTON_CODE_N_BINS)
            {
            jb = MORTON_CODE_N_BINS - 1;
            }
        }
    else if (jb == MORTON_CODE_N_BINS)
        {
        jb = 0;
        }

    // do as for y in z
    if (!periodic.z)
        {
        if (kb < 0)
            {
            kb = 0;
            }
        else if (kb >= MORTON_CODE_N_BINS)
            {
            kb = MORTON_CODE_N_BINS - 1;
            }
        }
    else if (kb == MORTON_CODE_N_BINS)
        {
        kb = 0;
        }

    // inline call to some bit swizzling arithmetic
    unsigned int ii = expandBits((unsigned int)ib);
    unsigned int jj = expandBits((unsigned int)jb);
    unsigned int kk = expandBits((unsigned int)kb);
    unsigned int morton_code = ii * 4 + jj * 2 + kk;

    // save the morton code and corresponding particle index for sorting
    // the morton codes hold both the type and the code to sort by both type and position simultaneously
    d_morton_types[idx] = (((uint64_t)type) << MORTON_CODE_BITS) + (uint64_t)morton_code;
    d_map_tree_pid[idx] = idx;
    }

/*!
 * \param d_morton_types Morton code-type keys per particle
 * \param d_map_tree_pid List to be overwritten with particle ids in ascending order
 * \param d_morton_conditions Flag if a local particle (not a ghost) is detected out of bounds
 * \param d_pos Particle positions
 * \param N Number of local particles
 * \param nghosts Number of ghost particles
 * \param box Local simulation box
 * \param ghost_width Anticipated size of the ghost layer for nonbonded interactions
 * \param block_size Requested thread block size of kernel launch
 *
 * \returns cudaSuccess on completion
 */
cudaError_t gpu_bvh_morton_types(uint64_t *d_morton_types,
                                   unsigned int *d_map_tree_pid,
                                   int *d_morton_conditions,
                                   const Scalar4 *d_pos,
                                   const unsigned int N,
                                   const unsigned int nghosts,
                                   const BoxDim& box,
                                   const Scalar3 ghost_width,
                                   const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_bvh_morton_types_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);

    gpu_bvh_morton_types_kernel<<<(N+nghosts)/run_block_size + 1, run_block_size>>>(d_morton_types,
                                                                                      d_map_tree_pid,
                                                                                      d_morton_conditions,
                                                                                      d_pos,
                                                                                      N,
                                                                                      nghosts,
                                                                                      box,
                                                                                      ghost_width);
    return cudaSuccess;
    }

/*!
 * \param d_morton_types Morton code-type keys per particle
 * \param d_morton_types_alt Auxiliary array of equal size to d_morton_types for double buffered sorting
 * \param d_map_tree_pid List of particle ids
 * \param d_map_tree_pid_alt Auxiliary array of equal size to d_map_tree_pid for double buffered sorting
 * \param d_tmp_storage Temporary storage in device memory
 * \param tmp_storage_bytes Number of bytes allocated for temporary storage
 * \param swap_morton Flag to switch real data from auxiliary array to primary array after sorting
 * \param swap_map Flag to switch real data from auxiliary array to primary array after sorting
 * \param Ntot Total number of keys to sort
 * \param n_type_bits Number of bits to check for particle types
 *
 * \returns cudaSuccess on completion
 *
 * \b Implementation
 * The CUB library is used for device-wide radix sorting. Radix sorting is O(kN) where k is the number of bits to check
 * in an unsigned integer key, and N is the number of keys. We restrict the number of bits checked in the max 64 bit
 * keys by only checking up to the MORTON_CODE_BITS + n_type_bits most significant bit. CUB DeviceRadixSort performs
 * its own tuning at run time.
 *
 * Because CUB requires temporary storage, this function must be called twice. First, when \a d_tmp_storage is NULL,
 * the number of bytes required for temporary storage is saved in \a tmp_storage_bytes. This memory must then be
 * allocated in \a d_tmp_storage. On the second call, the radix sort is performed. Because the radix sort may put the
 * active (sorted) buffer in either slot of the DoubleBuffer, a boolean flag is set in \a swap_morton and \a swap_map
 * for whether these data arrays should be swapped.
 */
cudaError_t gpu_bvh_morton_sort(uint64_t *d_morton_types,
                                  uint64_t *d_morton_types_alt,
                                  unsigned int *d_map_tree_pid,
                                  unsigned int *d_map_tree_pid_alt,
                                  void *d_tmp_storage,
                                  size_t &tmp_storage_bytes,
                                  bool &swap_morton,
                                  bool &swap_map,
                                  const unsigned int Ntot,
                                  const unsigned int n_type_bits)
    {
    // initialize memory as "double buffered"
    cub::DoubleBuffer<uint64_t> d_keys(d_morton_types, d_morton_types_alt);
    cub::DoubleBuffer<unsigned int> d_vals(d_map_tree_pid, d_map_tree_pid_alt);

    // on the first pass, this just sizes the temporary storage
    // on the second pass, it actually does the radix sort
    cub::DeviceRadixSort::SortPairs(d_tmp_storage,
                                    tmp_storage_bytes,
                                    d_keys,
                                    d_vals,
                                    Ntot,
                                    0,
                                    MORTON_CODE_BITS+n_type_bits);

    // we've only done something to the buffers on the second time when temporary storage is allocated
    if (d_tmp_storage != NULL)
        {
        // mark that the gpu arrays should be flipped if the final result is not in the right array
        swap_morton = (d_keys.selector == 1);
        swap_map = (d_vals.selector == 1);
        }

    return cudaSuccess;
    }


//! Computes the longest common prefix between Morton codes
/*!
 * \param d_morton_codes Array of Morton codes
 * \param i First Morton code index
 * \param j Second Morton code index
 * \param min_idx The smallest index considered "in range" (inclusive)
 * \param max_idx The last index considered "in range" (inclusive)
 *
 * \returns number of bits shared between the Morton codes of i and j
 *
 * delta(i,j) is defined as the largest number of bits shared between Morton codes i and j. When the Morton codes are
 * sorted, this implies delta(i',j') >= delta(i,j) for any i',j' in [i,j]. If i and j lie outside
 * of the range of Morton codes corresponding to this tree, then it always returns -1. If the Morton codes for i and j
 * are identical, then the longest prefix of i and j is used as a tie breaker.
 */
__device__ inline int delta(const uint32_t *d_morton_codes,
                            unsigned int i,
                            unsigned int j,
                            int min_idx,
                            int max_idx)
    {
    if (j > max_idx || j < min_idx)
        {
        return -1;
        }

    uint32_t first_code = d_morton_codes[i];
    uint32_t last_code = d_morton_codes[j];

    // if codes match, then use index as tie breaker
    // the number of shared bits is equal to the 32 bits in the integer, plus the number of bits shared between the
    // indexes (offset from the start of the node range to make things simpler)
    if (first_code == last_code)
        {
        return (32 + __clz((i-min_idx) ^ (j-min_idx)));
        }
    else
        {
        return __clz(first_code ^ last_code);
        }
    }


//! Determines the range of Morton codes that a node covers
/*!
 * \param d_morton_codes Array of Morton codes
 * \param min_idx The smallest Morton code index considered "in range" (inclusive)
 * \param max_idx The last Morton code index considered "in range" (inclusive)
 * \param idx Current node (Morton code) index
 *
 * \returns the minimum and maximum leafs covered by this node
 * \note This is a literal implementation of the Karras pseudocode, with no optimizations or refinement.
 *       Tero Karras, "Maximizing parallelism in the construction of BVHs, octrees, and k-d trees",
 *       High Performance Graphics (2012).
 */
__device__ inline uint2 determineRange(const uint32_t *d_morton_codes,
                                       const int min_idx,
                                       const int max_idx,
                                       const int idx)
    {
    int forward_prefix = delta(d_morton_codes, idx, idx+1, min_idx, max_idx);
    int backward_prefix = delta(d_morton_codes, idx, idx-1, min_idx, max_idx);

    // get direction of the range based on sign
    int d = ((forward_prefix - backward_prefix) > 0) ? 1 : -1;

    // get minimum prefix
    int min_prefix = delta(d_morton_codes, idx, idx-d, min_idx, max_idx);

    // get maximum prefix by binary search
    int lmax = 2;
    while( delta(d_morton_codes, idx, idx + d*lmax, min_idx, max_idx) > min_prefix)
        {
        lmax = lmax << 1;
        }

    unsigned int len = 0;
    unsigned int step = lmax;
    do
        {
        step = step >> 1;
        unsigned int new_len = len + step;
        if (delta(d_morton_codes, idx, idx + d*new_len, min_idx, max_idx) > min_prefix)
            len = new_len;
        }
    while (step > 1);

   // order range based on direction
    uint2 range;
    if (d > 0)
        {
        range.x = idx;
        range.y = idx + len;
        }
    else
        {
        range.x = idx - len;
        range.y = idx;
        }
    return range;
    }

//! Finds the split position in Morton codes covered by a range
/*!
 * \param d_morton_codes Array of Morton codes
 * \param first First leaf node in the range
 * \param last Last leaf node in the range
 *
 * \returns the leaf index corresponding to the split in Morton codes
 * See determineRange for original source of algorithm.
 */
__device__ inline unsigned int findSplit(const uint32_t *d_morton_codes,
                                         const unsigned int first,
                                         const unsigned int last)
    {
    uint32_t first_code = d_morton_codes[first];
    uint32_t last_code = d_morton_codes[last];

    // if codes match, then just split evenly
    if (first_code == last_code)
        return (first + last) >> 1;

    // get the length of the common prefix
    int common_prefix = __clz(first_code ^ last_code);

    // assume split starts at first, and begin binary search
    unsigned int split = first;
    unsigned int step = last - first;
    do
        {
        // exponential decrease (is factor of 2 best?)
        step = (step + 1) >> 1;
        unsigned int new_split = split + step;

        // if proposed split lies within range
        if (new_split < last)
            {
            unsigned int split_code = d_morton_codes[new_split];
            int split_prefix = __clz(first_code ^ split_code);

            // if new split shares a longer number of bits, accept it
            if (split_prefix > common_prefix)
                {
                split = new_split;
                }
            }
        }
    while (step > 1);

    return split;
    }


//! Kernel to generate the parent-child-sibling relationships between nodes
/*!
 * \param d_tree_parent_sib Parent and sibling for each node in the tree
 * \param d_morton_codes Morton codes for each leaf node
 * \param d_num_per_type Number of particles per type
 * \param ntypes Number of types
 * \param nleafs Number of leafs
 *
 * \b Implementation
 * One thread is called per internal node in a single kernel launch. Each thread first determines its "local" index
 * as an internal node within a tree based on the number of leafs per tree. The range of leafs covered by the internal
 * node is determined, and then its split position is identified. The split identifies the children of the node as
 * another internal node or as a leaf node.
 *
 * The parent and sibling of each child node is saved. The sibling id is bit shifted so as to use a single bit to encode
 * the sibling as a right child or left child (after shifting, we set the bit to 1 if the sibling is a right child).
 * If the child is a root node, it also saves information for itself (since no other node ever identifies a root as a
 * child node).
 */
__global__ void gpu_bvh_gen_hierarchy_kernel(uint2 *d_tree_parent_sib,
                                               const uint32_t *d_morton_codes,
                                               const unsigned int *d_num_per_type,
                                               const unsigned int ntypes,
                                               const unsigned int nleafs,
                                               const unsigned int ninternal,
                                               const unsigned int nparticles_per_leaf)
    {
    // compute the internal node index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per internal node
    if (idx >= ninternal)
        return;

    // get what type of leaf I am
    unsigned int min_idx = 0; // the "0" of the leaf node array
    unsigned int max_idx = 0; // the "N-1" of the leaf node array

    unsigned int node_idx = idx;
    unsigned int origin = 0;
    unsigned int end = 0;

    unsigned int cur_type=0;
    unsigned int active_types=0;
    for (cur_type=0; cur_type < ntypes; ++cur_type)
        {
        // current min index is the previous max index
        min_idx = max_idx;
        // max index adds the number of internal nodes in this type (nleaf - 1)
        const unsigned int cur_nleaf = (d_num_per_type[cur_type] + nparticles_per_leaf - 1)/nparticles_per_leaf;
        if (cur_nleaf > 0)
            {
            max_idx += cur_nleaf-1;
            ++active_types;
            }

        // we break the loop if we are in range
        if (idx < max_idx)
            {
            // decrement by 1 to get this back into the number we really need
            --active_types;

            // now, we repurpose the min and max index to now correspond to the *leaf* index.
            // the min index is the minimum *leaf* index
            origin = min_idx + active_types;
            end = max_idx + active_types;
            node_idx += active_types;
            break;
            }
        }

    // enact the magical split determining
    uint2 range = determineRange(d_morton_codes, origin, end, node_idx);
    unsigned int first = range.x;
    unsigned int last = range.y;
    unsigned int split = findSplit(d_morton_codes, first, last);

    uint2 children;
    // set the children, shifting ahead by nleafs - cur_type to account for leaf shifting
    // this factor comes out from resetting 0 = N_leaf,i each time, and then remapping this to
    // an internal node
    children.x = (split == first) ? split : (nleafs - active_types + split);
    children.y = ((split + 1) == last) ? (split + 1) : nleafs - active_types + split + 1;

    uint2 parent_sib;
    parent_sib.x = nleafs + idx;

    // encode the sibling as the right child
    parent_sib.y = children.y << 1;
    parent_sib.y |= 1;

    d_tree_parent_sib[children.x] = parent_sib;

    // encode the sibling as the left child
    parent_sib.y = children.x << 1;
    d_tree_parent_sib[children.y] = parent_sib;

    // root is always number "zero", but only it can set its parent / sibling
    // we mark both of these as the root for traversing, since only the root node
    // will be its own sibling
    if (node_idx == origin)
        {
        parent_sib.x = nleafs + idx;
        parent_sib.y = (nleafs + idx) << 1;

        d_tree_parent_sib[nleafs + idx] = parent_sib;
        }
    }

/*!
 * \param d_tree_parent_sib Parent and sibling for each node in the tree
 * \param d_morton_codes Morton codes for each leaf node
 * \param d_num_per_type Number of particles per type
 * \param ntypes Number of types
 * \param nleafs Number of leafs
 * \param block_size Requested thread block size
 *
 * \returns cudaSuccess on completion
 */
cudaError_t gpu_bvh_gen_hierarchy(uint2 *d_tree_parent_sib,
                                    const uint32_t *d_morton_codes,
                                    const unsigned int *d_num_per_type,
                                    const unsigned int ntypes,
                                    const unsigned int nleafs,
                                    const unsigned int ninternal,
                                    const unsigned int nparticles_per_leaf,
                                    const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_bvh_gen_hierarchy_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);

    // one thread per internal node
    gpu_bvh_gen_hierarchy_kernel<<<ninternal/run_block_size + 1, run_block_size>>>(d_tree_parent_sib,
                                                                                     d_morton_codes,
                                                                                     d_num_per_type,
                                                                                     ntypes,
                                                                                     nleafs,
                                                                                     ninternal,
                                                                                     nparticles_per_leaf);
    return cudaSuccess;
    }

//! Kernel to bubble up enclosing bounding volumes to internal nodes from leaf nodes
/*!
 * \param d_node_locks Atomic flags identifying when node has been visited
 * \param d_bv_nodes Bounding volume array for all tree nodes
 * \param d_tree_parent_sib Parent and sibling indexes of each node
 * \param ntypes Number of particle types
 * \param nleafs Number of leaf nodes
 *
 * \b Implementation
 * One thread is called per leaf node. The second thread to reach an internal node processes its two children,
 * which guarantees that no node AABB is prematurely processed. The arrival order at a node is controlled by an atomic
 * thread lock in global memory. This locking could be accelerated by using shared memory whenever a node is being
 * processed by threads in the same block.
 *
 * When processing the node, the thread also walks up the tree to find the "rope" that tells a traverser
 * how to navigate the tree. If a query AABB intersects the current node, then the traverser always moves the the left
 * child of the current node. If the AABB does not intersect, it moves along the "rope" to the next portion of the tree.
 * The "rope" is calculated by walking back up the tree to find the earliest ancestor that is a left child of its
 * parent. The rope then goes to that ancestor's sibling. If the root node is reached, then the rope is set to -1 to
 * indicate traversal should be aborted.
 *
 * This kernel also encodes the left child of a node into the AABB for internal nodes. The thread processing the node
 * checks if it arrived from a left child or right child of the node it is processing, and sets the left child of that
 * parent accordingly. A child is indicated by bit shifting, and setting the first bit to 1.
 */
template<class BVHNode>
__global__ void gpu_bvh_bubble_bounding_volumes_kernel(unsigned int *d_node_locks,
                                              BVHNode *d_tree_nodes,
                                              const uint2 *d_tree_parent_sib,
                                              const unsigned int ntypes,
                                              const unsigned int nleafs)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= nleafs)
        return;

    // okay, first we start from the leaf and set my bounding box
    auto cur_node = d_tree_nodes[idx];

    // zero the counters for internal nodes
    cur_node.rope = 0;
    cur_node.np_child_masked = 0;

    unsigned int cur_node_idx = idx;
    unsigned int lock_key = 0;
    do
        {
        uint2 cur_parent_sib = d_tree_parent_sib[cur_node_idx];
        unsigned int cur_parent = cur_parent_sib.x;

        // if the current sibling is a right child, then the current node is a left child
        bool cur_is_left = (cur_parent_sib.y & 1);

        unsigned int cur_sibling = cur_parent_sib.y >> 1;

        // first we compute the skip for this node always
        // back track up the tree until you find a left child
        // we have a check in place so that we don't stall on the root node
        uint2 backtrack = cur_parent_sib;
        while (!(backtrack.y & 1) && backtrack.x != (backtrack.y >> 1))
            {
            backtrack = d_tree_parent_sib[backtrack.x];
            }
        // then, the skip is to the sibling of that node, or else to quit
        if (backtrack.y & 1)
            {
            d_tree_nodes[cur_node_idx].rope = backtrack.y >> 1;
            }
        else
            {
            d_tree_nodes[cur_node_idx].rope = -1;
            }

        // then, we do an atomicAdd on the lock to see if we need to process the parent AABBs
        // check to make sure the parent is bigger than nleafs, or else the node lock always fails
        // so that we terminate the thread
        lock_key = (cur_parent >= nleafs) ? atomicAdd(d_node_locks + cur_parent - nleafs, 1) : 0;

        // process the node
        if (lock_key == 1)
            {
            // compute the max upper bound
            auto sibling_bv = d_tree_nodes[cur_sibling].bounding_volume;
            cur_node.bounding_volume = merge(sibling_bv, cur_node.bounding_volume);

            // this must always be some internal node, so stash the left child of this node here
            cur_node.np_child_masked = ((cur_is_left ? cur_node_idx : cur_sibling) << 1) | 1;

            // store in global memory
            d_tree_nodes[cur_parent] = cur_node;

            // bump the current node one level
            cur_node_idx = cur_parent;
            }
        }
    while (lock_key == 1);

    }

/*!
 * \param d_node_locks Atomic flags identifying when node has been visited
 * \param d_bounding_volumes AABB array for all tree nodes
 * \param d_tree_parent_sib Parent and sibling indexes of each node
 * \param ntypes Number of particle types
 * \param nleafs Number of leaf nodes
 * \param block_size Requested thread block size
 *
 * \returns cudaSuccess on completion
 */
template<class BVHNode>
cudaError_t gpu_bvh_bubble_bounding_volumes(unsigned int *d_node_locks,
                                   BVHNode *d_tree_nodes,
                                   const uint2 *d_tree_parent_sib,
                                   const unsigned int ntypes,
                                   const unsigned int nleafs,
                                   const unsigned int ninternal,
                                   const unsigned int block_size)
    {
    cudaMemset(d_node_locks, 0, sizeof(unsigned int)*ninternal);

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_bvh_bubble_bounding_volumes_kernel<BVHNode>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);

    gpu_bvh_bubble_bounding_volumes_kernel<BVHNode><<<nleafs/run_block_size + 1, run_block_size>>>(d_node_locks,
                                                                         d_tree_nodes,
                                                                         d_tree_parent_sib,
                                                                         ntypes,
                                                                         nleafs);

    return cudaSuccess;
    }

/*!
 * Construct an initial treelet of n leaves, with given root
 * \param root the index of the root node
 * \param leaves The array which will hold nleaves consecutive leaves
 * \param internal_nodes The array which will hold nleaves-1 internal nodes
 * \param d_tree_parent_sib The parent sibling information per tree node
 * \param d_tree_noes The node array in global memory
 *
 * \tparam n The maximum number treelet leaves
 *
 * \post leaves contains the leaf node indices, internal_nodes the internal nodes
 *
 * \returns the number of actual treelet leaves added
 */
template<unsigned int n, class BVHNode>
__device__ unsigned int formTreelet(unsigned int root,
    unsigned int *leaves,
    unsigned int *internal_nodes,
    const uint2 *d_tree_parent_sib,
    const BVHNode *d_tree_nodes)
    {
    unsigned int active_set[n];
    unsigned int is_active[n];

    // we always start with an internal node and its two children
    unsigned int left_child = d_tree_nodes[root].np_child_masked >> 1;
    unsigned int right_child = d_tree_parent_sib[left_child].y >> 1;

    // current number of internal nodes
    unsigned int n_internal = 0;

    // current number of treelet leaves
    unsigned int n_leaves = 0;

    // store in array of internal nodes
    internal_nodes[n_internal++] = root;

    // reset the active set
    for (unsigned int i = 0; i < n; ++i)
        is_active[i] = 0;

    // curent size of active set
    unsigned int n_active = 0;

    // push left and right child to active set
    active_set[n_active] = left_child;
    is_active[n_active++] = 1;

    active_set[n_active] = right_child;
    is_active[n_active++] = 1;

    // iteratively build the treelet, creating n-1 internal nodes
    while (n_active && n_internal < n-1 )
        {
        // the first empty slot in the active set
        unsigned int insert_pos = UINT_MAX;

        // find bounding volume in active set with largest surface area
        Scalar max_area(-FLT_MAX);
        unsigned int imax;
        for (unsigned int i = 0; i < n; ++i)
            {
            // skip holes in the set
            if (!is_active[i])
                {
                insert_pos = i;
                continue;
                }

            Scalar area = d_tree_nodes[active_set[i]].bounding_volume.getSurfaceArea();
            if (area > max_area)
                {
                max_area = area;
                imax = i;
                }
            }

        unsigned int cur_node_idx = active_set[imax];

        // have we arrived at an actual leaf node already?
        if (d_tree_nodes[cur_node_idx].np_child_masked & 1)
            {
            // no, pop the current node from the active set
            is_active[imax] = 0;
            n_active--;

            // stash it as an internal node for later recycling
            internal_nodes[n_internal++] = cur_node_idx;

            // push its two children

            // left child, replace the current node
            left_child = d_tree_nodes[cur_node_idx].np_child_masked >> 1;
            is_active[imax] = 1;
            active_set[imax] = left_child;
            n_active++;

            // right child, insert at next free position
            right_child = d_tree_parent_sib[left_child].y >> 1;
            is_active[insert_pos] = 1;
            active_set[insert_pos] = right_child;
            n_active++;
            }
        else
            {
            // pop this leaf node from the active set
            is_active[imax] = 0;
            n_active--;

            // write it to the output list and move on
            leaves[n_leaves++] = cur_node_idx;
            }
        }

    // dump the active set into the output list of leaves
    for (unsigned int i = 0; i < n; ++i)
        if (is_active[i])
            leaves[n_leaves++] = active_set[i]; // add it to the output

    return n_leaves;
    }

/*!
 * Computes the Hamming weight, i.e. the number of bits set to one
 *
 *https://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
 */
__device__ int numberOfSetBits(int i)
    {
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    }

/*!
 * Reconstruct the tree topology using the optimal treelet, as determined by a
 * a partition of the its nodes into left and right for every subtree
 *
 * \param nleaves The size of the treelet
 * \param leaves The array that holds nleaves consecutive leaves
 * \param internal_nodes The array that holds nleaves-1 internal nodes
 * \param p_opt_bar The optimal partition as a bitset
 * \param c_opt The optimal subset costs
 * \param d_tree_nodes the tree nodes array, holding child pointers
 * \param d_tree_parent_sib the parent sibling relationship array
 * \param d_tree_cost The cost metric per node
 *
 * \tparam n The maximum number treelet leaves
 *
 * \post d_tree_parent_sib will contain the right parent-sibling information, and
 *       d_tree_nodes will have updated parent-child relationships and skip pointers
 *       Bounding volumes still need to be updated
 */
template<unsigned int n, class BVHNode>
__device__ void reconstructTreelet(
    const unsigned int nleaves,
    const unsigned int *leaves,
    const unsigned int *internal_nodes,
    const unsigned int size,
    const int *p_opt_bar,
    const Scalar *c_opt,
    BVHNode *d_tree_nodes,
    uint2 *d_tree_parent_sib,
    Scalar *d_tree_cost)
    {
    // holds the current active subset
    int active_set[n];

    // the parent (.x) of the node corresponding to the active subset, and a
    // and a flag (.y) indicating whether we are a left node
    uint2 active_parent_sib[n];

    // a pointer into siblings per active set
    unsigned int active_sibling_ptr[n];

    // ==1 if this array element is in use
    unsigned int is_active[n];

    // the currently active subset
    int cur_set = size-1;

    // load the corresponding partition
    int cur_p_bar = p_opt_bar[cur_set];

    // and the cost
    Scalar cur_cost = c_opt[cur_set];

    // current number of internal nodes
    unsigned int n_internal = 0;

    // count the number of subsets traversed (not including the root)
    unsigned int n_subsets = 0;

    // a sibling node per traversed node, in the node_indices set
    unsigned int siblings[2*n-2];

    // an actual index per traversed node
    unsigned int node_indices[2*n-2];

    // load the (internal) root node as active parent
    unsigned int cur_parent = internal_nodes[n_internal++];
    uint2 cur_parent_sib;
    cur_parent_sib.x = cur_parent;

    // assign the cost metric
    d_tree_cost[cur_parent] = cur_cost;

    // reset the active set
    for (unsigned int i = 0; i < n; ++i)
        is_active[i] = 0;

    // curent size of active set
    unsigned int n_active = 0;

    // push the left and the right subsets into the active set, including a pointer to their parent and their
    // sibling identity, and a pointer to the siblings array
    unsigned int left_sibling = 0;
    unsigned int right_sibling = 1;

    active_set[n_active] = cur_p_bar;
    cur_parent_sib.y = 1;
    active_parent_sib[n_active] = cur_parent_sib;
    siblings[n_subsets] = right_sibling;
    active_sibling_ptr[n_active] = n_subsets++;
    is_active[n_active++] = 1;

    active_set[n_active] = cur_p_bar^cur_set;
    cur_parent_sib.y = 0;
    active_parent_sib[n_active] = cur_parent_sib;
    siblings[n_subsets] = left_sibling;
    active_sibling_ptr[n_active] = n_subsets++;
    is_active[n_active++] = 1;

    unsigned int leaves_count = 0;

    // iteratively reconstruct the treelet, creating nleaves-1 internal nodes and nleaves leaf nodes
    while (n_active && n_subsets < 2*nleaves-1 )
        {
        // the first empty slot in the active set
        unsigned int insert_pos = UINT_MAX;

        // find subset among active sets with largest surface area cost
        Scalar max_cost(-FLT_MAX);
        unsigned int imax;
        for (unsigned int i = 0; i < n; ++i)
            {
            // skip holes in the set
            if (!is_active[i])
                {
                insert_pos = i;
                continue;
                }

            Scalar cost = c_opt[active_set[i]];

            if (cost > max_cost)
                {
                max_cost = cost;
                imax = i;
                }
            }

        // the currently active subset which maximizes the cost function
        cur_set = active_set[imax];

        // is this subset a single node, i.e., a leaf?
        if (numberOfSetBits(cur_set) > 1)
            {
            // no, pop the current set from the active list
            is_active[imax] = 0;
            n_active--;

            // create an internal node, recycling from the stash
            unsigned int cur_node_idx = internal_nodes[n_internal++];

            // set the parent and sibling identity of this internal node
            cur_parent_sib = active_parent_sib[imax];
            d_tree_parent_sib[cur_node_idx] = cur_parent_sib;

            // assign tree cost
            d_tree_cost[cur_node_idx] = c_opt[cur_set];

            // set the left child pointer on the parent if this is a left child
            if (cur_parent_sib.y & 1)
                d_tree_nodes[cur_parent_sib.x].np_child_masked = 1 | (cur_node_idx << 1);

            // now that the node index has materialized, fill in the node_indices array
            unsigned int cur_ptr = active_sibling_ptr[imax];
            node_indices[cur_ptr] = cur_node_idx;

            // the internal node will become the next parent
            cur_parent_sib.x = cur_node_idx;

            // the partition corresponding to this subset (== subset of nodes in the left subtree)
            unsigned int cur_p_bar = p_opt_bar[cur_set];

            // the sibling pointers, inside the siblings array
            left_sibling = n_subsets;
            right_sibling = n_subsets + 1;

            // push the left subset
            unsigned int left_subset = cur_p_bar;
            is_active[imax] = 1;
            active_set[imax] = left_subset;
            cur_parent_sib.y = 1;
            active_parent_sib[imax] = cur_parent_sib;
            active_sibling_ptr[imax] = n_subsets;
            siblings[n_subsets++] = right_sibling;
            n_active++;

            // right subset, insert at next free position
            unsigned int right_subset = cur_p_bar^cur_set;
            is_active[insert_pos] = 1;
            active_set[insert_pos] = right_subset;
            cur_parent_sib.y = 0;
            active_parent_sib[insert_pos] = cur_parent_sib;
            active_sibling_ptr[insert_pos] = n_subsets;
            siblings[n_subsets++] = left_sibling;
            n_active++;
            }
        else
            {
            // pop this subset from the active list
            is_active[imax] = 0;
            n_active--;

            // get the corresponding leaf node
            unsigned int i = 0;
            unsigned int j = cur_set;
            while (j >> 1)
                {
                j >>= 1;
                i++;
                }
            unsigned int cur_node_idx = leaves[i];

            // fill in node index for later reconnecting siblings
            node_indices[active_sibling_ptr[imax]] = cur_node_idx;

            // set the parent and sibling identity of this internal node
            cur_parent_sib = active_parent_sib[imax];
            d_tree_parent_sib[cur_node_idx] = cur_parent_sib;

            // assign leave node cost
            d_tree_cost[cur_node_idx] = c_opt[cur_set];

            // set the left child pointer on the parent if this is a left child
            if (cur_parent_sib.y & 1)
                d_tree_nodes[cur_parent_sib.x].np_child_masked = 1 | (cur_node_idx << 1);

            leaves_count++;
            }
        }

    // all nodes have been processed, connect the siblings (we'll save updateing ropes for postprocessing)
    for (unsigned int i = 0; i < n_subsets; ++i)
        {
        d_tree_parent_sib[node_indices[i]].y |= (node_indices[siblings[i]] << 1);
        }
    }

/*!
 * Update the bounding volumes of all internal node of the treelet
 *
 * \param nleaves The size of the treelet
 * \param internal_nodes Array of internal treelet nodes
 * \param d_tree_nodes The nodes of the tree
 * \param d_tree_parent_sib the parent sibling relationship array
 * \tparam The type of BVH node
 *
 * \pre the tree topologye is current, so reconstructTreelet() has been called prior to this funciton
 * \post d_tree_nodes will contain updated bounding volumes
 *
 * Iterate over the internal nodes, merging their children's bounding volumes into the current node's volume,
 * and repeat until information has propagated to the root
 */
template<class BVHNode>
__device__ void propagateTreeletBoundingVolumes(
    const unsigned int nleaves,
    const unsigned int *internal_nodes,
    BVHNode *d_tree_nodes,
    const uint2 *d_tree_parent_sib)
    {
    // the longest possible propagation distance is when all internal nodes are arranged in a chain,
    // the path has length nleaves-1
    for (unsigned int i = 0; i < nleaves-1; ++i)
        {
        // iterate over all internal nodes
        for (unsigned int j = 0; j < nleaves-1; ++j)
            {
            unsigned int cur_node_idx = internal_nodes[j];

            unsigned int left_child = d_tree_nodes[cur_node_idx].np_child_masked >> 1;
            unsigned int right_child = d_tree_parent_sib[left_child].y >> 1;

            typename BVHNode::bounding_volume_type new_bv = merge(d_tree_nodes[left_child].bounding_volume,
                                                              d_tree_nodes[right_child].bounding_volume);

            // set this node's bounding volume
            d_tree_nodes[cur_node_idx].bounding_volume = new_bv;
            }
        }
    }

//! Kernel to optimize subtrees (treelets), employing a bottom-up traversal
/*!
 * \param d_node_locks Atomic flags identifying when node has been visited
 * \param d_bv_nodes Bounding volume array for all tree nodes
 * \param d_tree_parent_sib Parent and sibling indexes of each node
 * \param ntypes Number of particle types
 * \param nleafs Number of leaf nodes
 * \param C_i cost of traversing an internal node, per unit area of the bounding volume divided by root node area
 * \param C_l cost of traversing a leaf node, per unit area of the bounding volume divided by root node area
 * \param C_t cost of checking a primitive or particle contained in the leaf node, per primitive and unit area over root node area
 *
 * \tparam n the maximum treelet size (number of leaves)
 *
 * \b Implementation
 *
 * During the bottom-up traversal, which proceeds using atomic lock similar to the gpu_bvh_bubble_bounding_volumes_kernel(),
 * we first construct treelets up to n leaves and n-1 internal nodes. The construction is done iteratively by starting from
 * visited node and descending n-2 times. Then we do a brute-force search over all possible rearrangements of the treelet,
 * using some bitwise arithmetics and dynamic programming, to find the optimal arrangment that minimizes the surface area heuristic.
 * Finally we reconnect the nodes of the treelet.
 */
template<class BVHNode, unsigned int n>
__global__ void gpu_bvh_optimize_treelets_kernel(unsigned int *d_node_locks,
                                              BVHNode *d_tree_nodes,
                                              uint2 *d_tree_parent_sib,
                                              const unsigned int ntypes,
                                              const unsigned int nleafs,
                                              const Scalar C_i,
                                              const Scalar C_l,
                                              const Scalar C_t,
                                              Scalar *d_tree_cost)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= nleafs)
        return;

    // bottom-up traversal in analogy to the gpu_bvh_bubble_bounding_volumes_kernel()

    unsigned int cur_node_idx = idx;
    unsigned int lock_key = 0;
    do
        {
        uint2 cur_parent_sib = d_tree_parent_sib[cur_node_idx];
        unsigned int cur_parent = cur_parent_sib.x;

        // have we hit the tree root?
        bool is_root = cur_parent == cur_node_idx;

        // then, we do an atomicAdd on the lock to see if we need to process the parent nodes
        // check to make sure the parent is bigger than nleafs, or else the node lock always fails
        // so that we terminate the thread
        lock_key = (!is_root && cur_parent >= nleafs) ? atomicAdd(d_node_locks + cur_parent - nleafs, 1) : 0;

        // process the node
        if (lock_key == 1)
            {
            // the current parent becomes the treelet root
            cur_node_idx = cur_parent;

            // now we wish to find the optimal arrangment of the subtree formed by this node
            // and its immediate descendants, which we will search by brute-force

            // leaves of the initial treelet
            unsigned int leaves[n];

            // and its internal nodes
            unsigned int internal_nodes[n];

            // construct the initial treelet, computing the surface area of leaf nodes to be addded
            unsigned int nleaves = formTreelet<n>(cur_node_idx, leaves, internal_nodes, d_tree_parent_sib, d_tree_nodes);

            // now we have nleaves consecutive leaves stored in the leaves array
            // find the optimal partitioning according to the surface area heuristic

            // the maximum number of possible sets of leaves
            const unsigned int max_size = 1 << n;

            // the actual number of subsets
            unsigned int size = 1 << nleaves;

            // the area of the union of bounding volumes
            Scalar areas[max_size];

            // the optimal query cost
            Scalar c_opt[max_size];

            // the optimal partitioning, a bit indicates if the corresponding
            // leaf is part of the left subtree
            int p_opt_bar[max_size];

            // iterate over subsets of leaves
            // _bar indicates bitset
            for (int s_bar = 1; s_bar < size; ++s_bar)
                {
                typename BVHNode::bounding_volume_type bv;
                bool init = true;
                for (unsigned int i = 0; i < nleaves; ++i)
                    {
                    unsigned int cur_leaf = leaves[i];
                    // if this leaf is in the subset
                    if (s_bar & (1 << i))
                        {
                        if (init)
                            {
                            // initialize with leave bounding volume
                            bv = d_tree_nodes[cur_leaf].bounding_volume;
                            init = false;
                            }
                        else
                            {
                            // merge with current bounding volume
                            bv = merge(bv, d_tree_nodes[cur_leaf].bounding_volume);
                            }
                        }
                    }

                // store the area of the union in local memory
                areas[s_bar] = bv.getSurfaceArea();
                }

            // initialize leaf node costs
            for (unsigned int i = 0; i < nleaves; ++i)
                {
                // is this treelet leaf an actual leaf node?
                unsigned int leaf = leaves[i];
                unsigned int np_child_masked = d_tree_nodes[leaf].np_child_masked;
                bool is_leaf = ! (np_child_masked & 1);
                Scalar cost;
                if (is_leaf)
                    {
                    unsigned int npart = np_child_masked >> 1;
                    Scalar area = d_tree_nodes[leaf].bounding_volume.getSurfaceArea();

                    // compute cost metric and store in local memory
                    cost = area*(C_l + npart*C_t);
                    }
                else
                    {
                    // initialize with cost previously computed in bottom-up traversal
                    cost = d_tree_cost[leaf];
                    }

                c_opt[1 << i] = cost;
                }

            // iterate over size of subset
            for (unsigned int k = 2; k <= nleaves; ++k)
                {
                // iterate over subsets of size k
                for (int s_bar = 1; s_bar < size; ++s_bar)
                    {
                    if (numberOfSetBits(s_bar) != k)
                        continue;

                    // Try each way of partitioning the leaves using bit-shuffling arithmetics

                    // initialize the optimal cost for the current subset
                    Scalar c(FLT_MAX);

                    // initialize the bitset for the current partition
                    int p_bar = 0;

                    // Auxillary variables of Algorithm 3 in Karras and Alia 2013
                    int delta_bar = (s_bar - 1) & s_bar;
                    int cur_p_bar = (-delta_bar) & s_bar;

                    do {
                        Scalar cur_c = c_opt[cur_p_bar] + c_opt[s_bar^cur_p_bar];
                        if (cur_c < c)
                            {
                            c = cur_c;
                            p_bar = cur_p_bar;
                            }
                        cur_p_bar = (cur_p_bar - delta_bar) & s_bar;
                        } while (cur_p_bar);

                    #if 0
                    // get number of particles stored in the leaf nodes of this subset
                    unsigned int total_num_particles = 0;
                    for (unsigned int i = 0; i < nleaves; ++i)
                        {
                        if (s_bar & (1 << i))
                            {
                            // is this treelet leaf an actual leaf node?
                            unsigned int leaf = leaves[i];
                            unsigned int np_child_masked = d_tree_nodes[leaf].np_child_masked;
                            bool is_leaf = ! (np_child_masked & 1);
                            if (is_leaf)
                                total_num_particles += np_child_masked >> 1;
                            }
                        }
                    #endif

                    // compute the final surface area heuristic
//                    c_opt[s_bar] = min(C_i*areas[s_bar] + c,C_t*areas[s_bar]*total_num_particles);
                    c_opt[s_bar] = C_i*areas[s_bar] + c;
                    p_opt_bar[s_bar] = p_bar;
                    } // end loop over subsets
                } // end loop over size of subset

            // p_opt[size-1] contains the optimal partition of all nleaves leaves,
            // and c_opt[size-1] the associated cost

            // update the tree topology using the optimal treelet
            reconstructTreelet<n>(
                nleaves,
                leaves,
                internal_nodes,
                size,
                p_opt_bar,
                c_opt,
                d_tree_nodes,
                d_tree_parent_sib,
                d_tree_cost);

            // update the bounding volumes of the treelet
            propagateTreeletBoundingVolumes(
                nleaves,
                internal_nodes,
                d_tree_nodes,
                d_tree_parent_sib);

            // store cost for further accumulation
            d_tree_cost[cur_node_idx] = c_opt[size-1];
            }
        }
    while (lock_key == 1);

    }

/*!
 * \param d_node_locks Atomic flags identifying when node has been visited
 * \param d_bounding_volumes AABB array for all tree nodes
 * \param d_tree_parent_sib Parent and sibling indexes of each node
 * \param ntypes Number of particle types
 * \param nleafs Number of leaf nodes
 * \param n Treelet size (Number of leaves)
 * \param C_i cost of traversing an internal node, per unit area of the bounding volume divided by root node area
 * \param C_l cost of traversing a leaf node, per unit area of the bounding volume divided by root node area
 * \param C_t cost of checking a primitive or particle contained in the leaf node, per primitive and unit area over root node area
 * \param the accumulated SAH cost per tree node
 * \param block_size Requested thread block size
 *
 * \returns cudaSuccess on completion
 */
template<class BVHNode>
cudaError_t gpu_bvh_optimize_treelets(unsigned int *d_node_locks,
                                   BVHNode *d_tree_nodes,
                                   uint2 *d_tree_parent_sib,
                                   const unsigned int ntypes,
                                   const unsigned int nleafs,
                                   const unsigned int ninternal,
                                   const Scalar C_i,
                                   const Scalar C_l,
                                   const Scalar C_t,
                                   const unsigned int n,
                                   Scalar *d_tree_cost,
                                   const unsigned int block_size)
    {
    cudaMemset(d_node_locks, 0, sizeof(unsigned int)*ninternal);

    if (n == 5)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void *)gpu_bvh_optimize_treelets_kernel<BVHNode,5>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        int run_block_size = min(block_size,max_block_size);

        gpu_bvh_optimize_treelets_kernel<BVHNode,5><<<nleafs/run_block_size + 1, run_block_size>>>(d_node_locks,
                                                                             d_tree_nodes,
                                                                             d_tree_parent_sib,
                                                                             ntypes,
                                                                             nleafs,
                                                                             C_i,
                                                                             C_l,
                                                                             C_t,
                                                                             d_tree_cost);
        }
    else if (n == 7)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void *)gpu_bvh_optimize_treelets_kernel<BVHNode,7>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        int run_block_size = min(block_size,max_block_size);

        gpu_bvh_optimize_treelets_kernel<BVHNode,7><<<nleafs/run_block_size + 1, run_block_size>>>(d_node_locks,
                                                                             d_tree_nodes,
                                                                             d_tree_parent_sib,
                                                                             ntypes,
                                                                             nleafs,
                                                                             C_i,
                                                                             C_l,
                                                                             C_t,
                                                                             d_tree_cost);
        }
    else if (n == 9)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void *)gpu_bvh_optimize_treelets_kernel<BVHNode,9>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        int run_block_size = min(block_size,max_block_size);

        gpu_bvh_optimize_treelets_kernel<BVHNode,9><<<nleafs/run_block_size + 1, run_block_size>>>(d_node_locks,
                                                                             d_tree_nodes,
                                                                             d_tree_parent_sib,
                                                                             ntypes,
                                                                             nleafs,
                                                                             C_i,
                                                                             C_l,
                                                                             C_t,
                                                                             d_tree_cost);
        }
    else
        {
        throw std::runtime_error("Treelet size unsupported for optimization.");
        }

    return cudaSuccess;
    }


//! Kernel to find divisons between particle types in sorted order
/*!
 * \param d_type_head Index to first type in leaf ordered particles by type
 * \param d_pos Particle positions
 * \param d_map_tree_pid ParticleData indexes corresponding to a leaf particle id
 * \param N Total number of particles on rank (including ghosts)
 *
 * The starting index for each type of particles is the first particle where the left neighbor is not of the same type.
 */
__global__ void gpu_bvh_get_divisions_kernel(unsigned int *d_type_head,
                                               const Scalar4 *d_pos,
                                               const unsigned int *d_map_tree_pid,
                                               const unsigned int N)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per particle
    if (idx >= N)
        return;

    const unsigned int cur_pidx = d_map_tree_pid[idx];
    // get type of the current particle
    const Scalar4 cur_postype = d_pos[cur_pidx];
    const unsigned int cur_type = __scalar_as_int(cur_postype.w);

    // all particles except for the first one should look left
    if (idx > 0)
        {
        const unsigned int left_pidx = d_map_tree_pid[idx - 1];

        // get type of the particle to my left
        const Scalar4 left_postype = d_pos[left_pidx];
        const unsigned int left_type = __scalar_as_int(left_postype.w);

        // if the left has a different type, then this is a type boundary, and the type starts at the current thread index
        if (left_type != cur_type)
            {
            d_type_head[cur_type] = idx + 1; // offset the index +1 so that we can use 0 to mean "none of this found"
            }
        }
    else // the first particle just sets its type to be 1
        {
        d_type_head[cur_type] = 1;
        }
    }

/*!
 * \param d_type_head Index to first type in leaf ordered particles by type
 * \param d_num_per_type Number of particles per type
 * \param d_leaf_offset Offset for reading particles out of leaf order
 * \param d_tree_roots Root node of each tree
 * \param d_pos Particles positions
 * \param d_map_tree_pid ParticleData indexes corresponding to a leaf particle id
 * \param N Total number of particles on rank (including ghosts)
 * \param ntypes Number of types
 * \param block_size Requested thread block size
 *
 * \returns cudaSuccess on completion
 */
cudaError_t gpu_bvh_init_count(unsigned int *d_type_head,
                                 const Scalar4 *d_pos,
                                 const unsigned int *d_map_tree_pid,
                                 const unsigned int N,
                                 const unsigned int ntypes,
                                 const unsigned int block_size)
    {
    // apply the scan
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_bvh_get_divisions_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);

    // zero out the head list
    cudaMemset(d_type_head, 0, sizeof(unsigned int)*ntypes);

    // get the head list divisions
    gpu_bvh_get_divisions_kernel<<<N/run_block_size + 1, run_block_size>>>(d_type_head, d_pos, d_map_tree_pid, N);

    return cudaSuccess;
    }

//! Kernel to rearrange particle data into leaf order for faster traversal
/*!
 * \param d_leaf_xyzf Particle xyz coordinates + particle id in leaf order
 * \param d_leaf_db Particle diameter and body id in leaf order
 * \param d_pos Particle positions
 * \param d_diameter Particle diameters
 * \param d_body Particle body ids
 * \param d_map_tree_pid ParticleData indexes corresponding to a leaf particle id
 * \param Ntot Number of particles owned by this rank
 *
 * \b Implementation
 * One thread per particle is called. Writes are coalesced by writing in leaf order, and reading in a scattered way.
 */
__global__ void gpu_bvh_move_particles_kernel(Scalar4 *d_leaf_xyzf,
                                                Scalar2 *d_leaf_db,
                                                const Scalar4 *d_pos,
                                                const Scalar *d_diameter,
                                                const unsigned int *d_body,
                                                const unsigned int *d_map_tree_pid,
                                                const unsigned int Ntot)
    {
    // get thread index
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per particle
    if (idx >= Ntot)
        return;

    // read and write particle data
    unsigned int p_idx = d_map_tree_pid[idx];
    Scalar4 pos_i = d_pos[p_idx];
    d_leaf_xyzf[idx] = make_scalar4(pos_i.x, pos_i.y, pos_i.z, __int_as_scalar(p_idx));

    Scalar2 db = make_scalar2(d_diameter[p_idx], __int_as_scalar(d_body[p_idx]));
    d_leaf_db[idx] = db;
    }

/*!
 * \param d_leaf_xyzf Particle xyz coordinates + particle id in leaf order
 * \param d_leaf_db Particle diameter and body id in leaf order
 * \param d_pos Particle positions
 * \param d_diameter Particle diameters
 * \param d_body Particle body ids
 * \param d_map_tree_pid ParticleData indexes corresponding to a leaf particle id
 * \param Ntot Number of particles owned by this rank
 * \param block_size Requested thread block size
 *
 * \returns cudaSuccess on completion
 */
cudaError_t gpu_bvh_move_particles(Scalar4 *d_leaf_xyzf,
                                     Scalar2 *d_leaf_db,
                                     const Scalar4 *d_pos,
                                     const Scalar *d_diameter,
                                     const unsigned int *d_body,
                                     const unsigned int *d_map_tree_pid,
                                     const unsigned int Ntot,
                                     const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_bvh_move_particles_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);

    gpu_bvh_move_particles_kernel<<<Ntot/run_block_size + 1, run_block_size>>>(d_leaf_xyzf,
                                                                                 d_leaf_db,
                                                                                 d_pos,
                                                                                 d_diameter,
                                                                                 d_body,
                                                                                 d_map_tree_pid,
                                                                                 Ntot);
    return cudaSuccess;
    }


/*!
 * Template instantiations for various bounding volume types
 */

// AABB
template cudaError_t gpu_bvh_bubble_bounding_volumes<AABBNodeGPU>(unsigned int *d_node_locks,
                                   AABBNodeGPU *d_tree_nodes,
                                   const uint2 *d_tree_parent_sib,
                                   const unsigned int ntypes,
                                   const unsigned int nleafs,
                                   const unsigned int ninternal,
                                   const unsigned int block_size);

template cudaError_t gpu_bvh_merge_shapes<EmptyShape, AABBNodeGPU> (const hpmc_bvh_shapes_args_t& args,
                                 AABBNodeGPU *d_tree_nodes,
                                 const typename EmptyShape::param_type *d_params);

template cudaError_t gpu_bvh_optimize_treelets(unsigned int *d_node_locks,
                                   AABBNodeGPU *d_tree_nodes,
                                   uint2 *d_tree_parent_sib,
                                   const unsigned int ntypes,
                                   const unsigned int nleafs,
                                   const unsigned int ninternal,
                                   const Scalar C_i,
                                   const Scalar C_l,
                                   const Scalar C_t,
                                   const unsigned int n,
                                   Scalar *d_tree_cost,
                                   const unsigned int block_size);

// OBB
template cudaError_t gpu_bvh_bubble_bounding_volumes<OBBNodeGPU>(unsigned int *d_node_locks,
                                   OBBNodeGPU *d_tree_nodes,
                                   const uint2 *d_tree_parent_sib,
                                   const unsigned int ntypes,
                                   const unsigned int nleafs,
                                   const unsigned int ninternal,
                                   const unsigned int block_size);

template cudaError_t gpu_bvh_merge_shapes<EmptyShape, OBBNodeGPU> (const hpmc_bvh_shapes_args_t& args,
                                 OBBNodeGPU *d_tree_nodes,
                                 const typename EmptyShape::param_type *d_params);

template cudaError_t gpu_bvh_optimize_treelets(unsigned int *d_node_locks,
                                   OBBNodeGPU *d_tree_nodes,
                                   uint2 *d_tree_parent_sib,
                                   const unsigned int ntypes,
                                   const unsigned int nleafs,
                                   const unsigned int ninternal,
                                   const Scalar C_i,
                                   const Scalar C_l,
                                   const Scalar C_t,
                                   const unsigned int n,
                                   Scalar *d_tree_cost,
                                   const unsigned int block_size);

} // namespace detail
} // namespace hpmc
