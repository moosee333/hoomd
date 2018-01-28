// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


/*! \file BVHGPU.cu
    \brief Defines GPU kernel functions for BVH generation on the GPU
*/

#include "OBB.h"

#pragma once

namespace hpmc
{

namespace detail
{

#define BVH_GPU_INVALID_NODE 0xffffffff   //!< Sentinel for an invalid node
#define MORTON_TYPE_MASK_64 0x000000003fffffffu //!< 64 bit mask to turn morton code-type back to morton code

//! A parameter structure for gpu_bvh_merge_shapes
struct hpmc_bvh_shapes_args_t
    {
    hpmc_bvh_shapes_args_t(
        uint32_t *_d_morton_codes_red,
        uint2 *_d_tree_parent_sib,
        const uint64_t *_d_morton_types,
        const Scalar4 *_d_pos,
        const unsigned int *_d_num_per_type,
        const unsigned int _ntypes,
        const unsigned int *_d_map_tree_pid,
        const unsigned int *_d_leaf_offset,
        const unsigned int *_d_type_head,
        const unsigned int _Ntot,
        const unsigned int _nleafs,
        const unsigned int _nparticles_per_leaf,
        const unsigned int _block_size,
        const cudaStream_t _stream) :
        d_morton_codes_red(_d_morton_codes_red),
        d_tree_parent_sib(_d_tree_parent_sib),
        d_morton_types(_d_morton_types),
        d_pos(_d_pos),
        d_num_per_type(_d_num_per_type),
        ntypes(_ntypes),
        d_leaf_offset(_d_leaf_offset),
        d_type_head(_d_type_head),
        Ntot(_Ntot),
        nleafs(_nleafs),
        nparticles_per_leaf(_nparticles_per_leaf),
        block_size(_block_size),
        stream(_stream)
        { }

    uint32_t *d_morton_codes_red;            //!< The Morton codes corresponding to the merged leafs
    uint2 *d_tree_parent_sib;                //!< Parent and sibling indexes for all nodes
    const uint64_t *d_morton_types;          //!< Morton-code type keys for all particles
    const Scalar4 *d_pos;                    //!< Particle positions and types
    const unsigned int *d_num_per_type;      //!< Number of particles per type
    const unsigned int ntypes;               //!< Number of particle types
    const unsigned int *d_map_tree_pid;      //!< Sorted particle order (maps local index to ParticleData index
    const unsigned int *d_leaf_offset;       //!< Amount to subtract from the expected leaf starting index to make an array with no holes by type
    const unsigned int *d_type_head;         //!< Index to first type and leaf ordered particles by type
    const unsigned int Ntot;                 //!< Total number of keys to sort
    const unsigned int nleafs;               //!< Number of leaf nodes
    const unsigned int nparticles_per_leaf;  //!< Number of partices per leaf node
    const unsigned int block_size;           //!< Block size for kernel launch
    const cudaStream_t stream;               //!< The CUDA stream
    };

//! Definition of bounding volume hierarchies (BVHs)

//! OBBs
struct OBBNodeGPU
    {
    typedef OBB bounding_volume_type;

    OBB bounding_volume;//!< The bounding volume
    int rope;  //!< The 'rope' to the next node on the right
    int np_child_masked; //!< Either left child/sibling, or number of particles (for leaf nodes)
    };


// include fixed width integer types uint32_t and uint64_t
#include <stdint.h>

//! Kernel driver to generate morton code-type keys for particles and reorder by type
cudaError_t gpu_bvh_morton_types(uint64_t *d_morton_types,
                                   unsigned int *d_map_tree_pid,
                                   int *d_morton_conditions,
                                   const Scalar4 *d_pos,
                                   const unsigned int N,
                                   const unsigned int nghosts,
                                   const BoxDim& box,
                                   const Scalar3 ghost_width,
                                   const unsigned int block_size);

//! Wrapper to CUB sort for morton codes
cudaError_t gpu_bvh_morton_sort(uint64_t *d_morton_types,
                                  uint64_t *d_morton_types_alt,
                                  unsigned int *d_map_tree_pid,
                                  unsigned int *d_map_tree_pid_alt,
                                  void *d_tmp_storage,
                                  size_t &tmp_storage_bytes,
                                  bool &swap_morton,
                                  bool &swap_map,
                                  const unsigned int Ntot,
                                  const unsigned int n_type_bits);

//! Kernel driver functions
template<class Shape, class BVNode>
cudaError_t gpu_bvh_merge_shapes(const hpmc_bvh_shapes_args_t& args,
                                 BVNode *d_tree_nodes,
                                 const typename Shape::param_type *d_params);

template<class BVNode>
cudaError_t gpu_bvh_bubble_bounding_volumes(unsigned int *d_node_locks,
                                   BVNode *d_tree_nodes,
                                   const uint2 *d_tree_parent_sib,
                                   const unsigned int ntypes,
                                   const unsigned int nleafs,
                                   const unsigned int ninternal,
                                   const unsigned int block_size);

cudaError_t gpu_bvh_gen_hierarchy(uint2 *d_tree_parent_sib,
                                    const uint32_t *d_morton_codes,
                                    const unsigned int *d_num_per_type,
                                    const unsigned int ntypes,
                                    const unsigned int nleafs,
                                    const unsigned int ninternal,
                                    const unsigned int nparticles_per_leaf,
                                    const unsigned int block_size);

//! Kernel driver to rearrange particle data into leaf order
cudaError_t gpu_bvh_move_particles(Scalar4 *d_leaf_xyzf,
                                     Scalar2 *d_leaf_db,
                                     const Scalar4 *d_pos,
                                     const Scalar *d_diameter,
                                     const unsigned int *d_body,
                                     const unsigned int *d_map_tree_pid,
                                     const unsigned int N,
                                     const unsigned int block_size);

//! Kernel driver to initialize counting for types and nodes
cudaError_t gpu_bvh_init_count(unsigned int *d_type_head,
                               const Scalar4 *d_pos,
                               const unsigned int *d_map_tree_pid,
                               const unsigned int N,
                               const unsigned int ntypes,
                               const unsigned int block_size);

#ifdef NVCC
//! Kernel to merge adjacent codes into leaf nodes, and construct the leaf node bounding volume from the shapes
/*!
 * \param d_bvhs Flat array holding all BVHBs for the tree
 * \param d_morton_codes_red The Morton codes corresponding to the merged leafs
 * \param d_tree_parent_sib Parent and sibling indexes for all nodes
 * \param d_morton_types Morton-code type keys for all particles
 * \param d_pos Particle positions
 * \param d_num_per_type Number of particles per type
 * \param ntypes Number of particle types
 * \param d_map_tree_pid Sorted particle order (maps local index to ParticleData index)
 * \param d_leaf_offset Amount to subtract from the expected leaf starting index to make an array with no holes by type
 * \param d_type_head Index to first type and leaf ordered particles by type
 * \param Ntot Total number of keys to sort
 * \param nleafs Number of leaf nodes
 *
 * \b Implementation
 * One thread per leaf is called, and is responsible for merging NLIST_GPU_PARTICLES_PER_LEAF into an AABB. Each thread
 * first determines what type of leaf particle it is operating on by calculating and iterating on the number of leafs
 * of each type. Then, the starting index is determined by subtracting d_leaf_offset[type] from the starting index that
 * would be set in a nleaf x NLIST_GPU_PARTICLES_PER_LEAF array. The reason for this complexity is that the leaf particle
 * array is not permitted to have any "holes" in it for faster traversal. The AABB is merged from the particle
 * positions, and a Morton code is assigned to this AABB for determining tree hierarchy based on the Morton code of
 * the first particle in the leaf. Although this does not necessarily generate the best ordering along the Z order curve
 * for the newly merged leafs, it does guarantee that the leaf Morton codes are still in lexicographic ordering.
 *
 * AABBs are stored as two Scalar4s in a flat array. The first three coordinates of each Scalar4 correspond to the upper
 * and lower bounds of the AABB. The last value of the upper AABB will hold a "rope" for traversing the tree (see
 * gpu_nlist_bubble_aabbs_kernel), while the last value of the lower AABB holds the number of particles for a leaf node,
 * or the left child for an internal node. This is determined by setting a bit to mark this value as a rope or as child.
 */
template<class BVNode, class Shape>
__global__ void gpu_bvh_merge_shapes_kernel(BVNode *d_tree_nodes,
                                             uint32_t *d_morton_codes_red,
                                             uint2 *d_tree_parent_sib,
                                             const uint64_t *d_morton_types,
                                             const Scalar4 *d_pos,
                                             const unsigned int *d_num_per_type,
                                             const unsigned int ntypes,
                                             const unsigned int *d_map_tree_pid,
                                             const unsigned int *d_leaf_offset,
                                             const unsigned int *d_type_head,
                                             const unsigned int Ntot,
                                             const unsigned int nleafs,
                                             const unsigned int nparticles_per_leaf,
                                             const typename Shape::param_type *d_params)
    {
    extern __shared__ char s_data[];
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);

    if (d_params)
        {
        // copy over parameters one int per thread for fast loads
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
        unsigned int param_size = ntypes*sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)d_params)[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    // leaf index
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per leaf
    if (idx >= nleafs)
        return;

    // get what type of leaf I am
    unsigned int total_bins = 0;
    int leaf_type = -1;
    unsigned int max_idx = Ntot;
    for (unsigned int cur_type=0; leaf_type == -1 && cur_type < ntypes; ++cur_type)
        {
        total_bins += (d_num_per_type[cur_type] + nparticles_per_leaf - 1)/nparticles_per_leaf;

        if (idx < total_bins)
            {
            leaf_type = cur_type;
            for (unsigned int next_type=cur_type+1; next_type < ntypes; ++next_type)
                {
                if (d_type_head[next_type])
                    {
                    max_idx = d_type_head[next_type] - 1;
                    break; // quit out of this inner loop once a match is found
                    }
                }
            break; // quit the outer loop
            }
        }

    // get the starting particle index assuming naive leaf structure, and then subtract offset to eliminate "holes"
    unsigned int start_idx = idx*nparticles_per_leaf - d_leaf_offset[leaf_type];
    unsigned int end_idx = (max_idx - start_idx > nparticles_per_leaf) ? start_idx + nparticles_per_leaf : max_idx;

    BVNode node;
    unsigned int npart = end_idx - start_idx;
    node.np_child_masked = npart << 1;
    node.rope = 0; // we have no idea what the skip value is right now

    // compute the bounding valume of type BVNode for these shapes
    computeBoundingVolume<Shape>(node.bounding_volume, d_pos, d_map_tree_pid, start_idx, end_idx, s_params[leaf_type]);

    // store BVH node in global memory
    d_tree_nodes[idx] = node;

    // take logical AND with the 30 bit mask for the morton codes to extract just the morton code
    // no sense swinging around 64 bit integers anymore
    d_morton_codes_red[idx] = (unsigned int)(d_morton_types[start_idx] & MORTON_TYPE_MASK_64);

    // fill the parent/sib relationships as if everything is a single leaf at first, to be overridden by hierarchy gen
    // when this is not the case
    d_tree_parent_sib[idx] = make_uint2(idx, idx << 1);
    }

/*!
 * \param args Parameters to pass to the kernel
 * \param d_bv_nodes Flat array holding all BV's for the tree
 * \param d_params Shape parameter array
 * \returns cudaSuccess on completion
 */
template<class Shape, class BVNode>
cudaError_t gpu_bvh_merge_shapes(const hpmc_bvh_shapes_args_t& args,
                                 BVNode *d_tree_nodes,
                                 const typename Shape::param_type *d_params)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_bvh_merge_shapes_kernel<BVNode, Shape>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(args.block_size,max_block_size);

    if (d_params)
        cudaStreamAttachMemAsync(args.stream, d_params, 0, cudaMemAttachSingle);

    unsigned int shared_bytes = sizeof(typename Shape::param_type)*args.ntypes;
    gpu_bvh_merge_shapes_kernel<BVNode, Shape><<<args.nleafs/run_block_size + 1, run_block_size, shared_bytes, args.stream>>>(
                                                                                d_tree_nodes,
                                                                                args.d_morton_codes_red,
                                                                                args.d_tree_parent_sib,
                                                                                args.d_morton_types,
                                                                                args.d_pos,
                                                                                args.d_num_per_type,
                                                                                args.ntypes,
                                                                                args.d_map_tree_pid,
                                                                                args.d_leaf_offset,
                                                                                args.d_type_head,
                                                                                args.Ntot,
                                                                                args.nleafs,
                                                                                args.nparticles_per_leaf,
                                                                                d_params);
    return cudaSuccess;
    }
#undef MORTON_TYPE_MASK_64
#endif // NVCC

} // end namespace detail
} // end namespace hpmc
