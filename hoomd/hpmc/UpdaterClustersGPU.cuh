// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _UPDATER_CLUSTERS_GPU_CUH_
#define _UPDATER_CLUSTERS_GPU_CUH_

#include "HPMCPrecisionSetup.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/CachedAllocator.h"

#ifdef NVCC
#include "hoomd/TextureTools.h"
#endif

#include "hoomd/AABBTree.h"
#include "hoomd/ManagedArray.h"

#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>

namespace hpmc
{

namespace detail
{

/*! \file UpdaterClustersGPU.cuh
    \brief Declaration and implemtation of CUDA kernels for UpdaterClustersGPU
*/

//! Wraps arguments to gpu_hpmc_clusters
/*! \ingroup hpmc_data_structs */
struct hpmc_clusters_args_t
    {
    //! Construct a pair_args_t
    hpmc_clusters_args_t(
                unsigned int _N,
                unsigned int _ncollisions,
                const Scalar4 *_d_postype,
                const Scalar4 *_d_orientation,
                const int3 *_d_image,
                const unsigned int *_d_tag,
                const unsigned int _num_types,
                const unsigned int _timestep,
                const unsigned int _dim,
                const BoxDim& _box,
                const unsigned int _block_size,
                const unsigned int _stride,
                const unsigned int _group_size,
                const unsigned int _max_n,
                const unsigned int *_d_check_overlaps,
                Index2D _overlap_idx,
                bool _line,
                const Scalar4 *_d_postype_test,
                const Scalar4 *_d_orientation_test,
                const int3 *_d_image_test,
                const unsigned int *_d_tag_test,
                uint2 *_d_overlaps,
                uint3 *_d_collisions,
                unsigned int _max_n_overlaps,
                unsigned int *_d_n_overlaps,
                unsigned int *_d_reject,
                unsigned int _max_n_reject,
                unsigned int *_d_n_reject,
                uint2 *_d_conditions,
                bool _swap,
                unsigned int _type_A,
                unsigned int _type_B,
                const AABBTree& _aabb_tree,
                const ManagedArray<vec3<Scalar> >& _image_list,
                const ManagedArray<int3 >& _image_hkl,
                cudaStream_t _stream,
                const cudaDeviceProp& _devprop,
                const CachedAllocator &_alloc,
                unsigned int _N_old,
                Scalar *_d_begin,
                Scalar *_d_end,
                unsigned int *_d_aabb_idx)
                : N(_N),
                  ncollisions(_N),
                  d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  d_image(_d_image),
                  d_tag(_d_tag),
                  num_types(_num_types),
                  timestep(_timestep),
                  dim(_dim),
                  box(_box),
                  block_size(_block_size),
                  stride(_stride),
                  group_size(_group_size),
                  max_n(_max_n),
                  d_check_overlaps(_d_check_overlaps),
                  overlap_idx(_overlap_idx),
                  line(_line),
                  d_postype_test(_d_postype_test),
                  d_orientation_test(_d_orientation_test),
                  d_image_test(_d_image_test),
                  d_tag_test(_d_tag_test),
                  d_overlaps(_d_overlaps),
                  d_collisions(_d_collisions),
                  max_n_overlaps(_max_n_overlaps),
                  d_n_overlaps(_d_n_overlaps),
                  d_reject(_d_reject),
                  max_n_reject(_max_n_reject),
                  d_n_reject(_d_n_reject),
                  d_conditions(_d_conditions),
                  swap(_swap),
                  type_A(_type_A),
                  type_B(_type_B),
                  aabb_tree(_aabb_tree),
                  image_list(_image_list),
                  image_hkl(_image_hkl),
                  stream(_stream),
                  devprop(_devprop),
                  alloc(_alloc),
                  N_old(_N_old),
                  d_begin(_d_begin),
                  d_end(_d_end),
                  d_aabb_idx(_d_aabb_idx)
        {
        };

    unsigned int N;                   //!< Number of particles to test
    unsigned int ncollisions;         //!< Number of collisions from broad phase
    const Scalar4 *d_postype;         //!< postype array of configuration to test against
    const Scalar4 *d_orientation;     //!< orientation array
    const int3 *d_image;              //!< Particle images
    const unsigned int *d_tag;        //!< Particle tags
    const unsigned int num_types;     //!< Number of particle types
    const unsigned int timestep;      //!< Current time step
    const unsigned int dim;           //!< Number of dimensions
    const BoxDim& box;                //!< Current simulation box
    unsigned int block_size;          //!< Block size to execute for overlap checks
    unsigned int stride;              //!< Number of threads per overlap check
    unsigned int group_size;          //!< Size of the group to execute
    const unsigned int max_n;         //!< Maximum size of pdata arrays
    const unsigned int *d_check_overlaps;   //!< Interaction matrix
    Index2D overlap_idx;              //!< Interaction matrix indexer
    bool line;                        //!< True if line reflection
    const Scalar4 *d_postype_test;    //!< Positions and types to test
    const Scalar4 *d_orientation_test;//!< Orientations to test
    const int3 *d_image_test;         //!< Images of particles to test
    const unsigned int *d_tag_test;   //!< Tags of particles to test
    uint2 *d_overlaps;                //!< List of overlap pairs generated
    uint3 *d_collisions;              //!< List of overlap pairs generated (.x and .y, .x: index of peridoc image)
    unsigned int max_n_overlaps;      //!< Capacity of d_overlaps list
    unsigned int *d_n_overlaps;       //!< Number of particles inserted (return value)
    unsigned int *d_reject;           //!< Particle indices flagged for rejection
    unsigned int max_n_reject;        //!< Capacity of d_reject list
    unsigned int *d_n_reject;         //!< Number of particles flagged
    uint2 *d_conditions;              //!< Flags to indicate overflow
    bool swap;                        //!< If true, swap move
    unsigned int type_A;              //!< Type A of swap pair
    unsigned int type_B;              //!< Type B of swap pair
    const AABBTree& aabb_tree;        //!< AABB tree data structure for overlap checks
    const ManagedArray<vec3<Scalar> >& image_list; //!< Image list for periodic boundary conditions
    const ManagedArray<int3 >& image_hkl; //!< Image list shifts for periodic boundary conditions
    cudaStream_t stream;               //!< Stream for kernel execution
    const cudaDeviceProp& devprop;    //!< CUDA device properties
    const CachedAllocator &alloc;     //!< a caching allocator for thrust
    unsigned int N_old;               //!< Number of particles in configuration to test against
    Scalar *d_begin;                  //!< List of begin coordinates of AABBs along the sweep axis
    Scalar *d_end;                    //!< List of end coordinates of AABBs along the sweep axis
    unsigned int *d_aabb_idx;         //!< AABB indices corresponding to the (sorted) intervals
    };

template< class Shape >
cudaError_t gpu_hpmc_clusters(const hpmc_clusters_args_t &args, const typename Shape::param_type *d_params);

template< class Shape >
cudaError_t gpu_hpmc_clusters_overlaps(const hpmc_clusters_args_t &args, const typename Shape::param_type *d_params);

#ifdef NVGRAPH_AVAILABLE
//! Use nvGRAPH to find strongly connected components
cudaError_t gpu_connected_components(
    const uint2 *d_adj,
    unsigned int N,
    unsigned int n_elements,
    unsigned int *d_components,
    unsigned int &num_components,
    cudaStream_t stream,
    unsigned int max_ites,
    float tol,
    float jump_tol,
    const CachedAllocator& alloc);
#endif

#ifdef NVCC
//! Texture for reading postype
static scalar4_tex_t clusters_postype_tex;
//! Texture for reading orientation
static scalar4_tex_t clusters_orientation_tex;

//! Kernel to find overlaps between different configurations
template< class Shape >
__global__ void gpu_hpmc_clusters_kernel(unsigned int N,
                                     const Scalar4 *d_postype,
                                     const Scalar4 *d_orientation,
                                     const unsigned int *d_tag,
                                     const unsigned int num_types,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const unsigned int *d_check_overlaps,
                                     Index2D overlap_idx,
                                     const typename Shape::param_type *d_params,
                                     uint3 *d_collisions,
                                     unsigned int *d_n_overlaps,
                                     const Scalar4 *d_postype_test,
                                     const Scalar4 *d_orientation_test,
                                     const unsigned int *d_tag_test,
                                     bool line,
                                     bool swap,
                                     const AABBTree aabb_tree,
                                     const ManagedArray<vec3<Scalar> > image_list,
                                     unsigned int max_n_overlaps,
                                     uint2 *d_conditions)
    {
    // determine particle idx
    unsigned int i = (blockIdx.z * gridDim.y + blockIdx.y)*blockDim.y + threadIdx.y;

    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    unsigned int *s_check_overlaps = (unsigned int *) (s_params + num_types);
    unsigned int ntyppairs = overlap_idx.getNumElements();

    // copy over parameters one int per thread for fast loads
        {
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
        unsigned int param_size = num_types*sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)d_params)[cur_offset + tidx];
                }
            }

        for (unsigned int cur_offset = 0; cur_offset < ntyppairs; cur_offset += block_size)
            {
            if (cur_offset + tidx < ntyppairs)
                {
                s_check_overlaps[cur_offset + tidx] = d_check_overlaps[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    if (i >= N)
        return;

    // test particle position, orientation,..
    Scalar4 postype = d_postype_test[i];
    vec3<Scalar> pos_i(postype);
    unsigned int type = __scalar_as_int(postype.w);
    unsigned int tag = d_tag_test[i];

    Shape shape_i(quat<Scalar>(d_orientation_test[i]), s_params[type]);

    unsigned int num_nodes = aabb_tree.getNumNodes();
    detail::AABB aabb_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

    unsigned int n_images = image_list.size();

    // obtain a pointer to the managed memory holding the AABB nodes
    const AABB *aabbs = aabb_tree.getAABBs().get();

    for (unsigned int cur_image = 0; cur_image < n_images; ++cur_image)
        {
        vec3<Scalar> pos_image = pos_i + image_list[cur_image];
        detail::AABB aabb = aabb_local;
        aabb.translate(pos_image);

        for (unsigned int cur_node_idx = 0; cur_node_idx < num_nodes; ++cur_node_idx)
            {
            if (detail::overlap(aabbs[cur_node_idx], aabb))
                {
                if (aabb_tree.isNodeLeaf(cur_node_idx))
                    {
                    for (unsigned int cur_p = threadIdx.x; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p+=blockDim.x)
                        {
                        unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                        unsigned int tag_j = d_tag[j];
                        if (tag_j == tag)
                            continue;

                        Scalar4 postype_j = texFetchScalar4(d_postype, clusters_postype_tex, j);
                        Scalar4 orientation_j = make_scalar4(1,0,0,0);
                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(orientation_j), s_params[typ_j]);
                        if (shape_j.hasOrientation())
                            shape_j.orientation = quat<Scalar>(texFetchScalar4(d_orientation, clusters_orientation_tex, j));

                        vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_image;

                        // check for overlaps
                        OverlapReal rsq = dot(r_ij,r_ij);
                        OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                        if (rsq*OverlapReal(4.0) <= DaDb * DaDb &&
                            s_check_overlaps[overlap_idx(typ_j,type)])
                            {
                            // write to collision list
                            unsigned int typ_j = __scalar_as_int(postype_j.w);
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_image;

                            unsigned int n_overlaps = atomicAdd(d_n_overlaps, 1);
                            if (n_overlaps >= max_n_overlaps)
                                atomicMax(&(*d_conditions).x, n_overlaps+1);
                            else
                                {
                                // write indices, we'll later convert to tags
                                d_collisions[n_overlaps] = make_uint3(i,j, cur_image);
                                }

                            } // end if circumsphere overlap
                        } // end loop over particles in node
                    } // end isLeaf
                } // end AABB overlap
            else
                {
                // skip ahead
                cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                }
            } // end loop over nodes

        } // end loop over images
    }

//! Overlap checking in separate kernel to save on registers and avoid divergences
template< class Shape >
__global__ void gpu_hpmc_clusters_overlaps_kernel(
                                     const Scalar4 *d_postype,
                                     const Scalar4 *d_orientation,
                                     const int3 *d_image,
                                     const unsigned int *d_tag,
                                     const unsigned int num_types,
                                     const typename Shape::param_type *d_params,
                                     unsigned int n_collisions,
                                     const uint3 *d_collisions,
                                     uint2 *d_overlaps,
                                     unsigned int *d_n_overlaps,
                                     unsigned int *d_reject,
                                     unsigned int *d_n_reject,
                                     const Scalar4 *d_postype_test,
                                     const Scalar4 *d_orientation_test,
                                     const int3 *d_image_test,
                                     const unsigned int *d_tag_test,
                                     const ManagedArray<vec3<Scalar> > image_list,
                                     const ManagedArray<int3> image_hkl,
                                     bool line,
                                     bool swap,
                                     unsigned int type_A,
                                     unsigned int type_B,
                                     unsigned int max_n_reject,
                                     uint2 *d_conditions,
                                     unsigned int max_extra_bytes)
    {
    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);

    // copy over parameters one int per thread for fast loads
        {
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
        unsigned int param_size = num_types*sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)d_params)[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    // initialize extra shared mem
    char *s_extra = (char *)&s_params[num_types];

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    __syncthreads();

    // determine idx in collision list
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_collisions)
        return;

    unsigned int i = d_collisions[idx].x;
    unsigned int j = d_collisions[idx].y;
    unsigned int cur_image = d_collisions[idx].z;

    // load (test) particle i
    Scalar4 postype_i = d_postype_test[i];
    vec3<Scalar> pos_i(postype_i);
    vec3<Scalar> pos_image = pos_i + image_list[cur_image];
    unsigned int type_i = __scalar_as_int(postype_i.w);
    unsigned int tag_i = d_tag_test[i];
    int3 img_i = d_image_test[i];

    Shape shape_i(quat<Scalar>(d_orientation_test[i]), s_params[type_i]);

    // load particle j
    Scalar4 postype_j = d_postype[j];
    vec3<Scalar> pos_j(postype_j);
    unsigned int type_j = __scalar_as_int(postype_j.w);
    unsigned int tag_j = d_tag[j];

    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_image;
    Shape shape_j(quat<Scalar>(d_orientation[j]), s_params[type_j]);

    unsigned int err_count = 0;
    if (test_overlap(r_ij, shape_i, shape_j, err_count))
        {
        // write to overlaps list
        unsigned int n_overlaps = atomicAdd(d_n_overlaps, 1);
        d_overlaps[n_overlaps] = make_uint2(tag_i,tag_j);

        if (d_reject)
            {
            bool reject = false;

            if (line && !swap)
                {
                int3 delta_img = -image_hkl[cur_image] + img_i - d_image[j];
                reject = delta_img.x || delta_img.y || delta_img.z;
                }
            else if (swap)
                {
                reject = ((type_i != type_A && type_i != type_B) || (type_j != type_A && type_j != type_B));
                }

            if (reject)
                {
                unsigned int n_reject = atomicAdd(d_n_reject, 1);
                if (n_reject >= max_n_reject)
                    atomicMax(&(*d_conditions).y, n_reject+1);
                else
                    {
                    d_reject[n_reject] = tag_i;
                    }

                n_reject = atomicAdd(d_n_reject, 1);
                if (n_reject >= max_n_reject)
                     atomicMax(&(*d_conditions).y, n_reject+1);
                else
                    {
                    d_reject[n_reject] = tag_j;
                    }
                }
            } // end if (d_reject)
        }
    }

//! Perform sweep and prune along one axis
template<class Shape>
__global__ void gpu_sweep_and_prune_kernel(
    unsigned int N,
    unsigned int Nold,
    const Scalar4 *d_postype,
    const Scalar4 *d_postype_old,
    const unsigned int *d_tag,
    const unsigned int *d_tag_old,
    const typename Shape::param_type *d_params,
    unsigned int num_types,
    const Scalar *d_begin,
    const Scalar *d_end,
    unsigned int *d_aabb_idx,
    Scalar sweep_length,
    bool periodic,
    ManagedArray<vec3<Scalar> > image_list,
    const unsigned int *d_check_overlaps,
    Index2D overlap_idx,
    uint3 *d_collisions,
    unsigned int max_n_overlaps,
    unsigned int *d_n_overlaps,
    uint2 *d_conditions
    )
    {
    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    unsigned int *s_check_overlaps = (unsigned int *)(s_params + num_types);

    // copy over parameters one int per thread for fast loads
        {
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
        unsigned int param_size = num_types*sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)d_params)[cur_offset + tidx];
                }
            }

        unsigned int ntyppairs = overlap_idx.getNumElements();
        for (unsigned int cur_offset = 0; cur_offset < ntyppairs; cur_offset += block_size)
            {
            if (cur_offset + tidx < ntyppairs)
                {
                s_check_overlaps[cur_offset + tidx] = d_check_overlaps[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    // the index of the interval we're colliding
    unsigned int interval_i = blockIdx.x*blockDim.x+threadIdx.x;

    if (interval_i >= N+Nold)
        return;

    // the end coordinate of this interval
    Scalar end_i = d_end[interval_i];

    unsigned int interval_j = interval_i;

    // corresponding particle index
    unsigned int i = d_aabb_idx[interval_i];

    // the coordinates of particle i belonging to our interval
    Scalar4 postype_i = (i < Nold) ? d_postype_old[i] : d_postype[i - Nold];
    unsigned int tag_i = (i < Nold) ? d_tag_old[i] : d_tag[i - Nold];
    unsigned int typ_i = __scalar_as_int(postype_i.w);
    Shape shape_i(quat<Scalar>(), s_params[typ_i]);
    AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

    Scalar image = 0;
    do
        {
        interval_j++;
        if (interval_j == N+Nold)
            {
            if (periodic)
                {
                // wrap around periodic boundaries
                interval_j = 0;
                image+=sweep_length;
                }
            else
                {
                // no more intervals to the right
                break;
                }
            }

        // start coordinate of test interval
        Scalar begin_j = d_begin[interval_j] + image;

        if (begin_j > end_i)
            break; // we're done

        // get this interval's AABB index and AABB
        unsigned int j = d_aabb_idx[interval_j];

        // we're not reporting overlaps in the same configuration
        if ((i < Nold && j < Nold) || (i >= Nold && j >= Nold))
            continue;

        Scalar4 postype_j = (j < Nold) ? d_postype_old[j] : d_postype[j - Nold];
        unsigned int tag_j = (j < Nold) ? d_tag_old[j] : d_tag[j - Nold];
        unsigned int typ_j = __scalar_as_int(postype_j.w);

        if (tag_i == tag_j)
            continue;

        Shape shape_j(quat<Scalar>(), s_params[typ_j]);
        AABB aabb_j = shape_j.getAABB(vec3<Scalar>(postype_j));

        // check for AABB overlap
        if (!s_check_overlaps[overlap_idx(typ_i,typ_j)]) continue;

        // iterate over particle images to detect AABB overlap with periodic boundary conditions
        unsigned int n_images = image_list.size();

        for (unsigned int cur_image = 0; cur_image < n_images; ++cur_image)
            {
            // get the AABB for particle i in the current periodic image
            vec3<Scalar> pos_i_image = vec3<Scalar>(postype_i);

            if (i < Nold)
                pos_i_image -= image_list[cur_image];
            else
                pos_i_image += image_list[cur_image];

            AABB aabb_i = aabb_i_local;
            aabb_i.translate(pos_i_image);

            if (overlap(aabb_i, aabb_j))
                {
                // check further for circumsphere overlap
                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                OverlapReal rsq = dot(r_ij,r_ij);
                OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                if (rsq*OverlapReal(4.0) <= DaDb * DaDb)
                    {
                    // write to collision list
                    unsigned int n_overlaps = atomicAdd(d_n_overlaps, 1);
                    if (n_overlaps >= max_n_overlaps)
                        atomicMax(&(*d_conditions).x, n_overlaps+1);
                    else
                        {
                        // write indices to collision list, we'll later convert to tags

                        // .x index: test configuration
                        // .y index: old configuration
                        // .z index: image vector from test ptl to ptl from old configuration
                        unsigned int idx_test = i < Nold ? j - Nold : i - Nold;
                        unsigned int idx_old = i < Nold ? i : j;
                        d_collisions[n_overlaps] = make_uint3(idx_test,idx_old, cur_image);
                        }
                    }
                } // end if AABB overlap
            } // end loop over images
        } while (true);
    }

template<class Shape>
__global__ void gpu_get_aabb_extents_kernel(
    unsigned int N,
    const Scalar4 *d_postype,
    const typename Shape::param_type *d_params,
    unsigned int num_types,
    Scalar *d_begin,
    Scalar *d_end,
    vec3<Scalar> sweep_direction,
    Scalar sweep_length,
    const BoxDim box,
    unsigned int offs,
    unsigned int *d_aabb_idx)
    {
    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);

    // copy over parameters one int per thread for fast loads
        {
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
        unsigned int param_size = num_types*sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)d_params)[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // get the AABB extents along sweep axis for this particle
    Scalar4 postype = d_postype[idx];
    Shape shape(quat<Scalar>(), s_params[__scalar_as_int(postype.w)]);
    AABB aabb = shape.getAABB(vec3<Scalar>(postype));

    // and write them to global memory

    vec3<Scalar> lower = aabb.getLower();
    vec3<Scalar> upper = aabb.getUpper();

    Scalar begin = dot(lower, sweep_direction);
    Scalar end = dot(upper, sweep_direction);

    Scalar3 f = box.makeFraction(vec_to_scalar3(lower));
    if (box.getPeriodic().x && f.x < 0.0)
        {
        begin += sweep_length;
        end += sweep_length;
        }

    d_begin[idx] = begin;
    d_end[idx] = end;

    // fill index array, to be sorted later
    d_aabb_idx[idx] = offs+idx;
    }

//! Kernel driver for gpu_hpmc_clusters_kernel()
/*! \param args Bundled arguments
    \param d_params Per-type shape parameters
    \returns Error codes generated by any CUDA calls, or cudaSuccess when there is no error

    This templatized method is the kernel driver for parallel update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup hpmc_kernels
*/
template< class Shape >
cudaError_t gpu_hpmc_clusters(const hpmc_clusters_args_t& args, const typename Shape::param_type *d_params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_image);
    assert(args.d_tag);
    assert(args.d_cell_size);

    assert(args.d_postype_test);
    assert(args.d_orientation_test);
    assert(args.d_image_test);
    assert(args.d_tag_test);

    // attach the parameters to the kernel stream so that they are visible
    // when other kernels are called
    for (unsigned int i = 0; i < args.num_types; ++i)
        {
        // attach nested memory regions
        d_params[i].attach_to_stream(args.stream);
        }
    cudaStreamAttachMemAsync(args.stream, d_params, 0, cudaMemAttachSingle);

    #if 1

    // for now, as the sweep direction, choose the x axis
    vec3<Scalar> sweep_vector(args.box.getLatticeVector(0));
    Scalar sweep_length = args.box.getNearestPlaneDistance().x;
    vec3<Scalar> sweep_direction(sweep_vector/sqrt(dot(sweep_vector,sweep_vector)));
    bool periodic = args.box.getPeriodic().x;

    unsigned int shared_bytes_aabb_extents = args.num_types * sizeof(typename Shape::param_type);

    // extract AABB extents for old configuration
    unsigned int block_size_aabb = 256;
    unsigned int n_blocks = args.N_old/block_size_aabb+1;

    gpu_get_aabb_extents_kernel<Shape><<<n_blocks, block_size_aabb, shared_bytes_aabb_extents, args.stream>>>(
        args.N_old,
        args.d_postype,
        d_params,
        args.num_types,
        args.d_begin,
        args.d_end,
        sweep_direction,
        sweep_length,
        args.box,
        0,
        args.d_aabb_idx);

    // append AABB extents for new configuration
    block_size_aabb = 256;
    n_blocks = args.N/block_size_aabb+1;

    gpu_get_aabb_extents_kernel<Shape><<<n_blocks, block_size_aabb, shared_bytes_aabb_extents, args.stream>>>(
        args.N,
        args.d_postype_test,
        d_params,
        args.num_types,
        args.d_begin + args.N_old,
        args.d_end + args.N_old,
        sweep_direction,
        sweep_length,
        args.box,
        args.N_old,
        args.d_aabb_idx + args.N_old);

    // sort the interval ends and indices (==values) by their beginnings (==keys)
    thrust::device_ptr<Scalar> begin(args.d_begin);
    thrust::device_ptr<Scalar> end(args.d_end);
    thrust::device_ptr<unsigned int> aabb_idx(args.d_aabb_idx);

    // combine the end point and the index in one iterator
    auto values_it = thrust::make_zip_iterator(thrust::make_tuple(
        end,
        aabb_idx));

    thrust::sort_by_key(
        thrust::cuda::par(args.alloc),
        begin,
        begin + args.N_old + args.N,
        values_it);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size_sweep_and_prune = -1;
    static cudaFuncAttributes attr_sweep_and_prune;
    if (max_block_size_sweep_and_prune == -1)
        {
        cudaFuncGetAttributes(&attr_sweep_and_prune, gpu_sweep_and_prune_kernel<Shape>);
        max_block_size_sweep_and_prune = attr_sweep_and_prune.maxThreadsPerBlock;
        }

    // setup the grid to run the kernel
    unsigned int block_size_sweep_and_prune = min(args.block_size, (unsigned int)max_block_size_sweep_and_prune);

    unsigned int shared_bytes_sweep_and_prune = args.num_types * sizeof(typename Shape::param_type)
        + args.overlap_idx.getNumElements() * sizeof(unsigned int);

    cudaMemsetAsync(args.d_n_overlaps, 0, sizeof(unsigned int),args.stream);

    n_blocks = (args.N+args.N_old)/block_size_sweep_and_prune + 1;
    gpu_sweep_and_prune_kernel<Shape><<<n_blocks, block_size_sweep_and_prune, shared_bytes_sweep_and_prune,args.stream>>>(args.N,
        args.N_old,
        args.d_postype_test,
        args.d_postype,
        args.d_tag_test,
        args.d_tag,
        d_params,
        args.num_types,
        args.d_begin,
        args.d_end,
        args.d_aabb_idx,
        sweep_length,
        periodic,
        args.image_list,
        args.d_check_overlaps,
        args.overlap_idx,
        args.d_collisions,
        args.max_n_overlaps,
        args.d_n_overlaps,
        args.d_conditions);

    #else
    // AABB tree
    // bind the textures
    clusters_postype_tex.normalized = false;
    clusters_postype_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, clusters_postype_tex, args.d_postype, sizeof(Scalar4)*args.max_n);
    if (error != cudaSuccess)
        return error;

    clusters_orientation_tex.normalized = false;
    clusters_orientation_tex.filterMode = cudaFilterModePoint;
    error = cudaBindTexture(0, clusters_orientation_tex, args.d_orientation, sizeof(Scalar4)*args.max_n);
    if (error != cudaSuccess)
        return error;

    // determine the maximum block size and clamp the input block size down
    static int max_block_size_collisions = -1;
    static cudaFuncAttributes attr_collisions;
    if (max_block_size_collisions == -1)
        {
        cudaFuncGetAttributes(&attr_collisions, gpu_hpmc_clusters_kernel<Shape>);
        max_block_size_collisions = attr_collisions.maxThreadsPerBlock;
        }

    // setup the grid to run the kernel
    unsigned int block_size_collisions = min(args.block_size, (unsigned int)max_block_size_collisions);

    unsigned int group_size = args.group_size;
    while (block_size_collisions % group_size)
        group_size--;
    unsigned int n_groups = block_size_collisions/group_size;
    unsigned int n_blocks = args.N/n_groups+1;

    dim3 threads_collisions(group_size,n_groups,1);
    dim3 grid_collisions;

    if (n_blocks > (unsigned int) args.devprop.maxGridSize[1])
        grid_collisions = dim3(1, args.devprop.maxGridSize[1], n_blocks/args.devprop.maxGridSize[1] + 1);
    else
        grid_collisions = dim3(1, n_blocks, 1);

    unsigned int shared_bytes_collisions = args.num_types * sizeof(typename Shape::param_type) + args.overlap_idx.getNumElements()*sizeof(unsigned int);

    cudaMemsetAsync(args.d_n_overlaps, 0, sizeof(unsigned int),args.stream);

    // broad phase: detect collisions between circumspheres
    gpu_hpmc_clusters_kernel<Shape><<<grid_collisions, threads_collisions, shared_bytes_collisions, args.stream>>>(
                                                     args.N,
                                                     args.d_postype,
                                                     args.d_orientation,
                                                     args.d_tag,
                                                     args.num_types,
                                                     args.timestep,
                                                     args.dim,
                                                     args.box,
                                                     args.d_check_overlaps,
                                                     args.overlap_idx,
                                                     d_params,
                                                     args.d_collisions,
                                                     args.d_n_overlaps,
                                                     args.d_postype_test,
                                                     args.d_orientation_test,
                                                     args.d_tag_test,
                                                     args.line,
                                                     args.swap,
                                                     args.aabb_tree,
                                                     args.image_list,
                                                     args.max_n_overlaps,
                                                     args.d_conditions);
    #endif

    return cudaSuccess;
    }

//! Kernel driver for gpu_hpmc_clusters_overlaps_kernel()
/*! \param args Bundled arguments
    \param d_params Per-type shape parameters
    \returns Error codes generated by any CUDA calls, or cudaSuccess when there is no error

    This templatized method is the kernel driver for parallel update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup hpmc_kernels
*/
template< class Shape >
cudaError_t gpu_hpmc_clusters_overlaps(const hpmc_clusters_args_t& args, const typename Shape::param_type *d_params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_image);
    assert(args.d_tag);
    assert(args.d_cell_size);

    assert(args.d_postype_test);
    assert(args.d_orientation_test);
    assert(args.d_image_test);
    assert(args.d_tag_test);

    // bind the textures
    clusters_postype_tex.normalized = false;
    clusters_postype_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, clusters_postype_tex, args.d_postype, sizeof(Scalar4)*args.max_n);
    if (error != cudaSuccess)
        return error;

    clusters_orientation_tex.normalized = false;
    clusters_orientation_tex.filterMode = cudaFilterModePoint;
    error = cudaBindTexture(0, clusters_orientation_tex, args.d_orientation, sizeof(Scalar4)*args.max_n);
    if (error != cudaSuccess)
        return error;

    // required for memory coherency
    cudaDeviceSynchronize();

    // determine the maximum block size for the overlap check kernel and clamp the input block size down
    static int max_block_size_overlaps = -1;
    static cudaFuncAttributes attr_overlaps;
    if (max_block_size_overlaps == -1)
        {
        cudaFuncGetAttributes(&attr_overlaps, gpu_hpmc_clusters_overlaps_kernel<Shape>);
        max_block_size_overlaps = attr_overlaps.maxThreadsPerBlock;
        }

    unsigned int block_size_overlaps = min(args.block_size, (unsigned int)max_block_size_overlaps);

    unsigned int shared_bytes_overlaps = args.num_types * sizeof(typename Shape::param_type);
    unsigned int max_extra_bytes = max(0,(int) (32768 - attr_overlaps.sharedSizeBytes - shared_bytes_overlaps));

    // determine dynamically requested shared memory
    char *ptr = (char *)nullptr;
    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int i = 0; i < args.num_types; ++i)
        {
        d_params[i].load_shared(ptr, available_bytes);
        }
    unsigned int extra_bytes = max_extra_bytes - available_bytes;

    shared_bytes_overlaps += extra_bytes;

    dim3 threads_overlaps(block_size_overlaps,1,1);
    dim3 grid_overlaps( args.ncollisions / block_size_overlaps + 1, 1, 1);

    // narrow phase: check overlaps
    cudaMemsetAsync(args.d_n_overlaps, 0, sizeof(unsigned int),args.stream);

    if (args.d_n_reject)
        cudaMemsetAsync(args.d_n_reject, 0, sizeof(unsigned int),args.stream);

    gpu_hpmc_clusters_overlaps_kernel<Shape><<<grid_overlaps, threads_overlaps, shared_bytes_overlaps, args.stream>>>(
                                                     args.d_postype,
                                                     args.d_orientation,
                                                     args.d_image,
                                                     args.d_tag,
                                                     args.num_types,
                                                     d_params,
                                                     args.ncollisions,
                                                     args.d_collisions,
                                                     args.d_overlaps,
                                                     args.d_n_overlaps,
                                                     args.d_reject,
                                                     args.d_n_reject,
                                                     args.d_postype_test,
                                                     args.d_orientation_test,
                                                     args.d_image_test,
                                                     args.d_tag_test,
                                                     args.image_list,
                                                     args.image_hkl,
                                                     args.line,
                                                     args.swap,
                                                     args.type_A,
                                                     args.type_B,
                                                     args.max_n_reject,
                                                     args.d_conditions,
                                                     max_extra_bytes);


    // return control of managed memory
    cudaDeviceSynchronize();

    return cudaSuccess;
    }
#endif // NVCC

}; // end namespace detail

} // end namespace hpmc

#endif // _UPDATER_CLUSTERS_GPU_CUH_

