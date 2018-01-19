// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _UPDATER_MUVT_GPU_CUH_
#define _UPDATER_MUVT_GPU_CUH_

#include "HPMCCounters.h"
#include "HPMCPrecisionSetup.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/Saru.h"

#include "hoomd/AABBTree.h"

#include "hoomd/CachedAllocator.h"

#ifdef NVCC
#include "Moves.h"
#include "hoomd/TextureTools.h"
#endif

#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>

namespace hpmc
{

namespace detail
{

/*! \file IntegratorHPMCMonoImplicit.cuh
    \brief Declaration of CUDA kernels drivers
*/

//! Wraps arguments to gpu_hpmc_muvt
/*! \ingroup hpmc_data_structs */
struct hpmc_muvt_args_t
    {
    //! Construct a pair_args_t
    hpmc_muvt_args_t(
                unsigned int _n_insert,
                unsigned int _type,
                Scalar4 *_d_postype,
                Scalar4 *_d_orientation,
                const AABBTree& _aabb_tree,
                const ManagedArray<vec3<Scalar> >& _image_list,
                const unsigned int _N,
                const unsigned int _num_types,
                const unsigned int _seed,
                unsigned int _select,
                const unsigned int _timestep,
                const unsigned int _dim,
                const BoxDim& _box,
                const unsigned int _block_size,
                const unsigned int _stride,
                const unsigned int _max_n,
                const unsigned int *_d_check_overlaps,
                Index2D _overlap_idx,
                unsigned int *_d_overlap,
                Scalar4 *_d_postype_insert,
                Scalar4 *_d_orientation_insert,
                unsigned int &_n_ptls_inserted,
                cudaStream_t _stream,
                const cudaDeviceProp& _devprop,
                const CachedAllocator& _alloc
                )
                : n_insert(_n_insert),
                  type(_type),
                  d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  aabb_tree(_aabb_tree),
                  image_list(_image_list),
                  N(_N),
                  num_types(_num_types),
                  seed(_seed),
                  select(_select),
                  timestep(_timestep),
                  dim(_dim),
                  box(_box),
                  block_size(_block_size),
                  stride(_stride),
                  max_n(_max_n),
                  d_check_overlaps(_d_check_overlaps),
                  overlap_idx(_overlap_idx),
                  d_overlap(_d_overlap),
                  d_postype_insert(_d_postype_insert),
                  d_orientation_insert(_d_orientation_insert),
                  n_ptls_inserted(_n_ptls_inserted),
                  stream(_stream),
                  devprop(_devprop),
                  alloc(_alloc)
        {
        };

    unsigned int n_insert;            //!< Number of depletants particles to generate
    unsigned int type;                //!< Type of depletant particle
    Scalar4 *d_postype;               //!< postype array
    Scalar4 *d_orientation;           //!< orientation array
    const AABBTree& aabb_tree;        //!< Locality data structure
    const ManagedArray<vec3<Scalar> >& image_list;       //!< List of periodic images
    const unsigned int N;             //!< Number of particles
    const unsigned int num_types;     //!< Number of particle types
    const unsigned int seed;          //!< RNG seed
    unsigned int select;              //!< RNG select value
    const unsigned int timestep;      //!< Current time step
    const unsigned int dim;           //!< Number of dimensions
    const BoxDim& box;                //!< Current simulation box
    unsigned int block_size;          //!< Block size to execute
    unsigned int stride;              //!< Number of threads per overlap check
    const unsigned int max_n;         //!< Maximum size of pdata arrays
    const unsigned int *d_check_overlaps;   //!< Interaction matrix
    Index2D overlap_idx;              //!< Interaction matrix indexer
    unsigned int *d_overlap;          //!< Overlap flags
    Scalar4 *d_postype_insert;        //!< Inserted positions and types
    Scalar4 *d_orientation_insert;    //!< Inserteded orientations
    unsigned int &n_ptls_inserted;    //!< Number of particles inserted (return value)
    cudaStream_t stream;               //!< Stream for kernel execution
    const cudaDeviceProp& devprop;    //!< CUDA device properties
    const CachedAllocator& alloc;     //!< Caching allocator for thrust
    };

template< class Shape >
cudaError_t gpu_hpmc_muvt(const hpmc_muvt_args_t &args, const typename Shape::param_type *d_params);

cudaError_t gpu_muvt_set_particle_properties(
    unsigned int n_ptls_inserted,
    const unsigned int *d_rtag,
    const unsigned int *d_tags,
    Scalar4 *d_postype,
    Scalar4 *d_orientation,
    const Scalar4 *d_postype_insert,
    const Scalar4 *d_orientation_insert,
    unsigned int block_size);

#ifdef NVCC
//! Texture for reading postype
static scalar4_tex_t muvt_postype_tex;
//! Texture for reading orientation
static scalar4_tex_t muvt_orientation_tex;

//! Kernel to estimate the colloid overlap volume and the depletant free volume
/*! \param n_insert Number of probe depletant particles to generate
    \param type Type of depletant particle
    \param d_postype Particle positions and types by index
    \param d_orientation Particle orientation
    \param aabb_tree
    \param image_list
    \param N number of particles
    \param num_types Number of particle types
    \param seed User chosen random number seed
    \param a Size of rotation move (per type)
    \param timestep Current timestep of the simulation
    \param dim Dimension of the simulation box
    \param box Simulation box
    \param d_params Per-type shape parameters
    \param d_overlaps Per-type pair interaction matrix
*/
template< class Shape >
__global__ void gpu_hpmc_muvt_kernel(unsigned int n_insert,
                                     unsigned int type,
                                     Scalar4 *d_postype,
                                     Scalar4 *d_orientation,
                                     AABBTree aabb_tree,
                                     ManagedArray<vec3<Scalar> > image_list,
                                     const unsigned int N,
                                     const unsigned int num_types,
                                     const unsigned int seed,
                                     const unsigned int select,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const unsigned int *d_check_overlaps,
                                     Index2D overlap_idx,
                                     const typename Shape::param_type *d_params,
                                     unsigned int *d_overlap,
                                     Scalar4 *d_postype_insert,
                                     Scalar4 *d_orientation_insert,
                                     unsigned int max_extra_bytes)
    {
    // determine sample idx
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

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

    // initialize extra shared mem
    char *s_extra = (char *)(&s_check_overlaps[ntyppairs]);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    __syncthreads();

    if (i >= n_insert)
        return;

    // one RNG per particle
    hoomd::detail::Saru rng(i^0xf672b4ce, seed+select, timestep);

    // test depletant position
    quat<Scalar> orientation_i;
    Shape shape_i(orientation_i, s_params[type]);

    // select a random particle coordinate in the box
    Scalar xrand = rng.template s<Scalar>();
    Scalar yrand = rng.template s<Scalar>();
    Scalar zrand(0.5);

    if (dim==3)
        zrand = rng.template s<Scalar>();

    Scalar3 f = make_scalar3(xrand, yrand, zrand);
    vec3<Scalar> pos_i = vec3<Scalar>(box.makeCoordinates(f));

    if (shape_i.hasOrientation())
        {
        shape_i.orientation = generateRandomOrientation(rng);
        }

    // stash in global memory
    d_postype_insert[i] = make_scalar4(pos_i.x, pos_i.y, pos_i.z, __int_as_scalar(type));
    d_orientation_insert[i] = quat_to_scalar4(shape_i.orientation);

    // find cell the particle is in
    Scalar3 p = vec_to_scalar3(pos_i);

    bool overlap = false;

    unsigned int num_nodes = aabb_tree.getNumNodes();
    detail::AABB aabb_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

    unsigned int n_images = image_list.size();

    for (unsigned int cur_image = 0; cur_image < n_images; ++cur_image)
        {
        vec3<Scalar> pos_image = pos_i + image_list[cur_image];
        detail::AABB aabb = aabb_local;
        aabb.translate(pos_image);

        for (unsigned int cur_node_idx = 0; cur_node_idx < num_nodes; ++cur_node_idx)
            {
            if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                {
                if (aabb_tree.isNodeLeaf(cur_node_idx))
                    {
                    for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                        {
                        unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                        Scalar4 postype_j = texFetchScalar4(d_postype, muvt_postype_tex, j);
                        Scalar4 orientation_j = make_scalar4(1,0,0,0);
                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(orientation_j), s_params[typ_j]);
                        if (shape_j.hasOrientation())
                            shape_j.orientation = quat<Scalar>(texFetchScalar4(d_orientation, muvt_orientation_tex, j));

                        vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_image;

                        // check for overlaps
                        OverlapReal rsq = dot(r_ij,r_ij);
                        OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                        if (rsq*OverlapReal(4.0) <= DaDb * DaDb)
                            {
                            // circumsphere overlap
                            unsigned int err_count;
                            if (s_check_overlaps[overlap_idx(typ_j, type)] && test_overlap(r_ij, shape_i, shape_j, err_count))
                                {
                                overlap = true;
                                break;
                                }
                            }
                        }
                    }
                }
            else
                {
                // skipp ahead
                cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                }

            if (overlap)
                break;
            } // end loop over AABB nodes
        if (overlap)
            break;
        } // end loop over images

    // flag as overlapping in global mem
    if (overlap)
        {
        d_overlap[i] = 1;
        }
    }

//! Kernel driver for gpu_hpmc_muvt_kernel()
/*! \param args Bundled arguments
    \param d_params Per-type shape parameters
    \returns Error codes generated by any CUDA calls, or cudaSuccess when there is no error

    This templatized method is the kernel driver for parallel update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup hpmc_kernels
*/
template< class Shape >
cudaError_t gpu_hpmc_muvt(const hpmc_muvt_args_t& args, const typename Shape::param_type *d_params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_cell_size);

    // bind the textures
    muvt_postype_tex.normalized = false;
    muvt_postype_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, muvt_postype_tex, args.d_postype, sizeof(Scalar4)*args.max_n);
    if (error != cudaSuccess)
        return error;

    muvt_orientation_tex.normalized = false;
    muvt_orientation_tex.filterMode = cudaFilterModePoint;
    error = cudaBindTexture(0, muvt_orientation_tex, args.d_orientation, sizeof(Scalar4)*args.max_n);
    if (error != cudaSuccess)
        return error;

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static cudaFuncAttributes attr;
    if (max_block_size == -1)
        {
        cudaFuncGetAttributes(&attr, gpu_hpmc_muvt_kernel<Shape>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

    // setup the grid to run the kernel
    dim3 threads(block_size, 1, 1);
    dim3 grid( args.n_insert / block_size + 1, 1, 1);

    unsigned int shared_bytes = args.num_types * sizeof(typename Shape::param_type)
        + args.overlap_idx.getNumElements()*sizeof(unsigned int);

    // required for memory coherency
    cudaDeviceSynchronize();

    unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - attr.sharedSizeBytes - shared_bytes;

    // determine dynamically requested shared memory
    char *ptr = (char *)nullptr;
    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int i = 0; i < args.num_types; ++i)
        {
        d_params[i].load_shared(ptr, available_bytes);
        }
    unsigned int extra_bytes = max_extra_bytes - available_bytes;

    shared_bytes += extra_bytes;

    // attach the parameters to the kernel stream so that they are visible
    // when other kernels are called
    for (unsigned int i = 0; i < args.num_types; ++i)
        {
        // attach nested memory regions
        d_params[i].attach_to_stream(args.stream);
        }
    cudaStreamAttachMemAsync(args.stream, d_params, 0, cudaMemAttachSingle);

    cudaMemsetAsync(args.d_overlap, 0, sizeof(unsigned int)*args.n_insert, args.stream);

    gpu_hpmc_muvt_kernel<Shape><<<grid, threads, shared_bytes, args.stream>>>(
                                                     args.n_insert,
                                                     args.type,
                                                     args.d_postype,
                                                     args.d_orientation,
                                                     args.aabb_tree,
                                                     args.image_list,
                                                     args.N,
                                                     args.num_types,
                                                     args.seed,
                                                     args.select,
                                                     args.timestep,
                                                     args.dim,
                                                     args.box,
                                                     args.d_check_overlaps,
                                                     args.overlap_idx,
                                                     d_params,
                                                     args.d_overlap,
                                                     args.d_postype_insert,
                                                     args.d_orientation_insert,
                                                     max_extra_bytes);

    // return control of managed memory
    cudaDeviceSynchronize();

    // stream compact the output by removing overlapping elements
    thrust::device_ptr<unsigned int> overlap(args.d_overlap);
    thrust::device_ptr<Scalar4> postype_insert(args.d_postype_insert);
    thrust::device_ptr<Scalar4> orientation_insert(args.d_orientation_insert);
    auto itr = thrust::make_zip_iterator(thrust::make_tuple(postype_insert, orientation_insert));

    auto end = thrust::remove_if(thrust::cuda::par(args.alloc),
        itr,
        itr+args.n_insert,
        overlap,
        thrust::identity<unsigned int>());

    args.n_ptls_inserted = end - itr;

    return cudaSuccess;
    }
#endif // NVCC

}; // end namespace detail

} // end namespace hpmc

#endif // _UPDATER_MUVT_GPU_CUH_

