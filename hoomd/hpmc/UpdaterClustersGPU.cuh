// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _UPDATER_MUVT_GPU_CUH_
#define _UPDATER_MUVT_GPU_CUH_

#include "HPMCPrecisionSetup.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifdef NVCC
#include "hoomd/TextureTools.h"
#endif

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
                const Scalar4 *_d_postype,
                const Scalar4 *_d_orientation,
                const int3 *_d_image,
                const unsigned int *_d_tag,
                const unsigned int *_d_cell_idx,
                const unsigned int *_d_cell_size,
                const Index3D& _ci,
                const Index2D& _cli,
                const unsigned int *_d_excell_idx,
                const unsigned int *_d_excell_size,
                const Index2D& _excli,
                const uint3& _cell_dim,
                const unsigned int _num_types,
                const unsigned int _timestep,
                const unsigned int _dim,
                const BoxDim& _box,
                const unsigned int _block_size,
                const unsigned int _stride,
                const unsigned int _group_size,
                const unsigned int _max_n,
                const Scalar3 _ghost_width,
                const unsigned int *_d_check_overlaps,
                Index2D _overlap_idx,
                bool _line,
                const Scalar4 *_d_postype_test,
                const Scalar4 *_d_orientation_test,
                const int3 *_d_image_test,
                const unsigned int *_d_tag_test,
                uint2 *_d_overlaps,
                unsigned int _max_n_overlaps,
                unsigned int *_d_n_overlaps,
                unsigned int *_d_reject,
                unsigned int _max_n_reject,
                unsigned int *_d_n_reject,
                uint2 *_d_conditions,
                bool _swap,
                unsigned int _type_A,
                unsigned int _type_B,
                cudaStream_t _stream,
                const cudaDeviceProp& _devprop
                )
                : N(_N),
                  d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  d_image(_d_image),
                  d_tag(_d_tag),
                  d_cell_idx(_d_cell_idx),
                  d_cell_size(_d_cell_size),
                  ci(_ci),
                  cli(_cli),
                  d_excell_idx(_d_excell_idx),
                  d_excell_size(_d_excell_size),
                  excli(_excli),
                  cell_dim(_cell_dim),
                  num_types(_num_types),
                  timestep(_timestep),
                  dim(_dim),
                  box(_box),
                  block_size(_block_size),
                  stride(_stride),
                  group_size(_group_size),
                  max_n(_max_n),
                  ghost_width(_ghost_width),
                  d_check_overlaps(_d_check_overlaps),
                  overlap_idx(_overlap_idx),
                  line(_line),
                  d_postype_test(_d_postype_test),
                  d_orientation_test(_d_orientation_test),
                  d_image_test(_d_image_test),
                  d_tag_test(_d_tag_test),
                  d_overlaps(_d_overlaps),
                  max_n_overlaps(_max_n_overlaps),
                  d_n_overlaps(_d_n_overlaps),
                  d_reject(_d_reject),
                  max_n_reject(_max_n_reject),
                  d_n_reject(_d_n_reject),
                  d_conditions(_d_conditions),
                  swap(_swap),
                  type_A(_type_A),
                  type_B(_type_B),
                  stream(_stream),
                  devprop(_devprop)
        {
        };

    unsigned int N;                   //!< Number of particles to test
    const Scalar4 *d_postype;         //!< postype array of configuration to test against
    const Scalar4 *d_orientation;     //!< orientation array
    const int3 *d_image;              //!< Particle images
    const unsigned int *d_tag;        //!< Particle tags
    const unsigned int *d_cell_idx;   //!< Index data for each cell
    const unsigned int *d_cell_size;  //!< Number of particles in each cell
    const Index3D& ci;                //!< Cell indexer
    const Index2D& cli;               //!< Indexer for d_cell_idx
    const unsigned int *d_excell_idx; //!< Expanded cell neighbors
    const unsigned int *d_excell_size; //!< Size of expanded cell list per cell
    const Index2D excli;              //!< Expanded cell indexer
    const uint3& cell_dim;            //!< Cell dimensions
    const unsigned int num_types;     //!< Number of particle types
    const unsigned int timestep;      //!< Current time step
    const unsigned int dim;           //!< Number of dimensions
    const BoxDim& box;                //!< Current simulation box
    unsigned int block_size;          //!< Block size to execute
    unsigned int stride;              //!< Number of threads per overlap check
    unsigned int group_size;          //!< Size of the group to execute
    const unsigned int max_n;         //!< Maximum size of pdata arrays
    const Scalar3 ghost_width;       //!< Width of ghost layer
    const unsigned int *d_check_overlaps;   //!< Interaction matrix
    Index2D overlap_idx;              //!< Interaction matrix indexer
    bool line;                        //!< True if line reflection
    const Scalar4 *d_postype_test;    //!< Positions and types to test
    const Scalar4 *d_orientation_test;//!< Orientations to test
    const int3 *d_image_test;         //!< Images of particles to test
    const unsigned int *d_tag_test;   //!< Tags of particles to test
    uint2 *d_overlaps;                //!< List of overlap pairs generated
    unsigned int max_n_overlaps;      //!< Capacity of d_overlaps list
    unsigned int *d_n_overlaps;       //!< Number of particles inserted (return value)
    unsigned int *d_reject;           //!< Particle indices flagged for rejection
    unsigned int max_n_reject;        //!< Capacity of d_reject list
    unsigned int *d_n_reject;         //!< Number of particles flagged
    uint2 *d_conditions;              //!< Flags to indicate overflow
    bool swap;                        //!< If true, swap move
    unsigned int type_A;              //!< Type A of swap pair
    unsigned int type_B;              //!< Type B of swap pair
    cudaStream_t stream;               //!< Stream for kernel execution
    const cudaDeviceProp& devprop;    //!< CUDA device properties
    };

template< class Shape >
cudaError_t gpu_hpmc_clusters(const hpmc_clusters_args_t &args, const typename Shape::param_type *d_params);

#ifdef NVCC
//! Texture for reading postype
static scalar4_tex_t clusters_postype_tex;
//! Texture for reading orientation
static scalar4_tex_t clusters_orientation_tex;

//! Compute the cell that a particle sits in
__device__ inline unsigned int clusters_compute_cell_idx(const Scalar3 p,
                                               const BoxDim& box,
                                               const Scalar3& ghost_width,
                                               const uint3& cell_dim,
                                               const Index3D& ci)
    {
    // find the bin each particle belongs in
    Scalar3 f = box.makeFraction(p,ghost_width);
    uchar3 periodic = box.getPeriodic();
    int ib = (unsigned int)(f.x * cell_dim.x);
    int jb = (unsigned int)(f.y * cell_dim.y);
    int kb = (unsigned int)(f.z * cell_dim.z);

    // need to handle the case where the particle is exactly at the box hi
    if (ib == (int)cell_dim.x && periodic.x)
        ib = 0;
    if (jb == (int)cell_dim.y && periodic.y)
        jb = 0;
    if (kb == (int)cell_dim.z && periodic.z)
        kb = 0;

    // identify the bin
    return ci(ib,jb,kb);
    }


//! Kernel to find overlaps between different configurations
template< class Shape >
__global__ void gpu_hpmc_clusters_kernel(unsigned int N,
                                     const Scalar4 *d_postype,
                                     const Scalar4 *d_orientation,
                                     const int3 *d_image,
                                     const unsigned int *d_tag,
                                     const unsigned int *d_cell_size,
                                     const Index3D ci,
                                     const Index2D cli,
                                     const unsigned int *d_excell_idx,
                                     const unsigned int *d_excell_size,
                                     const Index2D excli,
                                     const uint3 cell_dim,
                                     const unsigned int num_types,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     Scalar3 ghost_width,
                                     const unsigned int *d_check_overlaps,
                                     Index2D overlap_idx,
                                     const typename Shape::param_type *d_params,
                                     uint2 *d_overlaps,
                                     unsigned int *d_n_overlaps,
                                     unsigned int *d_reject,
                                     unsigned int *d_n_reject,
                                     const Scalar4 *d_postype_test,
                                     const Scalar4 *d_orientation_test,
                                     const int3 *d_image_test,
                                     const unsigned int *d_tag_test,
                                     bool line,
                                     bool swap,
                                     unsigned int type_A,
                                     unsigned int type_B,
                                     unsigned int max_n_overlaps,
                                     unsigned int max_n_reject,
                                     uint2 *d_conditions,
                                     unsigned int max_extra_bytes)
    {
    unsigned int group = threadIdx.y;
    unsigned int offset = threadIdx.x;
    unsigned int group_size = blockDim.x;
    unsigned int n_groups = blockDim.y;

    // determine sample idx
    unsigned int i = blockIdx.x * n_groups + group;

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
    char *s_extra = (char *)&s_check_overlaps[ntyppairs];

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    __syncthreads();

    bool active = true;

    if (i >= N)
        {
        active = false;
        }

    unsigned int my_cell;

    // test particle position, orientation,..
    Scalar4 postype = d_postype_test[i];
    vec3<Scalar> pos_i(postype);
    unsigned int type = __scalar_as_int(postype.w);
    unsigned int tag = d_tag_test[i];
    int3 img = d_image_test[i];

    Shape shape_i(quat<Scalar>(d_orientation_test[i]), s_params[type]);

    if (active)
        {
        // find cell the particle is in
        Scalar3 p = vec_to_scalar3(pos_i);
        my_cell = clusters_compute_cell_idx(p, box, ghost_width, cell_dim, ci);
        }

    if (active)
        {
        // loop over neighboring cells and check for overlaps
        unsigned int excell_size = d_excell_size[my_cell];

        for (unsigned int k = 0; k < excell_size; k += group_size)
            {
            unsigned int local_k = k + offset;
            if (local_k < excell_size)
                {
                // read in position, and orientation of neighboring particle
                #if ( __CUDA_ARCH__ > 300)
                unsigned int j = __ldg(&d_excell_idx[excli(local_k, my_cell)]);
                #else
                unsigned int j = d_excell_idx[excli(local_k, my_cell)];
                #endif

                unsigned int tag_j = d_tag[j];
                if (tag == tag_j)
                    continue;

                Scalar4 postype_j = texFetchScalar4(d_postype, clusters_postype_tex, j);
                Scalar4 orientation_j = make_scalar4(1,0,0,0);
                unsigned int typ_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[typ_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(texFetchScalar4(d_orientation, clusters_orientation_tex, j));

                // put particle j into the coordinate system of particle i
                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i;
                vec3<Scalar> r_ij_wrap = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

                // check for overlaps
                OverlapReal rsq = dot(r_ij_wrap,r_ij_wrap);
                OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                if (rsq*OverlapReal(4.0) <= DaDb * DaDb)
                    {
                    // circumsphere overlap
                    unsigned int err_count;
                    if (s_check_overlaps[overlap_idx(typ_j, type)] && test_overlap(r_ij_wrap, shape_i, shape_j, err_count))
                        {
                        unsigned int n_overlaps = atomicAdd(d_n_overlaps, 1);
                        if (n_overlaps >= max_n_overlaps)
                            atomicMax(&(*d_conditions).x, n_overlaps+1);
                        else
                            d_overlaps[n_overlaps] = make_uint2(tag,tag_j);

                        if (d_reject)
                            {
                            bool reject = false;

                            if (line)
                                {
                                int3 delta_img = -box.getImage(r_ij - r_ij_wrap) + img - d_image[j];
                                reject = delta_img.x || delta_img.y || delta_img.z;
                                }
                            else if (swap)
                                {
                                reject = ((type != type_A && type != type_B) || (typ_j != type_A && typ_j != type_B));
                                }

                            if (reject)
                                {
                                unsigned int n_reject = atomicAdd(d_n_reject, 1);
                                if (n_reject >= max_n_reject)
                                    atomicMax(&(*d_conditions).y, n_reject+1);
                                else
                                    {
                                    d_reject[n_reject] = tag;
                                    }

                                n_reject = atomicAdd(d_n_reject, 1);
                                if (n_reject >= max_n_reject)
                                    atomicMax(&(*d_conditions).y, n_reject+1);
                                else
                                    {
                                    d_reject[n_reject] = tag_j;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
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
    assert(args.group_size >= 1);
    assert(args.group_size <= 32);  // note, really should be warp size of the device
    assert(args.block_size%(args.group_size)==0);

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

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static cudaFuncAttributes attr;
    if (max_block_size == -1)
        {
        cudaFuncGetAttributes(&attr, gpu_hpmc_clusters_kernel<Shape>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    // setup the grid to run the kernel
    unsigned int n_groups = min(args.block_size, (unsigned int)max_block_size) / args.group_size;

    dim3 threads(args.group_size, n_groups);
    dim3 grid( args.N / n_groups + 1, 1, 1);

    unsigned int shared_bytes = args.num_types * sizeof(typename Shape::param_type) + n_groups*sizeof(unsigned int)
        + args.overlap_idx.getNumElements()*sizeof(unsigned int);

    // required for memory coherency
    cudaDeviceSynchronize();

    unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - attr.sharedSizeBytes - shared_bytes;

    // attach the parameters to the kernel stream so that they are visible
    // when other kernels are called
    cudaStreamAttachMemAsync(args.stream, d_params, 0, cudaMemAttachSingle);
    for (unsigned int i = 0; i < args.num_types; ++i)
        {
        // attach nested memory regions
        d_params[i].attach_to_stream(args.stream);
        }

    // determine dynamically requested shared memory
    char *ptr = (char *)nullptr;
    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int i = 0; i < args.num_types; ++i)
        {
        d_params[i].load_shared(ptr, available_bytes);
        }
    unsigned int extra_bytes = max_extra_bytes - available_bytes;

    shared_bytes += extra_bytes;

    if (args.d_n_reject)
        cudaMemsetAsync(args.d_n_reject, 0, sizeof(unsigned int),args.stream);

    cudaMemsetAsync(args.d_n_overlaps, 0, sizeof(unsigned int),args.stream);

    gpu_hpmc_clusters_kernel<Shape><<<grid, threads, shared_bytes, args.stream>>>(
                                                     args.N,
                                                     args.d_postype,
                                                     args.d_orientation,
                                                     args.d_image,
                                                     args.d_tag,
                                                     args.d_cell_size,
                                                     args.ci,
                                                     args.cli,
                                                     args.d_excell_idx,
                                                     args.d_excell_size,
                                                     args.excli,
                                                     args.cell_dim,
                                                     args.num_types,
                                                     args.timestep,
                                                     args.dim,
                                                     args.box,
                                                     args.ghost_width,
                                                     args.d_check_overlaps,
                                                     args.overlap_idx,
                                                     d_params,
                                                     args.d_overlaps,
                                                     args.d_n_overlaps,
                                                     args.d_reject,
                                                     args.d_n_reject,
                                                     args.d_postype_test,
                                                     args.d_orientation_test,
                                                     args.d_image_test,
                                                     args.d_tag_test,
                                                     args.line,
                                                     args.swap,
                                                     args.type_A,
                                                     args.type_B,
                                                     args.max_n_overlaps,
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

#endif // _UPDATER_MUVT_GPU_CUH_

