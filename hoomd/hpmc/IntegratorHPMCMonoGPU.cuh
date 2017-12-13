// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _INTEGRATOR_HPMC_CUH_
#define _INTEGRATOR_HPMC_CUH_


#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/Saru.h"

#include <cassert>

#include "HPMCCounters.h"

#ifdef NVCC
#include "HPMCPrecisionSetup.h"
#include "Moves.h"
#include "hoomd/TextureTools.h"
#endif

namespace hpmc
{

namespace detail
{

#define OVERLAP_IN_OLD_CONFIG 1
#define OVERLAP_IN_NEW_CONFIG 2

/*! \file IntegratorHPMCMonoGPU.cuh
    \brief Declaration of CUDA kernels drivers
*/

//! Wraps arguments to gpu_hpmc_up
/*! \ingroup hpmc_data_structs */
struct hpmc_args_t
    {
    //! Construct a pair_args_t
    hpmc_args_t(Scalar4 *_d_postype,
                Scalar4 *_d_orientation,
                hpmc_counters_t *_d_counters,
                const unsigned int *_d_cell_idx,
                const unsigned int *_d_cell_size,
                const unsigned int *_d_excell_idx,
                const unsigned int *_d_excell_cell_set,
                unsigned int *_d_excell_overlap,
                const unsigned int *_d_excell_size,
                const Index3D& _ci,
                const Index2D& _cli,
                const Index2D& _excli,
                const uint3& _cell_dim,
                const Scalar3& _ghost_width,
                const unsigned int *_d_cell_set,
                const unsigned int _n_active_cells,
                const unsigned int _N,
                const unsigned int _num_types,
                const unsigned int _seed,
                const Scalar* _d,
                const Scalar* _a,
                const unsigned int *_check_overlaps,
                const Index2D& _overlap_idx,
                const unsigned int _move_ratio,
                const unsigned int _timestep,
                const unsigned int _dim,
                const BoxDim& _box,
                const unsigned int _select,
                const Scalar3 _ghost_fraction,
                const bool _domain_decomposition,
                const unsigned int _block_size,
                const unsigned int _stride,
                const unsigned int _group_size,
                const bool _has_orientation,
                const unsigned int _max_n,
                const cudaDeviceProp& _devprop,
                bool _update_shape_param,
                cudaStream_t _stream,
                unsigned int *_d_active_cell_ptl_idx = NULL,
                unsigned int *_d_active_cell_accept = NULL,
                unsigned int *_d_active_cell_move_type_translate = NULL,
                Index2D _queue_idx = Index2D(),
                unsigned int *_d_queue_active_cell_idx = NULL,
                Scalar4 *_d_queue_postype = NULL,
                Scalar4 *_d_queue_orientation = NULL,
                unsigned int *_d_queue_excell_idx = NULL,
                unsigned int *_d_cell_overlaps = NULL,
                unsigned int _max_gmem_queue_size = 0,
                bool _check_cuda_errors = false,
                unsigned int _block_size_overlaps = 0,
                bool _load_shared = false,
                Scalar4 *_d_trial_postype = 0,
                Scalar4 *_d_trial_orientation = 0,
                unsigned int *_d_trial_updated = 0,
                unsigned int *_d_trial_move_type_translate = 0,
                unsigned int *_d_update_order = 0,
                unsigned int _cur_set = 0,
                Index2D _csi = Index2D())
                : d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  d_counters(_d_counters),
                  d_cell_idx(_d_cell_idx),
                  d_cell_size(_d_cell_size),
                  d_excell_idx(_d_excell_idx),
                  d_excell_cell_set(_d_excell_cell_set),
                  d_excell_overlap(_d_excell_overlap),
                  d_excell_size(_d_excell_size),
                  ci(_ci),
                  cli(_cli),
                  excli(_excli),
                  cell_dim(_cell_dim),
                  ghost_width(_ghost_width),
                  d_cell_set(_d_cell_set),
                  n_active_cells(_n_active_cells),
                  N(_N),
                  num_types(_num_types),
                  seed(_seed),
                  d_d(_d),
                  d_a(_a),
                  d_check_overlaps(_check_overlaps),
                  overlap_idx(_overlap_idx),
                  move_ratio(_move_ratio),
                  timestep(_timestep),
                  dim(_dim),
                  box(_box),
                  select(_select),
                  ghost_fraction(_ghost_fraction),
                  domain_decomposition(_domain_decomposition),
                  block_size(_block_size),
                  stride(_stride),
                  group_size(_group_size),
                  has_orientation(_has_orientation),
                  max_n(_max_n),
                  devprop(_devprop),
                  update_shape_param(_update_shape_param),
                  stream(_stream),
                  d_active_cell_ptl_idx(_d_active_cell_ptl_idx),
                  d_active_cell_accept(_d_active_cell_accept),
                  d_active_cell_move_type_translate(_d_active_cell_move_type_translate),
                  queue_idx(_queue_idx),
                  d_queue_active_cell_idx(_d_queue_active_cell_idx),
                  d_queue_postype(_d_queue_postype),
                  d_queue_orientation(_d_queue_orientation),
                  d_queue_excell_idx(_d_queue_excell_idx),
                  d_cell_overlaps(_d_cell_overlaps),
                  max_gmem_queue_size(_max_gmem_queue_size),
                  check_cuda_errors(_check_cuda_errors),
                  block_size_overlaps(_block_size_overlaps),
                  load_shared(_load_shared),
                  d_trial_postype(_d_trial_postype),
                  d_trial_orientation(_d_trial_orientation),
                  d_trial_updated(_d_trial_updated),
                  d_trial_move_type_translate(_d_trial_move_type_translate),
                  d_update_order(_d_update_order),
                  cur_set(_cur_set),
                  csi(_csi)
        {
        };

    Scalar4 *d_postype;               //!< postype array
    Scalar4 *d_orientation;           //!< orientation array
    hpmc_counters_t *d_counters;      //!< Move accept/reject counters
    const unsigned int *d_cell_idx;   //!< Index data for each cell
    const unsigned int *d_cell_size;  //!< Number of particles in each cell
    const unsigned int *d_excell_idx; //!< Index data for each expanded cell
    const unsigned int *d_excell_cell_set; //!< active cell set for each expanded cell
    unsigned int *d_excell_overlap;   //!< Per-neighbor overlap flag, ==1 if overlap in old config, ==2 if overlap in new config
    const unsigned int *d_excell_size;//!< Number of particles in each expanded cell
    const Index3D& ci;                //!< Cell indexer
    const Index2D& cli;               //!< Indexer for d_cell_idx
    const Index2D& excli;             //!< Indexer for d_excell_idx
    const uint3& cell_dim;            //!< Cell dimensions
    const Scalar3& ghost_width;       //!< Width of the ghost layer
    const unsigned int *d_cell_set;   //!< List of active cells
    const unsigned int n_active_cells;//!< Number of active cells
    const unsigned int N;             //!< Number of particles
    const unsigned int num_types;     //!< Number of particle types
    const unsigned int seed;          //!< RNG seed
    const Scalar* d_d;                //!< Maximum move displacement
    const Scalar* d_a;                //!< Maximum move angular displacement
    const unsigned int *d_check_overlaps; //!< Interaction matrix
    const Index2D& overlap_idx;       //!< Indexer into interaction matrix
    const unsigned int move_ratio;    //!< Ratio of translation to rotation moves
    const unsigned int timestep;      //!< Current time step
    const unsigned int dim;           //!< Number of dimensions
    const BoxDim& box;                //!< Current simulation box
    const unsigned int select;        //!< Current selection
    const Scalar3 ghost_fraction;     //!< Width of the inactive layer
    const bool domain_decomposition;  //!< Is domain decomposition mode enabled?
    unsigned int block_size;          //!< Block size to execute
    unsigned int stride;              //!< Number of threads per overlap check
    unsigned int group_size;          //!< Size of the group to execute
    const bool has_orientation;       //!< True if the shape has orientation
    const unsigned int max_n;         //!< Maximum size of pdata arrays
    const cudaDeviceProp& devprop;    //!< CUDA device properties
    bool update_shape_param;          //!< If true, update size of shape param and synchronize GPU execution stream
    cudaStream_t stream;              //!< The CUDA stream associated with the update kernel
    unsigned int *d_active_cell_ptl_idx; //!< Updated particle index per active cell (ignore if NULL)
    unsigned int *d_active_cell_accept;//!< =1 if active cell move has been accepted, =0 otherwise (ignore if NULL)
    unsigned int *d_active_cell_move_type_translate;//!< =1 if active cell move was a translation, =0 if rotation
    Index2D queue_idx;                //!< The indexer for the work queue
    unsigned int *d_queue_active_cell_idx; //!< Queue of active cell indices
    Scalar4 *d_queue_postype;         //!< Queue of new particle position (and type)
    Scalar4 *d_queue_orientation;     //!< Queue of new particle orientation
    unsigned int *d_queue_excell_idx; //!< Queue of neighboring particle excell idx to test for overlap
    unsigned int *d_cell_overlaps;    //!< Result of overlap check per active cell
    unsigned int max_gmem_queue_size; //!< Maximum length of global memory queue
    bool check_cuda_errors;           //!< True if CUDA error checking in child kernel is enabled
    unsigned int block_size_overlaps; //!< Block size for overlap check kernel
    bool load_shared;                 //!< Whether to load extra shape data into shared mem or fetch from managed mem
    Scalar4 *d_trial_postype;         //!< New positions (and type) of particles
    Scalar4 *d_trial_orientation;     //!< New orientations of particles
    unsigned int *d_trial_updated;    //!< ==1 if this particle has been moved
    unsigned int *d_trial_move_type_translate;  //!< per cell flag, whether it is a translation or rotation (UINT_MAX if no move)
    unsigned int *d_update_order;     //!< The update sequence of cell sets
    unsigned int cur_set;             //!< the current active cell set
    Index2D csi;                      //!< The cell set indexer
    };

cudaError_t gpu_hpmc_excell(unsigned int *d_excell_idx,
                            unsigned int *d_excell_size,
                            const Index2D& excli,
                            const unsigned int *d_cell_idx,
                            const unsigned int *d_cell_size,
                            const unsigned int *d_cell_adj,
                            const Index3D& ci,
                            const Index2D& cli,
                            const Index2D& cadji,
                            const unsigned int block_size);

cudaError_t gpu_hpmc_excell_and_cell_set(unsigned int *d_inverse_cell_set,
                            unsigned int *d_excell_idx,
                            unsigned int *d_excell_cell_set,
                            unsigned int *d_excell_size,
                            const Index2D& excli,
                            const unsigned int *d_cell_idx,
                            const unsigned int *d_cell_size,
                            const unsigned int *d_cell_adj,
                            const Index3D& ci,
                            const Index2D& cli,
                            const Index2D& cadji,
                            const unsigned int block_size);

template< class Shape >
cudaError_t gpu_hpmc_update(const hpmc_args_t& args, const typename Shape::param_type *params);

template< class Shape >
cudaError_t gpu_hpmc_moves(const hpmc_args_t& args, const typename Shape::param_type *params);

template< class Shape >
cudaError_t gpu_hpmc_check_overlaps(const hpmc_args_t& args, const typename Shape::param_type *params);

template< class Shape >
cudaError_t gpu_hpmc_accept(const hpmc_args_t& args, const typename Shape::param_type *params);

cudaError_t gpu_hpmc_shift(Scalar4 *d_postype,
                           int3 *d_image,
                           const unsigned int N,
                           const BoxDim& box,
                           const Scalar3 shift,
                           const unsigned int block_size);

#ifdef NVCC
/*!
 * Definition of function templates and templated GPU kernels
 */

//! Texture for reading postype
static scalar4_tex_t postype_tex;
//! Texture for reading orientation
static scalar4_tex_t orientation_tex;
//! Texture for reading cell index data
static texture<unsigned int, 1, cudaReadModeElementType> cell_idx_tex;

//! Device function to compute the cell that a particle sits in
__device__ inline unsigned int computeParticleCell(const Scalar3& p,
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
    if (f.x >= Scalar(0.0) && f.x < Scalar(1.0) && f.y >= Scalar(0.0) && f.y < Scalar(1.0) && f.z >= Scalar(0.0) && f.z < Scalar(1.0))
        return ci(ib,jb,kb);
    else
        return 0xffffffff;
    }

//! HPMC  update kernel
/*! \param d_postype Particle positions and types by index
    \param d_orientation Particle orientation
    \param d_counters Acceptance counters to increment
    \param d_cell_idx Particle index stored in the cell list
    \param d_cell_size The size of each cell
    \param d_excell_idx Indices of particles in extended cells
    \param d_excell_size Number of particles in each extended cell
    \param ci Cell indexer
    \param cli Cell list indexer
    \param excli Extended cell list indexer
    \param cell_dim Dimensions of the cell list
    \param ghost_width Width of the ghost layer
    \param d_cell_set List of active cells
    \param n_active_cells Number of active cells
    \param N number of particles
    \param num_types Number of particle types
    \param seed User chosen random number seed
    \param d_d Array of maximum move displacements
    \param d_a Array of rotation move sizes
    \param d_check_overlaps Interaction matrix
    \parma overlap_idx Indexer into interaction matrix
    \param move_ratio Ratio of translation moves to rotation moves
    \param timestep Current timestep of the simulation
    \param dim Dimension of the simulation box
    \param box Simulation box
    \param select Current index within the loop over nselect selections (for RNG generation)
    \param ghost_fraction Width of the inactive layer in MPI domain decomposition simulations
    \param domain_decomposition True if executing with domain decomposition
    \param d_params Per-type shape parameters

    MPMC in its published form has a severe limit on the number of parallel threads in 3D. This implementation launches
    group_size threads per cell (1,2,4,8,16,32). Each thread in the group performs the same trial move on the same
    particle, and then checks for overlaps against different particles from the extended cell list. The entire extended
    cell list is covered in a batched loop. The group_size is autotuned to find the fastest performance. Smaller systems
    tend to run fastest with a large group_size due to the increased parallelism. Larger systems tend to run faster
    at smaller group_sizes because they already have the parallelism from the system size - however, even the largest
    systems benefit from group_size > 1 on K20. Shared memory is used to set an overlap flag to 1 if any of the threads
    in the group detect an overlap. After all checks are complete, the master thread in the group applies the trial move
    update if accepted.

    No __synchtreads is needed after the overlap checks because the group_size is always chosen to be a power of 2 and
    smaller than the warp size. Only a __threadfence_block() is needed to ensure memory consistency.

    Move stats are tallied in local memory, then totaled in shared memory at the end and finally a single thread in the
    block runs an atomicAdd on global memory to get the system wide total. This isn't as good as a reduction, but it
    is only a tiny fraction of the compute time.

    In order to simplify indexing and boundary checks, a list of active cells is determined on the host and passed into
    the kernel. That way, only a linear indexing of threads is needed to handle any geometry of active cells.

    Heavily divergent warps are avoided by pre-building a list of all particles in the neighboring region of any given
    cell. Otherwise, extremely non-uniform cell lengths (i.e. avg 1, max 4) don't cause massive performance degradation.

    **Indexing**
        - threadIdx.y indexes the current group in the block
        - threadIdx.x is the offset within the current group
        - blockIdx.x runs enough blocks so that all active cells are covered

    **Possible enhancements**
        - Use __ldg and not tex1Dfetch on sm35

    \ingroup hpmc_kernels
*/
template< class Shape >
__global__ void gpu_hpmc_mpmc_kernel(Scalar4 *d_postype,
                                     Scalar4 *d_orientation,
                                     hpmc_counters_t *d_counters,
                                     const unsigned int *d_cell_idx,
                                     const unsigned int *d_cell_size,
                                     const unsigned int *d_excell_idx,
                                     const unsigned int *d_excell_size,
                                     const Index3D ci,
                                     const Index2D cli,
                                     const Index2D excli,
                                     const uint3 cell_dim,
                                     const Scalar3 ghost_width,
                                     const unsigned int *d_cell_set,
                                     const unsigned int n_active_cells,
                                     const unsigned int N,
                                     const unsigned int num_types,
                                     const unsigned int seed,
                                     const Scalar* d_d,
                                     const Scalar* d_a,
                                     const unsigned int *d_check_overlaps,
                                     const Index2D overlap_idx,
                                     const unsigned int move_ratio,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const unsigned int select,
                                     const Scalar3 ghost_fraction,
                                     const bool domain_decomposition,
                                     unsigned int *d_active_cell_ptl_idx,
                                     unsigned int *d_active_cell_accept,
                                     unsigned int *d_active_cell_move_type_translate,
                                     const typename Shape::param_type *d_params,
                                     unsigned int max_queue_size,
                                     unsigned int max_extra_bytes)
    {
    // flags to tell what type of thread we are
    bool active = true;
    unsigned int group;
    unsigned int offset;
    unsigned int group_size;
    bool master;
    unsigned int n_groups;

    if (Shape::isParallel())
        {
        // use 3d thread block layout
        group = threadIdx.z;
        offset = threadIdx.y;
        group_size = blockDim.y;
        master = (offset == 0 && threadIdx.x == 0);
        n_groups = blockDim.z;
        }
    else
        {
        group = threadIdx.y;
        offset = threadIdx.x;
        group_size = blockDim.x;
        master = (offset == 0);
        n_groups = blockDim.y;
        }

    unsigned int err_count = 0;

    // shared arrays for per type pair parameters
    __shared__ unsigned int s_translate_accept_count;
    __shared__ unsigned int s_translate_reject_count;
    __shared__ unsigned int s_rotate_accept_count;
    __shared__ unsigned int s_rotate_reject_count;
    __shared__ unsigned int s_overlap_checks;
    __shared__ unsigned int s_overlap_err_count;

    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;

    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];

    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar4 *s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    Scalar *s_d = (Scalar *)(s_pos_group + n_groups);
    Scalar *s_a = (Scalar *)(s_d + num_types);
    unsigned int *s_check_overlaps = (unsigned int *) (s_a + num_types);
    unsigned int *s_queue_j =   (unsigned int*)(s_check_overlaps + overlap_idx.getNumElements());
    unsigned int *s_overlap =   (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int *s_queue_gid = (unsigned int*)(s_overlap + n_groups);
    unsigned int *s_type_group = (unsigned int*)(s_queue_gid + max_queue_size);

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

        for (unsigned int cur_offset = 0; cur_offset < num_types; cur_offset += block_size)
            {
            if (cur_offset + tidx < num_types)
                {
                s_a[cur_offset + tidx] = d_a[cur_offset + tidx];
                s_d[cur_offset + tidx] = d_d[cur_offset + tidx];
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

    // initialize extra shared mem
    char *s_extra = (char *)(s_type_group + n_groups);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    // initialize the shared memory array for communicating overlaps
    if (master && group == 0)
        {
        s_translate_accept_count = 0;
        s_translate_reject_count = 0;
        s_rotate_accept_count = 0;
        s_rotate_reject_count = 0;
        s_overlap_checks = 0;
        s_overlap_err_count = 0;
        s_queue_size = 0;
        s_still_searching = 1;
        }
    if (master)
        {
        s_overlap[group] = 0;
        }

    // identify the active cell that this thread handles
    unsigned int active_cell_idx = 0;
    if (gridDim.y > 1)
        {
        // if gridDim.y > 1, then the fermi workaround is in place, index blocks on a 2D grid
        active_cell_idx = (blockIdx.x + blockIdx.y * 65535) * n_groups + group;
        }
    else
        {
        active_cell_idx = blockIdx.x * n_groups + group;
        }


    // this thread is inactive if it indexes past the end of the active cell list
    if (active_cell_idx >= n_active_cells)
        active = false;

    // pull in the index of our cell
    unsigned int my_cell = 0;
    unsigned int my_cell_size = 0;
    if (active)
        {
        my_cell = d_cell_set[active_cell_idx];
        my_cell_size = d_cell_size[my_cell];
        }

    // need to deactivate if there are no particles in this cell
    if (my_cell_size == 0)
        active = false;

    __syncthreads();

    // initial implementation just moves one particle per cell (nselect=1).
    // these variables are ugly, but needed to get the updated quantities outside of the scope
    unsigned int i;
    unsigned int overlap_checks = 0;
    bool move_type_translate = false;
    bool move_active = true;
    int ignore_stats = 0;

    if (active)
        {
        // one RNG per cell
        hoomd::detail::Saru rng(my_cell, seed+select, timestep);

        // select one of the particles randomly from the cell
        unsigned int my_cell_offset = rand_select(rng, my_cell_size-1);
        i = tex1Dfetch(cell_idx_tex, cli(my_cell_offset, my_cell));

        // read in the position and orientation of our particle.
        Scalar4 postype_i = texFetchScalar4(d_postype, postype_tex, i);
        Scalar4 orientation_i = make_scalar4(1,0,0,0);

        unsigned int typ_i = __scalar_as_int(postype_i.w);
        Shape shape_i(quat<Scalar>(orientation_i), s_params[typ_i]);

        if (shape_i.hasOrientation())
            orientation_i = texFetchScalar4(d_orientation, orientation_tex, i);

        shape_i.orientation = quat<Scalar>(orientation_i);

        // if this looks funny, that is because it is. Using ignore_stats as a bool setting ignore_stats = ...
        // causes a compiler bug.
        if (shape_i.ignoreStatistics())
            ignore_stats = 1;

        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        // for domain decomposition simulations, we need to leave all particles in the inactive region alone
        // in order to avoid even more divergence, this is done by setting the move_active flag
        // overlap checks are still processed, but the final move acceptance will be skipped
        if (domain_decomposition && !isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
            move_active = false;

        // make the move
        unsigned int move_type_select = rng.u32() & 0xffff;
        move_type_translate = !shape_i.hasOrientation() || (move_type_select < move_ratio);

        if (move_type_translate)
            {
            move_translate(pos_i, rng, s_d[typ_i], dim);

            // need to reject any move that puts the particle in the inactive region
            if (domain_decomposition && !isActive(vec_to_scalar3(pos_i), box, ghost_fraction))
                move_active = false;
            }
        else
            {
            move_rotate(shape_i.orientation, rng, s_a[typ_i], dim);
            }

        // stash the trial move in shared memory so that other threads in this block can process overlap checks
        if (master)
            {
            s_pos_group[group] = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
            s_type_group[group] = typ_i;
            s_orientation_group[group] = quat_to_scalar4(shape_i.orientation);
            }
        }

    // sync so that s_postype_group and s_orientation are available before other threads might process overlap checks
    __syncthreads();

    // counters to track progress through the loop over potential neighbors
    unsigned int excell_size;
    unsigned int k = offset;
    if (active)
        {
        excell_size = d_excell_size[my_cell];
        overlap_checks += excell_size;
        }

    // loop while still searching
    while (s_still_searching)
        {
        // stage 1, fill the queue.
        // loop through particles in the excell list and add them to the queue if they pass the circumsphere check

        // active threads add to the queue
        if (active)
            {
            // prefetch j
            unsigned int j, next_j = 0;
            if (k < excell_size)
                {
                #if (__CUDA_ARCH__ > 300)
                next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);
                #else
                next_j = d_excell_idx[excli(k, my_cell)];
                #endif
                }

            // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
            // and as long as no overlaps have been found
            while (!s_overlap[group] && s_queue_size < max_queue_size && k < excell_size)
                {
                if (k < excell_size)
                    {
                    Scalar4 postype_j;
                    Scalar4 orientation_j;
                    vec3<Scalar> r_ij;

                    // build some shapes, but we only need them to get diameters, so don't load orientations
                    // build shape i from shared memory
                    Scalar3 pos_i = s_pos_group[group];
                    Shape shape_i(quat<Scalar>(), s_params[s_type_group[group]]);

                    // prefetch next j
                    k += group_size;
                    j = next_j;

                    if (k < excell_size)
                        {
                        #if (__CUDA_ARCH__ > 300)
                        next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);
                        #else
                        next_j = d_excell_idx[excli(k, my_cell)];
                        #endif
                        }

                    // read in position, and orientation of neighboring particle
                    postype_j = texFetchScalar4(d_postype, postype_tex, j);
                    Shape shape_j(quat<Scalar>(orientation_j), s_params[__scalar_as_int(postype_j.w)]);

                    // put particle j into the coordinate system of particle i
                    r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
                    r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

                    // test circumsphere overlap
                    OverlapReal rsq = dot(r_ij,r_ij);
                    OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                    if (i != j && rsq*OverlapReal(4.0) <= DaDb * DaDb)
                        {
                        // add this particle to the queue
                        unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                        if (insert_point < max_queue_size)
                            {
                            s_queue_gid[insert_point] = group;
                            s_queue_j[insert_point] = j;
                            }
                        else
                            {
                            // or back up if the queue is already full
                            // we will recheck and insert this on the next time through
                            k -= group_size;
                            }
                        }

                    } // end if k < excell_size
                } // end while (s_queue_size < max_queue_size && k < excell_size)
            } // end if active

        // sync to make sure all threads in the block are caught up
        __syncthreads();

        // when we get here, all threads have either finished their list, or encountered a full queue
        // either way, it is time to process overlaps
        // need to clear the still searching flag and sync first
        if (master && group == 0)
            s_still_searching = 0;

        unsigned int tidx_1d = offset + group_size*group;  // z component is for Shape parallelism

        // max_queue_size is always <= block size, so we just need an if here
        if (tidx_1d < min(s_queue_size, max_queue_size))
            {
            // need to extract the overlap check to perform out of the shared mem queue
            unsigned int check_group = s_queue_gid[tidx_1d];
            unsigned int check_j = s_queue_j[tidx_1d];
            Scalar4 postype_j;
            Scalar4 orientation_j;
            vec3<Scalar> r_ij;

            // build shape i from shared memory
            Scalar3 pos_i = s_pos_group[check_group];
            unsigned int type_i = s_type_group[check_group];
            Shape shape_i(quat<Scalar>(s_orientation_group[check_group]), s_params[type_i]);

            // build shape j from global memory
            postype_j = texFetchScalar4(d_postype, postype_tex, check_j);
            orientation_j = make_scalar4(1,0,0,0);
            unsigned int type_j = __scalar_as_int(postype_j.w);
            Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
            if (shape_j.hasOrientation())
                shape_j.orientation = quat<Scalar>(texFetchScalar4(d_orientation, orientation_tex, check_j));

            // put particle j into the coordinate system of particle i
            r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
            r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

            if (s_check_overlaps[overlap_idx(type_i, type_j)] && test_overlap(r_ij, shape_i, shape_j, err_count))
                {
                atomicAdd(&s_overlap[check_group], 1);
                }
            }

        // threads that need to do more looking set the still_searching flag
        __syncthreads();
        if (master && group == 0)
            s_queue_size = 0;

        if (active && !s_overlap[group] && k < excell_size)
            atomicAdd(&s_still_searching, 1);
        __syncthreads();

        } // end while (s_still_searching)

    // update the data if accepted
    if (master)
        {
        if (active && move_active)
            {
            // first need to check if the particle remains in its cell
            Scalar3 xnew_i = s_pos_group[group];
            unsigned int new_cell = computeParticleCell(xnew_i, box, ghost_width, cell_dim, ci);
            bool accepted=true;
            if (s_overlap[group])
                accepted=false;
            if (new_cell != my_cell)
                accepted=false;

            if (accepted)
                {
                // write out the updated position and orientation
                d_postype[i] = make_scalar4(xnew_i.x, xnew_i.y, xnew_i.z, __int_as_scalar(s_type_group[group]));
                d_orientation[i] = s_orientation_group[group];
                }

            if (d_active_cell_accept)
                {
                // store particle index
                d_active_cell_ptl_idx[active_cell_idx] = i;
                }

            if (d_active_cell_accept)
                {
                // store accept flag
                d_active_cell_accept[active_cell_idx] = accepted ? 1 : 0;
                }

            if (d_active_cell_move_type_translate)
                {
                // store move type
                d_active_cell_move_type_translate[active_cell_idx] = move_type_translate ? 1 : 0;
                }

            // if an auxillary array was provided, defer writing out statistics
            if (d_active_cell_ptl_idx)
                {
                ignore_stats = 1;
                }

            if (!ignore_stats && accepted && move_type_translate)
                atomicAdd(&s_translate_accept_count, 1);
            if (!ignore_stats && accepted && !move_type_translate)
                atomicAdd(&s_rotate_accept_count, 1);
            if (!ignore_stats && !accepted && move_type_translate)
                atomicAdd(&s_translate_reject_count, 1);
            if (!ignore_stats && !accepted && !move_type_translate)
                atomicAdd(&s_rotate_reject_count, 1);
            }
        else // active && move_active
            {
            if (d_active_cell_ptl_idx && active_cell_idx < n_active_cells)
                {
                // indicate that no particle was selected
                d_active_cell_ptl_idx[active_cell_idx] = UINT_MAX;
                }
            }

        // count the overlap checks
        atomicAdd(&s_overlap_checks, overlap_checks);
        }

    if (err_count > 0)
        atomicAdd(&s_overlap_err_count, err_count);

    __syncthreads();

    // final tally into global mem
    if (master && group == 0)
        {
        atomicAdd(&d_counters->translate_accept_count, s_translate_accept_count);
        atomicAdd(&d_counters->translate_reject_count, s_translate_reject_count);
        atomicAdd(&d_counters->rotate_accept_count, s_rotate_accept_count);
        atomicAdd(&d_counters->rotate_reject_count, s_rotate_reject_count);
        atomicAdd(&d_counters->overlap_checks, s_overlap_checks);
        atomicAdd(&d_counters->overlap_err_count, s_overlap_err_count);
        }
    }

//! Check for overlaps given a queue
//! This kernel is supposed to be launched as a single thread block from its parent kernel
template<class Shape>
__global__ void gpu_hpmc_check_overlaps_kernel(
                bool old_config,
                unsigned int i_queue,
                unsigned int offset,
                const Scalar4 *d_postype,
                const Scalar4 *d_orientation,
                hpmc_counters_t *d_counters,
                unsigned int num_types,
                const unsigned int *d_check_overlaps,
                const Index2D overlap_idx,
                const BoxDim box,
                const typename Shape::param_type *d_params,
                unsigned int max_extra_bytes,
                Index2D queue_idx,
                const unsigned int *d_queue_active_cell_idx,
                const Scalar4 *d_queue_postype,
                const Scalar4 *d_queue_orientation,
                const unsigned int *d_queue_excell_idx,
                const unsigned int *d_excell_idx,
                unsigned int *d_excell_overlap,
                unsigned int *d_cell_overlaps,
                bool load_shared,
                const Scalar4 *d_trial_postype,
                const Scalar4 *d_trial_orientation)
    {
    // fetch from queue
    unsigned int qidx = queue_idx(i_queue,offset);

    #if 0
    //no early exit for now
    unsigned int active_cell_idx = d_queue_active_cell_idx[qidx];

    // catch an opportunity at early exit using a global mem race
    __shared__ bool s_early_exit;
    if (threadIdx.x == 0 && threadIdx.y == 0)
        s_early_exit = false;
    __syncthreads();

    if (d_cell_overlaps[active_cell_idx])
        {
        s_early_exit = true;
        }
    __syncthreads();

    // if early exit, the entire thread block has to leave at once
    // so that we are not hitting an incomplete barrier below
    if (s_early_exit)
        return;
    #endif

    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];

    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    unsigned int *s_check_overlaps = (unsigned int *) (s_params + num_types);

    // copy over parameters one int per thread for fast loads
        {
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;

        #if (__CUDA_ARCH__ > 300)
        unsigned int param_size = num_types*sizeof(typename Shape::param_type) / sizeof(int);
        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = __ldg(&((int *)d_params)[cur_offset + tidx]);
                }
            }
        #endif

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

    if (load_shared)
        {
        // initialize extra shared mem
        char *s_extra = (char *)(s_check_overlaps + overlap_idx.getNumElements());

        unsigned int available_bytes = max_extra_bytes;
        for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
            s_params[cur_type].load_shared(s_extra, available_bytes);

        __syncthreads();
        }

    // load new configuration of particle i from queue
    Scalar4 postype_i = d_queue_postype[qidx];
    Scalar4 orientation_i = d_queue_orientation[qidx];
    unsigned int excell_idx = d_queue_excell_idx[qidx];

    // fetch particle index from expanded cells
    unsigned int j = d_excell_idx[excell_idx];

    unsigned int type_i = __scalar_as_int(postype_i.w);

    // perform overlap check
    Shape shape_i(quat<Scalar>(), s_params[type_i]);
    if (shape_i.hasOrientation())
        shape_i.orientation = quat<Scalar>(orientation_i);

    // build shape j from global memory
    Scalar4 postype_j = old_config ? d_postype[j] : d_trial_postype[j];

    unsigned int type_j = __scalar_as_int(postype_j.w);
    Shape shape_j(quat<Scalar>(), s_params[type_j]);
    if (shape_j.hasOrientation())
        shape_j.orientation = quat<Scalar>(old_config ? d_orientation[j] : d_trial_orientation[j]);

    // put particle j into the coordinate system of particle i
    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(postype_i);
    r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

    unsigned int err_count = 0;

    if (s_check_overlaps[overlap_idx(type_i, type_j)]
        && test_overlap(r_ij, shape_i, shape_j, err_count))
        {
        // NOTE if we are only checking against the old config, we could provide an early
        // exit code path here

        //d_cell_overlaps[active_cell_idx] = 1;

        d_excell_overlap[excell_idx] = old_config ? OVERLAP_IN_OLD_CONFIG : OVERLAP_IN_NEW_CONFIG;
        }

    if (err_count > 0)
        atomicAdd(&d_counters->overlap_err_count, err_count);
    }

//! Kernel driver for gpu_update_hpmc_kernel()
/*! \param args Bundled arguments
    \param d_params Per-type shape parameters
    \returns Error codes generated by any CUDA calls, or cudaSuccess when there is no error

    This templatized method is the kernel driver for HPMC update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup hpmc_kernels
*/
template< class Shape >
cudaError_t gpu_hpmc_update(const hpmc_args_t& args, const typename Shape::param_type *params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_counters);
    assert(args.d_cell_idx);
    assert(args.d_cell_size);
    assert(args.d_excell_idx);
    assert(args.d_excell_size);
    assert(args.d_cell_set);
    assert(args.d_d);
    assert(args.d_a);
    assert(args.d_check_overlaps);
    assert(args.group_size >= 1);
    assert(args.stride >= 1);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static int sm = -1;
    static cudaFuncAttributes attr;
    if (max_block_size == -1)
        {
        cudaFuncGetAttributes(&attr, gpu_hpmc_mpmc_kernel<Shape>);
        max_block_size = attr.maxThreadsPerBlock;
        sm = attr.binaryVersion;
        }

    // might need to modify group_size to make the kernel runnable
    unsigned int group_size = args.group_size;

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

    // the new block size might not fit the group size and stride, decrease group size until it is
    group_size = args.group_size;

    unsigned int stride = min(block_size, args.stride);
    while (stride*group_size > block_size)
        {
        group_size--;
        }

    unsigned int n_groups = block_size / (group_size * stride);
    unsigned int max_queue_size = n_groups*group_size;
    unsigned int shared_bytes = n_groups * (sizeof(unsigned int)*2 + sizeof(Scalar4) + sizeof(Scalar3)) +
                                max_queue_size*(sizeof(unsigned int) + sizeof(unsigned int)) +
                                args.num_types * (sizeof(typename Shape::param_type) + 2*sizeof(Scalar)) +
                                args.overlap_idx.getNumElements() * sizeof(unsigned int);

    unsigned int min_shared_bytes = args.num_types * (sizeof(typename Shape::param_type) + 2*sizeof(Scalar)) +
               args.overlap_idx.getNumElements() * sizeof(unsigned int);

    if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
        throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

    while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
        {
        block_size -= args.devprop.warpSize;
        if (block_size == 0)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel");

        // the new block size might not fit the group size and stride, decrease group size until it is
        stride = args.stride;
        group_size = args.group_size;

        unsigned int stride = min(block_size, args.stride);
        while (stride*group_size > block_size)
            {
            group_size--;
            }

        n_groups = block_size / (group_size * stride);
        max_queue_size = n_groups*group_size;
        shared_bytes = n_groups * (sizeof(unsigned int)*2 + sizeof(Scalar4) + sizeof(Scalar3)) +
                       max_queue_size*(sizeof(unsigned int) + sizeof(unsigned int)) +
                       min_shared_bytes;
        }

    static unsigned int base_shared_bytes = UINT_MAX;
    bool shared_bytes_changed = base_shared_bytes != shared_bytes + attr.sharedSizeBytes;
    base_shared_bytes = shared_bytes + attr.sharedSizeBytes;

    unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - base_shared_bytes;
    static unsigned int extra_bytes = UINT_MAX;
    if (extra_bytes == UINT_MAX || args.update_shape_param || shared_bytes_changed)
        {
        // required for memory coherency
        cudaDeviceSynchronize();

        // determine dynamically requested shared memory
        char *ptr = (char *)nullptr;
        unsigned int available_bytes = max_extra_bytes;
        for (unsigned int i = 0; i < args.num_types; ++i)
            {
            params[i].load_shared(ptr, available_bytes);
            }
        extra_bytes = max_extra_bytes - available_bytes;
        }

    shared_bytes += extra_bytes;

    // setup the grid to run the kernel
    dim3 threads;
    if (Shape::isParallel())
        {
        // use three-dimensional thread-layout with blockDim.z < 64
        threads = dim3(stride, group_size, n_groups);
        }
    else
        {
        threads = dim3(group_size, n_groups,1);
        }

    dim3 grid( args.n_active_cells / n_groups + 1, 1, 1);

    // hack to enable grids of more than 65k blocks
    if (sm < 30 && grid.x > 65535)
        {
        grid.y = grid.x / 65535 + 1;
        grid.x = 65535;
        }

    // bind the textures
    postype_tex.normalized = false;
    postype_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, postype_tex, args.d_postype, sizeof(Scalar4)*args.max_n);
    if (error != cudaSuccess)
        return error;

    if (args.has_orientation)
        {
        orientation_tex.normalized = false;
        orientation_tex.filterMode = cudaFilterModePoint;
        error = cudaBindTexture(0, orientation_tex, args.d_orientation, sizeof(Scalar4)*args.max_n);
        if (error != cudaSuccess)
            return error;
        }

    cell_idx_tex.normalized = false;
    cell_idx_tex.filterMode = cudaFilterModePoint;
    error = cudaBindTexture(0, cell_idx_tex, args.d_cell_idx, sizeof(Scalar4)*args.cli.getNumElements());
    if (error != cudaSuccess)
        return error;

    gpu_hpmc_mpmc_kernel<Shape><<<grid, threads, shared_bytes, args.stream>>>(args.d_postype,
                                                                 args.d_orientation,
                                                                 args.d_counters,
                                                                 args.d_cell_idx,
                                                                 args.d_cell_size,
                                                                 args.d_excell_idx,
                                                                 args.d_excell_size,
                                                                 args.ci,
                                                                 args.cli,
                                                                 args.excli,
                                                                 args.cell_dim,
                                                                 args.ghost_width,
                                                                 args.d_cell_set,
                                                                 args.n_active_cells,
                                                                 args.N,
                                                                 args.num_types,
                                                                 args.seed,
                                                                 args.d_d,
                                                                 args.d_a,
                                                                 args.d_check_overlaps,
                                                                 args.overlap_idx,
                                                                 args.move_ratio,
                                                                 args.timestep,
                                                                 args.dim,
                                                                 args.box,
                                                                 args.select,
                                                                 args.ghost_fraction,
                                                                 args.domain_decomposition,
                                                                 args.d_active_cell_ptl_idx,
                                                                 args.d_active_cell_accept,
                                                                 args.d_active_cell_move_type_translate,
                                                                 params,
                                                                 max_queue_size,
                                                                 max_extra_bytes);

    return cudaSuccess;
    }

//! Propose trial moves
template< class Shape >
__global__ void gpu_hpmc_moves_kernel(Scalar4 *d_postype,
                                     Scalar4 *d_orientation,
                                     hpmc_counters_t *d_counters,
                                     const unsigned int *d_cell_idx,
                                     const unsigned int *d_cell_size,
                                     const Index3D ci,
                                     const Index2D cli,
                                     const uint3 cell_dim,
                                     const Scalar3 ghost_width,
                                     const unsigned int *d_cell_set,
                                     const unsigned int num_types,
                                     const unsigned int seed,
                                     const Scalar* d_d,
                                     const Scalar* d_a,
                                     const unsigned int move_ratio,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const unsigned int select,
                                     const Scalar3 ghost_fraction,
                                     const bool domain_decomposition,
                                     Index2D csi,
                                     Scalar4 *d_trial_postype,
                                     Scalar4 *d_trial_orientation,
                                     unsigned int *d_trial_updated,
                                     unsigned int *d_trial_move_type_translate,
                                     const typename Shape::param_type *d_params)
    {
    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];

    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar *s_d = (Scalar *)(s_params + num_types);
    Scalar *s_a = (Scalar *)(s_d + num_types);

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

        for (unsigned int cur_offset = 0; cur_offset < num_types; cur_offset += block_size)
            {
            if (cur_offset + tidx < num_types)
                {
                s_a[cur_offset + tidx] = d_a[cur_offset + tidx];
                s_d[cur_offset + tidx] = d_d[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    // identify the cell that this thread handles
    unsigned int cell_set_idx = blockIdx.x*blockDim.x + threadIdx.x;

    // return early if we are not handling a cell
    if (cell_set_idx >= csi.getNumElements())
        return;

    // pull in the index of our cell
    unsigned int my_cell = d_cell_set[cell_set_idx];
    unsigned int my_cell_size = d_cell_size[my_cell];

    // return early if there are no particles in this cell
    if (my_cell_size == 0)
        {
        // flag as inactive in global mem
        d_trial_move_type_translate[my_cell] = UINT_MAX;
        return;
        }

    bool move_type_translate = false;
    bool move_active = true;

    // one RNG per cell
    hoomd::detail::Saru rng(my_cell, seed+select, timestep);

    // select one of the particles randomly from the cell
    unsigned int my_cell_offset = rand_select(rng, my_cell_size-1);
    unsigned int i = d_cell_idx[cli(my_cell_offset, my_cell)];

    // read in the position and orientation of our particle.
    Scalar4 postype_i = texFetchScalar4(d_postype, postype_tex, i);
    Scalar4 orientation_i = make_scalar4(1,0,0,0);

    unsigned int typ_i = __scalar_as_int(postype_i.w);
    Shape shape_i(quat<Scalar>(orientation_i), s_params[typ_i]);

    if (shape_i.hasOrientation())
        orientation_i = texFetchScalar4(d_orientation, orientation_tex, i);

    shape_i.orientation = quat<Scalar>(orientation_i);

    vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

    // for domain decomposition simulations, we need to leave all particles in the inactive region alone
    // in order to avoid even more divergence, this is done by setting the move_active flag
    // overlap checks are still processed, but the final move acceptance will be skipped
    if (domain_decomposition && !isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
        move_active = false;

    // make the move
    unsigned int move_type_select = rng.u32() & 0xffff;
    move_type_translate = !shape_i.hasOrientation() || (move_type_select < move_ratio);

    if (move_type_translate)
        {
        move_translate(pos_i, rng, s_d[typ_i], dim);

        // need to reject any move that puts the particle in the inactive region
        if (domain_decomposition && !isActive(vec_to_scalar3(pos_i), box, ghost_fraction))
            move_active = false;
        }
    else
        {
        move_rotate(shape_i.orientation, rng, s_a[typ_i], dim);
        }

    // stash the trial move in shared memory so that other threads in this block can process overlap checks
    d_trial_postype[i] = make_scalar4(pos_i.x, pos_i.y, pos_i.z, __int_as_scalar(typ_i));
    d_trial_orientation[i] = quat_to_scalar4(shape_i.orientation);
    d_trial_updated[i] = 1;
    d_trial_move_type_translate[my_cell] = move_active ? move_type_translate : UINT_MAX;
    }

//! Kernel driver for gpu_hpmc_moves_kernel
template< class Shape >
cudaError_t gpu_hpmc_moves(const hpmc_args_t& args, const typename Shape::param_type *params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_counters);
    assert(args.d_cell_idx);
    assert(args.d_cell_size);
    assert(args.d_excell_idx);
    assert(args.d_excell_size);
    assert(args.d_cell_set);
    assert(args.d_d);
    assert(args.d_a);
    assert(args.d_check_overlaps);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static cudaFuncAttributes attr;
    if (max_block_size == -1)
        {
        cudaFuncGetAttributes(&attr, gpu_hpmc_moves_kernel<Shape>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);
    unsigned int shared_bytes = args.num_types * (sizeof(typename Shape::param_type) + 2*sizeof(Scalar));

    // setup the grid to run the kernel
    dim3 threads( block_size, 1, 1);
    dim3 grid(args.csi.getNumElements()/block_size+1,1,1);

    // reset has_been_updated flags
    cudaMemsetAsync(args.d_trial_updated, 0, sizeof(unsigned int)*args.N, args.stream);

    gpu_hpmc_moves_kernel<Shape><<<grid, threads, shared_bytes, args.stream>>>(args.d_postype,
                                                                 args.d_orientation,
                                                                 args.d_counters,
                                                                 args.d_cell_idx,
                                                                 args.d_cell_size,
                                                                 args.ci,
                                                                 args.cli,
                                                                 args.cell_dim,
                                                                 args.ghost_width,
                                                                 args.d_cell_set,
                                                                 args.num_types,
                                                                 args.seed,
                                                                 args.d_d,
                                                                 args.d_a,
                                                                 args.move_ratio,
                                                                 args.timestep,
                                                                 args.dim,
                                                                 args.box,
                                                                 args.select,
                                                                 args.ghost_fraction,
                                                                 args.domain_decomposition,
                                                                 args.csi,
                                                                 args.d_trial_postype,
                                                                 args.d_trial_orientation,
                                                                 args.d_trial_updated,
                                                                 args.d_trial_move_type_translate,
                                                                 params
                                                            );

    return cudaSuccess;
    }

//! Kernel to schedule overlap checks between particles
template< class Shape >
__global__ void gpu_hpmc_schedule_overlaps_kernel(Scalar4 *d_postype,
                                     Scalar4 *d_orientation,
                                     hpmc_counters_t *d_counters,
                                     const unsigned int *d_cell_idx,
                                     const unsigned int *d_cell_size,
                                     const unsigned int *d_excell_idx,
                                     const unsigned int *d_excell_cell_set,
                                     unsigned int *d_excell_overlap,
                                     const unsigned int *d_excell_size,
                                     const Index3D ci,
                                     const Index2D cli,
                                     const Index2D excli,
                                     const uint3 cell_dim,
                                     const Scalar3 ghost_width,
                                     const unsigned int *d_cell_set,
                                     const unsigned int n_active_cells,
                                     const unsigned int N,
                                     const unsigned int num_types,
                                     const unsigned int seed,
                                     const unsigned int *d_check_overlaps,
                                     const Index2D overlap_idx,
                                     const unsigned int move_ratio,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const unsigned int select,
                                     const Scalar3 ghost_fraction,
                                     const bool domain_decomposition,
                                     const typename Shape::param_type *d_params,
                                     unsigned int max_queue_size,
                                     unsigned int extra_bytes,
                                     unsigned int max_extra_bytes,
                                     Index2D queue_idx,
                                     unsigned int *d_queue_active_cell_idx,
                                     Scalar4 *d_queue_postype,
                                     Scalar4 *d_queue_orientation,
                                     unsigned int *d_queue_excell_idx,
                                     unsigned int *d_cell_overlaps,
                                     unsigned int max_gmem_queue_size,
                                     bool check_cuda_errors,
                                     unsigned int child_block_size,
                                     bool load_shared,
                                     Index2D csi,
                                     Scalar4 *d_trial_postype,
                                     Scalar4 *d_trial_orientation,
                                     const unsigned int *d_trial_updated,
                                     const unsigned int *d_trial_move_type_translate,
                                     const unsigned int *d_update_order
                                     )
    {
    // flags to tell what type of thread we are
    bool active = true;

    unsigned int group = threadIdx.y;
    unsigned int offset = threadIdx.x;
    unsigned int group_size = blockDim.x;
    bool master = (offset == 0);
    unsigned int n_groups = blockDim.y;

    // shared arrays for per type pair parameters
    __shared__ unsigned int s_overlap_checks;

    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;

    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];

    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar4 *s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    unsigned int *s_update_order_group = (unsigned int *)(s_pos_group + n_groups);
    unsigned int *s_queue_type_j =   (unsigned int*)(s_update_order_group + n_groups);
    unsigned int *s_queue_excell_idx = (unsigned int *)(s_queue_type_j + max_queue_size);
    unsigned int *s_overlap =   (unsigned int*)(s_queue_excell_idx + max_queue_size);
    unsigned int *s_queue_gid = (unsigned int*)(s_overlap + n_groups);
    unsigned int *s_type_group = (unsigned int*)(s_queue_gid + max_queue_size);

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

    // initialize the shared memory array for communicating overlaps
    if (master && group == 0)
        {
        s_overlap_checks = 0;
        s_queue_size = 0;
        s_still_searching = 1;
        }
    if (master)
        {
        s_overlap[group] = 0;
        }

    // identify the active cell that this thread handles
    unsigned int cell_set_idx = blockIdx.x * n_groups + group;

    // this thread is inactive if it indexes past the end of the active cell list
    if (cell_set_idx >= csi.getNumElements())
        active = false;

    // get the active cell and the current cell set
    unsigned int cur_set = cell_set_idx / csi.getW();
    unsigned int active_cell_idx = cell_set_idx % csi.getW();

    // pull in the index of our cell
    unsigned int my_cell = 0;
    unsigned int my_cell_size = 0;
    if (active)
        {
        my_cell = d_cell_set[csi(active_cell_idx,cur_set)];
        my_cell_size = d_cell_size[my_cell];
        }

    // need to deactivate if there are no particles in this cell
    if (my_cell_size == 0)
        active = false;

    __syncthreads();

    // initial implementation just moves one particle per cell (nselect=1).
    // these variables are ugly, but needed to get the updated quantities outside of the scope
    unsigned int i = UINT_MAX;
    unsigned int overlap_checks = 0;
    if (active && master)
        {
        // one RNG per cell, reproduce the random number from the trial move kernel
        hoomd::detail::Saru rng(my_cell, seed+select, timestep);
        unsigned int my_cell_offset = rand_select(rng, my_cell_size-1);
        i = d_cell_idx[cli(my_cell_offset, my_cell)];

        Scalar4 trial_postype = d_trial_postype[i];
        s_pos_group[group] = make_scalar3(trial_postype.x, trial_postype.y, trial_postype.z);
        s_type_group[group] = __scalar_as_int(trial_postype.w);
        s_orientation_group[group] = d_trial_orientation[i];
        s_update_order_group[group] = d_update_order[cur_set];
        }

    // sync so that s_postype_group and s_orientation are available before other threads might process overlap checks
    __syncthreads();

    bool move_active = true;

    if (active)
        {
        unsigned int trial_move_type_translate = d_trial_move_type_translate[d_cell_set[csi(active_cell_idx,cur_set)]];
        if (trial_move_type_translate == UINT_MAX)
            move_active = false;
        }

    #if 0
    if (active && master)
        {
        // reset overlaps flag

        // NOTE currently not used
        d_cell_overlaps[active_cell_idx] = 0;
        }
    #endif

    // counters to track progress through the loop over potential neighbors
    unsigned int excell_size;
    unsigned int k = offset;
    if (active)
        {
        excell_size = d_excell_size[my_cell];
        overlap_checks += excell_size;
        }

    // current offset into global mem work queue
    __shared__ unsigned int s_queue_offset;
    __shared__ bool s_gmem_queue_full;

    if (master && group == 0)
        {
        s_queue_offset = 0;
        s_gmem_queue_full = false;
        }

    __syncthreads();

    #if (__CUDA_ARCH__ > 300)
    // create a device stream
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (check_cuda_errors)
        {
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess)
            {
            printf("Error creating device stream: %s\n", cudaGetErrorString(status));
            }
        }
    #endif

    // loop while still searching
    while (s_still_searching)
        {
        // stage 1, fill the queue.
        // loop through particles in the excell list and add them to the queue if they pass the circumsphere check

        // active threads add to the queue
        if (active)
            {
            // prefetch j
            unsigned int j, next_j = 0;
            unsigned int excell_idx, next_excell_idx = 0;

            if (k < excell_size)
                {
                #if (__CUDA_ARCH__ > 300)
                next_excell_idx = excli(k, my_cell);
                next_j = __ldg(&d_excell_idx[next_excell_idx]);
                #endif
                }

            // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
            // and as long as no overlaps have been found
            while (!s_overlap[group] && s_queue_size < max_queue_size && k < excell_size && !s_gmem_queue_full)
                {
                if (k < excell_size)
                    {
                    Scalar4 postype_j;
                    Scalar4 orientation_j;
                    vec3<Scalar> r_ij;

                    // build some shapes, but we only need them to get diameters, so don't load orientations
                    // build shape i from shared memory
                    Scalar3 pos_i = s_pos_group[group];
                    Shape shape_i(quat<Scalar>(), s_params[s_type_group[group]]);

                    // prefetch next j
                    k += group_size;
                    j = next_j;
                    excell_idx = next_excell_idx;

                    if (k < excell_size)
                        {
                        #if (__CUDA_ARCH__ > 300)
                        next_excell_idx = excli(k, my_cell);
                        next_j = __ldg(&d_excell_idx[next_excell_idx]);
                        #endif
                        }

                    // read in position, and orientation of neighboring particle
                    postype_j = texFetchScalar4(d_postype, postype_tex, j);
                    unsigned int type_j = __scalar_as_int(postype_j.w);
                    Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);

                    // put particle j into the coordinate system of particle i
                    r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
                    r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

                    // test circumsphere overlap
                    OverlapReal rsq = dot(r_ij,r_ij);
                    OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                    if (i != j && rsq*OverlapReal(4.0) <= DaDb * DaDb)
                        {
                        // add this particle to the queue
                        unsigned int insert_point = atomicAdd(&s_queue_size, 1);
                        if (insert_point + s_queue_offset >= max_gmem_queue_size)
                            s_gmem_queue_full = true;

                        if (insert_point < max_queue_size && !s_gmem_queue_full)
                            {
                            // global mem queue
                            unsigned int qidx = queue_idx(blockIdx.x, s_queue_offset + insert_point);
                            d_queue_active_cell_idx[qidx] = active_cell_idx;
                            Scalar3 pos_i = s_pos_group[group];
                            d_queue_postype[qidx] = make_scalar4(pos_i.x, pos_i.y, pos_i.z, __int_as_scalar(s_type_group[group]));
                            d_queue_orientation[qidx] = s_orientation_group[group];
                            d_queue_excell_idx[qidx] = excell_idx;

                            // shared mem queue
                            s_queue_gid[insert_point] = group;
                            s_queue_type_j[insert_point] = type_j;
                            s_queue_excell_idx[insert_point] = excell_idx;
                            }
                        else
                            {
                            // or back up if the queue is already full
                            // we will recheck and insert this on the next time through
                            k -= group_size;
                            }
                        }

                    } // end if k < excell_size
                } // end while (s_queue_size < max_queue_size && k < excell_size)
            } // end if active

        // sync to make sure all threads in the block are caught up
        __syncthreads();

        // when we get here, all threads have either finished their list, or encountered a full queue
        // either way, it is time to process overlaps
        // need to clear the still searching flag and sync first
        if (master && group == 0)
            s_still_searching = 0;

        unsigned int tidx_1d = offset + group_size*group;
        unsigned int n_overlap_checks = min(min(s_queue_size, max_queue_size),max_gmem_queue_size-s_queue_offset);
        if (tidx_1d < n_overlap_checks)
            {
            #if (__CUDA_ARCH__ > 300)
            // fetch neighbor info from the shared mem queueu
            unsigned int check_group = s_queue_gid[tidx_1d];
            unsigned int check_type_i = s_type_group[check_group];
            unsigned int check_type_j = s_queue_type_j[tidx_1d];
            unsigned int check_excell_idx = s_queue_excell_idx[tidx_1d];
            unsigned int check_update_order = d_update_order[d_excell_cell_set[check_excell_idx]];

            // reset the per-neighbor overlap flag
            d_excell_overlap[check_excell_idx] = 0;

            // get requested launch configuration
            Shape shape_i(quat<Scalar>(), s_params[check_type_i]);
            Shape shape_j(quat<Scalar>(), s_params[check_type_j]);

            unsigned int n_threads = get_num_requested_threads(shape_i, shape_j);

            // shape parallelism in .x index
            dim3 grid(n_threads/child_block_size+1, 1,1);
            dim3 threads(child_block_size, 1, 1);

            unsigned int shared_bytes = num_types*sizeof(Shape::param_type);
            shared_bytes += overlap_idx.getNumElements()*sizeof(unsigned int);
            if (load_shared) shared_bytes += extra_bytes;

            // check against old configuration of j
            gpu_hpmc_check_overlaps_kernel<Shape> <<<grid,threads,shared_bytes,stream>>>(
                false,
                blockIdx.x,
                s_queue_offset+tidx_1d,
                d_postype,
                d_orientation,
                d_counters,
                num_types,
                d_check_overlaps,
                overlap_idx,
                box,
                d_params,
                max_extra_bytes,
                queue_idx,
                d_queue_active_cell_idx,
                d_queue_postype,
                d_queue_orientation,
                d_queue_excell_idx,
                d_excell_idx,
                d_excell_overlap,
                d_cell_overlaps,
                load_shared,
                d_trial_postype,
                d_trial_orientation);

            if (check_cuda_errors)
                {
                cudaError_t status = cudaGetLastError();
                if (status != cudaSuccess)
                    {
                    printf("Error launching child kernel: %s\n", cudaGetErrorString(status));
                    }
                }

            // we only need to check against the new configuration of particle j
            // if that particle strictly precedes the current one in the chain and it has been updated
            unsigned int check_j = d_excell_idx[check_excell_idx];
            bool j_has_been_updated = d_trial_updated[check_j];
            if (j_has_been_updated && check_update_order < s_update_order_group[check_group])
                {
                // check against old configuration of j
                gpu_hpmc_check_overlaps_kernel<Shape> <<<grid,threads,shared_bytes,stream>>>(
                    true,
                    blockIdx.x,
                    s_queue_offset+tidx_1d,
                    d_postype,
                    d_orientation,
                    d_counters,
                    num_types,
                    d_check_overlaps,
                    overlap_idx,
                    box,
                    d_params,
                    max_extra_bytes,
                    queue_idx,
                    d_queue_active_cell_idx,
                    d_queue_postype,
                    d_queue_orientation,
                    d_queue_excell_idx,
                    d_excell_idx,
                    d_excell_overlap,
                    d_cell_overlaps,
                    load_shared,
                    d_trial_postype,
                    d_trial_orientation);

                if (check_cuda_errors)
                    {
                    cudaError_t status = cudaGetLastError();
                    if (status != cudaSuccess)
                        {
                        printf("Error launching child kernel: %s\n", cudaGetErrorString(status));
                        }
                    }
                }
            #endif
            }

        __syncthreads();
        if (master && group == 0)
            {
            // advance the parallel queue
            s_queue_offset += n_overlap_checks;
            }

        if (master && group==0 && s_gmem_queue_full)
            {
            #if (__CUDA_ARCH__ > 300)
            // catch up with child kernels
            cudaDeviceSynchronize();

            if (check_cuda_errors)
                {
                cudaError_t status = cudaGetLastError();
                if (status != cudaSuccess)
                    {
                    printf("Error on cudaDeviceSynchronize(): %s\n", cudaGetErrorString(status));
                    }
                }
            #endif

            // reset queue
            s_queue_offset = 0;
            s_gmem_queue_full = false;
            }

        __syncthreads();

        #if 0
        // early exit currently not implemented
        if (active && master)
            {
            // early exit via global mem race condition
            if (d_cell_overlaps[active_cell_idx])
                s_overlap[group] = true;
            }
        #endif

        // threads that need to do more looking set the still_searching flag
        __syncthreads();
        if (master && group == 0)
            s_queue_size = 0;

        if (active && !s_overlap[group] && k < excell_size)
            atomicAdd(&s_still_searching, 1);
        __syncthreads();

        } // end while (s_still_searching)

    if (master && active && move_active)
        {
        // count the overlap checks
        atomicAdd(&s_overlap_checks, overlap_checks);
        }

    __syncthreads();

    // final tally into global mem
    if (master && group == 0)
        {
        atomicAdd(&d_counters->overlap_checks, s_overlap_checks);
        }

    #if (__CUDA_ARCH__ > 300)
    cudaStreamDestroy(stream);
    #endif
    }


//! Kernel to evaluate the MC acceptance criteria, runs after overlaps have been determined
//! One grid per active cell set
template< class Shape >
__global__ void gpu_hpmc_accept_kernel(Scalar4 *d_postype,
                                     Scalar4 *d_orientation,
                                     hpmc_counters_t *d_counters,
                                     const unsigned int *d_cell_idx,
                                     const unsigned int *d_cell_size,
                                     const unsigned int *d_excell_idx,
                                     const unsigned int *d_excell_cell_set,
                                     const unsigned int *d_excell_overlap,
                                     const unsigned int *d_excell_size,
                                     const Index3D ci,
                                     const Index2D cli,
                                     const Index2D excli,
                                     const uint3 cell_dim,
                                     const Scalar3 ghost_width,
                                     const unsigned int *d_cell_set,
                                     const unsigned int n_active_cells,
                                     const unsigned int N,
                                     const unsigned int num_types,
                                     const unsigned int seed,
                                     const unsigned int move_ratio,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const unsigned int select,
                                     const Scalar3 ghost_fraction,
                                     const bool domain_decomposition,
                                     const typename Shape::param_type *d_params,
                                     Index2D csi,
                                     Scalar4 *d_trial_postype,
                                     Scalar4 *d_trial_orientation,
                                     unsigned int *d_trial_updated,
                                     unsigned int *d_trial_move_type_translate,
                                     unsigned int cur_set,
                                     unsigned int max_queue_size)
    {
    // flags to tell what type of thread we are
    bool active = true;

    unsigned int group = threadIdx.y;
    unsigned int offset = threadIdx.x;
    unsigned int group_size = blockDim.x;
    bool master = (offset == 0);
    unsigned int n_groups = blockDim.y;

    // shared arrays for per type pair parameters
    __shared__ unsigned int s_translate_accept_count;
    __shared__ unsigned int s_translate_reject_count;
    __shared__ unsigned int s_rotate_accept_count;
    __shared__ unsigned int s_rotate_reject_count;

    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;

    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];

    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar4 *s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    unsigned int *s_queue_excell_idx = (unsigned int *)(s_pos_group + n_groups);
    unsigned int *s_overlap =   (unsigned int*)(s_queue_excell_idx + max_queue_size);
    unsigned int *s_queue_gid = (unsigned int*)(s_overlap + n_groups);
    unsigned int *s_type_group = (unsigned int*)(s_queue_gid + max_queue_size);

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

    // initialize the shared memory array for communicating overlaps
    if (master && group == 0)
        {
        s_translate_accept_count = 0;
        s_translate_reject_count = 0;
        s_rotate_accept_count = 0;
        s_rotate_reject_count = 0;
        s_queue_size = 0;
        s_still_searching = 1;
        }
    if (master)
        {
        s_overlap[group] = 0;
        }

    // identify the active cell that this thread handles
    unsigned int active_cell_idx = blockIdx.x * n_groups + group;

    // this thread is inactive if it indexes past the end of the active cell list
    if (active_cell_idx >= csi.getW())
        active = false;

    // pull in the index of our cell
    unsigned int my_cell = 0;
    unsigned int my_cell_size = 0;
    if (active)
        {
        my_cell = d_cell_set[csi(active_cell_idx,cur_set)];
        my_cell_size = d_cell_size[my_cell];
        }

    // need to deactivate if there are no particles in this cell
    if (my_cell_size == 0)
        active = false;

    __syncthreads();

    // initial implementation just moves one particle per cell (nselect=1).
    // these variables are ugly, but needed to get the updated quantities outside of the scope
    unsigned int i = UINT_MAX;
    unsigned int overlap_checks = 0;
    bool move_type_translate = false;
    bool move_active = true;
    int ignore_stats = 0;

    if (active && master)
        {
        // one RNG per cell, reproduce the random number from the trial move kernel
        hoomd::detail::Saru rng(my_cell, seed+select, timestep);
        unsigned int my_cell_offset = rand_select(rng, my_cell_size-1);
        i = d_cell_idx[cli(my_cell_offset, my_cell)];

        Scalar4 trial_postype = d_trial_postype[i];
        s_pos_group[group] = make_scalar3(trial_postype.x, trial_postype.y, trial_postype.z);
        s_type_group[group] = __scalar_as_int(trial_postype.w);
        s_orientation_group[group] = d_trial_orientation[i];
        }

    // sync so that s_postype_group and s_orientation are available before other threads might process overlap checks
    __syncthreads();

    if (active)
        {
        unsigned int trial_move_type_translate = d_trial_move_type_translate[d_cell_set[csi(active_cell_idx,cur_set)]];
        if (trial_move_type_translate != UINT_MAX)
            {
            trial_move_type_translate = trial_move_type_translate;
            }
        else
            {
            move_active = false;
            }
        }

    #if 0
    if (active && master)
        {
        // reset overlaps flag

        // NOTE currently not used
        d_cell_overlaps[active_cell_idx] = 0;
        }
    #endif

    // counters to track progress through the loop over potential neighbors
    unsigned int excell_size;
    unsigned int k = offset;
    if (active)
        {
        excell_size = d_excell_size[my_cell];
        overlap_checks += excell_size;
        }

    __syncthreads();

    // loop while still searching
    while (s_still_searching)
        {
        // stage 1, fill the queue.
        // loop through particles in the excell list and add them to the queue if they pass the circumsphere check

        // active threads add to the queue
        if (active)
            {
            // prefetch j
            unsigned int j, next_j = 0;
            unsigned int excell_idx, next_excell_idx = 0;

            if (k < excell_size)
                {
                #if (__CUDA_ARCH__ > 300)
                next_excell_idx = excli(k, my_cell);
                next_j = __ldg(&d_excell_idx[next_excell_idx]);
                #endif
                }

            // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
            // and as long as no overlaps have been found
            while (!s_overlap[group] && s_queue_size < max_queue_size && k < excell_size)
                {
                if (k < excell_size)
                    {
                    Scalar4 postype_j;
                    Scalar4 orientation_j;
                    vec3<Scalar> r_ij;

                    // build some shapes, but we only need them to get diameters, so don't load orientations
                    // build shape i from shared memory
                    Scalar3 pos_i = s_pos_group[group];
                    Shape shape_i(quat<Scalar>(), s_params[s_type_group[group]]);

                    // prefetch next j
                    k += group_size;
                    j = next_j;
                    excell_idx = next_excell_idx;

                    if (k < excell_size)
                        {
                        #if (__CUDA_ARCH__ > 300)
                        next_excell_idx = excli(k, my_cell);
                        next_j = __ldg(&d_excell_idx[next_excell_idx]);
                        #endif
                        }

                    // read in position, and orientation of neighboring particle
                    postype_j = texFetchScalar4(d_postype, postype_tex, j);
                    unsigned int type_j = __scalar_as_int(postype_j.w);
                    Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);

                    // put particle j into the coordinate system of particle i
                    r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
                    r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

                    // test circumsphere overlap
                    OverlapReal rsq = dot(r_ij,r_ij);
                    OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                    if (i != j && rsq*OverlapReal(4.0) <= DaDb * DaDb)
                        {
                        // add this particle to the queue
                        unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                        if (insert_point < max_queue_size)
                            {
                            // shared mem queue
                            s_queue_gid[insert_point] = group;
                            s_queue_excell_idx[insert_point] = excell_idx;
                            }
                        else
                            {
                            // or back up if the queue is already full
                            // we will recheck and insert this on the next time through
                            k -= group_size;
                            }
                        }

                    } // end if k < excell_size
                } // end while (s_queue_size < max_queue_size && k < excell_size)
            } // end if active

        // sync to make sure all threads in the block are caught up
        __syncthreads();

        // when we get here, all threads have either finished their list, or encountered a full queue
        // either way, it is time to process overlaps
        // need to clear the still searching flag and sync first
        if (master && group == 0)
            s_still_searching = 0;

        unsigned int tidx_1d = offset + group_size*group;
        if (tidx_1d < min(s_queue_size, max_queue_size))
            {
            // use previously determined result from overlap checks
            unsigned int check_group = s_queue_gid[tidx_1d];
            unsigned int check_excell_idx = s_queue_excell_idx[tidx_1d];
            unsigned int check_j = d_excell_idx[check_excell_idx];
            bool j_has_been_updated = d_trial_updated[check_j];

            unsigned int excell_overlap = d_excell_overlap[check_excell_idx];

            if ( (j_has_been_updated && excell_overlap == OVERLAP_IN_NEW_CONFIG)
                ||  (!j_has_been_updated && excell_overlap == OVERLAP_IN_OLD_CONFIG))
                {
                // flag for overlap
                s_overlap[check_group] = 1;
                }
            }

        // threads that need to do more looking set the still_searching flag
        __syncthreads();
        if (master && group == 0)
            s_queue_size = 0;

        if (active && !s_overlap[group] && k < excell_size)
            atomicAdd(&s_still_searching, 1);
        __syncthreads();

        } // end while (s_still_searching)

    // update the data if accepted
    if (master)
        {
        if (active && move_active)
            {
            // first need to check if the particle remains in its cell
            Scalar3 xnew_i = s_pos_group[group];
            unsigned int new_cell = computeParticleCell(xnew_i, box, ghost_width, cell_dim, ci);
            bool accepted=true;
            if (s_overlap[group])
                accepted=false;
            if (new_cell != my_cell)
                accepted=false;

            if (accepted)
                {
                // write out the updated position and orientation
                d_postype[i] = make_scalar4(xnew_i.x, xnew_i.y, xnew_i.z, __int_as_scalar(s_type_group[group]));
                d_orientation[i] = s_orientation_group[group];
                }

            if (!accepted)
                {
                // flag rejection for subsequent cell sets
                d_trial_updated[i] = 0;
                }

            if (!ignore_stats && accepted && move_type_translate)
                atomicAdd(&s_translate_accept_count, 1);
            if (!ignore_stats && accepted && !move_type_translate)
                atomicAdd(&s_rotate_accept_count, 1);
            if (!ignore_stats && !accepted && move_type_translate)
                atomicAdd(&s_translate_reject_count, 1);
            if (!ignore_stats && !accepted && !move_type_translate)
                atomicAdd(&s_rotate_reject_count, 1);
            }
        }

    __syncthreads();

    // final tally into global mem
    if (master && group == 0)
        {
        atomicAdd(&d_counters->translate_accept_count, s_translate_accept_count);
        atomicAdd(&d_counters->translate_reject_count, s_translate_reject_count);
        atomicAdd(&d_counters->rotate_accept_count, s_rotate_accept_count);
        atomicAdd(&d_counters->rotate_reject_count, s_rotate_reject_count);
        }
    }



//! Kernel driver for gpu_hpmc_schedule_overlaps_kernel()
template< class Shape >
cudaError_t gpu_hpmc_check_overlaps(const hpmc_args_t& args, const typename Shape::param_type *params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_counters);
    assert(args.d_cell_idx);
    assert(args.d_cell_size);
    assert(args.d_excell_idx);
    assert(args.d_excell_size);
    assert(args.d_cell_set);
    assert(args.d_d);
    assert(args.d_a);
    assert(args.d_check_overlaps);
    assert(args.group_size >= 1);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static cudaFuncAttributes attr;
    if (max_block_size == -1)
        {
        cudaFuncGetAttributes(&attr, gpu_hpmc_schedule_overlaps_kernel<Shape>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    // might need to modify group_size to make the kernel runnable
    unsigned int group_size = args.group_size;

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

    // determine the maximum block size for the overlaps kernel and clamp the input block size down further
    static int max_block_size_overlaps = -1;
    static cudaFuncAttributes attr_overlaps;
    if (max_block_size_overlaps == -1)
        {
        cudaFuncGetAttributes(&attr_overlaps, gpu_hpmc_check_overlaps_kernel<Shape>);
        max_block_size_overlaps = attr_overlaps.maxThreadsPerBlock;
        }

    // the new block size might not fit the group size, decrease group size until it is
    while (block_size % group_size)
        {
        group_size--;
        }

    unsigned int n_groups = block_size / group_size;
    unsigned int max_queue_size = n_groups*group_size;
    unsigned int shared_bytes = n_groups * (sizeof(unsigned int)*3 + sizeof(Scalar4) + sizeof(Scalar3)) +
                                max_queue_size * 3 * sizeof(unsigned int) +
                                args.num_types * sizeof(typename Shape::param_type);

    unsigned int min_shared_bytes = args.num_types * sizeof(typename Shape::param_type);

    if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
        throw std::runtime_error("Insufficient shared memory for gpu_hpmc_schedule_overlaps_kernel kernel: reduce number of particle types or size of shape parameters");

    while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
        {
        block_size -= args.devprop.warpSize;
        if (block_size == 0)
            throw std::runtime_error("Insufficient shared memory for gpu_hpmc_schedule_overlaps_kernel kernel");

        // the new block size might not fit the group size, decrease group size until it is
        group_size = args.group_size;

        while (block_size % group_size)
            {
            group_size--;
            }

        n_groups = block_size / group_size;
        max_queue_size = n_groups*group_size;
        shared_bytes = n_groups * (sizeof(unsigned int)*3 + sizeof(Scalar4) + sizeof(Scalar3)) +
                       max_queue_size * 3 * sizeof(unsigned int) +
                       min_shared_bytes;
        }

    unsigned int shared_bytes_overlaps = args.num_types*sizeof(typename Shape::param_type)
        + args.overlap_idx.getNumElements()*sizeof(unsigned int);

    static unsigned int base_shared_bytes = UINT_MAX;
    bool shared_bytes_changed = base_shared_bytes != shared_bytes_overlaps + attr_overlaps.sharedSizeBytes;
    base_shared_bytes = shared_bytes_overlaps + attr_overlaps.sharedSizeBytes;

    // NVIDIA recommends not using more than 32k of shared memory per block
    // http://docs.nvidia.com/cuda/pascal-tuning-guide/index.html
    unsigned int max_extra_bytes = max(0,32768 - base_shared_bytes);
    static unsigned int extra_bytes = UINT_MAX;

    if (args.load_shared)
        {
        if (extra_bytes == UINT_MAX || args.update_shape_param || shared_bytes_changed)
            {
            // required for memory coherency
            cudaDeviceSynchronize();

            // determine dynamically requested shared memory
            char *ptr = (char *)nullptr;
            unsigned int available_bytes = max_extra_bytes;
            for (unsigned int i = 0; i < args.num_types; ++i)
                {
                params[i].load_shared(ptr, available_bytes);
                }
            extra_bytes = max_extra_bytes - available_bytes;
            }
        }

    // setup the grid to run the kernel
    dim3 threads(group_size, n_groups,1);
    dim3 grid( args.csi.getNumElements() / n_groups + 1, 1, 1);

    unsigned int block_size_overlaps = min(args.block_size_overlaps, max_block_size_overlaps);

    gpu_hpmc_schedule_overlaps_kernel<Shape><<<grid, threads, shared_bytes, args.stream>>>(args.d_postype,
                                                                 args.d_orientation,
                                                                 args.d_counters,
                                                                 args.d_cell_idx,
                                                                 args.d_cell_size,
                                                                 args.d_excell_idx,
                                                                 args.d_excell_cell_set,
                                                                 args.d_excell_overlap,
                                                                 args.d_excell_size,
                                                                 args.ci,
                                                                 args.cli,
                                                                 args.excli,
                                                                 args.cell_dim,
                                                                 args.ghost_width,
                                                                 args.d_cell_set,
                                                                 args.n_active_cells,
                                                                 args.N,
                                                                 args.num_types,
                                                                 args.seed,
                                                                 args.d_check_overlaps,
                                                                 args.overlap_idx,
                                                                 args.move_ratio,
                                                                 args.timestep,
                                                                 args.dim,
                                                                 args.box,
                                                                 args.select,
                                                                 args.ghost_fraction,
                                                                 args.domain_decomposition,
                                                                 params,
                                                                 max_queue_size,
                                                                 extra_bytes,
                                                                 max_extra_bytes,
                                                                 args.queue_idx,
                                                                 args.d_queue_active_cell_idx,
                                                                 args.d_queue_postype,
                                                                 args.d_queue_orientation,
                                                                 args.d_queue_excell_idx,
                                                                 args.d_cell_overlaps,
                                                                 args.max_gmem_queue_size,
                                                                 args.check_cuda_errors,
                                                                 block_size_overlaps,
                                                                 args.load_shared,
                                                                 args.csi,
                                                                 args.d_trial_postype,
                                                                 args.d_trial_orientation,
                                                                 args.d_trial_updated,
                                                                 args.d_trial_move_type_translate,
                                                                 args.d_update_order);

    return cudaSuccess;
    }

//! Kernel driver for gpu_hpmc_accept_kernel()
template< class Shape >
cudaError_t gpu_hpmc_accept(const hpmc_args_t& args, const typename Shape::param_type *params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_counters);
    assert(args.d_cell_idx);
    assert(args.d_cell_size);
    assert(args.d_excell_idx);
    assert(args.d_excell_size);
    assert(args.d_cell_set);
    assert(args.d_d);
    assert(args.d_a);
    assert(args.d_check_overlaps);
    assert(args.group_size >= 1);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static cudaFuncAttributes attr;
    if (max_block_size == -1)
        {
        cudaFuncGetAttributes(&attr, gpu_hpmc_accept_kernel<Shape>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    // might need to modify group_size to make the kernel runnable
    unsigned int group_size = args.group_size;

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

    // the new block size might not fit the group size, decrease group size until it is
    while (block_size % group_size)
        {
        group_size--;
        }

    unsigned int n_groups = block_size / group_size;
    unsigned int max_queue_size = n_groups*group_size;
    unsigned int shared_bytes = n_groups * (sizeof(unsigned int)*2 + sizeof(Scalar4) + sizeof(Scalar3)) +
                                max_queue_size * 2 * sizeof(unsigned int) +
                                args.num_types * (sizeof(typename Shape::param_type));

    unsigned int min_shared_bytes = args.num_types * sizeof(typename Shape::param_type);

    if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
        throw std::runtime_error("Insufficient shared memory for gpu_hpmc_accept_kernel: reduce number of particle types or size of shape parameters");

    while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
        {
        block_size -= args.devprop.warpSize;
        if (block_size == 0)
            throw std::runtime_error("Insufficient shared memory for gpu_hpmc_accept_kernel");

        // the new block size might not fit the group size, decrease group size until it is
        group_size = args.group_size;

        while (block_size % group_size)
            {
            group_size--;
            }

        n_groups = block_size / group_size;
        max_queue_size = n_groups*group_size;
        shared_bytes = n_groups * (sizeof(unsigned int)*2 + sizeof(Scalar4) + sizeof(Scalar3)) +
                       max_queue_size * 2 * sizeof(unsigned int) +
                       min_shared_bytes;
        }

    // setup the grid to run the kernel
    dim3 threads(group_size, n_groups,1);
    dim3 grid( args.n_active_cells / n_groups + 1, 1, 1);

    gpu_hpmc_accept_kernel<Shape><<<grid, threads, shared_bytes, args.stream>>>(args.d_postype,
                                                                 args.d_orientation,
                                                                 args.d_counters,
                                                                 args.d_cell_idx,
                                                                 args.d_cell_size,
                                                                 args.d_excell_idx,
                                                                 args.d_excell_cell_set,
                                                                 args.d_excell_overlap,
                                                                 args.d_excell_size,
                                                                 args.ci,
                                                                 args.cli,
                                                                 args.excli,
                                                                 args.cell_dim,
                                                                 args.ghost_width,
                                                                 args.d_cell_set,
                                                                 args.n_active_cells,
                                                                 args.N,
                                                                 args.num_types,
                                                                 args.seed,
                                                                 args.move_ratio,
                                                                 args.timestep,
                                                                 args.dim,
                                                                 args.box,
                                                                 args.select,
                                                                 args.ghost_fraction,
                                                                 args.domain_decomposition,
                                                                 params,
                                                                 args.csi,
                                                                 args.d_trial_postype,
                                                                 args.d_trial_orientation,
                                                                 args.d_trial_updated,
                                                                 args.d_trial_move_type_translate,
                                                                 args.cur_set,
                                                                 max_queue_size);

    return cudaSuccess;
    }
#endif //NVCC


}; // end namespace detail

} // end namespace hpmc

#endif // _INTEGRATOR_HPMC_CUH_

