// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

#include "NeighborListGPUStencil.cuh"
#include "hoomd/TextureTools.h"
#include "hoomd/extern/cub/cub/cub.cuh"

/*! \file NeighborListGPUStencil.cu
    \brief Defines GPU kernel code for O(N) neighbor list generation on the GPU with multiple bin stencils
*/

//! Texture for reading d_cell_xyzf
scalar4_tex_t cell_xyzf_1d_tex;

//! Texture for reading d_cell_tdb
scalar4_tex_t cell_tdb_1d_tex;

//! Texture for reading d_stencil
scalar4_tex_t stencil_1d_tex;

//! Warp-centric scan (Kepler and later)
template<int NT>
struct warp_scan_sm30_stencil
    {
    __device__ static int Scan(int tid, unsigned char x, unsigned char* total)
        {
        unsigned int laneid;
        //This command gets the lane ID within the current warp
        asm("mov.u32 %0, %%laneid;" : "=r"(laneid));

        int first = laneid - tid;

        #pragma unroll
        for(int offset = 1; offset < NT; offset += offset)
            {
            int y = __shfl(x,(first + tid - offset) &(WARP_SIZE -1));
            if(tid >= offset) x += y;
            }

        // all threads get the total from the last thread in the cta
        *total = __shfl(x,first + NT - 1);

        // shift by one (exclusive scan)
        int y = __shfl(x,(first + tid - 1) &(WARP_SIZE-1));
        x = tid ? y : 0;

        return x;
        }
    };

//! Kernel call for generating neighbor list on the GPU using multiple stencils (Kepler optimized version)
/*! \tparam flags Set bit 1 to enable body filtering. Set bit 2 to enable diameter filtering.
    \tparam threads_per_particle Number of threads cooperatively computing the neighbor list
    \param d_nlist Neighbor list data structure to write
    \param d_n_neigh Number of neighbors to write
    \param d_last_updated_pos Particle positions at this update are written to this array
    \param d_conditions Conditions array for writing overflow condition
    \param d_Nmax Maximum number of neighbors per type
    \param d_head_list List of indexes to access \a d_nlist
    \param d_pos Particle positions
    \param d_body Particle body indices
    \param d_diameter Particle diameters
    \param N Number of particles
    \param d_cell_size Number of particles in each cell
    \param d_cell_xyzf Cell contents (xyzf array from CellList with flag=type)
    \param d_cell_tdb Cell contents (tdb array from CellList with)
    \param ci Cell indexer for indexing cells
    \param cli Cell list indexer for indexing into d_cell_xyzf
    \param d_stencil 2D array of stencil offsets per type
    \param d_n_stencil Number of stencils per type
    \param stencil_idx Indexer into \a d_stencil
    \param box Simulation box dimensions
    \param d_r_cut Cutoff radius stored by pair type r_cut(i,j)
    \param r_buff The maximum radius for which to include particles as neighbors
    \param ntypes Number of particle types
    \param ghost_width Width of ghost cell layer

    \note optimized for Kepler
*/
template<unsigned char flags, int threads_per_particle>
__global__ void gpu_compute_nlist_stencil_kernel(unsigned int *d_nlist,
                                                 unsigned int *d_n_neigh,
                                                 Scalar4 *d_last_updated_pos,
                                                 unsigned int *d_conditions,
                                                 const unsigned int *d_Nmax,
                                                 const unsigned int *d_head_list,
                                                 const unsigned int *d_pid_map,
                                                 const Scalar4 *d_pos,
                                                 const unsigned int *d_body,
                                                 const Scalar *d_diameter,
                                                 const unsigned int N,
                                                 const unsigned int *d_cell_size,
                                                 const Scalar4 *d_cell_xyzf,
                                                 const Scalar4 *d_cell_tdb,
                                                 const Index3D ci,
                                                 const Index2D cli,
                                                 const Scalar4 *d_stencil,
                                                 const unsigned int *d_n_stencil,
                                                 const Index2D stencil_idx,
                                                 const BoxDim box,
                                                 const Scalar *d_r_cut,
                                                 const Scalar r_buff,
                                                 const unsigned int ntypes,
                                                 const Scalar3 ghost_width)
    {
    bool filter_body = flags & 1;
    bool diameter_shift = flags & 2;

    // cache the r_listsq parameters into shared memory
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared data for per type pair parameters
    extern __shared__ unsigned char s_data[];

    // pointer for the r_listsq data
    Scalar *s_r_list = (Scalar *)(&s_data[0]);
    unsigned int *s_Nmax = (unsigned int *)(&s_data[sizeof(Scalar)*num_typ_parameters]);

    // load in the per type pair r_list
    for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < num_typ_parameters)
            {
            Scalar r_cut = d_r_cut[cur_offset + threadIdx.x];
            // force the r_list(i,j) to a skippable value if r_cut(i,j) is skippable
            s_r_list[cur_offset + threadIdx.x] = (r_cut > Scalar(0.0)) ? r_cut+r_buff : Scalar(-1.0);
            }
        if (cur_offset + threadIdx.x < ntypes)
            {
            s_Nmax[cur_offset + threadIdx.x] = d_Nmax[cur_offset + threadIdx.x];
            }
        }
    __syncthreads();

    // each set of threads_per_particle threads is going to compute the neighbor list for a single particle
    int idx;
    if (gridDim.y > 1)
        {
        // fermi workaround
        idx = (blockIdx.x + blockIdx.y*65535) * (blockDim.x/threads_per_particle) + threadIdx.x/threads_per_particle;
        }
    else
        {
        idx = blockIdx.x * (blockDim.x/threads_per_particle) + threadIdx.x/threads_per_particle;
        }

    // one thread per particle
    if (idx >= N) return;

    // get the write particle id
    int my_pidx = d_pid_map[idx];

    Scalar4 my_postype = d_pos[my_pidx];
    Scalar3 my_pos = make_scalar3(my_postype.x, my_postype.y, my_postype.z);

    unsigned int my_type = __scalar_as_int(my_postype.w);
    unsigned int my_body = d_body[my_pidx];
    Scalar my_diam = d_diameter[my_pidx];
    unsigned int my_head = d_head_list[my_pidx];

    Scalar3 f = box.makeFraction(my_pos, ghost_width);

    // find the bin each particle belongs in
    int ib = (int)(f.x * ci.getW());
    int jb = (int)(f.y * ci.getH());
    int kb = (int)(f.z * ci.getD());

    uchar3 periodic = box.getPeriodic();

    // need to handle the case where the particle is exactly at the box hi
    if (ib == ci.getW() && periodic.x)
        ib = 0;
    if (jb == ci.getH() && periodic.y)
        jb = 0;
    if (kb == ci.getD() && periodic.z)
        kb = 0;

    int my_cell = ci(ib,jb,kb);

    // number of available stencils
    unsigned int n_stencil = d_n_stencil[my_type];

    // index of current stencil (-1 to initialize)
    int cur_adj = -1;
    Scalar cell_dist2 = 0.0;

    // current cell (0 to initialize)
    unsigned int neigh_cell = 0;

    // size of current cell (0 to initialize)
    unsigned int neigh_size = 0;

    // current index in cell
    int cur_offset = threadIdx.x % threads_per_particle;

    bool done = false;

    // total number of neighbors
    unsigned int nneigh = 0;

    while (! done)
        {
        // initalize with default
        unsigned int neighbor;
        unsigned char has_neighbor = 0;

        // advance neighbor cell
        while (cur_offset >= neigh_size && !done )
            {
            cur_offset -= neigh_size;
            cur_adj++;

            if (cur_adj < n_stencil)
                {
                // compute the stenciled cell cartesian coordinates
                Scalar4 stencil = texFetchScalar4(d_stencil, stencil_1d_tex, stencil_idx(cur_adj, my_type));
                int sib = ib + __scalar_as_int(stencil.x);
                int sjb = jb + __scalar_as_int(stencil.y);
                int skb = kb + __scalar_as_int(stencil.z);
                cell_dist2 = stencil.w;

                // wrap through the boundary
                if (sib >= (int)ci.getW() && periodic.x) sib -= ci.getW();
                if (sib < 0 && periodic.x) sib += ci.getW();
                if (sjb >= (int)ci.getH() && periodic.y) sjb -= ci.getH();
                if (sjb < 0 && periodic.y) sjb += ci.getH();
                if (skb >= (int)ci.getD() && periodic.z) skb -= ci.getD();
                if (skb < 0 && periodic.z) skb += ci.getD();

                neigh_cell = ci(sib, sjb, skb);
                neigh_size = d_cell_size[neigh_cell];
                }
            else
                // we are past the end of the cell neighbors
                done = true;
            }

        // if the first thread in the cta has no work, terminate the loop
        if (done && !(threadIdx.x % threads_per_particle)) break;

        if (!done)
            {
            // use a do {} while(0) loop to process this particle so we can break for exclusions
            // in microbenchmarks, this is was faster than using bool exclude because it saved flops
            // it's a little easier to read than having 4 levels of if{} statements nested
            do
                {
                // read in the particle type (diameter and body as well while we've got the Scalar4 in)
                const Scalar4 neigh_tdb = texFetchScalar4(d_cell_tdb, cell_tdb_1d_tex, cli(cur_offset, neigh_cell));
                const unsigned int type_j = __scalar_as_int(neigh_tdb.x);
                const Scalar diam_j = neigh_tdb.y;
                const unsigned int body_j = __scalar_as_int(neigh_tdb.z);

                // skip any particles belonging to the same rigid body if requested
                if (filter_body && my_body != 0xffffffff && my_body == body_j) break;

                // compute the rlist based on the particle type we're interacting with
                Scalar r_list = s_r_list[typpair_idx(my_type,type_j)];
                if (r_list <= Scalar(0.0)) break;
                Scalar sqshift = Scalar(0.0);
                if (diameter_shift)
                    {
                    const Scalar delta = (my_diam + diam_j) * Scalar(0.5) - Scalar(1.0);
                    // r^2 < (r_list + delta)^2
                    // r^2 < r_listsq + delta^2 + 2*r_list*delta
                    sqshift = (delta + Scalar(2.0) * r_list) * delta;
                    }
                Scalar r_listsq = r_list*r_list + sqshift;

                // compare the check distance to the minimum cell distance, and pass without distance check if unnecessary
                if (cell_dist2 > r_listsq) break;

                // only load in the particle position and id if distance check is required
                const Scalar4 neigh_xyzf = texFetchScalar4(d_cell_xyzf, cell_xyzf_1d_tex, cli(cur_offset, neigh_cell));
                const Scalar3 neigh_pos = make_scalar3(neigh_xyzf.x, neigh_xyzf.y, neigh_xyzf.z);
                unsigned int cur_neigh = __scalar_as_int(neigh_xyzf.w);

                // a particle cannot neighbor itself
                if (my_pidx == (int)cur_neigh) break;

                Scalar3 dx = my_pos - neigh_pos;
                dx = box.minImage(dx);

                Scalar dr_sq = dot(dx,dx);

                if (dr_sq <= r_listsq)
                    {
                    neighbor = cur_neigh;
                    has_neighbor = 1;
                    }
                } while (0); // particle is processed exactly once

            // advance cur_offset
            cur_offset += threads_per_particle;
            }

        // no syncthreads here, we assume threads_per_particle < warp size

        // scan over flags
        int k = 0;
        #if (__CUDA_ARCH__ >= 300)
        unsigned char n = 1;
        k = warp_scan_sm30_stencil<threads_per_particle>::Scan(threadIdx.x % threads_per_particle, has_neighbor, &n);
        #endif

        if (has_neighbor && (nneigh + k) < s_Nmax[my_type])
            d_nlist[my_head + nneigh + k] = neighbor;

        // increment total neighbor count
        #if (__CUDA_ARCH__ >= 300)
        nneigh += n;
        #else
        if (has_neighbor)
            nneigh++;
        #endif
        } // end while

    if (threadIdx.x % threads_per_particle == 0)
        {
        // flag if we need to grow the neighbor list
        if (nneigh >= s_Nmax[my_type])
            atomicMax(&d_conditions[my_type], nneigh);

        d_n_neigh[my_pidx] = nneigh;
        d_last_updated_pos[my_pidx] = my_postype;
        }
    }

//! determine maximum possible block size
template<typename T>
int get_max_block_size_stencil(T func)
    {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)func);
    int max_threads = attr.maxThreadsPerBlock;
    // number of threads has to be multiple of warp size
    max_threads -= max_threads % max_threads_per_particle;
    return max_threads;
    }

//! Bind the textures on sm <= 30
/*!
 * \param d_cell_xyzf Cell list particle array
 * \param d_cell_tdb Cell list type-diameter-body array
 * \param n_elements Number of elements in the cell list arrays
 * \param d_stencil Stencil offset array
 * \param n_stencil_elements Number of elements in the stencil offset array
 */
void gpu_nlist_stencil_bind_texture(const Scalar4 *d_cell_xyzf,
                                    const Scalar4 *d_cell_tdb,
                                    unsigned int n_elements,
                                    const Scalar4 *d_stencil,
                                    unsigned int n_stencil_elements)
    {
    // bind the position texture
    cell_xyzf_1d_tex.normalized = false;
    cell_xyzf_1d_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, cell_xyzf_1d_tex, d_cell_xyzf, sizeof(Scalar4)*n_elements);

    // bind the position texture
    cell_tdb_1d_tex.normalized = false;
    cell_tdb_1d_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, cell_tdb_1d_tex, d_cell_tdb, sizeof(Scalar4)*n_elements);

    // bind the stencil texture
    stencil_1d_tex.normalized = false;
    stencil_1d_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, stencil_1d_tex, d_stencil, sizeof(Scalar4)*n_stencil_elements);
    }

//! recursive template to launch neighborlist with given template parameters
/* \tparam cur_tpp Number of threads per particle (assumed to be power of two) */
template<int cur_tpp>
inline void stencil_launcher(unsigned int *d_nlist,
                             unsigned int *d_n_neigh,
                             Scalar4 *d_last_updated_pos,
                             unsigned int *d_conditions,
                             const unsigned int *d_Nmax,
                             const unsigned int *d_head_list,
                             const unsigned int *d_pid_map,
                             const Scalar4 *d_pos,
                             const unsigned int *d_body,
                             const Scalar *d_diameter,
                             const unsigned int N,
                             const unsigned int *d_cell_size,
                             const Scalar4 *d_cell_xyzf,
                             const Scalar4 *d_cell_tdb,
                             const Index3D& ci,
                             const Index2D& cli,
                             const Scalar4 *d_stencil,
                             const unsigned int *d_n_stencil,
                             const Index2D& stencil_idx,
                             const BoxDim& box,
                             const Scalar *d_r_cut,
                             const Scalar r_buff,
                             const unsigned int ntypes,
                             const Scalar3& ghost_width,
                             bool filter_body,
                             bool diameter_shift,
                             const unsigned int threads_per_particle,
                             const unsigned int block_size,
                             const unsigned int compute_capability)
    {
    // shared memory = r_listsq + Nmax + stuff needed for neighborlist (computed below)
    Index2D typpair_idx(ntypes);
    unsigned int shared_size = sizeof(Scalar)*typpair_idx.getNumElements() + sizeof(unsigned int)*ntypes;

    if (threads_per_particle == cur_tpp && cur_tpp != 0)
        {
        if (!diameter_shift && !filter_body)
            {
            static unsigned int max_block_size = UINT_MAX;
            if (max_block_size == UINT_MAX)
                max_block_size = get_max_block_size_stencil(gpu_compute_nlist_stencil_kernel<0,cur_tpp>);
            if (compute_capability < 35) gpu_nlist_stencil_bind_texture(d_cell_xyzf,
                                                                        d_cell_tdb,
                                                                        cli.getNumElements(),
                                                                        d_stencil,
                                                                        stencil_idx.getNumElements());

            unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;
            dim3 grid(N / (block_size/threads_per_particle) + 1);
            if (compute_capability < 30 && grid.x > 65535)
                {
                grid.y = grid.x/65535 + 1;
                grid.x = 65535;
                }

            gpu_compute_nlist_stencil_kernel<0,cur_tpp><<<grid,run_block_size,shared_size>>>(d_nlist,
                                                                                             d_n_neigh,
                                                                                             d_last_updated_pos,
                                                                                             d_conditions,
                                                                                             d_Nmax,
                                                                                             d_head_list,
                                                                                             d_pid_map,
                                                                                             d_pos,
                                                                                             d_body,
                                                                                             d_diameter,
                                                                                             N,
                                                                                             d_cell_size,
                                                                                             d_cell_xyzf,
                                                                                             d_cell_tdb,
                                                                                             ci,
                                                                                             cli,
                                                                                             d_stencil,
                                                                                             d_n_stencil,
                                                                                             stencil_idx,
                                                                                             box,
                                                                                             d_r_cut,
                                                                                             r_buff,
                                                                                             ntypes,
                                                                                             ghost_width);
            }
        else if (!diameter_shift && filter_body)
            {
            static unsigned int max_block_size = UINT_MAX;
            if (max_block_size == UINT_MAX)
                max_block_size = get_max_block_size_stencil(gpu_compute_nlist_stencil_kernel<1,cur_tpp>);
            if (compute_capability < 35) gpu_nlist_stencil_bind_texture(d_cell_xyzf,
                                                                        d_cell_tdb,
                                                                        cli.getNumElements(),
                                                                        d_stencil,
                                                                        stencil_idx.getNumElements());

            unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;
            dim3 grid(N / (block_size/threads_per_particle) + 1);
            if (compute_capability < 30 && grid.x > 65535)
                {
                grid.y = grid.x/65535 + 1;
                grid.x = 65535;
                }

            gpu_compute_nlist_stencil_kernel<1,cur_tpp><<<grid,run_block_size,shared_size>>>(d_nlist,
                                                                                             d_n_neigh,
                                                                                             d_last_updated_pos,
                                                                                             d_conditions,
                                                                                             d_Nmax,
                                                                                             d_head_list,
                                                                                             d_pid_map,
                                                                                             d_pos,
                                                                                             d_body,
                                                                                             d_diameter,
                                                                                             N,
                                                                                             d_cell_size,
                                                                                             d_cell_xyzf,
                                                                                             d_cell_tdb,
                                                                                             ci,
                                                                                             cli,
                                                                                             d_stencil,
                                                                                             d_n_stencil,
                                                                                             stencil_idx,
                                                                                             box,
                                                                                             d_r_cut,
                                                                                             r_buff,
                                                                                             ntypes,
                                                                                             ghost_width);
            }
        else if (diameter_shift && !filter_body)
            {
            static unsigned int max_block_size = UINT_MAX;
            if (max_block_size == UINT_MAX)
                max_block_size = get_max_block_size_stencil(gpu_compute_nlist_stencil_kernel<2,cur_tpp>);
            if (compute_capability < 35) gpu_nlist_stencil_bind_texture(d_cell_xyzf,
                                                                        d_cell_tdb,
                                                                        cli.getNumElements(),
                                                                        d_stencil,
                                                                        stencil_idx.getNumElements());

            unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;
            dim3 grid(N / (block_size/threads_per_particle) + 1);
            if (compute_capability < 30 && grid.x > 65535)
                {
                grid.y = grid.x/65535 + 1;
                grid.x = 65535;
                }

            gpu_compute_nlist_stencil_kernel<2,cur_tpp><<<grid,run_block_size,shared_size>>>(d_nlist,
                                                                                             d_n_neigh,
                                                                                             d_last_updated_pos,
                                                                                             d_conditions,
                                                                                             d_Nmax,
                                                                                             d_head_list,
                                                                                             d_pid_map,
                                                                                             d_pos,
                                                                                             d_body,
                                                                                             d_diameter,
                                                                                             N,
                                                                                             d_cell_size,
                                                                                             d_cell_xyzf,
                                                                                             d_cell_tdb,
                                                                                             ci,
                                                                                             cli,
                                                                                             d_stencil,
                                                                                             d_n_stencil,
                                                                                             stencil_idx,
                                                                                             box,
                                                                                             d_r_cut,
                                                                                             r_buff,
                                                                                             ntypes,
                                                                                             ghost_width);
            }
        else if (diameter_shift && filter_body)
            {
            static unsigned int max_block_size = UINT_MAX;
            if (max_block_size == UINT_MAX)
                max_block_size = get_max_block_size_stencil(gpu_compute_nlist_stencil_kernel<3,cur_tpp>);
            if (compute_capability < 35) gpu_nlist_stencil_bind_texture(d_cell_xyzf,
                                                                        d_cell_tdb,
                                                                        cli.getNumElements(),
                                                                        d_stencil,
                                                                        stencil_idx.getNumElements());

            unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;
            dim3 grid(N / (block_size/threads_per_particle) + 1);
            if (compute_capability < 30 && grid.x > 65535)
                {
                grid.y = grid.x/65535 + 1;
                grid.x = 65535;
                }

            gpu_compute_nlist_stencil_kernel<3,cur_tpp><<<grid,run_block_size,shared_size>>>(d_nlist,
                                                                                             d_n_neigh,
                                                                                             d_last_updated_pos,
                                                                                             d_conditions,
                                                                                             d_Nmax,
                                                                                             d_head_list,
                                                                                             d_pid_map,
                                                                                             d_pos,
                                                                                             d_body,
                                                                                             d_diameter,
                                                                                             N,
                                                                                             d_cell_size,
                                                                                             d_cell_xyzf,
                                                                                             d_cell_tdb,
                                                                                             ci,
                                                                                             cli,
                                                                                             d_stencil,
                                                                                             d_n_stencil,
                                                                                             stencil_idx,
                                                                                             box,
                                                                                             d_r_cut,
                                                                                             r_buff,
                                                                                             ntypes,
                                                                                             ghost_width);
            }
        }
    else
        {
        stencil_launcher<cur_tpp/2>(d_nlist,
                                    d_n_neigh,
                                    d_last_updated_pos,
                                    d_conditions,
                                    d_Nmax,
                                    d_head_list,
                                    d_pid_map,
                                    d_pos,
                                    d_body,
                                    d_diameter,
                                    N,
                                    d_cell_size,
                                    d_cell_xyzf,
                                    d_cell_tdb,
                                    ci,
                                    cli,
                                    d_stencil,
                                    d_n_stencil,
                                    stencil_idx,
                                    box,
                                    d_r_cut,
                                    r_buff,
                                    ntypes,
                                    ghost_width,
                                    filter_body,
                                    diameter_shift,
                                    threads_per_particle,
                                    block_size,
                                    compute_capability);
        }
    }

//! template specialization to terminate recursion
template<>
inline void stencil_launcher<min_threads_per_particle/2>(unsigned int *d_nlist,
                                                         unsigned int *d_n_neigh,
                                                         Scalar4 *d_last_updated_pos,
                                                         unsigned int *d_conditions,
                                                         const unsigned int *d_Nmax,
                                                         const unsigned int *d_head_list,
                                                         const unsigned int *d_pid_map,
                                                         const Scalar4 *d_pos,
                                                         const unsigned int *d_body,
                                                         const Scalar *d_diameter,
                                                         const unsigned int N,
                                                         const unsigned int *d_cell_size,
                                                         const Scalar4 *d_cell_xyzf,
                                                         const Scalar4 *d_cell_tdb,
                                                         const Index3D& ci,
                                                         const Index2D& cli,
                                                         const Scalar4 *d_stencil,
                                                         const unsigned int *d_n_stencil,
                                                         const Index2D& stencil_idx,
                                                         const BoxDim& box,
                                                         const Scalar *d_r_cut,
                                                         const Scalar r_buff,
                                                         const unsigned int ntypes,
                                                         const Scalar3& ghost_width,
                                                         bool filter_body,
                                                         bool diameter_shift,
                                                         const unsigned int threads_per_particle,
                                                         const unsigned int block_size,
                                                         const unsigned int compute_capability)
    { }

cudaError_t gpu_compute_nlist_stencil(unsigned int *d_nlist,
                                      unsigned int *d_n_neigh,
                                      Scalar4 *d_last_updated_pos,
                                      unsigned int *d_conditions,
                                      const unsigned int *d_Nmax,
                                      const unsigned int *d_head_list,
                                      const unsigned int *d_pid_map,
                                      const Scalar4 *d_pos,
                                      const unsigned int *d_body,
                                      const Scalar *d_diameter,
                                      const unsigned int N,
                                      const unsigned int *d_cell_size,
                                      const Scalar4 *d_cell_xyzf,
                                      const Scalar4 *d_cell_tdb,
                                      const Index3D& ci,
                                      const Index2D& cli,
                                      const Scalar4 *d_stencil,
                                      const unsigned int *d_n_stencil,
                                      const Index2D& stencil_idx,
                                      const BoxDim& box,
                                      const Scalar *d_r_cut,
                                      const Scalar r_buff,
                                      const unsigned int ntypes,
                                      const Scalar3& ghost_width,
                                      bool filter_body,
                                      bool diameter_shift,
                                      const unsigned int threads_per_particle,
                                      const unsigned int block_size,
                                      const unsigned int compute_capability)
    {
    stencil_launcher<max_threads_per_particle>(d_nlist,
                                               d_n_neigh,
                                               d_last_updated_pos,
                                               d_conditions,
                                               d_Nmax,
                                               d_head_list,
                                               d_pid_map,
                                               d_pos,
                                               d_body,
                                               d_diameter,
                                               N,
                                               d_cell_size,
                                               d_cell_xyzf,
                                               d_cell_tdb,
                                               ci,
                                               cli,
                                               d_stencil,
                                               d_n_stencil,
                                               stencil_idx,
                                               box,
                                               d_r_cut,
                                               r_buff,
                                               ntypes,
                                               ghost_width,
                                               filter_body,
                                               diameter_shift,
                                               threads_per_particle,
                                               block_size,
                                               compute_capability);
    return cudaSuccess;
    }

/*!
 * \param d_pids Unsorted particle indexes
 * \param d_types Unsorted particle types
 * \param d_pos Particle position array
 * \param N Number of particles
 *
 * \a d_pids and \a d_types are trivially initialized to their current (unsorted) values. They are later sorted in
 * gpu_compute_nlist_stencil_sort_types().
 */
__global__ void gpu_compute_nlist_stencil_fill_types_kernel(unsigned int *d_pids,
                                                            unsigned int *d_types,
                                                            const Scalar4 *d_pos,
                                                            const unsigned int N)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    Scalar4 pos_i = d_pos[idx];
    unsigned int type = __scalar_as_int(pos_i.w);
    d_types[idx] = type;
    d_pids[idx] = idx;
    }

/*!
 * \param d_pids Unsorted particle indexes
 * \param d_types Unsorted particle types
 * \param d_pos Particle position array
 * \param N Number of particles
 */
cudaError_t gpu_compute_nlist_stencil_fill_types(unsigned int *d_pids,
                                                 unsigned int *d_types,
                                                 const Scalar4 *d_pos,
                                                 const unsigned int N)
    {
    const unsigned int block_size = 128;

    gpu_compute_nlist_stencil_fill_types_kernel<<<N/block_size + 1, block_size>>>(d_pids, d_types, d_pos, N);

    return cudaSuccess;
    }

/*!
 * \param d_pids Array of unsorted particle indexes
 * \param d_pids_alt Double buffer for particle indexes
 * \param d_types Array of unsorted particle types
 * \param d_types_alt Double buffer for particle types
 * \param d_tmp_storage Temporary allocation for sorting
 * \param tmp_storage_bytes Size of temporary allocation
 * \param swap Flag to swap the sorted particle indexes into the correct buffer
 * \param N number of particles
 *
 * This wrapper calls the CUB radix sorting methods, and so it needs to be called twice. Initially, \a d_tmp_storage
 * should be NULL, and the necessary temporary storage is saved into \a tmp_storage_bytes. This space must then be
 * allocated into \a d_tmp_storage, and on the second call, the sorting is performed.
 */
void gpu_compute_nlist_stencil_sort_types(unsigned int *d_pids,
                                          unsigned int *d_pids_alt,
                                          unsigned int *d_types,
                                          unsigned int *d_types_alt,
                                          void *d_tmp_storage,
                                          size_t &tmp_storage_bytes,
                                          bool &swap,
                                          const unsigned int N)
    {
    cub::DoubleBuffer<unsigned int> d_keys(d_types, d_types_alt);
    cub::DoubleBuffer<unsigned int> d_vals(d_pids, d_pids_alt);
    cub::DeviceRadixSort::SortPairs(d_tmp_storage, tmp_storage_bytes, d_keys, d_vals, N);
    if (d_tmp_storage != NULL)
        {
        swap = (d_vals.selector == 1);
        }
    }
