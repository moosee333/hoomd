// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "UpdaterClustersGPU.cuh"

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

#include <cusparse.h>

#include "hoomd/extern/ECL.cuh"

/*! \file UpdaterClustersGPU.cu
    \brief Implements a spectral method for finding strongly connected components
*/

namespace hpmc
{

namespace detail
{

#define check_cusparse(a) \
    {\
    cusparseStatus_t status = (a);\
    if ((int)status != CUSPARSE_STATUS_SUCCESS)\
        {\
        printf("cusparse ERROR %d in file %s line %d\n",status,__FILE__,__LINE__);\
        throw std::runtime_error("Error during clusters update");\
        }\
    }

#define check_cuda(a) \
    {\
    cudaError_t status = (a);\
    if ((int)status != cudaSuccess)\
        {\
        return status;\
        }\
    }

struct get_source : public thrust::unary_function<uint2, unsigned int>
    {
    __host__ __device__
    unsigned int operator()(const uint2& u) const
        {
        return u.x;
        }
    };

struct get_destination : public thrust::unary_function<uint2, unsigned int>
    {
    __host__ __device__
    unsigned int operator()(const uint2& u) const
        {
        return u.y;
        }
    };

//! Generates edges such that the lower triangular adjacency matrix is filled
struct sort_pair : public thrust::unary_function<uint2, uint2>
    {
    __device__
    uint2 operator()(const uint2& p) const
        {
        unsigned int i = p.x;
        unsigned int j = p.y;
        return (j < i) ? make_uint2(i,j) : make_uint2(j,i);
        }
    };

struct pair_less : public thrust::binary_function<uint2, uint2, bool>
    {
    __device__ bool operator()(const uint2 &lhs, const uint2 &rhs) const
        {
        return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
        }
    };

struct pair_equal : public thrust::binary_function<uint2, uint2, bool>
    {
    __device__ bool operator()(const uint2 &lhs, const uint2 &rhs) const
        {
        return lhs.x == rhs.x && lhs.y == rhs.y;
        }
    };


cudaError_t gpu_connected_components(
    const uint2 *d_adj,
    unsigned int N,
    unsigned int n_elements,
    int *d_components,
    unsigned int &num_components,
    const cudaDeviceProp& dev_prop,
    const CachedAllocator& alloc)
    {
    // make a copy of the input
    uint2 *d_adj_copy;
    check_cuda(cudaMalloc((void **)&d_adj_copy,n_elements*sizeof(uint2)));
    check_cuda(cudaMemcpy(d_adj_copy, d_adj, sizeof(uint2)*n_elements,cudaMemcpyDeviceToDevice));

    // fill sparse matrix and make it symmetric
    thrust::device_ptr<uint2> adj_copy(d_adj_copy);

    // sort every pair
    thrust::transform(
        thrust::cuda::par(alloc),
        adj_copy,
        adj_copy + n_elements,
        adj_copy,
        sort_pair());

    // sort the list of pairs
    thrust::sort(
        thrust::cuda::par(alloc),
        adj_copy,
        adj_copy + n_elements,
        pair_less());

    // remove duplicates
    auto unique_end = thrust::unique(
        thrust::cuda::par(alloc),
        adj_copy,
        adj_copy + n_elements,
        pair_equal());

    unsigned int n_unique = unique_end - adj_copy;

    auto source = thrust::make_transform_iterator(adj_copy, get_source());
    auto destination = thrust::make_transform_iterator(adj_copy, get_destination());

    #if 0
    // label all endpoints of edges
    unsigned int *d_label;
    check_cuda(cudaMalloc((void **)&d_label,N*sizeof(unsigned int)));
    thrust::device_ptr<unsigned int> label(d_label);

    auto one_it = thrust::constant_iterator<unsigned int>(1);
    thrust::fill(
        thrust::cuda::par(alloc),
        label,
        label + N,
        0);

    thrust::scatter(
        thrust::cuda::par(alloc),
        one_it,
        one_it + n_unique,
        source,
        label);

    thrust::scatter(
        thrust::cuda::par(alloc),
        one_it,
        one_it + n_unique,
        destination,
        label);

    // partition the vertices into trivial ones, i.e. those not connected to any edge, and connected ones
    int *d_connected_vertices;
    int *d_trivial_components;
    check_cuda(cudaMalloc((void **)&d_connected_vertices,N*sizeof(int)));
    thrust::device_ptr<int> connected_vertices(d_connected_vertices);

    check_cuda(cudaMalloc((void **)&d_trivial_components,N*sizeof(int)));
    thrust::device_ptr<int> trivial_components(d_trivial_components);

    auto part = thrust::stable_partition_copy(
        thrust::cuda::par(alloc),
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(0)+N,
        label,
        connected_vertices,
        trivial_components,
        thrust::identity<int>()
        );
    unsigned int n_connected = part.first - connected_vertices;
    unsigned int n_trivial = part.second - trivial_components;

    // wrap the components output vector
    thrust::device_ptr<unsigned int> components(d_components);

    // already label the trivial components
    thrust::scatter(
        thrust::cuda::par(alloc),
        thrust::counting_iterator<unsigned int>(0),
        thrust::counting_iterator<unsigned int>(0)+n_trivial,
        trivial_components,
        components);

    num_components = n_trivial;

    if (!n_connected)
        {
        // return early
        return cudaSuccess;
        }
    #endif

    // input matrix in COO format
    int *d_rowidx;
    int *d_colidx;

    unsigned int nverts = N;
    unsigned int nedges = n_unique;

    check_cuda(cudaMalloc((void **)&d_rowidx,nedges*sizeof(int)));
    check_cuda(cudaMalloc((void **)&d_colidx, nedges*sizeof(int)));

    thrust::device_ptr<int> rowidx(d_rowidx);
    thrust::device_ptr<int> colidx(d_colidx);

    thrust::copy(source, source+n_unique, rowidx);
    thrust::copy(destination, destination+n_unique, colidx);

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    #if 0
    // transpose
    thrust::copy(source, source+n_unique, coo_destination+n_unique);
    thrust::copy(destination, destination+n_unique, coo_source+n_unique);
    #endif

    #if 0
    check_cuda(cudaMalloc((void **)&d_P, nedges*sizeof(int)));

    check_cusparse(cusparseCreateIdentityPermutation(
        handle,
        nedges,
        d_P);

    // get temporary buffer size
    size_t buffer_size = 0;
    check_cusparse(cusparseXcoosort_bufferSizeExt(
        handle,
        nvert,
        nvert,
        nedges,
        d_rowidx,
        d_colidx,
        &buffer_size ));

    void *d_buffer;
    check_cuda(cudaMalloc((void **)&d_buffer, sizeof(char)*buffer_size));

    // sort COO indices
    check_cusparse(cusparseXcoosortByRow(
        handle,
        nvert,
        nvert,
        nedges,
        d_rowidx,
        d_colidx,
        d_P,
        d_buffer));
    #endif

    // allocate CSR matrix topology
    int *d_csr_rowptr;
    check_cuda(cudaMalloc((void **)&d_csr_rowptr, (nverts+1)*sizeof(int)));

    check_cusparse(cusparseXcoo2csr(handle,
        d_rowidx,
        nedges,
        nverts,
        d_csr_rowptr,
        CUSPARSE_INDEX_BASE_ZERO));

    int *d_work;
    check_cuda(cudaMalloc((void **)&d_work, sizeof(int)*nverts));

    // compute the connected components
    ecl_connected_components(
        nverts,
        nedges,
        d_csr_rowptr,
        d_colidx,
        d_components,
        d_work,
        dev_prop);

    // reuse work array
    thrust::device_ptr<int> components(d_components);
    thrust::device_ptr<int> work(d_work);

    thrust::copy(
        thrust::cuda::par(alloc),
        components,
        components+nverts,
        work);
    thrust::sort(
        thrust::cuda::par(alloc),
        work,
        work+nverts);

    auto it = thrust::reduce_by_key(
        thrust::cuda::par(alloc),
        work,
        work+nverts,
        thrust::constant_iterator<int>(1),
        thrust::discard_iterator<int>(),
        thrust::discard_iterator<int>());

    num_components = it.first - thrust::discard_iterator<int>();

    // free device data
    check_cuda(cudaFree(d_rowidx));
    check_cuda(cudaFree(d_colidx));
    check_cuda(cudaFree(d_csr_rowptr));
//    check_cuda(cudaFree(d_P));
//    check_cuda(cudaFree(d_buffer));
    check_cuda(cudaFree(d_adj_copy));
    check_cuda(cudaFree(d_work));
//    check_cuda(cudaFree(d_label));
//    check_cuda(cudaFree(d_connected_vertices));
//    check_cuda(cudaFree(d_trivial_components));

    // clean cusparse
    cusparseDestroy(handle);

    return cudaSuccess;
    }

} // end namespace detail
} // end namespace hpmc
