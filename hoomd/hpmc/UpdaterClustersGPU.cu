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

struct pair_less : public thrust::binary_function<uint2, uint2, bool>
    {
    __device__ bool operator()(const uint2 &lhs, const uint2 &rhs) const
        {
        return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
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
    uint2 *d_adj_copy = alloc.getTemporaryBuffer<uint2>(n_elements);
    cudaMemcpy(d_adj_copy, d_adj, sizeof(uint2)*n_elements,cudaMemcpyDeviceToDevice);

    thrust::device_ptr<uint2> adj_copy(d_adj_copy);

    // sort the list of pairs
    thrust::sort(
        thrust::cuda::par(alloc),
        adj_copy,
        adj_copy + n_elements,
        pair_less());

    auto source = thrust::make_transform_iterator(adj_copy, get_source());
    auto destination = thrust::make_transform_iterator(adj_copy, get_destination());

    // input matrix in COO format
    unsigned int nverts = N;
    unsigned int nedges = n_elements;

    int *d_rowidx = alloc.getTemporaryBuffer<int>(nedges);
    int *d_colidx = alloc.getTemporaryBuffer<int>(nedges);

    thrust::device_ptr<int> rowidx(d_rowidx);
    thrust::device_ptr<int> colidx(d_colidx);

    thrust::copy(source, source+nedges, rowidx);
    thrust::copy(destination, destination+nedges, colidx);

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // allocate CSR matrix topology
    int *d_csr_rowptr = alloc.getTemporaryBuffer<int>(nverts+1);

    check_cusparse(cusparseXcoo2csr(handle,
        d_rowidx,
        nedges,
        nverts,
        d_csr_rowptr,
        CUSPARSE_INDEX_BASE_ZERO));

    int *d_work = alloc.getTemporaryBuffer<int>(nverts);

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

    // free temporary storage
    alloc.deallocate((char *)d_adj_copy);
    alloc.deallocate((char *)d_rowidx);
    alloc.deallocate((char *)d_colidx);
    alloc.deallocate((char *)d_csr_rowptr);
    alloc.deallocate((char *)d_work);

    // clean cusparse
    cusparseDestroy(handle);

    return cudaSuccess;
    }

} // end namespace detail
} // end namespace hpmc
