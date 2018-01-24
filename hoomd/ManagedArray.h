// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#ifndef NVCC
#include "managed_allocator.h"

#include <algorithm>
#endif

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <memory>
#include <string>

#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

#define check_cuda(a) \
    {\
    cudaError_t status = (a);\
    if ((int)status != cudaSuccess)\
        {\
        printf("CUDA ERROR %d (%s) in file %s line %d\n",status,cudaGetErrorString(status),__FILE__,__LINE__);\
        throw std::runtime_error("Error in ManagedArray");\
        }\
    }

//! A device-side, fixed-size array memory-managed through cudaMallocManaged
template<class T>
class ManagedArray
    {
    public:
        //! Default constructor
        HOSTDEVICE ManagedArray()
            : data(nullptr),
              managed_data(nullptr),
            #ifdef ENABLE_CUDA
            d_data(nullptr),
            cache_is_current(nullptr),
            #endif
            N(0), managed(0)
            { }

        #ifndef NVCC
        ManagedArray(unsigned int _N, bool _managed)
            : data(nullptr),
            managed_data(nullptr),
            #ifdef ENABLE_CUDA
            d_data(nullptr),
            cache_is_current(nullptr),
            #endif
            N(_N), managed(_managed)
            {
            if (N > 0)
                {
                allocate();

                // point data to managed array
                data = managed_data;
                }
            }

        //! Convenience constructor to fill array with values
        ManagedArray(unsigned int _N, bool _managed, const T& value)
            : data(nullptr),
            managed_data(nullptr),
            #ifdef ENABLE_CUDA
            d_data(nullptr),
            cache_is_current(nullptr),
            #endif
            N(_N), managed(_managed)
            {
            if (N > 0)
                {
                allocate();

                // point data to managed array
                data = managed_data;

                std::fill(data,data+N,value);
                }
            }
        #endif

        //! Destructor
        HOSTDEVICE ~ManagedArray()
            {
            #ifndef NVCC
            deallocate();

            #ifdef ENABLE_CUDA
            if (d_data)
                deallocate_cache();
            #endif

            #endif
            }

        //! Copy constructor
        HOSTDEVICE ManagedArray(const ManagedArray<T>& other)
            : data(nullptr),
            managed_data(nullptr),
            #ifdef ENABLE_CUDA
            d_data(nullptr),
            cache_is_current(nullptr),
            #endif
            N(other.N),
            managed(other.managed)
            {
            #ifndef NVCC
            if (N > 0)
                {
                allocate();

                // point data to managed array
                data = managed_data;

                std::copy(other.data, other.data+N, data);

                #ifdef ENABLE_CUDA
                if (managed)
                    *cache_is_current = *(other.cache_is_current);

                if (managed && other.d_data && *(other.cache_is_current))
                    {
                    allocate_cache();
                    assert(d_data);

                    // copy prefetch state
                    cudaMemcpy(d_data, other.d_data, sizeof(T)*N, cudaMemcpyDeviceToDevice);
                    }
                #endif
                }
            #else
            data = other.data;
            managed_data = other.managed_data;
            d_data = other.d_data;
            cache_is_current = other.cache_is_current;
            #endif
            }

        //! Assignment operator
        HOSTDEVICE ManagedArray& operator=(const ManagedArray<T>& other)
            {
            #ifndef NVCC
            deallocate();

            #ifdef ENABLE_CUDA
            if (d_data)
                deallocate_cache();
            #endif
            #endif

            N = other.N;
            managed = other.managed;

            #ifndef NVCC
            if (N > 0)
                {
                allocate();

                // point data to managed array
                data = managed_data;
                std::copy(other.data, other.data+N, data);

                #ifdef ENABLE_CUDA
                if (managed)
                    *cache_is_current = *(other.cache_is_current);

                if (managed && other.d_data && *(other.cache_is_current))
                    {
                    allocate_cache();
                    assert(d_data);

                    // copy prefetch state
                    cudaMemcpy(d_data, other.d_data, sizeof(T)*N, cudaMemcpyDeviceToDevice);
                    }
                #endif
                }
            #else
            data = other.data;
            managed_data = other.managed_data;
            #endif

            return *this;
            }

        //! read-only random access operator
        HOSTDEVICE inline const T& operator[](unsigned int i) const
            {
            return data[i];
            }

        //! Get read-only pointer to array data
        HOSTDEVICE inline const T* get() const
            {
            return data;
            }

        //! Return the size of the array
        HOSTDEVICE inline unsigned int size() const
            {
            return N;
            }

        #ifdef ENABLE_CUDA
        //! Attach to cuda stream, and prefetch, using driver if possible
        void attach_to_stream(cudaStream_t stream) const
            {
            if (managed && managed_data)
                {
                if (stream != 0)
                    cudaStreamAttachMemAsync(stream, managed_data, 0, cudaMemAttachSingle);

                #if (CUDART_VERSION >= 8000)
                int device = -1;
                check_cuda(cudaGetDevice(&device));
                check_cuda(cudaMemAdvise(managed_data, sizeof(T)*N, cudaMemAdviseSetReadMostly, device));

                // prefetch for managed access just in case, on devices that support it
                int concurrent_managed_access = 0;
                check_cuda(cudaDeviceGetAttribute(&concurrent_managed_access, cudaDevAttrConcurrentManagedAccess, device));
                if (concurrent_managed_access)
                    check_cuda(cudaMemPrefetchAsync(managed_data, sizeof(T)*N, device, stream));
                #endif

                // prefetch the device-side cache
                prefetchDeviceCache(stream);

                // point data to managed memory
                data = managed_data;
                }
            }

        #endif

        #ifdef ENABLE_CUDA
        #ifndef NVCC
        //! Return a pointer to an up-to-date copy of data in the device-side cache
        const T* getCachedDeviceHandle(cudaStream_t stream) const
            {
            if (! *cache_is_current)
                prefetchDeviceCache(stream);

            return d_data;
            }
        #endif
        #endif

        //! Request write access to the array
        /* \note This method may be called from inside a kernel. However, if that
           kernel simultaneously works on a cached copy of the data that copy will
           *not* reflect the updated data until the next call to getCachedDeviceHandle()
         */
        HOSTDEVICE T *requestWriteAccess()
            {
            // invalidate the cache
            invalidateCache();

            return managed_data;
            }

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory allocation
         */
        HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const
            {
            // size in ints (round up)
            unsigned int size_int = (sizeof(T)*N)/sizeof(int);
            if ((sizeof(T)*N) % sizeof(int)) size_int++;

            // align ptr to size of data type
            unsigned long int max_align_bytes = (sizeof(int) > sizeof(T) ? sizeof(int) : sizeof(T))-1;
            char *ptr_align = (char *)(((unsigned long int)ptr + max_align_bytes) & ~max_align_bytes);

            if (size_int*sizeof(int)+max_align_bytes > available_bytes)
                return;

            #if defined (__CUDA_ARCH__)
            // only in GPU code
            unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
            unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;

            // use cached data on the device if at all possible
            if (d_data && cache_is_current)
                {
                for (unsigned int cur_offset = 0; cur_offset < size_int; cur_offset += block_size)
                    {
                    if (cur_offset + tidx < size_int)
                        {
                        ((int *)ptr_align)[cur_offset + tidx] = ((int *)d_data)[cur_offset + tidx];
                        }
                    }
                }
            else
                {
                for (unsigned int cur_offset = 0; cur_offset < size_int; cur_offset += block_size)
                    {
                    if (cur_offset + tidx < size_int)
                        {
                        ((int *)ptr_align)[cur_offset + tidx] = ((int *)managed_data)[cur_offset + tidx];
                        }
                    }
                }

            // make sure all threads have read from data
            __syncthreads();

            // redirect data ptr
            if (tidx == 0)
                {
                data = (T *) ptr_align;
                }

            __syncthreads();
            #endif

            // increment pointer
            ptr = ptr_align + size_int*sizeof(int);
            available_bytes -= size_int*sizeof(int)+max_align_bytes;
            }

    protected:
        //! Set the cache to dirty
        HOSTDEVICE void invalidateCache()
            {
            #ifdef ENABLE_CUDA
            if (managed && d_data)
                *cache_is_current = 0;
            #endif
            }

        #ifndef NVCC
        void allocate()
            {
            managed_data = managed_allocator<T>::allocate_construct(N, managed);

            #ifdef ENABLE_CUDA
            if (managed)
                {
                // allocate caching flag in managed memory
                cache_is_current = managed_allocator<unsigned int>::allocate_construct(1, managed);

                // reset the caching flag
                *cache_is_current = 0;
                }
            #endif
            }
        #endif

        #ifdef ENABLE_CUDA
        void allocate_cache() const
            {
            check_cuda(cudaMalloc((void **) &d_data, sizeof(T)*N));
            }

        void deallocate_cache()
            {
            if (d_data)
                {
                check_cuda(cudaFree(d_data));
                d_data = nullptr;
                }
            }
        #endif

        #ifndef NVCC
        void deallocate()
            {
            if (N > 0)
                {
                managed_allocator<T>::deallocate_destroy(managed_data, N, managed);
                managed_data = nullptr;

                #ifdef ENABLE_CUDA
                if (managed)
                    {
                    managed_allocator<unsigned int>::deallocate_destroy(cache_is_current, 1, managed);
                    cache_is_current = nullptr;
                    }
                #endif
                }
            }
        #endif

        #ifdef ENABLE_CUDA
        // prefetch data into device cache
        void prefetchDeviceCache(cudaStream_t stream) const
            {
            // no need to prefetch if we are not using any CUDA
            if (! managed)
                return;

            if (! d_data)
                {
                // allocate cache first
                allocate_cache();
                }

            if (! *cache_is_current)
                {
                // update the cache
                check_cuda(cudaMemcpyAsync(d_data, managed_data, sizeof(T)*N, cudaMemcpyDeviceToDevice, stream));

                // update the caching flag
                *cache_is_current = 1;
                }
            }
        #endif


    private:
        mutable T *data;       //!< Pointer to current data source
        T *managed_data;       //!< The managed (or host) data array

        #ifdef ENABLE_CUDA
        mutable T *d_data;             //!< Pointer to read-only copy on device
        mutable unsigned int *cache_is_current; //!< True if the cache is current (resides in managed data space
                                                // to allow invalidating cache from the device side)
        #endif

        unsigned int N;        //!< Number of data elements
        unsigned int managed;  //!< True if we are CUDA managed
    };
