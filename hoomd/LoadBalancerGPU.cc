// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*! \file LoadBalancerGPU.cc
    \brief Defines the LoadBalancerGPU class
*/

#ifdef ENABLE_MPI
#ifdef ENABLE_CUDA

#include "LoadBalancerGPU.h"
#include "LoadBalancerGPU.cuh"

#include "CachedAllocator.h"

using namespace std;

namespace py = pybind11;


/*!
 * \param sysdef System definition
 * \param decomposition Domain decomposition
 */
LoadBalancerGPU::LoadBalancerGPU(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<DomainDecomposition> decomposition)
    : LoadBalancer(sysdef, decomposition)
    {
    // allocate data connected to the maximum number of particles
    m_pdata->getMaxParticleNumberChangeSignal().connect<LoadBalancerGPU, &LoadBalancerGPU::slotMaxNumChanged>(this);

    GPUArray<unsigned int> off_ranks(m_pdata->getMaxN(), m_exec_conf);
    m_off_ranks.swap(off_ranks);

    GPUFlags<unsigned int> n_off_rank(m_exec_conf);
    m_n_off_rank.swap(n_off_rank);

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "load_balance", this->m_exec_conf));
    }

LoadBalancerGPU::~LoadBalancerGPU()
    {
    // disconnect from the signal
    m_pdata->getMaxParticleNumberChangeSignal().disconnect<LoadBalancerGPU, &LoadBalancerGPU::slotMaxNumChanged>(this);
    }

void LoadBalancerGPU::countParticlesOffRank(std::map<unsigned int, unsigned int>& cnts)
    {
    // do nothing if rank doesn't own any particles
    if (m_pdata->getN() == 0) return;

    // mark the current ranks of each particle (hijack the comm flags array)
        {
        ArrayHandle<unsigned int> d_comm_flag(m_pdata->getCommFlags(), access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_cart_ranks(m_decomposition->getCartRanks(), access_location::device, access_mode::read);

        m_tuner->begin();
        gpu_load_balance_mark_rank(d_comm_flag.data,
                                   d_pos.data,
                                   d_cart_ranks.data,
                                   m_decomposition->getGridPos(),
                                   m_pdata->getBox(),
                                   m_decomposition->getDomainIndexer(),
                                   m_pdata->getN(),
                                   m_tuner->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tuner->end();

        }

    // select the particles that should be sent to other ranks
    vector<unsigned int> off_rank;
        {
        ArrayHandle<unsigned int> d_comm_flag(m_pdata->getCommFlags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_off_ranks(m_off_ranks, access_location::device, access_mode::overwrite);
        m_n_off_rank.resetFlags(0);

        // size the temporary storage
        void *d_tmp_storage = NULL;
        size_t tmp_storage_bytes = 0;
        gpu_load_balance_select_off_rank(d_off_ranks.data,
                                         m_n_off_rank.getDeviceFlags(),
                                         d_comm_flag.data,
                                         d_tmp_storage,
                                         tmp_storage_bytes,
                                         m_pdata->getN(),
                                         m_exec_conf->getRank());

        // always allocate a minimum of 4 bytes so that d_tmp_storage is never NULL
        size_t n_alloc = (tmp_storage_bytes > 0) ? tmp_storage_bytes : 4;
        ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), n_alloc);
        d_tmp_storage = (void*)d_alloc();

        // perform the selection
        gpu_load_balance_select_off_rank(d_off_ranks.data,
                                         m_n_off_rank.getDeviceFlags(),
                                         d_comm_flag.data,
                                         d_tmp_storage,
                                         tmp_storage_bytes,
                                         m_pdata->getN(),
                                         m_exec_conf->getRank());

        // copy just the subset of particles that are off rank on the device into host memory
        // this can save substantially on the memcpy if there are many particles on a rank
        const unsigned int n_off_rank = m_n_off_rank.readFlags();
        off_rank.resize(n_off_rank);
        cudaMemcpy(&off_rank[0], d_off_ranks.data, sizeof(unsigned int)*n_off_rank, cudaMemcpyDeviceToHost);
        }

    // perform the counting on the host
    for (unsigned int cur_p=0; cur_p < off_rank.size(); ++cur_p)
        {
        cnts[off_rank[cur_p]]++;
        }
    }

void export_LoadBalancerGPU(py::module& m)
    {
    py::class_<LoadBalancerGPU, std::shared_ptr<LoadBalancerGPU> >(m,"LoadBalancerGPU",py::base<LoadBalancer>())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<DomainDecomposition> >())
    ;
    }

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
