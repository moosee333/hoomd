// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "MolecularForceCompute.h"

#include "hoomd/CachedAllocator.h"
#include "hoomd/Autotuner.h"

#ifdef ENABLE_CUDA
#include "MolecularForceCompute.cuh"
#endif

#include <string.h>
#include <map>

namespace py = pybind11;

/*! \file MolecularForceCompute.cc
    \brief Contains code for the MolecularForceCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
MolecularForceCompute::MolecularForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceConstraint(sysdef), m_molecule_tag(m_exec_conf), m_n_molecules_global(0),
      m_molecule_list(m_exec_conf), m_molecule_length(m_exec_conf), m_molecule_order(m_exec_conf),
      m_molecule_idx(m_exec_conf), m_dirty(true)
    {
    // connect to the ParticleData to recieve notifications when particles change order in memory
    m_pdata->getParticleSortSignal().connect<MolecularForceCompute, &MolecularForceCompute::setDirty>(this);

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        // initialize autotuner
        std::vector<unsigned int> valid_params;
        for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
            valid_params.push_back(block_size);

        m_tuner_fill.reset(new Autotuner(valid_params, 5, 100000, "fill_molecule_table", this->m_exec_conf));
        }
    #endif
    }

//! Destructor
MolecularForceCompute::~MolecularForceCompute()
    {
    m_pdata->getParticleSortSignal().disconnect<MolecularForceCompute, &MolecularForceCompute::setDirty>(this);
    }

#ifdef ENABLE_CUDA
void MolecularForceCompute::initMoleculesGPU()
    {
    if (m_prof) m_prof->push(m_exec_conf,"init molecules");

    unsigned int nptl_local = m_pdata->getN() + m_pdata->getNGhosts();

    unsigned int n_local_molecules = 0;

    // maximum molecule length
    unsigned int nmax = 0;

    // number of local particles that are part of molecules
    unsigned int n_local_ptls_in_molecules = 0;

    // resize to maximum possible number of local molecules
    m_molecule_length.resize(nptl_local);
    m_molecule_idx.resize(nptl_local);

    ScopedAllocation<unsigned int> d_idx_sorted_by_tag(m_exec_conf->getCachedAllocator(), nptl_local);
    ScopedAllocation<unsigned int> d_local_molecule_tags(m_exec_conf->getCachedAllocator(), nptl_local);

        {
        ArrayHandle<unsigned int> d_molecule_tag(m_molecule_tag, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_molecule_length(m_molecule_length, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_molecule_idx(m_molecule_idx, access_location::device, access_mode::overwrite);

        // temporary buffers
        ScopedAllocation<unsigned int> d_local_unique_molecule_tags(m_exec_conf->getCachedAllocator(), m_n_molecules_global);
        ScopedAllocation<unsigned int> d_sorted_by_tag(m_exec_conf->getCachedAllocator(), nptl_local);

        gpu_sort_by_molecule(nptl_local,
            d_tag.data,
            d_molecule_tag.data,
            d_local_molecule_tags.data,
            d_local_unique_molecule_tags.data,
            d_molecule_idx.data,
            d_sorted_by_tag.data,
            d_idx_sorted_by_tag.data,
            d_molecule_length.data,
            n_local_molecules,
            nmax,
            n_local_ptls_in_molecules,
            m_exec_conf->getCachedAllocator());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // set up indexer
    m_molecule_indexer = Index2D(n_local_molecules, nmax);

    m_exec_conf->msg->notice(7) << "MolecularForceCompute: " << n_local_molecules << " molecules, "
        << n_local_ptls_in_molecules << " particles in molceules " << std::endl;

    // resize molecule list
    m_molecule_list.resize(m_molecule_indexer.getNumElements());

    // resize molecule lookup to size of local particle data
    m_molecule_order.resize(m_pdata->getMaxN());

        {
        // write out molecule list and order
        ArrayHandle<unsigned int> d_molecule_list(m_molecule_list, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_molecule_order(m_molecule_order, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_molecule_idx(m_molecule_idx, access_location::device, access_mode::read);

        m_tuner_fill->begin();
        unsigned int block_size = m_tuner_fill->getParam();

        gpu_fill_molecule_table(nptl_local,
            n_local_ptls_in_molecules,
            m_molecule_indexer,
            d_molecule_idx.data,
            d_local_molecule_tags.data,
            d_idx_sorted_by_tag.data,
            d_molecule_list.data,
            d_molecule_order.data,
            block_size,
            m_exec_conf->getCachedAllocator());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_fill->end();
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }
#endif

void MolecularForceCompute::initMolecules()
    {
    // return early if no molecules are defined
    if (!m_n_molecules_global) return;

    m_exec_conf->msg->notice(7) << "MolecularForceCompute initializing molecule table" << std::endl;

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        initMoleculesGPU();
        return;
        }
    #endif

    if (m_prof) m_prof->push("init molecules");

    // construct local molecule table
    unsigned int nptl_local = m_pdata->getN() + m_pdata->getNGhosts();

    ArrayHandle<unsigned int> h_molecule_tag(m_molecule_tag, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    std::set<unsigned int> local_molecule_tags;

    unsigned int n_local_molecules = 0;

    std::vector<unsigned int> local_molecule_idx(nptl_local, NO_MOLECULE);

    // resize molecule lookup to size of local particle data
    m_molecule_order.resize(m_pdata->getMaxN());

    // sort local molecules lexicographically by molecule and by ptl tag
    std::map<unsigned int, std::set<unsigned int> > local_molecules_sorted;

    for (unsigned int iptl = 0; iptl < nptl_local; ++iptl)
        {
        unsigned int tag = h_tag.data[iptl];
        assert(tag < m_molecule_tag.getNumElements());

        unsigned int mol_tag = h_molecule_tag.data[tag];
        if (mol_tag == NO_MOLECULE) continue;

        auto it = local_molecules_sorted.find(mol_tag);
        if (it == local_molecules_sorted.end())
            {
            auto res = local_molecules_sorted.insert(std::make_pair(mol_tag,std::set<unsigned int>()));
            assert(res.second);
            it = res.first;
            }

        it->second.insert(tag);
        }

    n_local_molecules = local_molecules_sorted.size();

    m_exec_conf->msg->notice(7) << "MolecularForceCompute: " << n_local_molecules << " molecules" << std::endl;

    m_molecule_length.resize(n_local_molecules);

    ArrayHandle<unsigned int> h_molecule_length(m_molecule_length, access_location::host, access_mode::overwrite);

    // reset lengths
    for (unsigned int imol = 0; imol < n_local_molecules; ++imol)
        {
        h_molecule_length.data[imol] = 0;
        }

    // count molecule lengths
    unsigned int i = 0;
    for (auto it = local_molecules_sorted.begin(); it != local_molecules_sorted.end(); ++it)
        {
        h_molecule_length.data[i++] = it->second.size();
        }

    // find maximum length
    unsigned nmax = 0;
    for (unsigned int imol = 0; imol < n_local_molecules; ++imol)
        {
        if (h_molecule_length.data[imol] > nmax)
            {
            nmax = h_molecule_length.data[imol];
            }
        }

    // set up indexer
    m_molecule_indexer = Index2D(n_local_molecules, nmax);

    // resize molecule list
    m_molecule_list.resize(m_molecule_indexer.getNumElements());

    // reset lengths again
    for (unsigned int imol = 0; imol < n_local_molecules; ++imol)
        {
        h_molecule_length.data[imol] = 0;
        }

    // reset molecule order
    ArrayHandle<unsigned int> h_molecule_order(m_molecule_order, access_location::host, access_mode::overwrite);
    memset(h_molecule_order.data, 0, sizeof(unsigned int)*(m_pdata->getN() + m_pdata->getNGhosts()));

    // resize reverse-lookup
    m_molecule_idx.resize(nptl_local);

    // fill molecule list
    ArrayHandle<unsigned int> h_molecule_list(m_molecule_list, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_molecule_idx(m_molecule_idx, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // reset reverse lookup
    memset(h_molecule_idx.data, 0, sizeof(unsigned int)*nptl_local);

    unsigned int i_mol = 0;
    for (auto it_mol = local_molecules_sorted.begin(); it_mol != local_molecules_sorted.end(); ++it_mol)
        {
        for (std::set<unsigned int>::iterator it_tag = it_mol->second.begin(); it_tag != it_mol->second.end(); ++it_tag)
            {
            unsigned int n = h_molecule_length.data[i_mol]++;
            unsigned int ptl_idx = h_rtag.data[*it_tag];
            assert(ptl_idx < m_pdata->getN() + m_pdata->getNGhosts());
            h_molecule_list.data[m_molecule_indexer(i_mol, n)] = ptl_idx;
            h_molecule_idx.data[ptl_idx] = i_mol;
            h_molecule_order.data[ptl_idx] = n;
            }
        i_mol ++;
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_MolecularForceCompute(py::module& m)
    {
    py::class_< MolecularForceCompute, std::shared_ptr<MolecularForceCompute> >(m, "MolecularForceCompute", py::base<ForceConstraint>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }
