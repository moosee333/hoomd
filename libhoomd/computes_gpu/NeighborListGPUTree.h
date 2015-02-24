/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: mphoward

#include "NeighborListGPU.h"
#include "AABBTree.h"
#include "AABBTreeGPU.h"
#include "Autotuner.h"

using namespace hpmc::detail;

/*! \file NeighborListGPUTree.h
    \brief Declares the NeighborListGPUTree class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __NEIGHBORLISTGPUTREE_H__
#define __NEIGHBORLISTGPUTREE_H__

class NeighborListGPUTree : public NeighborListGPU
    {
    public:
        //! Constructs the compute
        NeighborListGPUTree(boost::shared_ptr<SystemDefinition> sysdef,
                              Scalar r_cut,
                              Scalar r_buff);

        //! Destructor
        virtual ~NeighborListGPUTree();
    
    
        //! Set the autotuner period
        void setTuningParam(unsigned int param)
            {
            m_param = param;
            }

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            NeighborListGPU::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period/10);
            m_tuner->setEnabled(enable);
            }
            
        //! Notification of a box size change
        void slotBoxChanged()
            {
            m_box_changed = true;
            }
        
    protected:
        unsigned int m_param;               //!< Kernel tuning parameter
        boost::scoped_ptr<Autotuner> m_tuner; //!< Autotuner for block size and threads per particle
        unsigned int m_last_tuned_timestep; //!< Last tuning timestep
        
        GPUArray<AABBTree>   m_aabb_trees;              //!< Array of AABB trees
        GPUArray<AABBTreeGPU> m_aabb_trees_gpu;         //!< Array of trees for the GPU
        GPUArray<Scalar4> m_aabb_node_bounds;           //!< hold the upper and lower bounds as flat array of scalar4
        GPUArray<unsigned int> m_aabb_node_head_idx;    //!< head index for the current node in leaf data
        GPUArray<unsigned int> m_aabb_leaf_particles;   //!< Holds the particle tags as a flat array from the trees
        GPUArray<Scalar4> m_leaf_xyzf;
        GPUArray<Scalar2> m_leaf_db;
        GPUArray<AABB>          m_aabbs;                //!< Array of AABBs
        GPUArray<unsigned int>  m_num_per_type;         //!< Number of particles per type
        GPUArray<unsigned int>  m_type_head;            //!< Head list to each particle type
        GPUArray<unsigned int>  m_map_p_global_tree;    //!< maps global ids to tag in tree

        GPUArray<Scalar3>       m_image_list;           //!< list of translation vectors

        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);
        void buildTree();
        void traverseTree();
        void getNumPerType();
        
        void allocateTree(unsigned int n_local);
        
        bool m_box_changed;
        boost::signals2::connection m_boxchange_connection;   //!< Connection to the ParticleData box size change signal
        
        unsigned int m_max_n_local; //!< Maximum number of particles locally
        
        virtual void scheduleDistanceCheck(unsigned int timestep);
        
        // tree building on gpu
        GPUArray<unsigned int> m_map_tree_global;   //!< map a leaf order id to a global particle
        GPUArray<unsigned int> m_morton_codes;      //!< 30 bit morton codes for particles to sort on z-order curve
        GPUArray<unsigned int> m_morton_codes_red;  //!< Reduced capacity morton code array
        GPUArray<unsigned int> m_leaf_particles;    //!< holds the ids of the leaf particles to create a sorting map
        GPUArray<unsigned int> m_leaf_offset;       //!< total offset in particle index for leaf nodes by type
        GPUArray<Scalar4> m_tree_aabbs;             //!< aabbs for merged leaf nodes and internal nodes
        GPUArray<unsigned int> m_node_locks;        //!< node locks for if node has been visited or not
        GPUArray<unsigned int> m_node_left_child;   //!< left children of the internal nodes
        GPUArray<uint2> m_tree_parent_sib;          //!< parents and siblings of all nodes
        void buildTreeGPU();
        void calcMortonCodes();
        void sortMortonCodes();
        void updateLeafAABBCount();
        void mergeLeafParticles();
        void genTreeHierarchy();
        void bubbleAABBs();
        void moveLeafParticles();
        
        void traverseTree2();
        GPUArray<unsigned int> m_tree_roots;
        GPUArray<Scalar4> m_leaf_xyzf_alt;
        GPUArray<Scalar4> m_leaf_tdb_alt;
        
        unsigned int m_n_leaf;                      //!< total number of leaves in tree
        unsigned int m_n_node;                      //!< total number of nodes (including leaves) in tree
        
    private:
        void updateImageVectors();
        void copyCPUtoGPU();
    };

//! Exports NeighborListGPUBinned to python
void export_NeighborListGPUTree();
#endif //__NEIGHBORLISTGPUTREE_H__
