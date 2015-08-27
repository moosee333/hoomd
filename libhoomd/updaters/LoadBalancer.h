/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
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

/*! \file LoadBalancer.h
    \brief Declares an updater that changes the MPI domain decomposition to balance the load
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif
#include "Updater.h"
#include "BalancedDomainDecomposition.h"

#include <boost/shared_ptr.hpp>

#include <vector>

#ifndef __LOADBALANCER_H__
#define __LOADBALANCER_H__

//! Updates domain decompositions to balance the load
/*!
 * Adjusts the boundaries of the processor domains to distribute the load close to evenly between them.
 * Implements a hybrid of the LAMMPS and GROMACS load balancing algorithms, where boxes are rescaled slowly and only
 * one time per step (as in GROMACS), but using the particle numbers rather than explicit timings (as in LAMMPS).
 *
 * \ingroup updaters
 */
class LoadBalancer : public Updater
    {
#ifdef ENABLE_MPI
    public:
        //! Constructor
        LoadBalancer(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<BalancedDomainDecomposition> decomposition);
        //! Destructor
        virtual ~LoadBalancer();

        //! Notification of a max number of particle change    
//         void slotMaxNumChanged()
//             {
//             GPUArray<unsigned int> flag_own(m_pdata->getMaxN());
//             m_flag_own.swap(flag_own);
//             }

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

    protected:
        boost::shared_ptr<BalancedDomainDecomposition> m_decomposition; //!< The domain decomposition to balance

        const MPI_Comm m_mpi_comm;  //!< MPI communicator for all ranks
        MPI_Group m_mpi_comm_group; //!< MPI group for all ranks
        MPI_Comm m_mpi_comm_xy;      //!< MPI communicator for reducing down z to the xy plane
        MPI_Comm m_mpi_comm_xz;      //!< MPI communicator for reducing down y to the xz plane

        std::vector<MPI_Group> m_mpi_group_xy_red_y;   //!< Array of MPI groups for reducing xy down y
        std::vector<MPI_Comm> m_mpi_comm_xy_red_y;     //!< Array of MPI communicators for reducing xy down y

        std::vector<MPI_Group> m_mpi_group_xy_red_x;   //!< Array of MPI groups for reducing xy down x
        std::vector<MPI_Comm> m_mpi_comm_xy_red_x;     //!< Array of MPI communicators for reducing xy down x
        std::vector<MPI_Group> m_mpi_group_xz_red_x;   //!< Array of MPI groups for reducing xz down x
        std::vector<MPI_Comm> m_mpi_comm_xz_red_x;     //!< Array of MPI communicators for reducing xz down x
        
        MPI_Group m_mpi_group_x;    //!< Group for gathering and scattering in x
        MPI_Comm m_mpi_comm_x;      //!< Communicator for gathering and scattering in x

        MPI_Group m_mpi_group_y;    //!< Group for gathering and scattering in y
        MPI_Comm m_mpi_comm_y;      //!< Communicator for gathering and scattering in y

        MPI_Group m_mpi_group_z;    //!< Group for gathering and scattering in z
        MPI_Comm m_mpi_comm_z;      //!< Communicator for gathering and scattering in z

        //! Computes the maximum imbalance factor
        Scalar getMaxImbalance();

        //! Reduce the particle numbers per rank down to one dimension
        bool reduce(std::vector<unsigned int>& N_i, unsigned int dim);

        //! Adjust the partitioning along a single dimension
        bool adjust(std::vector<Scalar>& cum_frac_i,
                    const std::vector<Scalar>& frac_i,
                    const std::vector<unsigned int>& N_i,
                    Scalar L_i);

        //! Compute the number of particles on each rank after an adjustment along one dimension
        void computeParticleChange(unsigned int dim);
        //! Count the number of particles that have gone off either edge of the rank along a dimension
        virtual void countParticlesOffRank(unsigned int &n_up, unsigned int &n_down, unsigned int dim);
        unsigned int m_N_own;           //!< Number of particles owned by this rank
        bool m_needs_recount;           //!< Flag if a particle change needs to be computed
//         GPUArray<unsigned int> m_flag_own;    //!< Array to hold the flags of the particles owned by this rank
        
        //! Flags to determine if the particle needs to stay or move
//         enum owner_flags
//             {
//             stay = 0,
//             send_up = 1,
//             send_down = 2
//             };
        
//         boost::signals2::connection m_max_numchange_conn;   //!< Connection to max particle number change signal
#endif // ENABLE_MPI
    };

#ifdef ENABLE_MPI
//! Export the LoadBalancer to python
void export_LoadBalancer();
#endif

#endif // __LOADBALANCER_H__
