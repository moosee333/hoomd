/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: morozov

/**
powered by:
Moscow group.
*/

#include "EAMForceCompute.h"
#include "NeighborList.h"
#include "EAMForceGPU.cuh"

#include <boost/shared_ptr.hpp>

/*! \file EAMForceComputeGPU.h
    \brief Declares the class EAMForceComputeGPU
*/

#ifndef __EAMForceComputeGPU_H__
#define __EAMForceComputeGPU_H__

//! Computes Lennard-Jones forces on each particle using the GPU
/*! Calculates the same forces as EAMForceCompute, but on the GPU by using texture memory(cudaArray) with hardware interpolation.
*/
class EAMForceComputeGPU : public EAMForceCompute
    {
    public:
        //! Constructs the compute
        EAMForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, char *filename, int type_of_file);

        //! Destructor
        virtual ~EAMForceComputeGPU();

        //! Sets the block size to run at
        void setBlockSize(int block_size);

    protected:
        EAMTexInterData eam_data;                   //!< Undocumented parameter
        EAMtex eam_tex_data;                        //!< Undocumented parameter
        Scalar * d_atomDerivativeEmbeddingFunction; //!< array F'(rho) for each particle
        int m_block_size;                           //!< The block size to run on the GPU

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the EAMForceComputeGPU class to python
void export_EAMForceComputeGPU();

#endif
