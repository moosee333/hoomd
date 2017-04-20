// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "CellList.h"
#include "Autotuner.h"

/*! \file CellListGPU.h
    \brief Declares the CellListGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __CELLLISTGPU_H__
#define __CELLLISTGPU_H__

//! Computes a cell list from the particles in the system on the GPU
/*! Calls GPU functions in CellListGPU.cuh and CellListGPU.cu
    \sa CellList
    \ingroup computes
*/
class CellListGPU : public CellList
    {
    public:
        //! Construct a cell list
        CellListGPU(std::shared_ptr<SystemDefinition> sysdef);

        virtual ~CellListGPU() { };

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            CellList::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period/10);
            m_tuner->setEnabled(enable);
            }

    protected:
        //! Compute the cell list
        virtual void computeCellList();

        std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size

        mgpu::ContextPtr m_mgpu_context;      //!< moderngpu context
    };

//! Exports CellListGPU to python
void export_CellListGPU(pybind11::module& m);

#endif
