
#include "COMPForceCompute.h"
#include "COMPForceComputeGPU.cuh"
#include "hoomd/PotentialPairGPU.cuh"

#ifndef __COMPFORCECOMPUTEGPU_H__
#define __COMPFORCECOMPUTEGPU_H__

template<typename Potential>
class COMPForceComputeGPU: public COMPForceCompute<Potential>
{
public:
    //! Construct the pair potential
    COMPForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<NeighborList> nlist,
                     const std::string& log_suffix="");
    //! Destructor
    virtual ~COMPForceComputeGPU() {}

    virtual void setVertices(unsigned int typ, const boost::python::list &verts);

    //! Set the number of threads per particle to execute on the GPU
    /*! \param threads_per_particl Number of threads per particle
      \a threads_per_particle must be a power of two and smaller than 32.
    */
    void setTuningParam(unsigned int param)
    {
        m_param = param;
    }

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
      \param period period (approximate) in time steps when returning occurs

      Derived classes should override this to set the parameters of their autotuners.
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
    {
        COMPForceCompute<Potential>::setAutotunerParams(enable, period);
        m_tuner->setPeriod(period);
        m_tuner->setEnabled(enable);
    }

#ifdef ENABLE_MPI
    /*! Precompute the pair force without rebuilding the neighbor list
     *
     * \param timestep The time step
     */
    virtual void preCompute(unsigned int timestep)
    {
        m_precompute = true;
        this->forceCompute(timestep);
        m_precompute = false;
        m_has_been_precomputed = true;
    }
#endif

protected:
    boost::scoped_ptr<Autotuner> m_tuner; //!< Autotuner for block size and threads per particle
    unsigned int m_param;                 //!< Kernel tuning parameter
    bool m_precompute;                    //!< True if we are pre-computing the force
    bool m_has_been_precomputed;          //!< True if the forces have been precomputed

    //! Actually compute the forces
    virtual void computeForces(unsigned int timestep);

private:
    void rebuildGeometry();

    GPUVector<Scalar3> m_gpuVertices;
    GPUVector<unsigned int> m_typeStarts;
    GPUVector<unsigned int> m_typeCounts;
};

#include "COMPForceComputeGPU.cc"

#endif
