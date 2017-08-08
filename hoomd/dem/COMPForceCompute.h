
#include "hoomd/ForceCompute.h"
#include "hoomd/md/NeighborList.h"
#include "hoomd/VectorMath.h"

#include <stdexcept>
#include <string>
#include <vector>

//#include <boost/python.hpp>
//#include <boost/shared_ptr.hpp>

#include <iterator>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <memory>

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __COMPFORCECOMPUTE_H__
#define __COMPFORCECOMPUTE_H__

template<typename Potential>
class COMPForceCompute: public ForceCompute
{
public:
    //! Param type from evaluator
    typedef typename Potential::param_type param_type;

    typedef std::vector<vec3<Scalar> > Shape;
    typedef std::vector<Shape> Shapes;

    COMPForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<NeighborList> nlist,
                     const std::string &log_suffix="");

    virtual ~COMPForceCompute() {};

    //! Set the pair parameters for a single type pair
    virtual void setParams(unsigned int typ1, unsigned int typ2, const param_type &param);
    //! Set the rcut for a single type pair
    virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);
    //! Set ron for a single type pair
    virtual void setRon(unsigned int typ1, unsigned int typ2, Scalar ron);
    virtual void setVertices(unsigned int typ, const pybind11::list &verts);

    //! Returns a list of log quantities this compute calculates
    virtual std::vector<std::string> getProvidedLogQuantities();
    //! Calculates the requested log value and returns it
    virtual Scalar getLogValue(const std::string &quantity, unsigned int timestep);

    //! Shifting modes that can be applied to the energy
    enum energyShiftMode
    {
        no_shift = 0,
        shift,
        xplor
    };

    //! Set the mode to use for shifting the energy
    void setShiftMode(energyShiftMode mode)
    {
        m_shift_mode = mode;
    }

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    virtual CommFlags getRequestedCommFlags(unsigned int timestep);
#endif

    //! Returns true because we compute the torque
    virtual bool isAnisotropic()
    {
        return true;
    }

protected:
    //! Actually compute the forces
    virtual void computeForces(unsigned int timestep);

    //! Method to be called when number of types changes
    virtual void slotNumTypesChange();

    std::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
    energyShiftMode m_shift_mode;               //!< Store the mode with which to handle the energy shift at r_cut
    Index2D m_typpair_idx;                      //!< Helper class for indexing per type pair arrays
    GPUArray<Scalar> m_rcutsq;                  //!< Cuttoff radius squared per type pair
    GPUArray<Scalar> m_ronsq;                   //!< ron squared per type pair
    GPUArray<param_type> m_params;   //!< Pair parameters per type pair
    std::string m_prof_name;                    //!< Cached profiler name
    std::string m_log_name;                     //!< Cached log name

    Shapes m_typeVertices;
};

#include "COMPForceCompute.cc"

#endif
