#pragma once

// do not include python headers
#define HOOMD_NOPYTHON
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#include "KaleidoscopeJIT.h"

class EvalFactory
    {
    public:
        typedef float (*EvalFnPtr)(const vec3<float>& r_ij,
            unsigned int type_i,
            const quat<float>& q_i,
            float d_i,
            float charge_i,
            const quat<float>& quat_l_i,
            const quat<float>& quat_r_i,
            unsigned int type_j,
            const quat<float>& q_j,
            float d_j,
            float charge_j,
            const quat<float>& quat_l_j,
            const quat<float>& quat_r_j,
            float R);

        //! Constructor
        EvalFactory(const std::string& llvm_ir);

        //! Return the evaluator
        EvalFnPtr getEval()
            {
            return m_eval;
            }

        //! Get the error message from initialization
        const std::string& getError()
            {
            return m_error_msg;
            }

    private:
        std::unique_ptr<llvm::orc::KaleidoscopeJIT> m_jit; //!< The persistent JIT engine
        EvalFnPtr m_eval;         //!< Function pointer to evaluator

        std::string m_error_msg; //!< The error message if initialization fails
    };
