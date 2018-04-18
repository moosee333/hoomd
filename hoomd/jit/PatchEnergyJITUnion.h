#ifndef _PATCH_ENERGY_JIT_UNION_H_
#define _PATCH_ENERGY_JIT_UNION_H_

#include "PatchEnergyJIT.h"
#include "hoomd/hpmc/GPUTree.h"
#include "hoomd/SystemDefinition.h"

//! Evaluate patch energies via runtime generated code, using a tree accelerator structure for unions of particles
class PatchEnergyJITUnion : public PatchEnergyJIT
    {
    public:
        //! Constructor
        /*! \param r_cut Max rcut for constituent particles
         */
        PatchEnergyJITUnion(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& fname, Scalar r_cut)
            : PatchEnergyJIT(exec_conf, fname, r_cut), m_sysdef(sysdef),
              m_on_hypersphere(sysdef->getParticleData()->getBoundaryConditions() == ParticleData::hyperspherical)
            {
            // Connect to number of types change signal
            m_sysdef->getParticleData()->getNumTypesChangeSignal().connect<PatchEnergyJITUnion, &PatchEnergyJITUnion::slotNumTypesChange>(this);

            unsigned int ntypes = m_sysdef->getParticleData()->getNTypes();
            m_rcut_type.resize(ntypes,0.0);
            m_extent_type.resize(ntypes,0.0);
            m_type.resize(ntypes);
            m_position.resize(ntypes);
            m_orientation.resize(ntypes);
            m_diameter.resize(ntypes);
            m_charge.resize(ntypes);
            m_tree.resize(ntypes);
            }
        //! Destructor
        virtual ~PatchEnergyJITUnion()
            {
            m_sysdef->getParticleData()->getNumTypesChangeSignal().disconnect<PatchEnergyJITUnion, &PatchEnergyJITUnion::slotNumTypesChange>(this);
            }

        //! Set the per-type constituent particles
        /*! \param type The particle type to set the constituent particles for
            \param rcut The maximum cutoff over all constituent particles for this type
            \param types The type IDs for every constitutent particle
            \param positions The positions
            \param orientations The orientations
            \param leaf_capacity Number of particles in OBB tree leaf
         */
        void setParam(unsigned int type,
            pybind11::list types,
            pybind11::list positions,
            pybind11::list orientations,
            pybind11::list diameters,
            pybind11::list charges,
            unsigned int leaf_capacity=4);

        //! Set the global cutoff distance
        void setRCut(float rcut)
            {
            m_r_cut = rcut;
            }

        //! Get the cut-off for constituent particles
        virtual Scalar getRCut()
            {
            return m_r_cut;
            }

        //! Get the maximum geometric extent, which is added to the cutoff, per type
        virtual Scalar getAdditiveCutoff(unsigned int type)
            {
            assert(type <= m_extent_type.size());
            return m_extent_type[type];
            }

        //! evaluate the energy of the patch interaction
        /*! \param r_ij Vector pointing from particle i to j
            \param type_i Integer type index of particle i
            \param d_i Diameter of particle i
            \param charge_i Charge of particle i
            \param quat_l_i Left quaternion of particle i
            \param quat_r_i Right quaternion of particle i
            \param q_i Orientation quaternion of particle i
            \param type_j Integer type index of particle j
            \param q_j Orientation quaternion of particle j
            \param d_j Diameter of particle j
            \param charge_j Charge of particle j
            \param quat_l_j Left quaternion of particle j
            \param quat_r_j Right quaternion of particle j
            \param R radius of hypersphere
            \returns Energy of the patch interaction.
        */
        virtual float energy(const vec3<float>& r_ij,
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

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange()
            {
            unsigned int ntypes = m_sysdef->getParticleData()->getNTypes();
            m_rcut_type.resize(ntypes,0.0);
            m_extent_type.resize(ntypes,0.0);
            m_type.resize(ntypes);
            m_position.resize(ntypes);
            m_orientation.resize(ntypes);
            m_diameter.resize(ntypes);
            m_charge.resize(ntypes);
            m_tree.resize(ntypes);
            }

    protected:
        std::shared_ptr<SystemDefinition> m_sysdef;               // HOOMD's system definition
        std::vector<hpmc::detail::GPUTree> m_tree;                // The tree acceleration structure per particle type
        std::vector< float > m_rcut_type;                         // The per-type rcut
        std::vector< float > m_extent_type;                       // The per-type geometric extent
        std::vector< std::vector<vec3<float> > > m_position;      // The positions of the constituent particles
        std::vector< std::vector<quat<float> > > m_orientation;   // The orientations of the constituent particles
        std::vector< std::vector<float> > m_diameter;             // The diameters of the constituent particles
        std::vector< std::vector<float> > m_charge;               // The charges of the constituent particles
        std::vector< std::vector<unsigned int> > m_type;          // The type identifiers of the constituent particles

        bool m_on_hypersphere;                                   //!< If we use hyperspherical boundary conditions

        //! Compute the energy of two overlapping leaf nodes
        float compute_leaf_leaf_energy(vec3<float> dr,
                             unsigned int type_a,
                             unsigned int type_b,
                             const quat<float>& orientation_a,
                             const quat<float>& orientation_b,
                             unsigned int cur_node_a,
                             unsigned int cur_node_b,
                             const quat<float>& quat_l_a,
                             const quat<float>& quat_r_a,
                             const quat<float>& quat_l_b,
                             const quat<float>& quat_r_b,
                             float R);
    };

//! Exports the PatchEnergyJITUnion class to python
void export_PatchEnergyJITUnion(pybind11::module &m);
#endif // _PATCH_ENERGY_JIT_UNION_H_
