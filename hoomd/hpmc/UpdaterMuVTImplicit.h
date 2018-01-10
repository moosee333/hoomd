// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __UPDATER_MUVT_IMPLICIT_H__
#define __UPDATER_MUVT_IMPLICIT_H__

#include "UpdaterMuVT.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"
#include "Moves.h"

#include <random>

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

#ifdef TBB
#include <tbb/tbb.h>
#endif

namespace hpmc
{

/*!
 * This class implements an Updater for simulations in the grand-canonical ensemble (mu-V-T)
 * with depletant cluster moves
 *
 * for pure GC moves see Vink and Horbach JCP 2004
 * Bolhuis Frenkel JCP 1994, Biben/Hansen J. Phys. Cond. Mat. 1996
 */
template<class Shape, class Integrator>
class UpdaterMuVTImplicit : public UpdaterMuVT<Shape>
    {
    public:
        //! Constructor
        UpdaterMuVTImplicit(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<Integrator > mc_implicit,
            unsigned int seed,
            unsigned int npartition);

        //! Destructor
        virtual ~UpdaterMuVTImplicit()
            {
            if (m_aabbs != NULL)
                free(m_aabbs);
            }

    protected:
        std::poisson_distribution<unsigned int> m_poisson;   //!< Poisson distribution
        std::shared_ptr<Integrator > m_mc_implicit;   //!< The associated implicit depletants integrator

        /*! Check for overlaps in the new configuration
         * \param timestep  time step
         * \param type Type of particle to test
         * \param pos Position of fictitous particle
         * \param orientation Orientation of particle
         * \param lnboltzmann Log of Boltzmann weight of insertion attempt (return value)
         * \param if true, reduce final result
         * \returns True if boltzmann weight is non-zero
         */
        virtual bool tryInsertParticle(unsigned int timestep, unsigned int type, vec3<Scalar> pos,
            quat<Scalar> orientation, Scalar &lnboltzmann, bool communicate, unsigned int seed);

        /*! Remove particle and try to insert depletants
            \param timestep  time step
            \param tag Tag of particle being removed
            \param lnboltzmann Log of Boltzmann weight of insertion attempt (return value)
            \param communicate if true, reduce result with MPI
            \param seed Unique RNG seed per insertion
            \returns True if boltzmann weight is non-zero
         */
        virtual bool tryRemoveParticle(unsigned int timestep, unsigned int tag, Scalar &lnboltzmann,
            bool communicate, unsigned int seed,
            std::vector<unsigned int> types = std::vector<unsigned int>(),
            std::vector<vec3<Scalar> > positions = std::vector<vec3<Scalar> >(),
            std::vector<quat<Scalar> > orientations = std::vector<quat<Scalar> >());

        /*! Find overlapping particles of type type_remove in excluded volume of particle of type tag
            \param type_remove Type of particles to be removed
            \param tag tag of particle in whose excluded volume we search
            \returns List of particle tags in excluded volume
         */
        virtual std::set<unsigned int> findParticlesInExcludedVolume(unsigned int type_remove,
            unsigned int type,
            vec3<Scalar> pos,
            quat<Scalar> orientation);

        //! Generate a random configuration for a Gibbs sampler
        virtual void generateGibbsSamplerConfiguration(unsigned int timestep);

        /*! Check for overlaps of an inserted particle, with existing and with Gibbs sampler configuration
         * \param timestep  time step
         * \param type Type of particle to test
         * \param pos Position of fictitous particle
         * \param orientation Orientation of particle
         * \param lnboltzmann Log of Boltzmann weight of insertion attempt (return value)
         * \param if true, reduce final result
         * \returns True if boltzmann weight is non-zero
         */
        virtual bool tryInsertParticleGibbsSampling(unsigned int timestep, unsigned int type, vec3<Scalar> pos,
            quat<Scalar> orientation, Scalar &lnboltzmann, bool communicate, unsigned int seed);

        /*! Perform two-component perfect (Propp-Wilson) sampling in a sphere
         * \param timestep Current time step
         * \param maxit Maximum number of steps
         * \param type_insert Type of particle generated
         * \param type type of particle in whose excluded volume we sample
         * \param pos position of inserted particle
         * \param orientation orientation of inserted particle
         * \param types Types of generated particles (return value)
         * \param positions Positions of generated particles
         * \param orientations Orientations of generated particles
         * \returns True if boltzmann weight is non-zero
         */
        virtual unsigned int perfectSample(unsigned int timestep,
            unsigned int maxit,
            unsigned int type_insert,
            unsigned int type,
            vec3<Scalar> pos,
            quat<Scalar> orientation,
            std::vector<unsigned int>& types,
            std::vector<vec3<Scalar> >& positions,
            std::vector<quat<Scalar> >& orientations,
            const std::vector<unsigned int> & old_types,
            const std::vector<vec3<Scalar> >& old_pos,
            const std::vector<quat<Scalar> >& old_orientation);

        /*! Try inserting a particle in a two-species perfect sampling scheme
         * \param timestep Current time step
         * \param pos_sphere
         * \param diameter
         * \param bc
         * \param insert_types Types of particle to test
         * \param insert_pos Positions of particles to test
         * \param insert_orientation Orientations of particles to test
         * \param seed an additional RNG seed
         * \param start If true, begin sampling with species of this type (quasi-maximal element)
         * \param positions Positions of particles inserted in previous step
         * \param orientations Orientations of particles inserted in previous step
         * \returns True if boltzmann weight is non-zero
         */
        virtual std::vector<unsigned int> tryInsertPerfectSampling(unsigned int timestep,
            unsigned int type,
            vec3<Scalar> pos,
            quat<Scalar> orientation,
            const std::vector<unsigned int>& insert_types,
            const std::vector<vec3<Scalar> >& insert_pos,
            const std::vector< quat<Scalar> >& insert_orientation,
            unsigned int seed,
            bool start,
            const std::vector<unsigned int>& types = std::vector<unsigned int>(),
            const std::vector<vec3<Scalar> >& positions = std::vector<vec3<Scalar> >(),
            const std::vector<quat<Scalar> >& orientations = std::vector<quat<Scalar> >());

        /*! Try switching particle type
         * \param timestep  time step
         * \param tag Tag of particle that is considered for switching types
         * \param newtype New type of particle
         * \param lnboltzmann Log of Boltzmann weight (return value)
         * \returns True if boltzmann weight is non-zero
         *
         * \note The method has to check that getNGlobal() > 0, otherwise tag is invalid
         */
        virtual bool trySwitchType(unsigned int timestep, unsigned int tag, unsigned newtype, Scalar &lnboltzmann);

        /*! Rescale box to new dimensions and scale particles
         * \param timestep current timestep
         * \param old_box the old BoxDim
         * \param new_box the new BoxDim
         * \param extra_ndof Extra degrees of freedom added (depletants)
         * \param lnboltzmann Exponent of Boltzmann factor (-deltaE)
         * \returns true if box resize could be performed
         */
        virtual bool boxResizeAndScale(unsigned int timestep, const BoxDim old_box, const BoxDim new_box,
            unsigned int &extra_ndof, Scalar &lnboltzmann);

        /*! Try inserting depletants into space created by changing a particle type
         * \param timestep  time step
         * \param n_insert Number of depletants to insert
         * \param delta Sphere diameter
         * \param tag Particle that is replaced
         * \param new_type New type of particle (ignored, if ignore==True)
         * \param n_trial Number of insertion trials per depletant
         * \param lnboltzmann Log of Boltzmann factor for insertion (return value)
         * \returns True if Boltzmann factor is non-zero
         */
        bool moveDepletantsInUpdatedRegion(unsigned int timestep, unsigned int n_insert, Scalar delta,
            unsigned int tag, unsigned int new_type, unsigned int n_trial, Scalar &lnboltzmann);

        /*! Insert depletants into such that they overlap with a particle of given tag
         * \param timestep time step
         * \param n_insert Number of depletants to insert
         * \param delta Sphere diameter
         * \param tag Tag of the particle depletants must overlap with
         * \param n_trial Number of insertion trials per depletant
         * \param lnboltzmann Log of Boltzmann factor for insertion (return value)
         * \param need_overlap_shape If true, successful insertions need to overlap with shape at old position
         * \returns True if Boltzmann factor is non-zero
         */
        bool moveDepletantsIntoOldPosition(unsigned int timestep, unsigned int n_insert, Scalar delta, unsigned int tag,
            unsigned int n_trial, Scalar &lnboltzmann, bool need_overlap_shape);

        /*! Insert depletants such that they overlap with a fictitious particle at a specified position
         * \param timestep time step
         * \param n_insert Number of depletants to insert
         * \param delta Sphere diameter
         * \param pos Position of inserted particle
         * \param orientation Orientationof inserted particle
         * \param type Type of inserted particle
         * \param n_trial Number of insertion trials per depletant
         * \param lnboltzmann Log of Boltzmann factor for insertion (return value)
         * \returns True if Boltzmann factor is non-zero
         */
        bool moveDepletantsIntoNewPosition(unsigned int timestep, unsigned int n_insert, Scalar delta, vec3<Scalar> pos, quat<Scalar> orientation,
            unsigned int type, unsigned int n_trial, Scalar &lnboltzmann);

        /*! Count overlapping depletants due to insertion of a fictitious particle
         * \param timestep time step
         * \param n_insert Number of depletants in circumsphere
         * \param delta Sphere diameter
         * \param pos Position of new particle
         * \param orientation Orientation of new particle
         * \param type Type of new particle (ignored, if ignore==True)
         * \param n_free Depletants that were free in old configuration
         * \param communicate if true, reduce result with MPI
         * \param seed Unique RNG seed per insertion
         * \returns Number of overlapping depletants
         */
        unsigned int countDepletantOverlapsInNewPosition(unsigned int timestep, unsigned int n_insert, Scalar delta,
            vec3<Scalar>pos, quat<Scalar> orientation, unsigned int type, unsigned int &n_free, bool communicate, unsigned int seed);

        std::vector<unsigned int> checkDepletantOverlaps(unsigned int timestep, unsigned int n_insert,
            unsigned int seed,
            unsigned int type,
            vec3<Scalar> pos,
            quat<Scalar> orientation,
            const std::vector<unsigned int>& insert_type,
            const std::vector<vec3<Scalar> >& insert_position,
            const std::vector<quat<Scalar> >& insert_orientation,
            const std::vector<unsigned int>& old_types,
            const std::vector<vec3<Scalar> >& old_positions,
            const std::vector<quat<Scalar> >& old_orientations);

        /*! Count overlapping depletants due to re-insertion of an existing particle
         * \param timestep time step
         * \param n_insert Number of depletants in circumsphere
         * \param delta Sphere diameter
         * \param tag particle tag
         * \param communicate if true, reduce result with MPI
         * \param seed Unique RNG seed per insertion
         * \returns Number of overlapping depletants
         */
        unsigned int countDepletantOverlapsInOldPosition(unsigned int timestep, unsigned int n_insert, Scalar delta,
            unsigned int tag, bool communicate, unsigned int seed);

        /*! Count overlapping depletants in a sphere of diameter delta
         * \param timestep time step
         * \param n_insert Number of depletants in circumsphere
         * \param delta Sphere diameter
         * \param pos Center of sphere
         * \returns Number of overlapping depletants
         */
        unsigned int countDepletantOverlaps(unsigned int timestep, unsigned int n_insert, Scalar delta, vec3<Scalar>pos);


        //! Get the random number of depletants
        virtual unsigned int getNumDepletants(unsigned int timestep, Scalar V, unsigned int seed);

    private:
        //! Grow the m_aabbs list
        virtual void growAABBList(unsigned int N);

        //! Helper function to build the internal AABB tree
        const detail::AABBTree& buildGibbsSamplerAABBTree();

        detail::AABBTree m_aabb_tree;               //!< Bounding volume hierarchy for overlap checks
        detail::AABB* m_aabbs;                      //!< list of AABBs, one per particle
        unsigned int m_aabbs_capacity;              //!< Capacity of m_aabbs list

        #ifdef ENABLE_TBB
        tbb::concurrent_vector<vec3<Scalar> > m_gibbs_position;
        tbb::concurrent_vector<quat<Scalar> > m_gibbs_orientation;
        #else
        std::vector<Scalar3> m_gibbs_position;      //!< Internal list of coordinates for Gibbs sampler
        std::vector<quat<Scalar> > m_gibbs_orientation; //!< Internal list of orientations
        #endif

    };

//! Export the UpdaterMuVT class to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of UpdaterMuVTImplicit<Shape,Integrator> will be exported
*/
template < class Shape, class Integrator >
void export_UpdaterMuVTImplicit(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterMuVTImplicit<Shape, Integrator>, std::shared_ptr< UpdaterMuVTImplicit<Shape, Integrator> > >(m, name.c_str(),
          pybind11::base<UpdaterMuVT<Shape> >())
          .def(pybind11::init< std::shared_ptr<SystemDefinition>,
            std::shared_ptr< Integrator >, unsigned int, unsigned int>())
          ;
    }

/*! Constructor
    \param sysdef The system defintion
    \param mc_implict The HPMC integrator
    \param seed RNG seed
    \param npartition How many partitions to use in parallel for Gibbs ensemble (n=1 == grand canonical)
*/
template<class Shape, class Integrator>
UpdaterMuVTImplicit<Shape, Integrator>::UpdaterMuVTImplicit(std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<Integrator> mc_implicit,
    unsigned int seed,
    unsigned int npartition)
    : UpdaterMuVT<Shape>(sysdef, mc_implicit,seed,npartition), m_mc_implicit(mc_implicit)
    {
    m_aabbs_capacity = 0;
    m_aabbs = NULL;
    }

template <class Shape, class Integrator>
void UpdaterMuVTImplicit<Shape, Integrator>::growAABBList(unsigned int N)
    {
    if (N > m_aabbs_capacity)
        {
        m_aabbs_capacity = N;
        if (m_aabbs != NULL)
            free(m_aabbs);

        int retval = posix_memalign((void**)&m_aabbs, 32, N*sizeof(detail::AABB));
        if (retval != 0)
            {
            this->m_exec_conf->msg->error() << "Error allocating aligned memory" << std::endl;
            throw std::runtime_error("Error allocating AABB memory");
            }
        }
    }

template <class Shape, class Integrator>
const detail::AABBTree& UpdaterMuVTImplicit<Shape, Integrator>::buildGibbsSamplerAABBTree()
    {
    auto params = m_mc_implicit->getParams();

    this->m_exec_conf->msg->notice(8) << "UpdaterMuVT Building AABB tree: " << m_gibbs_position.size() << " Gibbs sampler ptls" << std::endl;
    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "AABB tree build");

    unsigned int type_d = m_mc_implicit->getDepletantType();

    // build the AABB tree
        {
        // grow the AABB list to the needed size
        unsigned int n_aabb = m_gibbs_position.size();
        if (n_aabb > 0)
            {
            growAABBList(n_aabb);
            for (unsigned int cur_particle = 0; cur_particle < n_aabb; cur_particle++)
                {
                unsigned int i = cur_particle;
                Shape shape(m_gibbs_orientation[i], params[type_d]);
                m_aabbs[i] = shape.getAABB(m_gibbs_position[i]);
                }
            m_aabb_tree.buildTree(m_aabbs, n_aabb);
            }
        }

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

    return m_aabb_tree;
    }

//! Generate a random configuration for a Gibbs sampler
template <class Shape, class Integrator>
void UpdaterMuVTImplicit<Shape, Integrator>::generateGibbsSamplerConfiguration(unsigned int timestep)
    {
    // reset existing configuration
    m_gibbs_position.clear();
    m_gibbs_orientation.clear();

    unsigned int type_d = m_mc_implicit->getDepletantType();
    Scalar fugacity = m_mc_implicit->getDepletantDensity();

    const BoxDim& box = this->m_pdata->getBox();
    Scalar V = box.getVolume(this->m_sysdef->getNDimensions() == 2);
    std::poisson_distribution<unsigned int> poisson(fugacity*V);

    std::vector<unsigned int> seed_seq(5);
    seed_seq[0] = this->m_seed;
    seed_seq[1] = timestep;
    seed_seq[2] = this->m_exec_conf->getPartition();
    seed_seq[3] = this->m_exec_conf->getRank();
    seed_seq[4] = 0x4823b4a1;
    std::seed_seq seed(seed_seq.begin(), seed_seq.end());

    std::mt19937 rng_mt(seed);

    // how many depletants to insert
    unsigned int n_insert = poisson(rng_mt);

    auto params = m_mc_implicit->getParams();
    Shape shape_insert(quat<Scalar>(), params[type_d]);

    // NOTE in MPI we will need to make sure to maintain inactive boundaries

    #ifdef ENABLE_TBB
    // avoid a race condition
    m_mc_implicit->updateImageList();
    if (this->m_pdata->getN()+this->m_pdata->getNGhosts())
        m_mc_implicit->buildAABBTree();
    #endif

    #ifdef ENABLE_TBB
    // create one RNG per thread
    tbb::enumerable_thread_specific< hoomd::detail::Saru > rng_parallel([=]
        {
        std::vector<unsigned int> seed_seq(5);
        seed_seq[0] = this->m_seed;
        seed_seq[1] = timestep;
        seed_seq[2] = this->m_exec_conf->getRank();
        std::hash<std::thread::id> hash;
        seed_seq[3] = hash(std::this_thread::get_id());
        seed_seq[4] = 0x1824d2df;

        std::seed_seq seed(seed_seq.begin(), seed_seq.end());
        std::vector<unsigned int> s(1);
        seed.generate(s.begin(),s.end());
        return s[0]; // initialize with single seed
        });
    #endif


    #ifdef ENABLE_TBB
    tbb::parallel_for((unsigned int)0,n_insert, [&](unsigned int i)
    #else
    for (unsigned int i = 0; i < n_insert; ++i)
    #endif
        {
        auto &rng = rng_parallel.local();

        // generate a position uniformly in the box
        Scalar3 f;
        f.x = rng.template s<Scalar>();
        f.y = rng.template s<Scalar>();
        f.z = rng.template s<Scalar>();
        vec3<Scalar> pos_insert(box.makeCoordinates(f));

        if (shape_insert.hasOrientation())
            {
            // set particle orientation
            shape_insert.orientation = generateRandomOrientation(rng);
            }

        // try inserting in existing configuration (not checking for depletant overlaps, hence base class method...)
        Scalar lnb(0.0);
        if (UpdaterMuVT<Shape>::tryInsertParticleGibbsSampling(timestep, type_d, pos_insert, shape_insert.orientation, lnb, false, 0))
            {
            // store particle in local list
            m_gibbs_position.push_back(pos_insert);
            m_gibbs_orientation.push_back(shape_insert.orientation);
            }
        }
    #ifdef ENABLE_TBB
        );
    #endif

    // update the AABB tree
    buildGibbsSamplerAABBTree();
    }

template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::tryInsertParticleGibbsSampling(unsigned int timestep, unsigned int type, vec3<Scalar> pos,
     quat<Scalar> orientation, Scalar &lnboltzmann, bool communicate, unsigned int seed)
    {
    // check overlaps with colloid particles first (call base class method)
    lnboltzmann = Scalar(0.0);
    Scalar lnb(0.0);
    bool nonzero = UpdaterMuVT<Shape>::tryInsertParticleGibbsSampling(timestep, type, pos, orientation, lnb, communicate, seed);
    if (nonzero)
        {
        lnboltzmann += lnb;
        }

    // Depletant type
    unsigned int type_d = m_mc_implicit->getDepletantType();

    unsigned int overlap = 0;

    bool is_local = true;
    #ifdef ENABLE_MPI
    if (communicate && this->m_pdata->getDomainDecomposition())
        {
        const BoxDim& global_box = this->m_pdata->getGlobalBox();
        is_local = this->m_exec_conf->getRank() == this->m_pdata->getDomainDecomposition()->placeParticle(global_box, vec_to_scalar3(pos));
        }
    #endif

    unsigned int nptl_gibbs = m_gibbs_position.size();

    if (nonzero && is_local && nptl_gibbs)
        {
        // update the image list
        const std::vector<vec3<Scalar> >&image_list = m_mc_implicit->updateImageList();

        // check for overlaps
        auto params = this->m_mc->getParams();

        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        // construct a shape for inserted particle
        Shape shape(orientation, params[type]);

        unsigned int err_count = 0;

        // Check particle against AABB tree of implicit particles for neighbors
        const unsigned int n_images = image_list.size();

        detail::AABB aabb_local = shape.getAABB(vec3<Scalar>(0,0,0));

        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_image = pos + image_list[cur_image];

            detail::AABB aabb = aabb_local;
            aabb.translate(pos_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (detail::overlap(m_aabb_tree.getNodeAABB(cur_node_idx), aabb))
                    {
                    if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(m_gibbs_position[j]) - pos_image;

                            Shape shape_j(quat<Scalar>(m_gibbs_orientation[j]), params[type_d]);

                            if (h_overlaps.data[overlap_idx(type_d, type)]
                                && check_circumsphere_overlap(r_ij, shape, shape_j)
                                && test_overlap(r_ij, shape, shape_j, err_count))
                                {
                                overlap = 1;
                                break;
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                    }

                if (overlap)
                    {
                    break;
                    }
                } // end loop over AABB nodes

            if (overlap)
                {
                break;
                }
            } // end loop over images
        } // end if local

    #ifdef ENABLE_MPI
    if (communicate && this->m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &overlap, 1, MPI_UNSIGNED, MPI_MAX, this->m_exec_conf->getMPICommunicator());
        }
    #endif

    return nonzero && !overlap;
    }

template<class Shape, class Integrator>
std::set<unsigned int> UpdaterMuVTImplicit<Shape,Integrator>::findParticlesInExcludedVolume(unsigned int type_remove,
            unsigned int type,
            vec3<Scalar> pos,
            quat<Scalar> orientation)
    {
    unsigned int nptl_local = this->m_pdata->getN() + this->m_pdata->getNGhosts();

    // update the image list
    const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

    // check for overlaps
    auto params = this->m_mc->getParams();

    unsigned int type_d = m_mc_implicit->getDepletantType();

    // shape of inserted particle
    Shape shape(orientation, params[type]);

    // NOTE we are assuming here the depletant orientation doesn't matter
    Shape shape_depletant(quat<Scalar>(), params[type_d]);

    if (shape_depletant.hasOrientation())
        throw std::runtime_error("Anisotropic depletants not supported with muVT");

    Scalar R_excl = 0.5*(shape.getCircumsphereDiameter() + shape_depletant.getCircumsphereDiameter());

    std::set<unsigned int> remove_tags;

    ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
    const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

    // we cannot rely on a valid AABB tree when there are 0 particles
    if (nptl_local > 0)
        {
        // Check particle against AABB tree for neighbors
        const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();
        const unsigned int n_images = image_list.size();

        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);

        detail::AABB aabb_local = detail::AABB(vec3<Scalar>(0,0,0),R_excl);

        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_image = pos + image_list[cur_image];

            detail::AABB aabb = aabb_local;
            aabb.translate(pos_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                    {
                    if (aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            Scalar4 postype_j = h_postype.data[j];

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_image;

                            unsigned int typ_j = __scalar_as_int(postype_j.w);
                            if (typ_j != type_remove) continue;

                            Shape shape_j(quat<Scalar>(h_orientation.data[j]), params[typ_j]);

                            OverlapReal rsq(dot(r_ij,r_ij));
                            bool circumsphere_overlap = rsq <= R_excl*R_excl;

                            if (circumsphere_overlap)
                                {
                                // is particle j in depletant-excluded volume? if we cannot insert a depletant
                                // at j's position without overlap
                                unsigned int err_count = 0;
                                if (h_overlaps.data[overlap_idx(type_d,type)] &&
                                    test_overlap(r_ij, shape, shape_depletant, err_count) && j < this->m_pdata->getN())
                                    remove_tags.insert(h_tag.data[j]);
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                    }

                } // end loop over AABB nodes

            } // end loop over images
        } // end if nptl_local > 0

    return remove_tags;
    }


template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::tryInsertParticle(unsigned int timestep, unsigned int type, vec3<Scalar> pos,
     quat<Scalar> orientation, Scalar &lnboltzmann, bool communicate, unsigned int seed)
    {
    // check overlaps with colloid particles first
    lnboltzmann = Scalar(0.0);
    Scalar lnb(0.0);
    bool nonzero = UpdaterMuVT<Shape>::tryInsertParticle(timestep, type, pos, orientation, lnb, communicate, seed);
    if (nonzero)
        {
        lnboltzmann += lnb;
        }

    // Depletant type
    unsigned int type_d = m_mc_implicit->getDepletantType();

    // Depletant and colloid diameter
    Scalar d_dep, d_colloid;
        {
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = this->m_mc->getParams();
        quat<Scalar> o;
        Shape tmp(o, params[type_d]);
        d_dep = tmp.getCircumsphereDiameter();
        Shape shape(o, params[type]);
        d_colloid = shape.getCircumsphereDiameter();
        }

    // test sphere diameter and volume
    Scalar delta = d_dep + d_colloid;
    Scalar V = Scalar(M_PI/6.0)*delta*delta*delta;


    unsigned int n_overlap = 0;
    #ifdef ENABLE_MPI
    // number of depletants to insert
    unsigned int n_insert = 0;

    if (this->m_gibbs)
        {
        // perform cluster move
        if (nonzero)
            {
            // generate random depletant number
            unsigned int n_dep = getNumDepletants(timestep, V, seed);

            unsigned int tmp = 0;

            // count depletants overlapping with new config (but ignore overlap in old one)
            n_overlap = countDepletantOverlapsInNewPosition(timestep, n_dep, delta, pos, orientation, type, tmp,
                communicate, 0);

            lnb = Scalar(0.0);

            // try inserting depletants in old configuration (compute configurational bias weight factor)
            if (moveDepletantsIntoNewPosition(timestep, n_overlap, delta, pos, orientation, type, m_mc_implicit->getNumTrials(), lnb))
                {
                lnboltzmann -= lnb;
                }
            else
                {
                nonzero = false;
                }

            }

        unsigned int other = this->m_gibbs_other;

        if (this->m_exec_conf->getRank() == 0)
            {
            MPI_Request req[2];
            MPI_Status status[2];
            MPI_Isend(&n_overlap, 1, MPI_UNSIGNED, other, 0, MPI_COMM_WORLD, &req[0]);
            MPI_Irecv(&n_insert, 1, MPI_UNSIGNED, other, 0, MPI_COMM_WORLD, &req[1]);
            MPI_Waitall(2, req, status);
            }
        if (this->m_comm)
            {
            bcast(n_insert, 0, this->m_exec_conf->getMPICommunicator());
            }

        // if we have to insert depletants in addition, reject
        if (n_insert)
            {
            nonzero = false;
            }
        }
    else
    #endif
        {
        // generate random depletant number
        unsigned int n_dep = getNumDepletants(timestep, V, seed);

        // count depletants overlapping with new config (but ignore overlap in old one)
        unsigned int n_free;
        n_overlap = countDepletantOverlapsInNewPosition(timestep, n_dep, delta, pos, orientation, type,
            n_free, communicate, seed);
        nonzero = !n_overlap;
        }

    return nonzero;
    }

template<class Shape, class Integrator>
std::vector<unsigned int> UpdaterMuVTImplicit<Shape,Integrator>::tryInsertPerfectSampling(unsigned int timestep,
    unsigned int type,
    vec3<Scalar> pos,
    quat<Scalar> orientation,
    const std::vector<unsigned int>& insert_type,
    const std::vector<vec3<Scalar> >& insert_pos,
    const std::vector<quat<Scalar> > & insert_orientation,
    unsigned int seed,
    bool start,
    const std::vector<unsigned int>& types,
    const std::vector<vec3<Scalar> >& positions,
    const std::vector<quat<Scalar> >& orientations)
    {
    // check overlaps with colloid particles first
    auto temp_result = UpdaterMuVT<Shape>::tryInsertPerfectSampling(timestep, type, pos, orientation,
        insert_type, insert_pos, insert_orientation, seed,
        start, types, positions, orientations);

    // test sphere diameter and volume
    auto params = this->m_mc->getParams();
    Shape tmp(quat<Scalar>(), params[m_mc_implicit->getDepletantType()]);
    Shape shape(quat<Scalar>(), params[type]);

    // need to investigate the role of BC here!
    Scalar delta = tmp.getCircumsphereDiameter() + shape.getCircumsphereDiameter();

    Scalar V = Scalar(M_PI/6.0)*delta*delta*delta;

    // generate random depletant number
    unsigned int n_dep = 0;

    // count depletants overlapping with new config (but ignore overlap in old one)
    std::vector<unsigned int> result;
    if (!start)
        {
        n_dep = getNumDepletants(timestep, V, seed);
        }

    std::vector<unsigned int> temp_type;
    std::vector<vec3<Scalar> > temp_pos;
    std::vector<quat<Scalar> > temp_orientation;
    std::vector<unsigned int> temp_idx;
    for (auto it = temp_result.begin(); it != temp_result.end(); ++it)
        {
        temp_type.push_back(insert_type[*it]);
        temp_pos.push_back(insert_pos[*it]);
        temp_orientation.push_back(insert_orientation[*it]);
        temp_idx.push_back(*it);
        }

    auto keep = checkDepletantOverlaps(timestep, n_dep, seed,
        type,
        pos,
        orientation,
        temp_type,
        temp_pos,
        temp_orientation,
        types,
        positions,
        orientations);

    // write result list of insert-able particle indices
    for (auto it = keep.begin(); it != keep.end(); ++it)
        result.push_back(temp_idx[*it]);

    return result;
    }

template<class Shape, class Integrator>
unsigned int UpdaterMuVTImplicit<Shape,Integrator>::perfectSample(unsigned int timestep,
    unsigned int maxit,
    unsigned int type_insert,
    unsigned int type,
    vec3<Scalar> pos,
    quat<Scalar> orientation,
    std::vector<unsigned int>& types,
    std::vector<vec3<Scalar> >& positions,
    std::vector<quat<Scalar> >& orientations,
    const std::vector<unsigned int> & old_types,
    const std::vector<vec3<Scalar> >& old_pos,
    const std::vector<quat<Scalar> >& old_orientation)
    {
    // Propp-Wilson (perfect) sampler

    unsigned int cur_sequence = 0;
    unsigned int l_sequence = 1; // length of current sequence

    // depletant-excluded volume circumsphere of inserted particle
    auto params = this->m_mc->getParams();

    Shape shape_depletant(quat<Scalar>(), params[m_mc_implicit->getDepletantType()]);
    Shape shape(orientation, params[type]);
    Scalar diameter = shape_depletant.getCircumsphereDiameter() + shape.getCircumsphereDiameter();

    Scalar fugacity = this->m_fugacity[type_insert]->getValue(timestep);
    Scalar V_sphere = Scalar(M_PI/6.0)*diameter*diameter*diameter;

    std::poisson_distribution<unsigned int> poisson(fugacity*V_sphere);

    // the two Gibbs samplers
    std::vector<vec3<Scalar> > cur_pos_A, cur_pos_B;
    std::vector<quat<Scalar> > cur_orientation_A, cur_orientation_B;
    std::vector<unsigned int> cur_types_A, cur_types_B;
    std::vector<unsigned int> cur_set_A, cur_set_B; // keep track of inserted particles

    std::vector<unsigned int> insert_types;
    std::vector<vec3<Scalar> > insert_pos;
    std::vector<quat<Scalar> > insert_orientation;

    Shape shape_insert(quat<Scalar>(), params[type_insert]);

    unsigned int last_seed;

    do
        {
        // seed for Poisson process
        unsigned int iseed = l_sequence;

        cur_types_A.clear(); cur_types_B.clear();
        cur_pos_A.clear(); cur_pos_B.clear();
        cur_orientation_A.clear(); cur_orientation_B.clear();

        for (unsigned int k = 0; k < l_sequence; ++k)
            {
            // generate random positions in sampling sphere
            hoomd::detail::Saru rng(timestep, this->m_seed+iseed, 0x9bc2ffe1 );

            // combine five seeds
            std::vector<unsigned int> seed_seq(5);
            seed_seq[0] = this->m_seed;
            seed_seq[1] = timestep;
            seed_seq[2] = this->m_exec_conf->getPartition();
            seed_seq[3] = 0x7ef9ab9a;
            seed_seq[4] = iseed;
            std::seed_seq seed(seed_seq.begin(), seed_seq.end());

            // RNG for poisson distribution
            std::mt19937 rng_mt(seed);

            unsigned int n_insert = poisson(rng_mt);
            this->m_exec_conf->msg->notice(7) << "UpdaterMuVTImplicit " << timestep << " sequence " << cur_sequence << " trying to insert " << n_insert
                 << " ptls of type " << this->m_pdata->getNameByType(type_insert) << std::endl;

            insert_types.clear();
            insert_pos.clear();
            insert_orientation.clear();

            for (unsigned int n = 0; n < n_insert; ++n)
                {
                // generate same positions on all ranks
                vec3<Scalar> pos_insert = generatePositionInSphere(rng, pos, 0.5*diameter);
                if (shape_insert.hasOrientation())
                    {
                    // set particle orientation
                    shape_insert.orientation = generateRandomOrientation(rng);
                    }
                insert_types.push_back(type_insert);
                insert_pos.push_back(pos_insert);
                insert_orientation.push_back(shape_insert.orientation);
                }

            // push back old positions to cur_pos_A, ...
            for (unsigned int i = 0; i < old_types.size(); ++i)
                {
                cur_types_A.push_back(old_types[i]);
                cur_pos_A.push_back(old_pos[i]);
                cur_orientation_A.push_back(old_orientation[i]);

                cur_types_B.push_back(old_types[i]);
                cur_pos_B.push_back(old_pos[i]);
                cur_orientation_B.push_back(old_orientation[i]);
                }

            bool start = k == 0;
            // try inserting in excluded volume, for the first Gibbs sampler chain
            cur_set_A = tryInsertPerfectSampling(timestep, type, pos, orientation, insert_types, insert_pos, insert_orientation, iseed,
                start, cur_types_A, cur_pos_A, cur_orientation_A);

            // retain successfully inserted particles
            cur_types_A.clear();
            cur_pos_A.clear();
            cur_orientation_A.clear();
            for (auto it = cur_set_A.begin(); it != cur_set_A.end(); it++)
                {
                cur_types_A.push_back(insert_types[*it]);
                cur_pos_A.push_back(insert_pos[*it]);
                cur_orientation_A.push_back(insert_orientation[*it]);
                }

            // try inserting in excluded volume for second chain, it is important to use the same seed
            cur_set_B =  tryInsertPerfectSampling(timestep, type, pos, orientation, insert_types, insert_pos, insert_orientation, iseed,
                false, cur_types_B, cur_pos_B, cur_orientation_B);

            // retain successfully inserted particles
            cur_types_B.clear();
            cur_pos_B.clear();
            cur_orientation_B.clear();
            for (auto it = cur_set_B.begin(); it != cur_set_B.end(); it++)
                {
                cur_types_B.push_back(insert_types[*it]);
                cur_pos_B.push_back(insert_pos[*it]);
                cur_orientation_B.push_back(insert_orientation[*it]);
                }

            last_seed = iseed--;
            }

        if (cur_set_A == cur_set_B || ++cur_sequence == maxit)
            {
            // converged, current Gibbs sampler state is final
            positions = cur_pos_B;
            orientations = cur_orientation_B;
            types = cur_types_B;
            break;
            }

        // step in powers of two
        l_sequence *= 2;
        } while (cur_sequence < maxit);

    // should always be one ..
    return last_seed;
    }



template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::trySwitchType(unsigned int timestep, unsigned int tag, unsigned int new_type,
    Scalar &lnboltzmann)
    {
    // check overlaps with colloid particles first
    lnboltzmann = Scalar(0.0);
    Scalar lnb(0.0);
    bool nonzero = UpdaterMuVT<Shape>::trySwitchType(timestep, tag, new_type, lnb);

    if (nonzero)
        {
        lnboltzmann += lnb;
        }

    #ifdef ENABLE_MPI
    quat<Scalar> orientation(this->m_pdata->getOrientation(tag));

    // getPosition() takes into account grid shift, correct for that
    Scalar3 p = this->m_pdata->getPosition(tag)+this->m_pdata->getOrigin();
    int3 tmp = make_int3(0,0,0);
    this->m_pdata->getGlobalBox().wrap(p,tmp);
    vec3<Scalar> pos(p);

    // Depletant type
    unsigned int type_d = m_mc_implicit->getDepletantType();

    // old type
    unsigned int type = this->m_pdata->getType(tag);

    // Depletant and colloid diameter
    Scalar d_dep, d_colloid, d_colloid_old;
        {
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> >&  params = this->m_mc->getParams();
        quat<Scalar> o;
        Shape tmp(o, params[type_d]);
        d_dep = tmp.getCircumsphereDiameter();

        Shape shape(o, params[new_type]);
        d_colloid = shape.getCircumsphereDiameter();

        Shape shape_old(o, params[type]);
        d_colloid_old = shape_old.getCircumsphereDiameter();
        }

    // test sphere diameter and volume
    Scalar delta = d_dep + d_colloid;
    Scalar delta_old = d_dep + d_colloid_old;
    Scalar V = Scalar(M_PI/6.0)*delta*delta*delta;

    // generate random depletant number
    unsigned int n_dep = getNumDepletants(timestep, V, 0);

    // count depletants overlapping with new config (but ignore overlaps with old one)
    unsigned int tmp_free = 0;
    unsigned int n_overlap = countDepletantOverlapsInNewPosition(timestep, n_dep, delta, pos, orientation,
        new_type, tmp_free, true, 0);

    // reject if depletant overlap
    if (! this->m_gibbs && n_overlap)
        {
        // FIXME: need to apply GC acceptance criterium here for muVT
        nonzero = false;
        }

    // number of depletants to insert
    unsigned int n_insert = 0;

    if (this->m_gibbs)
        {
        if (nonzero)
            {
            lnb = Scalar(0.0);
            // compute configurational bias weight
            if (moveDepletantsIntoNewPosition(timestep, n_overlap, delta, pos, orientation, new_type, m_mc_implicit->getNumTrials(), lnb))
                {
                lnboltzmann -= lnb;
                }
            else
                {
                nonzero = false;
                }
            }
        unsigned int other = this->m_gibbs_other;

        if (this->m_exec_conf->getRank() == 0)
            {
            MPI_Request req[2];
            MPI_Status status[2];
            MPI_Isend(&n_overlap, 1, MPI_UNSIGNED, other, 0, MPI_COMM_WORLD, &req[0]);
            MPI_Irecv(&n_insert, 1, MPI_UNSIGNED, other, 0, MPI_COMM_WORLD, &req[1]);
            MPI_Waitall(2, req, status);
            }
        if (this->m_comm)
            {
            bcast(n_insert, 0, this->m_exec_conf->getMPICommunicator());
            }

        // try inserting depletants in new configuration
        if (moveDepletantsInUpdatedRegion(timestep, n_insert, delta_old, tag, new_type, m_mc_implicit->getNumTrials(), lnb))
            {
            lnboltzmann += lnb;
            }
        else
            {
            nonzero = false;
            }
        }
    #endif

    return nonzero;
    }


template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::tryRemoveParticle(unsigned int timestep, unsigned int tag, Scalar &lnboltzmann,
    bool communicate, unsigned int seed,
    std::vector<unsigned int> types,
    std::vector<vec3<Scalar> > positions,
    std::vector<quat<Scalar> > orientations)
    {
    // call parent class method
    lnboltzmann = Scalar(0.0);
    Scalar lnb(0.0);
    bool nonzero = UpdaterMuVT<Shape>::tryRemoveParticle(timestep, tag, lnb, communicate, seed, types, positions, orientations);

    if (nonzero)
        {
        lnboltzmann += lnb;
        }

    // number of depletants to insert
    #ifdef ENABLE_MPI
    unsigned int n_insert = 0;

    // zero overlapping depletants after removal
    unsigned int n_overlap = 0;

    if (this->m_gibbs)
        {
        unsigned int other = this->m_gibbs_other;

        if (this->m_exec_conf->getRank() == 0)
            {
            MPI_Request req[2];
            MPI_Status status[2];
            MPI_Isend(&n_overlap, 1, MPI_UNSIGNED, other, 0, MPI_COMM_WORLD, &req[0]);
            MPI_Irecv(&n_insert, 1, MPI_UNSIGNED, other, 0, MPI_COMM_WORLD, &req[1]);
            MPI_Waitall(2, req, status);
            }
        if (this->m_comm)
            {
            bcast(n_insert, 0, this->m_exec_conf->getMPICommunicator());
            }
        }
    #endif

    // only if the particle to be removed actually exists
    #ifdef ENABLE_MPI
    if (this->m_gibbs)
        {
        if (tag != UINT_MAX)
            {
            // old type
            unsigned int type = this->m_pdata->getType(tag);

            // Depletant type
            unsigned int type_d = m_mc_implicit->getDepletantType();

            // Depletant and colloid diameter
            Scalar d_dep, d_colloid_old;
                {
                const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> >& params = this->m_mc->getParams();
                quat<Scalar> o;
                Shape tmp(o, params[type_d]);
                d_dep = tmp.getCircumsphereDiameter();

                Shape shape_old(o, params[type]);
                d_colloid_old = shape_old.getCircumsphereDiameter();
                }

            // try inserting depletants in new configuration (where particle is removed)
            Scalar delta = d_dep + d_colloid_old;
            if (moveDepletantsIntoOldPosition(timestep, n_insert, delta, tag, m_mc_implicit->getNumTrials(), lnb, true))
                {
                lnboltzmann += lnb;
                }
            else
                {
                nonzero = false;
                }
            }
        }
    #endif

    return nonzero;
    }

template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::moveDepletantsInUpdatedRegion(unsigned int timestep, unsigned int n_insert,
    Scalar delta, unsigned int tag, unsigned int new_type, unsigned int n_trial, Scalar &lnboltzmann)
    {
    lnboltzmann = Scalar(0.0);

    unsigned int type_d = m_mc_implicit->getDepletantType();

    // getPosition() takes into account grid shift, correct for that
    Scalar3 p = this->m_pdata->getPosition(tag)+this->m_pdata->getOrigin();
    int3 tmp = make_int3(0,0,0);
    this->m_pdata->getGlobalBox().wrap(p,tmp);
    vec3<Scalar> pos(p);

    bool is_local = this->m_pdata->isParticleLocal(tag);

    // initialize another rng
    #ifdef ENABLE_MPI
    hoomd::detail::Saru rng(timestep, this->m_seed, this->m_exec_conf->getPartition() ^0x974762fa );
    #else
    hoomd::detail::Saru rng(timestep, this->m_seed, 0x974762fa );
    #endif

    // update the aabb tree
    const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

    unsigned int zero = 0;

    if (is_local)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_rtag(this->m_pdata->getRTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> >& params = this->m_mc->getParams();

        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            unsigned int n_success = 0;

            for (unsigned int itrial = 0; itrial < n_trial; ++itrial)
                {
                // draw a random vector in the excluded volume sphere of the particle to be inserted
                Scalar theta = rng.template s<Scalar>(Scalar(0.0),Scalar(2.0*M_PI));
                Scalar z = rng.template s<Scalar>(Scalar(-1.0),Scalar(1.0));

                // random normalized vector
                vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

                // draw random radial coordinate in test sphere
                Scalar r3 = rng.template s<Scalar>();
                Scalar r = Scalar(0.5)*delta*powf(r3,1.0/3.0);

                // test depletant position
                vec3<Scalar> pos_test = pos+r*n;

                Shape shape_test(quat<Scalar>(), params[type_d]);
                if (shape_test.hasOrientation())
                    {
                    // if the depletant is anisotropic, generate orientation
                    shape_test.orientation = generateRandomOrientation(rng);
                    }

                bool overlap_old = false;

                detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

                unsigned int err_count = 0;
                // All image boxes (including the primary)
                const unsigned int n_images = image_list.size();
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image);

                    // stackless search
                    for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                        {
                        if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                            {
                            if (aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // load the old position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];

                                    if (h_tag.data[j] == tag)
                                        {
                                        // do not check against old particle configuration
                                        continue;
                                        }

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                    unsigned int type = __scalar_as_int(postype_j.w);

                                    Shape shape_j(quat<Scalar>(orientation_j), params[type]);

                                    if (h_overlaps.data[overlap_idx(type,type_d)]
                                        && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                        && test_overlap(r_ij, shape_test, shape_j, err_count))
                                        {
                                        overlap_old = true;
                                        break;
                                        }
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                            }

                        if (overlap_old)
                            break;
                        } // end loop over AABB nodes
                    if (overlap_old)
                        break;
                    } // end loop over images

                if (!overlap_old)
                    {
                    // resolve the updated particle tag
                    unsigned int j = h_rtag.data[tag];
                    assert(j < this->m_pdata->getN());

                    // load the old position and orientation of the udpated particle
                    Scalar4 postype_j = h_postype.data[j];
                    Scalar4 orientation_j = h_orientation.data[j];

                    // see if it overlaps with depletant
                    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                        {
                        vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                        vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                        // old particle shape
                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_old(quat<Scalar>(orientation_j), params[typ_j]);

                        if (h_overlaps.data[overlap_idx(type_d, typ_j)]
                            && check_circumsphere_overlap(r_ij, shape_test, shape_old)
                            && test_overlap(r_ij, shape_test, shape_old, err_count))
                            {
                            // overlap with old particle configuration

                            // new particle shape
                            Shape shape_new(quat<Scalar>(orientation_j), params[new_type]);

                            if (!(h_overlaps.data[overlap_idx(type_d,new_type)]
                                && check_circumsphere_overlap(r_ij, shape_test, shape_new)
                                && test_overlap(r_ij, shape_test, shape_new, err_count)))
                                {
                                // no overlap wth new configuration
                                n_success++;
                                }
                            }
                        }
                    }
                } // end loop over insertion attempts

            if (n_success)
                {
                lnboltzmann += log((Scalar) n_success/(Scalar)n_trial);
                }
            else
                {
                zero = 1;
                }
            } // end loop over test depletants
        } // end is_local

    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &lnboltzmann, 1, MPI_HOOMD_SCALAR, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &zero, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        }
    #endif

    return !zero;
    }

template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::moveDepletantsIntoNewPosition(unsigned int timestep, unsigned int n_insert,
    Scalar delta, vec3<Scalar> pos, quat<Scalar> orientation, unsigned int type, unsigned int n_trial, Scalar &lnboltzmann)
    {
    lnboltzmann = Scalar(0.0);
    unsigned int zero = 0;

    unsigned int type_d = m_mc_implicit->getDepletantType();

    bool is_local = true;
    #ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        const BoxDim& global_box = this->m_pdata->getGlobalBox();
        is_local = this->m_exec_conf->getRank() == this->m_pdata->getDomainDecomposition()->placeParticle(global_box, vec_to_scalar3(pos));
        }
    #endif

    // initialize another rng
    #ifdef ENABLE_MPI
    hoomd::detail::Saru rng(timestep, this->m_seed, this->m_exec_conf->getPartition() ^0x123b09af );
    #else
    hoomd::detail::Saru rng(timestep, this->m_seed, 0x123b09af );
    #endif

    // update the image list
    const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

    if (is_local)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> >& params = this->m_mc->getParams();

        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            // Number of successfully reinsert depletants

            // we start with one because of super-detailed balance (we already inserted one overlapping depletant in the trial move)
            unsigned int n_success = 1;

            // Number of allowed insertion trials (those which overlap with colloid at old position)
            unsigned int n_overlap_shape = 1;

            for (unsigned int itrial = 0; itrial < n_trial; ++itrial)
                {
                // draw a random vector in the excluded volume sphere of the particle to be inserted
                Scalar theta = rng.template s<Scalar>(Scalar(0.0),Scalar(2.0*M_PI));
                Scalar z = rng.template s<Scalar>(Scalar(-1.0),Scalar(1.0));

                // random normalized vector
                vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

                // draw random radial coordinate in test sphere
                Scalar r3 = rng.template s<Scalar>();
                Scalar r = Scalar(0.5)*delta*powf(r3,1.0/3.0);

                // test depletant position
                vec3<Scalar> pos_test = pos+r*n;

                Shape shape_test(quat<Scalar>(), params[type_d]);
                if (shape_test.hasOrientation())
                    {
                    // if the depletant is anisotropic, generate orientation
                    shape_test.orientation = generateRandomOrientation(rng);
                    }

                // check against overlap with old configuration
                bool overlap_old = false;

                detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

                unsigned int err_count = 0;
                if (this->m_pdata->getN()+this->m_pdata->getNGhosts())
                    {
                    // All image boxes (including the primary)
                    const unsigned int n_images = image_list.size();
                    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                        {
                        vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                        detail::AABB aabb = aabb_test_local;
                        aabb.translate(pos_test_image);

                        // update the aabb tree (only valid if N>0)
                        const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

                        // stackless search
                        for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                            {
                            if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                                {
                                if (aabb_tree.isNodeLeaf(cur_node_idx))
                                    {
                                    for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                        {
                                        // read in its position and orientation
                                        unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                        Scalar4 postype_j;
                                        Scalar4 orientation_j;

                                        // load the old position and orientation of the j particle
                                        postype_j = h_postype.data[j];
                                        orientation_j = h_orientation.data[j];

                                        // put particles in coordinate system of particle i
                                        vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                                        Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                        if (h_overlaps.data[overlap_idx(type_d,typ_j)]
                                            && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                            && test_overlap(r_ij, shape_test, shape_j, err_count))
                                            {
                                            overlap_old = true;
                                            break;
                                            }
                                        }
                                    }
                                }
                            else
                                {
                                // skip ahead
                                cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                                }
                            if (overlap_old)
                                break;
                            } // end loop over AABB nodes
                        if (overlap_old)
                            break;
                        } // end loop over images
                    } // end AABB check

                const unsigned int n_images = image_list.size();
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    // checking the (0,0,0) image is sufficient
                    Shape shape(orientation, params[type]);
                    vec3<Scalar> r_ij = pos - pos_test;
                    if (h_overlaps.data[overlap_idx(type, type_d)]
                        && check_circumsphere_overlap(r_ij, shape_test, shape)
                        && test_overlap(r_ij, shape_test, shape, err_count))
                        {
                        if (!overlap_old) n_success++;
                        n_overlap_shape++;
                        }
                    } // end loop over images
                } // end loop over insertion attempts

            if (n_success)
                {
                lnboltzmann += log((Scalar)n_success/(Scalar)n_overlap_shape);
                }
            else
                {
                zero = 1;
                }
            } // end loop over test depletants
        } // is_local

    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &lnboltzmann, 1, MPI_HOOMD_SCALAR, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &zero, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        }
    #endif

    return !zero;
    }

template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::moveDepletantsIntoOldPosition(unsigned int timestep, unsigned int n_insert,
    Scalar delta, unsigned int tag, unsigned int n_trial, Scalar &lnboltzmann, bool need_overlap_shape)
    {
    lnboltzmann = Scalar(0.0);

    unsigned int type_d = m_mc_implicit->getDepletantType();

    // getPosition() corrects for grid shift, add it back
    Scalar3 p = this->m_pdata->getPosition(tag)+this->m_pdata->getOrigin();
    int3 tmp = make_int3(0,0,0);
    this->m_pdata->getGlobalBox().wrap(p,tmp);
    vec3<Scalar> pos(p);

    bool is_local = this->m_pdata->isParticleLocal(tag);

    // initialize another rng
    #ifdef ENABLE_MPI
    hoomd::detail::Saru rng(timestep, this->m_seed, this->m_exec_conf->getPartition() ^0x64f123da );
    #else
    hoomd::detail::Saru rng(timestep, this->m_seed, 0x64f123da );
    #endif

    // update the aabb tree
    const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

    unsigned int zero = 0;

    if (is_local)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_rtag(this->m_pdata->getRTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = this->m_mc->getParams();

        const Index2D & overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            // Number of successfully reinsert depletants
            unsigned int n_success = 0;

            // Number of allowed insertion trials (those which overlap with colloid at old position)
            unsigned int n_overlap_shape = 0;

            for (unsigned int itrial = 0; itrial < n_trial; ++itrial)
                {
                // draw a random vector in the excluded volume sphere of the particle to be inserted
                Scalar theta = rng.template s<Scalar>(Scalar(0.0),Scalar(2.0*M_PI));
                Scalar z = rng.template s<Scalar>(Scalar(-1.0),Scalar(1.0));

                // random normalized vector
                vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

                // draw random radial coordinate in test sphere
                Scalar r3 = rng.template s<Scalar>();
                Scalar r = Scalar(0.5)*delta*powf(r3,1.0/3.0);

                // test depletant position
                vec3<Scalar> pos_test = pos+r*n;

                Shape shape_test(quat<Scalar>(), params[type_d]);
                if (shape_test.hasOrientation())
                    {
                    // if the depletant is anisotropic, generate orientation
                    shape_test.orientation = generateRandomOrientation(rng);
                    }

                bool overlap_old = false;
                bool overlap = false;

                detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

                unsigned int err_count = 0;
                // All image boxes (including the primary)
                const unsigned int n_images = image_list.size();
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image);

                    // stackless search
                    for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                        {
                        if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                            {
                            if (aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // load the old position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];

                                    if (h_tag.data[j] == tag)
                                        {
                                        // do not check against old particle configuration
                                        continue;
                                        }

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                    unsigned int type = __scalar_as_int(postype_j.w);
                                    Shape shape_j(quat<Scalar>(orientation_j), params[type]);

                                    if (h_overlaps.data[overlap_idx(type_d, type)]
                                        && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                        && test_overlap(r_ij, shape_test, shape_j, err_count))
                                        {
                                        overlap_old = true;
                                        break;
                                        }
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                            }

                        if (overlap_old)
                            break;
                        } // end loop over AABB nodes
                    if (overlap_old)
                        break;
                    } // end loop over images

                // resolve the updated particle tag
                unsigned int j = h_rtag.data[tag];
                assert(j < this->m_pdata->getN());

                // load the old position and orientation of the udpated particle
                Scalar4 postype_j = h_postype.data[j];
                Scalar4 orientation_j = h_orientation.data[j];

                // see if it overlaps with depletant
                // only need to consider (0,0,0) image
                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test;

                unsigned int typ_j = __scalar_as_int(postype_j.w);
                Shape shape(quat<Scalar>(orientation_j), params[typ_j]);

                if (h_overlaps.data[overlap_idx(type_d, typ_j)]
                    && check_circumsphere_overlap(r_ij, shape_test, shape)
                    && test_overlap(r_ij, shape_test, shape, err_count))
                    {
                    overlap = true;
                    n_overlap_shape++;
                    }

                if (!overlap_old && (overlap || !need_overlap_shape))
                    {
                    // success if it overlaps with the particle identified by the tag
                    n_success++;
                    }
                } // end loop over insertion attempts

            if (n_success)
                {
                lnboltzmann += log((Scalar) n_success/(Scalar)n_overlap_shape);
                }
            else
                {
                zero = 1;
                }
            } // end loop over test depletants
        } // end is_local

    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &lnboltzmann, 1, MPI_HOOMD_SCALAR, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &zero, 1, MPI_UNSIGNED, MPI_MAX, this->m_exec_conf->getMPICommunicator());
        }
    #endif

    return !zero;
    }

template<class Shape, class Integrator>
std::vector<unsigned int> UpdaterMuVTImplicit<Shape,Integrator>::checkDepletantOverlaps(
    unsigned int timestep, unsigned int n_insert, unsigned int seed,
    unsigned int type,
    vec3<Scalar> pos,
    quat<Scalar> orientation,
    const std::vector<unsigned int>& insert_type,
    const std::vector<vec3<Scalar> >& insert_position,
    const std::vector<quat<Scalar> >& insert_orientation,
    const std::vector<unsigned int>& old_types,
    const std::vector<vec3<Scalar> >& old_positions,
    const std::vector<quat<Scalar> >& old_orientations)
    {
    unsigned int type_d = m_mc_implicit->getDepletantType();

    std::vector<unsigned int> overlap(insert_type.size(),1);

    auto params = this->m_mc->getParams();

    ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
    const Index2D & overlap_idx = this->m_mc->getOverlapIndexer();

    // NOTE assume here depletant is isotropic
    Shape shape_tmp(quat<Scalar>(), params[type_d]);
    Shape shape(orientation, params[type]);
    Scalar delta = shape.getCircumsphereDiameter() + shape_tmp.getCircumsphereDiameter();

    #ifdef ENABLE_TBB
    // avoid race condition
    if (this->m_pdata->getN() + this->m_pdata->getNGhosts())
        this->m_mc->buildAABBTree();
    #endif

    // first reject the inserted particles that are not in depletant-excluded volume
    const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();
    const unsigned int n_images = image_list.size();

    tbb::atomic<unsigned int> count = 0;
    #ifdef ENABLE_TBB
    tbb::parallel_for((unsigned int)0,(unsigned int)insert_type.size(), [&](unsigned int l)
    #else
    for (unsigned int l = 0; l < insert_type.size(); ++l)
    #endif
        {
        // if we cannot insert a depletant at the inserted particle's position, we are good
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_test_image = insert_position[l] + image_list[cur_image];
            vec3<Scalar> r_ij = pos - pos_test_image;
            bool circumsphere_overlap = dot(r_ij,r_ij)*4.0 <= delta*delta;
            unsigned int err_count = 0;
            if (h_overlaps.data[overlap_idx(type,type_d)]
                && circumsphere_overlap
                && test_overlap(r_ij, shape_tmp, shape, err_count))
                {
                // this looks counterintuitive, but in the end we retain only those particles with overlap[l] == 0
                overlap[l] = 0;
                count++;
                break;
                }
            } // end loop over images
        } // end loop over images
    #ifdef ENABLE_TBB
        );
    #endif

    // for every test depletant

    #ifdef ENABLE_TBB
    tbb::parallel_for((unsigned int)0,n_insert, [&](unsigned int k)
    #else
    for (unsigned int k = 0; k < n_insert; ++k)
    #endif
        {
        // generate a reproducible pseudo RNG seed
        std::vector<unsigned int> seed_seq(4);
        seed_seq[0] = this->m_seed;
        seed_seq[1] = k;
        seed_seq[2] = seed;
        seed_seq[3] = 0x1412459a;

        std::seed_seq seed(seed_seq.begin(), seed_seq.end());
        std::vector<unsigned int> s(1);
        seed.generate(s.begin(),s.end());

        // draw a random vector in the excluded volume sphere of the particle to be inserted
        hoomd::detail::Saru rng(timestep, s[0], this->m_exec_conf->getRank());

        vec3<Scalar> pos_test = generatePositionInSphere(rng, pos, 0.5*delta);

        Shape shape_test(quat<Scalar>(), params[type_d]);
        if (shape_test.hasOrientation())
            {
            // if the depletant is anisotropic, generate orientation
            shape_test.orientation = generateRandomOrientation(rng);
            }

        // depletant has be in insertion volume (excluded volume)
        vec3<Scalar> r_ij(pos_test-pos);
        bool circumsphere_overlap = dot(r_ij,r_ij)*4.0 <= delta*delta;
        unsigned int err_count = 0;
        if (!(h_overlaps.data[overlap_idx(type_d, type)]
            && circumsphere_overlap
            && test_overlap(r_ij, shape_test, shape, err_count)))
            {
            #ifdef ENABLE_TBB
            return;
            #else
            continue;
            #endif
            }

        // check against overlap with old configuration
        bool overlap_old = false;

        detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

        if (this->m_pdata->getN()+this->m_pdata->getNGhosts())
            {
            // update the aabb tree (only valid with N > 0 particles)
            const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

            ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_comm_flags(this->m_pdata->getCommFlags(), access_location::host, access_mode::read);

            // All image boxes (including the primary)
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                detail::AABB aabb = aabb_test_local;
                aabb.translate(pos_test_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                        {
                        if (aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                // skip removed particles
                                if (h_comm_flags.data[j]) continue;

                                Scalar4 postype_j;
                                Scalar4 orientation_j;

                                // load the old position and orientation of the j particle
                                postype_j = h_postype.data[j];
                                orientation_j = h_orientation.data[j];

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                if (h_overlaps.data[overlap_idx(type_d, typ_j)]
                                    && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                    && test_overlap(r_ij, shape_test, shape_j, err_count))
                                    {
                                    overlap_old = true;
                                    break;
                                    }
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                        }
                    if (overlap_old)
                        break;
                    } // end loop over AABB nodes
                if (overlap_old)
                    break;
                }
            } // end AABB tree check

        // check against list of particles provided
        const unsigned int n_images = image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];

            for (unsigned int n = 0; n < old_positions.size(); ++n)
                {
                // put particles in coordinate system of particle i
                vec3<Scalar> r_ij = old_positions[n] - pos_test_image;

                Shape shape_j(old_orientations[n], params[old_types[n]]);

                if (h_overlaps.data[overlap_idx(type_d, old_types[n])]
                    && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                    && test_overlap(r_ij, shape_test, shape_j, err_count))
                    {
                    overlap_old = true;
                    break;
                    }
                } // end loop over extra particles

            if (overlap_old)
                break;
            } // end loop over images

        if (overlap_old)
            {
            #ifdef ENABLE_TBB
            return;
            #else
            continue;
            #endif
            }

        for (unsigned int l = 0; l < insert_type.size(); ++l)
            {
            if (!overlap[l])
                {
                // see if it overlaps with inserted particle
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    Shape shape_insert(insert_orientation[l], params[insert_type[l]]);

                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    vec3<Scalar> r_ij = insert_position[l] - pos_test_image;
                    if (h_overlaps.data[overlap_idx(type_d, insert_type[l])]
                        && check_circumsphere_overlap(r_ij, shape_test, shape)
                        && test_overlap(r_ij, shape_test, shape_insert, err_count))
                        {
                        overlap[l] = 1;
                        break;
                        }
                    }
                }
            } // end loop over inserted particles
        } // end loop over test depletants
    #ifdef ENABLE_TBB
        );
    #endif

    std::vector<unsigned int> result;
    for (unsigned int l = 0; l < overlap.size(); ++l)
        {
        if (!overlap[l])
            result.push_back(l);
        }

    return result;
    }

template<class Shape, class Integrator>
unsigned int UpdaterMuVTImplicit<Shape,Integrator>::countDepletantOverlapsInNewPosition(unsigned int timestep,
    unsigned int n_insert, Scalar delta, vec3<Scalar> pos, quat<Scalar> orientation, unsigned int type,
    unsigned int &n_free, bool communicate, unsigned int seed)
    {
    // number of depletants successfully inserted
    unsigned int n_overlap = 0;

    unsigned int type_d = m_mc_implicit->getDepletantType();

    bool is_local = true;
    #ifdef ENABLE_MPI
    if (communicate && this->m_pdata->getDomainDecomposition())
        {
        const BoxDim& global_box = this->m_pdata->getGlobalBox();
        is_local = this->m_exec_conf->getRank() == this->m_pdata->getDomainDecomposition()->placeParticle(global_box, vec_to_scalar3(pos));
        }
    #endif

    // initialize another rng
    hoomd::detail::Saru rng(timestep, this->m_seed+seed, this->m_exec_conf->getRank() ^0x1412459a );

    n_free = 0;

    if (is_local)
        {
        auto params = this->m_mc->getParams();

        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
        const Index2D & overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            // draw a random vector in the excluded volume sphere of the particle to be inserted
            Scalar theta = rng.template s<Scalar>(Scalar(0.0),Scalar(2.0*M_PI));
            Scalar z = rng.template s<Scalar>(Scalar(-1.0),Scalar(1.0));

            // random normalized vector
            vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

            // draw random radial coordinate in test sphere
            Scalar r3 = rng.template s<Scalar>();
            Scalar r = Scalar(0.5)*delta*powf(r3,1.0/3.0);

            // test depletant position
            vec3<Scalar> pos_test = pos+r*n;

            Shape shape_test(quat<Scalar>(), params[type_d]);
            if (shape_test.hasOrientation())
                {
                // if the depletant is anisotropic, generate orientation
                shape_test.orientation = generateRandomOrientation(rng);
                }

            // check against overlap with old configuration
            bool overlap_old = false;

            detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

            unsigned int err_count = 0;

            // update the image list
            const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

            if (this->m_pdata->getN()+this->m_pdata->getNGhosts())
                {
                // update the aabb tree (only valid with N > 0 particles)
                const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

                ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
                ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
                ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
                ArrayHandle<unsigned int> h_comm_flags(this->m_pdata->getCommFlags(), access_location::host, access_mode::read);

                // All image boxes (including the primary)
                const unsigned int n_images = image_list.size();
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image);

                    // stackless search
                    for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                        {
                        if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                            {
                            if (aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    // skip removed particles
                                    if (h_comm_flags.data[j]) continue;

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // load the old position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                                    Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                    if (h_overlaps.data[overlap_idx(type_d, typ_j)]
                                        && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                        && test_overlap(r_ij, shape_test, shape_j, err_count))
                                        {
                                        overlap_old = true;
                                        break;
                                        }
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                            }
                        if (overlap_old)
                            break;
                        } // end loop over AABB nodes
                    if (overlap_old)
                        break;
                    }
                } // end AABB tree check

            if (! overlap_old)
                {
                n_free++;

                // see if it overlaps with inserted particle
                const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();
                for (unsigned int cur_image = 0; cur_image < image_list.size(); cur_image++)
                    {
                    Shape shape(orientation, params[type]);

                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    vec3<Scalar> r_ij = pos - pos_test_image;
                    if (h_overlaps.data[overlap_idx(type_d, type)]
                        && check_circumsphere_overlap(r_ij, shape_test, shape)
                        && test_overlap(r_ij, shape_test, shape, err_count))
                        {
                        n_overlap++;
                        }
                    }
                }
            } // end loop over test depletants
        } // is_local

    #ifdef ENABLE_MPI
    if (communicate && this->m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &n_overlap, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &n_free, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        }
    #endif

    return n_overlap;
    }

template<class Shape, class Integrator>
unsigned int UpdaterMuVTImplicit<Shape,Integrator>::countDepletantOverlapsInOldPosition(unsigned int timestep,
     unsigned int n_insert, Scalar delta, unsigned int tag, bool communicate, unsigned int seed)
    {
    // number of depletants successfully inserted
    unsigned int n_overlap = 0;

    unsigned int type_d = m_mc_implicit->getDepletantType();

    // getPosition() corrects for grid shift, add it back
    Scalar3 p = this->m_pdata->getPosition(tag)+this->m_pdata->getOrigin();
    int3 tmp = make_int3(0,0,0);
    this->m_pdata->getGlobalBox().wrap(p,tmp);
    vec3<Scalar> pos(p);

    unsigned int type = this->m_pdata->getType(tag);
    quat<Scalar> orientation(this->m_pdata->getOrientation(tag));

    bool is_local = this->m_pdata->isParticleLocal(tag);

    // initialize another rng
    hoomd::detail::Saru rng(timestep, this->m_seed+seed, this->m_exec_conf->getRank() ^0x1412459a );

    if (is_local)
        {
        auto params = this->m_mc->getParams();

        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
        const Index2D & overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            // test depletant position
            vec3<Scalar> pos_test = generatePositionInSphere(rng, pos, 0.5*delta);

            Shape shape_test(quat<Scalar>(), params[type_d]);
            if (shape_test.hasOrientation())
                {
                // if the depletant is anisotropic, generate orientation
                shape_test.orientation = generateRandomOrientation(rng);
                }

            // check against overlap with old configuration
            bool overlap_old = false;

            detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

            unsigned int err_count = 0;

            if (this->m_pdata->getN()+this->m_pdata->getNGhosts())
                {
                // update the aabb tree (only valid with N > 0 particles)
                const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

                // update the image list
                const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

                ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
                ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
                ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
                ArrayHandle<unsigned int> h_comm_flags(this->m_pdata->getCommFlags(), access_location::host, access_mode::read);

                // All image boxes (including the primary)
                const unsigned int n_images = image_list.size();
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image);

                    // stackless search
                    for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                        {
                        if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                            {
                            if (aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    // skip removed particles
                                    if (h_comm_flags.data[j]) continue;

                                    if (h_tag.data[j] == tag) continue;

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // load the old position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                                    Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                    if (h_overlaps.data[overlap_idx(type_d, typ_j)]
                                        && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                        && test_overlap(r_ij, shape_test, shape_j, err_count))
                                        {
                                        overlap_old = true;
                                        break;
                                        }
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                            }
                        if (overlap_old)
                            break;
                        } // end loop over AABB nodes
                    if (overlap_old)
                        break;
                    }
                } // end AABB tree check

            if (! overlap_old)
                {
                // see if it overlaps with re-inserted particle
                const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();
                for (unsigned int cur_image = 0; cur_image < image_list.size(); cur_image++)
                    {
                    Shape shape(orientation, params[type]);

                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    vec3<Scalar> r_ij = pos - pos_test_image;
                    if (h_overlaps.data[overlap_idx(type_d, type)]
                        && check_circumsphere_overlap(r_ij, shape_test, shape)
                        && test_overlap(r_ij, shape_test, shape, err_count))
                        {
                        n_overlap++;
                        }
                    }
                }
            } // end loop over test depletants
        } // is_local

    #ifdef ENABLE_MPI
    if (communicate && this->m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &n_overlap, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        }
    #endif

    return n_overlap;
    }

template<class Shape, class Integrator>
unsigned int UpdaterMuVTImplicit<Shape,Integrator>::countDepletantOverlaps(unsigned int timestep, unsigned int n_insert, Scalar delta, vec3<Scalar> pos)
    {
    // number of depletants successfully inserted
    unsigned int n_overlap = 0;

    unsigned int type_d = m_mc_implicit->getDepletantType();

    bool is_local = true;
    #ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        const BoxDim& global_box = this->m_pdata->getGlobalBox();
        is_local = this->m_exec_conf->getRank() == this->m_pdata->getDomainDecomposition()->placeParticle(global_box, vec_to_scalar3(pos));
        }
    #endif

    // initialize another rng
    #ifdef ENABLE_MPI
    hoomd::detail::Saru rng(timestep, this->m_seed, this->m_exec_conf->getPartition() ^0x1412459a );
    #else
    hoomd::detail::Saru rng(timestep, this->m_seed, 0x1412459a);
    #endif

    // update the aabb tree
    const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

    if (is_local)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);

        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = this->m_mc->getParams();

        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            // draw a random vector in the excluded volume sphere of the particle to be inserted
            Scalar theta = rng.template s<Scalar>(Scalar(0.0),Scalar(2.0*M_PI));
            Scalar z = rng.template s<Scalar>(Scalar(-1.0),Scalar(1.0));

            // random normalized vector
            vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

            // draw random radial coordinate in test sphere
            Scalar r3 = rng.template s<Scalar>();
            Scalar r = Scalar(0.5)*delta*powf(r3,1.0/3.0);

            // test depletant position
            vec3<Scalar> pos_test = pos+r*n;

            Shape shape_test(quat<Scalar>(), params[type_d]);
            if (shape_test.hasOrientation())
                {
                // if the depletant is anisotropic, generate orientation
                shape_test.orientation = generateRandomOrientation(rng);
                }

            // check against overlap with present configuration
            bool overlap = false;

            detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

            unsigned int err_count = 0;
            // All image boxes (including the primary)
            const unsigned int n_images = image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                detail::AABB aabb = aabb_test_local;
                aabb.translate(pos_test_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                        {
                        if (aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                Scalar4 postype_j;
                                Scalar4 orientation_j;

                                // load the old position and orientation of the j particle
                                postype_j = h_postype.data[j];
                                orientation_j = h_orientation.data[j];

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                if (h_overlaps.data[overlap_idx(typ_j, type_d)]
                                    && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                    && test_overlap(r_ij, shape_test, shape_j, err_count))
                                    {
                                    overlap = true;
                                    break;
                                    }
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                        }
                    if (overlap)
                        break;
                    } // end loop over AABB nodes
                if (overlap)
                    break;
                } // end loop over images

            if (overlap)
                {
                n_overlap++;
                }
            } // end loop over test depletants
        } // is_local

    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &n_overlap, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        }
    #endif

    return n_overlap;
    }


//! Get a poisson-distributed number of depletants
template<class Shape, class Integrator>
unsigned int UpdaterMuVTImplicit<Shape,Integrator>::getNumDepletants(unsigned int timestep,  Scalar V, unsigned int seed)
    {
    // parameter for Poisson distribution
    Scalar lambda = this->m_mc_implicit->getDepletantDensity()*V;

    unsigned int n = 0;
    if (lambda>Scalar(0.0))
        {
        std::poisson_distribution<unsigned int> poisson =
            std::poisson_distribution<unsigned int>(lambda);

        // combine five seeds
        std::vector<unsigned int> seed_seq(4);
        seed_seq[0] = this->m_seed;
        seed_seq[1] = timestep;
        seed_seq[2] = this->m_exec_conf->getRank();
        seed_seq[3] = seed;
        std::seed_seq s(seed_seq.begin(), seed_seq.end());

        // RNG for poisson distribution
        std::mt19937 rng_poisson(s);

        n = poisson(rng_poisson);
        }
    return n;
    }

template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::boxResizeAndScale(unsigned int timestep, const BoxDim old_box, const BoxDim new_box,
    unsigned int &extra_ndof, Scalar &lnboltzmann)
    {
    // call parent class method
    lnboltzmann = Scalar(0.0);

    unsigned int partition = 0;
    #ifdef ENABLE_MPI
    partition = this->m_exec_conf->getPartition();
    #endif

    bool result = UpdaterMuVT<Shape>::boxResizeAndScale(timestep, old_box, new_box, extra_ndof, lnboltzmann);

    if (result)
        {
        // update the aabb tree
        const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

        // update the image list
        const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

        if (this->m_prof) this->m_prof->push(this->m_exec_conf, "HPMC implicit volume move ");

        // access particle data and system box
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);

        // access parameters
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = this->m_mc->getParams();
        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        bool overlap = false;

        // get old local box
        BoxDim old_local_box = old_box;
        #ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            old_local_box = this->m_pdata->getDomainDecomposition()->calculateLocalBox(old_box);
            }
        #endif

        // draw number from Poisson distribution (using old box)
        unsigned int n = getNumDepletants(timestep, old_local_box.getVolume(), partition);

        // Depletant type
        unsigned int type_d = m_mc_implicit->getDepletantType();

        // place a cut-off on the result to avoid long-running loops
        unsigned int err_count = 0;

        // draw a random vector in the box
        #ifdef ENABLE_MPI
        hoomd::detail::Saru rng(this->m_seed, this->m_exec_conf->getNPartitions()*this->m_exec_conf->getRank()+this->m_exec_conf->getPartition(), timestep);
        #else
        hoomd::detail::Saru rng(this->m_seed, timestep);
        #endif

        uint3 dim = make_uint3(1,1,1);
        uint3 grid_pos = make_uint3(0,0,0);
        #ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            Index3D didx = this->m_pdata->getDomainDecomposition()->getDomainIndexer();
            dim = make_uint3(didx.getW(), didx.getH(), didx.getD());
            grid_pos = this->m_pdata->getDomainDecomposition()->getGridPos();
            }
        #endif

        // for every test depletant
        for (unsigned int k = 0; k < n; ++k)
            {
            Scalar xrand = rng.template s<Scalar>();
            Scalar yrand = rng.template s<Scalar>();
            Scalar zrand = rng.template s<Scalar>();

            Scalar3 f_test = make_scalar3(xrand, yrand, zrand);
            f_test = (f_test + make_scalar3(grid_pos.x,grid_pos.y,grid_pos.z))/make_scalar3(dim.x,dim.y,dim.z);
            vec3<Scalar> pos_test = vec3<Scalar>(new_box.makeCoordinates(f_test));

            Shape shape_test(quat<Scalar>(), params[type_d]);
            if (shape_test.hasOrientation())
                {
                // if the depletant is anisotropic, generate orientation
                shape_test.orientation = generateRandomOrientation(rng);
                }

            // check against overlap in old box
            overlap=false;
            bool overlap_old = false;
            detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

            // All image boxes (including the primary)
            const unsigned int n_images = image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                Scalar3 f = new_box.makeFraction(vec_to_scalar3(pos_test_image));
                vec3<Scalar> pos_test_image_old = vec3<Scalar>(old_box.makeCoordinates(f));

                // set up AABB in old coordinates
                detail::AABB aabb = aabb_test_local;
                aabb.translate(pos_test_image_old);

                // scale AABB to new coordinates (the AABB tree contains new coordinates)
                vec3<Scalar> lower, upper;
                lower = aabb.getLower();
                f = old_box.makeFraction(vec_to_scalar3(lower));
                lower = vec3<Scalar>(new_box.makeCoordinates(f));
                upper = aabb.getUpper();
                f = old_box.makeFraction(vec_to_scalar3(upper));
                upper = vec3<Scalar>(new_box.makeCoordinates(f));
                aabb = detail::AABB(lower,upper);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                        {
                        if (aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                Scalar4 postype_j;
                                Scalar4 orientation_j;

                                // load the old position and orientation of the j particle
                                postype_j = h_postype.data[j];
                                orientation_j = h_orientation.data[j];

                                // compute the particle position scaled in the old box
                                f = new_box.makeFraction(make_scalar3(postype_j.x,postype_j.y,postype_j.z));
                                vec3<Scalar> pos_j_old(old_box.makeCoordinates(f));

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j_old - pos_test_image_old;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                if (h_overlaps.data[overlap_idx(typ_j, type_d)]
                                    && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                    && test_overlap(r_ij, shape_test, shape_j, err_count))
                                    {
                                    overlap = true;

                                    // depletant is ignored for any overlap in the old configuration
                                    overlap_old = true;
                                    break;
                                    }
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                        }
                    if (overlap)
                        break;
                    }  // end loop over AABB nodes

                if (overlap)
                    break;

                } // end loop over images

            if (!overlap)
                {
                // depletant in free volume
                extra_ndof++;

                // check for overlap in new configuration

                // new depletant coordinates
                vec3<Scalar> pos_test(new_box.makeCoordinates(f_test));

                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image);

                    // stackless search
                    for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                       {
                       if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                            {
                            if (aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // load the new position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                                    Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                    if (h_overlaps.data[overlap_idx(typ_j, type_d)]
                                         && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                         && test_overlap(r_ij, shape_test, shape_j, err_count))
                                         {
                                         overlap = true;
                                         break;
                                         }
                                     }
                                 }
                             }
                         else
                             {
                             // skip ahead
                             cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                             }

                         }  // end loop over AABB nodes

                    if (overlap)
                        break;
                    } // end loop over images
               } // end overlap check in new configuration

            if (overlap_old)
                {
                overlap = false;
                continue;
                }

            if (overlap)
                break;
            } // end loop over test depletants

        unsigned int overlap_count = overlap;

        #ifdef ENABLE_MPI
        if (this->m_comm)
            {
            MPI_Allreduce(MPI_IN_PLACE, &overlap_count, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE, &extra_ndof, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
            }
        #endif

        if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

        result = !overlap_count;
        }
    return result;
    }
} // end namespace

#endif
