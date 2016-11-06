// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _GIBBS_SAMPLER_H
#define _GIBBS_SAMPLER_H

/*! \file GibbsSampler.h
\brief Declaration of GibbsSampler base class
*/

#include "ExternalField.h"
#include <hoomd/SnapshotSystemData.h>
//NOTE: Check if we prefer to do this internally for some reason
#include <hoomd/extern/pybind/include/pybind11/stl.h>
#include <chrono>
#include <thread>

/*
Oustanding issues;
//NOTE: I'm not sure why it's necessary to always use the "this" keyword in this->m_pdata and this->m_exec_conf since they should be part of this class. Worth checking with Jens.
//NOTE: Still need to allow MPI decomposition
//NOTE: The way I'm dealing with multiple GPUArray acquires is generally clean for thing like positions, but not for the overlaps matrix. The fact that I re-store that locally in the compute function each time is pretty hackish. Ideally I need to implement something in GPUArrays that allows release and reacquire, but that may just be a separate project, and then I can come back here after to fix it
//NOTE: Is it worth defining a central method for checking overlaps and having my two functions call that? The main reason not to is that the underlying particle data positions are stored as vec<Scalar3>, whereas I store the Gibbs particles internally as vec3<Scalar>. In order to have a centralized, consistent interface, I would have to interconvert, which seems computationally inefficient and unnecessary
//NOTE: The size of the AABB list is now just double the poisson parameter, which is a bit hackish and is really not the smartest solution, especially since with some (probably vanishingly small, but still finite) probability it could fail if we pull an unusually large value from the distribution. A smarter way would be to dynamically resize whenever the AABB list was in danger of getting too long.
 */
namespace hpmc
{
template< class Shape >
class GibbsSampler : public ExternalFieldMono<Shape>
    {
    public:
         /**
          * Constructor for class
          * @param sysdef       The HOOMD system definition for that we are associating this compute with
          * @param mc           The integrator instance we are associating this compute with
          * @param type_indices An array of the type indices that we want to include in the Gibbs Sampler. Each one corresponds to a separate species in the sampler
          * @param type_densities  The desired number of each type to be sampled each time
          */
        GibbsSampler(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<IntegratorHPMCMono<Shape> > mc, std::vector<unsigned int> type_indices, std::vector<Scalar> type_densities, unsigned int seed);

        /**
        * Destructor method. Simply frees the space used by the various class level constructs
        */
        ~GibbsSampler();

        /**
         * Determines whether the move of the particle at index is accepted or rejected
         * @param  index        The index in the ParticleData arrays
         * @param  position_old The old position of the particle
         * @param  shape_old    The old shape of the particle
         * @param  position_new The new position of the particle
         * @param  shape_new    The new shape of the particle
         * @return              A boolean value indicated whether or not the move was accepted
         */
        bool accept(const unsigned int& index, const unsigned int type_i, const vec3<Scalar>& position_old, const Shape& shape_old, const vec3<Scalar>& position_new, const Shape& shape_new, Saru& rng)
            {
                //printf("The boltzmann value is %f\n", boltzmann(index, type_i, position_old, shape_old, position_new, shape_new));
            return boltzmann(index, type_i, position_old, shape_old, position_new, shape_new) == 1;
            }

        // Unfortunately there's no clean way around having both of these functions because partial specialization is forbidden in C++ and we have to have specialized templates for pybind. I've encapsulated all of the actual logic into one helper function, though
        /**
         * Adds the Gibbs particles to a provides snapshot for visualization purposes. Used when python is in float mode
         * @param snapshot A snapshot to add the particle data to
         */
        void appendSnapshot_float(std::shared_ptr< SnapshotSystemData<float> > snapshot)
            {
            this->appendSnapshot<float>(snapshot);
            }

        /**
         * Adds the Gibbs particles to a provides snapshot for visualization purposes. Used when python is in double mode
         * @param snapshot A snapshot to add the particle data to
         */
        void appendSnapshot_double(std::shared_ptr< SnapshotSystemData<double> > snapshot)
            {
            this->appendSnapshot<double>(snapshot);
            }

        /**
         * Computes the boltzmann factor. Since we are only dealing with hard particles, the boltzmann factor is solely determined by whether or not there are overlaps
         * @param  index        The index in the ParticleData arrays
         * @param  typ_i        The type of the particle
         * @param  position_old The old position of the particle
         * @param  shape_old    The old shape of the particle
         * @param  position_new The new position of the particle
         * @param  shape_new    The new shape of the particle
         * @return              The boltzmann factor associated with the two states. Is either 0 (new state is higher energy) or 1 (new state is lower energy)
         */
        Scalar boltzmann(const unsigned int& index, const unsigned int typ_i, const vec3<Scalar>& position_old, const Shape& shape_old, const vec3<Scalar>& position_new, const Shape& shape_new);

        /**
        * Required compute function
        * @param timestep The current timestep in the simulation
        */
        void compute(unsigned int timestep);

        /**
         * How many particles of type i exist in the system
         * @param type_i   The type of particle for which we want counts
         */
        unsigned int getCount(const unsigned int type_i)
            {
            return m_type_counts[std::find(m_type_indices.begin(), m_type_indices.end(), type_i) - m_type_indices.begin()];
            }

        /**
         * Change the densities of the particle types specified in type_indices to the new densities
         * @param type_indices   The types (the CPP indices) we want to change densities for
         * @param type_densities The corresponding new densities
         */
        void setParams(std::vector<unsigned int> type_indices, std::vector<Scalar> type_densities);

    protected:
        /**
         * Adds the Gibbs particles to a provides snapshot for visualization purposes. Used when python is in double mode
         * @param snapshot A snapshot to add the particle data to
         */
        template<typename Real>
        void appendSnapshot(std::shared_ptr< SnapshotSystemData<Real> > snapshot);

        /**
         * Determine whether or not a particle of typ_i at position_i with shape_i will overlap with any particles of the type corresponding to typ_index_j
         * @param  typ_i       The type of the particle being checked
         * @param  position_i  The position of the particle being checked
         * @param  shape_i     The orientation of the particle being checked
         * @param  typ_index_j The type index to check against
         * @return             Whether the particle overlaps with any other particles in aabb_tree
         */
        bool checkGibbsOverlaps(unsigned int typ_i, const vec3<Scalar>& position_i, const Shape& shape_i, int typ_index_j);

        /**
         * Determine whether or not a particle of typ_i at position_i with shape_i will overlap with any system particles
         * @param  typ_i       The type of the particle being checked
         * @param  position_i  The position of the particle being checked
         * @param  shape_i     The orientation of the particle being checked
         * @return             Whether the particle overlaps with any other particles in aabb_tree
         */
        bool checkOverlaps(unsigned int typ_i, const vec3<Scalar>& position_i, const Shape& shape_i);

         /**
          * Helper function to generate a random configuration of particles. Used to place solvent
          * @param type_index_i The index in m_type_indices of the current type we are generating particles for. Used to determine counts and check overlaps
          * @param timestep   The current simulation timestep
          */
        std::pair<std::vector<vec3<Scalar> >, std::vector<quat<Scalar> > > generateRandomConfiguration(unsigned int type_index_i, unsigned int timestep);

        /**
         * Updates the list of  poisson distributions with the mean values
         */
        void initializePoissonDistribution();

    private:
        std::vector<std::vector<vec3<Scalar> >, managed_allocator<std::vector<vec3<Scalar> > > > m_positions;
        std::vector<std::vector<quat<Scalar> >, managed_allocator<std::vector<quat<Scalar> > > > m_orientations;
        std::vector<detail::AABBTree, managed_allocator<detail::AABBTree> > m_aabb_trees;
        std::vector<detail::AABB*, managed_allocator<detail::AABB*> > m_aabbs;

        std::shared_ptr<IntegratorHPMCMono<Shape> > m_mc;
        std::vector<unsigned int> m_type_indices;
        std::vector<Scalar> m_type_densities;
        std::vector<unsigned int> m_type_samples; // The number of particles sampled
        std::vector<unsigned int> m_type_counts; // The number of particles actually placed
        unsigned int m_num_species; // The number of distinct species in the Gibbs Sampler

        GPUArray<unsigned int> m_overlaps;          //!< Interaction matrix (0/1) for overlap checks

        std::vector<std::poisson_distribution<unsigned int> > m_poisson;   //!< Poisson distribution
        unsigned int m_seed;                       //!< Random number seed
    };

template<class Shape>
GibbsSampler<Shape>::GibbsSampler(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<IntegratorHPMCMono<Shape> > mc, std::vector<unsigned int> type_indices, std::vector<Scalar> type_densities, unsigned int seed) : ExternalFieldMono<Shape>(sysdef), m_mc(mc), m_type_indices(type_indices), m_type_densities(type_densities), m_num_species(type_densities.size()), m_seed(seed)
    {
    // Have to ensure that the vector of configurations and the aabb trees are identically indexed
    //NOTE: Do these need to have managed allocators too?
    m_type_samples = std::vector<unsigned int>();
    m_type_counts = std::vector<unsigned int>();
    m_positions = std::vector<std::vector<vec3<Scalar> >, managed_allocator<std::vector<vec3<Scalar> > > >();
    m_orientations = std::vector<std::vector<quat<Scalar> >, managed_allocator<std::vector<quat<Scalar> > > >();
    m_aabb_trees = std::vector<detail::AABBTree, managed_allocator<detail::AABBTree> >();
    m_aabbs = std::vector<detail::AABB*, managed_allocator<detail::AABB*> >();
    m_positions.clear();
    m_orientations.clear();
    m_aabb_trees.clear();
    m_aabbs.clear();
    detail::AABB* temp_aabb;

    const BoxDim& box = this->m_pdata->getBox();
    Scalar vol = box.getVolume();

    for(unsigned int i = 0; i < m_num_species; i++)
        {
        m_type_samples.push_back(std::round(vol*m_type_densities[i]));
        m_type_counts.push_back(0);
        m_positions.push_back(std::vector<vec3<Scalar> >());
        m_orientations.push_back(std::vector<quat<Scalar> >());
        m_aabb_trees.push_back(detail::AABBTree());

        int retval = posix_memalign((void**)&temp_aabb, 32, 2*m_type_samples[i]*sizeof(detail::AABB));

        if (retval != 0)
            {
            this->m_exec_conf->msg->error() << "Error allocating aligned memory" << std::endl;
            throw std::runtime_error("Error allocating AABB memory");
            }
        else
            {
            m_aabbs.push_back(temp_aabb);
            }

        }

    m_poisson = std::vector<std::poisson_distribution<unsigned int> >();
    this->initializePoissonDistribution();
    }

template<class Shape>
GibbsSampler<Shape>::~GibbsSampler()
    {
    m_positions.clear();
    m_orientations.clear();
    m_aabb_trees.clear();
    for(unsigned int i = 0; i < m_num_species; i++)
        {
        free(m_aabbs[i]);
        }
    m_aabbs.clear();
    }

template<class Shape>
Scalar GibbsSampler<Shape>::boltzmann(const unsigned int& index, const unsigned int typ_i, const vec3<Scalar>& position_old, const Shape& shape_old, const vec3<Scalar>& position_new, const Shape& shape_new)
    {
    // Use the overlap indexer to map the (i, j) type pairs to a linear index in the overlaps array
    bool overlap = false;
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);
    const Index2D& indexer = m_mc->getOverlapIndexer();

    // Now loop over the different Gibbs Sampler particles and test those for overlaps
    for (unsigned int j = 0; j < m_num_species; j++)
        {
        unsigned int typ_j = m_type_indices[j];
        unsigned int overlap_index = indexer(typ_i, typ_j);
        if (h_overlaps.data[overlap_index]) // Can skip this for pairs that can overlap
            {
            overlap = checkGibbsOverlaps(typ_i, position_new, shape_new, j);
            if (overlap) { return 0;}
            }
        }
    // We won't get this far unless all overlap checks were successful
    return 1;
    }

template<class Shape>
void GibbsSampler<Shape>::compute(unsigned int timestep)
    {
    m_overlaps = m_mc->getInteractionMatrix();
    const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc->getParams();
    for (unsigned int i = 0; i < m_num_species; i++)
        {
        // For each element, we can just generate a new configurationkkk
        std::pair<std::vector<vec3<Scalar> >, std::vector<quat<Scalar> > > new_configuration(generateRandomConfiguration(i, timestep));
        m_positions[i] = new_configuration.first;
        m_orientations[i] = new_configuration.second;

        // We have to reconstruct AABBs for each particle and save them so we can pass a full list of AABBs to the AABBTree construction
        for(unsigned int j = 0; j < m_positions[i].size(); j++)
            {
                Shape shape_j(m_orientations[i][j], params[m_type_indices[i]]);
                m_aabbs[i][j] = shape_j.getAABB(m_positions[i][j]);
            } // End loop over particles

        // Now construct an AABB tree for each of them
        m_aabb_trees[i].buildTree(m_aabbs[i], m_type_counts[i]);
        } // End loop over types
    }

template<class Shape>
void GibbsSampler<Shape>::setParams(std::vector<unsigned int> type_indices, std::vector<Scalar> type_densities)
    {
    // Most naive algorithm, but this should never be a performance bottleneck anyway
    for(unsigned int i = 0; i < type_indices.size(); i++)
        {
        for(unsigned int j = 0; j < m_type_indices.size(); j++)
            {
            if(m_type_indices[j] == type_indices[i])
                {
                detail::AABB* temp_aabb;
                m_type_densities[j] = type_densities[i];
                const BoxDim& box = this->m_pdata->getBox();
                Scalar vol = box.getVolume();
                unsigned int new_sample_size = std::round(vol*m_type_densities[j]);

                if (new_sample_size > m_type_samples[j])
                    {
                    if (m_aabbs[j] != NULL)
                        {
                        int retval = posix_memalign((void**)&temp_aabb, 32, 2*new_sample_size*sizeof(detail::AABB));
                        if (retval != 0)
                            {
                            this->m_exec_conf->msg->error() << "Error allocating aligned memory" << std::endl;
                            throw std::runtime_error("Error allocating AABB memory");
                            }
                        else
                            {
                            free(m_aabbs[j]);
                            m_aabbs[j] = temp_aabb;
                            }
                        }
                    }
                m_type_samples[j] = new_sample_size;
                this->initializePoissonDistribution();
                }
            } // End loop over included types
        } // End loops over types whose params are getting set
    }

/*
template<class Shape>
bool GibbsSampler<Shape>::checkOverlapsHelper(const detail::AABBTree& aabb_tree, const vec3<Scalar>& position_i, const Shape& shape_i, const vec3*)
    {
        // List of things I need:
        // The aabb_tree, which is fully specified by the typ_index_j
        // Position and shape of the particle I'm checking
    detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

    const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc->getParams();
    bool overlap = false;
    unsigned int err_count = 0;
    // All image boxes (including the primary)
    std::vector<vec3<Scalar> > image_list = this->m_mc->updateImageList();
    const unsigned int n_images = image_list.size();
    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
        {
        vec3<Scalar> pos_i_image = position_i + image_list[cur_image];
        detail::AABB aabb = aabb_i_local;
        aabb.translate(pos_i_image);

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

                        // load the position and orientation of the j particle
                        vec3<Scalar> position_j = m_positions[typ_index_j][j];
                        quat<Scalar> orientation_j = m_orientations[typ_index_j][j];
                        Shape shape_j(quat<Scalar>(orientation_j), params[m_type_indices[typ_index_j]]);

                        // put particles in coordinate system of particle i
                        vec3<Scalar> r_ij = position_j - pos_i_image;

                        if (check_circumsphere_overlap(r_ij, shape_i, shape_j)
                            && test_overlap(r_ij, shape_i, shape_j, err_count))
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

            if (overlap){break;}
            }  // end loop over AABB nodes

        if (overlap){break;}
        }
        return overlap;
    }
    */

template<class Shape>
bool GibbsSampler<Shape>::checkGibbsOverlaps(unsigned int typ_i, const vec3<Scalar>& position_i, const Shape& shape_i, int typ_index_j)
    {
    // Set up the aabb tree we're testing against
    const detail::AABBTree& aabb_tree = m_aabb_trees[typ_index_j];

    detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

    const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc->getParams();
    bool overlap = false;
    unsigned int err_count = 0;

    // All image boxes (including the primary)
    std::vector<vec3<Scalar> > image_list = this->m_mc->updateImageList();
    const unsigned int n_images = image_list.size();
    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
        {
        vec3<Scalar> pos_i_image = position_i + image_list[cur_image];
        detail::AABB aabb = aabb_i_local;
        aabb.translate(pos_i_image);

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

                        // load the position and orientation of the j particle
                        vec3<Scalar> position_j = m_positions[typ_index_j][j];
                        quat<Scalar> orientation_j = m_orientations[typ_index_j][j];
                        Shape shape_j(quat<Scalar>(orientation_j), params[m_type_indices[typ_index_j]]);

                        // put particles in coordinate system of particle i
                        vec3<Scalar> r_ij = position_j - pos_i_image;

                        if (check_circumsphere_overlap(r_ij, shape_i, shape_j)
                            && test_overlap(r_ij, shape_i, shape_j, err_count))
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

            if (overlap){break;}
            }  // end loop over AABB nodes

        if (overlap){break;}
        } // end loop over images
    return overlap;
    }

template<class Shape>
bool GibbsSampler<Shape>::checkOverlaps(unsigned int typ_i, const vec3<Scalar>& position_i, const Shape& shape_i)
    {
    // Set up the aabb tree we're testing against
    const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

    const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc->getParams();
    ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    const Index2D& indexer = m_mc->getOverlapIndexer();
    bool overlap = false;
    unsigned int err_count = 0;

    // All image boxes (including the primary)
    std::vector<vec3<Scalar> > image_list = this->m_mc->updateImageList();
    const unsigned int n_images = image_list.size();
    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
        {
        vec3<Scalar> pos_i_image = position_i + image_list[cur_image];
        detail::AABB aabb = aabb_i_local;
        aabb.translate(pos_i_image);

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

                        // load the position and orientation of the j particle
                        Scalar4 postype_j = h_postype.data[j];
                        vec3<Scalar> position_j = vec3<Scalar>(postype_j);
                        quat<Scalar> orientation_j = quat<Scalar>(h_orientation.data[j]);
                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                        // put particles in coordinate system of particle i
                        vec3<Scalar> r_ij = position_j - pos_i_image;

                        if (h_overlaps.data[indexer(typ_i, typ_j)]
                            && check_circumsphere_overlap(r_ij, shape_i, shape_j)
                            && test_overlap(r_ij, shape_i, shape_j, err_count))
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

            if (overlap){break;}
            }  // end loop over AABB nodes

        if (overlap){break;}
        } // end loop over images
    return overlap;
    }


template<class Shape>
std::pair<std::vector<vec3<Scalar> >, std::vector<quat<Scalar> > > GibbsSampler<Shape>::generateRandomConfiguration(unsigned int type_index_i, unsigned int timestep)
    {
    //NOTE: Still have to account for the number of ranks for MPI
    //#ifdef ENABLE_MPI
    //num_particles /= m_mc.m_exec_conf->getNRanks();
    //#endif

    // access parameters and interaction matrix
    const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc->getParams();
    const BoxDim& box = this->m_pdata->getBox();

    // Interaction matrix
    const Index2D& indexer = m_mc->getOverlapIndexer();
    unsigned int type_j, overlap_index;
    unsigned int type_i = m_type_indices[type_index_i];

    // update the image list
    std::vector<vec3<Scalar> > image_list = this->m_mc->updateImageList();

    // We need to add particles to save each time we find a valid one
    std::vector<vec3<Scalar> > positions_new = std::vector<vec3<Scalar> >();
    std::vector<quat<Scalar> > orientations_new = std::vector<quat<Scalar> >();
    bool overlap;

    // Generate Poisson value
    std::vector<unsigned int> seed_seq(3);
    seed_seq[0] = this->m_seed;
    seed_seq[1] = timestep;
    seed_seq[2] = this->m_exec_conf->getRank();
    std::seed_seq seed(seed_seq.begin(), seed_seq.end());
    std::mt19937 rng_poisson(seed);
    unsigned int random_num_samples = m_poisson[type_index_i](rng_poisson);
    //unsigned int random_num_samples = m_type_samples[type_index_i];

    // This is looping over the number of particles to try to insert
    for (unsigned int i = 0; i < random_num_samples ; i++)
        {
        overlap = false;
        // select a random particle coordinate in the box
        Saru rng_i(i, m_seed + this->m_exec_conf->getRank() + type_index_i, timestep);

        Scalar xrand = rng_i.f();
        Scalar yrand = rng_i.f();
        Scalar zrand = rng_i.f();

        Scalar3 f = make_scalar3(xrand, yrand, zrand);
        vec3<Scalar> pos_i = vec3<Scalar>(box.makeCoordinates(f));

        Shape shape_i(quat<Scalar>(), params[type_i]);
        if (shape_i.hasOrientation())
            {
            shape_i.orientation = generateRandomOrientation(rng_i);
            }

        // Check whether we overlap with the actual system particles. This isn't necessary for accepting moves because those overlaps are checked by the integrator, but when we generate new configs we need to make sure the Gibbs particles don't overlap with system particles
        overlap = checkOverlaps(type_i, pos_i, shape_i);

        // If the main particles overlapped then no need to continue
        if(overlap) { continue;}

        // Now loop over the other Gibbs types and check those as well
        for (unsigned int j = 0; j < m_num_species; j++)
            {
            type_j = m_type_indices[j];
            bool check_overlaps = false;
            overlap_index = indexer(type_i, type_j);
            // Scope this so ArrayHandle doesn't cause problems
                {
                ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
                check_overlaps = h_overlaps.data[overlap_index];
                }
            if (check_overlaps) // Can skip this for pairs that can overlap
                {
                overlap = checkGibbsOverlaps(type_i, pos_i, shape_i, j);
                if (overlap) { break;}
                }
            }

        if (overlap)
            {
            continue;
            }
        else
            {
            positions_new.push_back(pos_i);
            orientations_new.push_back(shape_i.orientation);
            }
        } // end loop through all particles

    // Update the actual counts now
    m_type_counts[type_index_i] = positions_new.size();

    std::pair<std::vector<vec3<Scalar> >, std::vector<quat<Scalar> > > return_val(positions_new, orientations_new);
    return return_val;
    }

template<class Shape>
template<class Real>
void GibbsSampler<Shape>::appendSnapshot(std::shared_ptr< SnapshotSystemData<Real> > snapshot)
//NOTE: I don't know why I have to explicitly include the SnapshotSystemData file again. It doesn't make sense to me since it's already included by ExternalField
    {
    unsigned int n_gibbs_particles = 0;
    SnapshotParticleData<Real>& pdata = snapshot->particle_data;
    unsigned int snap_id = pdata.size;
    for(unsigned int i = 0; i < m_num_species; i++)
        {
        n_gibbs_particles += m_type_counts[i];
        }
    pdata.resize(pdata.size + n_gibbs_particles);

    for(unsigned int i = 0; i < m_num_species; i++)
        {
        for(unsigned int j = 0; j < m_type_counts[i]; j++)
            {
            pdata.pos[snap_id] = m_positions[i][j];
            pdata.orientation[snap_id] = m_orientations[i][j];
            pdata.diameter[snap_id] = 1;
            pdata.type[snap_id] = m_type_indices[i];
            snap_id++;
            }
        }
    }

template<class Shape>
void GibbsSampler< Shape >::initializePoissonDistribution()
    {
    m_poisson.resize(this->m_num_species);

    for (unsigned int i = 0; i < this->m_num_species; i++)
        {
        // parameter for Poisson distribution
        m_poisson[i] = std::poisson_distribution<unsigned int>(this->m_type_samples[i]);
        }
    }

/*
 * Function to create the pybind function to expose this to python
 */
template<class Shape> void export_GibbsSampler(pybind11::module& m, std::string name)
    {
   pybind11::class_<GibbsSampler<Shape>, std::shared_ptr< GibbsSampler<Shape> > >(m, name.c_str(), pybind11::base< ExternalFieldMono<Shape> >())
    .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<IntegratorHPMCMono<Shape> >, std::vector<unsigned int>, std::vector<Scalar>, unsigned int >())
    .def("setParams", &GibbsSampler<Shape >::setParams)
    .def("getCount", &GibbsSampler<Shape >::getCount)
    .def("appendSnapshot_float", &GibbsSampler<Shape>::appendSnapshot_float)
    .def("appendSnapshot_double", &GibbsSampler<Shape>::appendSnapshot_double)
    ;
    }
}

#endif // _GIBBS_SAMPLER_H
