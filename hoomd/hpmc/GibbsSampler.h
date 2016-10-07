// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _GIBBS_SAMPLER_H
#define _GIBBS_SAMPLER_H

/*! \file GibbsSampler.h
\brief Declaration of GibbsSampler base class
*/

#include "ExternalField.h"

// May automatically convert list to scalar
//NOTE:MAY NOT WORK
#include <hoomd/extern/pybind/include/pybind11/stl.h>

/*
Oustanding issues;
The checkoverlaps function requires some type information, but I'm not sure how to pass that in since it's entirely possible that I'll be passing in vec3 of positions that doesn't contain types
NEed to be very careful everywhere as to whether I"m correctly passing references vs actual objects, or I'll get lots of compile errors
//NOTE: Will have to add logic to grow or shrink AABB trees; I don't know how likely it is that people will change the concentration, if we think it'll be infrequent then maybe we just remake it every time
//NOTE: Ask Jens whether my use of references in the checkOverlaps function signature is correct
//NOTE: I'm not sure why it's necessary to always use this->m_pdata and this->m_exec_conf. Worth checking with Jens.
//NOTE: Allow dumping out the information of how many particles of each type were actually inserted
//NOTE: Maybe we shouldn't make it an error if the overlap settings are off, that will allow things like what Jens was doing before
 */
namespace hpmc
{
template< class Shape > class GibbsSampler : public ExternalFieldMono<Shape>
    {
    public:
         /**
          * Constructor for class
          * @param sysdef       The HOOMD system definition for that we are associating this compute with
          * @param mc           The integrator instance we are associating this compute with
          * @param type_indices An array of the type indices that we want to include in the Gibbs Sampler. Each one corresponds to a separate species in the sampler
          * @param type_counts  The desired number of each type to be sampled each time
          */
        GibbsSampler(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<IntegratorHPMCMono<Shape>> mc, std::vector<unsigned int> type_indices, std::vector<unsigned int> type_counts) : ExternalFieldMono<Shape>(sysdef), m_mc(mc), m_type_indices(type_indices), m_type_counts(type_counts), m_num_species(type_counts.size())
            {
            if (m_type_indices.size() != m_type_counts.size())
                {
                    this->m_exec_conf->msg->error() << "You must provide desired quantities for every type you want included in the Gibbs Sampler" << std::endl;
                }
            // Using the Gibbs Sampler for a type that overlap checks itself makes no sense, so check that first
            ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
            const Index2D& indexer = m_mc->getOverlapIndexer();
            unsigned int type_i, overlap_index;
            for(unsigned int i = 0; i < m_num_species; i++)
                {
                type_i = m_type_indices[i];
                overlap_index = indexer(type_i, type_i);
                if (h_overlaps.data[overlap_index] != 0) // Assuming 0 means overlap checks are not occurring
                    {
                    this->m_exec_conf->msg->error() << "A type was specified for the Gibbs Sampler, but the interaction matrix currently specifies that overlap checks should be performed between two particles of this type. This is likely an error" << std::endl; //NOTE: Would be useful to add in the actual type index to the warning
                    }
                }

            // Have to ensure that the vector of configurations and the aabb trees are identically indexed
            m_positions = std::vector<std::vector<vec3<Scalar>>, managed_allocator<std::vector<vec3<Scalar>>>>();
            m_orientations = std::vector<std::vector<quat<Scalar>>, managed_allocator<std::vector<quat<Scalar>>>>();
            m_aabb_trees = std::vector<detail::AABBTree, managed_allocator<detail::AABBTree>>();
            m_aabbs = std::vector<detail::AABB*, managed_allocator<detail::AABB*>>();
            m_positions.clear();
            m_orientations.clear();
            m_aabb_trees.clear();
            m_aabbs.clear();
            detail::AABB* temp_aabb;
            for(unsigned int i = 0; i < m_type_counts.size(); i++)
                {
                //std::vector<vec3<Scalar>> test = std::vector<vec3<Scalar>>();
                //m_positions.push_back(test);
                m_positions.push_back(std::vector<vec3<Scalar>>());
                //test = new std::vector<quat<Scalar>>();
                m_orientations.push_back(std::vector<quat<Scalar>>());
                m_aabb_trees.push_back(detail::AABBTree());

                int retval = posix_memalign((void**)&temp_aabb, 32, m_type_counts[i]*sizeof(detail::AABB));
                //NOTE: Need to make sure that the above (void***)&(*m_aabbs) makes sense; it used to be (void**)&m_aabbs, but now my m_aabbs is a pointer to a pointer since I need one aabbs for each type

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
            }

        /**
        * Destructor method. Simply frees the space used by the various class level constructs
        */
        ~GibbsSampler()
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

        /**
         * Determines whether the move of the particle at index is accepted or rejected
         * @param  index        The index in the ParticleData arrays
         * @param  position_old The old position of the particle
         * @param  shape_old    The old shape of the particle
         * @param  position_new The new position of the particle
         * @param  shape_new    The new shape of the particle
         * @return              A boolean value indicated whether or not the move was accepted
         */
        Scalar accept(const unsigned int& index, const vec3<Scalar>& position_old, const Shape& shape_old, const vec3<Scalar>& position_new, const Shape& shape_new)
            {
            return boltzmann(index, position_old, shape_old, position_new, shape_new) == 1;
            }

        /**
         * Computes the boltzmann factor. Since we are only dealing with hard particles, the boltzmann factor is solely determined by whether or not there are overlaps
         * @param  index        The index in the ParticleData arrays
         * @param  position_old The old position of the particle
         * @param  shape_old    The old shape of the particle
         * @param  position_new The new position of the particle
         * @param  shape_new    The new shape of the particle
         * @return              The boltzmann factor associated with the two states. Is either 0 (new state is higher energy) or 1 (new state is lower energy)
         */
        Scalar boltzmann(const unsigned int& index, const vec3<Scalar>& position_old, const Shape& shape_old, const vec3<Scalar>& position_new, const Shape& shape_new)
            {
            // We'll need to know the type of this particle
            ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            Scalar4 postype_i = h_postype.data[index];
            unsigned int typ_i = __scalar_as_int(postype_i.w);

            // Some variables we'll need later
            unsigned int typ_j;
            bool overlap = false;

            // Use the overlap indexer to map the (i, j) type pairs to a linear index in the overlaps array
            ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
            const Index2D& indexer = m_mc->getOverlapIndexer();
            unsigned int overlap_index = indexer(typ_i, typ_i);

            /*
OOK this is actually not quite so simple. For the iteration over the main set of particles, each particle could be of a different type, so I can't just do a quick overlap check here. But that also means that I need to give it a way to figure out what type each particle is once I pull it out of the AABB tree. For the iteration over the Gibbs particles, though, I have a separate aabb tree for each one.
There is also an efficiency loss if checkOverlaps checks every time whether or not there are overlaps allowed if i already know ahead of time. So maybe I can add a boolean flag to improve efficiency. That would save some time on check
Another related issue is the fact that I also have to call checkOverlaps from the generateRandomConfiguration function, and in that case the input particle is not indexed to the main array, so I can't pull information that way
Simplest solution might just be to provide an array of types corresponding to the positions array
             */
            // First check whether the particle overlaps with other particles
            const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();
            overlap = checkOverlaps(typ_i, position_new, shape_new, aabb_tree);
            if (overlap) { return 0;}

            // Now loop over the different Gibbs Sampler particles and test those for overlaps
            for (unsigned int j = 0; j < m_num_species; j++)
                {
                typ_j = m_type_indices[j];
                overlap_index = indexer(typ_i, typ_j);
                if (h_overlaps.data[overlap_index]) // Can skip this for pairs that can overlap
                    {
                    overlap = checkOverlaps(typ_i, position_new, shape_new, m_aabb_trees[j], j);
                    if (overlap) { return 0;}
                    }
                }
            // We won't get this far unless all overlap checks were successful
            return 1;
            }

        /**
        * Required compute function
        * @param timestep The current timestep in the simulation
        */
        void compute(unsigned int timestep)
            {
            const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type>> & params = m_mc->getParams();
            for (unsigned int i = 0; i < m_num_species; i++)
                {
                // For each element, we can just generate a new configurationkkk
                std::pair<std::vector<vec3<Scalar>>, std::vector<quat<Scalar>>> new_configuration(generateRandomConfiguration(i, timestep));
                m_positions[i] = new_configuration.first;
                m_orientations[i] = new_configuration.second;

                // We have to reconstruct AABBs for each particle and save them so we can pass a full list of AABBs to the AABBTree construction
                for(unsigned int j = 0; j < m_positions[i].size(); j++)
                    {
                        //NOTE: I think the second argument is pulling the parameters for the jth type, which works out here
                        Shape shape_j(m_orientations[i][j], params[m_type_indices[j]]);
                        m_aabbs[i][j] = shape_j.getAABB(vec3<Scalar>(0,0,0)); //NOTE: I'm not sure what the argument is. It seems like where we want to center the box?
                    } // End loop over particles

                // Now construct an AABB tree for each of them
                m_aabb_trees[i].buildTree(m_aabbs[i], m_type_counts[i]);
                } // End loop over types
            }

    protected:
         /**
          * Function to check whether a particle overlaps with any particle in a provided AABBTree
          * @param  postype_i      [description]
          * @param  shape_i    [description]
          * @param  aabb_tree  [description]
          * @param  image_list [description]
          * @return            [description]
          */
        bool checkOverlaps(unsigned int typ_i, const vec3<Scalar>& position_i, const Shape& shape_i, detail::AABBTree aabb_tree, int typ_index = -1)
        //NOTE: Currently my solution for the type issue is pretty hackish. I can try and come up with a better interface in a bit, but let's get this working first
            {
            //NOTE:It may be more efficient to pass the image list as an arg. I'm not sure how intelligent updateImageList is

            // Get positions array. Note that we only use this to get the type of the current particle, the position is passed in
            //NOTE: Can we also take the position? Will the position still be the old one? Given that it's passed into accept() and boltzmann() I'm inclined to just use that value to be safe
            //NOTE: I would like to declare typ_j here and initialize it just once if we are only working with one type of particle, but since that's in an if statement it causes g++ to throw a warning because in principle I could be using it uninitialized. So for now I'm doing it in multiple places. This entire methodology to avoid duplicating countOverlaps code is a hack anyway... so for now I'll let this stand

            // This is the aabb tree of the current particle
            detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

            const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type>> & params = m_mc->getParams();
            ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
            const Index2D& indexer = m_mc->getOverlapIndexer();
            bool overlap = false;
            unsigned int err_count = 0;

            // All image boxes (including the primary)
            std::vector<vec3<Scalar>> image_list = this->m_mc->updateImageList();
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

                                vec3<Scalar> position_j;
                                quat<Scalar> orientation_j;
                                unsigned int typ_j;
                                // load the position and orientation of the j particle
                                if(typ_index == -1)
                                    {
                                    ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
                                    ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
                                    Scalar4 postype_j = h_postype.data[j];
                                    //NOTE: Are these typecasts OK? In particular, does casting a Scalar4 to a vec3<Scalar> work as expected? I assume so since I've seen it elsewhere, but we'll have to see...
                                    position_j = vec3<Scalar>(postype_j);
                                    orientation_j = quat<Scalar>(h_orientation.data[j]);
                                    typ_j = __scalar_as_int(postype_j.w);
                                    }
                                else
                                    {
                                    position_j = m_positions[typ_index][j];
                                    orientation_j = m_orientations[typ_index][j];
                                    typ_j = m_type_indices[typ_index];
                                    }
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
            if (overlap)
                {
                    return true;
                }
            else
                {
                    return false;
                }
            }

         /**
          * Helper function to generate a random configuration of particles. Used to place solvent
          * @param type_index_i The index in m_type_indices of the current type we are generating particles for. Used to determine counts and check overlaps
          * @param timestep   The current simulation timestep
          */
        std::pair<std::vector<vec3<Scalar>>, std::vector<quat<Scalar>>> generateRandomConfiguration(unsigned int type_index_i, unsigned int timestep)
            {
            //NOTE: Still have to account for the number of ranks for MPI
            //#ifdef ENABLE_MPI
            //num_particles /= m_mc.m_exec_conf->getNRanks();
            //#endif

            // access parameters and interaction matrix
            const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type>> & params = m_mc->getParams();
            const BoxDim& box = this->m_pdata->getBox();

            // Interaction matrix
            ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
            const Index2D& indexer = m_mc->getOverlapIndexer();
            unsigned int type_j, overlap_index;
            unsigned int type_i = m_type_indices[type_index_i];

            // update AABB tree of the integrator
            const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

            // update the image list
            std::vector<vec3<Scalar>> image_list = this->m_mc->updateImageList();

            // We need to add particles to save each time we find a valid one
            std::vector<vec3<Scalar>> positions_new = std::vector<vec3<Scalar>>();
            std::vector<quat<Scalar>> orientations_new = std::vector<quat<Scalar>>();
            bool overlap;

            // This is looping over the number of particles of type m_type_indices[type_index_i]
            for (unsigned int i = 0; i < m_type_counts[type_index_i]; i++)
                {
                overlap = false;
                // select a random particle coordinate in the box
                //NOTE: We could request a random seed for this, but we're currently not.
                //Saru rng_i(i, m_seed + m_exec_conf->getRank(), timestep);
                Saru rng_i(i, this->m_exec_conf->getRank(), timestep);

                Scalar xrand = rng_i.f();
                Scalar yrand = rng_i.f();
                Scalar zrand = rng_i.f();

                Scalar3 f = make_scalar3(xrand, yrand, zrand);
                //NOTE: This is not a postype... this is just a pos
                vec3<Scalar> pos_i = vec3<Scalar>(box.makeCoordinates(f));

                //NOTE: MAKE SURE I"M referencing the righttype
                Shape shape_i(quat<Scalar>(), params[type_i]);
                // This should never have an orientation, right? They're ideal gas particles. So I think I can just ignore this altogether
                if (shape_i.hasOrientation())
                    {
                    shape_i.orientation = generateRandomOrientation(rng_i);
                    }

                // access particle data and system box
                ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
                ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
                //NOTE: The first arg is the type of this a
                overlap = checkOverlaps(type_i, pos_i, shape_i, aabb_tree);

                // If the main particles overlapped then no need to continue
                if(overlap) { continue;}

                // Now loop over the other Gibbs types and check those as well
                for (unsigned int j = 0; j < m_num_species; j++)
                    {
                    type_j = m_type_indices[j];
                    overlap_index = indexer(type_i, type_j);
                    if (h_overlaps.data[overlap_index]) // Can skip this for pairs that can overlap
                        {
                        overlap = checkOverlaps(type_i, pos_i, shape_i, m_aabb_trees[j], j);
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

            // If there were overlaps at any point, return an empty configuration. Otherwise return the new configuration
            //NOTE: This is a pretty hackish way to do this, but not sure what the preferred method would be
            std::pair<std::vector<vec3<Scalar>>, std::vector<quat<Scalar>>> return_val(positions_new, orientations_new);
            return return_val;
            }

    private:
        //NOTE: Do I need to include typename in all of these templates?
        std::vector<std::vector<vec3<Scalar>>, managed_allocator<std::vector<vec3<Scalar>>>> m_positions;
        std::vector<std::vector<quat<Scalar>>, managed_allocator<std::vector<quat<Scalar>>>> m_orientations;
        std::vector<detail::AABBTree, managed_allocator<detail::AABBTree>> m_aabb_trees;
        std::vector<detail::AABB*, managed_allocator<detail::AABB*>> m_aabbs;

        std::shared_ptr<IntegratorHPMCMono<Shape>> m_mc;
        std::vector<unsigned int> m_type_indices;
        std::vector<unsigned int> m_type_counts;
        unsigned int m_num_species;
    };

    template<class Shape>
    void export_GibbsSampler(pybind11::module& m, std::string name)
        {
       pybind11::class_<GibbsSampler<Shape>, std::shared_ptr< GibbsSampler<Shape> > >(m, name.c_str(), pybind11::base< ExternalFieldMono<Shape> >())
        .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<IntegratorHPMCMono<Shape>>, std::vector<unsigned int>, std::vector<unsigned int>>())
        ;
        }

}

#endif // _GIBBS_SAMPLER_H
