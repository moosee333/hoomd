// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: vramasub

#include "SparseFieldUpdater.h"

using namespace std;
namespace py = pybind11;

/*! \file SparseFieldUpdater.cc
    \brief Contains code for the SparseFieldUpdater class
*/

namespace solvent
{

//! Constructor
SparseFieldUpdater::SparseFieldUpdater(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<GridData> grid, char num_layers)
    : m_sysdef(sysdef),
      m_pdata(sysdef->getParticleData()),
      m_exec_conf(m_pdata->getExecConf()),
      m_grid(grid),
      m_num_layers(num_layers)
    {
    // Initialize the layers and the index
    m_layers = std::vector<std::vector<uint3> >();
    for (char i = -m_num_layers; i <= m_num_layers; i++)
        {
        m_layers.push_back(std::vector<uint3>());
        m_index[i] = i + (char) m_num_layers;
        }
    }

//! Destructor
SparseFieldUpdater::~SparseFieldUpdater()
    {
    for (std::vector<std::vector<uint3> >::iterator layer = m_layers.begin(); layer != m_layers.end(); layer++)
        layer->clear();
    }

void SparseFieldUpdater::computeInitialField()
    {
    initializeLz();
    initializeL1();

    // Can't be unsigned because we need to negate
    for (char i = 2; i <= (char) m_num_layers; i++)
        {
        initializeLayer(i);
        initializeLayer(-i);
        }

    /*//DEBUGGING CODE
    for (int i = -m_num_layers; i <= m_num_layers; i++)
    {
        std::vector<uint3> tmp = m_layers[m_index[i]];
        printf("The number of items in layer %d is %zu\n", i, tmp.size());
        printf("The number of unique items in layer %d is %zu\n", i, std::distance(tmp.begin(), std::unique(tmp.begin(), tmp.end())));
    }
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);
    Index3D indexer = this->m_grid->getIndexer();
    this->m_grid->setGridValues(GridData::DISTANCES, -10);

    for (int j = -m_num_layers; j <= m_num_layers; j++)
        {
        std::vector<uint3> L = m_layers[m_index[j]];
        unsigned int len_L = L.size();
        for (unsigned int i = 0; i < len_L; i++)
            {
            unsigned int x = L[i].x;
            unsigned int y = L[i].y;
            unsigned int z = L[i].z;
            h_phi.data[indexer(x, y, z)] = j;
            }
        }
    //END DEBUGGING CODE*/
    }

void SparseFieldUpdater::initializeLz()
    {
    // Access grid data
    ArrayHandle<Scalar> h_fn(m_grid->getVelocityGrid(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);
    uint3 dims = m_grid->getDimensions();
    Index3D indexer = this->m_grid->getIndexer();

    unsigned int zero_index = m_index[0];

    // Loop over all cells, and compute neighbors
    for(unsigned int i = 0; i < dims.x; i++)
        {
        for(unsigned int j = 0; j < dims.y; j++)
            {
            for(unsigned int k = 0; k < dims.z; k++)
                {
                unsigned int cur_cell = indexer(i, j, k);
                int cur_sign = sgn(h_fn.data[cur_cell]);

                // In very sparse systems where the r_cut is much smaller than
                // the box dimensions, it is possible for there to be cells that
                // appear to overlap with the vdW surface simply because no
                // potential reaches there. Since there is no definite way to
                // detect this, a notice is simply printed in the default case.
                // The ignore_zero option can be set to avoid these cells.
                if (cur_sign == 0)
                    {
                    if (!m_grid->ignoreZero())
                        {
                        m_exec_conf->msg->notice(5) << "Your system has grid cells with an exactly zero value of the energy."
                            "This is likely because your system is very sparse relative to the specified r_cut."
                            "These cells will be added to L_0. If you wish to ignore them, please set ignore_zero to True" << std::endl;
                        }
                    else
                        {
                        continue;
                        }
                    }

                int3 neighbor_indices[6] =
                    {
                    make_int3(i+1, j, k),
                    make_int3(i-1, j, k),
                    make_int3(i, j+1, k),
                    make_int3(i, j-1, k),
                    make_int3(i, j, k+1),
                    make_int3(i, j, k-1),
                    }; // The neighboring cells

                // Check all directions for changes in the sign of the energy (h_fn)
                for(unsigned int idx = 0; idx < sizeof(neighbor_indices)/sizeof(int3); idx++)
                    {
                        int3 neighbor_idx = neighbor_indices[idx];
                        uint3 periodic_neighbor = m_grid->wrap(neighbor_idx);

                        unsigned int neighbor_cell = indexer(periodic_neighbor.x, periodic_neighbor.y, periodic_neighbor.z);

                        if(cur_sign != sgn(h_fn.data[neighbor_cell]))
                            {
                            // If the sign change is 1 instead of 2, we are on a zero cell and react accordingly.
                            if (std::abs(cur_sign - sgn(h_fn.data[neighbor_cell])) == 1)
                                {
                                if (!m_grid->ignoreZero())
                                    {
                                    m_exec_conf->msg->notice(5) << "Your system has grid cells with an exactly zero value of the energy."
                                        "This is likely because your system is very sparse relative to the specified r_cut."
                                        "These cells will be added to L_0. If you wish to ignore them, please set ignore_zero to True" << std::endl;
                                    m_layers[zero_index].push_back(make_uint3(i, j, k));
                                    }
                                }
                            else
                                {
                                m_layers[zero_index].push_back(make_uint3(i, j, k));
                                }
                            // Once we've found some direction of change, we are finished with this cell
                            break;
                            }
                    } // End loop over neighbors
                }
            }
        } // End loop over grid
    }

void SparseFieldUpdater::initializeL1()
    {
    // Access grid data
    ArrayHandle<Scalar> h_fn(m_grid->getVelocityGrid(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);
    Index3D indexer = this->m_grid->getIndexer();

    int layer = 1;
    unsigned int pos_layer_index = m_index[layer];
    unsigned int neg_layer_index = m_index[-layer];
    int prev_layer = 0;
    unsigned int prev_layer_index = m_index[prev_layer];

    // Loop over previous layer
    for (std::vector<uint3>::iterator L_prev_element = m_layers[prev_layer_index].begin();
            L_prev_element != m_layers[prev_layer_index].end(); L_prev_element++)
        {
        unsigned int i = L_prev_element->x;
        unsigned int j = L_prev_element->y;
        unsigned int k = L_prev_element->z;

        int3 neighbor_indices[6] =
            {
            make_int3(i+1, j, k),
            make_int3(i-1, j, k),
            make_int3(i, j+1, k),
            make_int3(i, j-1, k),
            make_int3(i, j, k+1),
            make_int3(i, j, k-1),
            }; // The neighboring cells

        // Loop over each neighbor
        for(unsigned int idx = 0; idx < sizeof(neighbor_indices)/sizeof(int3); idx++)
            {
            int3 neighbor_idx = neighbor_indices[idx];
            uint3 periodic_neighbor = m_grid->wrap(neighbor_idx);
            unsigned int neighbor_cell = indexer(periodic_neighbor.x, periodic_neighbor.y, periodic_neighbor.z);

            // Make sure that this cell is not already in Lz, Lp1, or Ln1
            // NOTE: Is there a way to make this efficient? Possibly better data structure than list?
            // May need to use some combination of vector and hashmap to get good add and removal speed
            auto test_pos = std::find(std::begin(m_layers[pos_layer_index]), std::end(m_layers[pos_layer_index]), periodic_neighbor);
            if (test_pos != std::end(m_layers[pos_layer_index]))
                continue;

            auto test_neg = std::find(std::begin(m_layers[neg_layer_index]), std::end(m_layers[neg_layer_index]), periodic_neighbor);
            if (test_neg != std::end(m_layers[neg_layer_index]))
                continue;

            auto test_prev = std::find(std::begin(m_layers[prev_layer_index]), std::end(m_layers[prev_layer_index]), periodic_neighbor);
            if (test_prev != std::end(m_layers[prev_layer_index]))
                continue;

            if (h_fn.data[neighbor_cell] > 0)
                m_layers[neg_layer_index].push_back(periodic_neighbor);
            else
                m_layers[pos_layer_index].push_back(periodic_neighbor);
            } // End loop over neighbors
        } // End m_layers[prev_layer_index] loop
    }

void SparseFieldUpdater::initializeLayer(int layer)
    {
    assert(layer != 0);

    // Access grid data
    ArrayHandle<Scalar> h_fn(m_grid->getVelocityGrid(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);

    unsigned int cur_layer_index = m_index[layer];
    int prev_layer = layer > 0 ? layer - 1 : layer + 1;
    unsigned int prev_layer_index = m_index[prev_layer];
    // Need the layer before to ensure that we don't add e.g. Lz items to L2 when testing the neighbors of L1
    int second_layer = layer > 0 ? layer - 2 : layer + 2;
    unsigned int second_layer_index = m_index[second_layer];

    // Loop over previous layer and check all neighbors for addition
    for (std::vector<uint3>::iterator L_prev_element = m_layers[prev_layer_index].begin();
            L_prev_element != m_layers[prev_layer_index].end(); L_prev_element++)
        {
        // Loop over each neighbor
        unsigned int i = L_prev_element->x;
        unsigned int j = L_prev_element->y;
        unsigned int k = L_prev_element->z;

        int3 neighbor_indices[6] =
            {
            make_int3(i+1, j, k),
            make_int3(i-1, j, k),
            make_int3(i, j+1, k),
            make_int3(i, j-1, k),
            make_int3(i, j, k+1),
            make_int3(i, j, k-1),
            }; // The neighboring cells

        // Check all directions for changes in the sign of the energy (h_fn)
        for(unsigned int idx = 0; idx < sizeof(neighbor_indices)/sizeof(int3); idx++)
            {
            int3 neighbor_idx = neighbor_indices[idx];
            uint3 periodic_neighbor = m_grid->wrap(neighbor_idx);

            // Make sure that this cell is not in any of the previous layers. Have to test the "previous" two layers
            // NOTE: Is there a way to make this efficient? Possibly better data structure than list?
            // May need to use some combination of vector and hashmap to get good add and removal speed
            auto test_cur = std::find(std::begin(m_layers[cur_layer_index]), std::end(m_layers[cur_layer_index]), periodic_neighbor);
            if (test_cur != std::end(m_layers[cur_layer_index]))
                continue;

            auto test_prev = std::find(std::begin(m_layers[prev_layer_index]), std::end(m_layers[prev_layer_index]), periodic_neighbor);
            if (test_prev != std::end(m_layers[prev_layer_index]))
                continue;

            auto test_second = std::find(std::begin(m_layers[second_layer_index]), std::end(m_layers[second_layer_index]), periodic_neighbor);
            if (test_second != std::end(m_layers[second_layer_index]))
                continue;

            m_layers[cur_layer_index].push_back(periodic_neighbor);
            } // End loop over neighbors
        } // End m_layers[prev_layer_index] loop
    }

void SparseFieldUpdater::updateField()
    {
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);
    Index3D indexer = m_grid->getIndexer();
    Scalar3 spacing = m_grid->getSpacing();

    //NOTE: ARE THERE ANY OPTIMIZATIONS TO BE HAD HERE TO PREVENT REUPDATING THE FIELD IF NOTHING HAS CHANGED?
    // Initialize the status lists such that we can use identical indexing to the layers
    std::vector<std::vector<uint3> > status_lists;
    for (unsigned int i = 0; i < m_layers.size(); i++)
        status_lists.push_back(std::vector<uint3>());

    // Build up the status lists by looping layer-by-layer
    for (int i = 0; i <= m_num_layers; i++)
        {
        for (int sgn = 1; sgn != -3; sgn-=2)
            { //NOTE:SHOULD WRITE A MORE ELEGANT LOOP, BUT THIS WILL WORK FOR NOW
            // Don't do 0 twice
            if (i == 0 && sgn == -1)
                continue;

            int list_index = i*sgn; 
            unsigned int layer_index = m_index.find(list_index)->second, 
                         increase_index = m_index.find(list_index+1)->second, 
                         decrease_index = m_index.find(list_index-1)->second;

            //NOTE: For non-orthorhombic boxes does this bound work?
            //I COULD IMPLEMENT A MORE COMPLEX GEOMETRIC TEST, BUT NOT SURE IF THAT'S WORTHWHILE (CERTAINLY NOT YET)
            Scalar distance_bound = sqrt(spacing.x*spacing.x + spacing.y*spacing.y + spacing.z*spacing.z);
            // //NOTE: THE BOUNDS ARE WRONG FOR NEGATIVE LAYERS (WRONG ORDER)
            throw std::runtime_error("Fix bounds to work for negative layers!");
            Scalar upper_bound = (i+0.5)*distance_bound;
            Scalar lower_bound = (i-0.5)*distance_bound;

            // Since deleting items one at a time from vectors is relatively inefficient, we apply a modification of
            // the erase-remove idiom (with an intermediate loop to move the additional items)
            std::vector<uint3>::iterator removal_iterator = std::remove_if(m_layers[layer_index].begin(), m_layers[layer_index].end(),
                    [&](uint3 tmp) {
                        return (h_phi.data[indexer(tmp.x, tmp.y, tmp.z)] >= upper_bound || h_phi.data[indexer(tmp.x, tmp.y, tmp.z)] < lower_bound);
                    });
            for (std::vector<uint3>::iterator move_iterator = removal_iterator; move_iterator != m_layers[layer_index].end(); move_iterator++)
                {
                // Check which sign condition was violated
                if (h_phi.data[indexer(move_iterator->x, move_iterator->y, move_iterator->z)] >= upper_bound)
                    status_lists[increase_index].push_back(*move_iterator);
                else if (h_phi.data[indexer(move_iterator->x, move_iterator->y, move_iterator->z)] < lower_bound)
                    status_lists[decrease_index].push_back(*move_iterator);
                else
                    throw std::runtime_error("Trying to add an element to the status list that shoulnd't move");
                }
            m_layers[layer_index].erase(removal_iterator, m_layers[layer_index].end());
            }
        }

    // Now update the layers from the status lists
    for (int i = 0; i <= m_num_layers; i++)
        {
        for (int sgn = 1; sgn != -3; sgn -= 2)
            { //NOTE:SHOULD WRITE A MORE ELEGANT LOOP, BUT THIS WILL WORK FOR NOW
            if (i == 0 && sgn == -1)
                continue;

            int list_index = i*sgn; 
            unsigned int layer_index = m_index.find(list_index)->second;

            for (std::vector<uint3>::const_iterator element = status_lists[layer_index].begin(); element != status_lists[layer_index].end(); element++)
                {
                m_layers[layer_index].push_back(*element);
                }

            // For the final layer, we may have to add cells that were not part of the original sparse field
            if (i == m_num_layers - 1)
                {
                for (std::vector<uint3>::const_iterator element = m_layers[layer_index].begin(); element != m_layers[layer_index].end(); element++)
                    {
                    // Any neighbor of the second to last layer (either inner or outer) that is not already a member of either the last layer or the
                    // third to last year needs to be added to the last layer now
                    unsigned int x = element->x, y = element->y, z = element->z;
                    int3 neighbor_indices[6] =
                        {
                        make_int3(x+1, y, z),
                        make_int3(x-1, y, z),
                        make_int3(x, y+1, z),
                        make_int3(x, y-1, z),
                        make_int3(x, y, z+1),
                        make_int3(x, y, z-1),
                        }; // The neighboring cells

                    for(unsigned int idx = 0; idx < sizeof(neighbor_indices)/sizeof(int3); idx++)
                        {
                        int3 neighbor_idx = neighbor_indices[idx];
                        uint3 periodic_neighbor = m_grid->wrap(neighbor_idx);

                        //NOTE: Make sure the cast works when multiplying by sgn
                        int inner_test_list_index = (int) (m_num_layers - 2) * (sgn), outer_test_list_index = (int) (m_num_layers) * (sgn);
                        unsigned int inner_test_layer_index = m_index.find(inner_test_list_index)->second, outer_test_layer_index = m_index.find(outer_test_list_index)->second;

                        if (std::find(m_layers[inner_test_layer_index].begin(), m_layers[inner_test_layer_index].end(), periodic_neighbor) == m_layers[inner_test_layer_index].end()
                                &&
                            std::find(m_layers[outer_test_layer_index].begin(), m_layers[outer_test_layer_index].end(), periodic_neighbor) == m_layers[outer_test_layer_index].end()
                                )
                            {
                            m_layers[outer_test_layer_index].push_back(periodic_neighbor);
                            }
                        }
                    }
                }
            }
        }
    }
} // end namespace solvent
