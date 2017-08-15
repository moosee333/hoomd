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
    // Initialize the layers
    m_layers = std::vector<std::vector<uint3> >(); //!< Sparse field layers
    for (unsigned char i = 0; i < (2*m_num_layers+1); i++)
        m_layers.push_back(std::vector<uint3>());
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
        std::vector<uint3> tmp = m_layers[get_layer_index(i)];
        printf("The number of items in layer %d is %zu\n", i, tmp.size());
        printf("The number of unique items in layer %d is %zu\n", i, std::distance(tmp.begin(), std::unique(tmp.begin(), tmp.end())));
    }
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);
    Index3D indexer = this->m_grid->getIndexer();
    this->m_grid->setGridValues(GridData::energies, -10);

    for (int j = -m_num_layers; j <= m_num_layers; j++)
        {
        std::vector<uint3> L = m_layers[get_layer_index(j)];
        unsigned int len_L = L.size();
        for (unsigned int i = 0; i < len_L; i++)
            {
            unsigned int x = L[i].x;
            unsigned int y = L[i].y;
            unsigned int z = L[i].z;
            h_phi.data[indexer(x, y, z)] = j;
            }
        }*/
    }

void SparseFieldUpdater::initializeLz()
    {
    // Access grid data
    ArrayHandle<Scalar> h_fn(m_grid->getVelocityGrid(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);

    // Use periodic flags
    const BoxDim& box = this->m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();

    // Loop over all cells, and compute neighbors
    uint3 dims = m_grid->getDimensions();
    Index3D indexer = this->m_grid->getIndexer();

    unsigned int zero_index = get_layer_index(0);

    for(unsigned int i = 0; i < dims.x; i++)
        {
        for(unsigned int j = 0; j < dims.y; j++)
            {
            for(unsigned int k = 0; k < dims.z; k++)
                {
                unsigned int cur_cell = indexer(i, j, k);
                int cur_sign = sgn(h_fn.data[cur_cell]);

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
                        int x = neighbor_idx.x;
                        int y = neighbor_idx.y;
                        int z = neighbor_idx.z;

                        if (periodic.x && x < 0)
                            x += dims.x;
                        if (periodic.y && neighbor_idx.y < 0)
                            y += dims.y;
                        if (periodic.z && neighbor_idx.z < 0)
                            z += dims.z;
                        if (periodic.x && x >= (int) dims.x)
                            x -= dims.x;
                        if (periodic.y && y >= (int) dims.y)
                            y -= dims.y;
                        if (periodic.z && z >= (int) dims.z)
                            z -= dims.z;

                        unsigned int neighbor_cell = indexer(x, y, z);
                        sgn(h_fn.data[neighbor_cell]);
                        if(cur_sign != sgn(h_fn.data[neighbor_cell]))
                            {
                            m_layers[zero_index].push_back(make_uint3(i, j, k));
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

    // Use periodic flags
    const BoxDim& box = this->m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();

    uint3 dims = m_grid->getDimensions();
    Index3D indexer = this->m_grid->getIndexer();

    int layer = 1;
    unsigned int pos_layer_index = get_layer_index(layer);
    unsigned int neg_layer_index = get_layer_index(-layer);
    int prev_layer = 0;
    unsigned int prev_layer_index = get_layer_index(prev_layer);

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
            int x = neighbor_idx.x;
            int y = neighbor_idx.y;
            int z = neighbor_idx.z;

            if (periodic.x && x < 0)
                x += dims.x;
            if (periodic.y && neighbor_idx.y < 0)
                y += dims.y;
            if (periodic.z && neighbor_idx.z < 0)
                z += dims.z;
            if (periodic.x && x >= (int) dims.x)
                x -= dims.x;
            if (periodic.y && y >= (int) dims.y)
                y -= dims.y;
            if (periodic.z && z >= (int) dims.z)
                z -= dims.z;
            unsigned int neighbor_cell = indexer(x, y, z);

            // Make sure that this cell is not already in L0, Lp1, or Ln1
            // NOTE: Is there a way to make this efficient? Possibly better data structure than list?
            // May need to use some combination of vector and hashmap to get good add and removal speed
            auto test_pos = std::find(std::begin(m_layers[pos_layer_index]), std::end(m_layers[pos_layer_index]), make_uint3(x, y, z));
            if (test_pos != std::end(m_layers[pos_layer_index]))
                continue;

            auto test_neg = std::find(std::begin(m_layers[neg_layer_index]), std::end(m_layers[neg_layer_index]), make_uint3(x, y, z));
            if (test_neg != std::end(m_layers[neg_layer_index]))
                continue;

            auto test_prev = std::find(std::begin(m_layers[prev_layer_index]), std::end(m_layers[prev_layer_index]), make_uint3(x, y, z));
            if (test_prev != std::end(m_layers[prev_layer_index]))
                continue;

            if (h_fn.data[neighbor_cell] > 0)
                m_layers[neg_layer_index].push_back(make_uint3(x, y, z));
            else
                m_layers[pos_layer_index].push_back(make_uint3(x, y, z));
            } // End loop over neighbors
        } // End m_layers[prev_layer_index] loop
    }

void SparseFieldUpdater::initializeLayer(int layer)
    {
    assert(layer != 0);

    // Access grid data
    ArrayHandle<Scalar> h_fn(m_grid->getVelocityGrid(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);

    // Use periodic flags
    const BoxDim& box = this->m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();

    uint3 dims = m_grid->getDimensions();

    unsigned int cur_layer_index = get_layer_index(layer);
    int prev_layer = layer > 0 ? layer - 1 : layer + 1;
    unsigned int prev_layer_index = get_layer_index(prev_layer);
    // Need the layer before to ensure that we don't add e.g. L0 items to L2 when testing the neighbors of L1
    int second_layer = layer > 0 ? layer - 2 : layer + 2;
    unsigned int second_layer_index = get_layer_index(second_layer);

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
            int x = neighbor_idx.x;
            int y = neighbor_idx.y;
            int z = neighbor_idx.z;

            if (periodic.x && x < 0)
                x += dims.x;
            if (periodic.y && neighbor_idx.y < 0)
                y += dims.y;
            if (periodic.z && neighbor_idx.z < 0)
                z += dims.z;
            if (periodic.x && x >= (int) dims.x)
                x -= dims.x;
            if (periodic.y && y >= (int) dims.y)
                y -= dims.y;
            if (periodic.z && z >= (int) dims.z)
                z -= dims.z;

            // Make sure that this cell is not in any of the previous layers. Have to test the "previous" two layers
            // NOTE: Is there a way to make this efficient? Possibly better data structure than list?
            // May need to use some combination of vector and hashmap to get good add and removal speed
            auto test_cur = std::find(std::begin(m_layers[cur_layer_index]), std::end(m_layers[cur_layer_index]), make_uint3(x, y, z));
            if (test_cur != std::end(m_layers[cur_layer_index]))
                continue;

            auto test_prev = std::find(std::begin(m_layers[prev_layer_index]), std::end(m_layers[prev_layer_index]), make_uint3(x, y, z));
            if (test_prev != std::end(m_layers[prev_layer_index]))
                continue;

            auto test_second = std::find(std::begin(m_layers[second_layer_index]), std::end(m_layers[second_layer_index]), make_uint3(x, y, z));
            if (test_second != std::end(m_layers[second_layer_index]))
                continue;

            m_layers[cur_layer_index].push_back(make_uint3(x, y, z));
            } // End loop over neighbors
        } // End m_layers[prev_layer_index] loop
    }

void export_SparseFieldUpdater(py::module& m)
    {
    pybind11::class_<SparseFieldUpdater, std::shared_ptr<SparseFieldUpdater> >(m, "SparseFieldUpdater")
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<GridData>, Scalar>())
    ;
    }

} // end namespace solvent
