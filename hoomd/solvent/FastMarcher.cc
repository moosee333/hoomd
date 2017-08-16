// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: vramasub

#include "FastMarcher.h"

using namespace std;
namespace py = pybind11;

/*! \file FastMarcher.cc
    \brief Contains code for the FastMarcher class
*/

namespace solvent
{

//! Constructor
FastMarcher::FastMarcher(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<SparseFieldUpdater> field)
    : 
    m_sysdef(sysdef), 
    m_pdata(sysdef->getParticleData()), 
    m_exec_conf(m_pdata->getExecConf()),
    m_field(field),
    m_grid(field->getGrid())
    { }

//! Destructor
FastMarcher::~FastMarcher()
    { }

void FastMarcher::estimateL0Distances()
    { 
    // Access grid data
    ArrayHandle<Scalar> h_fn(m_grid->getVelocityGrid(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);
    Scalar3 grid_spacing = m_grid->getSpacing();

    // Access field data
    const std::vector<std::vector<uint3> > layers = m_field->getLayers();
    const std::map<char, char> layer_indexer = m_field->getIndex();

    // Use periodic flags
    const BoxDim& box = this->m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();

    // Loop over all cells, and compute neighbors
    uint3 dims = m_grid->getDimensions();
    Index3D indexer = this->m_grid->getIndexer();

    // Interpolate distances on Lz
    const std::vector<uint3> Lz = layers[layer_indexer.find(0)->second];
    for (std::vector<uint3>::const_iterator element = Lz.begin(); element != Lz.end(); element++)
        {
        unsigned int x = element->x;
        unsigned int y = element->y;
        unsigned int z = element->z;
        unsigned int cur_cell = indexer(x, y, z);

        // The distance estimate, initialized to infinity
        Scalar min_distance = std::numeric_limits<Scalar>::infinity();

        // Currently using a somewhat convoluted construct to ensure that when
        // considering a particular neighbor we also know which grid spacing
        // to use. For this purpose x = 0, y = 1, z = 2
        std::vector<vec3<Scalar> > directions;
        directions.push_back(vec3<Scalar>(1, 0, 0));
        directions.push_back(vec3<Scalar>(0, 1, 0));
        directions.push_back(vec3<Scalar>(0, 0, 1));

        Scalar shifts[2] = {1, -1};

        for (unsigned int i = 0; i < directions.size(); i++)
            { 
            Scalar step;
            switch (i) 
                {
                case 0:
                    step = grid_spacing.x;
                    break;
                case 1:
                    step = grid_spacing.y;
                    break;
                case 2:
                    step = grid_spacing.z;
                    break;
                default:
                    assert(false);
                }
            vec3<Scalar> direction = directions[i];

            for (unsigned int j = 0; j < sizeof(shifts)/sizeof(Scalar); j++)
                {
                vec3<Scalar> total_shift = direction*shifts[j];
                int neighbor_x = x+total_shift.x;
                int neighbor_y = y+total_shift.y;
                int neighbor_z = z+total_shift.z;

                if (periodic.x && neighbor_x < 0)
                    neighbor_x += dims.x;
                if (periodic.y && neighbor_y < 0)
                    neighbor_y += dims.y;
                if (periodic.z && neighbor_z < 0)
                    neighbor_z += dims.z;
                if (periodic.x && neighbor_x >= (int) dims.x)
                    neighbor_x -= dims.x;
                if (periodic.y && neighbor_y >= (int) dims.y)
                    neighbor_y -= dims.y;
                if (periodic.z && neighbor_z >= (int) dims.z)
                    neighbor_z -= dims.z;

                // Distance calculation is performed by taking the change in energy between cells,
                // then finding where along that line the zero energy point lies, and then multiplying
                // that fraction into the grid spacing
                unsigned int neighbor_cell = indexer(neighbor_x, neighbor_y, neighbor_z);

                // If there is no difference between this cell and its neighbor, there cannot be a 
                // crossing, so we can skip it. This also accounts for the special case where the two
                // values are precisely the same, so the total_energy_difference is 0
                if (sgn(h_fn.data[neighbor_cell]) == sgn(h_fn.data[cur_cell]))
                    continue;
                Scalar total_energy_difference = h_fn.data[neighbor_cell] - h_fn.data[cur_cell];
                Scalar boundary_distance = sgn(total_energy_difference) * step * h_fn.data[cur_cell] / total_energy_difference; // multiply by the sign so that cur_cell determines total sign
                if (std::abs(boundary_distance) < std::abs(min_distance))
                    min_distance = boundary_distance;
                }
            } // End loops over neighbors

        h_phi.data[cur_cell] = min_distance;
        }
    }

void FastMarcher::march()
    { 
    estimateL0Distances();
    }

} // end namespace solvent
