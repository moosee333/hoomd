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

void FastMarcher::march()
    {
    // The "exact" Lz values are required before anything else can be done.
    estimateLzDistances();

    // Access grid data
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);
    uint3 dims = m_grid->getDimensions();

    // Access field data
    const std::vector<std::vector<uint3> > layers = m_field->getLayers();
    const std::map<char, char> layer_indexer = m_field->getIndex();
    Index3D indexer = this->m_grid->getIndexer();

    // Use periodic flags
    const BoxDim& box = this->m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();

    /**************************
     * March outwards
     *************************/
    // Initialize tentative distances in the outward direction
            //NOTE: NEED TO TEST MY COMPARATORS
            //NOTE: may be worth implementing a better PQ that allows updating, for now I'm doing the clunky thing again
    std::priority_queue< std::pair<uint3, Scalar>, std::vector< std::pair<uint3, Scalar> >, CompareScalarGreater > outer_tentative_distances;
    std::map<uint3, Scalar, CompareInt> current_outer_tentative_distances;
    const std::vector<uint3> Lp1 = layers[layer_indexer.find(1)->second];
    for (std::vector<uint3>::const_iterator element = Lp1.begin(); element != Lp1.end(); element++)
        {
            //NOTE: need to decide whether to just march positive and negative separately, or to instead
            //use absolute values
        Scalar d = calculateTentativeDistance(*element, true);
        outer_tentative_distances.push(std::pair<uint3, Scalar>(*element, d));
        current_outer_tentative_distances[*element] = d;
        }

    // Build lists of eligible points
    std::vector<uint3> eligible_positive;
    unsigned int num_eligible_positive = 0;
    for (unsigned int layer = 1; layer <= m_field->getNumLayers(); layer++)
        num_eligible_positive += layers[layer_indexer.find(layer)->second].size();
    eligible_positive.reserve(num_eligible_positive);
    for (unsigned int layer = 1; layer <= m_field->getNumLayers(); layer++)
        {
        std::vector<uint3> L = layers[layer_indexer.find(layer)->second];
        eligible_positive.insert(eligible_positive.end(), L.begin(), L.end());
        }

    // Then March
    while (!outer_tentative_distances.empty())
        {
        // This is how we ensure that we get the most updated value
        std::pair<uint3, Scalar> next = outer_tentative_distances.top();
        outer_tentative_distances.pop();
        uint3 next_point = next.first;
        Scalar next_distance = next.second;
        while (next_distance != current_outer_tentative_distances[next_point])
            {
            next = outer_tentative_distances.top();
            outer_tentative_distances.pop();
            next_point = next.first;
            next_distance = next.second;
            }

        int i = next_point.x, j = next_point.y, k = next_point.z;
        h_phi.data[indexer(i, j, k)] = next_distance;

        int3 neighbor_indices[6] =
            {
            make_int3(i+1, j, k),
            make_int3(i-1, j, k),
            make_int3(i, j+1, k),
            make_int3(i, j-1, k),
            make_int3(i, j, k+1),
            make_int3(i, j, k-1),
            }; // The neighboring cells

        for(unsigned int idx = 0; idx < sizeof(neighbor_indices)/sizeof(int3); idx++)
            {
            int3 neighbor_idx = neighbor_indices[idx];
            int x = neighbor_idx.x;
            int y = neighbor_idx.y;
            int z = neighbor_idx.z;

            if (periodic.x && x < 0)
                x += dims.x;
            if (periodic.y && y < 0)
                y += dims.y;
            if (periodic.z && z < 0)
                z += dims.z;
            if (periodic.x && x >= (int) dims.x)
                x -= dims.x;
            if (periodic.y && y >= (int) dims.y)
                y -= dims.y;
            if (periodic.z && z >= (int) dims.z)
                z -= dims.z;

            // Only bother with points in the layers
            //NOTE: Make sure that make_uint3 is ok here; x, y, z are ints, but should be effectively unsigned
            uint3 current_neighbor = make_uint3(x, y, z);
            auto existence_iterator = std::find(std::begin(eligible_positive), std::end(eligible_positive), current_neighbor);
            if (existence_iterator == std::end(eligible_positive))
                continue;
            eligible_positive.erase(existence_iterator); //NOTE: Make sure that this is doing what I expect
            Scalar d = calculateTentativeDistance(current_neighbor, true);
            outer_tentative_distances.push(std::pair<uint3, Scalar>(current_neighbor, d));
            current_outer_tentative_distances[current_neighbor] = d;
            }
        }

    /**************************
     * March inwards
     *************************/
    // Initialize tentative distances in the inward direction
            //NOTE: NEED TO TEST MY COMPARATORS
            //NOTE: may be worth implementing a better PQ that allows updating, for now I'm doing the clunky thing again
    std::priority_queue< std::pair<uint3, Scalar>, std::vector< std::pair<uint3, Scalar> >, CompareScalarLess > inner_tentative_distances;
    std::map<uint3, Scalar, CompareInt> current_inner_tentative_distances;
    const std::vector<uint3> Ln1 = layers[layer_indexer.find(-1)->second];
    for (std::vector<uint3>::const_iterator element = Ln1.begin(); element != Ln1.end(); element++)
        {
            //NOTE: need to decide whether to just march negative and negative separately, or to instead
            //use absolute values
        Scalar d = calculateTentativeDistance(*element, false);
        inner_tentative_distances.push(std::pair<uint3, Scalar>(*element, d));
        current_inner_tentative_distances[*element] = d;
        }

    // Build lists of eligible points
    std::vector<uint3> eligible_negative;
    unsigned int num_eligible_negative = 0;
    for (int layer = 1; layer <= m_field->getNumLayers(); layer++)
        num_eligible_negative += layers[layer_indexer.find(-layer)->second].size();
    eligible_negative.reserve(num_eligible_negative);
    for (int layer = 1; layer <= m_field->getNumLayers(); layer++)
        {
        std::vector<uint3> L = layers[layer_indexer.find(-layer)->second];
        eligible_negative.insert(eligible_negative.end(), L.begin(), L.end());
        }

    // Then March
    while (!inner_tentative_distances.empty())
        {
        // This is how we ensure that we get the most updated value
        std::pair<uint3, Scalar> next = inner_tentative_distances.top();
        inner_tentative_distances.pop();
        uint3 next_point = next.first;
        Scalar next_distance = next.second;
        while (next_distance != current_inner_tentative_distances[next_point])
            {
            next = inner_tentative_distances.top();
            inner_tentative_distances.pop();
            next_point = next.first;
            next_distance = next.second;
            }

        int i = next_point.x, j = next_point.y, k = next_point.z;
        h_phi.data[indexer(i, j, k)] = next_distance;

        int3 neighbor_indices[6] =
            {
            make_int3(i+1, j, k),
            make_int3(i-1, j, k),
            make_int3(i, j+1, k),
            make_int3(i, j-1, k),
            make_int3(i, j, k+1),
            make_int3(i, j, k-1),
            }; // The neighboring cells

        for(unsigned int idx = 0; idx < sizeof(neighbor_indices)/sizeof(int3); idx++)
            {
            int3 neighbor_idx = neighbor_indices[idx];
            int x = neighbor_idx.x;
            int y = neighbor_idx.y;
            int z = neighbor_idx.z;

            if (periodic.x && x < 0)
                x += dims.x;
            if (periodic.y && y < 0)
                y += dims.y;
            if (periodic.z && z < 0)
                z += dims.z;
            if (periodic.x && x >= (int) dims.x)
                x -= dims.x;
            if (periodic.y && y >= (int) dims.y)
                y -= dims.y;
            if (periodic.z && z >= (int) dims.z)
                z -= dims.z;

            // Only bother with points in the layers
            //NOTE: Make sure that make_uint3 is ok here; x, y, z are ints, but should be effectively unsigned
            uint3 current_neighbor = make_uint3(x, y, z);
            auto existence_iterator = std::find(std::begin(eligible_negative), std::end(eligible_negative), current_neighbor);
            if (existence_iterator == std::end(eligible_negative))
                continue;
            eligible_negative.erase(existence_iterator); //NOTE: Make sure that this is doing what I expect
            Scalar d = calculateTentativeDistance(current_neighbor, false);
            inner_tentative_distances.push(std::pair<uint3, Scalar>(current_neighbor, d));
            current_inner_tentative_distances[current_neighbor] = d;
            }
        }
    }

void FastMarcher::estimateLzDistances()
    {
    // Access grid data
    ArrayHandle<Scalar> h_fn(m_grid->getVelocityGrid(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);
    Scalar3 grid_spacing = m_grid->getSpacing();
    uint3 dims = m_grid->getDimensions();
    Index3D indexer = this->m_grid->getIndexer();

    // Access field data
    const std::vector<std::vector<uint3> > layers = m_field->getLayers();
    const std::map<char, char> layer_indexer = m_field->getIndex();

    // Use periodic flags
    const BoxDim& box = this->m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();

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
                if (abs(boundary_distance) < abs(min_distance))
                    min_distance = boundary_distance;
                }
            } // End loops over neighbors

        h_phi.data[cur_cell] = min_distance;
        /*NOTE:
         * It may be worth adding a little logic here that removes L0 cells when the distance is clearly too far; currently, it will find both cells adjacent to the interface, so being more specific would help reduce the load.
         */
        }
    }

Scalar FastMarcher::calculateTentativeDistance(uint3 cell, bool positive)
    {
    // Access grid data
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);
    Scalar3 grid_spacing = m_grid->getSpacing();
    Index3D indexer = this->m_grid->getIndexer();

    Scalar missing_value = 0; // The grid value that indicates that a cell's distance has not yet been finalized
    Scalar guessed_distance = std::numeric_limits<Scalar>::infinity();

    // Loop over possible phi values
    for (int i = -1; i <= 1; i+=2)
        for (int j = -1; j <= 1; j+=2)
            for (int k = -1; k <= 1; k+=2)
                {
                    //NOTE: FOR ALL OF THIS TO WORK, I MAY NEED TO INITIALIZE THE GRID TO INFINITY RATHER THAN 0

                // The neighboring phis to use
                Scalar phi1 = h_phi.data[indexer(cell.x+i, cell.y, cell.z)];
                Scalar phi2 = h_phi.data[indexer(cell.x, cell.y+j, cell.z)];
                Scalar phi3 = h_phi.data[indexer(cell.x, cell.y, cell.z+k)];

                // The distance estimate, initialized to infinity
                Scalar calculated_distance = std::numeric_limits<Scalar>::infinity();
                std::pair<Scalar, Scalar> calculated_distances;
                // Have to go through a long line of conditionals
                if (phi1 != missing_value)
                    {
                    if (phi2 != missing_value)
                        {
                        if (phi3 != missing_value)
                            {
                            // All values exist, can do 3 point solution
                            calculated_distances = lagrange3D(grid_spacing.x, phi1, grid_spacing.y, phi2, grid_spacing.z, phi3);
                            calculated_distance = positive ? max(calculated_distances.first, calculated_distances.second) : min(calculated_distances.first, calculated_distances.second);
                            }
                        else
                            {
                            // Solve with phi1 and phi2
                            calculated_distances = lagrange2D(grid_spacing.x, phi1, grid_spacing.y, phi2);
                            calculated_distance = positive ? max(calculated_distances.first, calculated_distances.second) : min(calculated_distances.first, calculated_distances.second);
                            }
                        }
                    else
                        {
                        if (phi3 != missing_value)
                            {
                            // Solve with phi1 and phi3
                            calculated_distances = lagrange2D(grid_spacing.x, phi1, grid_spacing.z, phi3);
                            calculated_distance = positive ? max(calculated_distances.first, calculated_distances.second) : min(calculated_distances.first, calculated_distances.second);
                            }
                        else
                            {
                            // Solve with just phi1
                            calculated_distances = lagrange1D(grid_spacing.x, phi1);
                            calculated_distance = positive ? max(calculated_distances.first, calculated_distances.second) : min(calculated_distances.first, calculated_distances.second);
                            }
                        }
                    }
                else
                    {
                    if (phi2 != missing_value)
                        {
                        if (phi3 != missing_value)
                            {
                            // Solve with phi2 and phi3
                            calculated_distances = lagrange2D(grid_spacing.y, phi2, grid_spacing.z, phi3);
                            calculated_distance = positive ? max(calculated_distances.first, calculated_distances.second) : min(calculated_distances.first, calculated_distances.second);
                            }
                        else
                            {
                            // Solve with just phi2
                            calculated_distances = lagrange1D(grid_spacing.y, phi2);
                            calculated_distance = positive ? max(calculated_distances.first, calculated_distances.second) : min(calculated_distances.first, calculated_distances.second);
                            }
                        }
                    else
                        {
                        if (phi3 != missing_value)
                            {
                            // Solve with just phi3
                            calculated_distances = lagrange1D(grid_spacing.z, phi3);
                            calculated_distance = positive ? max(calculated_distances.first, calculated_distances.second) : min(calculated_distances.first, calculated_distances.second);
                            }
                        else
                            {
                            // We are looking in an octant that has no values yet.
                            // An example (in 2d) would be the following scenario,
                            // with Inf representing cells without known distances:
                            //
                            // ----------------
                            // |    |    |    |
                            // |    | 1.2|    |
                            // |----|----|----|
                            // |    |    |    |
                            // | 1.4|    | Inf|
                            // |----|----|----|
                            // |    |    |    |
                            // |    | Inf|    |
                            // ----------------
                            // In this case, trying to calculate using the bottom and
                            // right cells for phi1 and phi2 would lead to no possible
                            // guess. In this case, this set has nothing to contribute
                            continue;
                            }
                        }
                    }

                //NOTE: THIS NEEDS TO BE INTELLIGENT WITH RESPECT TO THE SIGN (i.e. sometimes compares with greater, sometimes with less than)
                if (abs(calculated_distance) < abs(guessed_distance))
                    guessed_distance = calculated_distance;
                } // End loop over phi values
    assert(guessed_distance != std::numeric_limits<Scalar>::infinity);
    return guessed_distance;
    }

std::pair<Scalar, Scalar> FastMarcher::lagrange1D(Scalar delta_x1, Scalar phi1)
    {
        assert(delta_x1 > 0);
        return std::pair<Scalar, Scalar>(phi1 + delta_x1, phi1 - delta_x1);
    }

std::pair<Scalar, Scalar> FastMarcher::lagrange2D(Scalar delta_x1, Scalar phi1, Scalar delta_x2, Scalar phi2)
    {
        assert(delta_x1 > 0);
        assert(delta_x2 > 0);

        // First make sure we don't have to fall back to the 2D solution
        Scalar P1 = lagrangeP2(phi1, delta_x1, phi1, delta_x2, phi2);
        Scalar P2 = lagrangeP2(phi2, delta_x1, phi1, delta_x2, phi2);
        if (P1 > 1)
            return lagrange1D(delta_x2, phi2);
        else if (P2 > 1)
            return lagrange1D(delta_x1, phi1);
        else
            {
            // The usual elements of a quadratic equation
            Scalar a = 1.0/(delta_x1*delta_x1) + 1.0/(delta_x2*delta_x2);
            Scalar b = -2.0*(phi1/(delta_x1*delta_x1) + phi2/(delta_x2*delta_x2));
            Scalar c = (-1.0 + (phi1*phi1)/(delta_x1*delta_x1) + (phi2*phi2)/(delta_x2*delta_x2));

            // Make sure we won't get imaginary solutions before proceeding
            Scalar bsquare = b*b;
            Scalar four_ac = 4.0*a*c;
            Scalar discriminant = bsquare - four_ac;
            assert(discriminant > 0);
            Scalar sqrt_discriminant = sqrt(discriminant);

            return std::pair<Scalar, Scalar>((-b + sqrt_discriminant)/(2.0*a), (-b - sqrt_discriminant)/(2.0*a));
            }
    }

std::pair<Scalar, Scalar> FastMarcher::lagrange3D(Scalar delta_x1, Scalar phi1, Scalar delta_x2, Scalar phi2, Scalar delta_x3, Scalar phi3)
    {
        //NOTE: HOPEFULLY THE USE OF THE LAGRANGEP2 AND LAGRANGEP3 FUNCTIONS GETS RID OF THE COMPLEX NUMBERS ISSUE, BUT IF NOT WE CAN REVISIT
        assert(delta_x1 > 0);
        assert(delta_x2 > 0);
        assert(delta_x3 > 0);

        // First make sure we don't have to fall back to the 2D solution
        Scalar P1 = lagrangeP3(phi1, delta_x1, phi1, delta_x2, phi2, delta_x3, phi3);
        Scalar P2 = lagrangeP3(phi2, delta_x1, phi1, delta_x2, phi2, delta_x3, phi3);
        Scalar P3 = lagrangeP3(phi3, delta_x1, phi1, delta_x2, phi2, delta_x3, phi3);
        if (P1 > 1)
            return lagrange2D(delta_x2, phi2, delta_x3, phi3);
        else if (P2 > 1)
            return lagrange2D(delta_x1, phi1, delta_x3, phi3);
        else if (P3 > 1)
            return lagrange2D(delta_x1, phi1, delta_x2, phi2);
        else
            {
            // The usual elements of a quadratic equation
            Scalar a = 1.0/(delta_x1*delta_x1) + 1.0/(delta_x2*delta_x2) + 1.0/(delta_x3*delta_x3);
            Scalar b = -2.0*(phi1/(delta_x1*delta_x1) + phi2/(delta_x2*delta_x2) + phi3/(delta_x3*delta_x3));
            Scalar c = (-1.0 + (phi1*phi1)/(delta_x1*delta_x1) + (phi2*phi2)/(delta_x2*delta_x2) + (phi3*phi3)/(delta_x3*delta_x3));

            // Make sure we won't get imaginary solutions before proceeding
            Scalar bsquare = b*b;
            Scalar four_ac = 4.0*a*c;
            Scalar discriminant = bsquare - four_ac;
            assert(discriminant > 0);
            Scalar sqrt_discriminant = sqrt(discriminant);

            return std::pair<Scalar, Scalar>((-b + sqrt_discriminant)/(2.0*a), (-b - sqrt_discriminant)/(2.0*a));
            }
    }

Scalar FastMarcher::lagrangeP2(Scalar phi, Scalar delta_x1, Scalar phi1, Scalar delta_x2, Scalar phi2)
    {
        Scalar t1 = (phi - phi1)/delta_x1;
        Scalar t2 = (phi - phi2)/delta_x2;
        return t1*t1 + t2*t2;
    }
Scalar FastMarcher::lagrangeP3(Scalar phi, Scalar delta_x1, Scalar phi1, Scalar delta_x2, Scalar phi2, Scalar delta_x3, Scalar phi3)
    {
        Scalar t1 = (phi - phi1)/delta_x1;
        Scalar t2 = (phi - phi2)/delta_x2;
        Scalar t3 = (phi - phi3)/delta_x3;
        return t1*t1 + t2*t2 + t3*t3;
    }
} // end namespace solvent
