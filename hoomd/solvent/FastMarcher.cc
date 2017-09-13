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
FastMarcher::FastMarcher(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<SparseFieldUpdater> updater)
    :
    m_sysdef(sysdef),
    m_pdata(sysdef->getParticleData()),
    m_exec_conf(m_pdata->getExecConf()),
    m_updater(updater),
    m_grid(updater->getGrid())
    { }

//! Destructor
FastMarcher::~FastMarcher()
    { }

void FastMarcher::march()
    {
    // The "exact" Lz values are required before anything else can be done.
    estimateLzDistances();

    // Access field data
    const std::vector<std::vector<uint3> > layers = m_updater->getLayers();
    const std::map<char, char> layer_indexer = m_updater->getIndex();
    Index3D indexer = this->m_grid->getIndexer();

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
        Scalar d = calculateTentativeDistance(*element, true);
        outer_tentative_distances.push(std::pair<uint3, Scalar>(*element, d));
        current_outer_tentative_distances[*element] = d;
        }

    // Build lists of eligible points
    std::vector<uint3> eligible_positive;
    unsigned int num_eligible_positive = 0;
    for (unsigned int layer = 1; layer <= m_updater->getNumLayers(); layer++)
        num_eligible_positive += layers[layer_indexer.find(layer)->second].size();
    eligible_positive.reserve(num_eligible_positive);
    for (unsigned int layer = 1; layer <= m_updater->getNumLayers(); layer++)
        {
        std::vector<uint3> L = layers[layer_indexer.find(layer)->second];
        eligible_positive.insert(eligible_positive.end(), L.begin(), L.end());
        }

    // Then march
    while (!outer_tentative_distances.empty())
        {
        // To ensure that we get the most updated value, we have to check
        // the updated values saved in the map for every point. The outdated
        // tentative values will still exist in the PQ, but we just ignore
        // them when they are removed.
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
            { // Scope usage of phi grid to avoid conflicting with calculateTentativeDistances
            ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);
            h_phi.data[indexer(i, j, k)] = next_distance;
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

        for(unsigned int idx = 0; idx < sizeof(neighbor_indices)/sizeof(int3); idx++)
            {
            int3 neighbor_idx = neighbor_indices[idx];
            uint3 current_neighbor = m_grid->wrap(neighbor_idx);

            // Only bother with points in the layers
            //NOTE: Make sure that make_uint3 is ok here; x, y, z are ints, but should be effectively unsigned
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
    std::priority_queue< std::pair<uint3, Scalar>, std::vector< std::pair<uint3, Scalar> >, CompareScalarLess > inner_tentative_distances;
    std::map<uint3, Scalar, CompareInt> current_inner_tentative_distances;
    const std::vector<uint3> Ln1 = layers[layer_indexer.find(-1)->second];
    for (std::vector<uint3>::const_iterator element = Ln1.begin(); element != Ln1.end(); element++)
        {
        Scalar d = calculateTentativeDistance(*element, false);
        inner_tentative_distances.push(std::pair<uint3, Scalar>(*element, d));
        current_inner_tentative_distances[*element] = d;
        }

    // Build lists of eligible points
    std::vector<uint3> eligible_negative;
    unsigned int num_eligible_negative = 0;
    for (int layer = 1; layer <= m_updater->getNumLayers(); layer++)
        num_eligible_negative += layers[layer_indexer.find(-layer)->second].size();
    eligible_negative.reserve(num_eligible_negative);
    for (int layer = 1; layer <= m_updater->getNumLayers(); layer++)
        {
        std::vector<uint3> L = layers[layer_indexer.find(-layer)->second];
        eligible_negative.insert(eligible_negative.end(), L.begin(), L.end());
        }

    // Then march
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
            { // Scope usage of phi grid to avoid conflicting with calculateTentativeDistances
            ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);
            h_phi.data[indexer(i, j, k)] = next_distance;
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

        for(unsigned int idx = 0; idx < sizeof(neighbor_indices)/sizeof(int3); idx++)
            {
            int3 neighbor_idx = neighbor_indices[idx];
            uint3 current_neighbor = m_grid->wrap(neighbor_idx);

            // Only bother with points in the layers
            //NOTE: Make sure that make_uint3 is ok here; x, y, z are ints, but should be effectively unsigned
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
    Index3D indexer = this->m_grid->getIndexer();

    // Access field data
    const std::vector<std::vector<uint3> > layers = m_updater->getLayers();
    const std::map<char, char> layer_indexer = m_updater->getIndex();

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

        // Check forward and backwards for changes in sign
        Scalar shifts[2] = {1, -1};

        for (unsigned int i = 0; i < directions.size(); i++)
            {
            Scalar step = 0;
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
                uint3 neighbor_idx = m_grid->wrap(make_int3(x+total_shift.x, y+total_shift.y, z+total_shift.z));

                // Distance calculation is performed by taking the change in energy between cells,
                // then finding where along that line the zero energy point lies, and then multiplying
                // that fraction into the grid spacing
                unsigned int neighbor_cell = indexer(neighbor_idx.x, neighbor_idx.y, neighbor_idx.z);

                // If there is no difference between this cell and its neighbor, there cannot be a
                // crossing, so we can skip it. This also accounts for the special case where the two
                // values are precisely the same, so the total_energy_difference is 0
                if (sgn(h_fn.data[neighbor_cell]) == sgn(h_fn.data[cur_cell]))
                    continue;
                Scalar total_energy_difference = h_fn.data[neighbor_cell] - h_fn.data[cur_cell];
                Scalar boundary_distance = sgn(total_energy_difference) * step * h_fn.data[cur_cell] / total_energy_difference; // multiply by the sign so that cur_cell determines total sign
                if (abs(boundary_distance) < abs(min_distance))
                    min_distance = boundary_distance;
                } // End loop over directions 
            } // End loops over neighbors

        h_phi.data[cur_cell] = min_distance;
        /*NOTE:
         * It may be worth adding a little logic here that removes L0 cells when the distance is clearly too far; currently, it will find both cells adjacent to the interface, so being more specific would help reduce the load.
         */
        }
    }

void FastMarcher::extend_velocities(GPUArray<Scalar>& velocities)
    {
        /*
         * We already have the distances. Now we reconstruct a PQ of these distances
         * and go in the same order, computing gradients as we go. This marching is
         * much simpler since there are no tentative updates, just immediate ones.
         */
    
    // Access grid data
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);

    // Access field data
    const std::vector<std::vector<uint3> > layers = m_updater->getLayers();
    const std::map<char, char> layer_indexer = m_updater->getIndex();
    Index3D indexer = this->m_grid->getIndexer();
    Scalar3 spacing = this->m_grid->getSpacing();

    // Access velocities
    ArrayHandle<Scalar> h_velocities(velocities, access_location::host, access_mode::readwrite);

    /**************************
     * March outwards
     *************************/
    //NOTE: assumes that layer0 is already filled in the velocities array
    // Insert all existing positive distances into the grid
    std::priority_queue< std::pair<uint3, Scalar>, std::vector< std::pair<uint3, Scalar> >, CompareScalarGreater > outer_distances;
    for (unsigned int layer_idx = 1; layer_idx < m_updater->getNumLayers(); layer_idx++)
        {
        std::vector<uint3> layer = layers[layer_indexer.find(layer_idx)->second];
        for (std::vector<uint3>::const_iterator element = layer.begin(); element != layer.end(); element++)
            {
            unsigned int idx = indexer(element->x, element->y, element->z);
            outer_distances.push(std::pair<uint3, Scalar>(*element, h_phi.data[idx]));
            }
        }

    // Now march in this order
    while (!outer_distances.empty())
        {
        // This is how we ensure that we get the most updated value
        std::pair<uint3, Scalar> current = outer_distances.top();
        outer_distances.pop();
        uint3 current_point = current.first;
        Scalar current_distance = current.second;

        int i = current_point.x, j = current_point.y, k = current_point.z;
        unsigned int cur_idx = indexer(i, j, k);

        //NOTE: When assigning the S_i I'm assigning a value for when phi_i = 0
        //by default I'm just using the D+ operator. This should be irrelevant
        //because it should cancel; the only case where it could be a problem is
        //if all of the phi_i are 0, but I don't think that this is possible
        //since by construction we are propagating outwards from the interface

        //NOTE: Consider switching to central differences rather than simple upwind
        //for the phi terms since I should be able to compute those

        // Note that while the phi{x,y,z} are the proper finite differences, the
        // S{x,y,z} variables are actually the values of S at the neighbor. This
        // is because when solving the propagation equation for S_{ijk}, the 
        // finite difference terms in S get split.

        // x direction
        Scalar phix;
        unsigned int idx_xp = indexer(i+1, j, k), idx_xm = indexer(i-1, j, k);
        Scalar dist_xp = h_phi.data[idx_xp], dist_xm = h_phi.data[idx_xm];

        // If both neighbors are further from the interface, there is no information to be gained here
        if (min(dist_xp, dist_xm) > current_distance)
            phix = 0;
        else
            {
            // Choose the closer of the two neighbors to the interface
            if (dist_xp < dist_xm)
                phix = (dist_xp - current_distance)/spacing.x;
            else
                phix = (current_distance - dist_xm)/spacing.x;
            }

        // Choose the appropriate differencing operator depending on the sign of the phi derivative
        Scalar Sx = phix > 0 ? h_velocities.data[idx_xm] : h_velocities.data[idx_xp];

        // y direction
        Scalar phiy;
        unsigned int idx_yp = indexer(i, j+1, k) ,idx_ym = indexer(i, j-1, k);
        Scalar dist_yp = h_phi.data[idx_yp], dist_ym = h_phi.data[idx_ym];

        if (min(dist_yp, dist_ym) > current_distance)
            phiy = 0;
        else
            {
            if (dist_yp < dist_ym)
                phiy = (dist_yp - current_distance)/spacing.y;
            else
                phiy = (current_distance - dist_ym)/spacing.y;
            }

        Scalar Sy = phiy > 0 ? h_velocities.data[idx_ym] : h_velocities.data[idx_yp];

        // z direction
        Scalar phiz;
        unsigned int idx_zp = indexer(i, j, k+1) ,idx_zm = indexer(i, j, k-1);
        Scalar dist_zp = h_phi.data[idx_zp], dist_zm = h_phi.data[idx_zm];

        if (min(dist_zp, dist_zm) > current_distance)
            phiz = 0;
        else
            {
            if (dist_zp < dist_zm)
                phiz = (dist_zp - current_distance)/spacing.x;
            else
                phiz  = (current_distance - dist_zm)/spacing.x;
            }

        Scalar Sz = phiz > 0 ? h_velocities.data[idx_zm] : h_velocities.data[idx_zp];

        h_velocities.data[cur_idx] = ((Sx*phix/spacing.x) + (Sy*phiy/spacing.y) + (Sz*phiz/spacing.z)) /
            (phix/spacing.x + phiy/spacing.y + phiz/spacing.z);
        }

    /**************************
     * March inwards
     *************************/
    std::priority_queue< std::pair<uint3, Scalar>, std::vector< std::pair<uint3, Scalar> >, CompareScalarLess > inner_distances;
    // Make sure to use int (not unsigned int) since I'll have to negate
    for (int layer_idx = 1; layer_idx < m_updater->getNumLayers(); layer_idx++)
        {
        std::vector<uint3> layer = layers[layer_indexer.find(-layer_idx)->second];
        for (std::vector<uint3>::const_iterator element = layer.begin(); element != layer.end(); element++)
            {
            unsigned int idx = indexer(element->x, element->y, element->z);
            inner_distances.push(std::pair<uint3, Scalar>(*element, h_phi.data[idx]));
            }
        }

    // Now march in this order
    while (!inner_distances.empty())
        {
        // This is how we ensure that we get the most updated value
        std::pair<uint3, Scalar> current = inner_distances.top();
        inner_distances.pop();
        uint3 current_point = current.first;
        Scalar current_distance = current.second;

        int i = current_point.x, j = current_point.y, k = current_point.z;
        unsigned int cur_idx = indexer(i, j, k);

        // x direction
        Scalar phix;
        unsigned int idx_xp = indexer(i+1, j, k), idx_xm = indexer(i-1, j, k);
        Scalar dist_xp = h_phi.data[idx_xp], dist_xm = h_phi.data[idx_xm];

        // If both neighbors are further from the interface, there is no information to be gained here
        if (max(dist_xp, dist_xm) < current_distance)
            phix = 0;
        else
            {
            // Choose the closer of the two neighbors to the interface
            if (dist_xp > dist_xm)
                phix = (dist_xp - current_distance)/spacing.x;
            else
                phix = (current_distance - dist_xm)/spacing.x;
            }

        // Choose the appropriate differencing operator depending on the sign of the phi derivative.
        // The sign of the comparator is switched relative to the outward marching because all phi
        // values are negative inside the boundary
        Scalar Sx = phix < 0 ? h_velocities.data[idx_xm] : h_velocities.data[idx_xp];

        // y direction
        Scalar phiy;
        unsigned int idx_yp = indexer(i, j+1, k) ,idx_ym = indexer(i, j-1, k);
        Scalar dist_yp = h_phi.data[idx_yp], dist_ym = h_phi.data[idx_ym];

        if (max(dist_yp, dist_ym) < current_distance)
            phiy = 0;
        else
            {
            if (dist_yp > dist_ym)
                phiy = (dist_yp - current_distance)/spacing.y;
            else
                phiy = (current_distance - dist_ym)/spacing.y;
            }

        Scalar Sy = phiy < 0 ? h_velocities.data[idx_ym] : h_velocities.data[idx_yp];

        // z direction
        Scalar phiz;
        unsigned int idx_zp = indexer(i, j, k+1) ,idx_zm = indexer(i, j, k-1);
        Scalar dist_zp = h_phi.data[idx_zp], dist_zm = h_phi.data[idx_zm];

        if (max(dist_zp, dist_zm) < current_distance)
            phiz = 0;
        else
            {
            if (dist_zp > dist_zm)
                phiz = (dist_zp - current_distance)/spacing.x;
            else
                phiz  = (current_distance - dist_zm)/spacing.x;
            }

        Scalar Sz = phiz < 0 ? h_velocities.data[idx_zm] : h_velocities.data[idx_zp];

        //NOTE: Make sure that this exact formulat treats signs correctly when pointing inward (I think it should)
        h_velocities.data[cur_idx] = ((Sx*phix/spacing.x) + (Sy*phiy/spacing.y) + (Sz*phiz/spacing.z)) /
            (phix/spacing.x + phiy/spacing.y + phiz/spacing.z);
        }
    }

GPUArray<Scalar> FastMarcher::boundaryInterp(GPUArray<Scalar>& B_Lz)
    {

    // Access field data
    const std::vector<std::vector<uint3> > layers = m_updater->getLayers();
    const std::map<char, char> layer_indexer = m_updater->getIndex();
    Index3D indexer = this->m_grid->getIndexer();
    Scalar3 spacing = m_grid->getSpacing();
    uint3 dims = m_grid->getDimensions();

    const std::vector<uint3> Lz = layers[layer_indexer.find(0)->second];
    GPUArray<Scalar3> vectors_to_boundary = m_grid->vecToBoundary(Lz);
    Scalar missing_value = GridData::MISSING_VALUE;

    ArrayHandle<Scalar> h_B_Lz(B_Lz, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_vectors_to_boundary(vectors_to_boundary, access_location::host, access_mode::readwrite);

    unsigned int n_elements = dims.x*dims.y*dims.z;
    GPUArray<Scalar> B_interp(n_elements, m_exec_conf);
    ArrayHandle<Scalar> h_B_interp(B_interp, access_location::host, access_mode::readwrite);
    for (unsigned int i = 0; i < Lz.size(); i++)
        {
        Scalar3 vector_to_boundary = h_vectors_to_boundary.data[i];
        int3 point = make_int3(Lz[i].x, Lz[i].y, Lz[i].z);
        //Scalar3 boundary_locations = make_scalar3(Lz[i].x, Lz[i].y, Lz[i].z)*spacing + spacing/2 + vectors_to_boundary[i];

        // For a consistent mathematical formulation of the interpolation,
        // we determine whether to use the current cell or the previous one
        // in each direction as the basis of the interpolation depending on
        // where the boundary is determined to be relative to the current cell 
        vec3<bool> use_previous = vec3<bool>(vector_to_boundary.x < 0, vector_to_boundary.y < 0, vector_to_boundary.z < 0);
        int3 index_shifter = make_int3(use_previous.x * (-1), use_previous.y * (-1), use_previous.z * (-1)); 
        
        // When use_previous is true, epsilon_unsigned will be negative, so it can be directly
        // added to 1 rather than subtracted
        uint3 point_to_use = m_grid->wrap(point + index_shifter);
        Scalar3 epsilon_unsigned = vector_to_boundary/spacing;
        Scalar epsilon1  = use_previous.x ? 1 + epsilon_unsigned.x : epsilon_unsigned.x;
        Scalar epsilon2  = use_previous.y ? 1 + epsilon_unsigned.y : epsilon_unsigned.y;
        Scalar epsilon3  = use_previous.z ? 1 + epsilon_unsigned.z : epsilon_unsigned.z;
        
        // For clarity, the full interpolation formula is constructed very
        // explicitly term-by-term. This can be changed later if it is too
        // verbose.
        Scalar eps000 = (1-epsilon1)*(1-epsilon2)*(1-epsilon3);
        Scalar eps001 = (1-epsilon1)*(1-epsilon2)*epsilon3;
        Scalar eps010 = (1-epsilon1)*epsilon2*(1-epsilon3);
        Scalar eps011 = (1-epsilon1)*epsilon2*epsilon3;
        Scalar eps100 = epsilon1*(1-epsilon2)*(1-epsilon3);
        Scalar eps101 = epsilon1*(1-epsilon2)*epsilon3;
        Scalar eps110 = epsilon1*epsilon2*(1-epsilon3);
        Scalar eps111 = epsilon1*epsilon2*epsilon3;

        uint3 index000 = point_to_use;
        uint3 index001 = make_uint3(point_to_use.x, point_to_use.y, point_to_use.z + 1);
        uint3 index010 = make_uint3(point_to_use.x, point_to_use.y + 1, point_to_use.z);
        uint3 index011 = make_uint3(point_to_use.x, point_to_use.y + 1, point_to_use.z + 1);
        uint3 index100 = make_uint3(point_to_use.x + 1, point_to_use.y, point_to_use.z);
        uint3 index101 = make_uint3(point_to_use.x + 1, point_to_use.y, point_to_use.z + 1);
        uint3 index110 = make_uint3(point_to_use.x + 1, point_to_use.y + 1, point_to_use.z);
        uint3 index111 = make_uint3(point_to_use.x + 1, point_to_use.y + 1, point_to_use.z + 1);

        // As a best guess, for cells that don't have well defined energies (e.g. anything 
        // not on Lz), we just replace it with the original cell (which is guaranteed to have
        // values)
        //NOTE: Does this work if use_previous is true in any direction? Or does that make the
        //base cell also possibly invalid (e.g. missing a value)?
        Scalar energy000 = h_B_Lz.data[indexer(index000.x, index000.y, index000.z)];
        Scalar energy001 = h_B_Lz.data[indexer(index001.x, index001.y, index001.z)];
        energy001 = (energy001 == missing_value) ? energy000 : energy001;
        Scalar energy010 = h_B_Lz.data[indexer(index010.x, index010.y, index010.z)];
        energy010 = (energy010 == missing_value) ? energy000 : energy010;
        Scalar energy011 = h_B_Lz.data[indexer(index011.x, index011.y, index011.z)];
        energy011 = (energy011 == missing_value) ? energy000 : energy011;
        Scalar energy100 = h_B_Lz.data[indexer(index100.x, index100.y, index100.z)];
        energy100 = (energy100 == missing_value) ? energy000 : energy100;
        Scalar energy101 = h_B_Lz.data[indexer(index101.x, index101.y, index101.z)];
        energy101 = (energy101 == missing_value) ? energy000 : energy101;
        Scalar energy110 = h_B_Lz.data[indexer(index110.x, index110.y, index110.z)];
        energy110 = (energy110 == missing_value) ? energy000 : energy110;
        Scalar energy111 = h_B_Lz.data[indexer(index111.x, index111.y, index111.z)];
        energy111 = (energy111 == missing_value) ? energy000 : energy111;

        unsigned int idx = indexer(point.x, point.y, point.z);
        h_B_interp.data[idx] = eps000*energy000 + eps001*energy001 + eps010*energy010 + eps011*energy011
            + eps100*energy100 + eps101*energy101 + eps110*energy110 + eps111*energy111;
        }
    return B_interp;
    }

Scalar FastMarcher::calculateTentativeDistance(uint3 cell, bool positive)
    {
    // Access grid data
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);
    Scalar3 grid_spacing = m_grid->getSpacing();
    Index3D indexer = this->m_grid->getIndexer();

    Scalar missing_value = GridData::MISSING_VALUE;
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
    assert(guessed_distance != std::numeric_limits<Scalar>::infinity());
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
