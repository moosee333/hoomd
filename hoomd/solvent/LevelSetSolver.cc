// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: vramasub

#include "LevelSetSolver.h"

using namespace std;
namespace py = pybind11;

/*! \file LevelSetSolver.cc
    \brief Contains code for the LevelSetSolver class
*/

namespace solvent
{

//! Constructor
LevelSetSolver::LevelSetSolver(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<GridData> grid)
    : ForceCompute(sysdef),
      m_sysdef(sysdef),
      m_pdata(sysdef->getParticleData()),
      m_exec_conf(m_pdata->getExecConf()),
      m_updater(new SparseFieldUpdater(m_sysdef, grid)),
      m_marcher(new FastMarcher(m_sysdef, m_updater)),
      m_grid(grid)
    { }

//! Destructor
LevelSetSolver::~LevelSetSolver()
    { }

void LevelSetSolver::addGridForceCompute(std::shared_ptr<GridForceCompute> gfc)
    {
    assert(gfc);
    m_grid_forces.push_back(gfc);
    }

void LevelSetSolver::computeForces(unsigned int timestep)
    {
    // We need to precompute the energy for each of grid forces before performing any level set operations
	for(std::vector<std::shared_ptr<GridForceCompute> >::iterator grid_force = m_grid_forces.begin(); grid_force != m_grid_forces.end(); ++grid_force)
        {
		(*grid_force)->compute(timestep);
        }

    // Utilize the SparseFieldUpdater to keep track of the level sets of interest
    m_updater->clearField();
    m_updater->computeInitialField();

    // Now use the fast marcher to compute the distances
    m_marcher->march();

    // Once the initial grid is established we compute the numerical derivatives
    /*
     * The next steps once the distances are computed is to actually compute the move.
     * For that, we need to do a number of things
     * I'm breaking up the computation like the VISM paper, so I need A and B terms separately
     */
    //NOTE: FOR THE BELOW FUNCTIONS, I AM CURRENTLY USING LZ INTERNALLY. HOWEVER, IF 
    //I START USING A WIDER BAND, THEN I SHOULD BE ABLE TO ALSO HAVE CURVATURE INFORMATION
    //FOR MORE LAYERS, AND THEREFORE I SHOULD ALSO BE ABLE TO HAVE BETTER DATA TO COMPUTE
    //THE INTERPOLATIONS ETC THAT I NEED
    GPUArray<Scalar> A = computeA();
    GPUArray<Scalar> Bphi = computeBphi();

    //NOTE: Jens doesn't actually use the parabolic A right now; I think this is because of what he said
    //vis-a-vis computing the timestep with it but then using the non-linearized version, but it still
    //seems weird
    
    
    // Now that we have A and BPhi, we should be able to compute the forces. However, I don't think this
    // is quite good enough; I'm no longer making use of the interpolation I was doing before in order to
    // get the values, I'm just taking the exact values at the grid point, which is not what I want to do.
    // I think I need to use the boundary vector calculation to find the boundary, and then interpolate
    // somehow; I'll have to look at my prototype.
    // I'm actually not sure how this should work now; since the A term includes the norm internally, and
    // that norm is computed to accomodate various stability criteria, I'm not sure that applying an
    // interpolation in the same way that I was before is still appropriate.
    // I think I have to apply this to the A term before multiplying by the norm grad, and then do the 
    // linearization, and then multiply. For the B term, I should be able to do it with the existing B term.
    // Doing it for each of them independently should work.
    //NOTE: Avoid recomputing gradient; should find one spot where I calculate and then pass it everywhere

    // The total change in boundary force is just the sum of the two terms; now we can take the Euler step
    const std::vector<std::vector<uint3> > layers = m_updater->getLayers();
    const std::map<char, char> layer_indexer = m_updater->getIndex();
    const std::vector<uint3> Lz = layers[layer_indexer.find(0)->second];
    GPUArray<Scalar> Fn_phi(Lz.size(), m_exec_conf);
    ArrayHandle<Scalar> h_Fn_phi(Fn_phi, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_A(A, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_Bphi(Bphi, access_location::host, access_mode::readwrite);
    for (unsigned int i = 0; i < Lz.size(); i++)
        {
        h_Fn_phi.data[i] = h_A.data[i] + h_Bphi.data[i];
        }
    
    // Extend velocities to rest of grid
    // Euler step
    ArrayHandle<Scalar> h_phi(m_grid->getPhiGrid(), access_location::host, access_mode::readwrite);

    }

GPUArray<Scalar> LevelSetSolver::computeA()
    {
        /*
         * For the A term, we can compute it on all points on the grid, not just Lz,
         * because the computation should still be mathematically equivalent to actually
         * extending (and therefore is probably first-order accurate numerically). Therefore,
         * we can loop over the layers and update them individually
         */
    const std::vector<std::vector<uint3> > layers = m_updater->getLayers();
    const std::map<char, char> layer_indexer = m_updater->getIndex();


    Index3D indexer = m_grid->getIndexer();
    uint3 dims = m_grid->getDimensions();
    unsigned int n_elements = dims.x*dims.y*dims.z;
    GPUArray<Scalar> A_all(n_elements, m_exec_conf); 
    ArrayHandle<Scalar> h_A_all(A_all, access_location::host, access_mode::readwrite);

    //NOTE: Make sure the typecasting here will work as expected
    for (int i = (-1)*(int)m_updater->getNumLayers(); i <= m_updater->getNumLayers(); i++)
        {
        const std::vector<uint3> layer = layers[layer_indexer.find(i)->second];
        unsigned int n_layer_elements = layer.size();

        // Get gradient
        GPUArray<Scalar> dx(n_layer_elements, m_exec_conf), dy(n_layer_elements, m_exec_conf), dz(n_layer_elements, m_exec_conf); 
        m_grid->grad(dx, dy, dz, layer);

        // Compute norm of gradient
        GPUArray<Scalar> norm_grad(n_layer_elements, m_exec_conf); 
        ArrayHandle<Scalar> h_norm_grad(norm_grad, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_dx(dx, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_dy(dy, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_dz(dz, access_location::host, access_mode::readwrite);
        for (unsigned int i = 0; i < n_layer_elements; i++)
            {
            h_norm_grad.data[i] = sqrt(h_dx.data[i]*h_dx.data[i] + h_dy.data[i]*h_dy.data[i] + h_dz.data[i]*h_dz.data[i]);
            }

        // Compute hessian
        GPUArray<Scalar> ddxx(n_layer_elements, m_exec_conf), ddxy(n_layer_elements, m_exec_conf), ddxz(n_layer_elements, m_exec_conf), ddyy(n_layer_elements, m_exec_conf), ddyz(n_layer_elements, m_exec_conf), ddzz(n_layer_elements, m_exec_conf);
        m_grid->hessian(ddxx, ddxy, ddxz, ddyy, ddyz, ddzz, layer);

        // Compute both curvature terms
        GPUArray<Scalar> H(n_layer_elements, m_exec_conf); 
        GPUArray<Scalar> K(n_layer_elements, m_exec_conf); 
        m_grid->getMeanCurvature(H, dx, dy, dz, ddxx, ddxy, ddxz, ddyy, ddyz, ddzz, layer);
        m_grid->getGaussianCurvature(K, dx, dy, dz, ddxx, ddxy, ddxz, ddyy, ddyz, ddzz, layer);

        // Need the values of B1 on the surface to determine the timestep
        //NOTE: Make sure that this works for non-Lz cells (i.e. whether using the current energies for B1 is the right approach). I think it's fine, only B (not B1) needs to worry about extension.
        GPUArray<Scalar> B1(n_layer_elements, m_exec_conf);
        this->computeB1(B1, layer);

        // Perform the linearization to ensure parabolicity of the tau matrix
        GPUArray<Scalar> tau(n_layer_elements, m_exec_conf);
        Scalar dt = 0; // Not sure if this requires initialization
        this->linearizeParabolicTerm(n_layer_elements, H, K, B1, tau, dt);

        GPUArray<Scalar> A(n_layer_elements, m_exec_conf); 
        ArrayHandle<Scalar> h_A(A, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_H(H, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_K(K, access_location::host, access_mode::readwrite);
        for (unsigned int i = 0; i < n_layer_elements; i++)
            {
            h_A.data[i] = 2*m_gamma_0*(h_H.data[i] - m_tau*h_K.data[i])*h_norm_grad.data[i];
            }

        // Insert values for this layer into the overall grid
        for (unsigned int j = 0; j < n_layer_elements; j++)
            {
            uint3 point = layer[j];
            unsigned int idx = indexer(point.x, point.y, point.z);
            h_A_all.data[idx] = h_A.data[j];
            }
        } // End loop over layers

    return A_all;
    }

GPUArray<Scalar> LevelSetSolver::computeBphi()
    {
        /*
         * Computing the B term involves marching outwards to find the best approximation of the normal.
         * Need to use an upwind scheme for this as well.
         * First, we compute the values on Lz; then we extrapolate to the rest of the grid.
         */
    const std::vector<std::vector<uint3> > layers = m_updater->getLayers();
    const std::map<char, char> layer_indexer = m_updater->getIndex();

    const std::vector<uint3> Lz = layers[layer_indexer.find(0)->second];
    Index3D grid_indexer = m_grid->getIndexer();
    uint3 dims = m_grid->getDimensions();
    unsigned int n_elements = dims.x*dims.y*dims.z;

    /*
     * The computation of the B term itself is simple.
     * The trick is regularizing the derivative of phi to maintain
     * the stability of integrating the hyperbolic term
     */

    // We first compute compute the approximate B term on the Lz grid points
    // Note that we place it directly on a grid rather than storing it in a vector
    // for use with the boundary interpolation
    GPUArray<Scalar> B_estimate(n_elements, m_exec_conf);

        { // Scope the ArrayHandles to avoid conflicts when calling grid functions
        ArrayHandle<Scalar> h_B_estimate(B_estimate, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_fn(m_grid->getVelocityGrid(), access_location::host, access_mode::readwrite);
        Index3D indexer = m_grid->getIndexer();
        
        for (unsigned int i = 0; i < Lz.size(); i++)
            {
            uint3 point = Lz[i];
            unsigned int idx = indexer(point.x, point.y, point.z);

            auto coulomb_term = 0;
            m_exec_conf->msg->notice(1) << "Currently setting coulomb term to 0 in the computation of the B term; this must be updated" << std::endl;

            h_B_estimate.data[idx] = (m_delta_p - m_rho_water*h_fn.data[idx] + coulomb_term);
            }
        }

    // Since we have B on a grid, we can use it directly for interpolation. Once we
    // interpolate the "exact" values of B, we use the fast marching method to extend
    // these velocities along characteristics to the rest of the sparse field
    GPUArray<Scalar> B_final = m_marcher->boundaryInterp(B_estimate);
    m_marcher->extend_velocities(B_final);

    // Finally, use upwind finite differencing to compute the appropriately normalized norm of
    // phi on all layers, then multiply this by the B value to get the finalized Bphi value
    GPUArray<Scalar> Bphi(n_elements, m_exec_conf);
    ArrayHandle<Scalar> h_Bphi(Bphi, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_B_final(B_final, access_location::host, access_mode::readwrite);

    for (std::vector<std::vector<uint3>>::const_iterator layer = layers.begin(); layer != layers.end(); layer++)
        {
        GPUArray<Scalar> layer_norm_phi_upwind = m_grid->getNormUpwind(*layer);
        ArrayHandle<Scalar> h_layer_norm_phi_upwind(layer_norm_phi_upwind, access_location::host, access_mode::readwrite);
        for (unsigned int i = 0; i < layer->size(); i++)
            {
            uint3 point = (*layer)[i];
            unsigned int idx = grid_indexer(point.x, point.y, point.z);
            h_Bphi.data[idx] = h_B_final.data[idx]*h_layer_norm_phi_upwind.data[i];
            }
        }

    return Bphi;
    }

void LevelSetSolver::linearizeParabolicTerm(unsigned int n_elements, GPUArray<Scalar>& H, GPUArray<Scalar>& K, GPUArray<Scalar>& B1, GPUArray<Scalar>& tau, Scalar& dt)
    {
    //NOTE: tau and dt are changed internally by reference
    /*
     * In order to ensure that the integration scheme is stable, the A term
     * of the step must be adjusted such that the differential operator is
     * parabolic. This restriction is reflected in the eigenvalues of the
     * operator, which correspond to the Tolman length; therefore, we can
     * ensure parabolicity by scaling the Tolman length.
     */
    ArrayHandle<Scalar> h_tau(tau, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_H(H, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_K(K, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_B1(B1, access_location::host, access_mode::readwrite);

    Scalar denom_max = 0;

    auto spacing = m_grid->getSpacing();
    auto h = min(min(spacing.x, spacing.y), spacing.z);

    for(unsigned int i = 0; i < n_elements; i++)
        {
        auto kappa1 = h_H.data[i] + sqrt(h_H.data[i]*h_H.data[i] - h_K.data[i]);
        auto kappa2 = h_H.data[i] - sqrt(h_H.data[i]*h_H.data[i] - h_K.data[i]);

        auto a1 = 1 - m_tau*kappa1;
        auto a2 = 1 - m_tau*kappa2;

        if (a1 < 0.5 && a2 >= 0.5)
            {
            h_tau.data[i] = 0.5/kappa1;
            a1 = 0.5;
            a2 = 1 - 0.5*kappa2/kappa1;
            }
        else if (a1 >= 0.5 && a2 < 0.5)
            {
            h_tau.data[i] = 0.5/kappa2;
            a1 = 1 - 0.5*kappa1/kappa2;
            a2 = 0.5;
            }
        else if (a1 < 0.5 && a2 < 0.5)
            {
            h_tau.data[i] = min(0.5/kappa1, 0.5/kappa2);
            a1 = 1 - h_tau.data[i]/kappa1;
            a2 = 1 - h_tau.data[i]/kappa2;
            }
        else
            {
            h_tau.data[i] = m_tau;
            }

        //NOTE: I'm not sure why this has a gamma_0 term in it. May arise from the linearization,
        //look into this further.
        //NOTE: I need to figure out how I'm getting B1 since I don't have the Coulomb terms at
        //this stage.
        auto denom_current = m_gamma_0*(a1+a2)/h + abs(h_B1.data[i]);
        if (denom_current > denom_max)
            {
            denom_max = denom_current;
            }
        }
    //NOTE: Not sure how to choose m_alpha.
    dt = m_alpha*h/denom_max;
    }

void LevelSetSolver::computeB1(GPUArray<Scalar> B1, std::vector<uint3> points)
    {
    // Access grid data
    ArrayHandle<Scalar> h_fn(m_grid->getVelocityGrid(), access_location::host, access_mode::read);
    Index3D indexer = m_grid->getIndexer();

    ArrayHandle<Scalar> h_B1(B1, access_location::host, access_mode::read);

    //NOTE: Need to figure out to compute the Coulomb term
    for (unsigned int i = 0; i < points.size(); i++)
        {
        uint3 point = points[i];
        auto coulomb_term = 0;
        m_exec_conf->msg->notice(1) << "Currently setting coulomb term to 0 in the computation of the B1 term; this must be updated" << std::endl;
        h_B1.data[i] = m_delta_p + m_rho_water*h_fn.data[indexer(point.x, point.y, point.z)] + coulomb_term;
        }
    }

void export_LevelSetSolver(py::module& m)
    {
    pybind11::class_<LevelSetSolver, std::shared_ptr<LevelSetSolver> >(m, "LevelSetSolver", pybind11::base<ForceCompute>())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<GridData> >())
        .def("addGridForceCompute", &LevelSetSolver::addGridForceCompute)
    ;
    }

} // end namespace solvent
