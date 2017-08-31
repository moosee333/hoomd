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
		// (*grid_force)->setGrid(m_grid); // Happens on the python side.
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
    GPUArray<Scalar> A = computeA();
    GPUArray<Scalar> Bphi = computeBphi();

    //NOTE: Jens doesn't actually use the parabolic A right now; I think this is because of what he said
    //vis-a-vis computing the timestep with it but then using the non-linearized version, but it still
    //seems weird
    }

GPUArray<Scalar> LevelSetSolver::computeA()
    {
    // Use the gradient to find the direction to the boundary, then multiply by the phi grid's value to compute the vector
    const std::vector<std::vector<uint3> > layers = m_updater->getLayers();
    const std::map<char, char> layer_indexer = m_updater->getIndex();

    const std::vector<uint3> Lz = layers[layer_indexer.find(0)->second];
    unsigned int n_elements = Lz.size();

    // Get gradient
    GPUArray<Scalar> dx(n_elements, m_exec_conf), dy(n_elements, m_exec_conf), dz(n_elements, m_exec_conf); 
    ArrayHandle<Scalar> h_dx(dx, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dy(dy, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_dz(dz, access_location::host, access_mode::readwrite);
    m_grid->grad(dx, dy, dz, Lz);

    // Compute norm of gradient
    GPUArray<Scalar> norm_grad(n_elements, m_exec_conf); 
    ArrayHandle<Scalar> h_norm_grad(norm_grad, access_location::host, access_mode::readwrite);
    for (unsigned int i = 0; i < n_elements; i++)
        {
        h_norm_grad.data[i] = sqrt(h_dx.data[i]*h_dx.data[i] + h_dy.data[i]*h_dy.data[i] + h_dz.data[i]*h_dz.data[i]);
        }

    // Compute hessian
    GPUArray<Scalar> ddxx(n_elements, m_exec_conf), ddxy(n_elements, m_exec_conf), ddxz(n_elements, m_exec_conf), ddyy(n_elements, m_exec_conf), ddyz(n_elements, m_exec_conf), ddzz(n_elements, m_exec_conf);
    m_grid->hessian(ddxx, ddxy, ddxz, ddyy, ddyz, ddzz, Lz);

    // Compute both curvature terms
    GPUArray<Scalar> H(n_elements, m_exec_conf); 
    GPUArray<Scalar> K(n_elements, m_exec_conf); 
    m_grid->getMeanCurvature(H, dx, dy, dz, ddxx, ddxy, ddxz, ddyy, ddyz, ddzz, Lz);
    m_grid->getGaussianCurvature(K, dx, dy, dz, ddxx, ddxy, ddxz, ddyy, ddyz, ddzz, Lz);

    // Need the values of B1 on the surface to determine the timestep
    GPUArray<Scalar> B1(n_elements, m_exec_conf);
    this->computeB1(B1, Lz);

    // Perform the linearization to ensure parabolicity of the tau matrix
    GPUArray<Scalar> tau(n_elements, m_exec_conf);
    Scalar dt = 0; // Not sure if this requires initialization
    this->linearizeParabolicTerm(n_elements, H, K, B1, tau, dt);

    GPUArray<Scalar> A(n_elements, m_exec_conf); 
    ArrayHandle<Scalar> h_A(A, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_H(H, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_K(K, access_location::host, access_mode::readwrite);
    for (unsigned int i = 0; i < n_elements; i++)
        {
        h_A.data[i] = 2*m_gamma_0*(h_H.data[i] - m_tau*h_K.data[i])*h_norm_grad.data[i];
        }
    return A;
    }

GPUArray<Scalar> LevelSetSolver::computeBphi()
    {
    const std::vector<std::vector<uint3> > layers = m_updater->getLayers();
    const std::map<char, char> layer_indexer = m_updater->getIndex();

    const std::vector<uint3> Lz = layers[layer_indexer.find(0)->second];
    unsigned int n_elements = Lz.size();

    /*
     * The computation a of the B term itself is simple.
     * The trick is regularizing the derivative of phi to maintain
     * the stability of integrating the hyperbolic term
     */
    GPUArray<Scalar> Bphi(n_elements, m_exec_conf);
    ArrayHandle<Scalar> h_Bphi(Bphi, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_fn(m_grid->getVelocityGrid(), access_location::host, access_mode::readwrite);
    Index3D indexer = m_grid->getIndexer();

    // To finalize the calculation of B we need the normalized version of nabla phi
    GPUArray<Scalar> norm_phi_upwind = m_grid->getNormUpwind(Lz);
    ArrayHandle<Scalar> h_norm_phi_upwind(norm_phi_upwind, access_location::host, access_mode::readwrite);
    
    for (unsigned int i = 0; i < n_elements; i++)
        {
        uint3 point = Lz[i];
        auto coulomb_term = 0;
        m_exec_conf->msg->notice(1) << "Currently setting coulomb term to 0 in the computation of the B term; this must be updated" << std::endl;
        h_Bphi.data[i] = (m_delta_p - m_rho_water*h_fn.data[indexer(point.x, point.y, point.z)] + coulomb_term)*h_norm_phi_upwind.data[i];
        }

    return Bphi;
    }

void LevelSetSolver::linearizeParabolicTerm(unsigned int n_elements, GPUArray<Scalar>& H, GPUArray<Scalar>& K, GPUArray<Scalar>& B1, GPUArray<Scalar>& tau, Scalar& dt)
    {
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
