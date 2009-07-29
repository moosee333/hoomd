/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id: nve_updater_test.cc 1622 2009-01-28 22:51:01Z joaander $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/unit_tests/nve_updater_test.cc $

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! name the boost unit test module
#define BOOST_TEST_MODULE NVERigidUpdaterTests
#include "boost_utf_configure.h"

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "NVEUpdater.h"
#ifdef ENABLE_CUDA
#include "NVEUpdaterGPU.h"
#endif

#include "BinnedNeighborList.h"
#include "Initializers.h"
#include "LJForceCompute.h"

#ifdef ENABLE_CUDA
#include "BinnedNeighborListGPU.h"
#include "LJForceComputeGPU.h"
#endif

#include "saruprng.h"
#include <math.h>
#include <time.h>

using namespace std;
using namespace boost;

/*! \file nve_rigid_updater_test.cc
	\brief Implements unit tests for NVERigidUpdater 
	\ingroup unit_tests
*/


//! Tolerance for floating point comparisons
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(1e-2);
#else
const Scalar tol = 1e-3;
#endif

//! Typedef'd NVEUpdator class factory
typedef boost::function<shared_ptr<NVEUpdater> (shared_ptr<SystemDefinition> sysdef, Scalar deltaT)> nveup_creator;

void nve_updater_integrate_tests(nveup_creator nve_creator, ExecutionConfiguration exec_conf)
	{
	#ifdef ENABLE_CUDA
	g_gpu_error_checking = true;
	#endif
	
	// check that the nve updater can actually integrate particle positions and velocities correctly
	// start with a 2 particle system to keep things simple: also put everything in a huge box so boundary conditions
	// don't come into play
	shared_ptr<SystemDefinition> sysdef(new SystemDefinition(10, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
	shared_ptr<ParticleData> pdata = sysdef->getParticleData();
	
	ParticleDataArrays arrays = pdata->acquireReadWrite();
	
	// setup a simple initial state
	arrays.x[0] = Scalar(-1.0); arrays.y[0] = 0.0; arrays.z[0] = 0.0;
	arrays.vx[0] = Scalar(-0.5); arrays.body[0] = 0;
	arrays.x[1] =  Scalar(-1.0); arrays.y[1] = 1.0; arrays.z[1] = 0.0;
	arrays.vx[1] = Scalar(0.2); arrays.body[1] = 0;
	arrays.x[2] = Scalar(-1.0); arrays.y[2] = 2.0; arrays.z[2] = 0.0;
	arrays.vy[2] = Scalar(-0.1); arrays.body[2] = 0;
	arrays.x[3] = Scalar(-1.0); arrays.y[3] = 3.0; arrays.z[3] = 0.0;
	arrays.vy[3] = Scalar(0.3);  arrays.body[3] = 0;
	arrays.x[4] = Scalar(-1.0); arrays.y[4] = 4.0; arrays.z[4] = 0.0;
	arrays.vz[4] = Scalar(-0.2); arrays.body[4] = 0;
	
	arrays.x[5] = 0.0; arrays.y[5] = Scalar(0.0); arrays.z[5] = 0.0;
	arrays.vx[5] = Scalar(0.2); arrays.body[5] = 1;
	arrays.x[6] = 0.0; arrays.y[6] = Scalar(1.0); arrays.z[6] = 0.0;
	arrays.vy[6] = Scalar(0.8); arrays.body[6] = 1;
	arrays.x[7] = 0.0; arrays.y[7] = Scalar(2.0); arrays.z[7] = 0.0;
	arrays.vy[7] = Scalar(-0.6); arrays.body[7] = 1;
	arrays.x[8] = 0.0; arrays.y[8] = Scalar(3.0); arrays.z[8] = 0.0;
	arrays.vz[8] = Scalar(0.7); arrays.body[8] = 1;
	arrays.x[9] = 0.0; arrays.y[9] = Scalar(4.0); arrays.z[9] = 0.0;
	arrays.vy[9] = Scalar(-0.5); arrays.body[9] = 1;
	
	pdata->release();
	
	Scalar deltaT = Scalar(0.001);
	shared_ptr<NVEUpdater> nve_up = nve_creator(sysdef, deltaT);
	shared_ptr<NeighborList> nlist(new NeighborList(sysdef, Scalar(3.0), Scalar(0.8)));
	shared_ptr<LJForceCompute> fc(new LJForceCompute(sysdef, nlist, Scalar(3.0)));
	
	// setup some values for alpha and sigma
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(1.0);
	Scalar alpha = Scalar(1.0);
	Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma, Scalar(12.0));
	Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma, Scalar(6.0));
	
	// specify the force parameters
	fc->setParams(0,0,lj1,lj2);

	nve_up->addForceCompute(fc);
	
	unsigned int steps = 1000;
	unsigned int sampling = 100;

	sysdef->init();
		
	shared_ptr<RigidData> rdata = sysdef->getRigidData();
	unsigned int nbodies = rdata->getNumBodies();
	cout << "Number of particles = " << arrays.nparticles << "; Number of rigid bodies = " << nbodies << "\n";
	
	for (unsigned int i = 0; i < steps; i++)
		{
		if (i % sampling == 0) cout << "Step " << i << "\n";
		
		nve_up->update(i);
		}

	ArrayHandle<Scalar4> com_handle(rdata->getCOM(), access_location::host, access_mode::read);
	cout << "Rigid body center of mass:\n";
	for (unsigned int i = 0; i < nbodies; i++) 
		cout << i << "\t " << com_handle.data[i].x << "\t" << com_handle.data[i].y << "\t" << com_handle.data[i].z << "\n";

	// Output coordinates
	arrays = pdata->acquireReadWrite();
		
	FILE *fp = fopen("test_integrate.xyz", "w");
	BoxDim box = pdata->getBox();
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;
	fprintf(fp, "%d\n%f\t%f\t%f\n", arrays.nparticles, Lx, Ly, Lz);
	for (unsigned int i = 0; i < arrays.nparticles; i++) 
		fprintf(fp, "N\t%f\t%f\t%f\n", arrays.x[i], arrays.y[i], arrays.z[i]);
		
	fclose(fp);
	pdata->release();

	}


void nve_updater_energy_tests(nveup_creator nve_creator, ExecutionConfiguration exec_conf)
{
	#ifdef ENABLE_CUDA
	g_gpu_error_checking = true;
	#endif
	
	// check that the nve updater can actually integrate particle positions and velocities correctly
	// start with a 2 particle system to keep things simple: also put everything in a huge box so boundary conditions
	// don't come into play
	unsigned int nbodies = 12000;
	unsigned int nparticlesperbody = 5;
	unsigned int N = nbodies * nparticlesperbody;
	Scalar box_length = 80.0;
	shared_ptr<SystemDefinition> sysdef(new SystemDefinition(N, BoxDim(box_length), 1, 0, 0, 0, 0, exec_conf));
	shared_ptr<ParticleData> pdata = sysdef->getParticleData();
	BoxDim box = pdata->getBox();
	
	// setup a simple initial state
	unsigned int ibody = 0;
	unsigned int iparticle = 0;
	Scalar x0 = box.xlo + 0.01;
	Scalar y0 = box.ylo + 0.01;
	Scalar z0 = box.zlo + 0.01;
	Scalar xspacing = 6.0f;
	Scalar yspacing = 2.0f;
	Scalar zspacing = 2.0f;
	
	unsigned int seed = 10483;
	boost::shared_ptr<Saru> random = boost::shared_ptr<Saru>(new Saru(seed));
	Scalar temperature = 1.0;
	Scalar KE = Scalar(0.0);
	
	ParticleDataArrays arrays = pdata->acquireReadWrite();
	
	// initialize bodies in a cubic lattice with some velocity profile
	for (unsigned int i = 0; i < nbodies; i++)
	{
		for (unsigned int j = 0; j < nparticlesperbody; j++)
		{
			/*
			arrays.x[iparticle] = x0 + 1.0 * j;
                        arrays.y[iparticle] = y0 + 0.0;
                        arrays.z[iparticle] = z0 + 0.0;
			*/
			/*
			if (j == 0)
			{
				arrays.x[iparticle] = x0 + 0.0; 
				arrays.y[iparticle] = y0 + 0.0;
				arrays.z[iparticle] = z0 + 0.0;
			}
			else if (j == 1)
			{
				arrays.x[iparticle] = x0 + 1.0;
                                arrays.y[iparticle] = y0 + 0.0;
                                arrays.z[iparticle] = z0 + 0.0;

			}
			else if (j == 2)
			{
				arrays.x[iparticle] = x0 + 2.0;
                                arrays.y[iparticle] = y0 + 0.0;
                                arrays.z[iparticle] = z0 + 0.0;

			}
			else if (j == 3)
			{
				arrays.x[iparticle] = x0 + 2.76;
                                arrays.y[iparticle] = y0 + 0.642;
                                arrays.z[iparticle] = z0 + 0.0;

			}
			else if (j == 4)
			{
				arrays.x[iparticle] = x0 + 3.52;
                                arrays.y[iparticle] = y0 + 1.284;
                                arrays.z[iparticle] = z0 + 0.0;

			}
			*/

			
			if (j == 0)
                        {
                                arrays.x[iparticle] = x0 + 0.577;
                                arrays.y[iparticle] = y0 + 0.577;
                                arrays.z[iparticle] = z0 + 0.577;
                        }               
                        else if (j == 1)
                        {
                                arrays.x[iparticle] = x0 + 1.154;  
                                arrays.y[iparticle] = y0 + 1.154;
                                arrays.z[iparticle] = z0 + 1.154;

                        } 
                        else if (j == 2)
                        {
                                arrays.x[iparticle] = x0 + 0.0;
                                arrays.y[iparticle] = y0 + 0.0;
                                arrays.z[iparticle] = z0 + 1.154;

                        }
                        else if (j == 3)
                        {
                                arrays.x[iparticle] = x0 + 0.0;
                                arrays.y[iparticle] = y0 + 1.154;
                                arrays.z[iparticle] = z0 + 0.0;

                        }           
                        else if (j == 4)
                        {
                                arrays.x[iparticle] = x0 + 1.154;
                                arrays.y[iparticle] = y0 + 0.0;
				arrays.z[iparticle] = z0 + 0.0;
			}


			arrays.vx[iparticle] = random->d(); 
			arrays.vy[iparticle] = random->d();  
			arrays.vz[iparticle] = random->d();  
			
			KE += Scalar(0.5) * (arrays.vx[iparticle]*arrays.vx[iparticle] + arrays.vy[iparticle]*arrays.vy[iparticle] + arrays.vz[iparticle]*arrays.vz[iparticle]);
			
			arrays.body[iparticle] = ibody;
						
			iparticle++;
		}
		
		x0 += xspacing;
		if (x0 + xspacing >= box.xhi) 
		{
			x0 = box.xlo + 2.5;
		
			y0 += yspacing;
			if (y0 + yspacing >= box.yhi) 
			{
				y0 = box.ylo + 2.5;
				
				z0 += zspacing;
				if (z0 + zspacing >= box.zhi) 
					z0 = box.zlo + 2.5;
			}
		}
		
		ibody++;
	}
	
	assert(iparticle == N);
	
	pdata->release();

	Scalar deltaT = Scalar(0.001);
	shared_ptr<NVEUpdater> nve_up = nve_creator(sysdef, deltaT);
	shared_ptr<BinnedNeighborListGPU> nlist(new BinnedNeighborListGPU(sysdef, Scalar(2.5), Scalar(0.3)));
	shared_ptr<LJForceComputeGPU> fc(new LJForceComputeGPU(sysdef, nlist, Scalar(2.5)));
	
	// setup some values for alpha and sigma
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(1.0);
	Scalar alpha = Scalar(1.0);
	Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma, Scalar(12.0));
	Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma, Scalar(6.0));
	
	// specify the force parameters
	fc->setParams(0,0,lj1,lj2);
	
	nve_up->addForceCompute(fc);
	
	sysdef->init();

	Scalar PE;
	unsigned int steps = 100000;
	unsigned int sampling = 10000;
	
	shared_ptr<RigidData> rdata = sysdef->getRigidData();
	unsigned int nrigid_dof = rdata->getNumDOF();
	
	// Rescale particle velocities to desired temperature:
	Scalar current_temp = 2.0 * KE / nrigid_dof;
	Scalar factor = sqrt(temperature / current_temp);
	
	arrays = pdata->acquireReadWrite();
	for (unsigned int j = 0; j < N; j++) 
	{
		arrays.vx[j] *= factor;
		arrays.vy[j] *= factor;
		arrays.vz[j] *= factor;
	}
	
	pdata->release();
	
	cout << "Number of particles = " << N << "; Number of rigid bodies = " << rdata->getNumBodies() << "\n";
	cout << "Step\tTemp\tPotEng\tKinEng\tTotalE\n";
		
	clock_t start = clock();

	for (unsigned int i = 0; i <= steps; i++)
		{

		nve_up->update(i);
			
		if (i % sampling == 0) 
			{			
			arrays = pdata->acquireReadWrite();
			KE = Scalar(0.0);
			for (unsigned int j = 0; j < N; j++) 
				KE += Scalar(0.5) * (arrays.vx[j]*arrays.vx[j] + arrays.vy[j]*arrays.vy[j] + arrays.vz[j]*arrays.vz[j]);
			PE = fc->calcEnergySum();
				
			current_temp = 2.0 * KE / nrigid_dof;
			printf("%8d\t%12.8g\t%12.8g\t%12.8g\t%12.8g\n", i, current_temp, PE / N, KE / N, (PE + KE) / N); 	
			
			pdata->release();
			}
		}
	
	clock_t end = clock();
	double elapsed = (double)(end - start) / (double)CLOCKS_PER_SEC;	
	printf("Elapased time: %f sec or %f TPS\n", elapsed, (double)steps / elapsed);

	// Output coordinates
	arrays = pdata->acquireReadWrite();
	
	FILE *fp = fopen("test_energy.xyz", "w");
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;
	fprintf(fp, "%d\n%f\t%f\t%f\n", arrays.nparticles, Lx, Ly, Lz);
	for (unsigned int i = 0; i < arrays.nparticles; i++) 
		fprintf(fp, "N\t%f\t%f\t%f\n", arrays.x[i], arrays.y[i], arrays.z[i]);
	
	fclose(fp);
	pdata->release();

	}

//! NVEUpdater factory for the unit tests
shared_ptr<NVEUpdater> base_class_nve_creator(shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
	{
	return shared_ptr<NVEUpdater>(new NVEUpdater(sysdef, deltaT));
	}

#ifdef ENABLE_CUDA
//! NVEUpdaterGPU factory for the unit tests
shared_ptr<NVEUpdater> gpu_nve_creator(shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
{
	return shared_ptr<NVEUpdater>(new NVEUpdaterGPU(sysdef, deltaT));
}
#endif
/*
//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( NVEUpdater_integrate_tests )
	{
	printf("\nTesting integration on CPU...\n");
	nveup_creator nve_creator = bind(base_class_nve_creator, _1, _2);
	nve_updater_integrate_tests(nve_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	}
*/
/*
BOOST_AUTO_TEST_CASE( NVEUpdater_energy_tests )
{
	printf("\nTesting energy conservation on CPU...\n");
	nveup_creator nve_creator = bind(base_class_nve_creator, _1, _2);
	nve_updater_energy_tests(nve_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
}
*/
#ifdef ENABLE_CUDA
/*
//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( NVEUpdaterGPU_integrate_tests )
{
	printf("\nTesting integration on GPU...\n");
	nveup_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
	nve_updater_integrate_tests(nve_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU, ExecutionConfiguration::getDefaultGPU()));
}
*/

//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( NVEUpdaterGPU_energy_tests )
{
	printf("\nTesting energy conservation on GPU...\n");
	nveup_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
	nve_updater_energy_tests(nve_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU, ExecutionConfiguration::getDefaultGPU()));
}

#endif

#ifdef WIN32
#pragma warning( pop )
#endif
