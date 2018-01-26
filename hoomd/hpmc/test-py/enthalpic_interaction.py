from __future__ import division
from __future__ import print_function

import hoomd
from hoomd import context, data, init, analyze
from hoomd import hpmc, jit

import unittest
import os
import numpy as np
import itertools
import sys

context.initialize();

class enthalpic_dipole_interaction(unittest.TestCase):
        def setUp(self):
            self.lamb = 1      # Dipolar coupling constant: mu0*m^2/ (2 * pi * d^3 * K * T)
            self.r_cut = 15;   # interaction cut-off
            self.diameter = 1; # particles diameter
            # interaction between point dipoles
            self.dipole_dipole = """ float rsq = dot(r_ij, r_ij);
                                float r_cut = {0};
                                if (rsq < r_cut*r_cut)
                                    {{
                                    float lambda = {1};
                                    float r = fast::sqrt(rsq);
                                    float r3inv = 1.0 / rsq / r;
                                    vec3<float> t = r_ij / r;
                                    vec3<float> pi_o(1,0,0);
                                    vec3<float> pj_o(1,0,0);
                                    rotmat3<float> Ai(q_i);
                                    rotmat3<float> Aj(q_j);
                                    vec3<float> pi = Ai * pi_o;
                                    vec3<float> pj = Aj * pj_o;
                                    float Udd = (lambda*r3inv/2)*( dot(pi,pj)
                                                 - 3 * dot(pi,t) * dot(pj,t));
                                    return Udd;
                                   }}
                                else
                                    return 0.0f;
                            """.format(self.r_cut,self.lamb);
            self.snapshot = data.make_snapshot(N=2, box=data.boxdim(L=2*self.r_cut, dimensions=3), particle_types=['A']);

        # a) particle 1 on top of particle 2: parallel dipole orientations
        def test_head_to_tail_parallel(self):
            self.snapshot.particles.position[0,:]    = (0,0,0);
            self.snapshot.particles.position[1,:]    = (self.diameter,0,0);
            self.snapshot.particles.orientation[0,:] = (1,0,0,0);
            self.snapshot.particles.orientation[1,:] = (1,0,0,0);
            init.read_snapshot(self.snapshot);
            self.mc = hpmc.integrate.sphere(seed=10,a=0,d=0);
            self.mc.shape_param.set('A', diameter=self.diameter,orientable=True);
            self.patch = jit.patch.user(mc=self.mc,r_cut=self.r_cut, code=self.dipole_dipole);
            self.log = analyze.log(filename=None, quantities=['hpmc_patch_energy'],period=0,overwrite=True)
            hoomd.run(0, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), -self.lamb);

        # b) particle 1 on top of particle 2: antiparallel dipole orientations
        def test_head_to_tail_antiparallel(self):
            self.snapshot.particles.position[0,:]    = (0,0,0);
            self.snapshot.particles.position[1,:]    = (self.diameter,0,0);
            self.snapshot.particles.orientation[0,:] = (1,0,0,0);
            self.snapshot.particles.orientation[1,:] = (0,0,1,0);
            init.read_snapshot(self.snapshot);
            self.mc = hpmc.integrate.sphere(seed=10,a=0,d=0);
            self.mc.shape_param.set('A', diameter=self.diameter,orientable=True);
            self.patch = jit.patch.user(mc=self.mc,r_cut=self.r_cut, code=self.dipole_dipole);
            self.log = analyze.log(filename=None, quantities=['hpmc_patch_energy'],period=0,overwrite=True)
            hoomd.run(0, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), self.lamb);

        # c) particles side by side: parallel dipole orientations
        def test_side_to_side_parallel(self):
            self.snapshot.particles.position[0,:]    = (0,0,0);
            self.snapshot.particles.position[1,:]    = (0,0,self.diameter);
            self.snapshot.particles.orientation[0,:] = (1,0,0,0);
            self.snapshot.particles.orientation[1,:] = (1,0,0,0);
            init.read_snapshot(self.snapshot);
            self.mc = hpmc.integrate.sphere(seed=10,a=0,d=0);
            self.mc.shape_param.set('A', diameter=self.diameter,orientable=True);
            self.patch = jit.patch.user(mc=self.mc,r_cut=self.r_cut, code=self.dipole_dipole);
            self.log = analyze.log(filename=None, quantities=['hpmc_patch_energy'],period=0,overwrite=True)
            hoomd.run(0, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), self.lamb/2);

        # d) particles side by side: antiparallel dipole orientations
        def test_side_to_side_parallel(self):
           self.snapshot.particles.position[0,:]    = (0,0,0);
           self.snapshot.particles.position[1,:]    = (0,0,self.diameter);
           self.snapshot.particles.orientation[0,:] = (1,0,0,0);
           self.snapshot.particles.orientation[1,:] = (0,0,1,0);
           init.read_snapshot(self.snapshot);
           self.mc = hpmc.integrate.sphere(seed=10,a=0,d=0);
           self.mc.shape_param.set('A', diameter=self.diameter,orientable=True);
           self.patch = jit.patch.user(mc=self.mc,r_cut=self.r_cut, code=self.dipole_dipole);
           self.log = analyze.log(filename=None, quantities=['hpmc_patch_energy'],period=0,overwrite=True)
           hoomd.run(0, quiet=True);
           self.assertEqual(self.log.query('hpmc_patch_energy'), -self.lamb/2);

        def tearDown(self):
            del self.mc;
            del self.patch;
            del self.snapshot;
            del self.log;
            context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
