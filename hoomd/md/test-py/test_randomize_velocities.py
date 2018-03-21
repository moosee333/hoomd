# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os
import numpy

# unit tests for velocity randomization
class velocity_randomization_tests (unittest.TestCase):

    def setUp(self):
        print

        # Target a packing fraction of 0.05
        # Set an orientation and a moment of inertia for the anisotropic test
        a = 2.1878096788957757
        if '2d' not in self._testMethodName:
            self.D = 3
            self.rot_dof = 3
            uc = lattice.unitcell(N=1,
                                  a1=[a, 0, 0],
                                  a2=[0, a, 0],
                                  a3=[0, 0, a],
                                  position=[[0, 0, 0]],
                                  type_name=['A'],
                                  mass=[1],
                                  moment_inertia=[[1, 1, 1]],
                                  orientation=[[1, 0, 0, 0]])
            n = [50, 50, 40]
        else:
            self.D = 2
            self.rot_dof = 1
            uc = lattice.unitcell(N=1,
                                  a1=[a, 0, 0],
                                  a2=[0, a, 0],
                                  a3=[0, 0, 1],
                                  dimensions=2,
                                  position=[[0, 0, 0]],
                                  type_name=['A'],
                                  mass=[1],
                                  moment_inertia=[[0, 0, 1]],
                                  orientation=[[1, 0, 0, 0]])
            n = 316

        # The system has to be reasonably large (~100k particles in brief
        # tests) to ensure that the measured temperature is approximately
        # equal to the desired temperature after removing the center of mass
        # momentum (eliminating drift)
        self.system = init.create_lattice(unitcell=uc, n=n)

        # Set one particle to be really massive for validation
        self.system.particles[0].mass = 10000

        md.integrate.mode_standard(dt=0.)
        self.all = group.all()
        self.aniso = False
        self.quantities = ['N',
                           'kinetic_energy',
                           'translational_kinetic_energy',
                           'rotational_kinetic_energy',
                           'temperature',
                           'momentum']
        self.log = analyze.log(filename=None, quantities=self.quantities, period=None)

    def aniso_prep(self):
        # Sets up an anisotropic pair potential, so that rotational degrees of
        # freedom are given some energy
        nlist = md.nlist.cell()
        dipole = md.pair.dipole(r_cut=2, nlist=nlist)
        dipole.pair_coeff.set('A', 'A', mu=0.0, A=1.0, kappa=1.0)
        self.aniso = True

    def check_quantities(self):
        N = self.log.query('N')
        self.assertAlmostEqual(self.log.query('temperature'), self.kT, 2)
        self.assertAlmostEqual(self.log.query('momentum'), 0, 6)

        avg_trans_KE = self.log.query('translational_kinetic_energy') / N
        # We expect D * (1/2 kT) translational energy per particle
        self.assertAlmostEqual(avg_trans_KE, 0.5*self.D*self.kT, 2)

        if self.aniso:
            avg_rot_KE = self.log.query('rotational_kinetic_energy') / N
            # We expect rot_dof * (1/2 kT) rotational energy per particle
            self.assertAlmostEqual(avg_rot_KE, 0.5*self.rot_dof*self.kT, 2)

    def test_nvt(self):
        self.kT = 1.0
        integrator = md.integrate.nvt(group=self.all, kT=self.kT, tau=0.5)
        integrator.randomize_velocities(seed=42)
        run(1)
        self.check_quantities()

    def test_berendsen(self):
        if comm.get_num_ranks() == 1:
            self.kT = 1.0
            integrator = md.integrate.berendsen(group=self.all, kT=self.kT, tau=0.5)
            integrator.randomize_velocities(seed=42)
            run(1)
            self.check_quantities()
        else:
            # We can only run the berendsen thermostat if we have one rank.
            # Ignore this test on MPI with more than one rank.
            pass

    def test_npt(self):
        self.kT = 1.0
        integrator = md.integrate.npt(group=self.all, kT=self.kT, tau=0.5, tauP=1.0, P=2.0)
        integrator.randomize_velocities(seed=42)
        run(1)
        self.check_quantities()

    def test_nph(self):
        self.kT = 1.0
        integrator = md.integrate.nph(group=self.all, P=2.0, tauP=1.0)
        integrator.randomize_velocities(kT=self.kT, seed=42)
        run(1)
        self.check_quantities()

    def test_nve(self):
        self.kT = 1.0
        integrator = md.integrate.nve(group=self.all)
        integrator.randomize_velocities(kT=self.kT, seed=42)
        run(1)
        self.check_quantities()

    def test_nvt_2d(self):
        self.kT = 1.0
        integrator = md.integrate.nvt(group=self.all, kT=self.kT, tau=0.5)
        integrator.randomize_velocities(seed=42)
        run(1)
        self.check_quantities()

    def test_nvt_anisotropic(self):
        self.aniso_prep()
        self.kT = 1.0
        integrator = md.integrate.nvt(group=self.all, kT=self.kT, tau=0.5)
        integrator.randomize_velocities(seed=42)
        run(1)
        self.check_quantities()

    def test_nvt_anisotropic_2d(self):
        self.aniso_prep()
        self.kT = 1.0
        integrator = md.integrate.nvt(group=self.all, kT=self.kT, tau=0.5)
        integrator.randomize_velocities(seed=42)
        run(1)
        self.check_quantities()

    def tearDown(self):
        context.initialize()


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
