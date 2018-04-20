# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
import hoomd;
context.initialize()
import unittest
import os

# tests for update.box_resize
class update_box_resize_tests (unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=100, sphere=data.spheredim(R=20,dimensions=3),particle_types=['A'])
        self.s = init.read_snapshot(snap)

    # tests basic creation of the updater
    def test(self):
        update.sphere_resize(R = variant.linear_interp([(0, 20), (99, 50)]))
        run(100);
        self.assertAlmostEqual(self.s.sphere.R,50)

    # tests with phase
    def test_phase(self):
        update.sphere_resize(R = variant.linear_interp([(0, 20), (1e6, 50)]), period=10, phase=0)
        run(100);

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
