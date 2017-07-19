# -*- coding: iso-8859-1 -*-
# this file exists to mark this directory as a python module

R""" Implicit solvation

The solvent module computes equilibrium solute-solvent interfaces using a variational approach.

.. rubric:: Overview

The solvent package allows to compute the properties of particles in an implicit
solvent, which is represented by the equilibrium interface it has with the solute.
The solvent is represented by a surface tension, and an energy function, such as LJ,
between the continuum solvent and the solute centers.

Every HOOMD time step, the solvation free energy is minimized, and the particle forces
and free energy are computed from the minimized configuration. The minimization
proceeds iteratively.

.. rubric:: Logging

.. rubric:: Stability

:py:mod:`hoomd.hpmc` is **stable**. When upgrading from version 2.x to 2.y (y > x),
existing job scripts that follow *documented* interfaces for functions and classes
will not require any modifications.

**Maintainer:** Jens Glaser and Vyas Ramasubramani
"""

# need to import all submodules defined in this directory
from hoomd.solvent import pair

import hoomd
