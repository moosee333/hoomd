# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: vramasub

R""" Grid objects

"""

from hoomd import _hoomd
import hoomd
from hoomd.solvent import _solvent

import math
import sys

class grid(hoomd.meta._metadata):
    R""" Grid object

    Args:
        sigma (float): The grid spacing

    :py:class:`grid` represents a grid that can be utilized by the ls_solver class to
    find solutions of the variational implicit solvent model.

    Grids are reflections of underlying GridData classes. The purpose of the grids is
    to provide the core data structure on which the rest of the variational implicit
    solvent model is build. Snapshots of grids are available for direct viewing. Both the energies and the
    forces on the grid can be accessed through these snapshots

    """
    def __init__(self, sigma):
        hoomd.util.print_status_line();

        self.sigma = sigma

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_grid = _solvent.GridData(hoomd.context.current.system_definition, self.sigma);
            self.cpp_class = _solvent.GridData;
        else:
            raise NotImplementedError("Grid pair potentials are not yet GPU enabled!")

    def set_sigma(self, sigma):
        R""" Sets the grid spacing values

        Examples::

            g = grid(0.01)
            grid.set_sigma(0.02)
        """
        self.sigma = sigma
        self.cpp_grid.setSigma(self.sigma)
