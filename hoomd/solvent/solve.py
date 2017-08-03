# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: vramasub / All Developers are free to add commands for new features

R""" Level set solver for solvent potential
"""

from hoomd import _hoomd
from hoomd.solvent import _solvent
import sys
import hoomd

## \internal
# \brief Level set solver class 
#
# An instance of ls_solver reflect a LevelSetSolver in C++. It is responsible
# for using grid information and applying the requisite algorithms for
# computing the resulting solvent interface. Note that the underlying
# implementation is as a ForceCompute, because ultimately the solver serves
# to compute the interfacial forces on the system. As a result, the
# implementation here closely parallels that of md._force
class ls_solver(hoomd.meta._metadata):
    # default counter
    cur_id = 0

    def __init__(self, grid, name = None):
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot create ls_solver before initialization\n")
            raise RuntimeError('Error creating ls_solver')

        # Instantiate underlying objects
        self.grid = grid
        self.cpp_force = _solvent.LevelSetSolver(hoomd.context.current.system_definition, self.grid.cpp_grid)
        self.cpp_class = _solvent.LevelSetSolver

        # Allow force to store a name.  Used for discombobulation in the logger
        if name is None:
            self.name = ""
        else:
            self.name = "_" + name

        obj_id = ls_solver.cur_id
        ls_solver.cur_id += 1

        self.force_name = "grid_force%d" % (obj_id)
        self.enabled = True
        self.log = True
        hoomd.context.current.forces.append(self)

        # base class constructor
        hoomd.meta._metadata.__init__(self)

    ## \internal
    # \brief Checks that proper initialization has completed
    def check_initialization(self):
        # check that we have been initialized properly
        if self.cpp_force is None:
            hoomd.context.msg.error('Bug in solvent.grid_force: cpp_force not set, please report\n')
            raise RuntimeError()

    def disable(self, log=False):
        R""" Disable the force.

        Args:
            log (bool): Set to True if you plan to continue logging the potential energy associated with this force.

        Examples::

            force.disable()
            force.disable(log=True)

        Executing the disable command will remove the force from the simulation.
        Any :py:func:`hoomd.run()` command executed after disabling a force will not calculate or
        use the force during the simulation. A disabled force can be re-enabled
        with :py:meth:`enable()`.

        By setting *log* to True, the values of the force can be logged even though the forces are not applied
        in the simulation.  For forces that use cutoff radii, setting *log=True* will cause the correct *r_cut* values
        to be used throughout the simulation, and therefore possibly drive the neighbor list size larger than it
        otherwise would be. If *log* is left False, the potential energy associated with this force will not be
        available for logging.

        """
        hoomd.util.print_status_line()
        self.check_initialization()

        # check if we are already disabled
        if not self.enabled:
            hoomd.context.msg.warning("Ignoring command to disable a force that is already disabled")
            return

        self.enabled = False
        self.log = log

        # remove the compute from the system if it is not going to be logged
        if not log:
            hoomd.context.current.system.removeCompute(self.cpp_force, self.force_name)
            hoomd.context.current.forces.remove(self)

    def enable(self):
        R""" Enable the grid force.

        Examples::

            force.enable()

        See :py:meth:`disable()`.
        """
        hoomd.util.print_status_line()
        self.check_initialization()

        # check if we are already disabled
        if self.enabled:
            hoomd.context.msg.warning("Ignoring command to enable a force that is already enabled")
            return

        # add the compute back to the system if it was removed
        if not self.log:
            hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)
            hoomd.context.current.forces.append(self)

        self.enabled = True
        self.log = True

    ## \internal
    # \brief updates force coefficients
    # This is a dummy function required to allow the level set solver to coexist
    # with other HOOMD forces; the underlying grid forces have coefficients, but
    # the solver itself does not
    def update_coeffs(self):
        pass

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['enabled'] = self.enabled
        data['log'] = self.log
        if self.name is not "":
            data['name'] = self.name

        return data
