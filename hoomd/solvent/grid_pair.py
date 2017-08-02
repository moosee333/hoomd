# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: vramasub

R""" Grid pair potentials.

These potentials are modeled on the MD pair potentials, but are instead calculated
as the effective potential experienced on a grid due to existing particles. As a
result, the computations do not operate on particle pairs, but rather particle-grid
point pairs.

Generally, pair forces are short range and are summed over all non-bonded particles
within a certain cutoff radius of each grid point. Any number of pair forces
can be defined in a single simulation. The net force on each grid point due to
all types of grid forces is summed.

Grid pair forces require that parameters be set for each unique particle type in
the simulation. Coefficients are set through the aid of the :py:class:`coeff`
class. To set these coefficients, specify a pair force and save it in a variable::

    my_force = grid_pair.some_pair_force(arguments...)

Then the coefficients can be set using the saved variable::

    my_force.grid_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    my_force.grid_coeff.set('A', 'B', epsilon=1.0, sigma=2.0)
    my_force.grid_coeff.set('B', 'B', epsilon=2.0, sigma=1.0)

This example set the parameters *epsilon* and *sigma*
(which are used in :py:class:`lj`). Different pair forces require that different
coefficients are set. Check the documentation of each to see the definition
of the coefficients.
"""

from hoomd import _hoomd
import hoomd
from hoomd.solvent import grid_force
from hoomd.solvent import _solvent

import math
import sys

class coeff(object):
    R""" Define potential coefficients.

    The coefficients for all potentials are specified using this class. Coefficients are
    specified per particle type.

    There are two ways to set the coefficients for a particular potential.
    The first way is to save the potential in a variable and call :py:meth:`set()` directly.
    See below for an example of this.

    The second method is to build the coeff class first and then assign it to the
    potential. There are some advantages to this method in that you could specify a
    complicated set of potential coefficients in a separate python file and import
    it into your job script.

    Example::

        my_coeffs = hoomd.md.bond.coeff();
        my_force.grid_coeff.set('polymer', k=330.0, r=0.84)
        my_force.grid_coeff.set('backbone', k=330.0, r=0.84)

    """

    ## \internal
    # \brief Initializes the class
    # \details
    # The main task to be performed during initialization is just to init some variables
    # \param self Python required class instance variable
    def __init__(self):
        self.values = {};
        self.default_coeff = {}

    ## \var values
    # \internal
    # \brief Contains the vector of set values in a dictionary

    ## \var default_coeff
    # \internal
    # \brief default_coeff['coeff'] lists the default value for \a coeff, if it is set

    ## \internal
    # \brief Sets a default value for a given coefficient
    # \details
    # \param name Name of the coefficient to for which to set the default
    # \param value Default value to set
    #
    # Some coefficients have reasonable default values and the user should not be burdened with typing them in
    # all the time. set_default_coeff() sets
    def set_default_coeff(self, name, value):
        self.default_coeff[name] = value;

    def set(self, type, **coeffs):
        R""" Sets parameters for bond types.

        Args:
            type (str): Type of particle (or a list of type names)
            coeffs: Named coefficients (see below for examples)

        Calling :py:meth:`set()` results in one or more parameters being set for a particle type. Types are identified
        by name, and parameters are also added by name. Which parameters you need to specify depends on the particle
        potential you are setting these coefficients for, see the corresponding documentation.

        All possible particle types as defined in the simulation box must be specified before executing run().
        You will receive an error if you fail to do so. It is not an error, however, to specify coefficients for
        particle types that do not exist in the simulation. This can be useful in defining a potential field for many
        different types of particle even when some simulations only include a subset.

        Examples::

            my_force.grid_coeff.set('polymer', k=330.0, r0=0.84)
            my_force.grid_coeff.set('backbone', k=1000.0, r0=1.0)
            my_force.grid_coeff.set(['bondA','bondB'], k=100, r0=0.0)

        Note:
            Single parameters can be updated. If both ``k`` and ``r0`` have already been set for a particle type,
            then executing ``coeff.set('polymer', r0=1.0)`` will update the value of ``r0`` and leave the other
            parameters as they were previously set.

        """
        hoomd.util.print_status_line();

        # listify the input
        if isinstance(type, str):
            type = [type];

        for typei in type:
            self.set_single(typei, coeffs);

    ## \internal
    # \brief Sets a single parameter
    def set_single(self, type, coeffs):
        type = str(type);

        # create the type identifier if it hasn't been created yet
        if (not type in self.values):
            self.values[type] = {};

        # update each of the values provided
        if not coeffs:
            hoomd.context.msg.error("No coefficents specified\n");
        for name, val in coeffs.items():
            self.values[type][name] = val;

        # set the default values
        for name, val in self.default_coeff.items():
            # don't override a coeff if it is already set
            if not name in self.values[type]:
                self.values[type][name] = val;

    ## \internal
    # \brief Verifies that all values are set
    # \details
    # \param self Python required self variable
    # \param required_coeffs list of required variables
    #
    # This can only be run after the system has been initialized
    def verify(self, required_coeffs):
        # first, check that the system has been initialized
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot verify bond coefficients before initialization\n");
            raise RuntimeError('Error verifying force coefficients');

        # get a list of types from the particle data
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        valid = True;
        # loop over all possible types and verify that all required variables are set
        for i in range(0,ntypes):
            type = type_list[i];

            if type not in self.values.keys():
                hoomd.context.msg.error("No coeffs found for particle type " + str(type) + "\n");
                valid = False;
                continue;

            # verify that all required values are set by counting the matches
            coeffs_missing = required_coeffs.copy()
            for coeff_name in self.values[type].keys():
                if not coeff_name in required_coeffs:
                    hoomd.context.msg.notice(2, "Notice: Possible typo? Coefficient " + str(coeff_name) + " is specified for particle type " + str(type) + \
                          ", but is not used by the force.\n");
                else:
                    coeffs_missing.remove(coeff_name)

            if coeffs_missing:
                hoomd.context.msg.error("Particle type " + str(type) + " is missing the following coefficients: {}\n".format(", ".join(coeffs_missing)))
                valid = False;

        return valid;

    ## \internal
    # \brief Gets the value of a single %bond %force coefficient
    # \detail
    # \param type Name of bond type
    # \param coeff_name Coefficient to get
    def get(self, type, coeff_name):
        if type not in self.values.keys():
            hoomd.context.msg.error("Bug detected in grid_force.coeff. Please report\n");
            raise RuntimeError("Error setting bond coeff");

        coeff_val = self.values[type].get(coeff_name)

        if not coeff_val:
            raise RuntimeError("Bug detected, coefficient {coeff} not set for particle type {type}".format(coeff = coeff_name, type = type))
        else:
            return coeff_val

    ## \internal
    # \brief Return metadata
    def get_metadata(self):
        return self.values

## \internal
# \brief Base class for grid pair potentials
#
# A grid_pair in hoomd reflects a GridPotentialPair in C++. It is responsible
# for all high-level management that happens behind the scenes for hoomd
# potentials, including:
#   1) The instance of the c++ bond force itself is tracked and added to the system 
#   2) methods are provided for disabling the force from being added to the net force on each particle
class _grid_pair(grid_force._grid_force):
    ## \internal
    # \brief Constructs the grid pair potential
    #
    # \param name name of the grid pair potential instance
    #
    # Initializes the cpp_force to None.
    # If specified, assigns a name to the instance
    # Assigns a name to the force in force_name;
    def __init__(self, nlist):
        # initialize the base class
        super(_grid_pair, self).__init__();

        self.cpp_force = None;
        self.nlist = nlist;
        #self.nlist.subscribe(lambda:self.get_rcut())
        #self.nlist.update_rcut()

        # setup the coefficient vector
        self.grid_coeff = coeff();

        self.enabled = True;

    def update_coeffs(self):
        coeff_list = self.required_coeffs
        # check that the force coefficients are valid
        if not self.grid_coeff.verify(coeff_list):
           hoomd.context.msg.error("Not all force coefficients are set\n");
           raise RuntimeError("Error updating force coefficients");

        # set all the params
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            coeff_dict = {};
            typ = hoomd.context.current.system_definition.getParticleData().getNameByType(i)
            for name in coeff_list:
                coeff_dict[name] = self.grid_coeff.get(typ, name);
            param = self.process_coeff(coeff_dict);
            self.cpp_force.setParams(i, param);

    def process_coeff(coeffs):
        raise NotImplementedError("The process_coeff function must be implemented in each grid pair potential")

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = grid_force._force.get_metadata(self)

        # make sure coefficients are up-to-date
        self.update_coeffs()

        data['grid_coeff'] = self.grid_coeff
        return data

    def take_snapshot(self, dtype='float'):
        R""" Take a snapshot of the current grid

        Returns:
            The snapshot object.

        Examples::

            snapshot = grid.take_snapshot()

        """
        hoomd.util.print_status_line();

        # take the snapshot
        if dtype == 'float':
            cpp_snapshot = self.cpp_force.getSnapshot()
        elif dtype == 'double':
            cpp_snapshot = self.cpp_force.getSnapshot()
        else:
            raise ValueError("dtype must be float or double");

        return cpp_snapshot

class lj(_grid_pair):
    R""" Lennard-Jones grid pair potential.

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        name (str): Name of the force instance.

    :py:class:`lj` specifies that a Lennard-Jones pair potential should be applied between every
    non-excluded particle type in the simulation and the grid.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{LJ}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
                          \alpha \left( \frac{\sigma}{r} \right)^{6} \right] & r < r_{\mathrm{cut}} \\
                            = & 0 & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    See :py:class:`pair` for details on how forces are calculated and the available energy shifting and smoothing modes.
    Use :py:meth:`pair_coeff.set <coeff.set>` to set potential coefficients.

    The following coefficients must be set per unique particle type:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`\alpha` - *alpha* (unitless) - *optional*: defaults to 1.0
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command
    - :math:`r_{\mathrm{on}}`- *r_on* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    Example::

        nl = nlist.cell()
        lj = grid_pair.lj(r_cut=3.0, nlist=nl)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        lj.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, alpha=0.5, r_cut=3.0, r_on=2.0);
        lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, r_cut=2**(1.0/6.0), r_on=2.0);
        lj.pair_coeff.set(['A', 'B'], ['C', 'D'], epsilon=1.5, sigma=2.0)

    """
    def __init__(self, r_cut, nlist):
        hoomd.util.print_status_line();

        # initialize the base class
        super(lj, self).__init__(nlist);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _solvent.GridPotentialPairLJ(hoomd.context.current.system_definition, self.nlist.cpp_cl);
            self.cpp_class = _solvent.GridPotentialPairLJ;
        else:
            raise NotImplementedError("Grid pair potentials are not yet GPU enabled!")

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'alpha'];
        self.grid_coeff.set_default_coeff('alpha', 1.0);

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        alpha = coeff['alpha'];

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0);
        return _hoomd.make_scalar2(lj1, lj2);
