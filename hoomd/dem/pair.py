# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R"""DEM pair potentials.
"""

import hoomd;
import hoomd.md;
import hoomd.md.nlist as nl;

from math import sqrt;

from hoomd.dem import _dem;
from hoomd.dem import params;
from hoomd.dem import utils;

class _DEMBase:
    def __init__(self, nlist):
        self.nlist = nlist;
        self.nlist.subscribe(self.get_rcut);
        self.nlist.update_rcut();

    def _initialize_types(self):
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        if self.dimensions == 2:
            for typ in type_list:
                self.setParams2D(typ, [[0, 0]], False);
        else:
            for typ in type_list:
                self.setParams3D(typ, [[0, 0, 0]], [], False);

    def setParams2D(self, type, vertices, center=False):
        """Set the vertices for a given particle type.

        Args:
            type (str): Name of the type to set the shape of
            vertices (list): List of (2D) points specifying the coordinates of the shape
            center (bool): If True, subtract the center of mass of the shape from the vertices before setting them for the shape

        Shapes are specified as a list of 2D coordinates. Edges will
        be made between all adjacent pairs of vertices, including one
        between the last and first vertex.
        """
        itype = hoomd.context.current.system_definition.getParticleData().getTypeByName(type);

        if not len(vertices):
            vertices = [(0, 0)];
            center = False;

        # explicitly turn into a list of tuples
        if center:
            vertices = [(float(p[0]), float(p[1])) for p in utils.center(vertices)];
        else:
            vertices = [(float(p[0]), float(p[1])) for p in vertices];

        # update the neighbor list
        rcutmax = 2*(sqrt(max(x*x + y*y for (x, y) in vertices)) + self.radius*2**(1./6));
        self.r_cut = max(self.r_cut, rcutmax);

        self.vertices[type] = vertices;
        self.cpp_force.setRcut(self.r_cut);
        self.cpp_force.setParams(itype, vertices);

    def setParams3D(self, type, vertices, faces, center=False):
        """Set the vertices for a given particle type.

        Args:
            type (str): Name of the type to set the shape of
            vertices (list): List of (3D) points specifying the coordinates of the shape
            faces (list): List of lists of indices specifying which coordinates comprise each face of a shape.
            center (bool): If True, subtract the center of mass of the shape from the vertices before setting them for the shape

        Shapes are specified as a list of coordinates (`vertices`) and
        another list containing one list for each polygonal face
        (`faces`). The elements of each list inside `faces` are
        integer indices specifying which vertex in `vertices` comprise
        the face.
        """
        itype = hoomd.context.current.system_definition.getParticleData().getTypeByName(type);

        if not len(vertices):
            vertices = [(0, 0, 0)];
            faces = [];
            center = False;

        # explicitly turn into python lists
        if center:
            vertices = [(float(p[0]), float(p[1]), float(p[2])) for p in utils.center(vertices, faces)];
        else:
            vertices = [(float(p[0]), float(p[1]), float(p[2])) for p in vertices];
        faces = [[int(i) for i in face] for face in faces];

        # update the neighbor list
        rcutmax = 2*(sqrt(max(x*x + y*y + z*z for (x, y, z) in vertices)) + self.radius*2**(1./6));
        self.r_cut = max(self.r_cut, rcutmax);

        self.vertices[type] = vertices;
        self.cpp_force.setRcut(self.r_cut);
        self.cpp_force.setParams(itype, vertices, faces);

    def get_type_shapes(self):
        """Returns a list of shape descriptions with one element for each
        unique particle type in the system. Currently assumes that all
        3D shapes are convex.
        """
        result = []

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();

        for i in range(ntypes):
            typename = hoomd.context.current.system_definition.getParticleData().getNameByType(i);
            shape = self.vertices[typename]
            if self.dimensions == 2:
                if len(shape) < 2:
                    result.append(dict(type='Disk',
                                       diameter=2*self.radius))
                else:
                    result.append(dict(type='Polygon',
                                       rounding_radius=self.radius,
                                       vertices=list(shape)))
            else:
                if len(shape) < 2:
                    result.append(dict(type='Sphere',
                                       diameter=2*self.radius))
                else:
                    result.append(dict(type='ConvexPolyhedron',
                                       rounding_radius=self.radius,
                                       vertices=list(shape)))

        return result

class WCA(hoomd.md.force._force, _DEMBase):
    R"""Specify a purely repulsive Weeks-Chandler-Andersen DEM force with a constant rounding radius.

    Args:
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list to use
        radius (float): Rounding radius :math:`r` to apply to the shape vertices

    The effect is as if a :py:class:`hoomd.md.pair.lj` interaction
    with :math:`r_{cut}=2^{1/6}\sigma` and :math:`\sigma=2\cdot r`
    were applied between the contact points of each pair of particles.

    Examples::

        # 2D system of squares
        squares = hoomd.dem.pair.WCA(radius=.5)
        squares.setParams('A', [[1, 1], [-1, 1], [-1, -1], [1, -1]])
        # 3D system of rounded square plates
        squarePlates = hoomd.dem.pair.WCA(radius=.5)
        squarePlates.setParams('A',
            vertices=[[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]],
            faces=[[0, 1, 2, 3]], center=False)
        # 3D system of some convex shape specified by vertices
        (vertices, faces) = hoomd.dem.utils.convexHull(vertices)
        shapes = hoomd.dem.pair.WCA(radius=.5)
        shapes.setParams('A', vertices=vertices, faces=faces)

    """

    def __init__(self, nlist, radius=1.):
        hoomd.util.print_status_line();
        friction = None;

        self.radius = radius;
        self.autotunerEnabled = True;
        self.autotunerPeriod = 100000;
        self.vertices = {};

        self.onGPU = hoomd.context.exec_conf.isCUDAEnabled();
        cppForces = {(2, None, 'cpu'): _dem.WCADEM2D,
             (2, None, 'gpu'): (_dem.WCADEM2DGPU if self.onGPU else None),
             (3, None, 'cpu'): _dem.WCADEM3D,
             (3, None, 'gpu'): (_dem.WCADEM3DGPU if self.onGPU else None)};

        self.dimensions = hoomd.context.current.system_definition.getNDimensions();

        # initialize the base class
        hoomd.md.force._force.__init__(self);

        # interparticle cutoff radius, will be updated as shapes are added
        self.r_cut = 2*radius*2**(1./6);

        if friction is None:
            potentialParams = params.WCA(radius=radius);
        else:
            raise RuntimeError('Unknown friction type: {}'.format(friction));

        _DEMBase.__init__(self, nlist);

        key = (self.dimensions, friction, 'gpu' if self.onGPU else 'cpu');
        cpp_force = cppForces[key];

        self.cpp_force = cpp_force(hoomd.context.current.system_definition,
                                   self.nlist.cpp_nlist, self.r_cut,
                                   potentialParams);

        if self.dimensions == 2:
            self.setParams = self.setParams2D;
        else:
            self.setParams = self.setParams3D;

        self._initialize_types();

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

    def update_coeffs(self):
        """Noop for this potential"""
        pass

    def setAutotunerParams(self, enable=None, period=None):
        if not self.onGPU:
            return;
        if enable is not None:
            self.autotunerEnabled = enable;
        if period is not None:
            self.autotunerPeriod = period;
        self.cpp_force.setAutotunerParams(self.autotunerEnabled, self.autotunerPeriod);

    def get_rcut(self):
        # self.log is True if the force is enabled
        if not self.log:
            return None;

        # go through the list of only the active particle types in the sim
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # update the rcut by pair type
        r_cut_dict = nl.rcut();
        r_max_dict = {typ: sqrt(max(sum(p*p for p in point)
                                    for point in self.vertices[typ]))
                      for typ in self.vertices};
        for i in range(ntypes):
            for j in range(i, ntypes):
                (typei, typej) = type_list[i], type_list[j];
                r_cut_dict.set_pair(typei, typej,
                                    r_max_dict.get(typei, 0) + r_max_dict.get(typej, 0) + self.radius*2*2.0**(1./6));

        r_cut_dict.fill();

        return r_cut_dict;

class SWCA(hoomd.md.force._force, _DEMBase):
    R"""Specify a purely repulsive Weeks-Chandler-Andersen DEM force with a particle-varying rounding radius.

    Args:
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list to use
        radius (float): Unshifted rounding radius :math:`r` to apply to the shape vertices
        d_max (float): maximum rounding diameter among all particles in the system

    The SWCA potential enables simulation of particles with
    heterogeneous rounding radii. The effect is as if a
    :py:class:`hoomd.md.pair.slj` interaction with
    :math:`r_{cut}=2^{1/6}\sigma` and :math:`\sigma=2\cdot r` were
    applied between the contact points of each pair of particles.

    Examples::

        # 2D system of squares
        squares = hoomd.dem.pair.SWCA(radius=.5)
        squares.setParams('A', [[1, 1], [-1, 1], [-1, -1], [1, -1]])
        # 3D system of rounded square plates
        squarePlates = hoomd.dem.pair.SWCA(radius=.5)
        squarePlates.setParams('A',
            vertices=[[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]],
            faces=[[0, 1, 2, 3]], center=False)
        # 3D system of some convex shape specified by vertices
        (vertices, faces) = hoomd.dem.utils.convexHull(vertices)
        shapes = hoomd.dem.pair.SWCA(radius=.5)
        shapes.setParams('A', vertices=vertices, faces=faces)

    """
    def __init__(self, nlist, radius=1., d_max=None):
        hoomd.util.print_status_line();
        friction = None;

        self.radius = radius;
        self.autotunerEnabled = True;
        self.autotunerPeriod = 100000;
        self.vertices = {};

        self.onGPU = hoomd.context.exec_conf.isCUDAEnabled();
        cppForces = {(2, None, 'cpu'): _dem.SWCADEM2D,
             (2, None, 'gpu'): (_dem.SWCADEM2DGPU if self.onGPU else None),
             (3, None, 'cpu'): _dem.SWCADEM3D,
             (3, None, 'gpu'): (_dem.SWCADEM3DGPU if self.onGPU else None)};

        self.dimensions = hoomd.context.current.system_definition.getNDimensions();

        # Error out in MPI simulations
        if (hoomd._hoomd.is_MPI_available()):
            if hoomd.context.current.system_definition.getParticleData().getDomainDecomposition():
                hoomd.context.msg.error("pair.SWCA is not supported in multi-processor simulations.\n\n");
                raise RuntimeError("Error setting up pair potential.");

        # initialize the base class
        hoomd.md.force._force.__init__(self);

        # update the neighbor list
        if d_max is None :
            sysdef = hoomd.context.current.system_definition;
            self.d_max = max(x.diameter for x in hoomd.data.particle_data(sysdef.getParticleData()));
            hoomd.context.msg.notice(2, "Notice: swca set d_max=" + str(self.d_max) + "\n");

        # interparticle cutoff radius, will be updated as shapes are added
        self.r_cut = 2*2*self.radius*2**(1./6);

        if friction is None:
            potentialParams = params.SWCA(radius=radius);
        else:
            raise RuntimeError('Unknown friction type: {}'.format(friction));

        _DEMBase.__init__(self, nlist);

        key = (self.dimensions, friction, 'gpu' if self.onGPU else 'cpu');
        cpp_force = cppForces[key];

        self.cpp_force = cpp_force(hoomd.context.current.system_definition,
                                   self.nlist.cpp_nlist, self.r_cut,
                                   potentialParams);

        if self.dimensions == 2:
            self.setParams = self.setParams2D;
        else:
            self.setParams = self.setParams3D;

        self._initialize_types();

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

    def update_coeffs(self):
        """Noop for this potential"""
        pass

    def setAutotunerParams(self, enable=None, period=None):
        if not self.onGPU:
            return;
        if enable is not None:
            self.autotunerEnabled = enable;
        if period is not None:
            self.autotunerPeriod = period;
        self.cpp_force.setAutotunerParams(self.autotunerEnabled, self.autotunerPeriod);

    def get_rcut(self):
        if not self.log:
            return None;

        # go through the list of only the active particle types in the sim
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # update the rcut by pair type
        r_cut_dict = nl.rcut();
        r_max_dict = {typ: sqrt(max(sum(p*p for p in point)
                                    for point in self.vertices[typ]))
                      for typ in self.vertices};
        for i in range(ntypes):
            for j in range(i, ntypes):
                (typei, typej) = type_list[i], type_list[j];
                r_cut_dict.set_pair(typei, typej,
                                    r_max_dict.get(typei, 0) + r_max_dict.get(typej, 0) +
                                    self.radius*2*2.0**(1./6) + self.d_max - 1);

        r_cut_dict.fill();

        return r_cut_dict;

# Stuff from compositebodies/pymodule/pair.py

def shiftRcut(fun):
    def result(self):
        originalRcut = fun(self)

        if originalRcut is None:
            return None

        r_maxs = {globals.system_definition.getParticleData().getNameByType(i):
                  math.sqrt(max(sum(p*p for p in point)
                                for point in shape))
                  for (i, shape) in enumerate(self._shapes)}

        r_cut = nlist.rcut()

        for (i, j) in originalRcut.values:
            r_cut.set_pair(i, j, originalRcut.get_pair(i, j) + r_maxs[i] + r_maxs[j])

        return r_cut

    return result

class _compositeBase:
    def get_max_rcut(self):
        return (2*math.sqrt(max(max(x*x + y*y + z*z for (x, y, z) in shape) if shape else 0. for shape in self._shapes)) +
                self._get_point_max_rcut(self))

    def setVertices(self, type, vertices):
        itype = globals.system_definition.getParticleData().getTypeByName(type)

        vertices = [(float(p[0]), float(p[1]), float(p[2])) for p in vertices]
        self._shapes[itype] = vertices

        # update the neighbor list
        # rcutmax = 2*(sqrt(max(x*x + y*y for (x, y) in vertices)) + self.radius*2**(1./6))
        # self.r_cut = max(self.r_cut, rcutmax)
        # neighbor_list = hoomd_script.pair._update_global_nlist(self.r_cut);

        self.cpp_force.setVertices(itype, vertices)

    get_rcut = shiftRcut(hoomd_script.pair.pair.get_rcut)

## Lennard-Jones %pair %force
#
# The command pair.lj specifies that a Lennard-Jones type %pair %force should be added to every
# non-bonded particle %pair in the simulation.
#
# \f{eqnarray*}
# V_{\mathrm{LJ}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
#                   \alpha \left( \frac{\sigma}{r} \right)^{6} \right] & r < r_{\mathrm{cut}} \\
#                     = & 0 & r \ge r_{\mathrm{cut}} \\
# \f}
#
# For an exact definition of the %force and potential calculation and how cutoff radii are handled, see pair.
#
# The following coefficients must be set per unique %pair of particle types. See hoomd_script.pair or
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon (in energy units)
# - \f$ \sigma \f$ - \c sigma (in distance units)
# - \f$ \alpha \f$ - \c alpha (unitless)
#   - <i>optional</i>: defaults to 1.0
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut (in distance units)
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
# - \f$ r_{\mathrm{on}} \f$ - \c r_on (in distance units)
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
#
# pair.lj is a standard %pair potential and supports a number of energy shift / smoothing modes. See hoomd_script.pair.pair for a full
# description of the various options.
#
# \b Example:
# \code
# lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
# lj.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, alpha=0.5, r_cut=3.0, r_on=2.0);
# lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, r_cut=2**(1.0/6.0), r_on=2.0);
# lj.pair_coeff.set(['A', 'B'], ['C', 'D'], epsilon=1.5, sigma=2.0)
# \endcode
#
# For more information on setting pair coefficients, including examples with <i>wildcards</i>, see
# \link hoomd_script.pair.coeff.set() pair_coeff.set()\endlink.
#
# The cutoff radius \a r_cut passed into the initial pair.lj command sets the default \a r_cut for all %pair
# interactions. Smaller (or larger) cutoffs can be set individually per each type %pair. The cutoff distances used for
# the neighbor list will by dynamically determined from the maximum of all \a r_cut values specified among all type
# %pair parameters among all %pair potentials.
#
# \MPI_SUPPORTED
class lj(_compositeBase, hoomd_script.pair.lj):
    ## Specify the Lennard-Jones %pair %force
    #
    # \param r_cut Default cutoff radius (in distance units)
    # \param name Name of the force instance
    #
    # \b Example:
    # \code
    # lj = pair.lj(r_cut=3.0)
    # lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    # lj.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, alpha=0.5, r_cut=3.0, r_on=2.0);
    # lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, r_cut=2**(1.0/6.0), r_on=2.0);
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
    def __init__(self, r_cut, name=None):
        util.print_status_line();

        self._shapes = [[(0, 0, 0)]]*globals.system_definition.getParticleData().getNTypes()

        # tell the base class how we operate

        # initialize the base class
        hoomd_script.pair.pair.__init__(self, r_cut, name);

        self._get_point_max_rcut = hoomd_script.pair.lj.get_max_rcut

        # update the neighbor list
        neighbor_list = nlist._subscribe_global_nlist(self.get_rcut)

        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = _compositeBodies.COMPPairLJ(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = _compositeBodies.COMPPairLJ;
        else:
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = _compositeBodies.COMPPairLJGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = _compositeBodies.COMPPairLJGPU;

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'alpha'];
        self.pair_coeff.set_default_coeff('alpha', 1.0);

## Gaussian %pair %force
#
# The command pair.gauss specifies that a Gaussian %pair %force should be added to every
# non-bonded particle %pair in the simulation.
#
# \f{eqnarray*}
#  V_{\mathrm{gauss}}(r)  = & \varepsilon \exp \left[ -\frac{1}{2}\left( \frac{r}{\sigma} \right)^2 \right]
#                                         & r < r_{\mathrm{cut}} \\
#                     = & 0 & r \ge r_{\mathrm{cut}} \\
# \f}
#
# For an exact definition of the %force and potential calculation and how cutoff radii are handled, see pair.
#
# The following coefficients must be set per unique %pair of particle types. See hoomd_script.pair or
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon (in energy units)
# - \f$ \sigma \f$ - \c sigma (in distance units)
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
# - \f$ r_{\mathrm{on}} \f$ - \c r_on
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
#
# pair.gauss is a standard %pair potential and supports a number of energy shift / smoothing modes. See hoomd_script.pair.pair for a full
# description of the various options.
#
# \b Example:
# \code
# gauss.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
# gauss.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, r_cut=3.0, r_on=2.0);
# gauss.pair_coeff.set(['A', 'B'], ['C', 'D'], epsilon=3.0, sigma=0.5)
# \endcode
#
# For more information on setting pair coefficients, including examples with <i>wildcards</i>, see
# \link hoomd_script.pair.coeff.set() pair_coeff.set()\endlink.
#
# The cutoff radius \a r_cut passed into the initial pair.gauss command sets the default \a r_cut for all %pair
# interactions. Smaller (or larger) cutoffs can be set individually per each type %pair. The cutoff distances used for
# the neighbor list will by dynamically determined from the maximum of all \a r_cut values specified among all type
# %pair parameters among all %pair potentials.
#
# \MPI_SUPPORTED
class gauss(_compositeBase, hoomd_script.pair.gauss):
    ## Specify the Gaussian %pair %force
    #
    # \param r_cut Default cutoff radius (in distance units)
    # \param name Name of the force instance
    #
    # \b Example:
    # \code
    # gauss = pair.gauss(r_cut=3.0)
    # gauss.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    # gauss.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, r_cut=3.0, r_on=2.0);
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
    def __init__(self, r_cut, name=None):
        util.print_status_line();

        self._shapes = [[(0, 0, 0)]]*globals.system_definition.getParticleData().getNTypes()

        # tell the base class how we operate

        # initialize the base class
        hoomd_script.pair.pair.__init__(self, r_cut, name);

        self._get_point_max_rcut = hoomd_script.pair.gauss.get_max_rcut

        # update the neighbor list
        neighbor_list = nlist._subscribe_global_nlist(self.get_rcut)

        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = _compositeBodies.COMPPairGauss(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = _compositeBodies.COMPPairGauss;
        else:
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = _compositeBodies.COMPPairGaussGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = _compositeBodies.COMPPairGaussGPU;

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma'];

## Shifted Gaussian %pair %force
#
# The command pair.gauss specifies that a Gaussian %pair %force should be added to every
# non-bonded particle %pair in the simulation.
#
# \f{eqnarray*}
#  V_{\mathrm{gauss}}(r)  = & \varepsilon \exp \left[ -\frac{1}{2}\left( \frac{r - r_0}{\sigma} \right)^2 \right]
#                                         & r - r_0 < r_{\mathrm{cut}} \\
#                     = & 0 & r - r_0 \ge r_{\mathrm{cut}} \\
# \f}
#
# For an exact definition of the %force and potential calculation and how cutoff radii are handled, see pair.
#
# The following coefficients must be set per unique %pair of particle types. See hoomd_script.pair or
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon (in energy units)
# - \f$ \sigma \f$ - \c sigma (in distance units)
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
# - \f$ r_{\mathrm{on}} \f$ - \c r_on
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
#
# pair.shifted_gauss is a standard %pair potential and supports a number of energy shift / smoothing modes. See hoomd_script.pair.pair for a full
# description of the various options.
#
# \b Example:
# \code
# shifted_gauss.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
# shifted_gauss.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, r_cut=3.0, r_on=2.0);
# shifted_gauss.pair_coeff.set(['A', 'B'], ['C', 'D'], epsilon=3.0, sigma=0.5)
# \endcode
#
# For more information on setting pair coefficients, including examples with <i>wildcards</i>, see
# \link hoomd_script.pair.coeff.set() pair_coeff.set()\endlink.
#
# The cutoff radius \a r_cut passed into the initial pair.shifted_gauss command sets the default \a r_cut for all %pair
# interactions. Smaller (or larger) cutoffs can be set individually per each type %pair. The cutoff distances used for
# the neighbor list will by dynamically determined from the maximum of all \a r_cut values specified among all type
# %pair parameters among all %pair potentials.
#
# \MPI_SUPPORTED
class shifted_gauss(_compositeBase, hoomd_script.pair.pair):
    ## Specify the Gaussian %pair %force
    #
    # \param r_cut Default cutoff radius (in distance units)
    # \param name Name of the force instance
    #
    # \b Example:
    # \code
    # shifted_gauss = pair.shifted_gauss(r_cut=3.0)
    # shifted_gauss.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    # shifted_gauss.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, r_cut=3.0, r_on=2.0);
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
    def __init__(self, r_cut, name=None):
        util.print_status_line();

        self._shapes = [[(0, 0, 0)]]*globals.system_definition.getParticleData().getNTypes()

        # tell the base class how we operate

        # initialize the base class
        hoomd_script.pair.pair.__init__(self, r_cut, name);

        self._get_point_max_rcut = hoomd_script.pair.gauss.get_max_rcut

        # update the neighbor list
        neighbor_list = nlist._subscribe_global_nlist(self.get_rcut)

        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = _compositeBodies.COMPPairShiftedGauss(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = _compositeBodies.COMPPairShiftedGauss;
        else:
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = _compositeBodies.COMPPairShiftedGaussGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = _compositeBodies.COMPPairShiftedGaussGPU;

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'r_0'];

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        r_0 = coeff['r_0'];

        return hoomd.make_scalar3(epsilon, sigma, r_0);

    get_shifted_rcut = shiftRcut(hoomd_script.pair.pair.get_rcut)

    def get_rcut(self):
        originalRcut = self.get_shifted_rcut()

        if originalRcut is None:
            return None

        r_cut = nlist.rcut()
        r_cut.merge(originalRcut)

        for (i, j) in originalRcut.values:
            r_0 = self.pair_coeff.get(i, j, 'r_0')
            if r_0 is not None:
                r_cut.set_pair(i, j, originalRcut.get_pair(i, j) + r_0)

        return r_cut

    def update_coeffs(self):
        coeff_list = self.required_coeffs + ["r_cut", "r_on"];
        # check that the pair coefficents are valid
        if not self.pair_coeff.verify(coeff_list):
            globals.msg.error("Not all pair coefficients are set\n");
            raise RuntimeError("Error updating pair coefficients");

        # set all the params
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));

        for i in range(0,ntypes):
            for j in range(i,ntypes):
                # build a dict of the coeffs to pass to process_coeff
                coeff_dict = {};
                for name in coeff_list:
                    coeff_dict[name] = self.pair_coeff.get(type_list[i], type_list[j], name);

                param = self.process_coeff(coeff_dict);
                self.cpp_force.setParams(i, j, param);
                self.cpp_force.setRcut(i, j, coeff_dict['r_cut'] + coeff_dict['r_0']);
                self.cpp_force.setRon(i, j, coeff_dict['r_on'] + coeff_dict['r_0']);



