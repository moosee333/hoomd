
import hoomd;
import hoomd.md;
import hoomd.md.nlist as nl;

from math import sqrt;

from hoomd.dem import _dem;

import math;
import sys;

def shiftRcut(fun):
    def result(self):
        originalRcut = fun(self)

        if originalRcut is None:
            return None

        r_maxs = {hoomd.context.current.system_definition.getParticleData().getNameByType(i):
                  math.sqrt(max(sum(p*p for p in point)
                                for point in shape))
                  for (i, shape) in enumerate(self._shapes)}

        r_cut = nl.rcut()

        for (i, j) in originalRcut.values:
            r_cut.set_pair(i, j, originalRcut.get_pair(i, j) + r_maxs[i] + r_maxs[j])

        return r_cut

    return result

class _compositeBase:
    def __init__(self, nlist):
        self.nlist = nlist;
        self.nlist.subscribe(self.get_rcut);
        self.nlist.update_rcut();

    def get_max_rcut(self):
        return (2*math.sqrt(max(max(x*x + y*y + z*z for (x, y, z) in shape) if shape else 0. for shape in self._shapes)) +
                self._get_point_max_rcut(self))

    def setVertices(self, type, vertices):
        itype = hoomd.context.current.system_definition.getParticleData().getTypeByName(type)

        vertices = [(float(p[0]), float(p[1]), float(p[2])) for p in vertices]
        self._shapes[itype] = vertices

        # update the neighbor list
        # rcutmax = 2*(sqrt(max(x*x + y*y for (x, y) in vertices)) + self.radius*2**(1./6))
        # self.r_cut = max(self.r_cut, rcutmax)
        # neighbor_list = hoomd.md.pair._update_global_nlist(self.r_cut);

        self.cpp_force.setVertices(itype, vertices)

    get_rcut = shiftRcut(hoomd.md.pair.pair.get_rcut)

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
# The following coefficients must be set per unique %pair of particle types. See hoomd.md.pair or
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
# pair.lj is a standard %pair potential and supports a number of energy shift / smoothing modes. See hoomd.md.pair.pair for a full
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
# \link hoomd.md.pair.coeff.set() pair_coeff.set()\endlink.
#
# The cutoff radius \a r_cut passed into the initial pair.lj command sets the default \a r_cut for all %pair
# interactions. Smaller (or larger) cutoffs can be set individually per each type %pair. The cutoff distances used for
# the neighbor list will by dynamically determined from the maximum of all \a r_cut values specified among all type
# %pair parameters among all %pair potentials.
#
# \MPI_SUPPORTED
class lj(_compositeBase, hoomd.md.pair.lj):
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
    def __init__(self, nlist, r_cut, name=None):
        util.print_status_line();

        self._shapes = [[(0, 0, 0)]]*hoomd.context.current.system_definition.getParticleData().getNTypes()

        # tell the base class how we operate

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name);

        self._get_point_max_rcut = hoomd.md.pair.lj.get_max_rcut

        _compositeBase.__init__(self, nlist)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _dem.COMPPairLJ(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name);
            self.cpp_class = _dem.COMPPairLJ;
        else:
            self.nlist.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = _dem.COMPPairLJGPU(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name);
            self.cpp_class = _dem.COMPPairLJGPU;

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

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
# The following coefficients must be set per unique %pair of particle types. See hoomd.md.pair or
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon (in energy units)
# - \f$ \sigma \f$ - \c sigma (in distance units)
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
# - \f$ r_{\mathrm{on}} \f$ - \c r_on
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
#
# pair.gauss is a standard %pair potential and supports a number of energy shift / smoothing modes. See hoomd.md.pair.pair for a full
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
# \link hoomd.md.pair.coeff.set() pair_coeff.set()\endlink.
#
# The cutoff radius \a r_cut passed into the initial pair.gauss command sets the default \a r_cut for all %pair
# interactions. Smaller (or larger) cutoffs can be set individually per each type %pair. The cutoff distances used for
# the neighbor list will by dynamically determined from the maximum of all \a r_cut values specified among all type
# %pair parameters among all %pair potentials.
#
# \MPI_SUPPORTED
class gauss(_compositeBase, hoomd.md.pair.gauss):
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
    def __init__(self, nlist, r_cut, name=None):
        util.print_status_line();

        self._shapes = [[(0, 0, 0)]]*hoomd.context.current.system_definition.getParticleData().getNTypes()

        # tell the base class how we operate

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name);

        self._get_point_max_rcut = hoomd.md.pair.gauss.get_max_rcut

        _compositeBase.__init__(self, nlist)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _dem.COMPPairGauss(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name);
            self.cpp_class = _dem.COMPPairGauss;
        else:
            self.nlist.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = _dem.COMPPairGaussGPU(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name);
            self.cpp_class = _dem.COMPPairGaussGPU;

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

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
# The following coefficients must be set per unique %pair of particle types. See hoomd.md.pair or
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon (in energy units)
# - \f$ \sigma \f$ - \c sigma (in distance units)
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
# - \f$ r_{\mathrm{on}} \f$ - \c r_on
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
#
# pair.shifted_gauss is a standard %pair potential and supports a number of energy shift / smoothing modes. See hoomd.md.pair.pair for a full
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
# \link hoomd.md.pair.coeff.set() pair_coeff.set()\endlink.
#
# The cutoff radius \a r_cut passed into the initial pair.shifted_gauss command sets the default \a r_cut for all %pair
# interactions. Smaller (or larger) cutoffs can be set individually per each type %pair. The cutoff distances used for
# the neighbor list will by dynamically determined from the maximum of all \a r_cut values specified among all type
# %pair parameters among all %pair potentials.
#
# \MPI_SUPPORTED
class shifted_gauss(_compositeBase, hoomd.md.pair.pair):
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
    def __init__(self, nlist, r_cut, name=None):
        util.print_status_line();

        self._shapes = [[(0, 0, 0)]]*hoomd.context.current.system_definition.getParticleData().getNTypes()

        # tell the base class how we operate

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name);

        self._get_point_max_rcut = hoomd.md.pair.gauss.get_max_rcut

        _compositeBase.__init__(self, nlist)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _dem.COMPPairShiftedGauss(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name);
            self.cpp_class = _dem.COMPPairShiftedGauss;
        else:
            self.nlist.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = _dem.COMPPairShiftedGaussGPU(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name);
            self.cpp_class = _dem.COMPPairShiftedGaussGPU;

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'r_0'];

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        r_0 = coeff['r_0'];

        return hoomd.make_scalar3(epsilon, sigma, r_0);

    get_shifted_rcut = shiftRcut(hoomd.md.pair.pair.get_rcut)

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
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

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
