# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Apply forces to particles.
"""

from hoomd import _hoomd
from hoomd.md import _md;
import sys;
import hoomd;

## \internal
# \brief Base class for forces
#
# A force in hoomd_script reflects a ForceCompute in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
# writers. 1) The instance of the c++ analyzer itself is tracked and added to the
# System 2) methods are provided for disabling the force from being added to the
# net force on each particle
class _force(hoomd.meta._metadata):
    ## \internal
    # \brief Constructs the force
    #
    # \param name name of the force instance
    #
    # Initializes the cpp_analyzer to None.
    # If specified, assigns a name to the instance
    # Assigns a name to the force in force_name;
    def __init__(self, name=None):
        # check if initialization has occured
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot create force before initialization\n");
            raise RuntimeError('Error creating force');

        # Allow force to store a name.  Used for discombobulation in the logger
        if name is None:
            self.name = "";
        else:
            self.name="_" + name;

        self.cpp_force = None;

        # increment the id counter
        id = _force.cur_id;
        _force.cur_id += 1;

        self.force_name = "force%d" % (id);
        self.enabled = True;
        self.log =True;
        hoomd.context.current.forces.append(self);

        # base class constructor
        hoomd.meta._metadata.__init__(self)

    ## \var enabled
    # \internal
    # \brief True if the force is enabled

    ## \var cpp_force
    # \internal
    # \brief Stores the C++ side ForceCompute managed by this class

    ## \var force_name
    # \internal
    # \brief The Force's name as it is assigned to the System

    ## \internal
    # \brief Checks that proper initialization has completed
    def check_initialization(self):
        # check that we have been initialized properly
        if self.cpp_force is None:
            hoomd.context.msg.error('Bug in hoomd_script: cpp_force not set, please report\n');
            raise RuntimeError();

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
        hoomd.util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if not self.enabled:
            hoomd.context.msg.warning("Ignoring command to disable a force that is already disabled");
            return;

        self.enabled = False;
        self.log = log;

        # remove the compute from the system if it is not going to be logged
        if not log:
            hoomd.context.current.system.removeCompute(self.force_name);
            hoomd.context.current.forces.remove(self)

    def enable(self):
        R""" Enable the force.

        Examples::

            force.enable()

        See :py:meth:`disable()`.
        """
        hoomd.util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if self.enabled:
            hoomd.context.msg.warning("Ignoring command to enable a force that is already enabled");
            return;

        # add the compute back to the system if it was removed
        if not self.log:
            hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);
            hoomd.context.current.forces.append(self)

        self.enabled = True;
        self.log = True;

    def get_energy(self,group):
        R""" Get the energy of a particle group.

        Args:
            group (:py:mod:`hoomd.group`): The particle group to query the energy for.

        Returns:
            The last computed energy for the members in the group.

        Examples::

            g = group.all()
            energy = force.get_energy(g)
        """
        return self.cpp_force.calcEnergyGroup(group.cpp_group)

    ## \internal
    # \brief updates force coefficients
    def update_coeffs(self):
        pass
        raise RuntimeError("_force.update_coeffs should not be called");
        # does nothing: this is for derived classes to implement

    ## \internal
    # \brief Returns the force data
    #
    def __forces(self):
        return hoomd.data.force_data(self);

    forces = property(__forces);

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['enabled'] = self.enabled
        data['log'] = self.log
        if self.name is not "":
            data['name'] = self.name

        return data

# set default counter
_force.cur_id = 0;

class constant(_force):
    R""" Constant force.

    Args:
        fvec (tuple): force vector (in force units)
        tvec (tuple): torque vector (in torque units)
        fx (float): x compent of force, retained for backwards compatibility
        fy (float): y compent of force, retained for backwards compatibility
        fz (float): z compent of force, retained for backwards compatibility
        group (:py:mod:`hoomd.group`): Group for which the force will be set.

    :py:class:`constant` specifies that a constant force should be added to every
    particle in the simulation or optionally to all particles in a group.

    Examples::

        force.constant(fx=1.0, fy=0.5, fz=0.25))
        const = force.constant(fvec=(0.4,1.0,0.5))
        const = force.constant(fvec=(0.4,1.0,0.5),group=fluid)
        const = force.constant(fvec=(0.4,1.0,0.5), tvec=(0,0,1) ,group=fluid)
    """
    def __init__(self, fx=None, fy=None, fz=None, fvec=None, tvec=None, group=None):
        hoomd.util.print_status_line();

        if (fx is not None) and (fy is not None) and (fz is not None):
            self.fvec = (fx,fy,fz)
        elif (fvec is not None):
            self.fvec = fvec
        else:
            self.fvec = (0,0,0)

        if (tvec is not None):
            self.tvec = tvec
        else:
            self.tvec = (0,0,0)

        if (self.fvec == (0,0,0)) and (self.tvec == (0,0,0)):
            hoomd.context.msg.warning("The constant force specified has no non-zero components\n");

        # initialize the base class
        _force.__init__(self);

        # create the c++ mirror class
        if (group is not None):
            self.cpp_force = _hoomd.ConstForceCompute(hoomd.context.current.system_definition, group.cpp_group, self.fvec[0],
                                                                                                                self.fvec[1],
                                                                                                                self.fvec[2],
                                                                                                                self.tvec[0],
                                                                                                                self.tvec[1],
                                                                                                                self.tvec[2]);
        else:
            self.cpp_force = _hoomd.ConstForceCompute(hoomd.context.current.system_definition, hoomd.context.current.group_all.cpp_group, self.fvec[0],
                                                                                                                                          self.fvec[1],
                                                                                                                                          self.fvec[2],
                                                                                                                                          self.tvec[0],
                                                                                                                                          self.tvec[1],
                                                                                                                                          self.tvec[2]);

        # store metadata
        self.metadata_fields = ['fvec', 'tvec']
        if group is not None:
            self.metadata_fields.append('group')
            self.group = group

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

    ## Change the value of the force
    #
    # \param fx New x-component of the force (in force units)
    # \param fy New y-component of the force (in force units)
    # \param fz New z-component of the force (in force units)
    # \param fvec New force vector
    # \param tvec New torque vector

    # \param group Group for which the force will be set
    #
    # Using set_force() requires that you saved the created constant force in a variable. i.e.
    # \code
    # const = force.constant(fx=0.4, fy=1.0, fz=0.5)
    # \endcode
    #
    # \b Example:
    # \code
    # const.set_force(fx=0.2, fy=0.1, fz=-0.5)
    # const.set_force(fx=0.2, fy=0.1, fz=-0.5, group=fluid)
    # const.set_force(fvec=(0.2,0.1,-0.5), tvec=(0,0,1), group=fluid)
    # \endcode
    def set_force(self, fx=None, fy=None, fz=None, fvec=None, tvec=None, group=None):

        if (fx is not None) and (fy is not None) and (fx is not None):
            self.fvec = (fx,fy,fz)
        elif fvec is not None:
            self.fvec = fvec
        else:
            self.fvec = (0,0,0)

        if tvec is not None:
            self.tvec = tvec
        else:
            self.tvec = (0,0,0)

        if (fvec==(0,0,0)) and (tvec==(0,0,0)):
            hoomd.contex.msg.warning("You are setting the constant force to have no non-zero components\n")

        self.check_initialization();
        if (group is not None):
            self.cpp_force.setGroupForce(group.cpp_group, self.fvec[0], self.fvec[1], self.fvec[2],
                                                          self.tvec[0], self.tvec[1], self.tvec[2]);
        else:
            self.cpp_force.setForce(self.fvec[0], self.fvec[1], self.fvec[2], self.tvec[0], self.tvec[1], self.tvec[2]);

    # there are no coeffs to update in the constant force compute
    def update_coeffs(self):
        pass

class active(_force):
    R""" Active force.

    Args:
        seed (int): required user-specified seed number for random number generator.
        f_list (list): An array of (x,y,z) tuples for the active force vector for each individual particle.
        t_list (list): An array of (x,y,z) tuples that indicate active torque vectors for each particle
        group (:py:mod:`hoomd.group`): Group for which the force will be set
        orientation_link (bool): if True then forces and torques are applied in the particle's reference frame. If false, then the box
         reference frame is used. Only relevant for non-point-like anisotropic particles.
        orientation_reverse_link (bool): When True, the particle's orientation is set to match the active force vector. Useful for
         for using a particle's orientation to log the active force vector. Not recommended for anisotropic particles. Quaternion rotation
         assumes base vector of (0,0,1).
        rotation_diff (float): rotational diffusion constant, :math:`D_r`, for all particles in the group.
        constraint (:py:class:`hoomd.md.update.constraint_ellipsoid`) specifies a constraint surface, to which particles are confined,
          such as update.constraint_ellipsoid.

    :py:class:`active` specifies that an active force should be added to all particles.
    Obeys :math:`\delta {\bf r}_i = \delta t v_0 \hat{p}_i`, where :math:`v_0` is the active velocity. In 2D
    :math:`\hat{p}_i = (\cos \theta_i, \sin \theta_i)` is the active force vector for particle :math:`i`; and the
    diffusion of the active force vector follows :math:`\delta \theta / \delta t = \sqrt{2 D_r / \delta t} \Gamma`,
    where :math:`D_r` is the rotational diffusion constant, and the gamma function is a unit-variance random variable,
    whose components are uncorrelated in time, space, and between particles.
    In 3D, :math:`\hat{p}_i` is a unit vector in 3D space, and diffusion follows
    :math:`\delta \hat{p}_i / \delta t = \sqrt{2 D_r / \delta t} \Gamma (\hat{p}_i (\cos \theta - 1) + \hat{p}_r \sin \theta)`, where
    :math:`\hat{p}_r` is an uncorrelated random unit vector. The persistence length of an active particle's path is
    :math:`v_0 / D_r`.

    .. attention::
        :py:meth:`active` does not support MPI execution.

    Examples::

        force.active( seed=13, f_list=[tuple(3,0,0) for i in range(N)])

        ellipsoid = update.constraint_ellipsoid(group=groupA, P=(0,0,0), rx=3, ry=4, rz=5)
        force.active( seed=7, f_list=[tuple(1,2,3) for i in range(N)], orientation_link=False, rotation_diff=100, constraint=ellipsoid)
    """
    def __init__(self, seed, group, f_lst=None, t_lst=None, orientation_link=True, orientation_reverse_link=False, rotation_diff=0, constraint=None):
        hoomd.util.print_status_line();

        # initialize the base class
        _force.__init__(self);

        if (f_lst is None) and (t_lst is None):
            raise RuntimeError('No forces or torques are being set')

        # input check
        if (f_lst is not None):
            for element in f_lst:
                if type(element) != tuple or len(element) != 3:
                    raise RuntimeError("Active force passed in should be a list of 3-tuples (fx, fy, fz)")
        else:
            f_lst = []
            for element in t_lst:
                f_lst.append((0,0,0))

        if (t_lst is not None):
            for element in t_lst:
                if type(element) != tuple or len(element) != 3:
                    raise RuntimeError("Active torque passed in should be a list of 3-tuples (tx, ty, tz)")
        else:
            t_lst = []
            for element in f_lst:
                t_lst.append((0,0,0))

        # assign constraints
        if (constraint is not None):
            if (constraint.__class__.__name__ is "constraint_ellipsoid"):
                P = constraint.P
                rx = constraint.rx
                ry = constraint.ry
                rz = constraint.rz
            else:
                raise RuntimeError("Active force constraint is not accepted (currently only accepts ellipsoids)")
        else:
            P = _hoomd.make_scalar3(0, 0, 0)
            rx = 0
            ry = 0
            rz = 0

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.ActiveForceCompute(hoomd.context.current.system_definition, group.cpp_group, seed, f_lst, t_lst,
                                                      orientation_link, orientation_reverse_link, rotation_diff, P, rx, ry, rz);

        else:
            self.cpp_force = _md.ActiveForceComputeGPU(hoomd.context.current.system_definition, group.cpp_group, seed, f_lst, t_lst,
                                                         orientation_link, orientation_reverse_link, rotation_diff, P, rx, ry, rz);


        # store metadata
        self.metdata_fields = ['group', 'seed', 'orientation_link', 'rotation_diff', 'constraint']
        self.group = group
        self.seed = seed
        self.orientation_link = orientation_link
        self.orientation_reverse_link = orientation_reverse_link
        self.rotation_diff = rotation_diff
        self.constraint = constraint

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

    # there are no coeffs to update in the active force compute
    def update_coeffs(self):
        pass

class dipole(_force):
    R""" Treat particles as dipoles in an electric field.

    Args:
        field_x (float): x-component of the field (units?)
        field_y (float): y-component of the field (units?)
        field_z (float): z-component of the field (units?)
        p (float): magnitude of the particles' dipole moment in the local z direction

    Examples::

        force.external_field_dipole(field_x=0.0, field_y=1.0 ,field_z=0.5, p=1.0)
        const_ext_f_dipole = force.external_field_dipole(field_x=0.0, field_y=1.0 ,field_z=0.5, p=1.0)
    """
    def __init__(self, field_x,field_y,field_z,p):
        hoomd.util.print_status_line()

        # initialize the base class
        _force.__init__(self)

        # create the c++ mirror class
        self.cpp_force = _md.ConstExternalFieldDipoleForceCompute(hoomd.context.current.system_definition, field_x, field_y, field_z, p)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # store metadata
        self.metdata_fields = ['field_x', 'field_y', 'field_z']
        self.field_x = field_x
        self.field_y = field_y
        self.field_z = field_z

    def set_params(field_x, field_y,field_z,p):
        R""" Change the constant field and dipole moment.

        Args:
            field_x (float): x-component of the field (units?)
            field_y (float): y-component of the field (units?)
            field_z (float): z-component of the field (units?)
            p (float): magnitude of the particles' dipole moment in the local z direction

        Examples::

            const_ext_f_dipole = force.external_field_dipole(field_x=0.0, field_y=1.0 ,field_z=0.5, p=1.0)
            const_ext_f_dipole.setParams(field_x=0.1, field_y=0.1, field_z=0.0, p=1.0))

        """
        self.check_initialization()

        self.cpp_force.setParams(field_x,field_y,field_z,p)

    # there are no coeffs to update in the constant ExternalFieldDipoleForceCompute
    def update_coeffs(self):
        pass
