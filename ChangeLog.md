# HOOMD-blue Change Log

[TOC]

## v2.3.0

Not yet released

*New features*

* General:
    * Store `BUILD_*` CMake variables in the hoomd cmake cache for use in external plugins.
    * `init.read_gsd` and `data.gsd_snapshot` now accept negative frame indices to index from the end of the trajectory.

* MD:
    * Improve performance with `md.constrain.rigid` in multi-GPU simulations.

* HPMC:
    * Enabled simulations involving spherical walls and convex spheropolyhedral particle shapes.
    * Support patchy energetic interactions between particles (CPU only)

* JIT:
    * Add new experimental `jit` module that uses LLVM to compile and execute user provided C++ code at runtime. (CPU only)
    * Add `jit.patch.user`: Compute arbitrary patch energy between particles in HPMC (CPU only)
    * Add `jit.patch.user_union`: Compute arbitrary patch energy between rigid unions of points in HPMC (CPU only)

*Deprecated*

*Other changes*

* Eigen is now provided as a submodule. Plugins that use Eigen headers need to update include paths.

## v2.2.4

Released 2018/03/05

*Bug fixes*

* Fix a rare error in `md.nlist.tree` when particles are very close to each other.
* Fix deadlock when `init.read_getar` is given different file names on different ranks.
* Sample from the correct uniform distribution of depletants in a sphere cap with `depletant_mode='overlap_regions'` on the CPU
* Fix a bug where ternary (or higher order) mixtures of small and large particles were not correctly handled with `depletant_mode='overlap_regions'` on the CPU
* Improve acceptance rate in depletant simulations with `depletant_mode='overlap_regions'`

## v2.2.3

Released 2018/01/25

*Bug fixes*

* Write default values to gsd frames when non-default values are present in frame 0.
* `md.wall.force_shifted_lj` now works.
* Fix a bug in HPMC where `run()` would not start after `restore_state` unless shape parameters were also set from python.
* Fix a bug in HPMC Box MC updater where moves were attempted with zero weight.
* `dump.gsd()` now writes `hpmc` shape state correctly when there are multiple particle types.
* `hpmc.integrate.polyhedron()` now produces correct results on the GPU.
* Fix binary compatibility across python minor versions.

## v2.2.2

Released 2017/12/04

*Bug fixes*

* `md.dihedral.table.set_from_file` now works.
* Fix a critical bug where forces in MPI simulations with rigid bodies or anisotropic particles were incorrectly calculated
* Ensure that ghost particles are updated after load balancing.
* `meta.dump_metadata` no longer reports an error when used with `md.constrain.rigid`
* Miscellaneous documentation fixes
* `dump.gsd` can now write GSD files with 0 particles in a frame
* Explicitly report MPI synchronization delays due to load imbalance with `profile=True`
* Correctly compute net torque of rigid bodies with anisotropic constituent particles in MPI execution on multiple ranks
* Fix `PotentialPairDPDThermoGPU.h` for use in external plugins
* Use correct ghost region with `constrain.rigid` in MPI execution on multiple ranks
* `hpmc.update.muvt()` now works with `depletant_mode='overlap_regions'`
* Fix the sampling of configurations with in `hpmc.update.muvt` with depletants
* Fix simulation crash after modifying a snapshot and re-initializing from it
* The pressure in simulations with rigid bodies (`md.constrain.rigid()`) and MPI on multiple ranks is now computed correctly

## v2.2.1

Released 2017/10/04

*Bug fixes*

* Add special pair headers to install target
* Fix a bug where `hpmc.integrate.convex_polyhedron`, `hpmc.integrate.convex_spheropolyhedron`, `hpmc.integrate.polyedron`, `hpmc.integrate.faceted_sphere`, `hpmc.integrate.sphere_union` and `hpmc.integrate.convex_polyhedron_union` produced spurious overlaps on the GPU

## v2.2.0

Released 2017/09/08

*New features*

* General:
    * Add `hoomd.hdf5.log` to log quantities in hdf5 format. Matrix quantities can be logged.
    * `dump.gsd` can now save internal state to gsd files. Call `dump_state(object)` to save the state for a particular object. The following objects are supported:
        * HPMC integrators save shape and trial move size state.
    * Add *dynamic* argument to `hoomd.dump.gsd` to specify which quantity categories should be written every frame.
    * HOOMD now inter-operates with other python libraries that set the active CUDA device.
    * Add generic capability for bidirectional ghost communication, enabling multi body potentials in MPI simulation.

* MD:
    * Added support for a 3 body potential that is harmonic in the local density.
    * `force.constant` and `force.active` can now apply torques.
    * `quiet` option to `nlist.tune` to quiet the output of the embedded `run()` commands.
    * Add special pairs as exclusions from neighbor lists.
    * Add cosine squared angle potential `md.angle.cosinesq`.
    * Add `md.pair.DLVO()` for evaluation of colloidal dispersion and electrostatic forces.
    * Add Lennard-Jones 12-8 pair potential.
    * Add Buckingham (exp-6) pair potential.
    * Add Coulomb 1-4 special_pair potential.
    * Check that composite body dimensions are consistent with minimum image convention and generate an error if they are not.
    * `md.integrate.mode.minimize_fire()` now supports anisotropic particles (i.e. composite bodies)
    * `md.integrate.mode.minimize_fire()` now supports flexible specification of integration methods
    * `md.integrate.npt()/md.integrate.nph()` now accept a friction parameter (gamma) for damping out box fluctuations during minimization runs
    * Add new command `integrate.mode_standard.reset_methods()` to clear NVT and NPT integrator variables


* HPMC:
    * `hpmc.integrate.sphere_union()` takes new capacity parameter to optimize performance for different shape sizes
    * `hpmc.integrate.polyhedron()` takes new capacity parameter to optimize performance for different shape sizes
    * `hpmc.integrate.convex_polyhedron` and `convex_spheropolyhedron` now support arbitrary numbers of vertices, subject only to memory limitations (`max_verts` is now ignored).
    * HPMC integrators restore state from a gsd file read by `init.read_gsd` when the option `restore_state` is `True`.
    * Deterministic HPMC integration on the GPU (optional): `mc.set_params(deterministic=True)`.
    * New `hpmc.update.boxmc.ln_volume()` move allows logarithmic volume moves for fast equilibration.
    * New shape: `hpmc.integrate.convex_polyhedron_union` performs simulations of unions of convex polyhedra.
    * `hpmc.field.callback()` now enables MC energy evaluation in a python function
    * The option `depletant_mode='overlap_regions'` for `hpmc.integrate.*` allows the selection of a new depletion algorithm that restores the diffusivity of dilute colloids in dense depletant baths

*Deprecated*

* HPMC: `hpmc.integrate.sphere_union()` no longer needs the `max_members` parameter.
* HPMC: `hpmc.integrate.convex_polyhedron` and `convex_spheropolyhedron` no longer needs the `max_verts` parameter.
* The *static* argument to `hoomd.dump.gsd` should no longer be used. Use *dynamic* instead.

*Bug fixes*

* HPMC:
    * `hpmc.integrate.sphere_union()` and `hpmc.integrate.polyhedron()` missed overlaps.
    * Fix alignment error when running implicit depletants on GPU with ntrial > 0.
    * HPMC integrators now behave correctly when the user provides different RNG seeds on different ranks.
    * Fix a bug where overlapping configurations were produced with `hpmc.integrate.faceted_sphere()`

* MD:
    * `charge.pppm()` with `order=7` now gives correct results
    * The PPPM energy for particles excluded as part of rigid bodies now correctly takes into account the periodic boundary conditions

* EAM:
    * `metal.pair.eam` now produces correct results.

*Other changes*

* Optimized performance of HPMC sphere union overlap check and polyhedron shape
* Improved performance of rigid bodies in MPI simulations
* Support triclinic boxes with rigid bodies
* Raise an error when an updater is given a period of 0
* Revised compilation instructions
* Misc documentation improvements
* Fully document `constrain.rigid`
* `-march=native` is no longer set by default (this is now a suggestion in the documentation)
* Compiler flags now default to CMake defaults
* `ENABLE_CUDA` and `ENABLE_MPI` CMake options default OFF. User must explicitly choose to enable optional dependencies.
* HOOMD now builds on powerpc+CUDA platforms (tested on summitdev)
* Improve performance of GPU PPPM force calculation
* Use sphere tree to further improve performance of `hpmc.integrate.sphere_union()`

## v2.1.9

Released 2017/08/22

*Bug fixes*

* Fix a bug where the log quantity `momentum` was incorrectly reported in MPI simulations.
* Raise an error when the user provides inconsistent  `charge` or `diameter` lists to `md.constrain.rigid`.
* Fix a bug where `pair.compute_energy()` did not report correct results in MPI parallel simulations.
* Fix a bug where make rigid bodies with anisotropic constituent particles did not work on the GPU.
* Fix hoomd compilation after the rebase in the cub repository.
* `deprecated.dump.xml()` now writes correct results when particles have been added or deleted from the simulation.
* Fix a critical bug where `charge.pppm()` calculated invalid forces on the GPU

## v2.1.8

Released 2017/07/19

*Bug fixes*

* `init.read_getar` now correctly restores static quantities when given a particular frame.
* Fix bug where many short calls to `run()` caused incorrect results when using `md.integrate.langevin`.
* Fix a bug in the Saru pseudo-random number generator that caused some double-precision values to be drawn outside the valid range [0,1) by a small amount. Both floats and doubles are now drawn on [0,1).
* Fix a bug where coefficients for multi-character unicode type names failed to process in Python 2.

*Other changes*

* The Saru generator has been moved into `hoomd/Saru.h`, and plugins depending on Saru or SaruGPU will need to update their includes. The `SaruGPU` class has been removed. Use `hoomd::detail::Saru` instead for both CPU and GPU plugins.

## v2.1.7

Released 2017/05/11

*Bug fixes*

* Fix PPM exclusion handling on the CPU
* Handle `r_cut` for special pairs correctly
* Fix tauP reference in NPH documentation
* Fixed ``constrain.rigid`` on compute 5.x.
* Fixed random seg faults when using sqlite getar archives with LZ4 compression
* Fixed XZ coupling with ``hoomd.md.integrate.npt`` integration
* Fixed aspect ratio with non-cubic boxes in ``hoomd.hpmc.update.boxmc``

## v2.1.6

Released 2017/04/12

*Bug fixes*

* Document `hpmc.util.tune_npt`
* Fix dump.getar.writeJSON usage with MPI execution
* Fix a bug where integrate.langevin and integrate.brownian correlated RNGs between ranks in multiple CPU execution
* Bump CUB to version 1.6.4 for improved performance on Pascal architectures. CUB is now embedded using a git submodule. Users upgrading existing git repositories should reinitialize their git submodules with ``git submodule update --init``
* CMake no longer complains when it finds a partial MKL installation.

## v2.1.5

Released 2017/03/09

*Bug fixes*

* Fixed a compile error on Mac

## v2.1.4

Released 2017/03/09

*Bug fixes*

* Fixed a bug re-enabling disabled integration methods
* Fixed a bug where adding particle types to the system failed for anisotropic pair potentials
* scipy is no longer required to execute DEM component unit tests
* Issue a warning when a subsequent call to context.initialize is given different arguments
* DPD now uses the seed from rank 0 to avoid incorrect simulations when users provide different seeds on different ranks
* Miscellaneous documentation updates
* Defer initialization message until context.initialize
* Fixed a problem where a momentary dip in TPS would cause walltime limited jobs to exit prematurely
* HPMC and DEM components now correctly print citation notices

## v2.1.3

Released 2017/02/07

*Bug fixes*

* Fixed a bug where the WalltimeLimitReached was ignored

## v2.1.2

Released 2017/01/11

*Bug fixes*

* (HPMC) Implicit depletants with spheres and faceted spheres now produces correct ensembles
* (HPMC) Implicit depletants with ntrial > 0 now produces correct ensembles
* (HPMC) NPT ensemble in HPMC (`hpmc.update.boxmc`) now produces correct ensembles
* Fix a bug where multiple nvt/npt integrators caused warnings from analyze.log.
* update.balance() is properly ignored when only one rank is available
* Add missing headers to plugin install build
* Fix a bug where charge.pppm calculated an incorrect pressure

* Other changes *

* Drop support for compute 2.0 GPU devices
* Support cusolver with CUDA 8.0

## v2.1.1

Released 2016/10/23

*Bug fixes*

* Fix `force.active` memory allocation bug
* Quiet Python.h warnigns when building (python 2.7)
* Allow multi-character particle types in HPMC (python 2.7)
* Enable `dump.getar.writeJSON` in MPI
* Allow the flow to change directions in `md.update.mueller_plathe_flow`
* Fix critical bug in MPI communication when using HPMC integrators

## v2.1.0

Released 2016/10/04

*New features*

* enable/disable overlap checks between pairs of constituent particles for `hpmc.integrate.sphere_union()`
* Support for non-additive mixtures in HPMC, overlap checks can now be enabled/disabled per type-pair
* Add `md.constrain.oned` to constrain particles to move in one dimension
* `hpmc.integrate.sphere_union()` now takes max_members as an optional argument, allowing to use GPU memory more efficiently
* Add `md.special_pair.lj()` to support scaled 1-4 (or other) exclusions in all-atom force fields
* `md.update.mueller_plathe_flow()`: Method to create shear flows in MD simulations
* `use_charge` option for `md.pair.reaction_field`
* `md.charge.pppm()` takes a Debye screening length as an optional parameter
* `md.charge.pppm()` now computes the rigid body correction to the PPPM energy

*Deprecated*

* HPMC: the `ignore_overlaps` flag is replaced by `hpmc.integrate.interaction_matrix`

*Other changes*

* Optimized MPI simulations of mixed systems with rigid and non-rigid bodies
* Removed dependency on all boost libraries. Boost is no longer needed to build hoomd
* Intel compiler builds are no longer supported due to c++11 bugs
* Shorter compile time for HPMC GPU kernels
* Include symlinked external components in the build process
* Add template for external components
* Optimized dense depletant simulations with HPMC on CPU

*Bug fixes*

* fix invalid mesh energy in non-neutral systems with `md.charge.pppm()`
* Fix invalid forces in simulations with many bond types (on GPU)
* fix rare cases where analyze.log() would report a wrong pressure
* fix possible illegal memory access when using `md.constrain.rigid()` in GPU MPI simulations
* fix a bug where the potential energy is misreported on the first step with `md.constrain.rigid()`
* Fix a bug where the potential energy is misreported in MPI simulations with `md.constrain.rigid()`
* Fix a bug where the potential energy is misreported on the first step with `md.constrain.rigid()`
* `md.charge.pppm()` computed invalid forces
* Fix a bug where PPPM interactions on CPU where not computed correctly
* Match logged quantitites between MPI and non-MPI runs on first time step
* Fix `md.pair.dpd` and `md.pair.dpdlj` `set_params`
* Fix diameter handling in DEM shifted WCA potential
* Correctly handle particle type names in lattice.unitcell
* Validate `md.group.tag_list` is consistent across MPI ranks

## v2.0.3

Released 2016/08/30

* hpmc.util.tune now works with particle types as documented
* Fix pressure computation with pair.dpd() on the GPU
* Fix a bug where dump.dcd corrupted files on job restart
* Fix a bug where HPMC walls did not work correctly with MPI
* Fix a bug where stdout/stderr did not appear in MPI execution
* HOOMD will now report an human readable error when users forget context.initialize()
* Fix syntax errors in frenkel ladd field

## v2.0.2

Released 2016/08/09

* Support CUDA Toolkit 8.0
* group.rigid()/nonrigid() did not work in MPI simulations
* Fix builds with ENABLE_DOXYGEN=on
* Always add -std=c++11 to the compiler command line arguments
* Fix rare infinite loops when using hpmc.integrate.faceted_sphere
* Fix hpmc.util.tune to work with more than one tunable
* Fix a bug where dump.gsd() would write invalid data in simulations with changing number of particles
* replicate() sometimes did not work when restarting a simulation

## v2.0.1

Released 2016/07/15

*Bug fixes*

* Fix acceptance criterion in mu-V-T simulations with implicit depletants (HPMC).
* References to disabled analyzers, computes, updaters, etc. are properly freed from the simulation context.
* Fix a bug where `init.read_gsd` ignored the `restart` argument.
* Report an error when HPMC kernels run out of memory.
* Fix ghost layer when using rigid constraints in MPI runs.
* Clarify definition of the dihedral angle.

## v2.0.0

Released 2016/06/22

HOOMD-blue v2.0 is released under a clean BSD 3-clause license.

*New packages*

* `dem` - simulate faceted shapes with dynamics
* `hpmc` - hard particle Monte Carlo of a variety of shape classes.

*Bug fixes*

* Angles, dihedrals, and impropers no longer initialize with one default type.
* Fixed a bug where integrate.brownian gave the same x,y, and z velocity components.
* Data proxies verify input types and vector lengths.
* dump.dcd no longer generates excessive metadata traffic on lustre file systems

*New features*

* Distance constraints `constrain.distance` - constrain pairs of particles to a fixed separation distance
* Rigid body constraints `constrain.rigid` - rigid bodies now have central particles, and support MPI and replication
* Multi-GPU electrostatics `charge.pppm` - the long range electrostatic forces are now supported in MPI runs
* `context.initialize()` can now be called multiple times - useful in jupyter notebooks
* Manage multiple simulations in a single job script with `SimulationContext` as a python context manager.
* `util.quiet_status() / util.unquiet_status()` allow users to control if line status messages are output.
* Support executing hoomd in Jupyter (ipython) notebooks. Notice, warning, and error messages now show up in the
  notebook output blocks.
* `analyze.log` can now register python callback functions as sources for logged quantities.
* The GSD file format (http://gsd.readthedocs.io) is fully implemented in hoomd
    * `dump.gsd` writes GSD trajectories and restart files (use `truncate=true` for restarts).
    * `init.read_gsd` reads GSD file and initializes the system, and can start the simulation
       from any frame in the GSD file.
    * `data.gsd_snapshot` reads a GSD file into a snapshot which can be modified before system
      initialization with `init.read_snapshot`.
    * The GSD file format is capable of storing all particle and topology data fields in hoomd,
      either static at frame 0, or varying over the course of the trajectory. The number of
      particles, types, bonds, etc. can also vary over the trajectory.
* `force.active` applies an active force (optionally with rotational diffusion) to a group of particles
* `update.constrain_ellipsoid` constrains particles to an ellipsoid
* `integrate.langevin` and `integrate.brownian` now apply rotational noise and damping to anisotropic particles
* Support dynamically updating groups. `group.force_update()` forces the group to rebuild according
  to the original selection criteria. For example, this can be used to periodically update a cuboid
  group to include particles only in the specified region.
* `pair.reaction_field` implements a pair force for a screened electrostatic interaction of a charge pair in a
  dielectric medium.
* `force.get_energy` allows querying the potential energy of a particle group for a specific force
* `init.create_lattice` initializes particles on a lattice.
    * `lattice.unitcell` provides a generic unit cell definition for `create_lattice`
    * Convenience functions for common lattices: sq, hex, sc, bcc, fcc.
* Dump and initialize commands for the GTAR file format (http://libgetar.readthedocs.io).
    * GTAR can store trajectory data in zip, tar, sqlite, or bare directories
    * The current version stores system properties, later versions will be able to capture log, metadata, and other
      output to reduce the number of files that a job script produces.
* `integrate.npt` can now apply a constant stress tensor to the simulation box.
* Faceted shapes can now be simulated through the `dem` component.

*Changes that require job script modifications*

* `context.initialize()` is now required before any other hoomd script command.
* `init.reset()` no longer exists. Use `context.initialize()` or activate a `SimulationContext`.
* Any scripts that relied on undocumented members of the `globals` module will fail. These variables have been moved to
  the `context` module and members of the currently active `SimulationContext`.
* bonds, angles, dihedrals, and impropers no longer use the `set_coeff` syntax. Use `bond_coeff.set`, `angle_coeff.set`,
  `dihedral_coeff.set`, and `improper_coeff.set` instead.
* `hoomd_script` no longer exists, python commands are now spread across `hoomd`, `hoomd.md`, and other sub packages.
* `integrate.\*_rigid()` no longer exists. Use a standard integrator on `group.rigid_center()`, and define rigid bodies
  using `constrain.rigid()`
* All neighbor lists must be explicitly created using `nlist.\*`, and each pair potential must be attached explicitly
  to a neighbor list. A default global neighbor list is no longer created.
* Moved cgcmm into its own package.
* Moved eam into the metal package.
* Integrators now take `kT` arguments for temperature instead of `T` to avoid confusion on the units of temperature.
* phase defaults to 0 for updaters and analyzers so that restartable jobs are more easily enabled by default.
* `dump.xml` (deprecated) requires a particle group, and can dump subsets of particles.

*Other changes*

* CMake minimum version is now 2.8
* Convert particle type names to `str` to allow unicode type name input
* `__version__` is now available in the top level package
* `boost::iostreams` is no longer a build dependency
* `boost::filesystem` is no longer a build dependency
* New concepts page explaining the different styles of neighbor lists
* Default neighbor list buffer radius is more clearly shown to be r_buff = 0.4
* Memory usage of `nlist.stencil` is significantly reduced
* A C++11 compliant compiler is now required to build HOOMD-blue

*Removed*

* Removed `integrate.bdnvt`: use `integrate.langevin`
* Removed `mtk=False` option from `integrate.nvt` - The MTK NVT integrator is now the only implementation.
* Removed `integrate.\*_rigid()`: rigid body functionality is now contained in the standard integration methods
* Removed the global neighbor list, and thin wrappers to the neighbor list in `nlist`.
* Removed PDB and MOL2 dump writers.
* Removed init.create_empty

*Deprecated*

* Deprecated analyze.msd.
* Deprecated dump.xml.
* Deprecated dump.pos.
* Deprecated init.read_xml.
* Deprecated init.create_random.
* Deprecated init.create_random_polymers.

## v1.3.3

Released 2016/03/06

*Bug fixes*

* Fix problem incluing `hoomd.h` in plugins
* Fix random memory errors when using walls

## v1.3.2

Released 2016/02/08

*Bug fixes*

* Fix wrong access to system.box
* Fix kinetic energy logging in MPI
* Fix particle out of box error if particles are initialized on the boundary in MPI
* Add integrate.brownian to the documentation index
* Fix misc doc typos
* Fix runtime errors with boost 1.60.0
* Fix corrupt metadata dumps in MPI runs

## v1.3.1

Released 2016/1/14

*Bug fixes*

* Fix invalid MPI communicator error with Intel MPI
* Fix python 3.5.1 seg fault

## v1.3.0

Released 2015/12/8

*New features*

* Automatically load balanced domain decomposition simulations.
* Anisotropic particle integrators.
* Gay-Berne pair potential.
* Dipole pair potential.
* Brownian dynamics `integrate.brownian`
* Langevin dynamics `integrate.langevin` (formerly `bdnvt`)
* `nlist.stencil` to compute neighbor lists using stencilled cell lists.
* Add single value scale, `min_image`, and `make_fraction` to `data.boxdim`
* `analyze.log` can optionally not write a file and now supports querying current quantity values.
* Rewritten wall potentials.
    * Walls are now sums of planar, cylindrical, and spherical half-spaces.
    * Walls are defined and can be modified in job scripts.
    * Walls execute on the GPU.
    * Walls support per type interaction parameters.
    * Implemented for: lj, gauss, slj, yukawa, morse, force_shifted_lj, and mie potentials.
* External electric field potential: `external.e_field`

*Bug fixes*

* Fixed a bug where NVT integration hung when there were 0 particles in some domains.
* Check SLURM environment variables for local MPI rank identification
* Fixed a typo in the box math documentation
* Fixed a bug where exceptions weren't properly passed up to the user script
* Fixed a bug in the velocity initialization example
* Fixed an openmpi fork() warning on some systems
* Fixed segfaults in PPPM
* Fixed a bug where compute.thermo failed after reinitializing a system
* Support list and dict-like objects in init.create_random_polymers.
* Fall back to global rank to assign GPUs if local rank is not available

*Deprecated commands*

* `integrate.bdnvt` is deprecated. Use `integrate.langevin` instead.
* `dump.bin` and `init.bin` are now removed. Use XML files for restartable jobs.

*Changes that may break existing scripts*

* `boxdim.wrap` now returns the position and image in a tuple, where it used to return just the position.
* `wall.lj` has a new API
* `dump.bin` and `init.bin` have been removed.

## v1.2.1

Released 2015/10/22

*Bug fixes*

* Fix a crash when adding or removing particles and reinitializing
* Fix a bug where simulations hung on sm 5.x GPUs with CUDA 7.5
* Fix compile error with long tests enabled
* Issue a warning instead of an error for memory allocations greater than 4 GiB.
* Fix invalid RPATH when building inside `zsh`.
* Fix incorrect simulations with `integrate.npt_rigid`
* Label mie potential correctly in user documentation

## v1.2.0

Released 2015/09/30

*New features*

* Performance improvements for systems with large particle size disparity
* Bounding volume hierarchy (tree) neighbor list computation
* Neighbor lists have separate `r_cut` values for each pair of types
* addInfo callback for dump.pos allows user specified information in pos files

*Bug fixes*

* Fix `test_pair_set_energy` unit test, which failed on numpy < 1.9.0
* Analyze.log now accepts unicode strings.
* Fixed a bug where calling `restore_snapshot()` during a run zeroed potential parameters.
* Fix segfault on exit with python 3.4
* Add `cite.save()` to documentation
* Fix a problem were bond forces are computed incorrectly in some MPI configurations
* Fix bug in pair.zbl
* Add pair.zbl to the documentation
* Use `HOOMD_PYTHON_LIBRARY` to avoid problems with modified CMake builds that preset `PYTHON_LIBRARY`

## v1.1.1

Released 2015/07/21

*Bug fixes*

* `dump.xml(restart=True)` now works with MPI execution
* Added missing documentation for `meta.dump_metadata`
* Build all unit tests by default
* Run all script unit tests through `mpirun -n 1`

## v1.1.0

Released 2015/07/14

*New features*

* Allow builds with ninja.
* Allow K=0 FENE bonds.
* Allow number of particles types to change after initialization.
```system.particles.types.add('newType')```
* Allow number of particles to change after initialization.
```
system.particles.add('A')
del system.particles[0]
```
* OPLS dihedral
* Add `phase` keyword to analyzers and dumps to make restartable jobs easier.
* `HOOMD_WALLTIME_STOP` environment variable to stop simulation runs before they hit a wall clock limit.
* `init.read_xml()` Now accepts an initialization and restart file.
* `dump.xml()` can now write restart files.
* Added documentation concepts page on writing restartable jobs.
* New citation management infrastructure. `cite.save()` writes `.bib` files with a list of references to features
  actively used in the current job script.
* Snapshots expose data as numpy arrays for high performance access to particle properties.
* `data.make_snapshot()` makes a new empty snapshot.
* `analyze.callback()` allows multiple python callbacks to operate at different periods.
* `comm.barrier()` and `comm.barrier_all()` allow users to insert barriers into their scripts.
* Mie pair potential.
* `meta.dump_metadata()` writes job metadata information out to a json file.
* `context.initialize()` initializes the execution context.
* Restart option for `dump.xml()`

*Bug fixes*

* Fix slow performance when initializing `pair.slj()`in MPI runs.
* Properly update particle image when setting position from python.
* PYTHON_SITEDIR hoomd shell launcher now calls the python interpreter used at build time.
* Fix compile error on older gcc versions.
* Fix a bug where rigid bodies had 0 velocity when restarting jobs.
* Enable `-march=native` builds in OS X clang builds.
* Fix `group.rigid()` and `group.nonrigid()`.
* Fix image access from the python data access proxies.
* Gracefully exit when launching MPI jobs with mixed execution configurations.

*Changes that may require updated job scripts*

* `context.initialize()` **must** be called before any `comm` method that queries the MPI rank. Call it as early as
  possible in your job script (right after importing `hoomd_script`) to avoid problems.

*Deprecated*

* `init.create_empty()` is deprecated and will be removed in a future version. Use `data.make_snapshot()` and
  `init.read_snapshot()` instead.
* Job scripts that do not call `context.initialize()` will result in a warning message. A future version of HOOMD
  will require that you call `context.initialize()`.

*Removed*

* Several `option` commands for controlling the execution configuration. Replaced with `context.initialize`.

## v1.0.5

Released 2015/05/19

*Bug fixes*

* Fix segfault when changing integrators
* Fix system.box to indicate the correct number of dimensions
* Fix syntax error in comm.get_rank with --nrank
* Enable CUDA enabled builds with the intel compiler
* Use CMake builtin FindCUDA on recent versions of CMake
* GCC_ARCH env var sets the -march command line option to gcc at configure time
* Auto-assign GPU-ids on non-compute exclusive systems even with --mode=gpu
* Support python 3.5 alpha
* Fix a bug where particle types were doubled with boost 1.58.0
* Fix a bug where angle_z=true dcd output was inaccurate near 0 angles
* Properly handle lj.wall potentials with epsilon=0.0 and particles on top of the walls

## v1.0.4

Released 2015/04/07

*Bug fixes*

* Fix invalid virials computed in rigid body simulations when multi-particle bodies crossed box boundaries
* Fix invalid forces/torques for rigid body simulations caused by race conditions
* Fix compile errors on Mac OS X 10.10
* Fix invalid pair force computations caused by race conditions
* Fix invalid neighbour list computations caused by race conditions on Fermi generation GPUs

*Other*

* Extremely long running unit tests are now off by default. Enable with -DHOOMD_SKIP_LONG_TESTS=OFF
* Add additional tests to detect race conditions and memory errors in kernels

## v1.0.3

Released 2015/03/18

**Bug fixes**

* Enable builds with intel MPI
* Silence warnings coming from boost and python headers

## v1.0.2

Released 2015/01/21

**Bug fixes**

* Fixed a bug where `linear_interp` would not take a floating point value for *zero*
* Provide more useful error messages when cuda drivers are not present
* Assume device count is 0 when `cudaGetDeviceCount()` returns an error
* Link to python statically when `ENABLE_STATIC=on`
* Misc documentation updates

## v1.0.1

Released 2014/09/09

**Bug fixes**

1. Fixed bug where error messages were truncated and HOOMD exited with a segmentation fault instead (e.g. on Blue Waters)
1. Fixed bug where plug-ins did not load on Blue Waters
1. Fixed compile error with gcc4.4 and cuda5.0
1. Fixed syntax error in `read_snapshot()`
1. Fixed a bug where `init.read_xml throwing` an error (or any other command outside of `run()`) would hang in MPI runs
1. Search the install path for hoomd_script - enable the hoomd executable to be outside of the install tree (useful with cray aprun)
1. Fixed CMake 3.0 warnings
1. Removed dependancy on tr1/random
1. Fixed a bug where `analyze.msd` ignored images in the r0_file
1. Fixed typos in `pair.gauss` documentation
1. Fixed compile errors on Ubuntu 12.10
1. Fix failure of `integrate.nvt` to reach target temperature in analyze.log. The fix is a new symplectic MTK integrate.nvt integrator. Simulation results in hoomd v1.0.0 are correct, just the temperature and velocity outputs are off slightly.
1. Remove MPI from Mac OS X dmg build.
1. Enable `import hoomd_script as ...`

*Other changes*

1. Added default compile flag -march=native
1. Support CUDA 6.5
1. Binary builds for CentOS/RHEL 6, Fedora 20, Ubuntu 14.04 LTS, and Ubuntu 12.04 LTS.

## Version 1.0.0

Released 2014/05/25

*New features*

* Support for python 3
* New NPT integrator capable of flexible coupling schemes
* Triclinic unit cell support
* MPI domain decomposition
* Snapshot save/restore
* Autotune block sizes at run time
* Improve performance in small simulation boxes
* Improve performance with smaller numbers of particles per GPU
* Full double precision computations on the GPU (compile time option must be enabled, binary builds provided on the download page are single precision)
* Tabulated bond potential `bond.table`
* Tabulated angle potential `angle.table`
* Tabulated dihedral potental `dihedral.table`
* `update.box_resize` now accepts `period=None` to trigger an immediate update of the box without creating a periodic updater
* `update.box_resize` now replaces *None* arguments with the current box parameters
* `init.create_random` and `init.create_random_polymers` can now create random configurations in triclinc and 2D boxes
* `init.create_empty` can now create triclinic boxes
* particle, bond, angle, dihedral, and impropers types can now be named in `init.create_empty`
* `system.replicate` command replicates the simulation box

*Bug fixes*

* Fixed a bug where init.create_random_polymers failed when lx,ly,lz were not equal.
* Fixed a bug in init.create_random_polymers and init.create_random where the separation radius was not accounted for correctly
* Fixed a bug in bond.* where random crashes would occur when more than one bond type was defined
* Fixed a bug where dump.dcd did not write the period to the file

*Changes that may require updated job scripts*

* `integrate.nph`: A time scale `tau_p` for the relaxation of the barostat is now required instead of the barostat mass *W* of the previous release.
The time scale is the relaxation time the barostat would have at an average temperature `T_0 = 1`, and it is related to the internally used
(Andersen) Barostat mass *W* via `W = d N T_0 tau_p^2`, where *d* is the dimensionsality and *N* the number of particles.
* `sorter` and `nlist` are now modules, not variables in the `__main__` namespace.
* Data proxies function correctly in MPI simulations, but are extremely slow. If you use `init.create_empty`, consider separating the generation step out to a single rank short execution that writes an XML file for the main run.
* `update.box_resize(Lx=...)` no longer makes cubic box updates, instead it will keep the current **Ly** and **Lz**. Use the `L=...` shorthand for cubic box updates.
* All `init.*` commands now take `data.boxdim` objects, instead of `hoomd.boxdim` (or *3-tuples*). We strongly encourage the use of explicit argument names for `data.boxdim()`. In particular, if `hoomd.boxdim(123)` was previously used to create a cubic box, it is now required to use `data.boxdim(L=123)` (CORRECT) instead of `data.boxdim(123)` (INCORRECT), otherwise a box with unit dimensions along the y and z axes will be created.
* `system.dimensions` can no longer be set after initialization. System dimensions are now set during initialization via the `data.boxdim` interface. The dimensionality of the system can now be queried through `system.box`.
* `system.box` no longer accepts 3-tuples. It takes `data.boxdim` objects.
* `system.dimensions` no longer exists. Query the dimensionality of the system from `system.box`. Set the dimensionality of the system by passing an appropriate `data.boxdim` to an `init` method.
* `init.create_empty` no longer accepts `n_*_types`. Instead, it now takes a list of strings to name the types.

*Deprecated*

* Support for G80, G200 GPUs.
* `dump.bin` and `read.bin`. These will be removed in v1.1 and replaced with a new binary format.

*Removed*

* OpenMP mult-core execution (replaced with MPI domain decomposition)
* `tune.find_optimal_block_size` (replaced by Autotuner)

## Version 0.11.3

Released 2013/05/10

*Bug fixes*

* Fixed a bug where charge.pppm could not be used after init.reset()
* Data proxies can now set body angular momentum before the first run()
* Fixed a bug where PPPM forces were incorrect on the GPU

## Version 0.11.2

Released 2012/12/19

*New features*

* Block sizes tuned for K20

*Bug fixes*

* Warn user that PPPM ignores rigid body exclusions
* Document that proxy iterators need to be deleted before init.reset()
* Fixed a bug where body angular momentum could not be set
* Fixed a bug where analyze.log would report nan for the pressure tensor in nve and nvt simulations

## Version 0.11.1

Released 2012/11/2

*New features*

* Support for CUDA 5.0
* Binary builds for Fedora 16 and OpenSUSE 12.1
* Automatically specify /usr/bin/gcc to nvcc when the configured gcc is not supported

*Bug fixes*

* Fixed a compile error with gcc 4.7
* Fixed a bug where PPPM forces were incorrect with neighborlist exclusions
* Fixed an issue where boost 1.50 and newer were not detected properly when BOOST_ROOT is set
* Fixed a bug where accessing force data in python prevented init.reset() from working
* Fixed a bug that prevented pair.external from logging energy
* Fixed a unit test that failed randomly

## Version 0.11.0

2012-07-27

*New features*

1. Support for Kepler GPUs (GTX 680)
1. NPH integration (*integrate.nph*)
1. Compute full pressure tensor
1. Example plugin for new bond potentials
1. New syntax for bond coefficients: *_bond_.bond_coeff.set('type', _params_)*
1. New external potential: *external.periodic* applies a periodic potential along one direction (uses include inducing lamellar phases in copolymer systems)
1. Significant performance increases when running *analyze.log*, *analyze.msd*, *update.box_resize*, *update.rescale_temp*, or *update.zero_momentum* with a small period
1. Command line options may now be overwritten by scripts, ex: *options.set_gpu(2)*
1. Added *--user* command line option to allow user defined options to be passed into job scripts, ex: *--user="-N=5 -phi=0.56"*
1. Added *table.set_from_file* method to enable reading table based pair potentials from a file
1. Added *--notice-level* command line option to control how much extra information is printed during a run. Set to 0 to disable, or any value up to 10. At 10, verbose debugging information is printed.
1. Added *--msg-file* command line option which redirects the message output to a file
1. New pair potential *pair.force_shifted_lj* : Implements http://dx.doi.org/10.1063/1.3558787

*Bug fixes*

1. Fixed a bug where FENE bonds were sometimes computed incorrectly
1. Fixed a bug where pressure was computed incorrectly when using pair.dpd or pair.dpdlj
1. Fixed a bug where using OpenMP and CUDA at the same time caused invalid memory accesses
1. Fixed a bug where RPM packages did not work on systems where the CUDA toolkit was not installed
1. Fixed a bug where rigid body velocities were not set from python
1. Disabled OpenMP builds on Mac OS X. HOOMD-blue w/ openmp enabled crashes due to bugs in Apple's OpenMP implementation.
1. Fixed a bug that allowed users to provide invalid rigid body data and cause a seg fault.
1. Fixed a bug where using PPPM resulted in error messages on program exit.

*API changes*

1. Bond potentials rewritten with template evaluators
1. External potentials use template evaluators
1. Complete rewrite of ParticleData - may break existing plugins
1. Bond/Angle/Dihedral data structures rewritten
    * The GPU specific data structures are now generated on the GPU
1. DPDThermo and DPDLJThermo are now processed by the same template class
1. Headers that cannot be included by nvcc now throw an error when they are
1. CUDA 4.0 is the new minimum requirement
1. Rewrote BoxDim to internally handle minimum image conventions
1. HOOMD now only compiles ptx code for the newest architecture, this halves the executable file size
1. New Messenger class for global control of messages printed to the screen / directed to a file.

*Testing changes*

1. Automated test suite now performs tests on OpenMPI + CUDA builds
1. Valgrind tests added back into automated test suite
1. Added CPU test in bd_ridid_updater_tests
1. ctest -S scripts can now set parallel makes (with cmake > 2.8.2)

## Version 0.10.1

2012-02-10

1. Add missing entries to credits page
1. Add `dist_check` option to neighbor list. Can be used to force neighbor list builds at a specified frequency (useful in profiling runs with nvvp).
1. Fix typos in ubuntu compile documentation
1. Add missing header files to hoomd.h
1. Add torque to the python particle data access API
1. Support boost::filesystem API v3
1. Expose name of executing gpu, n_cpu, hoomd version, git sha1, cuda version, and compiler version to python
1. Fix a bug where multiple `nvt_rigid` or `npt_rigid` integrators didn't work correctly
1. Fix missing pages in developer documentation

## Version 0.10.0

2011-12-14

*New features*

1. Added *pair.dpdlj* which uses the DPD thermostat and the Lennard-Jones potential. In previous versions, this could be accomplished by using two pair commands but at the cost of reduced performance.
1. Additional example scripts are now present in the documentation. The example scripts are cross-linked to the commands that are used in them.
1. Most dump commands now accept the form: *dump.ext(filename="filename.ext")* which immediately writes out filename.ext.
1. Added _vis_ parameter to dump.xml which enables output options commonly used in files written for the purposes of visulization. dump.xml also now accepts parameters on the instantiation line. Combined with the previous feature, *dump.xml(filename="file.xml", vis=True)* is now a convenient short hand for what was previously
<pre><code class="python">
xml = dump.xml()
xml.set_params(position = True, mass = True, diameter = True, \
                         type = True, bond = True, angle = True, \
                         dihedral = True, improper = True, charge = True)
xml.write(filename="file.xml")
</code></pre>
1. Specify rigid bodies in XML input files
1. Simulations that contain rigid body constraints applied to groups of particles in BDNVT, NVE, NVT, and NPT ensembles.
    * *integrate.bdnvt_rigid*
    * *integrate.nve_rigid*
    * *integrate.nvt_rigid*
    * *integrate.npt_rigid*
1. Energy minimization of rigid bodies (*integrate.mode_minimize_rigid_fire*)
1. Existing commands are now rigid-body aware
    * update.rescale_temp
    * update.box_resize
    * update.enforce2d
    * update.zero_momentum
1. NVT integration using the Berendsen thermostat (*integrate.berendsen*)
1. Bonds, angles, dihedrals, and impropers can now be created and deleted with the python data access API.
1. Attribution clauses added to the HOOMD-blue license.

*Changes that may break existing job scripts*

1. The _wrap_ option to *dump.dcd* has been changed to _unwrap_full_ and its meaning inverted. *dump.dcd* now offers two options for unwrapping particles, _unwrap_full_ fully unwraps particles into their box image and _unwrap_rigid_ unwraps particles in rigid bodies so that bodies are not broken up across a box boundary.

*Bug/fixes small enhancements*

1. Fixed a bug where launching hoomd on mac os X 10.5 always resulted in a bus error.
1. Fixed a bug where DCD output restricted to a group saved incorrect data.
1. force.constant may now be applied to a group of particles, not just all particles
1. Added C++ plugin example that demonstrates how to add a pair potential in a plugin
1. Fixed a bug where box.resize would always transfer particle data even in a flat portion of the variant
1. OpenMP builds re-enabled on Mac OS X
1. Initial state of integrate.nvt and integrate.npt changed to decrease oscillations at startup.
1. Fixed a bug where the polymer generator would fail to initialize very long polymers
1. Fixed a bug where images were passed to python as unsigned ints.
1. Fixed a bug where dump.pdb wrote coordinates in the wrong order.
1. Fixed a rare problem where a file written by dump.xml would not be read by init.read_xml due to round-off errors.
1. Increased the number of significant digits written out to dump.xml to make them more useful for ad-hoc restart files.
1. Potential energy and pressure computations that slow performance are now only performed on those steps where the values are actually needed.
1. Fixed a typo in the example C++ plugin
1. Mac build instructions updated to work with the latest version of macports
1. Fixed a bug where set_period on any dump was ineffective.
1. print_status_line now handles multiple lines
1. Fixed a bug where using bdnvt tally with per type gammas resulted in a race condition.
1. Fix an issue where ENABLE_CUDA=off builds gave nonsense errors when --mode=gpu was requested.
1. Fixed a bug where dumpl.xml could produce files that init.xml would not read
1. Fixed a typo in the example plugin
1. Fix example that uses hoomd as a library so that it compiles.
1. Update maintainer lines
1. Added message to nlist exclusions that notifies if diameter or body exclusions are set.
1. HOOMD-blue is now hosted in a git repository
1. Added bibtex bibliography to the user documentation
1. Converted user documentation examples to use doxygen auto cross-referencing \example commands
1. Fix a bug where particle data is not released in dump.binary
1. ENABLE_OPENMP can now be set in the ctest builds
1. Tuned block sizes for CUDA 4.0
1. Removed unsupported GPUS from CUDA_ARCH_LIST

## Version 0.9.2

2011-04-04

*Note:* only major changes are listed here.

*New features*

1. *New exclusion option:* Particles can now be excluded from the neighbor list based on diameter consistent with pair.slj.
1. *New pair coeff syntax:* Coefficients for multiple type pairs can be specified conveniently on a single line.
<pre><code class="python">
coeff.set(['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'], epsilon=1.0)
</code></pre>
1. *New documentation:* HOOMD-blue's system of units is now fully documented, and every coefficient in the documentation is labeled with the appropriate unit.
1. *Performance improvements:* Performance has been significantly boosted for simulations of medium sized systems (5,000-20,000 particles). Smaller performance boosts were made to larger runs.
1. *CUDA 3.2 support:* HOOMD-blue is now fully tested and performance tuned for use with CUDA 3.2.
1. *CUDA 4.0 support:* HOOMD-blue compiles with CUDA 4.0 and passes initial tests.
1. *New command:* tune.r_buff performs detailed auto-tuning of the r_buff neighborlist parameter.
1. *New installation method:* RPM, DEB, and app bundle packages are now built for easier installation
1. *New command:* charge.pppm computes the full long range electrostatic interaction using the PPPM method

*Bug/fixes small enhancements*

1. Fixed a bug where the python library was linked statically.
1. Added the PYTHON_SITEDIR setting to allow hoomd builds to install into the native python site directory.
1. FIRE energy minimization convergence criteria changed to require both energy *and* force to converge
1. Clarified that groups are static in the documentation
1. Updated doc comments for compatibility with Doxygen#7.3
1. system.particles.types now lists the particle types in the simulation
1. Creating a group of a non-existant type is no longer an error
1. Mention XML file format for walls in wall.lj documentation
1. Analyzers now profile themselves
1. Use "\n" for newlines in dump.xml - improves performance when writing many XML files on a NFS file system
1. Fixed a bug where the neighbor list build could take an exceptionally long time (several seconds) to complete the first build.
1. Fixed a bug where certain logged quantities always reported as 0 on the first step of the simulation.
1. system.box can now be used to read and set the simulation box size from python
1. Numerous internal API updates
1. Fixed a bug the resulted in incorrect behavior when using integrate.npt on the GPU.
1. Removed hoomd launcher shell script. In non-sitedir installs, ${HOOMD_ROOT}/bin/hoomd is now the executable itself
1. Creating unions of groups of non-existent types no longer produces a seg fault
1. hoomd now builds on all cuda architectures. Modify CUDA_ARCH_LIST in cmake to add or remove architectures from the build
1. hoomd now builds with boost#46.0
1. Updated hoomd icons to maize/blue color scheme
1. hoomd xml file format bumped to#3, adds support for charge.
1. FENE and harmonic bonds now handle 0 interaction parameters and 0 length bonds more gracefully
1. The packaged plugin template now actually builds and installs into a recent build of hoomd

## Version 0.9.1

2010-10-08

*Note:* only major changes are listed here.

*New features*

1. *New constraint*: constrain.sphere constrains a group of particles to the surface of a sphere
1. *New pair potential/thermostat*: pair.dpd implements the standard DPD conservative, random, and dissipative forces
1. *New pair potential*: pair.dpd_conservative applies just the conservative DPD potential
1. *New pair potential*: pair.eam implements the Embedded Atom Method (EAM) and supports both *alloy* and *FS* type computations.
1. *Faster performance*: Cell list and neighbor list code has been rewritten for performance.
    * In our benchmarks, *performance increases* ranged from *10-50%* over HOOMD-blue 0.9.0. Simulations with shorter cutoffs tend to attain a higher performance boost than those with longer cutoffs.
    * We recommended that you *re-tune r_buff* values for optimal performance with 0.9.1.
    * Due to the nature of the changes, *identical runs* may produce *different trajectories*.
1. *Removed limitation*: The limit on the number of neighbor list exclusions per particle has been removed. Any number of exclusions can now be added per particle. Expect reduced performance when adding excessive numbers of exclusions.

*Bug/fixes small enhancements*

1. Pressure computation is now correct when constraints are applied.
1. Removed missing files from hoomd.h
1. pair.yukawa is no longer referred to by "gaussian" in the documentation
1. Fermi GPUs are now prioritized over per-Fermi GPUs in systems where both are present
1. HOOMD now compiles against CUDA 3.1
1. Momentum conservation significantly improved on compute#x hardware
1. hoomd plugins can now be installed into user specified directories
1. Setting r_buff=0 no longer triggers exclusion list updates on every step
1. CUDA 2.2 and older are no longer supported
1. Workaround for compiler bug in 3.1 that produces extremely high register usage
1. Disabled OpenMP compile checks on Mac OS X
1. Support for compute 2.1 devices (such as the GTX 460)

## Version 0.9.0

2010-05-18

*Note:* only major changes are listed here.

*New features*

1. *New pair potential*: Shifted LJ potential for particles of varying diameters (pair.slj)
1. *New pair potential*: Tabulated pair potential (pair.table)
1. *New pair potential*: Yukawa potential (pair.yukawa)
1. *Update to pair potentials*: Most pair potentials can now accept different values of r_cut for different type pairs. The r_cut specified in the initial pair.*** command is now treated as the default r_cut, so no changes to scripts are necessary.
1. *Update to pair potentials*: Default pair coeff values are now supported. The parameter alpha for lj now defaults to#0, so there is no longer a need to specify it for a majority of simulations.
1. *Update to pair potentials*: The maximum r_cut needed for the neighbor list is now determined at the start of each run(). In simulations where r_cut may decrease over time, increased performance will result.
1. *Update to pair potentials*: Pair potentials are now specified via template evaluator classes. Adding a new pair potential to hoomd now only requires a small amount of additional code.
1. *Plugin API* : Advanced users/developers can now write, install, and use plugins for hoomd without needing to modify core hoomd source code
1. *Particle data access*: User-level hoomd scripts can now directly access the particle data. For example, one can change all particles in the top half of the box to be type B:
<pre><code class="python">
top = group.cuboid(name="top", zmin=0)
for p in top:
    p.type = 'B'
</code></pre>
    . *All* particle data including position, velocity, type, ''et cetera'', can be read and written in this manner. Computed forces and energies can also be accessed in a similar way.
1. *New script command*: init.create_empty() can be used in conjunction with the particle data access above to completely initialize a system within the hoomd script.
1. *New script command*: dump.bin() writes full binary restart files with the entire system state, including the internal state of integrators.
    - File output can be gzip compressed (if zlib is available) to save space
    - Output can alternate between two different output files for safe crash recovery
1. *New script command*: init.read_bin() reads restart files written by dump.bin()
1. *New option*: run() now accepts a quiet option. When True, it eliminates the status information printouts that go to stdout.
1. *New example script*: Example 6 demonstrates the use of the particle data access routines to initialize a system. It also demonstrates how to initialize velocities from a gaussian distribution
1. *New example script*: Example 7 plots the pair.lj potential energy and force as evaluated by hoomd. It can trivially be modified to plot any potential in hoomd.
1. *New feature*: Two dimensional simulations can now be run in hoomd: #259
1. *New pair potential*: Morse potential for particles of varying diameters (pair.morse)
1. *New command*: run_upto will run a simulation up to a given time step number (handy for breaking long simulations up into many independent jobs)
1. *New feature*: HOOMD on the CPU is now accelerated with OpenMP.
1. *New feature*: integrate.mode_minimize_fire performs energy minimization using the FIRE algorithm
1. *New feature*: analyze.msd can now accept an xml file specifying the initial particle positions (for restarting jobs)
1. *Improved feature*: analyze.imd now supports all IMD commands that VMD sends (pause, kill, change trate, etc.)
1. *New feature*: Pair potentials can now be given names, allowing multiple potentials of the same type to be logged separately. Additionally, potentials that are disabled and not applied to the system dynamics can be optionally logged.
1. *Performance improvements*: Simulation performance has been increased across the board, but especially when running systems with very low particle number densities.
1. *New hardware support*: 0.9.0 and newer support Fermi GPUs
1. *Deprecated hardware support*: 0.9.x might continue run on compute#1 GPUs but that hardware is no longer officially supported
1. *New script command*: group.tag_list() takes a python list of particle tags and creates a group
1. *New script command*: compute.thermo() computes thermodynamic properties of a group of particles for logging
1. *New feature*: dump.dcd can now optionally write out only those particles that belong to a specified group

*Changes that will break jobs scripts written for 0.8.x*

1. Integration routines have changed significantly to enable new use cases. Where scripts previously had commands like:
<pre><code class="python">
integrate.nve(dt=0.005)
</code></pre>
    they now need
<pre><code class="python">
all = group.all()
integrate.mode_standard(dt=0.005)
integrate.nve(group=all)
</code></pre>
    . Integrating only specific groups of particles enables simulations to fix certain particles in place or integrate different parts of the system at different temperatures, among many other possibilities.
1. sorter.set_params no longer takes the ''bin_width'' argument. It is replaced by a new ''grid'' argument, see the documentation for details.
1. conserved_quantity is no longer a quantity available for logging. Instead log the nvt reservoir energy and compute the total conserved quantity in post processing.

*Bug/fixes small enhancements*

1. Fixed a bug where boost#38 is not found on some machines
1. dump.xml now has an option to write particle accelerations
1. Fixed a bug where periods like 1e6 were not accepted by updaters
1. Fixed a bug where bond.fene forces were calculated incorrectly between particles of differing diameters
1. Fixed a bug where bond.fene energies were computed incorrectly when running on the GPU
1. Fixed a bug where comments in hoomd xml files were not ignored as they aught to be: #331
1. It is now possible to prevent bond exclusions from ever being added to the neighbor list: #338
1. init.create_random_polymers can now generate extremely dense systems and will warn the user about large memory usage
1. variant.linear_interp now accepts a user-defined zero (handy for breaking long simulations up into many independent jobs)
1. Improved installation and compilation documentation
1. Integration methods now silently ignore when they are given an empty group
1. Fixed a bug where disabling all forces resulted in some forces still being applied
1. Integrators now behave in a reasonable way when given empty groups
1. Analyzers now accept a floating point period
1. run() now aborts immediately if limit_hours=0 is specified.
1. Pair potentials that diverge at r=0 will no longer result in invalid simulations when the leading coefficients are set to zero.
1. integrate.bdnvt can now tally the energy transferred into/out of the "reservoir", allowing energy conservation to be monitored during bd simulation runs.
1. Most potentials now prevent NaN results when computed for overlapping particles
1. Stopping a simulation from a callback or time limit no longer produces invalid simulations when continued
1. run() commands limited with limit_hours can now be set to only stop on given timestep multiples
1. Worked around a compiler bug where pair.morse would crash on Fermi GPUs
1. ULF stability improvements for G200 GPUs.


## Version 0.8.2

2009-09-10

*Note:* only major changes are listed here.

*New features*

1. Quantities that vary over time can now be specified easily in scripts with the variant.linear_interp command.
1. Box resizing updater (update.box_resize) command that uses the time varying quantity command to grow or shrink the simulation box.
1. Individual run() commands can be limited by wall-clock time
1. Angle forces can now be specified
1. Dihedral forces can now be specified
1. Improper forces can now be specified
1. 1-3 and 1-4 exclusions from the cutoff pair force can now be chosen
1. New command line option: --minimize-cpu-usage cuts the CPU usage of HOOMD down to 10% of one CPU core while only decreasing overall performance by 10%
1. Major changes have been made in the way HOOMD chooses the device on which to run (all require CUDA 2.2 or newer)
   * there are now checks that an appropriate NVIDIA drivers is installed
   * running without any command line options will now correctly revert to running on the CPU if no capable GPUs are installed
   * when no gpu is explicitly specified, the default choice is now prioritized to choose the fastest GPU and one that is not attached to a display first
   * new command line option: --ignore-display-gpu will prevent HOOMD from executing on any GPU attached to a display
   * HOOMD now prints out a short description of the GPU(s) it is running on
   * on linux, devices can be set to compute-exclusive mode and HOOMD will then automatically choose the first free GPU (see the documentation for details)
1. nlist.reset_exclusions command to control the particles that are excluded from the neighbor list


*Bug/fixes small enhancements*

1. Default block size change to improve stability on compute#3 devices
1. ULF workaround on GTX 280 now works with CUDA 2.2
1. Standalone benchmark executables have been removed and replaced by in script benchmarking commands
1. Block size tuning runs can now be performed automatically using the python API and results can be saved on the local machine
1. Fixed a bug where GTX 280 bug workarounds were not properly applied in CUDA 2.2
1. The time step read in from the XML file can now be optionally overwritten with a user-chosen one
1. Added support for CUDA 2.2
1. Fixed a bug where the WCA forces included in bond.fene had an improper cutoff
1. Added support for a python callback to be executed periodically during a run()
1. Removed demos from the hoomd downloads. These will be offered separately on the webpage now to keep the required download size small.
1. documentation improvements
1. Significantly increased performance of dual-GPU runs when build with CUDA 2.2 or newer
1. Numerous stability and performance improvements
1. Temperatures are now calculated based on 3N-3 degrees of freedom. See #283 for a more flexible system that is coming in the future.
1. Emulation mode builds now work on systems without an NVIDIA card (CUDA 2.2 or newer)
1. HOOMD now compiles with CUDA 2.3
1. Fixed a bug where uninitialized memory was written to dcd files
1. Fixed a bug that prevented the neighbor list on the CPU from working properly with non-cubic boxes
1. There is now a compile time hack to allow for more than 4 exclusions per particle
1. Documentation added to aid users in migrating from LAMMPS
1. hoomd_script now has an internal version number useful for third party scripts interfacing with it
1. VMD#8.7 is now found by the live demo scripts
1. live demos now run in vista 64-bit
1. init.create_random_polymers can now create polymers with more than one type of bond

## Version 0.8.1

2009-03-24

*Note:* only major changes are listed here.

*New features*

1. Significant performance enhancements
1. New build option for compiling on UMich CAC clusters: ENABLE_CAC_GPU_ID compiles HOOMD to read in the *$CAC_GPU_ID* environment variable and use it to determine which GPUs to execute on. No --gpu command line required in job scripts any more.
1. Particles can now be assigned a *non-unit mass*
1. *init.reset()* command added to allow for the creation of a looped series of simulations all in python
1. *dump.pdb()* command for writing PDB files
1. pair.lj now comes with an option to *shift* the potential energy to 0 at the cutoff
1. pair.lj now comes with an opiton to *smoothly switch* both the *potential* and *force* to 0 at the cutoff with the XPLOR smoothing function
1. *Gaussian pair potential* computation added (pair.gauss)
1. update and analyze commands can now be given a function to determine a non-linear rate to run at
1. analyze.log, and dump.dcd can now append to existing files

*Changes that will break scripts from 0.8.0*

1. *dump.mol2()* has been changed to be more consistent with other dump commands. In order to get the same result as the previous behavior, replace
<pre><code class="python">
 dump.mol2(filename="file.mol2")
</code></pre>
 with
 <pre><code class="python">
 mol2 = dump.mol2()
 mol2.write(filename="file.mol2")
</code></pre>
1. Grouping commands have been moved to their own package for organizational purposes. *group_all()* must now be called as *group.all()* and similarly for tags and type.

*Bug/fixes small enhancements*

1. Documentation updates
1. DCD file writing no longer crashes HOOMD in windows
1. !FindBoost.cmake is patched upstream. Use CMake 2.6.3 if you need BOOST_ROOT to work correctly
1. Validation tests now run with --gpu_error_checking
1. ULF bug workarounds are now enabled only on hardware where they are needed. This boosts performance on C1060 and newer GPUs.
1. !FindPythonLibs now always finds the shared python libraries, if they exist
1. "make package" now works fine on mac os x
1. Fixed erroneously reported dangerous neighbor list builds when using --mode=cpu
1. Small tweaks to the XML file format.
1. Numerous performance enhancements
1. Workaround for ULF on compute#1 devices in place
1. dump.xml can now be given the option "all=true" to write all fields
1. total momentum can now be logged by analyze.log
1. HOOMD now compiles with boost#38 (and hopefully future versions)
1. Updaters can now be given floating point periods such as 1e5
1. Additional warnings are now printed when HOOMD is about to allocate a large amount of memory due to the specification of an extremely large box size
1. run() now shows up in the documentation index
1. Default sorter period is now 100 on CPUs to improve performance on chips with small caches


## Version 0.8.0

2008-12-22

*Note:* only major changes are listed here.

*New features*

1. Addition of FENE bond potential
1. Addition of update.zero_momentum command to zero a system's linear momentum
1. Brownian dynamics integration implemented
1. Multi-GPU simulations
1. Particle image flags are now tracked. analyze.msd command added to calculate the mean squared displacement.

*Changes that will break scripts from 0.7.x*

1. analyze.log quantity names have changed

*Bug/fixes small enhancements*

1. Performance of the neighbor list has been increased significantly on the GPU (overall performance improvements are approximately 10%)
1. Profile option added to the run() command
1. Warnings are now correctly printed when negative coefficients are given to bond forces
1. Simulations no longer fail on G200 cards
1. Mac OS X binaries will be provided for download: new documentation for installing on Mac OS x has been written
1. Two new demos showcasing large systems
1. Particles leaving the simulation box due to bad initial conditions now generate an error
1. win64 installers will no longer attempt to install on win32 and vice-versa
1. neighborlist check_period now defaults to 1
1. The elapsed time counter in run() now continues counting time over multiple runs.
1. init.create_random_polymers now throws an error if the bond length is too small given the specified separation radii
1. Fixed a bug where a floating point value for the count field in init.create_random_polymers produced an error
1. Additional error checking to test if particles go NaN
1. Much improved status line printing for identifying hoomd_script commands
1. Numerous documentation updates
1. The VS redistributable package no longer needs to be installed to run HOOMD on windows (these files are distributed with HOOMD)
1. Now using new features in doxygen#5.7 to build pdf user documentation for download.
1. Performance enhancements of the Lennard-Jones pair force computation, thanks to David Tarjan
1. A header prefix can be added to log files to make them more gnuplot friendly
1. Log quantities completely revamped. Common quantities (i.e. kinetic energy, potential energy can now be logged in any simulation)
1. Particle groups can now be created. Currently only analyze.msd makes use of them.
1. The CUDA toolkit no longer needs to be installed to run a packaged HOOMD binary in windows.
1. User documentation can now be downloaded as a pdf.
1. Analyzers and updaters now count time 0 as being the time they were created, instead of time step 0.
1. Added job test scripts to aid in validating HOOMD
1. HOOMD will now build with default settings on a linux/unix-like OS where the boost static libraries are not installed, but the dynamic ones are.

----

## Version 0.7.1

2008-09-12

1. Fixed bug where extremely large box dimensions resulted in an argument error - ticket:118
1. Fixed bug where simulations ran incorrectly with extremely small box dimensions - ticket:138

----

## Version 0.7.0

2008-08-12

*Note:* only major changes are listed here.

1. Stability and performance improvements.
1. Cleaned up the hoomd_xml file format.
1. Improved detection of errors in hoomd_xml files significantly.
1. Users no longer need to manually specify HOOMD_ROOT, unless their installation is non-standard
1. Particle charge can now be read in from a hoomd_xml file
1. Consistency changes in the hoomd_xml file format: HOOMD 0.6.0 XML files are not compatible. No more compatibility breaking changes are planned after 0.7.0
1. Enabled parallel builds in MSVC for faster compilation times on multicore systems
1. Numerous small bug fixes
1. New force compute for implementing walls
1. Documentation updates
1. Support for CUDA 2.0
1. Bug fixed allowing simulations with no integrator
1. Support for boost#35.0
1. Cleaned up GPU code interface
1. NVT integrator now uses tau (period) instead of Q (the mass of the extra degree of freedom).
1. Added option to NVE integration to limit the distance a particle moves in a single time step
1. Added code to dump system snapshots in the DCD file format
1. Particle types can be named by strings
1. A snapshot of the initial configuration can now be written in the .mol2 file format
1. The default build settings now enable most of the optional features
1. Separated the user and developer documentation
1. Mixed polymer systems can now be generated inside HOOMD
1. Support for CMake 2.6.0
1. Wrote the user documentation
1. GPU selection from the command line
1. Implementation of the job scripting system
1. GPU can now handle neighbor lists that overflow
1. Energies are now calculated
1. Added a logger for logging energies during a simulation run
1. Code now actually compiles on Mac OS X
1. Benchmark and demo scripts now use the new scripting system
1. Consistent error message format that is more visible.
1. Multiple types of bonds each with the own coefficients are now supported
1. Added python scripts to convert from HOOMD's XML file format to LAMMPS input and dump files
1. Fixed a bug where empty xml nodes in input files resulted in an error message
1. Fixed a bug where HOOMD seg faulted when a particle left the simulation , vis=True)* is now a convenient short hand for what was previously
box now works fine on mac os x
1. Fixed erroneously reported dangerous neighbor list builds when using --mode=cpu
1. Small tweaks to the XML file format.
1. Numerous performance enhancements
1. Workaround for ULF on compute#1 devices in place
1. dump.xml can now be given the option
