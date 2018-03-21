# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: csadorf / All Developers are free to add commands for new features

R""" Manage execution contexts.

Every hoomd simulation needs an execution context that describes what hardware it should execute on,
the MPI configuration for the job, etc...
"""

import os
import hoomd
from hoomd import _hoomd
from hoomd import cite
import socket
import getpass
import platform

# The following global variables keep track of the walltime and processing time since the import of hoomd_script
import time
TIME_START = time.time()
CLOCK_START = time.clock()

## Global Messenger
# \note This is initialized to a default messenger on load so that python code may have a unified path for sending
# messages
msg = _hoomd.Messenger();

# only use python stdout/stderr in non-mpi runs
if not (  'OMPI_COMM_WORLD_RANK' in os.environ
        or 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ
        or 'PMI_RANK' in os.environ
        or 'ALPS_APP_PE' in os.environ):
    msg.openPython();

## Global bibliography
bib = None;

## Global options
options = None;

## Global variable that holds the execution configuration for reference by the python API
exec_conf = None;

## Current simulation context
current = None;

_prev_args = None;

class SimulationContext(object):
    R""" Simulation context

    Store all of the context related to a single simulation, including the system state, forces, updaters, integration
    methods, and all other commands specified on this simulation. All such commands in hoomd apply to the currently
    active simulation context. You swap between simulation contexts by using this class as a context manager::


        sim1 = context.SimulationContext();
        sim2 = context.SimulationContext();
        with sim1:
          init.read_xml('init1.xml');
          lj = pair.lj(...)
          ...

        with sim2:
          init.read_xml('init2.xml');
          gauss = pair.gauss(...)
          ...

        # run simulation 1 for a bit
        with sim1:
           run(100)

        # run simulation 2 for a bit
        with sim2:
           run(100)

        # set_current sets the current context without needing to use with
        sim1.set_current()
        run(100)


    If you do not need to maintain multiple contexts, you can call `context.initialize()` to  initialize a new context
    and erase the existing one::

        context.initialize()
        init.read_xml('init1.xml');
        lj = pair.lj(...)
        ...
        run(100);

        context.initialize()
        init.read_xml('init2.xml');
        gauss = pair.gauss(...)
        ...
        run(100)

    Attributes:
        sorter (:py:class:`hoomd.update.sort`): Global particle sorter.
        system_definition (:py:class:`hoomd.data.system_data`): System definition.

    The attributes are global to the context. User scripts may access documented attributes to control settings,
    access particle data, etc... See the linked documentation of each attribute for more details. For example,
    to disable the global sorter::

        c = context.initialize();
        c.sorter.disable();

    """
    def __init__(self):
        ## Global variable that holds the SystemDefinition shared by all parts of hoomd_script
        self.system_definition = None;

        ## Global variable that holds the System shared by all parts of hoomd_script
        self.system = None;

        ## Global variable that holds the balanced domain decomposition in MPI runs if it is requested
        self.decomposition = None

        ## Global variable that holds the sorter
        self.sorter = None;

        ## Global variable that tracks the all of the force computes specified in the script so far
        self.forces = [];

        ## Global variable that tracks the all of the constraint force computes specified in the script so far
        self.constraint_forces = [];

        ## Global variable that tracks all the integration methods that have been specified in the script so far
        self.integration_methods = [];

        ## Global variable tracking the last _integrator set
        self.integrator = None;

        ## Global variable tracking all neighbor lists that have been created
        self.neighbor_lists = []

        ## Global variable tracking all the loggers that have been created
        self.loggers = [];

        ## Global variable tracking all the analyzers that have been created
        self.analyzers = [];

        ## Global variable tracking all the updaters that have been created
        self.updaters = [];

        ## Global variable tracking all the compute thermos that have been created
        self.thermos = [];

        ## Cached all group
        self.group_all = None;

        ## MPCD system
        self.mpcd = None;

        ## Stored reference to the reader that was used to initialize the system
        self.state_reader = None;

    def set_current(self):
        R""" Force this to be the current context
        """
        global current

        current = self;

    def on_gpu(self):
        R""" Test whether this job is running on a GPU.

        Returns:
            True if this invocation of HOOMD-blue is executing on a GPU. False if it is on the CPU.
        """
        global exec_conf
        return exec_conf.isCUDAEnabled()

    def __enter__(self):
        global current

        self.prev = current;
        current = self;

    def __exit__(self, exc_type, exc_value, traceback):
        global current

        current = self.prev;

def initialize(args=None):
    R""" Initialize the execution context

    Args:
        args (str): Arguments to parse. When *None*, parse the arguments passed on the command line.

    :py:func:`hoomd.context.initialize()` parses the command line arguments given, sets the options and initializes MPI and GPU execution
    (if any). By default, :py:func:`hoomd.context.initialize()` reads arguments given on the command line. Provide a string to :py:func:`hoomd.context.initialize()`
    to set the launch configuration within the job script.

    :py:func:`hoomd.context.initialize()` can be called more than once in a script. However, the execution parameters are fixed on the first call
    and *args* is ignored. Subsequent calls to :py:func:`hoomd.context.initialize()` create a new :py:class:`SimulationContext` and set it current. This
    behavior is primarily to support use of hoomd in jupyter notebooks, so that a new clean simulation context is
    set when rerunning the notebook within an existing kernel.

    Example::

        from hoomd import *
        context.initialize();
        context.initialize("--mode=gpu --nrank=64");
        context.initialize("--mode=cpu --nthreads=64");

    """
    global exec_conf, msg, options, current, _prev_args

    if exec_conf is not None:
        if args != _prev_args:
            msg.warning("Ignoring new options, cannot change execution mode after initialization.\n");
        current = SimulationContext();
        return current

    _prev_args = args;

    options = hoomd.option.options();
    hoomd.option._parse_command_line(args);

    # Check to see if we are built without MPI support and the user used mpirun
    if (not _hoomd.is_MPI_available() and not options.single_mpi
        and (    'OMPI_COMM_WORLD_RANK' in os.environ
              or 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ
              or 'PMI_RANK' in os.environ
              or 'ALPS_APP_PE' in os.environ)
       ):
        print('HOOMD-blue is built without MPI support, but seems to have been launched with mpirun');
        print('exiting now to prevent many sequential jobs from starting');
        raise RuntimeError('Error launching hoomd')

    # output the version info on initialization
    msg.notice(1, _hoomd.output_version_info())

    # ensure creation of global bibliography to print HOOMD base citations
    cite._ensure_global_bib()

    _create_exec_conf();

    current = SimulationContext();
    return current

## Get the current processor name
#
# platform.node() can spawn forked processes in some version of MPI.
# This avoids that problem by using MPI information about the hostname directly
# when it is available. MPI is initialized on module load if it is available,
# so this data is accessible immediately.
#
# \returns String name for the current processor
# \internal
def _get_proc_name():
    if _hoomd.is_MPI_available():
        return _hoomd.get_mpi_proc_name()
    else:
        return platform.node()

## Initializes the execution configuration
#
# \internal
def _create_exec_conf():
    global exec_conf, options, msg

    # use a cached execution configuration if available
    if exec_conf is not None:
        return exec_conf

    mpi_available = _hoomd.is_MPI_available();

    # error out on nyx/flux if the auto mode is set
    if options.mode == 'auto':
        host = _get_proc_name()
        if "flux" in host or "nyx" in host:
            msg.error("--mode=gpu or --mode=cpu must be specified on nyx/flux\n");
            raise RuntimeError("Error initializing");
        exec_mode = _hoomd.ExecutionConfiguration.executionMode.AUTO;
    elif options.mode == "cpu":
        exec_mode = _hoomd.ExecutionConfiguration.executionMode.CPU;
    elif options.mode == "gpu":
        exec_mode = _hoomd.ExecutionConfiguration.executionMode.GPU;
    else:
        raise RuntimeError("Invalid mode");

    # convert None options to defaults
    if options.gpu is None:
        gpu_id = -1;
    else:
        gpu_id = int(options.gpu);

    if options.nrank is None:
        nrank = 0;
    else:
        nrank = int(options.nrank);

    # create the specified configuration
    exec_conf = _hoomd.ExecutionConfiguration(exec_mode, gpu_id, options.min_cpu, options.ignore_display, msg, nrank);

    # if gpu_error_checking is set, enable it on the GPU
    if options.gpu_error_checking:
       exec_conf.setCUDAErrorChecking(True);

    if _hoomd.is_TBB_available():
        # set the number of TBB threads as necessary
        if options.nthreads != None:
            exec_conf.setNumThreads(options.nthreads)

    exec_conf = exec_conf;

    return exec_conf;

## \internal
# \brief Throw an error if the context is not initialized
def _verify_init():
    global exec_conf, msg, current

    if exec_conf is None:
        msg.error("call context.initialize() before any other method in hoomd.")
        raise RuntimeError("hoomd execution context is not available")

## \internal
# \brief Gather context from the environment
class ExecutionContext(hoomd.meta._metadata):
    ## \internal
    # \brief Constructs the context object
    def __init__(self):
        hoomd.meta._metadata.__init__(self)
        self.metadata_fields = [
            'hostname', 'gpu', 'mode', 'num_ranks',
            'username', 'wallclocktime', 'cputime',
            'job_id', 'job_name'
            ]
        if _hoomd.is_TBB_available():
            self.metadata_fields.append('num_threads')

    ## \internal
    # \brief Return the execution configuration if initialized or raise exception.
    def _get_exec_conf(self):
        global exec_conf
        if exec_conf is None:
            raise RuntimeError("Not initialized.")
        else:
            return exec_conf

    # \brief Return the network hostname.
    @property
    def hostname(self):
        return socket.gethostname()

    # \brief Return the name of the GPU used in GPU mode.
    @property
    def gpu(self):
        return self._get_exec_conf().getGPUName()

    # \brief Return the execution mode
    @property
    def mode(self):
        if self._get_exec_conf().isCUDAEnabled():
            return 'gpu';
        else:
            return 'cpu';

    # \brief Return the number of ranks.
    @property
    def num_ranks(self):
        return hoomd.comm.get_num_ranks()

    # \brief Return the username.
    @property
    def username(self):
        return getpass.getuser()

    # \brief Return the wallclock time since the import of hoomd_script
    @property
    def wallclocktime(self):
        return time.time() - TIME_START

    # \brief Return the CPU clock time since the import of hoomd_script
    @property
    def cputime(self):
        return time.clock() - CLOCK_START

    # \brief Return the job id
    @property
    def job_id(self):
        if 'PBS_JOBID' in os.environ:
            return os.environ['PBS_JOBID'];
        elif 'SLURM_JOB_ID' in os.environ:
            return os.environ['SLURM_JOB_ID'];
        else:
            return '';

    # \brief Return the job name
    @property
    def job_name(self):
        if 'PBS_JOBNAME' in os.environ:
            return os.environ['PBS_JOBNAME'];
        elif 'SLURM_JOB_NAME' in os.environ:
            return os.environ['SLURM_JOB_NAME'];
        else:
            return '';

    # \brief Return the number of CPU threads
    @property
    def num_threads(self):
        if not _hoomd.is_TBB_available():
            msg.warning("HOOMD was compiled without thread support, returning None\n");
            return None
        else:
            return self._get_exec_conf().getNumThreads();

## \internal
# \brief Gather context about HOOMD
class HOOMDContext(hoomd.meta._metadata):
    ## \internal
    # \brief Constructs the context object
    def __init__(self):
        hoomd.meta._metadata.__init__(self)
        self.metadata_fields = [
            'hoomd_version', 'hoomd_git_sha1', 'hoomd_git_refspec',
            'hoomd_compile_flags', 'cuda_version', 'compiler_version',
            ]

    # \brief Return the hoomd version.
    @property
    def hoomd_version(self):
        return _hoomd.__version__

    # \brief Return the hoomd git hash
    @property
    def hoomd_git_sha1(self):
        return _hoomd.__git_sha1__

    # \brief Return the hoomd git refspec
    @property
    def hoomd_git_refspec(self):
        return _hoomd.__git_refspec__

    # \brief Return the hoomd compile flags
    @property
    def hoomd_compile_flags(self):
        return _hoomd.hoomd_compile_flags();

    # \brief Return the cuda version
    @property
    def cuda_version(self):
        return _hoomd.__cuda_version__

    # \brief Return the compiler version
    @property
    def compiler_version(self):
        return _hoomd.__compiler_version__
