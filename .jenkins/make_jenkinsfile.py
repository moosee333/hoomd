import jinja2

env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))
template = env.get_template('Jenkinsfile.jinja')

tests = []
tests.append(dict(name='unit-gcc49-py27-cd75',
                  agent='gpu',
                  CC = '/usr/bin/gcc-4.9',
                  CXX = '/usr/bin/g++-4.9',
                  PYVER = '2.7',
                  CMAKE_BIN = '/usr/bin',
                  ENABLE_CUDA = 'ON',
                  ENABLE_MPI = 'ON',
                  ENABLE_TBB = 'OFF',
                  BUILD_VALIDATION = 'OFF',
                  CONTAINER = 'ci-20171130-cuda75.img',
                  BUILD_JIT = 'OFF',
                  timeout=1))

tests.append(dict(name='unit-clang38-py35-cd80',
                  agent='gpu',
                  CC = '/usr/bin/clang',
                  CXX = '/usr/bin/clang++',
                  PYVER = '3.5',
                  CMAKE_BIN = '/usr/bin',
                  ENABLE_CUDA = 'ON',
                  ENABLE_MPI = 'OFF',
                  ENABLE_TBB = 'OFF',
                  BUILD_VALIDATION = 'OFF',
                  CONTAINER = 'ci-20171130-cuda80.img',
                  BUILD_JIT = 'ON',
                  timeout=1))

tests.append(dict(name='vld-gcc6-py36-mpi-tbb-cd90',
                  agent='gpu',
                  CC = '/usr/sbin/gcc-6',
                  CXX = '/usr/sbin/g++-6',
                  PYVER = '3.6',
                  CMAKE_BIN = '/usr/sbin',
                  ENABLE_CUDA = 'ON',
                  ENABLE_MPI = 'ON',
                  ENABLE_TBB = 'OFF',
                  BUILD_VALIDATION = 'OFF',
                  CONTAINER = 'ci-20171206-arch-2.img',
                  BUILD_JIT = 'ON',
                  timeout=15))

tests.append(dict(name='vld-clang50-py36-mpi-tbb',
                  agent='linux-cpu',
                  CC = '/usr/sbin/clang',
                  CXX = '/usr/sbin/clang++',
                  PYVER = '3.6',
                  CMAKE_BIN = '/usr/sbin',
                  ENABLE_CUDA = 'OFF',
                  ENABLE_MPI = 'ON',
                  ENABLE_TBB = 'ON',
                  BUILD_VALIDATION = 'ON',
                  CONTAINER = 'ci-20171206-arch-2.img',
                  BUILD_JIT = 'ON',
                  timeout=15))

print(template.render(tests=tests))
