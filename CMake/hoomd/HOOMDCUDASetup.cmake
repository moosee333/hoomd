# Maintainer: joaander

##################################
## Find CUDA
# If CUDA is enabled, set it up
if (ENABLE_CUDA)
	find_package(CUDA 7.0 REQUIRED)
    find_package(Thrust 1.5.0 REQUIRED)

    # first thrust, then CUDA (to allow for local thrust installation
    # that overrides CUDA toolkit)
    include_directories(${THRUST_INCLUDE_DIR})
	include_directories(${CUDA_INCLUDE_DIRS})

    get_directory_property(DIRS INCLUDE_DIRECTORIES SYSTEM)
    # hide some variables users don't need to see
    mark_as_advanced(CUDA_SDK_ROOT_DIR)
    if (CUDA_TOOLKIT_ROOT_DIR)
        mark_as_advanced(CUDA_TOOLKIT_ROOT_DIR)
    endif (CUDA_TOOLKIT_ROOT_DIR)
    mark_as_advanced(CUDA_VERBOSE_BUILD)
    mark_as_advanced(CUDA_BUILD_EMULATION)
    mark_as_advanced(CUDA_HOST_COMPILER)
    mark_as_advanced(CUDA_dl_LIBRARY)
    mark_as_advanced(CUDA_rt_LIBRARY)
    mark_as_advanced(THRUST_INCLUDE_DIR)

    if (ENABLE_NVTOOLS)
        find_library(CUDA_nvToolsExt_LIBRARY
                     NAMES nvToolsExt
                     PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
                           "${CUDA_TOOLKIT_ROOT_DIR}/lib"
                     ENV CUDA_LIB_PATH
                     DOC "nvTools library"
                     NO_DEFAULT_PATH
                     )

        mark_as_advanced(CUDA_nvToolsExt_LIBRARY)
    endif()
endif (ENABLE_CUDA)

# setup CUDA compile options
if (ENABLE_CUDA)
    # setup nvcc to build for all CUDA architectures. Allow user to modify the list if desired
    if (CUDA_VERSION VERSION_GREATER 7.99)
        set(CUDA_ARCH_LIST 30 35 50 60 CACHE STRING "List of target sm_ architectures to compile CUDA code for. Separate with semicolons.")
    elseif (CUDA_VERSION VERSION_GREATER 6.99)
        set(CUDA_ARCH_LIST 30 35 50 CACHE STRING "List of target sm_ architectures to compile CUDA code for. Separate with semicolons.")
    endif()

    foreach(_cuda_arch ${CUDA_ARCH_LIST})
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${_cuda_arch},code=sm_${_cuda_arch}")
    endforeach (_cuda_arch)

    # need to know the minumum supported CUDA_ARCH
    set(_cuda_arch_list_sorted ${CUDA_ARCH_LIST})
    list(SORT _cuda_arch_list_sorted)
    list(GET _cuda_arch_list_sorted 0 _cuda_min_arch)
    add_definitions(-DCUDA_ARCH=${_cuda_min_arch})

    if (_cuda_min_arch LESS 20)
        message(SEND_ERROR "SM1x builds are not supported")
    endif ()

    # only generage ptx code for the maximum supported CUDA_ARCH (saves on file size)
    list(REVERSE _cuda_arch_list_sorted)
    list(GET _cuda_arch_list_sorted 0 _cuda_max_arch)
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${_cuda_max_arch},code=compute_${_cuda_max_arch}")

# embed the CUDA libraries into the lib dir
if (ENABLE_EMBED_CUDA)
    # determine the directory of the found cuda libs
    get_filename_component(_cuda_libdir ${CUDA_CUDART_LIBRARY} PATH)
    FILE(GLOB _cuda_libs ${_cuda_libdir}/libcurand.* ${_cuda_libdir}/libcufft.* ${_cuda_libdir}/libcusolver.* ${_cuda_libdir}/libcusparse.*)
    install(PROGRAMS ${_cuda_libs} DESTINATION ${PYTHON_MODULE_BASE_DIR})
endif ()

endif (ENABLE_CUDA)

# set CUSOLVER_AVAILABLE depending on CUDA Toolkit version
if (ENABLE_CUDA)
    if(${CUDA_VERSION} VERSION_LESS 7.5)
        set(CUSOLVER_AVAILABLE FALSE)
    elseif(${CUDA_VERSION} VERSION_LESS 8.0)
        # CUDA 7.5 has a functioning cusolver, if cmake found it
        if (NOT ${CUDA_cusolver_LIBRARY} STREQUAL "")
            set(CUSOLVER_AVAILABLE TRUE)
        else()
            set(CUSOLVER_AVAILABLE FALSE)
        endif()
    else()
        # CUDA 8.0 requires that libgomp be linked in - see if we can link it
        try_compile(_can_link_gomp
                    ${CMAKE_CURRENT_BINARY_DIR}/tmp
                    ${CMAKE_CURRENT_LIST_DIR}/test.cc
                    LINK_LIBRARIES gomp
                   )

        if (NOT ${CUDA_cusolver_LIBRARY} STREQUAL "" AND _can_link_gomp)
            set(CUSOLVER_AVAILABLE TRUE)
        else()
            set(CUSOLVER_AVAILABLE FALSE)
        endif()
    endif()

if (NOT CUSOLVER_AVAILABLE)
    message(STATUS "Could not find cusolver library, constraints will be slower. Perhaps old CMake or missing gomp library.")
else()
    message(STATUS "Found cusolver: ${CUDA_cusolver_LIBRARY}")
endif()

endif()
