# Maintainer: jglaser

set(_solvent_sources module-solvent.cc
                   )

set(_solvent_headers                )

if (ENABLE_CUDA)
list(APPEND _solvent_sources )
endif()

set(_solvent_cu_sources                      )

# Need to define NO_IMPORT_ARRAY in every file but module.cc
set_source_files_properties(${_solvent_sources} ${_solvent_cu_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_solvent_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

add_library (_solvent SHARED ${_solvent_sources} ${_CUDA_GENERATED_FILES} ${DFFT_SOURCES})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_solvent PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_solvent PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

##################################
# change the name of the library to be a valid python module
# tweak the properties of the output to make a functional python module
set_target_properties(_solvent PROPERTIES PREFIX "" OUTPUT_NAME "_solvent")

# .dylib is not recognized as a python module by python on Mac OS X
if(APPLE)
    set_target_properties(_solvent PROPERTIES SUFFIX ".so")
endif(APPLE)
fix_cudart_rpath(_solvent)

# install the library
install(TARGETS _solvent
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/solvent
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/solvent/${file}"
    )
ENDMACRO(copy_file)

set(files __init__.py
          angle.py
          bond.py
          charge.py
          constrain.py
          dihedral.py
          external.py
          force.py
          improper.py
          integrate.py
          nlist.py
          pair.py
          update.py
          wall.py
          special_pair.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/solvent
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_solvent ALL DEPENDS ${files})

# copy python modules to the build directory to make it a working python package
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/../include/hoomd/solvent/${file}
        COMMENT    "Copy hoomd/include/hoomd/solvent/${file}"
    )
ENDMACRO(copy_header)

# install headers in installation target
install(FILES ${_solvent_headers}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/solvent
       )

if (COPY_HEADERS)
# copy headers into build directory
foreach(file ${_solvent_headers})
    copy_header(${file})
endforeach()

add_custom_target(copy_solvent_headers ALL DEPENDS ${_solvent_headers})
endif()

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()
