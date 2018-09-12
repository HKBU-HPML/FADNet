include(ExternalProject)

ExternalProject_Add(
        webp
        PREFIX "${CMAKE_BINARY_DIR}/webp"
        URL "http://downloads.webmproject.org/releases/webp/libwebp-0.6.0.tar.gz"
        # do not update
        UPDATE_COMMAND ""
        # compile with PIC because caffe builds a shared object
        CONFIGURE_COMMAND CFLAGS=-fPIC ../webp/configure --disable-gl --disable-png --disable-jpeg  --disable-tiff --disable-gif --disable-wic --disable-shared
        BUILD_COMMAND "make"
        # do not install
        INSTALL_COMMAND ""
)
ExternalProject_Get_Property( webp SOURCE_DIR BINARY_DIR )
set( webp_INCLUDE_DIR "${SOURCE_DIR}/src" )
set( webp_LIBRARY "${BINARY_DIR}/src/.libs/libwebp.a" )

# lz4
ExternalProject_Add(
        lz4
        PREFIX "${CMAKE_BINARY_DIR}/lz4"
        URL "https://github.com/lz4/lz4/archive/r131.tar.gz"
        # do not update
        UPDATE_COMMAND ""
        # fix missing CMakeLists.txt in source root dir
        PATCH_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/patches/lz4_CMakeLists.txt <SOURCE_DIR>/CMakeLists.txt
        # do not install
        INSTALL_COMMAND ""
)
ExternalProject_Get_Property( lz4 SOURCE_DIR BINARY_DIR )
set( lz4_INCLUDE_DIR "${SOURCE_DIR}/lib" )
set( lz4_STATIC_LIB ${BINARY_DIR}/cmake_unofficial/liblz4.a )


find_package( PythonInterp REQUIRED )
find_package( OpenMP REQUIRED )


# retrieve tensorflow include dir and lib dir
execute_process( COMMAND ${PYTHON_EXECUTABLE} "-c" "from __future__ import print_function; import tensorflow as tf; print(tf.sysconfig.get_include(), end='')" 
        OUTPUT_VARIABLE TENSORFLOW_INCLUDE_DIR )
execute_process( COMMAND ${PYTHON_EXECUTABLE} "-c" "from __future__ import print_function; import tensorflow as tf; print(tf.sysconfig.get_lib(), end='')" 
        OUTPUT_VARIABLE TENSORFLOW_LIB_DIR )
find_library( TENSORFLOW_FRAMEWORK_LIB tensorflow_framework PATHS "${TENSORFLOW_LIB_DIR}" NO_DEFAULT_PATH )
message( STATUS "${TENSORFLOW_INCLUDE_DIR}" )
message( STATUS "${TENSORFLOW_LIB_DIR}" )
message( STATUS "${TENSORFLOW_FRAMEWORK_LIB}" )

configure_file( config.h.in config.h )
if( BUILD_WITH_CUDA )
        find_package( CUDA REQUIRED )
        include( "${CMAKE_SOURCE_DIR}/cmake/cuda_gencode_options.cmake" )
        list( APPEND CUDA_NVCC_FLAGS "-std=c++11")
        list( APPEND CUDA_NVCC_FLAGS "${CUDA_GENCODE_STRING}" )
        list( APPEND CUDA_NVCC_FLAGS_DEBUG -g -G -O0 --ptxas-options=-v )

endif()

file( GLOB lmbspecialops_SOURCES  *.cc *.c )
file( GLOB lmbspecialops_CUSOURCES *.cu )


include_directories( 
    ${TENSORFLOW_INCLUDE_DIR}
    ${TENSORFLOW_INCLUDE_DIR}/external/nsync/public/  # fix nsync headers not found bug
        ${PROJECT_BINARY_DIR}/lib
)

if( BUILD_WITH_CUDA )
        cuda_add_library( lmbspecialops SHARED ${lmbspecialops_SOURCES} ${lmbspecialops_CUSOURCES} )
else()
        add_library( lmbspecialops SHARED ${lmbspecialops_SOURCES} )
endif()
# do not add 'lib' prefix
set_target_properties( lmbspecialops PROPERTIES PREFIX "" )
set_target_properties( lmbspecialops PROPERTIES DEBUG_POSTFIX "_debug" )
set_target_properties( lmbspecialops PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}" )
set_target_properties( lmbspecialops PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}" )
# use old ABI with gcc 5.x
set_target_properties( lmbspecialops PROPERTIES COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=0" )
# enable c++11
set_target_properties( lmbspecialops PROPERTIES 
                CXX_STANDARD 11 
                CXX_STANDARD_REQUIRED ON 
                CXX_EXTENSIONS OFF)

add_dependencies( lmbspecialops lz4 webp )

target_include_directories( lmbspecialops PUBLIC
        ${lz4_INCLUDE_DIR}
        ${webp_INCLUDE_DIR}
)

target_link_libraries( lmbspecialops
        ${lz4_STATIC_LIB}
        ${webp_LIBRARY}
)
if( TENSORFLOW_FRAMEWORK_LIB )
        target_link_libraries( lmbspecialops ${TENSORFLOW_FRAMEWORK_LIB} )
endif()
