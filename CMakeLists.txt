#
# Copyright (C) 2022 Embecosm Limited
#
# Contributor William Jones <william.jones@embecosm.com>
# Contributor Elliot Stein <E.Stein@soton.ac.uk>
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

cmake_minimum_required(VERSION 3.16)

project(dcm_demo LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

# Options
option(BUILD_DOCS "Build docs using Sphinx and Doxygen" ON)
option(STATIC_BUILD "Build static binaries" OFF)
option(USE_OPENMP "Build with OpenMP for Eigen parallelisation" ON)
option(OPTIMIZE_FASTMATH_NATIVE "Build with -march=native and -ffast-math" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Basic CMake setup
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

if(OPTIMIZE_FASTMATH_NATIVE)
  set(FAST_MATH_FLAGS -ffast-math -fno-unsafe-math-optimizations)
  CHECK_CXX_COMPILER_FLAG("${FAST_MATH_FLAGS}" COMPILER_SUPPORTS_FAST_MATH)
  CHECK_CXX_COMPILER_FLAG(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)

  if (COMPILER_SUPPORTS_FAST_MATH)
    add_compile_options(${FAST_MATH_FLAGS})
  endif()

  if (COMPILER_SUPPORTS_MARCH_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

if(STATIC_BUILD)
  set(BUILD_SHARED_LIBS OFF)
  link_libraries(-static-libgcc -static-libstdc++)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
endif()

# Configure for OpenMP
if(USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
    add_compile_definitions(_GLIBCXX_PARALLEL)
  endif()
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(dcEmb)

# Includes
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/COVID
    ${CMAKE_SOURCE_DIR}/include/3body
    ${CMAKE_SOURCE_DIR}/include/weather
    ${CMAKE_SOURCE_DIR}/include/tests
)

# Sources
set(SOURCES_COVID
    src/COVID/run_COVID_dcm.cc
    src/COVID/DEM_COVID.cc
    src/COVID/import_COVID.cc
    src/COVID/dynamic_COVID_model.cc
    src/utility.cc
)

set(SOURCES_3BODY
    src/3body/run_3body_dcm.cc
    src/3body/DEM_3body.cc
    src/3body/dynamic_3body_model.cc
    src/utility.cc
)

set(SOURCES_WEATHER
    src/weather/run_weather_dcm.cc
    src/weather/DEM_weather.cc
    src/weather/dynamic_weather_model.cc
    src/utility.cc
)

# Binaries
set(EXE_LIBS Eigen3::Eigen dcEmb)
add_executable(dcm_covid ${SOURCES_COVID})
add_executable(dcm_3body ${SOURCES_3BODY})
add_executable(dcm_weather ${SOURCES_WEATHER})
target_link_libraries(dcm_covid PUBLIC ${EXE_LIBS})
target_link_libraries(dcm_3body PUBLIC ${EXE_LIBS})
target_link_libraries(dcm_weather PUBLIC ${EXE_LIBS})

if(BUILD_DOCS)
 # look for Doxygen package
 find_package(Doxygen)

 if (DOXYGEN_FOUND)
   # set input and output files
   set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/source/doxygen/Doxyfile.in)
   set(DOXYGEN_OUT ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.out)
   message(${DOXYGEN_IN})
   # request to configure the file
   configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
   message("Doxygen build started")

   # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
   add_custom_target( doxy
     COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
     COMMENT "Generating API documentation with Doxygen"
     VERBATIM )
 else (DOXYGEN_FOUND)
   message("Doxygen need to be installed to generate the doxygen documentation")
 endif (DOXYGEN_FOUND)

 include(UseSphinxDoc)
 include(FindSphinx)

 add_sphinx_doc(
   SOURCE_DIR
     ${CMAKE_CURRENT_SOURCE_DIR}/docs/source/sphinx/source
   BUILD_DIR
     ${CMAKE_CURRENT_SOURCE_DIR}/docs/
   CACHE_DIR
     ${CMAKE_CURRENT_SOURCE_DIR}/docs/.doctrees
   HTML_DIR
     ${CMAKE_CURRENT_SOURCE_DIR}/docs/
   CONF_FILE
     ${CMAKE_CURRENT_SOURCE_DIR}/docs/source/sphinx/source/conf.py
   TARGET_NAME
     sphinx
   COMMENT
     "HTML documentation"
   )

 # Documentation
 add_custom_target(docs)
 add_dependencies(docs doxy sphinx)
endif()
