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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(UseSphinxDoc)
include(FindSphinx)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}-std=c++14 -O3 -ffast-math -fno-unsafe-math-optimizations -flto -march=native\
 -D_GLIBCXX_PARALLEL -g")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}-O3 -g")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}-O3 -g")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(dcEmb)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/COVID
    ${CMAKE_SOURCE_DIR}/include/3body
    ${CMAKE_SOURCE_DIR}/include/weather
    ${CMAKE_SOURCE_DIR}/include/tests
)

set(SOURCES_COVID
    # src/dynamic_model.cc
    src/COVID/run_COVID_dcm.cc
    src/COVID/DEM_COVID.cc
    src/COVID/import_COVID.cc
    src/COVID/dynamic_COVID_model.cc
    src/utility.cc
)

set(SOURCES_3BODY
    # src/dynamic_model.cc
    src/3body/run_3body_dcm.cc
    src/3body/DEM_3body.cc
    src/3body/dynamic_3body_model.cc
    src/utility.cc
)

set(SOURCES_WEATHER
    # src/dynamic_model.cc
    src/weather/run_weather_dcm.cc
    src/weather/DEM_weather.cc
    src/weather/dynamic_weather_model.cc
    src/utility.cc
)



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

# COVID-19 Model
add_executable(dcm_covid ${SOURCES_COVID})

if(OpenMP_FOUND)
    target_link_libraries(dcm_covid PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

target_link_libraries(dcm_covid PUBLIC Eigen3::Eigen dcEmb)
set_target_properties(dcm_covid PROPERTIES COMPILE_FLAGS "")

# 3Body model
add_executable(dcm_3body ${SOURCES_3BODY})

if(OpenMP_FOUND)
    target_link_libraries(dcm_3body PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

target_link_libraries(dcm_3body PUBLIC Eigen3::Eigen dcEmb)
set_target_properties(dcm_3body PROPERTIES COMPILE_FLAGS "")

# Weather model
add_executable(dcm_weather ${SOURCES_WEATHER})

if(OpenMP_FOUND)
    target_link_libraries(dcm_weather PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

target_link_libraries(dcm_weather PUBLIC Eigen3::Eigen dcEmb)
set_target_properties(dcm_weather PROPERTIES COMPILE_FLAGS "")
