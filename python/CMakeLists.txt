# Copyright 2026 TGS

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.24)

if(NOT Python3_EXECUTABLE)
  find_program(Python3_EXECUTABLE NAMES python3 REQUIRED)
endif()

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
  OUTPUT_VARIABLE MDIO_PYTHON_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_VARIABLE MDIO_PYTHON_EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE MDIO_NUMPY_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE MDIO_NUMPY_RESULT
)
if(NOT MDIO_NUMPY_RESULT EQUAL 0)
  message(FATAL_ERROR "NumPy is required to build MDIO Python bindings")
endif()

message(STATUS "MDIO Python executable: ${Python3_EXECUTABLE}")
message(STATUS "MDIO Python include: ${MDIO_PYTHON_INCLUDE}")
message(STATUS "MDIO NumPy include: ${MDIO_NUMPY_INCLUDE}")

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

set(MDIO_PYTHON_SOURCES
  src/bindings/module.cc
  src/bindings/bind_variable.cc
  src/bindings/bind_dataset.cc
  src/bindings/bind_support.cc
)

add_library(_core MODULE ${MDIO_PYTHON_SOURCES})

target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${MDIO_PYTHON_INCLUDE}
  ${MDIO_NUMPY_INCLUDE}
)

target_compile_features(_core PRIVATE cxx_std_17)
target_compile_options(_core PRIVATE ${mdio_DEFAULT_COPTS})
target_compile_definitions(_core PRIVATE MAX_NUM_SLICES=${MAX_NUM_SLICES})

target_link_libraries(_core PRIVATE
  pybind11::headers
  mdio
  ${mdio_TENSORSTORE_DEPS}
  ${mdio_INTERNAL_GCS_DRIVER_DEPS}
  ${mdio_INTERNAL_S3_DRIVER_DEPS}
  nlohmann_json_schema_validator
)

set_target_properties(_core PROPERTIES
  PREFIX ""
  SUFFIX "${MDIO_PYTHON_EXT_SUFFIX}"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/mdio_cpp
  LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/src/mdio_cpp
  LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/src/mdio_cpp
)

install(TARGETS _core DESTINATION mdio_cpp)
install(FILES src/mdio_cpp/__init__.py DESTINATION mdio_cpp)

if(BUILD_TESTING)
  add_test(
    NAME mdio_python_bindings
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -v
  )
  set_tests_properties(mdio_python_bindings PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
endif()
