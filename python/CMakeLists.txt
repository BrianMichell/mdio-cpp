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

set(PYBIND11_FINDPYTHON ON)
# PATH python3. scikit-build can pass -DPython_EXECUTABLE.
if(NOT Python_EXECUTABLE)
  execute_process(
    COMMAND python3 -c "import sys; print(sys.executable)"
    OUTPUT_VARIABLE Python_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

find_package(pybind11 2.13 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

message(STATUS "MDIO Python executable: ${Python_EXECUTABLE}")

set(MDIO_PYTHON_SOURCES
  src/bindings/module.cc
  src/bindings/bind_variable.cc
  src/bindings/bind_dataset.cc
  src/bindings/bind_support.cc
)

pybind11_add_module(_core ${MDIO_PYTHON_SOURCES})

target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(_core PRIVATE cxx_std_17)
target_compile_options(_core PRIVATE ${mdio_DEFAULT_COPTS})

# mdio is INTERFACE headers. Drivers must be linked so the MODULE can open
# local / GCS / S3 stores (same set C++ tests use, minus status_testutil).
target_link_libraries(_core PRIVATE
  mdio
  ${mdio_TENSORSTORE_DEPS}
  ${mdio_INTERNAL_GCS_DRIVER_DEPS}
  ${mdio_INTERNAL_S3_DRIVER_DEPS}
  nlohmann_json_schema_validator
)

# In-tree output keeps PYTHONPATH=python/src working for local builds.
set(_mdio_py_out ${CMAKE_CURRENT_SOURCE_DIR}/src/mdio_cpp)
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${_mdio_py_out})
if(CMAKE_CONFIGURATION_TYPES)
  foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${_cfg}" _cfg_u)
    set_property(TARGET _core PROPERTY LIBRARY_OUTPUT_DIRECTORY_${_cfg_u}
                 ${_mdio_py_out})
  endforeach()
endif()

install(TARGETS _core DESTINATION mdio_cpp)
install(FILES src/mdio_cpp/__init__.py DESTINATION mdio_cpp)

if(BUILD_TESTING)
  add_test(
    NAME mdio_python_bindings
    COMMAND ${Python_EXECUTABLE} -m unittest discover
            -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -v
  )
  set_tests_properties(mdio_python_bindings PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
endif()
