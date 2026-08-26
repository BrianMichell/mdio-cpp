# Copyright 2026 TGS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Stand-alone WEBKNOSSOS sidecars. Same consume path as USER_GUIDE
# "Consuming the installed package": find_package(mdio) + mdio::monolith.
# Older sibling: examples/seismic_reader (installer + manual -l flags).
cmake_minimum_required(VERSION 3.27)
project(mdio_tools VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(mdio REQUIRED)
if(NOT TARGET mdio::monolith)
  message(FATAL_ERROR
    "mdio::monolith not found. Install mdio-cpp with "
    "-DMDIO_BUILD_MONOLITHIC_SHARED=ON, then pass "
    "-DCMAKE_PREFIX_PATH=<mdio-install> to this project.")
endif()

# mdio_DIR is <prefix>/lib/cmake/mdio. Binaries need libmdio_monolith.so at runtime.
get_filename_component(_mdio_lib_dir "${mdio_DIR}/../.." ABSOLUTE)
set(CMAKE_BUILD_RPATH "${_mdio_lib_dir}")
set(CMAKE_INSTALL_RPATH "${_mdio_lib_dir}")

set(_mdio_tools_src "${CMAKE_CURRENT_SOURCE_DIR}/../../tools")

function(add_mdio_tool name)
  add_executable(${name} "${_mdio_tools_src}/${name}.cc")
  target_link_libraries(${name} PRIVATE mdio::monolith)
endfunction()

add_mdio_tool(mdio_load)
add_mdio_tool(mdio_optimize)
add_mdio_tool(mdio_labels)

install(TARGETS mdio_load mdio_optimize mdio_labels RUNTIME DESTINATION bin)
