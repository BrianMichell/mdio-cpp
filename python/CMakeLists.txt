cmake_minimum_required(VERSION 3.24)

project(mdio_python_bindings LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
  FIND_PACKAGE_ARGS NAMES pybind11 CONFIG
)

FetchContent_MakeAvailable(pybind11)

find_package(Python3 COMPONENTS Development REQUIRED)

pybind11_add_module(
  mdio_cpp
  mdio_pybind.cc
)

target_link_libraries(
  mdio_cpp
  PRIVATE
    mdio
    tensorstore::tensorstore
    tensorstore::driver_zarr3
    tensorstore::driver_zarr3_metadata
    tensorstore::driver_zarr_dtype
    tensorstore_index_space_dimension_selection
    tensorstore_index_space_interval_slice_op
    tensorstore_index_space_label_op
    ${mdio_INTERNAL_DEPS}
    ${mdio_INTERNAL_GCS_DRIVER_DEPS}
    ${mdio_INTERNAL_S3_DRIVER_DEPS}
    nlohmann_json_schema_validator::nlohmann_json_schema_validator
)

target_compile_options(mdio_cpp PRIVATE ${mdio_DEFAULT_COPTS})

target_compile_definitions(mdio_cpp PRIVATE MAX_NUM_SLICES=${MAX_NUM_SLICES})

target_include_directories(mdio_cpp PRIVATE ${CMAKE_SOURCE_DIR})

set_target_properties(
  mdio_cpp
  PROPERTIES
    OUTPUT_NAME "mdio_cpp"
)

