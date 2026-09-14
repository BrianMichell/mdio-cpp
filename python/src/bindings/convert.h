// Copyright 2026 TGS

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PYTHON_SRC_BINDINGS_CONVERT_H_
#define PYTHON_SRC_BINDINGS_CONVERT_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "bindings/status.h"
#include "mdio/variable.h"
#include "mdio/zarr/zarr_driver.h"
#include "tensorstore/data_type.h"

// clang-format off
#include <nlohmann/json.hpp>  // NOLINT
// clang-format on

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace mdio_py {

template <typename T>
std::string StreamToString(const T& obj) {
  std::ostringstream stream;
  stream << obj;
  return stream.str();
}

inline nlohmann::json PythonToJson(const py::handle& obj) {
  if (obj.is_none()) {
    return nullptr;
  }
  if (py::isinstance<py::bool_>(obj)) {
    return obj.cast<bool>();
  }
  if (py::isinstance<py::int_>(obj)) {
    try {
      return obj.cast<std::int64_t>();
    } catch (const py::cast_error&) {
      try {
        return obj.cast<std::uint64_t>();
      } catch (const py::cast_error&) {
        throw MdioError("Python integer exceeds JSON 64-bit range");
      }
    }
  }
  if (py::isinstance<py::float_>(obj)) {
    return obj.cast<double>();
  }
  if (py::isinstance<py::str>(obj)) {
    return obj.cast<std::string>();
  }
  if (py::isinstance<py::bytes>(obj)) {
    return obj.cast<std::string>();
  }
  if (py::isinstance<py::dict>(obj)) {
    nlohmann::json json = nlohmann::json::object();
    for (auto item : obj.cast<py::dict>()) {
      json[item.first.cast<std::string>()] = PythonToJson(item.second);
    }
    return json;
  }
  if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    nlohmann::json json = nlohmann::json::array();
    for (auto item : obj) {
      json.push_back(PythonToJson(item));
    }
    return json;
  }
  if (py::isinstance<py::array>(obj)) {
    return PythonToJson(obj.attr("tolist")());
  }
  if (py::hasattr(obj, "item") && py::hasattr(obj, "shape")) {
    return PythonToJson(obj.attr("item")());
  }
  throw MdioError("Cannot convert Python type '" +
                  std::string(py::str(obj.get_type())) + "' to JSON");
}

inline py::object JsonToPython(const nlohmann::json& json) {
  if (json.is_null()) {
    return py::none();
  }
  if (json.is_boolean()) {
    return py::bool_(json.get<bool>());
  }
  if (json.is_number_unsigned()) {
    return py::int_(json.get<std::uint64_t>());
  }
  if (json.is_number_integer()) {
    return py::int_(json.get<std::int64_t>());
  }
  if (json.is_number_float()) {
    return py::float_(json.get<double>());
  }
  if (json.is_string()) {
    return py::str(json.get<std::string>());
  }
  if (json.is_array()) {
    py::list list;
    for (const auto& item : json) {
      list.append(JsonToPython(item));
    }
    return list;
  }
  if (json.is_object()) {
    py::dict dict;
    for (auto it = json.begin(); it != json.end(); ++it) {
      dict[py::str(it.key())] = JsonToPython(it.value());
    }
    return dict;
  }
  throw MdioError("Unsupported JSON value");
}

// id, C++ tag, NumPy name, Python attr. EXTRA cases are NumPy-only.
#define MDIO_PY_DTYPE_CASES(NUMERIC, EXTRA)          \
  NUMERIC(bool_t, bool_t, "bool", BOOL)              \
  NUMERIC(int8_t, int8_t, "int8", INT8)              \
  NUMERIC(int16_t, int16_t, "int16", INT16)          \
  NUMERIC(int32_t, int32_t, "int32", INT32)          \
  NUMERIC(int64_t, int64_t, "int64", INT64)          \
  NUMERIC(uint8_t, uint8_t, "uint8", UINT8)          \
  NUMERIC(uint16_t, uint16_t, "uint16", UINT16)      \
  NUMERIC(uint32_t, uint32_t, "uint32", UINT32)      \
  NUMERIC(uint64_t, uint64_t, "uint64", UINT64)      \
  NUMERIC(float16_t, float_16_t, "float16", FLOAT16) \
  NUMERIC(float32_t, float32_t, "float32", FLOAT32)  \
  NUMERIC(float64_t, float64_t, "float64", FLOAT64)  \
  EXTRA(complex64_t, "complex64", COMPLEX64)         \
  EXTRA(complex128_t, "complex128", COMPLEX128)

inline py::dtype DataTypeToNumpy(mdio::DataType dtype) {
  using tensorstore::DataTypeId;
  switch (dtype.id()) {
#define MDIO_PY_NP_NUM(id, tag, name, attr) \
  case DataTypeId::id:                      \
    return py::dtype(name);
#define MDIO_PY_NP_EXTRA(id, name, attr) \
  case DataTypeId::id:                   \
    return py::dtype(name);
    MDIO_PY_DTYPE_CASES(MDIO_PY_NP_NUM, MDIO_PY_NP_EXTRA)
#undef MDIO_PY_NP_NUM
#undef MDIO_PY_NP_EXTRA
    case DataTypeId::byte_t:
    default:
      return py::dtype("V" + std::to_string(dtype.size()));
  }
}

inline std::string DataTypeName(mdio::DataType dtype) {
  return std::string(dtype.name());
}

template <typename F>
decltype(auto) VisitNumericDtype(mdio::DataType dtype, F&& func) {
  using tensorstore::DataTypeId;
  switch (dtype.id()) {
#define MDIO_PY_VISIT(id, tag, name, attr) \
  case DataTypeId::id:                     \
    return func(mdio::dtypes::tag{});
#define MDIO_PY_VISIT_IGNORE(id, name, attr)
    MDIO_PY_DTYPE_CASES(MDIO_PY_VISIT, MDIO_PY_VISIT_IGNORE)
#undef MDIO_PY_VISIT
#undef MDIO_PY_VISIT_IGNORE
    default:
      throw MdioError("Unsupported dtype '" + std::string(dtype.name()) +
                      "' for this operation");
  }
}

inline void BindDtypeNames(py::module_& m) {
  py::module_ dtypes = m.def_submodule("dtypes", "MDIO dtype names");
#define MDIO_PY_ATTR_NUM(id, tag, name, pyattr) dtypes.attr(#pyattr) = name;
#define MDIO_PY_ATTR_EXTRA(id, name, pyattr) dtypes.attr(#pyattr) = name;
  MDIO_PY_DTYPE_CASES(MDIO_PY_ATTR_NUM, MDIO_PY_ATTR_EXTRA)
#undef MDIO_PY_ATTR_NUM
#undef MDIO_PY_ATTR_EXTRA
  dtypes.attr("BYTE") = "byte";
}

#undef MDIO_PY_DTYPE_CASES

template <typename T>
T CastNumeric(const py::handle& obj) {
  if constexpr (std::is_same_v<T, mdio::dtypes::float_16_t>) {
    return static_cast<T>(obj.cast<float>());
  } else {
    return obj.cast<T>();
  }
}

template <typename F>
decltype(auto) WithHistogramDtype(const std::string& name, F&& func) {
  if (name == "int32") {
    return func(mdio::dtypes::int32_t{});
  }
  if (name == "float32") {
    return func(mdio::dtypes::float32_t{});
  }
  throw MdioError("histogram_dtype must be 'float32' or 'int32', got '" + name +
                  "'");
}

struct IndexDomainInfo {
  mdio::DimensionIndex rank = 0;
  std::vector<mdio::Index> origin;
  std::vector<mdio::Index> shape;
  std::vector<std::string> labels;
};

template <typename Domain>
inline IndexDomainInfo DomainInfo(const Domain& domain) {
  IndexDomainInfo info;
  info.rank = domain.rank();
  info.origin.reserve(info.rank);
  info.shape.reserve(info.rank);
  info.labels.reserve(info.rank);
  for (mdio::DimensionIndex i = 0; i < info.rank; ++i) {
    info.origin.push_back(domain.origin()[i]);
    info.shape.push_back(domain.shape()[i]);
    info.labels.emplace_back(domain.labels()[i]);
  }
  return info;
}

template <typename Domain>
inline py::object DomainToPython(const Domain& domain) {
  const IndexDomainInfo info = DomainInfo(domain);
  py::dict dict;
  dict["rank"] = info.rank;
  dict["origin"] = info.origin;
  dict["shape"] = info.shape;
  dict["labels"] = info.labels;
  return dict;
}

inline py::array VariableDataToNumpy(const py::object& holder) {
  auto& data = holder.cast<mdio::VariableData<>&>();
  auto accessor = data.get_data_accessor();
  const mdio::DimensionIndex rank = accessor.rank();
  std::vector<ssize_t> shape(rank);
  std::vector<ssize_t> strides(rank);
  for (mdio::DimensionIndex i = 0; i < rank; ++i) {
    shape[i] = static_cast<ssize_t>(accessor.shape()[i]);
    strides[i] = static_cast<ssize_t>(accessor.byte_strides()[i]);
  }
  void* pointer = accessor.byte_strided_origin_pointer().get();
  return py::array(DataTypeToNumpy(accessor.dtype()), shape, strides, pointer,
                   holder);
}

inline void FillVariableDataFromNumpy(mdio::VariableData<>& data,
                                      const py::array& array) {
  const auto info = DomainInfo(data.dimensions());
  if (array.ndim() != static_cast<ssize_t>(info.rank)) {
    throw MdioError("NumPy rank " + std::to_string(array.ndim()) +
                    " does not match VariableData rank " +
                    std::to_string(info.rank));
  }
  for (mdio::DimensionIndex i = 0; i < info.rank; ++i) {
    if (array.shape(i) != static_cast<ssize_t>(info.shape[i])) {
      throw MdioError("NumPy shape does not match VariableData shape");
    }
  }
  py::array dest =
      VariableDataToNumpy(py::cast(data, py::return_value_policy::reference));
  dest[py::ellipsis()] =
      array.attr("astype")(dest.attr("dtype"), py::arg("copy") = false);
}

enum class PyOpenMode {
  kOpen = 0,
  kCreate = 1,
  kCreateClean = 2,
};

inline tensorstore::OpenMode ToOpenMode(PyOpenMode mode) {
  switch (mode) {
    case PyOpenMode::kOpen:
      return mdio::constants::kOpen;
    case PyOpenMode::kCreate:
      return mdio::constants::kCreate;
    case PyOpenMode::kCreateClean:
      return mdio::constants::kCreateClean;
  }
  throw MdioError("Unknown OpenMode");
}

inline PyOpenMode ParseOpenMode(const py::object& mode) {
  if (mode.is_none()) {
    throw MdioError("OpenMode cannot be None");
  }
  if (py::isinstance<py::str>(mode)) {
    const std::string value = mode.cast<std::string>();
    if (value == "open" || value == "r") {
      return PyOpenMode::kOpen;
    }
    if (value == "create" || value == "w-") {
      return PyOpenMode::kCreate;
    }
    if (value == "create_clean" || value == "w") {
      return PyOpenMode::kCreateClean;
    }
    throw MdioError("Unknown open mode string '" + value + "'");
  }
  return mode.cast<PyOpenMode>();
}

inline std::optional<mdio::zarr::ZarrVersion> ParseZarrVersion(
    const py::object& version) {
  if (version.is_none()) {
    return std::nullopt;
  }
  if (py::isinstance<py::int_>(version)) {
    const int value = version.cast<int>();
    if (value == 2) {
      return mdio::zarr::ZarrVersion::kV2;
    }
    if (value == 3) {
      return mdio::zarr::ZarrVersion::kV3;
    }
    throw MdioError("Zarr version must be 2 or 3");
  }
  return version.cast<mdio::zarr::ZarrVersion>();
}

}  // namespace mdio_py

#endif  // PYTHON_SRC_BINDINGS_CONVERT_H_
