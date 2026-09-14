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

#include <string>

#include "bindings/bind.h"
#include "bindings/bind_helpers.h"
#include "bindings/slice.h"

namespace {

mdio::Variable<> OpenVariable(const py::object& spec, const py::object& mode) {
  nlohmann::json json = mdio_py::PythonToJson(spec);
  const tensorstore::OpenMode open_mode =
      mdio_py::ToOpenMode(mdio_py::ParseOpenMode(mode));
  return mdio_py::Await(
      [&] { return mdio::Variable<>::Open(json, open_mode); });
}

mdio::VariableData<> AllocateVariableData(const mdio::Variable<>& variable) {
  return mdio_py::CheckResult(mdio::from_variable(variable));
}

mdio::VariableData<> ReadVariableData(mdio::Variable<>& variable) {
  return mdio_py::Await([&] { return variable.Read(); });
}

py::array ReadVariableNumpy(mdio::Variable<>& variable) {
  return mdio_py::VariableDataToNumpy(py::cast(ReadVariableData(variable)));
}

void WriteVariableData(mdio::Variable<>& variable,
                       const mdio::VariableData<>& data) {
  mdio_py::Await([&] { return variable.Write(data); });
}

void WriteVariableNumpy(mdio::Variable<>& variable, const py::array& array) {
  mdio::VariableData<> data = AllocateVariableData(variable);
  mdio_py::FillVariableDataFromNumpy(data, array);
  mdio_py::Await([&] { return variable.Write(data); });
}

mdio::Variable<> SliceVariable(mdio::Variable<>& variable, const py::args& args,
                               const py::kwargs& kwargs) {
  const auto domain = mdio_py::DomainInfo(variable.dimensions());
  auto slices = mdio_py::ParseISelArgs(args, kwargs, domain);
  if (slices.empty()) {
    return variable;
  }
  return mdio_py::CheckResult(variable.slice(mdio_py::ToDescriptors(slices)));
}

template <typename Variable>
void UpdateAttributes(Variable& variable, const py::object& attrs,
                      const std::string& histogram_dtype) {
  nlohmann::json json = mdio_py::PythonToJson(attrs);
  mdio_py::WithHistogramDtype(histogram_dtype, [&](auto tag) {
    using T = decltype(tag);
    mdio_py::CheckResult(variable.template UpdateAttributes<T>(json));
  });
}

}  // namespace

namespace mdio_py {

void BindVariable(py::module_& m) {
  py::class_<mdio::Variable<>::Interval>(m, "Interval")
      .def_property_readonly("label",
                             [](const mdio::Variable<>::Interval& interval) {
                               return std::string(interval.label.label());
                             })
      .def_readonly("inclusive_min", &mdio::Variable<>::Interval::inclusive_min)
      .def_readonly("exclusive_max", &mdio::Variable<>::Interval::exclusive_max)
      .def("__repr__", [](const mdio::Variable<>::Interval& interval) {
        return mdio_py::StreamToString(interval);
      });

  auto variable_data = py::class_<mdio::VariableData<>>(m, "VariableData");
  mdio_py::BindDomainAttrs(variable_data, [](const mdio::VariableData<>& data) {
    return data.dimensions();
  });
  variable_data
      .def_property_readonly(
          "name",
          [](const mdio::VariableData<>& data) { return data.variableName; })
      .def_property_readonly(
          "long_name",
          [](const mdio::VariableData<>& data) { return data.longName; })
      .def_property_readonly("metadata",
                             [](const mdio::VariableData<>& data) {
                               return mdio_py::JsonToPython(data.metadata);
                             })
      .def_property_readonly("rank", &mdio::VariableData<>::rank)
      .def_property_readonly("num_samples", &mdio::VariableData<>::num_samples)
      .def_property_readonly("dtype",
                             [](mdio::VariableData<>& data) {
                               return mdio_py::DataTypeToNumpy(data.dtype());
                             })
      .def_property_readonly("dtype_name",
                             [](mdio::VariableData<>& data) {
                               return mdio_py::DataTypeName(data.dtype());
                             })
      .def_property_readonly("flattened_offset",
                             [](mdio::VariableData<>& data) {
                               return data.get_flattened_offset();
                             })
      .def_property(
          "numpy",
          [](py::object self) { return mdio_py::VariableDataToNumpy(self); },
          [](mdio::VariableData<>& data, const py::array& array) {
            mdio_py::FillVariableDataFromNumpy(data, array);
          },
          "Zero-copy NumPy view of the in-memory array.")
      .def_static("allocate", &AllocateVariableData, py::arg("variable"))
      .def("__repr__", [](const mdio::VariableData<>& data) {
        return mdio_py::StreamToString(data);
      });

  m.def("from_variable", &AllocateVariableData, py::arg("variable"),
        "Allocate an in-memory VariableData filled with defaults/NaN.");

  auto variable = py::class_<mdio::Variable<>>(m, "Variable");
  BindDomainAttrs(variable,
                  [](const mdio::Variable<>& obj) { return obj.dimensions(); });
  BindSharedMetadata(variable, &UpdateAttributes<mdio::Variable<>>);
  variable
      .def_static("open", &OpenVariable, py::arg("spec"),
                  py::arg("mode") = PyOpenMode::kOpen)
      .def_property_readonly("num_samples", &mdio::Variable<>::num_samples)
      .def_property_readonly("dtype",
                             [](const mdio::Variable<>& obj) {
                               return DataTypeToNumpy(obj.dtype());
                             })
      .def_property_readonly(
          "dtype_name",
          [](const mdio::Variable<>& obj) { return DataTypeName(obj.dtype()); })
      .def_property_readonly("spec",
                             [](const mdio::Variable<>& obj) {
                               return JsonToPython(CheckResult(obj.get_spec()));
                             })
      .def_property_readonly("chunk_shape",
                             [](const mdio::Variable<>& obj) {
                               return CheckResult(obj.get_chunk_shape());
                             })
      .def_property_readonly("store_shape",
                             [](const mdio::Variable<>& obj) {
                               return CheckResult(obj.get_store_shape());
                             })
      .def_property_readonly(
          "units",
          [](const mdio::Variable<>& obj) {
            return JsonToPython(CheckResult(obj.get_units()));
          })
      .def("read", &ReadVariableNumpy, "Read the array into a NumPy ndarray.")
      .def("read_data", &ReadVariableData,
           "Read the array into a VariableData object.")
      .def("write", &WriteVariableData, py::arg("source"),
           "Write a VariableData object.")
      .def("write", &WriteVariableNumpy, py::arg("source"),
           "Write a NumPy array.")
      .def("slice", &SliceVariable)
      .def("isel", &SliceVariable)
      .def("intervals",
           [](const mdio::Variable<>& obj, const py::args& labels) {
             return CollectIntervals(obj, labels);
           })
      .def(
          "has_label",
          [](const mdio::Variable<>& obj, const std::string& label) {
            return obj.hasLabel(label);
          },
          py::arg("label"));

  BindCollection<mdio::VariableCollection>(m, "VariableCollection");

  auto header = py::class_<mdio::HeaderVariable<>>(m, "HeaderVariable");
  BindDomainAttrs(header, [](const mdio::HeaderVariable<>& obj) {
    return obj.dimensions();
  });
  BindSharedMetadata(header, &UpdateAttributes<mdio::HeaderVariable<>>);
  header.def_property_readonly("dtype_name",
                               &mdio::HeaderVariable<>::get_dtype_name);

  BindCollection<mdio::HeaderVariableCollection>(m, "HeaderVariableCollection");
}

}  // namespace mdio_py
