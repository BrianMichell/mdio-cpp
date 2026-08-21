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
  return mdio_py::WaitFuture(mdio::Variable<>::Open(
      json, mdio_py::ToOpenMode(mdio_py::ParseOpenMode(mode))));
}

mdio::VariableData<> AllocateVariableData(const mdio::Variable<>& variable) {
  return mdio_py::CheckResult(mdio::from_variable(variable));
}

mdio::VariableData<> ReadVariableData(mdio::Variable<>& variable) {
  return mdio_py::WaitFuture(variable.Read());
}

py::array ReadVariableNumpy(mdio::Variable<>& variable) {
  mdio::VariableData<> data = ReadVariableData(variable);
  return mdio_py::VariableDataToNumpy(data, py::cast(data));
}

void WriteVariable(mdio::Variable<>& variable, const py::object& source) {
  if (py::isinstance<mdio::VariableData<>>(source)) {
    mdio_py::WaitWrite(variable.Write(source.cast<mdio::VariableData<>>()));
    return;
  }
  if (py::isinstance<py::array>(source)) {
    mdio::VariableData<> data = AllocateVariableData(variable);
    mdio_py::FillVariableDataFromNumpy(data, source.cast<py::array>());
    mdio_py::WaitWrite(variable.Write(data));
    return;
  }
  throw mdio_py::MdioError("write() expects a VariableData or NumPy array");
}

mdio::Variable<> SliceVariable(mdio::Variable<>& variable, const py::args& args,
                               const py::kwargs& kwargs) {
  const auto domain = mdio_py::DomainInfo(variable.dimensions());
  auto slices = mdio_py::ParseISelArgs(args, kwargs, &domain);
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
          [](py::object self) {
            auto& data = self.cast<mdio::VariableData<>&>();
            return mdio_py::VariableDataToNumpy(data, self);
          },
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
  mdio_py::BindDomainAttrs(
      variable, [](const mdio::Variable<>& obj) { return obj.dimensions(); });
  variable
      .def_static("open", &OpenVariable, py::arg("spec"),
                  py::arg("mode") = mdio_py::PyOpenMode::kOpen)
      .def_property_readonly("name", &mdio::Variable<>::get_variable_name)
      .def_property_readonly("long_name", &mdio::Variable<>::get_long_name)
      .def_property_readonly("rank", &mdio::Variable<>::rank)
      .def_property_readonly("num_samples", &mdio::Variable<>::num_samples)
      .def_property_readonly("dtype",
                             [](const mdio::Variable<>& obj) {
                               return mdio_py::DataTypeToNumpy(obj.dtype());
                             })
      .def_property_readonly("dtype_name",
                             [](const mdio::Variable<>& obj) {
                               return mdio_py::DataTypeName(obj.dtype());
                             })
      .def_property_readonly("metadata",
                             [](const mdio::Variable<>& obj) {
                               return mdio_py::JsonToPython(obj.getMetadata());
                             })
      .def_property_readonly(
          "attributes",
          [](const mdio::Variable<>& obj) {
            return mdio_py::JsonToPython(obj.GetAttributes());
          })
      .def_property_readonly(
          "spec",
          [](const mdio::Variable<>& obj) {
            return mdio_py::JsonToPython(mdio_py::CheckResult(obj.get_spec()));
          })
      .def_property_readonly(
          "chunk_shape",
          [](const mdio::Variable<>& obj) {
            return mdio_py::CheckResult(obj.get_chunk_shape());
          })
      .def_property_readonly(
          "store_shape",
          [](const mdio::Variable<>& obj) {
            return mdio_py::CheckResult(obj.get_store_shape());
          })
      .def_property_readonly(
          "units",
          [](const mdio::Variable<>& obj) {
            return mdio_py::JsonToPython(mdio_py::CheckResult(obj.get_units()));
          })
      .def_property_readonly("was_updated", &mdio::Variable<>::was_updated)
      .def("read", &ReadVariableNumpy, "Read the array into a NumPy ndarray.")
      .def("read_data", &ReadVariableData,
           "Read the array into a VariableData object.")
      .def("write", &WriteVariable, py::arg("source"),
           "Write a VariableData or NumPy array.")
      .def("slice", &SliceVariable)
      .def("isel", &SliceVariable)
      .def("update_attributes", &UpdateAttributes<mdio::Variable<>>,
           py::arg("attrs"), py::arg("histogram_dtype") = "float32")
      .def("intervals",
           [](const mdio::Variable<>& obj, const py::args& labels) {
             return mdio_py::CollectIntervals(obj, labels);
           })
      .def(
          "has_label",
          [](const mdio::Variable<>& obj, const std::string& label) {
            return obj.hasLabel(label);
          },
          py::arg("label"))
      .def("__repr__", [](const mdio::Variable<>& obj) {
        return mdio_py::StreamToString(obj);
      });

  mdio_py::BindCollection<mdio::VariableCollection>(m, "VariableCollection");

  auto header = py::class_<mdio::HeaderVariable<>>(m, "HeaderVariable");
  mdio_py::BindDomainAttrs(header, [](const mdio::HeaderVariable<>& obj) {
    return obj.dimensions();
  });
  header
      .def_property_readonly("name", &mdio::HeaderVariable<>::get_variable_name)
      .def_property_readonly("long_name",
                             &mdio::HeaderVariable<>::get_long_name)
      .def_property_readonly("rank", &mdio::HeaderVariable<>::rank)
      .def_property_readonly("dtype_name",
                             &mdio::HeaderVariable<>::get_dtype_name)
      .def_property_readonly("metadata",
                             [](const mdio::HeaderVariable<>& obj) {
                               return mdio_py::JsonToPython(obj.getMetadata());
                             })
      .def_property_readonly(
          "attributes",
          [](const mdio::HeaderVariable<>& obj) {
            return mdio_py::JsonToPython(obj.GetAttributes());
          })
      .def_property_readonly("was_updated",
                             &mdio::HeaderVariable<>::was_updated)
      .def("update_attributes", &UpdateAttributes<mdio::HeaderVariable<>>,
           py::arg("attrs"), py::arg("histogram_dtype") = "float32")
      .def("__repr__", [](const mdio::HeaderVariable<>& obj) {
        return mdio_py::StreamToString(obj);
      });

  mdio_py::BindCollection<mdio::HeaderVariableCollection>(
      m, "HeaderVariableCollection");
}
