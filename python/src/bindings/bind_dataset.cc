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

mdio::Dataset DatasetFromJson(const py::object& schema, const std::string& path,
                              const py::object& mode,
                              const py::object& zarr_version) {
  nlohmann::json json = mdio_py::PythonToJson(schema);
  const tensorstore::OpenMode open_mode =
      mdio_py::ToOpenMode(mdio_py::ParseOpenMode(mode));
  auto version = mdio_py::ParseZarrVersion(zarr_version);
  if (version.has_value()) {
    return mdio_py::WaitFuture(
        mdio::Dataset::from_json(json, path, version, open_mode));
  }
  return mdio_py::WaitFuture(mdio::Dataset::from_json(json, path, open_mode));
}

mdio::Dataset DatasetOpen(const std::string& path, const py::object& mode) {
  return mdio_py::WaitFuture(mdio::Dataset::Open(
      path, mdio_py::ToOpenMode(mdio_py::ParseOpenMode(mode))));
}

mdio::Dataset DatasetISel(mdio::Dataset& dataset, const py::args& args,
                          const py::kwargs& kwargs) {
  const auto domain = mdio_py::DomainInfo(dataset.domain);
  return mdio_py::ApplyISel(dataset,
                            mdio_py::ParseISelArgs(args, kwargs, &domain));
}

mdio::Dataset DatasetSel(mdio::Dataset dataset, const py::args& args,
                         const py::kwargs& kwargs) {
  return mdio_py::ApplySel(std::move(dataset),
                           mdio_py::ParseSelArgs(args, kwargs));
}

py::dict CoordinatesToDict(const mdio::coordinate_map& coordinates) {
  py::dict dict;
  for (const auto& [key, values] : coordinates) {
    dict[py::str(key)] = values;
  }
  return dict;
}

}  // namespace

void BindDataset(py::module_& m) {
  py::class_<mdio::Dataset>(m, "Dataset")
      .def_static("from_json", &DatasetFromJson, py::arg("schema"),
                  py::arg("path"),
                  py::arg("mode") = mdio_py::PyOpenMode::kCreate,
                  py::arg("zarr_version") = py::none(),
                  "Create or open a Dataset from an MDIO v1 schema.")
      .def_static("open", &DatasetOpen, py::arg("path"),
                  py::arg("mode") = mdio_py::PyOpenMode::kOpen,
                  "Open an existing Dataset.")
      .def_property_readonly(
          "variables",
          [](mdio::Dataset& dataset) -> mdio::VariableCollection& {
            return dataset.variables;
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "header_variables",
          [](mdio::Dataset& dataset) -> mdio::HeaderVariableCollection& {
            return dataset.header_variables;
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly("coordinates",
                             [](const mdio::Dataset& dataset) {
                               return CoordinatesToDict(dataset.coordinates);
                             })
      .def_property_readonly("domain",
                             [](const mdio::Dataset& dataset) {
                               return mdio_py::DomainToPython(dataset.domain);
                             })
      .def_property_readonly(
          "metadata",
          [](const mdio::Dataset& dataset) {
            return mdio_py::JsonToPython(dataset.getMetadata());
          })
      .def(
          "get_variable",
          [](mdio::Dataset& dataset, const std::string& name) {
            return mdio_py::CheckResult(dataset.get_variable(name));
          },
          py::arg("name"))
      .def(
          "get_header_variable",
          [](mdio::Dataset& dataset, const std::string& name) {
            return mdio_py::CheckResult(dataset.get_header_variable(name));
          },
          py::arg("name"))
      .def("isel", &DatasetISel,
           "Index-based slice. kwargs are dimension slices. An integer index "
           "keeps a size-1 dimension.")
      .def("sel", &DatasetSel,
           "Label/coordinate-value slice. kwargs are dimension values. "
           "Lists are not implemented.")
      .def(
          "select_field",
          [](mdio::Dataset& dataset, const std::string& variable_name,
             const std::string& field_name) {
            return mdio_py::WaitFuture(
                dataset.SelectField(variable_name, field_name));
          },
          py::arg("variable_name"), py::arg("field_name"))
      .def("commit_metadata",
           [](mdio::Dataset& dataset) {
             mdio_py::WaitFuture(dataset.CommitMetadata());
           })
      .def("intervals",
           [](const mdio::Dataset& dataset, const py::args& labels) {
             return mdio_py::CollectIntervals(dataset, labels);
           })
      .def("__getitem__",
           [](mdio::Dataset& dataset, const std::string& label) {
             return mdio_py::CheckResult(dataset[label]);
           })
      .def("__repr__", [](const mdio::Dataset& dataset) {
        return mdio_py::StreamToString(dataset);
      });
}
