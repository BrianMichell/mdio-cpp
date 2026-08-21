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

#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "bindings/bind.h"
#include "bindings/slice.h"
#include "mdio/coordinate_selector.h"
#include "mdio/dataset_factory.h"
#include "mdio/dataset_validator.h"
#include "mdio/stats.h"
#include "mdio/utils/delete.h"

namespace {

struct PyCoordinateSelector {
  mdio::Dataset* dataset;
  mdio::CoordinateSelector selector;

  explicit PyCoordinateSelector(mdio::Dataset& dataset_ref)
      : dataset(&dataset_ref), selector(dataset_ref) {}
};

}  // namespace

void BindConstants(py::module_& m) {
  py::enum_<mdio_py::PyOpenMode>(m, "OpenMode")
      .value("OPEN", mdio_py::PyOpenMode::kOpen)
      .value("CREATE", mdio_py::PyOpenMode::kCreate)
      .value("CREATE_CLEAN", mdio_py::PyOpenMode::kCreateClean);

  py::enum_<mdio::zarr::ZarrVersion>(m, "ZarrVersion")
      .value("V2", mdio::zarr::ZarrVersion::kV2)
      .value("V3", mdio::zarr::ZarrVersion::kV3);

  py::module_ units = m.def_submodule("units", "MDIO unit strings");
  units.attr("DEGREES") = std::string(mdio::units::kDegrees);
  units.attr("RADIANS") = std::string(mdio::units::kRadians);
  units.attr("GRAMS_PER_CUBIC_CENTIMETER") =
      std::string(mdio::units::kGramsPerCubicCentimeter);
  units.attr("KILOGRAMS_PER_CUBIC_METER") =
      std::string(mdio::units::kKilogramsPerCubicMeter);
  units.attr("POUNDS_PER_GALLON") = std::string(mdio::units::kPoundsPerGallon);
  units.attr("HERTZ") = std::string(mdio::units::kHertz);
  units.attr("MILLIMETERS") = std::string(mdio::units::kMillimeters);
  units.attr("CENTIMETERS") = std::string(mdio::units::kCentimeters);
  units.attr("METERS") = std::string(mdio::units::kMeters);
  units.attr("KILOMETERS") = std::string(mdio::units::kKilometers);
  units.attr("INCHES") = std::string(mdio::units::kInches);
  units.attr("FEET") = std::string(mdio::units::kFeet);
  units.attr("YARDS") = std::string(mdio::units::kYards);
  units.attr("MILES") = std::string(mdio::units::kMiles);
  units.attr("METERS_PER_SECOND") = std::string(mdio::units::kMetersPerSecond);
  units.attr("FEET_PER_SECOND") = std::string(mdio::units::kFeetPerSecond);
  units.attr("NANOSECONDS") = std::string(mdio::units::kNanoseconds);
  units.attr("MICROSECONDS") = std::string(mdio::units::kMicroseconds);
  units.attr("MILLISECONDS") = std::string(mdio::units::kMilliseconds);
  units.attr("SECONDS") = std::string(mdio::units::kSeconds);
  units.attr("MINUTES") = std::string(mdio::units::kMinutes);
  units.attr("HOURS") = std::string(mdio::units::kHours);
  units.attr("DAYS") = std::string(mdio::units::kDays);
  units.attr("MICROVOLTS") = std::string(mdio::units::kMicrovolts);
  units.attr("MILLIVOLTS") = std::string(mdio::units::kMillivolts);
  units.attr("VOLTS") = std::string(mdio::units::kVolts);

  mdio_py::BindDtypeNames(m);
}

void BindDescriptors(py::module_& m) {
  py::class_<mdio_py::PyRange>(m, "Range")
      .def(py::init<std::string, mdio::Index, mdio::Index, mdio::Index>(),
           py::arg("label"), py::arg("start"), py::arg("stop"),
           py::arg("step") = 1)
      .def_readwrite("label", &mdio_py::PyRange::label)
      .def_readwrite("start", &mdio_py::PyRange::start)
      .def_readwrite("stop", &mdio_py::PyRange::stop)
      .def_readwrite("step", &mdio_py::PyRange::step)
      .def("__repr__", [](const mdio_py::PyRange& desc) {
        return "Range(label=" + desc.label +
               ", start=" + std::to_string(desc.start) +
               ", stop=" + std::to_string(desc.stop) +
               ", step=" + std::to_string(desc.step) + ")";
      });

  py::class_<mdio_py::PyValue>(m, "Value")
      .def(py::init<std::string, py::object>(), py::arg("label"),
           py::arg("value"))
      .def_readwrite("label", &mdio_py::PyValue::label)
      .def_readwrite("value", &mdio_py::PyValue::value);
}

void BindUtils(py::module_& m) {
  m.def(
      "validate_dataset",
      [](const py::object& schema) {
        nlohmann::json json = mdio_py::PythonToJson(schema);
        mdio_py::ThrowIfError(validate_dataset(json));
      },
      py::arg("schema"), "Validate an MDIO v1 Dataset schema.");

  m.def(
      "construct",
      [](const py::object& schema, const std::string& path,
         const py::object& zarr_version) {
        nlohmann::json json = mdio_py::PythonToJson(schema);
        auto version = mdio_py::ParseZarrVersion(zarr_version);
        auto result = Construct(json, path, version);
        mdio_py::ThrowIfError(result.status());
        auto [metadata, specs] = result.value();
        py::list spec_list;
        for (const auto& spec : specs) {
          spec_list.append(mdio_py::JsonToPython(spec));
        }
        return py::make_tuple(mdio_py::JsonToPython(metadata), spec_list);
      },
      py::arg("schema"), py::arg("path"), py::arg("zarr_version") = py::none(),
      "Build Variable specs from an MDIO Dataset schema.");

  m.def(
      "delete_dataset",
      [](const std::string& path) {
        mdio_py::CheckResult(mdio::utils::DeleteDataset(path));
      },
      py::arg("path"), "Delete a valid MDIO dataset.");

  m.def(
      "trim_dataset",
      [](const std::string& path, const py::object& slices,
         bool delete_sliced_out_chunks) {
        mdio_py::TrimWithVector(path, delete_sliced_out_chunks,
                                mdio_py::ParseTrimSlices(slices));
      },
      py::arg("path"), py::arg("slices"),
      py::arg("delete_sliced_out_chunks") = false,
      "DANGER: mutate on-disk shape. `slices` is {label: stop}, Range, or "
      "slice.");

  py::class_<mdio::UserAttributes>(m, "UserAttributes")
      .def_static(
          "from_json",
          [](const py::object& attrs, const std::string& histogram_dtype) {
            nlohmann::json json = mdio_py::PythonToJson(attrs);
            return mdio_py::WithHistogramDtype(histogram_dtype, [&](auto tag) {
              using T = decltype(tag);
              return mdio_py::CheckResult(
                  mdio::UserAttributes::FromJson<T>(json));
            });
          },
          py::arg("attrs"), py::arg("histogram_dtype") = "float32")
      .def("to_json",
           [](const mdio::UserAttributes& attrs) {
             return mdio_py::JsonToPython(attrs.ToJson());
           })
      .def_property_readonly("stats",
                             [](const mdio::UserAttributes& attrs) {
                               return mdio_py::JsonToPython(attrs.getStatsV1());
                             })
      .def_property_readonly("units",
                             [](const mdio::UserAttributes& attrs) {
                               return mdio_py::JsonToPython(attrs.getUnitsV1());
                             })
      .def_property_readonly("attrs", [](const mdio::UserAttributes& attrs) {
        return mdio_py::JsonToPython(attrs.getAttrs());
      });

  py::class_<PyCoordinateSelector>(m, "CoordinateSelector")
      .def(py::init<mdio::Dataset&>(), py::arg("dataset"),
           py::keep_alive<1, 2>())
      .def("reset",
           [](PyCoordinateSelector& selector) { selector.selector.reset(); })
      .def(
          "filter_by_coordinate",
          [](PyCoordinateSelector& selector, const std::string& label,
             const py::object& value) {
            auto variable =
                mdio_py::CheckResult(selector.dataset->variables.at(label));
            mdio_py::VisitNumericDtype(variable.dtype(), [&](auto tag) {
              using T = decltype(tag);
              mdio::ValueDescriptor<T> desc{label,
                                            mdio_py::CastNumeric<T>(value)};
              mdio_py::WaitFuture(selector.selector.filterByCoordinate(desc));
            });
          },
          py::arg("label"), py::arg("value"))
      .def(
          "sort_by_key",
          [](PyCoordinateSelector& selector, const std::string& sort_key) {
            auto variable =
                mdio_py::CheckResult(selector.dataset->variables.at(sort_key));
            mdio_py::VisitNumericDtype(variable.dtype(), [&](auto tag) {
              using T = decltype(tag);
              mdio_py::WaitFuture(
                  selector.selector.sortSelectionByKey<T>(sort_key));
            });
          },
          py::arg("sort_key"))
      .def(
          "read_selection",
          [](PyCoordinateSelector& selector,
             const std::string& output_variable) -> py::array {
            auto variable = mdio_py::CheckResult(
                selector.dataset->variables.at(output_variable));
            return mdio_py::VisitNumericDtype(
                variable.dtype(), [&](auto tag) -> py::array {
                  using T = decltype(tag);
                  if constexpr (std::is_same_v<T, bool>) {
                    throw mdio_py::MdioError(
                        "read_selection does not support bool arrays");
                  } else {
                    std::vector<T> values = mdio_py::WaitFuture(
                        selector.selector.readSelection<T>(output_variable));
                    py::array_t<T> array(values.size());
                    if (!values.empty()) {
                      std::memcpy(array.mutable_data(), values.data(),
                                  values.size() * sizeof(T));
                    }
                    return py::array(array);
                  }
                });
          },
          py::arg("output_variable"));
}
