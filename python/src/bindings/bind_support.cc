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
#include <string_view>
#include <type_traits>
#include <utility>
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
  mdio::Dataset& dataset;
  mdio::CoordinateSelector selector;

  explicit PyCoordinateSelector(mdio::Dataset& dataset_ref)
      : dataset(dataset_ref), selector(dataset_ref) {}
};

template <typename F>
decltype(auto) WithCoordinateDtype(PyCoordinateSelector& selector,
                                   const std::string& name, F&& func) {
  auto variable = mdio_py::CheckResult(selector.dataset.variables.at(name));
  return mdio_py::VisitNumericDtype(variable.dtype(), std::forward<F>(func));
}

struct UnitAttr {
  const char* name;
  std::string_view value;
};

constexpr UnitAttr kUnitAttrs[] = {
    {"DEGREES", mdio::units::kDegrees},
    {"RADIANS", mdio::units::kRadians},
    {"GRAMS_PER_CUBIC_CENTIMETER", mdio::units::kGramsPerCubicCentimeter},
    {"KILOGRAMS_PER_CUBIC_METER", mdio::units::kKilogramsPerCubicMeter},
    {"POUNDS_PER_GALLON", mdio::units::kPoundsPerGallon},
    {"HERTZ", mdio::units::kHertz},
    {"MILLIMETERS", mdio::units::kMillimeters},
    {"CENTIMETERS", mdio::units::kCentimeters},
    {"METERS", mdio::units::kMeters},
    {"KILOMETERS", mdio::units::kKilometers},
    {"INCHES", mdio::units::kInches},
    {"FEET", mdio::units::kFeet},
    {"YARDS", mdio::units::kYards},
    {"MILES", mdio::units::kMiles},
    {"METERS_PER_SECOND", mdio::units::kMetersPerSecond},
    {"FEET_PER_SECOND", mdio::units::kFeetPerSecond},
    {"NANOSECONDS", mdio::units::kNanoseconds},
    {"MICROSECONDS", mdio::units::kMicroseconds},
    {"MILLISECONDS", mdio::units::kMilliseconds},
    {"SECONDS", mdio::units::kSeconds},
    {"MINUTES", mdio::units::kMinutes},
    {"HOURS", mdio::units::kHours},
    {"DAYS", mdio::units::kDays},
    {"MICROVOLTS", mdio::units::kMicrovolts},
    {"MILLIVOLTS", mdio::units::kMillivolts},
    {"VOLTS", mdio::units::kVolts},
};

}  // namespace

namespace mdio_py {

void BindConstants(py::module_& m) {
  py::enum_<PyOpenMode>(m, "OpenMode")
      .value("OPEN", PyOpenMode::kOpen)
      .value("CREATE", PyOpenMode::kCreate)
      .value("CREATE_CLEAN", PyOpenMode::kCreateClean);

  py::enum_<mdio::zarr::ZarrVersion>(m, "ZarrVersion")
      .value("V2", mdio::zarr::ZarrVersion::kV2)
      .value("V3", mdio::zarr::ZarrVersion::kV3);

  py::module_ units = m.def_submodule("units", "MDIO unit strings");
  for (const auto& unit : kUnitAttrs) {
    units.attr(unit.name) = std::string(unit.value);
  }

  BindDtypeNames(m);
}

void BindDescriptors(py::module_& m) {
  py::class_<mdio_py::PyRange>(m, "Range")
      .def(py::init<std::string, mdio::Index, mdio::Index, mdio::Index>(),
           py::arg("label"), py::arg("start"), py::arg("stop"),
           py::arg("step") = 1)
      .def_readwrite("label", &mdio_py::PyRange::label)
      .def_readwrite("start", &mdio_py::PyRange::start)
      .def_readwrite("stop", &mdio_py::PyRange::stop)
      .def_readonly("step", &mdio_py::PyRange::step)
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
        mdio_py::Await([&] { return mdio::utils::DeleteDataset(path); });
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
      "DANGER: mutate on-disk shape. `slices` is {label: stop}.");

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
            WithCoordinateDtype(selector, label, [&](auto tag) {
              using T = decltype(tag);
              mdio::ValueDescriptor<T> desc{label, CastNumeric<T>(value)};
              Await([&] { return selector.selector.filterByCoordinate(desc); });
            });
          },
          py::arg("label"), py::arg("value"))
      .def(
          "sort_by_key",
          [](PyCoordinateSelector& selector, const std::string& sort_key) {
            WithCoordinateDtype(selector, sort_key, [&](auto tag) {
              using T = decltype(tag);
              Await([&] {
                return selector.selector.sortSelectionByKey<T>(sort_key);
              });
            });
          },
          py::arg("sort_key"))
      .def(
          "read_selection",
          [](PyCoordinateSelector& selector,
             const std::string& output_variable) -> py::array {
            return WithCoordinateDtype(
                selector, output_variable, [&](auto tag) -> py::array {
                  using T = decltype(tag);
                  if constexpr (std::is_same_v<T, bool>) {
                    throw MdioError(
                        "read_selection does not support bool arrays");
                  } else {
                    std::vector<T> values = Await([&] {
                      return selector.selector.readSelection<T>(
                          output_variable);
                    });
                    return py::array_t<T>(
                        static_cast<py::ssize_t>(values.size()), values.data());
                  }
                });
          },
          py::arg("output_variable"));
}

}  // namespace mdio_py
