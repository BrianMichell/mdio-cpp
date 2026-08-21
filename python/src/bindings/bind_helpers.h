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

#ifndef PYTHON_SRC_BINDINGS_BIND_HELPERS_H_
#define PYTHON_SRC_BINDINGS_BIND_HELPERS_H_

#include <pybind11/pybind11.h>

#include <string>
#include <vector>

#include "bindings/convert.h"

namespace py = pybind11;

namespace mdio_py {

template <typename Cls, typename GetDomain>
Cls& BindDomainAttrs(Cls& cls, GetDomain getter) {
  using Obj = typename Cls::type;
  cls.def_property_readonly("domain", [getter](const Obj& obj) {
    return DomainToPython(getter(obj));
  });
  cls.def_property_readonly("shape", [getter](const Obj& obj) {
    return DomainInfo(getter(obj)).shape;
  });
  cls.def_property_readonly("origin", [getter](const Obj& obj) {
    return DomainInfo(getter(obj)).origin;
  });
  cls.def_property_readonly("labels", [getter](const Obj& obj) {
    return DomainInfo(getter(obj)).labels;
  });
  return cls;
}

template <typename Collection>
void BindCollection(py::module_& m, const char* name) {
  py::class_<Collection>(m, name)
      .def(py::init<>())
      .def("add", &Collection::add, py::arg("label"), py::arg("variable"))
      .def(
          "get",
          [](const Collection& collection, const std::string& label) {
            return CheckResult(collection.get(label));
          },
          py::arg("label"))
      .def(
          "at",
          [](const Collection& collection, const std::string& label) {
            return CheckResult(collection.at(label));
          },
          py::arg("label"))
      .def("contains_key", &Collection::contains_key, py::arg("label"))
      .def("keys", &Collection::get_iterable_accessor)
      .def("__getitem__",
           [](const Collection& collection, const std::string& label) {
             return CheckResult(collection.at(label));
           })
      .def("__contains__", &Collection::contains_key)
      .def("__iter__",
           [](const py::object& self) {
             return self.attr("keys")().attr("__iter__")();
           })
      .def("__len__", [](const Collection& collection) {
        return collection.get_keys().size();
      });
}

inline py::list IntervalsToList(
    const std::vector<mdio::Variable<>::Interval>& intervals) {
  py::list list;
  for (const auto& interval : intervals) {
    list.append(interval);
  }
  return list;
}

template <typename Owner>
py::list CollectIntervals(const Owner& owner, const py::args& labels) {
  if (labels.size() == 0) {
    return IntervalsToList(CheckResult(owner.get_intervals()));
  }
  py::list list;
  for (const auto& label : labels) {
    auto intervals =
        CheckResult(owner.get_intervals(label.cast<std::string>()));
    for (const auto& interval : intervals) {
      list.append(interval);
    }
  }
  return list;
}

}  // namespace mdio_py

#endif  // PYTHON_SRC_BINDINGS_BIND_HELPERS_H_
