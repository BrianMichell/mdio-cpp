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

#ifndef PYTHON_SRC_BINDINGS_BIND_H_
#define PYTHON_SRC_BINDINGS_BIND_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindConstants(py::module_& m);
void BindDescriptors(py::module_& m);
void BindVariable(py::module_& m);
void BindDataset(py::module_& m);
void BindUtils(py::module_& m);

#endif  // PYTHON_SRC_BINDINGS_BIND_H_
