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

#include "bindings/bind.h"
#include "bindings/status.h"
#include "mdio/dataset.h"

PYBIND11_MODULE(_core, m) {
  m.doc() = "Python bindings for the MDIO-cpp library";
  m.attr("__version__") = MDIO_API_VERSION;

  py::register_exception<mdio_py::MdioError>(m, "MdioError");

  BindConstants(m);
  BindDescriptors(m);
  BindVariable(m);
  BindDataset(m);
  BindUtils(m);
}
