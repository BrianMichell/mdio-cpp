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

#ifndef PYTHON_SRC_BINDINGS_STATUS_H_
#define PYTHON_SRC_BINDINGS_STATUS_H_

#include <pybind11/pybind11.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "mdio/impl.h"

namespace py = pybind11;

namespace mdio_py {

class MdioError : public std::runtime_error {
 public:
  explicit MdioError(const std::string& message)
      : std::runtime_error(message) {}
};

inline void ThrowIfError(const absl::Status& status) {
  if (!status.ok()) {
    throw MdioError(std::string(status.ToString()));
  }
}

template <typename T>
T CheckResult(mdio::Result<T> result) {
  ThrowIfError(result.status());
  return std::move(result).value();
}

inline void CheckResult(const mdio::Result<void>& result) {
  ThrowIfError(result.status());
}

template <typename T>
T WaitFuture(mdio::Future<T> future) {
  absl::Status status;
  std::optional<T> value;
  {
    py::gil_scoped_release release;
    auto result = future.result();
    status = result.status();
    if (result.ok()) {
      value.emplace(std::move(result).value());
    }
  }
  ThrowIfError(status);
  return std::move(*value);
}

inline void WaitFuture(const mdio::Future<void>& future) {
  absl::Status status;
  {
    py::gil_scoped_release release;
    status = future.status();
  }
  ThrowIfError(status);
}

inline void WaitWrite(const mdio::WriteFutures& write) {
  absl::Status status;
  {
    py::gil_scoped_release release;
    status = write.status();
  }
  ThrowIfError(status);
}

}  // namespace mdio_py

#endif  // PYTHON_SRC_BINDINGS_STATUS_H_
